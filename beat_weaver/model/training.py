"""Training loop with mixed-precision, checkpointing, and TensorBoard logging."""

from __future__ import annotations

import json
import logging
import sys
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from beat_weaver.model.config import ModelConfig
from beat_weaver.model.dataset import BeatSaberDataset, build_weighted_sampler, collate_fn
from beat_weaver.model.tokenizer import (
    BAR,
    LEFT_BASE,
    LEFT_COUNT,
    PAD,
    POS_BASE,
    POS_COUNT,
    RIGHT_BASE,
    RIGHT_COUNT,
    SUBDIVISIONS_PER_BAR,
)
from beat_weaver.model.transformer import BeatWeaverModel

logger = logging.getLogger(__name__)


def _get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def _build_lr_scheduler(
    optimizer: torch.optim.Optimizer, config: ModelConfig, steps_per_epoch: int,
) -> torch.optim.lr_scheduler.LRScheduler:
    """Cosine LR schedule with linear warmup.

    steps_per_epoch should be the number of *optimizer steps* per epoch
    (i.e. len(train_loader) // gradient_accumulation_steps), not the raw
    batch count.
    """
    total_steps = config.max_epochs * steps_per_epoch

    def lr_lambda(step: int) -> float:
        if step < config.warmup_steps:
            return step / max(1, config.warmup_steps)
        progress = (step - config.warmup_steps) / max(1, total_steps - config.warmup_steps)
        return 0.5 * (1.0 + __import__("math").cos(__import__("math").pi * progress))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


def _expand_onset_input_proj(model: BeatWeaverModel, state_dict: dict) -> dict:
    """Pad encoder.input_proj when loading an 80-bin checkpoint into 81-bin.

    New onset column is zeros so the extra channel starts as a no-op.
    """
    key = "encoder.input_proj.weight"
    if key not in state_dict:
        return state_dict
    ckpt_w = state_dict[key]
    model_w = model.encoder.input_proj.weight
    if ckpt_w.shape == model_w.shape:
        return state_dict
    if ckpt_w.shape[0] != model_w.shape[0]:
        raise RuntimeError(
            f"encoder.input_proj out_features mismatch: ckpt {tuple(ckpt_w.shape)} "
            f"vs model {tuple(model_w.shape)}"
        )
    if model_w.shape[1] != ckpt_w.shape[1] + 1:
        raise RuntimeError(
            f"Cannot expand encoder.input_proj {tuple(ckpt_w.shape)} "
            f"into {tuple(model_w.shape)}"
        )
    expanded = model_w.detach().clone()
    expanded[:, : ckpt_w.shape[1]] = ckpt_w
    expanded[:, ckpt_w.shape[1] :] = 0
    out = dict(state_dict)
    out[key] = expanded
    logger.info(
        "Expanded encoder.input_proj %s → %s (onset column zero-init)",
        tuple(ckpt_w.shape), tuple(expanded.shape),
    )
    return out


def mix_scheduled_sampling_inputs(
    input_tokens: torch.Tensor,
    pred_tokens: torch.Tensor,
    input_mask: torch.Tensor,
    prob: float,
) -> torch.Tensor:
    """Replace some decoder inputs with the model's previous-step prediction.

    ``input_tokens[:, t]`` is gold[t]. ``pred_tokens[:, t]`` is the model's
    prediction of gold[t+1], so position t+1 is replaced with pred[:, t].
    Position 0 (START / DIFF) is never replaced. PAD positions stay PAD.
    ``prob <= 0`` returns ``input_tokens`` unchanged.
    """
    if prob <= 0:
        return input_tokens
    if input_tokens.size(1) < 2:
        return input_tokens
    mixed = input_tokens.clone()
    bernoulli = torch.rand(
        input_tokens.size(0), input_tokens.size(1) - 1,
        device=input_tokens.device,
    ) < prob
    valid = input_mask[:, 1:] & (input_tokens[:, 1:] != PAD)
    replace = bernoulli & valid
    mixed[:, 1:] = torch.where(replace, pred_tokens[:, :-1], input_tokens[:, 1:])
    return mixed


def cube_target_from_tokens(
    tokens: torch.Tensor, token_mask: torch.Tensor, max_frames: int,
) -> torch.Tensor:
    """Binary (batch, max_frames) target: 1.0 at audio frames with an active POS event.

    Derived directly from the token sequence — BAR advances the bar index,
    POS_p sets the position within the bar — the exact same position math
    encode_beatmap uses to place notes/bombs, so this needs no extra data
    from dataset.py (no raw note/bomb list, no collate_fn change). Frame
    index = bar * SUBDIVISIONS_PER_BAR + p, matching how generate_full_song
    already maps audio frames to beats (frame / 16 = beat).
    """
    batch = tokens.size(0)
    target = torch.zeros(batch, max_frames, device=tokens.device)
    tokens_cpu = tokens.detach().cpu()
    mask_cpu = token_mask.detach().cpu()
    for b in range(batch):
        bar = -1
        seq = tokens_cpu[b]
        m = mask_cpu[b]
        for i in range(seq.size(0)):
            if not m[i]:
                break
            tok = int(seq[i].item())
            if tok == BAR:
                bar += 1
            elif POS_BASE <= tok < POS_BASE + POS_COUNT and bar >= 0:
                frame = bar * SUBDIVISIONS_PER_BAR + (tok - POS_BASE)
                if frame < max_frames:
                    target[b, frame] = 1.0
    return target


def cube_head_loss(
    cube_logits: torch.Tensor, tokens: torch.Tensor, token_mask: torch.Tensor,
    mel_mask: torch.Tensor, pos_weight: float = 1.0,
) -> torch.Tensor:
    """BCE loss for the cube head, masked to valid audio frames only.

    pos_weight upweights the positive (active-frame) class — active frames
    are only ~4-5% of all frames, so pos_weight=1.0 (the default) lets the
    head trivially collapse to "always inactive" and still score >95%
    accuracy without learning anything. Confirmed empirically: a first cube
    head run plateaued at val_cube_accuracy=0.9528 from epoch 2 onward,
    matching 1 - true_positive_rate (0.9554) almost exactly, and its
    inference-time bias — built from that degenerate signal — collapsed
    generate onset F1 to 0.0 (see HANDOFF.md). Raise pos_weight toward the
    true imbalance ratio (~1/positive_rate) to force real discrimination.
    """
    target = cube_target_from_tokens(tokens, token_mask, cube_logits.size(1))
    pw = torch.tensor(pos_weight, device=cube_logits.device, dtype=cube_logits.dtype)
    per_frame = F.binary_cross_entropy_with_logits(
        cube_logits, target, reduction="none", pos_weight=pw,
    )
    weight = mel_mask.float()
    denom = weight.sum().clamp(min=1.0)
    return (per_frame * weight).sum() / denom


def _color_balance_loss(logits: torch.Tensor) -> torch.Tensor:
    """Penalize deviation from 50/50 LEFT/RIGHT token probability.

    Only considers positions where a note token (LEFT or RIGHT) is likely.
    """
    probs = torch.softmax(logits, dim=-1)  # (batch, seq, vocab)
    left_prob = probs[:, :, LEFT_BASE : LEFT_BASE + LEFT_COUNT].sum(dim=-1)
    right_prob = probs[:, :, RIGHT_BASE : RIGHT_BASE + RIGHT_COUNT].sum(dim=-1)
    total = left_prob + right_prob + 1e-8
    # Only count positions where note tokens have meaningful probability
    note_mask = total > 0.1
    if not note_mask.any():
        return torch.tensor(0.0, device=logits.device)
    left_ratio = left_prob[note_mask] / total[note_mask]
    return ((left_ratio - 0.5) ** 2).mean()


class Trainer:
    """Wraps training state and provides train/validate methods."""

    def __init__(
        self,
        model: BeatWeaverModel,
        config: ModelConfig,
        output_dir: Path,
        device: torch.device | None = None,
    ):
        self.model = model
        self.config = config
        self.output_dir = Path(output_dir)
        self.device = device or _get_device()

        self.model.to(self.device)

        self.optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay,
        )
        self.criterion = nn.CrossEntropyLoss(
            ignore_index=PAD,
            label_smoothing=config.label_smoothing,
        )
        self.scaler = torch.amp.GradScaler(enabled=self.device.type == "cuda")
        self.writer = SummaryWriter(log_dir=str(self.output_dir / "logs"))

        self.epoch = 0
        self.global_step = 0
        self.best_val_loss = float("inf")
        self.patience_counter = 0

    def _scheduled_sampling_prob(self) -> float:
        """Mix rate for this epoch. 0 = pure teacher forcing."""
        p_max = self.config.scheduled_sampling_prob
        if p_max <= 0:
            return 0.0
        ramp = self.config.scheduled_sampling_ramp_epochs
        if ramp <= 0:
            return p_max
        return p_max * min(1.0, (self.epoch + 1) / ramp)

    def train_epoch(self, dataloader: DataLoader) -> float:
        """Run one training epoch. Returns average loss."""
        self.model.train()
        total_loss = 0.0
        n_batches = 0
        accum_steps = self.config.gradient_accumulation_steps

        self.optimizer.zero_grad()

        for batch_idx, (mel, mel_mask, tokens, token_mask) in enumerate(dataloader):
            mel = mel.to(self.device)
            mel_mask = mel_mask.to(self.device)
            tokens = tokens.to(self.device)
            token_mask = token_mask.to(self.device)

            # Teacher forcing: input is tokens[:-1], target is tokens[1:]
            input_tokens = tokens[:, :-1]
            target_tokens = tokens[:, 1:]
            input_mask = token_mask[:, :-1]

            mix_p = self._scheduled_sampling_prob()
            if mix_p > 0:
                # Parallel scheduled sampling: one no-grad TF pass to get
                # discrete predictions, then mix them into the real forward.
                with torch.no_grad():
                    with torch.amp.autocast(
                        device_type=self.device.type, enabled=self.device.type == "cuda",
                    ):
                        tf_logits = self.model(mel, input_tokens, mel_mask, input_mask)
                    pred_tokens = tf_logits.argmax(dim=-1)
                input_tokens = mix_scheduled_sampling_inputs(
                    input_tokens, pred_tokens, input_mask, mix_p,
                )

            use_cube = self.config.use_cube_head and self.config.cube_head_weight > 0
            with torch.amp.autocast(device_type=self.device.type, enabled=self.device.type == "cuda"):
                if use_cube:
                    logits, cube_logits = self.model(
                        mel, input_tokens, mel_mask, input_mask, return_cube=True,
                    )
                else:
                    logits = self.model(mel, input_tokens, mel_mask, input_mask)
                # logits: (batch, seq_len-1, vocab_size)
                # target: (batch, seq_len-1)
                loss = self.criterion(
                    logits.reshape(-1, logits.size(-1)),
                    target_tokens.reshape(-1),
                )
                # Color balance auxiliary loss
                if self.config.color_balance_weight > 0:
                    loss = loss + self.config.color_balance_weight * _color_balance_loss(logits)
                # Cube-head auxiliary loss — supervises the encoder directly on
                # "is there an event here", independent of the AR decoder's own
                # willingness to emit a POS token instead of skipping to BAR.
                if use_cube and cube_logits is not None:
                    c_loss = cube_head_loss(
                        cube_logits, tokens, token_mask, mel_mask,
                        pos_weight=self.config.cube_head_pos_weight,
                    )
                    loss = loss + self.config.cube_head_weight * c_loss
                loss = loss / accum_steps  # Scale for accumulation

            self.scaler.scale(loss).backward()

            if (batch_idx + 1) % accum_steps == 0 or (batch_idx + 1) == len(dataloader):
                self.scaler.unscale_(self.optimizer)
                nn.utils.clip_grad_norm_(
                    self.model.parameters(), self.config.gradient_clip_norm,
                )
                self.scaler.step(self.optimizer)
                self.scaler.update()
                self.optimizer.zero_grad()

                if hasattr(self, "scheduler"):
                    self.scheduler.step()

            total_loss += loss.item() * accum_steps  # Unscale for logging
            n_batches += 1
            self.global_step += 1

            # Log every 50 steps
            if self.global_step % 50 == 0:
                self.writer.add_scalar("train/loss_step", loss.item() * accum_steps, self.global_step)
                lr = self.optimizer.param_groups[0]["lr"]
                self.writer.add_scalar("train/lr", lr, self.global_step)

        avg_loss = total_loss / max(1, n_batches)
        return avg_loss

    @torch.no_grad()
    def validate(self, dataloader: DataLoader) -> dict[str, float]:
        """Run validation. Returns dict of metrics."""
        self.model.eval()
        total_loss = 0.0
        total_correct = 0
        total_tokens = 0
        n_batches = 0
        use_cube = self.config.use_cube_head
        cube_correct = 0
        cube_total = 0
        cube_loss_sum = 0.0
        cube_true_positive = 0
        cube_actual_positive = 0

        for mel, mel_mask, tokens, token_mask in dataloader:
            mel = mel.to(self.device)
            mel_mask = mel_mask.to(self.device)
            tokens = tokens.to(self.device)
            token_mask = token_mask.to(self.device)

            input_tokens = tokens[:, :-1]
            target_tokens = tokens[:, 1:]
            input_mask = token_mask[:, :-1]
            target_mask = token_mask[:, 1:]

            if use_cube:
                logits, cube_logits = self.model(
                    mel, input_tokens, mel_mask, input_mask, return_cube=True,
                )
            else:
                logits = self.model(mel, input_tokens, mel_mask, input_mask)
            loss = self.criterion(
                logits.reshape(-1, logits.size(-1)),
                target_tokens.reshape(-1),
            )
            total_loss += loss.item()
            n_batches += 1

            # Token accuracy (ignoring padding)
            preds = logits.argmax(dim=-1)
            mask = target_mask & (target_tokens != PAD)
            total_correct += (preds == target_tokens)[mask].sum().item()
            total_tokens += mask.sum().item()

            # Cube-head diagnostic: BCE loss + accuracy at >0.5 threshold,
            # over valid audio frames only. Not used for early stopping —
            # see HANDOFF.md on why val_loss/token acc aren't the ship metric.
            if use_cube and cube_logits is not None:
                cube_target = cube_target_from_tokens(tokens, token_mask, cube_logits.size(1))
                cube_loss_sum += cube_head_loss(
                    cube_logits, tokens, token_mask, mel_mask,
                    pos_weight=self.config.cube_head_pos_weight,
                ).item()
                cube_preds = (cube_logits > 0).float()
                valid = mel_mask.float()
                cube_correct += ((cube_preds == cube_target).float() * valid).sum().item()
                cube_total += valid.sum().item()
                # Recall on the positive (active-frame) class — the number
                # that actually catches a collapse to "always inactive",
                # which raw accuracy hides under class imbalance (~95%
                # accuracy for predicting nothing, since only ~4-5% of
                # frames are truly active).
                is_positive = (cube_target > 0.5) & (valid > 0.5)
                cube_true_positive += (cube_preds[is_positive] > 0.5).sum().item()
                cube_actual_positive += is_positive.sum().item()

        avg_loss = total_loss / max(1, n_batches)
        accuracy = total_correct / max(1, total_tokens)
        metrics = {"val_loss": avg_loss, "val_token_accuracy": accuracy}
        if use_cube:
            metrics["val_cube_loss"] = cube_loss_sum / max(1, n_batches)
            metrics["val_cube_accuracy"] = cube_correct / max(1.0, cube_total)
            metrics["val_cube_recall"] = cube_true_positive / max(1, cube_actual_positive)
        return metrics

    def save_checkpoint(self, name: str) -> Path:
        """Save model + optimizer + scheduler + training state."""
        ckpt_dir = self.output_dir / "checkpoints" / name
        ckpt_dir.mkdir(parents=True, exist_ok=True)

        torch.save(self.model.state_dict(), ckpt_dir / "model.pt")
        torch.save(self.optimizer.state_dict(), ckpt_dir / "optimizer.pt")
        if hasattr(self, "scheduler"):
            torch.save(self.scheduler.state_dict(), ckpt_dir / "scheduler.pt")
        torch.save(self.scaler.state_dict(), ckpt_dir / "scaler.pt")
        self.config.save(ckpt_dir / "config.json")

        state = {
            "epoch": self.epoch,
            "global_step": self.global_step,
            "best_val_loss": self.best_val_loss,
        }
        (ckpt_dir / "training_state.json").write_text(json.dumps(state, indent=2), encoding="utf-8")
        return ckpt_dir

    def load_checkpoint(self, ckpt_dir: Path) -> None:
        """Resume from a checkpoint."""
        ckpt_dir = Path(ckpt_dir)
        self._resume_dir = ckpt_dir
        self.model.load_state_dict(
            torch.load(ckpt_dir / "model.pt", map_location=self.device, weights_only=True),
        )
        self.optimizer.load_state_dict(
            torch.load(ckpt_dir / "optimizer.pt", map_location=self.device, weights_only=True),
        )
        scaler_path = ckpt_dir / "scaler.pt"
        if scaler_path.exists():
            self.scaler.load_state_dict(
                torch.load(scaler_path, map_location=self.device, weights_only=True),
            )
            logger.info("Restored GradScaler state from %s", scaler_path)
        elif self.device.type == "cuda":
            # No scaler.pt — use conservative scale=1.0 to avoid overflow
            self.scaler = torch.amp.GradScaler(init_scale=1.0, growth_interval=1000)
            logger.info("No scaler.pt found; using conservative init_scale=1.0")
        state = json.loads((ckpt_dir / "training_state.json").read_text(encoding="utf-8"))
        self.epoch = state["epoch"]
        self.global_step = state["global_step"]
        self.best_val_loss = state["best_val_loss"]

    def load_weights(self, ckpt_dir: Path) -> None:
        """Load model weights only — fresh optimizer, scaler, epoch, LR.

        Use this for a fine-tune (scheduled sampling, low LR) from a finished
        run. ``load_checkpoint`` would reuse the decayed cosine schedule.

        If this model has ``use_onset_features`` and the checkpoint does not,
        the encoder input projection is expanded by one zero-initialized
        column so the onset channel starts unused and can learn.

        If this model has ``use_cube_head`` and the checkpoint does not, the
        head loads randomly initialized (strict=False) — expected when
        fine-tuning a new architecture cut onto an existing run, per
        HANDOFF.md's "--init-from SS/best" recipe. Any *other* missing or
        unexpected key still raises, so a real mismatch isn't silently eaten.
        """
        ckpt_dir = Path(ckpt_dir)
        state = torch.load(ckpt_dir / "model.pt", map_location=self.device, weights_only=True)
        state = _expand_onset_input_proj(self.model, state)
        expected_missing = {"cube_head.weight", "cube_head.bias"} if self.config.use_cube_head else set()
        result = self.model.load_state_dict(state, strict=False)
        unexpected_missing = set(result.missing_keys) - expected_missing
        if unexpected_missing or result.unexpected_keys:
            raise RuntimeError(
                f"Unexpected state_dict mismatch loading {ckpt_dir}: "
                f"missing={sorted(unexpected_missing)} unexpected={sorted(result.unexpected_keys)}"
            )
        if result.missing_keys:
            logger.info(
                "%s randomly initialized (not present in %s)",
                sorted(result.missing_keys), ckpt_dir,
            )
        self.epoch = 0
        self.global_step = 0
        self.best_val_loss = float("inf")
        self.patience_counter = 0
        logger.info("Loaded weights from %s (fresh optimizer / LR)", ckpt_dir)

    def restore_scheduler(self) -> None:
        """Restore scheduler state if resuming and scheduler.pt exists.

        If no scheduler.pt is found, fast-forward the scheduler to the
        current global_step so the LR matches where training left off.
        """
        resume_dir = getattr(self, "_resume_dir", None)
        if resume_dir is None:
            return
        if not hasattr(self, "scheduler"):
            return
        scheduler_path = resume_dir / "scheduler.pt"
        if scheduler_path.exists():
            self.scheduler.load_state_dict(
                torch.load(scheduler_path, map_location=self.device, weights_only=True),
            )
            logger.info("Restored LR scheduler state from %s", scheduler_path)
        elif self.global_step > 0:
            # No scheduler.pt — fast-forward to match resumed global_step
            for _ in range(self.global_step):
                self.scheduler.step()
            lr = self.optimizer.param_groups[0]["lr"]
            logger.info(
                "Fast-forwarded LR scheduler to step %d (lr=%.2e)", self.global_step, lr,
            )


def train(
    config: ModelConfig,
    train_dataset: BeatSaberDataset,
    val_dataset: BeatSaberDataset,
    output_dir: Path,
    resume_from: Path | None = None,
    init_from: Path | None = None,
) -> Path:
    """Main training entry point.

    Returns path to the best checkpoint directory.
    """
    if resume_from is not None and init_from is not None:
        raise ValueError("Pass only one of resume_from or init_from")

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    model = BeatWeaverModel(config)
    logger.info("Model parameters: %s", f"{model.count_parameters():,}")

    trainer = Trainer(model, config, output_dir)

    if resume_from:
        trainer.load_checkpoint(resume_from)
        logger.info("Resumed from %s (epoch %d)", resume_from, trainer.epoch)
    elif init_from:
        trainer.load_weights(init_from)

    sampler = build_weighted_sampler(train_dataset, config.official_ratio)
    use_cuda = trainer.device.type == "cuda"
    # Windows spawn-based multiprocessing causes DataLoader worker deadlocks
    # between epochs with persistent_workers. Use num_workers=0 on Windows.
    num_workers = 0 if sys.platform == "win32" else (2 if use_cuda else 0)
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=sampler is None,  # shuffle only when no weighted sampler
        sampler=sampler,
        collate_fn=collate_fn,
        num_workers=num_workers,
        pin_memory=use_cuda,
        persistent_workers=num_workers > 0,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=num_workers,
        pin_memory=use_cuda,
        persistent_workers=num_workers > 0,
    )

    # Build scheduler after knowing steps_per_epoch.
    # The scheduler steps once per optimizer step, which happens every
    # gradient_accumulation_steps batches (not every batch).
    optimizer_steps_per_epoch = max(1, len(train_loader) // config.gradient_accumulation_steps)
    trainer.scheduler = _build_lr_scheduler(
        trainer.optimizer, config, optimizer_steps_per_epoch,
    )
    trainer.restore_scheduler()

    config.save(output_dir / "config.json")

    training_start = time.time()
    epoch_times: list[float] = []

    logger.info(
        "Training: %d train samples, %d val samples, %d batches/epoch, device=%s",
        len(train_dataset), len(val_dataset), len(train_loader), trainer.device,
    )

    for epoch in range(trainer.epoch, config.max_epochs):
        trainer.epoch = epoch
        mix_p = trainer._scheduled_sampling_prob()
        if config.scheduled_sampling_prob > 0:
            logger.info(
                "Scheduled sampling: epoch %d mix_p=%.3f (max=%.3f, ramp=%d)",
                epoch + 1, mix_p, config.scheduled_sampling_prob,
                config.scheduled_sampling_ramp_epochs,
            )
            trainer.writer.add_scalar("train/scheduled_sampling_p", mix_p, epoch)
        t0 = time.time()

        train_loss = trainer.train_epoch(train_loader)
        val_metrics = trainer.validate(val_loader)

        elapsed = time.time() - t0
        epoch_times.append(elapsed)
        total_elapsed = time.time() - training_start

        cube_suffix = ""
        if "val_cube_accuracy" in val_metrics:
            cube_suffix = (
                f" cube_loss={val_metrics['val_cube_loss']:.4f} "
                f"cube_acc={val_metrics['val_cube_accuracy']:.4f} "
                f"cube_recall={val_metrics['val_cube_recall']:.4f}"
            )
        logger.info(
            "Epoch %d/%d (%.1fs, total %.0fs): train_loss=%.4f val_loss=%.4f val_acc=%.4f%s",
            epoch + 1, config.max_epochs, elapsed, total_elapsed,
            train_loss, val_metrics["val_loss"], val_metrics["val_token_accuracy"], cube_suffix,
        )

        # TensorBoard
        trainer.writer.add_scalar("train/loss_epoch", train_loss, epoch)
        trainer.writer.add_scalar("val/loss", val_metrics["val_loss"], epoch)
        trainer.writer.add_scalar("val/token_accuracy", val_metrics["val_token_accuracy"], epoch)
        if "val_cube_accuracy" in val_metrics:
            trainer.writer.add_scalar("val/cube_loss", val_metrics["val_cube_loss"], epoch)
            trainer.writer.add_scalar("val/cube_accuracy", val_metrics["val_cube_accuracy"], epoch)
            trainer.writer.add_scalar("val/cube_recall", val_metrics["val_cube_recall"], epoch)
        trainer.writer.add_scalar("timing/epoch_seconds", elapsed, epoch)
        trainer.writer.add_scalar("timing/total_seconds", total_elapsed, epoch)

        # Checkpoint every epoch
        trainer.save_checkpoint(f"epoch_{epoch + 1:03d}")

        # Best model
        if val_metrics["val_loss"] < trainer.best_val_loss:
            trainer.best_val_loss = val_metrics["val_loss"]
            trainer.patience_counter = 0
            trainer.save_checkpoint("best")
            logger.info("New best model (val_loss=%.4f)", trainer.best_val_loss)
        else:
            trainer.patience_counter += 1
            if trainer.patience_counter >= config.early_stopping_patience:
                logger.info("Early stopping at epoch %d", epoch + 1)
                break

    trainer.writer.close()

    # Write training summary
    total_time = time.time() - training_start
    epochs_completed = len(epoch_times)
    avg_epoch = sum(epoch_times) / max(1, epochs_completed)
    summary = {
        "train_samples": len(train_dataset),
        "val_samples": len(val_dataset),
        "batches_per_epoch": len(train_loader),
        "batch_size": config.batch_size,
        "epochs_completed": epochs_completed,
        "total_time_seconds": round(total_time, 1),
        "avg_epoch_seconds": round(avg_epoch, 1),
        "samples_per_second": round(len(train_dataset) / avg_epoch, 1),
        "best_val_loss": round(trainer.best_val_loss, 4),
        "device": str(trainer.device),
        "model_parameters": model.count_parameters(),
    }
    summary_path = output_dir / "training_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    logger.info(
        "Training complete: %d epochs in %.0fs (avg %.1fs/epoch, %.1f samples/s)",
        epochs_completed, total_time, avg_epoch,
        len(train_dataset) / avg_epoch,
    )

    return output_dir / "checkpoints" / "best"
