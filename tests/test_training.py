"""Tests for training utilities — color balance loss."""

import pytest

torch = pytest.importorskip("torch")

from beat_weaver.model.config import ModelConfig
from beat_weaver.model.training import (
    Trainer,
    _color_balance_loss,
    _expand_onset_input_proj,
    cube_head_loss,
    cube_target_from_tokens,
    mix_scheduled_sampling_inputs,
    train,
)
from beat_weaver.model.tokenizer import (
    BAR,
    LEFT_BASE,
    LEFT_COUNT,
    PAD,
    POS_BASE,
    RIGHT_BASE,
    RIGHT_COUNT,
    START,
    SUBDIVISIONS_PER_BAR,
)
from beat_weaver.model.transformer import BeatWeaverModel


class TestTrainerWeightDecay:
    def test_config_weight_decay_reaches_optimizer(self, tmp_path):
        """A non-default weight_decay in config must actually reach AdamW —
        small-dataset training relies on this being tunable per-config."""
        config = ModelConfig(
            vocab_size=32, max_seq_len=16, n_mels=8, max_audio_len=16,
            encoder_layers=1, encoder_dim=8, encoder_heads=2, encoder_ff_dim=16,
            decoder_layers=1, decoder_dim=8, decoder_heads=2, decoder_ff_dim=16,
            weight_decay=0.05,
        )
        model = BeatWeaverModel(config)
        trainer = Trainer(model, config, tmp_path, device=torch.device("cpu"))
        assert trainer.optimizer.param_groups[0]["weight_decay"] == 0.05

    def test_default_weight_decay_unchanged(self, tmp_path):
        """Default weight_decay must match the previous hardcoded value (0.01)."""
        config = ModelConfig(
            vocab_size=32, max_seq_len=16, n_mels=8, max_audio_len=16,
            encoder_layers=1, encoder_dim=8, encoder_heads=2, encoder_ff_dim=16,
            decoder_layers=1, decoder_dim=8, decoder_heads=2, decoder_ff_dim=16,
        )
        model = BeatWeaverModel(config)
        trainer = Trainer(model, config, tmp_path, device=torch.device("cpu"))
        assert trainer.optimizer.param_groups[0]["weight_decay"] == 0.01


class TestScheduledSamplingMix:
    def test_prob_zero_is_identity(self):
        tokens = torch.tensor([[1, 2, 3, 4]])
        preds = torch.tensor([[9, 9, 9, 9]])
        mask = torch.ones_like(tokens, dtype=torch.bool)
        mixed = mix_scheduled_sampling_inputs(tokens, preds, mask, prob=0.0)
        assert torch.equal(mixed, tokens)

    def test_prob_one_replaces_from_position_1(self):
        tokens = torch.tensor([[1, 2, 3, 4]])
        preds = torch.tensor([[10, 20, 30, 40]])
        mask = torch.ones_like(tokens, dtype=torch.bool)
        mixed = mix_scheduled_sampling_inputs(tokens, preds, mask, prob=1.0)
        # pos 0 kept; pos 1.. = pred of previous step
        assert mixed[0, 0].item() == 1
        assert mixed[0, 1].item() == 10
        assert mixed[0, 2].item() == 20
        assert mixed[0, 3].item() == 30

    def test_pad_is_never_replaced(self):
        tokens = torch.tensor([[1, 2, PAD, PAD]])
        preds = torch.tensor([[9, 9, 9, 9]])
        mask = torch.tensor([[True, True, False, False]])
        mixed = mix_scheduled_sampling_inputs(tokens, preds, mask, prob=1.0)
        assert mixed[0, 2].item() == PAD
        assert mixed[0, 3].item() == PAD

    def test_ramp_zero_is_constant(self, tmp_path):
        config = ModelConfig(
            vocab_size=32, max_seq_len=16, n_mels=8, max_audio_len=16,
            encoder_layers=1, encoder_dim=8, encoder_heads=2, encoder_ff_dim=16,
            decoder_layers=1, decoder_dim=8, decoder_heads=2, decoder_ff_dim=16,
            scheduled_sampling_prob=0.4, scheduled_sampling_ramp_epochs=0,
        )
        model = BeatWeaverModel(config)
        trainer = Trainer(model, config, tmp_path, device=torch.device("cpu"))
        trainer.epoch = 0
        assert trainer._scheduled_sampling_prob() == pytest.approx(0.4)
        trainer.epoch = 7
        assert trainer._scheduled_sampling_prob() == pytest.approx(0.4)

    def test_ramp_reaches_max(self, tmp_path):
        config = ModelConfig(
            vocab_size=32, max_seq_len=16, n_mels=8, max_audio_len=16,
            encoder_layers=1, encoder_dim=8, encoder_heads=2, encoder_ff_dim=16,
            decoder_layers=1, decoder_dim=8, decoder_heads=2, decoder_ff_dim=16,
            scheduled_sampling_prob=0.5, scheduled_sampling_ramp_epochs=5,
        )
        model = BeatWeaverModel(config)
        trainer = Trainer(model, config, tmp_path, device=torch.device("cpu"))
        trainer.epoch = 0
        assert trainer._scheduled_sampling_prob() == pytest.approx(0.1)
        trainer.epoch = 4
        assert trainer._scheduled_sampling_prob() == pytest.approx(0.5)
        trainer.epoch = 9
        assert trainer._scheduled_sampling_prob() == pytest.approx(0.5)


class TestExpandOnsetInputProj:
    def _tiny(self, onset: bool, tmp_path):
        config = ModelConfig(
            vocab_size=32, max_seq_len=16, n_mels=8, max_audio_len=16,
            encoder_layers=1, encoder_dim=8, encoder_heads=2, encoder_ff_dim=16,
            decoder_layers=1, decoder_dim=8, decoder_heads=2, decoder_ff_dim=16,
            use_onset_features=onset, use_conformer=False, use_rope=False,
        )
        return BeatWeaverModel(config), config

    def test_same_shape_is_noop(self, tmp_path):
        model, _ = self._tiny(False, tmp_path)
        state = model.state_dict()
        out = _expand_onset_input_proj(model, state)
        assert torch.equal(out["encoder.input_proj.weight"], state["encoder.input_proj.weight"])

    def test_expands_zero_onset_column(self, tmp_path):
        src, _ = self._tiny(False, tmp_path)
        dst, _ = self._tiny(True, tmp_path)
        expanded = _expand_onset_input_proj(dst, src.state_dict())
        w = expanded["encoder.input_proj.weight"]
        assert w.shape[1] == src.encoder.input_proj.weight.shape[1] + 1
        assert torch.equal(w[:, :-1], src.encoder.input_proj.weight)
        assert torch.all(w[:, -1] == 0)

    def test_load_weights_from_no_onset_checkpoint(self, tmp_path):
        src, src_cfg = self._tiny(False, tmp_path)
        ckpt = tmp_path / "ckpt"
        ckpt.mkdir()
        torch.save(src.state_dict(), ckpt / "model.pt")
        dst, dst_cfg = self._tiny(True, tmp_path)
        trainer = Trainer(dst, dst_cfg, tmp_path, device=torch.device("cpu"))
        trainer.load_weights(ckpt)
        w = trainer.model.encoder.input_proj.weight
        assert w.shape[1] == 9
        assert torch.all(w[:, -1] == 0)


class TestCubeTargetFromTokens:
    def test_marks_pos_frames(self):
        # START DIFF BAR POS_2 LEFT_EMPTY RIGHT_EMPTY BAR POS_5 LEFT_EMPTY RIGHT_EMPTY END
        tokens = torch.tensor([[
            START, 4, BAR, POS_BASE + 2, 73, 182, BAR, POS_BASE + 5, 73, 182, 2,
        ]])
        mask = torch.ones_like(tokens, dtype=torch.bool)
        target = cube_target_from_tokens(tokens, mask, max_frames=SUBDIVISIONS_PER_BAR * 2)
        expected = torch.zeros(1, SUBDIVISIONS_PER_BAR * 2)
        expected[0, 2] = 1.0  # bar 0, sub 2
        expected[0, SUBDIVISIONS_PER_BAR + 5] = 1.0  # bar 1, sub 5
        assert torch.equal(target, expected)

    def test_no_pos_tokens_all_zero(self):
        tokens = torch.tensor([[START, 4, 2, PAD, PAD]])
        mask = torch.tensor([[True, True, True, False, False]])
        target = cube_target_from_tokens(tokens, mask, max_frames=64)
        assert torch.all(target == 0)

    def test_stops_at_mask_boundary(self):
        """A POS token past the valid mask (padding) must not be counted."""
        tokens = torch.tensor([[START, 4, BAR, POS_BASE + 10, 73, 182]])
        mask = torch.tensor([[True, True, True, False, False, False]])
        target = cube_target_from_tokens(tokens, mask, max_frames=64)
        assert torch.all(target == 0)

    def test_frame_beyond_max_frames_ignored(self):
        """A POS position past max_frames must not raise or wrap around."""
        tokens = torch.tensor([[START, 4, BAR, POS_BASE + 63, 73, 182, 2]])
        mask = torch.ones_like(tokens, dtype=torch.bool)
        target = cube_target_from_tokens(tokens, mask, max_frames=10)
        assert target.shape == (1, 10)
        assert torch.all(target == 0)  # frame 63 is out of range


class TestCubeHeadLoss:
    def test_perfect_prediction_near_zero_loss(self):
        target_tokens = torch.tensor([[BAR, POS_BASE + 2, 73, 182, 2]])
        mask = torch.ones_like(target_tokens, dtype=torch.bool)
        cube_logits = torch.full((1, 64), -10.0)
        cube_logits[0, 2] = 10.0
        mel_mask = torch.ones(1, 64, dtype=torch.bool)
        loss = cube_head_loss(cube_logits, target_tokens, mask, mel_mask)
        assert loss.item() < 0.01

    def test_wrong_prediction_high_loss(self):
        target_tokens = torch.tensor([[BAR, POS_BASE + 2, 73, 182, 2]])
        mask = torch.ones_like(target_tokens, dtype=torch.bool)
        cube_logits = torch.full((1, 64), 10.0)  # predicts "active" everywhere
        cube_logits[0, 2] = -10.0  # ...except the one frame that's actually active
        mel_mask = torch.ones(1, 64, dtype=torch.bool)
        loss = cube_head_loss(cube_logits, target_tokens, mask, mel_mask)
        assert loss.item() > 5.0

    def test_pos_weight_penalizes_missed_positives_more(self):
        """A missed positive frame (predicted inactive) should cost more
        loss under a higher pos_weight — this is the fix for the head
        collapsing to 'always inactive' under the true ~4-5% class balance."""
        target_tokens = torch.tensor([[BAR, POS_BASE + 2, 73, 182, 2]])
        mask = torch.ones_like(target_tokens, dtype=torch.bool)
        mel_mask = torch.ones(1, 64, dtype=torch.bool)
        # Confidently wrong on the one active frame (frame 2), correct everywhere else.
        cube_logits = torch.full((1, 64), -10.0)

        loss_unweighted = cube_head_loss(cube_logits, target_tokens, mask, mel_mask, pos_weight=1.0)
        loss_weighted = cube_head_loss(cube_logits, target_tokens, mask, mel_mask, pos_weight=20.0)
        assert loss_weighted.item() > loss_unweighted.item()

    def test_default_pos_weight_matches_unweighted(self):
        """pos_weight=1.0 (the default) must match plain BCE exactly."""
        target_tokens = torch.tensor([[BAR, POS_BASE + 2, 73, 182, 2]])
        mask = torch.ones_like(target_tokens, dtype=torch.bool)
        mel_mask = torch.ones(1, 64, dtype=torch.bool)
        cube_logits = torch.randn(1, 64)

        loss_default = cube_head_loss(cube_logits, target_tokens, mask, mel_mask)
        loss_explicit = cube_head_loss(cube_logits, target_tokens, mask, mel_mask, pos_weight=1.0)
        assert loss_default.item() == pytest.approx(loss_explicit.item())

    def test_ignores_padded_frames(self):
        """Garbage logits in padded (invalid) audio frames must not affect the loss."""
        target_tokens = torch.tensor([[BAR, POS_BASE + 2, 73, 182, 2]])
        mask = torch.ones_like(target_tokens, dtype=torch.bool)
        cube_logits = torch.full((1, 64), -10.0)
        cube_logits[0, 2] = 10.0
        cube_logits[0, 32:] = 999.0  # garbage in the padded region
        mel_mask = torch.zeros(1, 64, dtype=torch.bool)
        mel_mask[0, :32] = True
        loss = cube_head_loss(cube_logits, target_tokens, mask, mel_mask)
        assert loss.item() < 0.01


class TestLoadWeightsCubeHead:
    def _tiny(self, use_cube_head: bool):
        config = ModelConfig(
            vocab_size=32, max_seq_len=16, n_mels=8, max_audio_len=16,
            encoder_layers=1, encoder_dim=8, encoder_heads=2, encoder_ff_dim=16,
            decoder_layers=1, decoder_dim=8, decoder_heads=2, decoder_ff_dim=16,
            use_conformer=False, use_rope=False, use_cube_head=use_cube_head,
        )
        return BeatWeaverModel(config), config

    def test_load_into_model_with_new_cube_head(self, tmp_path):
        """Fine-tuning onto a model with use_cube_head=True from a checkpoint
        that predates the head must succeed, randomly initializing just that
        head — the exact scenario HANDOFF.md's --init-from SS/best recipe
        needs for this architecture cut."""
        src, _ = self._tiny(use_cube_head=False)
        ckpt = tmp_path / "ckpt"
        ckpt.mkdir()
        torch.save(src.state_dict(), ckpt / "model.pt")

        dst, dst_cfg = self._tiny(use_cube_head=True)
        original_cube_weight = dst.cube_head.weight.clone()
        trainer = Trainer(dst, dst_cfg, tmp_path, device=torch.device("cpu"))
        trainer.load_weights(ckpt)  # must not raise

        # Non-cube weights transferred from the checkpoint...
        assert torch.equal(
            trainer.model.encoder.input_proj.weight, src.encoder.input_proj.weight,
        )
        # ...cube_head stayed at its own random init, untouched by the load.
        assert torch.equal(trainer.model.cube_head.weight, original_cube_weight)

    def test_genuine_mismatch_still_raises(self, tmp_path):
        """A real architecture mismatch (not just the expected cube_head gap)
        must still raise, not be silently swallowed by strict=False."""
        config_a = ModelConfig(
            vocab_size=32, max_seq_len=16, n_mels=8, max_audio_len=16,
            encoder_layers=1, encoder_dim=8, encoder_heads=2, encoder_ff_dim=16,
            decoder_layers=1, decoder_dim=16, decoder_heads=2, decoder_ff_dim=16,
            use_conformer=False, use_rope=False, use_cube_head=True,
        )
        config_b = ModelConfig(
            vocab_size=32, max_seq_len=16, n_mels=8, max_audio_len=16,
            encoder_layers=1, encoder_dim=8, encoder_heads=2, encoder_ff_dim=16,
            decoder_layers=1, decoder_dim=32, decoder_heads=2, decoder_ff_dim=16,
            use_conformer=False, use_rope=False, use_cube_head=True,
        )
        src = BeatWeaverModel(config_a)
        ckpt = tmp_path / "ckpt"
        ckpt.mkdir()
        torch.save(src.state_dict(), ckpt / "model.pt")

        dst = BeatWeaverModel(config_b)
        trainer = Trainer(dst, config_b, tmp_path, device=torch.device("cpu"))
        with pytest.raises(RuntimeError):
            trainer.load_weights(ckpt)


class TestInitFromVsResume:
    def test_init_from_and_resume_conflict(self, tmp_path):
        config = ModelConfig(
            vocab_size=32, max_seq_len=16, n_mels=8, max_audio_len=16,
            encoder_layers=1, encoder_dim=8, encoder_heads=2, encoder_ff_dim=16,
            decoder_layers=1, decoder_dim=8, decoder_heads=2, decoder_ff_dim=16,
            max_epochs=1, batch_size=1,
        )
        with pytest.raises(ValueError, match="only one"):
            train(
                config, None, None, tmp_path,  # type: ignore[arg-type]
                resume_from=tmp_path, init_from=tmp_path,
            )


class TestColorBalanceLoss:
    def test_balanced_near_zero(self):
        """Loss near zero for 50/50 left/right predictions."""
        # Create logits that predict equal LEFT and RIGHT probability
        batch, seq, vocab = 2, 10, 291
        logits = torch.zeros(batch, seq, vocab)
        # Set equal logits for all LEFT and RIGHT tokens
        logits[:, :, LEFT_BASE:LEFT_BASE + LEFT_COUNT] = 1.0
        logits[:, :, RIGHT_BASE:RIGHT_BASE + RIGHT_COUNT] = 1.0
        loss = _color_balance_loss(logits)
        assert loss.item() < 0.01

    def test_imbalanced_positive(self):
        """Loss > 0 for 100% left predictions."""
        batch, seq, vocab = 2, 10, 291
        logits = torch.full((batch, seq, vocab), -10.0)
        # All probability on LEFT tokens
        logits[:, :, LEFT_BASE:LEFT_BASE + LEFT_COUNT] = 5.0
        loss = _color_balance_loss(logits)
        assert loss.item() > 0.1

    def test_gradient_flows(self):
        """Gradients flow through auxiliary loss."""
        batch, seq, vocab = 2, 10, 291
        logits = torch.randn(batch, seq, vocab, requires_grad=True)
        loss = _color_balance_loss(logits)
        loss.backward()
        assert logits.grad is not None
        assert logits.grad.abs().sum() > 0

    def test_no_note_tokens(self):
        """Returns 0 when no positions have significant note probability."""
        batch, seq, vocab = 2, 10, 291
        logits = torch.zeros(batch, seq, vocab)
        # All probability on PAD/START/END tokens, none on notes
        logits[:, :, 0:8] = 10.0
        loss = _color_balance_loss(logits)
        assert loss.item() == 0.0
