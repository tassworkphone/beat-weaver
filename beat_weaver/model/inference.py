"""Autoregressive generation with grammar-constrained decoding."""

from __future__ import annotations

import torch
import torch.nn.functional as F

from beat_weaver.model.config import ModelConfig
from beat_weaver.model.tokenizer import (
    BAR,
    BOMB_BASE,
    BOMB_COUNT,
    BOMB_EMPTY,
    DIFF_EASY,
    DIFF_EXPERT_PLUS,
    END,
    LEFT_BASE,
    LEFT_COUNT,
    LEFT_EMPTY,
    POS_BASE,
    POS_COUNT,
    RIGHT_BASE,
    RIGHT_COUNT,
    RIGHT_EMPTY,
    START,
    SUBDIVISIONS_PER_BAR,
    VOCAB_SIZE,
    VOCAB_SIZE_WITH_BOMBS,
    decode_tokens,
    difficulty_to_token,
)
from beat_weaver.model.transformer import BeatWeaverModel
from beat_weaver.schemas.normalized import Note


def _build_grammar_mask(
    last_token: int, last_pos_in_bar: int = -1, include_bombs: bool = False,
) -> torch.Tensor:
    """Build a boolean mask over the vocabulary for valid next tokens.

    Returns a tensor of shape (vocab_size,) where True = allowed.

    Args:
        last_token: The most recently generated token.
        last_pos_in_bar: The last POS offset used in the current bar (-1 if none).
            Used to enforce strictly increasing positions within a bar,
            preventing multiple notes at the same beat.
        include_bombs: Must match the model's config.include_bombs — inserts a
            required BOMB slot between RIGHT and the next POS/BAR/END.

    Grammar rules (include_bombs=False):
        START      → DIFF_*
        DIFF_*     → BAR
        BAR        → POS_* | BAR | END
        POS_*      → LEFT_* | LEFT_EMPTY
        LEFT_*     → RIGHT_* | RIGHT_EMPTY
        RIGHT_*    → POS_* (strictly >) | BAR | END

    With include_bombs=True, RIGHT_* instead transitions to BOMB_*/BOMB_EMPTY,
    which then transitions to POS_*/BAR/END.
    """
    vocab_size = VOCAB_SIZE_WITH_BOMBS if include_bombs else VOCAB_SIZE
    mask = torch.zeros(vocab_size, dtype=torch.bool)

    if last_token == START:
        # After START → only difficulty tokens
        mask[DIFF_EASY: DIFF_EXPERT_PLUS + 1] = True

    elif DIFF_EASY <= last_token <= DIFF_EXPERT_PLUS:
        # After DIFF → only BAR
        mask[BAR] = True

    elif last_token == BAR:
        # After BAR → POS, BAR, or END
        mask[POS_BASE: POS_BASE + POS_COUNT] = True
        mask[BAR] = True
        mask[END] = True

    elif POS_BASE <= last_token < POS_BASE + POS_COUNT:
        # After POS → LEFT note or LEFT_EMPTY
        mask[LEFT_EMPTY] = True
        mask[LEFT_BASE: LEFT_BASE + LEFT_COUNT] = True

    elif last_token == LEFT_EMPTY or (LEFT_BASE <= last_token < LEFT_BASE + LEFT_COUNT):
        # After LEFT → RIGHT note or RIGHT_EMPTY
        mask[RIGHT_EMPTY] = True
        mask[RIGHT_BASE: RIGHT_BASE + RIGHT_COUNT] = True

    elif last_token == RIGHT_EMPTY or (RIGHT_BASE <= last_token < RIGHT_BASE + RIGHT_COUNT):
        if include_bombs:
            # After RIGHT → BOMB or BOMB_EMPTY
            mask[BOMB_EMPTY] = True
            mask[BOMB_BASE: BOMB_BASE + BOMB_COUNT] = True
        else:
            # After RIGHT → POS (strictly increasing), BAR, or END
            min_next = last_pos_in_bar + 1
            if min_next < POS_COUNT:
                mask[POS_BASE + min_next: POS_BASE + POS_COUNT] = True
            mask[BAR] = True
            mask[END] = True

    elif include_bombs and (
        last_token == BOMB_EMPTY or (BOMB_BASE <= last_token < BOMB_BASE + BOMB_COUNT)
    ):
        # After BOMB → POS (strictly increasing), BAR, or END
        min_next = last_pos_in_bar + 1
        if min_next < POS_COUNT:
            mask[POS_BASE + min_next: POS_BASE + POS_COUNT] = True
        mask[BAR] = True
        mask[END] = True

    else:
        # Unknown state — allow everything except PAD/START
        mask[2:] = True

    return mask


# Generous per-bar token allowance for sizing the generation budget: 1 BAR
# token + up to ~6 active note positions, each up to 4 tokens (POS/LEFT/
# RIGHT/BOMB). Real maps essentially never exceed this even at ExpertPlus
# density, so this is a safety margin, not a typical case.
_TOKENS_PER_BAR_BUDGET = 25


def _max_generation_tokens(n_frames: int, max_seq_len: int) -> int:
    """Token budget for one generate() call, sized to the window it covers.

    config.max_seq_len (the training-time sequence length) is too small to
    guarantee covering a full audio window at anything but low note density:
    a max_audio_len=4096 window spans 64 bars, and the grammar requires one
    BAR token per bar plus per-note tokens — a moderately dense window can
    exceed 1024 tokens well before the last bar, silently truncating the
    back half of the window (which is exactly the half kept by
    generate_full_song's midpoint-ownership stitching, producing dead gaps
    followed by a pile-up wherever the next window happens to catch up).
    Scale the budget to the window's actual bar count instead, using
    max_seq_len only as a floor so short single-window songs are unaffected.
    """
    bars_in_window = max(1, -(-n_frames // SUBDIVISIONS_PER_BAR))  # ceil div
    return max(max_seq_len, 4 + bars_in_window * _TOKENS_PER_BAR_BUDGET)


def _sample_with_filter(
    logits: torch.Tensor,
    temperature: float = 1.0,
    top_k: int = 0,
    top_p: float = 1.0,
) -> int:
    """Sample a token from logits with temperature, top-k, and top-p filtering."""
    if temperature <= 0:
        return logits.argmax().item()

    logits = logits / temperature

    # Top-k filtering
    if top_k > 0:
        top_k = min(top_k, logits.size(-1))
        values, _ = torch.topk(logits, top_k)
        min_val = values[-1]
        logits = torch.where(logits < min_val, torch.full_like(logits, float("-inf")), logits)

    # Top-p (nucleus) filtering
    if top_p < 1.0:
        sorted_logits, sorted_indices = torch.sort(logits, descending=True)
        cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
        # Remove tokens with cumulative prob above threshold
        sorted_mask = cumulative_probs - F.softmax(sorted_logits, dim=-1) >= top_p
        sorted_logits[sorted_mask] = float("-inf")
        # Scatter back
        logits = torch.zeros_like(logits).scatter(0, sorted_indices, sorted_logits)

    probs = F.softmax(logits, dim=-1)
    return torch.multinomial(probs, 1).item()


@torch.no_grad()
def generate(
    model: BeatWeaverModel,
    mel_spectrogram: torch.Tensor,
    difficulty: str,
    config: ModelConfig,
    temperature: float = 1.0,
    top_k: int = 0,
    top_p: float = 1.0,
    seed: int | None = None,
    mel_mask: torch.Tensor | None = None,
) -> list[int]:
    """Generate a token sequence autoregressively.

    Args:
        model: Trained BeatWeaverModel.
        mel_spectrogram: (n_mels, T_audio) — single spectrogram (no batch dim).
        difficulty: Difficulty name (e.g., "Expert").
        config: Model configuration.
        temperature: Sampling temperature (0 = greedy).
        top_k: Top-k filtering (0 = disabled).
        top_p: Top-p / nucleus filtering (1.0 = disabled).
        seed: Random seed for reproducibility.
        mel_mask: (T_audio,) — True for valid positions.

    Returns:
        List of token IDs including START and END.
    """
    if seed is not None:
        torch.manual_seed(seed)

    model.eval()
    device = next(model.parameters()).device

    # Prepare mel: add batch dimension
    mel = mel_spectrogram.unsqueeze(0).to(device)  # (1, n_mels, T_audio)
    if mel_mask is not None:
        mel_mask = mel_mask.unsqueeze(0).to(device)  # (1, T_audio)

    # Encode audio once
    memory = model.encoder(mel, mel_mask)

    # Binary "cube here" head bias: one logit per audio frame, added to the
    # matching POS_p logit so the decoder is nudged toward real onsets
    # instead of skipping straight to BAR. Computed once up front since the
    # audio encoding doesn't change across steps. No-op unless the model has
    # the head and config.cube_head_bias_scale > 0 (see ModelConfig).
    cube_bias = None
    if (
        config.use_cube_head
        and config.cube_head_bias_scale > 0
        and getattr(model, "cube_head", None) is not None
    ):
        cube_bias = (
            model.cube_head(memory).squeeze(-1).squeeze(0) * config.cube_head_bias_scale
        )  # (T_audio,)

    # Start with [START, DIFF_x]
    diff_token = difficulty_to_token(difficulty)
    tokens = [START, diff_token]
    last_pos_in_bar = -1  # Track last POS offset in current bar
    current_bar = -1  # Track bar index to map POS_p -> absolute audio frame

    max_tokens = _max_generation_tokens(mel_spectrogram.shape[1], config.max_seq_len)
    for _ in range(max_tokens - 2):
        # Prepare decoder input
        token_tensor = torch.tensor([tokens], dtype=torch.long, device=device)
        token_mask = torch.ones(1, len(tokens), dtype=torch.bool, device=device)

        logits = model.decoder(token_tensor, memory, token_mask, mel_mask)
        # logits: (1, seq_len, vocab_size) — take last position
        next_logits = logits[0, -1]  # (vocab_size,)

        # Apply grammar mask (with position tracking for one-note-per-color-per-beat)
        grammar_mask = _build_grammar_mask(
            tokens[-1], last_pos_in_bar, include_bombs=config.include_bombs,
        ).to(device)
        next_logits[~grammar_mask] = float("-inf")

        # Bias POS logits toward frames the cube head thinks are real onsets.
        # Frame = current_bar * SUBDIVISIONS_PER_BAR + p, matching how
        # generate_full_song already maps frames to beats.
        if cube_bias is not None and current_bar >= 0:
            frame_start = current_bar * SUBDIVISIONS_PER_BAR
            frame_end = min(frame_start + POS_COUNT, cube_bias.size(0))
            n = frame_end - frame_start
            if n > 0:
                next_logits[POS_BASE: POS_BASE + n] += cube_bias[frame_start:frame_end]

        # Sample
        next_token = _sample_with_filter(next_logits, temperature, top_k, top_p)
        tokens.append(next_token)

        # Update position tracking
        if next_token == BAR:
            last_pos_in_bar = -1  # Reset on new bar
            current_bar += 1
        elif POS_BASE <= next_token < POS_BASE + POS_COUNT:
            last_pos_in_bar = next_token - POS_BASE

        if next_token == END:
            break

    return tokens


def generate_full_song(
    model: BeatWeaverModel,
    mel_spectrogram: torch.Tensor,
    difficulty: str,
    config: ModelConfig,
    bpm: float,
    temperature: float = 1.0,
    top_k: int = 0,
    top_p: float = 1.0,
    seed: int | None = None,
) -> list[Note]:
    """Generate a complete Beat Saber map by processing audio in overlapping windows.

    For short audio that fits in a single window, this is equivalent to calling
    generate() + decode_tokens(). For longer audio, the mel is split into
    overlapping windows, each generating a token sequence that is decoded to
    notes and merged using midpoint ownership in the overlap zones.

    Args:
        model: Trained BeatWeaverModel.
        mel_spectrogram: (n_mels, T_audio) — full beat-aligned spectrogram.
        difficulty: Difficulty name (e.g., "Expert").
        config: Model configuration.
        bpm: Song BPM (needed for beat offset calculation and token decoding).
        temperature: Sampling temperature.
        top_k: Top-k filtering (0 = disabled).
        top_p: Top-p / nucleus filtering (1.0 = disabled).
        seed: Random seed for reproducibility.

    Returns:
        List of Note objects spanning the full song, sorted by beat.
    """
    total_frames = mel_spectrogram.shape[1]
    max_len = config.max_audio_len

    # Single window — generate directly
    if total_frames <= max_len:
        tokens = generate(
            model, mel_spectrogram, difficulty, config,
            temperature=temperature, top_k=top_k, top_p=top_p, seed=seed,
        )
        return decode_tokens(tokens, bpm, include_bombs=config.include_bombs)

    # Multi-window generation
    overlap = min(max_len // 4, 1024)
    stride = max_len - overlap

    # Compute window start positions
    starts: list[int] = []
    pos = 0
    while pos < total_frames:
        starts.append(pos)
        if pos + max_len >= total_frames:
            break
        pos += stride

    all_window_notes: list[tuple[int, list[Note]]] = []

    for i, start in enumerate(starts):
        end = start + max_len
        window_mel = mel_spectrogram[:, start:end]
        valid_len = window_mel.shape[1]

        # Zero-pad last window if needed, and mask the padded tail so the
        # encoder doesn't attend to it as if it were real (very quiet) audio —
        # unmasked padding previously let the model generate a long tail of
        # notes over pure silence past the real end of the song.
        window_mask = None
        if valid_len < max_len:
            pad_size = max_len - valid_len
            window_mel = torch.nn.functional.pad(window_mel, (0, pad_size))
            window_mask = torch.zeros(max_len, dtype=torch.bool)
            window_mask[:valid_len] = True

        # Use different seed per window for variety (if seed provided)
        window_seed = seed + i if seed is not None else None

        tokens = generate(
            model, window_mel, difficulty, config,
            temperature=temperature, top_k=top_k, top_p=top_p, seed=window_seed,
            mel_mask=window_mask,
        )

        # Decode tokens — notes have beats relative to window start (bar 0)
        window_notes = decode_tokens(tokens, bpm, include_bombs=config.include_bombs)

        # Offset all beats by the window's start position in frames
        # Each frame = 1/16th note subdivision, beat = frame / 16
        beat_offset = start / 16.0
        for note in window_notes:
            note.beat += beat_offset
            note.time_seconds = note.beat * 60.0 / bpm

        all_window_notes.append((start, window_notes))

    # Merge with midpoint ownership — each window owns notes up to the
    # midpoint of its overlap with the next window, and from the midpoint
    # of its overlap with the previous window.
    result: list[Note] = []
    for i, (start, notes) in enumerate(all_window_notes):
        min_beat = 0.0
        max_beat = float("inf")

        if i > 0:
            prev_start = all_window_notes[i - 1][0]
            # Overlap zone between prev and current: [start, prev_start + max_len)
            overlap_mid_frame = (start + prev_start + max_len) / 2.0
            min_beat = overlap_mid_frame / 16.0

        if i < len(all_window_notes) - 1:
            next_start = all_window_notes[i + 1][0]
            # Overlap zone between current and next: [next_start, start + max_len)
            overlap_mid_frame = (next_start + start + max_len) / 2.0
            max_beat = overlap_mid_frame / 16.0

        result.extend(n for n in notes if min_beat <= n.beat < max_beat)

    # Hard safety net: never emit notes past the real (unpadded) audio length,
    # regardless of what the last, heavily-padded window generated.
    song_end_beat = total_frames / 16.0
    result = [n for n in result if n.beat < song_end_beat]

    result.sort(key=lambda n: (n.beat, n.color))
    return result
