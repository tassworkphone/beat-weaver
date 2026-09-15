"""Token vocabulary and encode/decode for Beat Saber maps.

Converts between NormalizedBeatmap and integer token sequences.
No PyTorch dependency — pure Python + existing dataclasses.

Token vocabulary (291 base tokens, 304 when include_bombs=True):
    0       PAD
    1       START
    2       END
    3-7     DIFF_Easy .. DIFF_ExpertPlus
    8       BAR
    9-72    POS_0 .. POS_63  (1/16th note positions in a 4-beat bar)
    73      LEFT_EMPTY
    74-181  LEFT_x_y_d  (4 cols × 3 rows × 9 dirs)
    182     RIGHT_EMPTY
    183-290 RIGHT_x_y_d (4 cols × 3 rows × 9 dirs)
    291     BOMB_EMPTY       (only used when include_bombs=True)
    292-303 BOMB_x_y         (4 cols × 3 rows, no direction)

Compound note encoding: base + x * 27 + y * 9 + direction
Compound bomb encoding: BOMB_BASE + x * 3 + y  (no direction — bombs aren't handed)

include_bombs is opt-in (default False everywhere) so existing configs/checkpoints
trained on the 291-token vocabulary keep working unchanged.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

from beat_weaver.schemas.normalized import Bomb, Note, NormalizedBeatmap

logger = logging.getLogger(__name__)

# ── Special tokens ──────────────────────────────────────────────────────────

PAD = 0
START = 1
END = 2

# ── Difficulty tokens ───────────────────────────────────────────────────────

DIFF_EASY = 3
DIFF_NORMAL = 4
DIFF_HARD = 5
DIFF_EXPERT = 6
DIFF_EXPERT_PLUS = 7

_DIFF_NAMES = ["Easy", "Normal", "Hard", "Expert", "ExpertPlus"]
_DIFF_TO_TOKEN = {name: DIFF_EASY + i for i, name in enumerate(_DIFF_NAMES)}
_TOKEN_TO_DIFF = {v: k for k, v in _DIFF_TO_TOKEN.items()}

# Case-insensitive lookup + aliases (BeatSaver maps use inconsistent casing)
_DIFF_LOOKUP = {name.lower(): token for name, token in _DIFF_TO_TOKEN.items()}
_DIFF_LOOKUP["expert+"] = DIFF_EXPERT_PLUS

# ── Bar / position tokens ──────────────────────────────────────────────────

BAR = 8

POS_BASE = 9
POS_COUNT = 64  # 4 beats × 16 subdivisions

# ── Note tokens ─────────────────────────────────────────────────────────────

LEFT_EMPTY = 73
LEFT_BASE = 74
LEFT_COUNT = 108  # 4 × 3 × 9

RIGHT_EMPTY = 182
RIGHT_BASE = 183
RIGHT_COUNT = 108

VOCAB_SIZE = 291

# ── Bomb tokens (opt-in via include_bombs) ──────────────────────────────────

BOMB_EMPTY = 291
BOMB_BASE = 292
BOMB_COUNT = 12  # 4 cols × 3 rows, no direction — bombs aren't handed

VOCAB_SIZE_WITH_BOMBS = BOMB_BASE + BOMB_COUNT  # 304

# Grid constants
COLS = 4
ROWS = 3
DIRS = 9
SUBDIVISIONS_PER_BEAT = 16
BEATS_PER_BAR = 4
SUBDIVISIONS_PER_BAR = SUBDIVISIONS_PER_BEAT * BEATS_PER_BAR  # 64


def vocab_size(include_bombs: bool = False) -> int:
    """Return the vocabulary size for a given feature set. Use for config.vocab_size."""
    return VOCAB_SIZE_WITH_BOMBS if include_bombs else VOCAB_SIZE


def _encode_note_token(base: int, x: int, y: int, direction: int) -> int:
    """Encode a note placement into a compound token ID."""
    assert 0 <= x < COLS
    assert 0 <= y < ROWS
    assert 0 <= direction < DIRS
    return base + x * 27 + y * 9 + direction


def _decode_note_token(token: int, base: int) -> tuple[int, int, int]:
    """Decode a compound note token into (x, y, direction)."""
    offset = token - base
    x = offset // 27
    y = (offset % 27) // 9
    direction = offset % 9
    return x, y, direction


def _encode_bomb_token(x: int, y: int) -> int:
    """Encode a bomb placement into a compound token ID (no direction)."""
    assert 0 <= x < COLS
    assert 0 <= y < ROWS
    return BOMB_BASE + x * ROWS + y


def _decode_bomb_token(token: int) -> tuple[int, int]:
    """Decode a compound bomb token into (x, y)."""
    offset = token - BOMB_BASE
    x = offset // ROWS
    y = offset % ROWS
    return x, y


def difficulty_to_token(difficulty: str) -> int:
    """Map difficulty name to token ID. Raises ValueError for unknown."""
    token = _DIFF_LOOKUP.get(difficulty.lower())
    if token is None:
        raise ValueError(
            f"Unknown difficulty {difficulty!r}. "
            f"Expected one of: {_DIFF_NAMES}"
        )
    return token


def token_to_difficulty(token: int) -> str:
    """Map token ID to difficulty name. Raises ValueError for non-difficulty token."""
    name = _TOKEN_TO_DIFF.get(token)
    if name is None:
        raise ValueError(f"Token {token} is not a difficulty token (expected {DIFF_EASY}-{DIFF_EXPERT_PLUS})")
    return name


def _quantize_beat(beat: float) -> tuple[int, int]:
    """Quantize a beat value to (bar_index, subdivision_within_bar).

    Returns 0-based bar index and 0-63 subdivision position.
    """
    total_subdivisions = round(beat * SUBDIVISIONS_PER_BEAT)
    bar_index = total_subdivisions // SUBDIVISIONS_PER_BAR
    sub_in_bar = total_subdivisions % SUBDIVISIONS_PER_BAR
    return bar_index, sub_in_bar


def encode_beatmap(beatmap: NormalizedBeatmap, include_bombs: bool = False) -> list[int]:
    """Encode a NormalizedBeatmap into a token ID sequence.

    Steps:
        1. Sort notes (and bombs, if include_bombs) by beat
        2. Quantize beats to 1/16th note grid
        3. Group notes/bombs by quantized position
        4. Emit: START DIFF_x BAR POS_p LEFT_tok RIGHT_tok [BOMB_tok] ... BAR ... END
        5. Handle multiple notes at same position (one per hand); one bomb per
           position (rare simultaneous multi-bomb clusters collapse to the first).
    """
    difficulty = beatmap.difficulty_info.difficulty
    diff_token = difficulty_to_token(difficulty)

    # Sort notes by beat
    notes = sorted(beatmap.notes, key=lambda n: n.beat)
    bombs = sorted(beatmap.bombs, key=lambda b: b.beat) if include_bombs else []

    if not notes and not bombs:
        return [START, diff_token, END]

    # Group notes by (bar_index, subdivision)
    groups: dict[tuple[int, int], list[Note]] = {}
    for note in notes:
        key = _quantize_beat(note.beat)
        groups.setdefault(key, []).append(note)

    # Group bombs by (bar_index, subdivision) — may occur at positions with no notes
    bomb_groups: dict[tuple[int, int], list[Bomb]] = {}
    for bomb in bombs:
        key = _quantize_beat(bomb.beat)
        bomb_groups.setdefault(key, []).append(bomb)

    # Find the range of bars we need (union of note and bomb positions)
    all_keys = set(groups) | set(bomb_groups)
    max_bar = max(bar for bar, _ in all_keys)

    tokens: list[int] = [START, diff_token]

    for bar_idx in range(max_bar + 1):
        tokens.append(BAR)

        # Collect all subdivisions in this bar that have notes or bombs
        bar_subs = sorted(
            sub for (b, sub) in all_keys if b == bar_idx
        )

        for sub in bar_subs:
            tokens.append(POS_BASE + sub)

            group_notes = groups.get((bar_idx, sub), [])
            left_note: Note | None = None
            right_note: Note | None = None

            for note in group_notes:
                if note.color == 0:  # Red/Left
                    if left_note is not None:
                        logger.debug(
                            "Duplicate left note at beat %s, keeping first",
                            note.beat,
                        )
                        continue
                    left_note = note
                elif note.color == 1:  # Blue/Right
                    if right_note is not None:
                        logger.debug(
                            "Duplicate right note at beat %s, keeping first",
                            note.beat,
                        )
                        continue
                    right_note = note

            # Emit left token
            if left_note is not None:
                tokens.append(
                    _encode_note_token(
                        LEFT_BASE, left_note.x, left_note.y,
                        left_note.cut_direction,
                    )
                )
            else:
                tokens.append(LEFT_EMPTY)

            # Emit right token
            if right_note is not None:
                tokens.append(
                    _encode_note_token(
                        RIGHT_BASE, right_note.x, right_note.y,
                        right_note.cut_direction,
                    )
                )
            else:
                tokens.append(RIGHT_EMPTY)

            # Emit bomb token
            if include_bombs:
                position_bombs = bomb_groups.get((bar_idx, sub), [])
                if position_bombs:
                    if len(position_bombs) > 1:
                        logger.debug(
                            "Multiple bombs at beat %s, keeping first",
                            position_bombs[0].beat,
                        )
                    bomb = position_bombs[0]
                    tokens.append(_encode_bomb_token(bomb.x, bomb.y))
                else:
                    tokens.append(BOMB_EMPTY)

    tokens.append(END)
    return tokens


def decode_tokens(token_ids: list[int], bpm: float, include_bombs: bool = False) -> list[Note]:
    """Decode a token ID sequence back into a list of Note objects.

    Args:
        token_ids: Token sequence (should start with START and end with END).
        bpm: Beats per minute for computing time_seconds.
        include_bombs: Must match the value used to encode/generate this sequence
            (whether a BOMB slot follows RIGHT at each position). Decoded bombs are
            returned as Note objects with color=3 — matching Beat Saber v2's own
            convention that bombs are just another `_notes` entry with `_type: 3` —
            so callers (exporter, evaluation) don't need a separate bomb type.

    Returns:
        List of Note objects sorted by beat. Bomb entries (color=3) have
        cut_direction=0 (ignored by the game for bombs).
    """
    notes: list[Note] = []
    current_bar = -1
    current_sub = 0

    i = 0
    while i < len(token_ids):
        tok = token_ids[i]

        if tok in (PAD, START, END) or DIFF_EASY <= tok <= DIFF_EXPERT_PLUS:
            i += 1
            continue

        if tok == BAR:
            current_bar += 1
            i += 1
            continue

        if POS_BASE <= tok < POS_BASE + POS_COUNT:
            current_sub = tok - POS_BASE
            beat = (current_bar * SUBDIVISIONS_PER_BAR + current_sub) / SUBDIVISIONS_PER_BEAT
            time_seconds = beat * 60.0 / bpm

            # Expect LEFT then RIGHT (then BOMB, if include_bombs) token next
            i += 1
            if i < len(token_ids):
                left_tok = token_ids[i]
                if LEFT_BASE <= left_tok < LEFT_BASE + LEFT_COUNT:
                    x, y, d = _decode_note_token(left_tok, LEFT_BASE)
                    notes.append(Note(
                        beat=beat, time_seconds=time_seconds,
                        x=x, y=y, color=0, cut_direction=d,
                    ))
                # LEFT_EMPTY — skip
                i += 1

            if i < len(token_ids):
                right_tok = token_ids[i]
                if RIGHT_BASE <= right_tok < RIGHT_BASE + RIGHT_COUNT:
                    x, y, d = _decode_note_token(right_tok, RIGHT_BASE)
                    notes.append(Note(
                        beat=beat, time_seconds=time_seconds,
                        x=x, y=y, color=1, cut_direction=d,
                    ))
                # RIGHT_EMPTY — skip
                i += 1

            if include_bombs and i < len(token_ids):
                bomb_tok = token_ids[i]
                if BOMB_BASE <= bomb_tok < BOMB_BASE + BOMB_COUNT:
                    x, y = _decode_bomb_token(bomb_tok)
                    notes.append(Note(
                        beat=beat, time_seconds=time_seconds,
                        x=x, y=y, color=3, cut_direction=0,
                    ))
                # BOMB_EMPTY — skip
                i += 1
            continue

        # Unknown token — skip
        i += 1

    return sorted(notes, key=lambda n: n.beat)


@dataclass
class TokenInfo:
    """Human-readable description of a token for debugging."""

    id: int
    name: str
    category: str


def describe_token(token_id: int) -> TokenInfo:
    """Return human-readable info about a token ID."""
    if token_id == PAD:
        return TokenInfo(token_id, "PAD", "special")
    if token_id == START:
        return TokenInfo(token_id, "START", "special")
    if token_id == END:
        return TokenInfo(token_id, "END", "special")
    if DIFF_EASY <= token_id <= DIFF_EXPERT_PLUS:
        name = _TOKEN_TO_DIFF[token_id]
        return TokenInfo(token_id, f"DIFF_{name}", "difficulty")
    if token_id == BAR:
        return TokenInfo(token_id, "BAR", "structure")
    if POS_BASE <= token_id < POS_BASE + POS_COUNT:
        pos = token_id - POS_BASE
        beat_in_bar = pos / SUBDIVISIONS_PER_BEAT
        return TokenInfo(token_id, f"POS_{pos} (beat {beat_in_bar:.4g})", "position")
    if token_id == LEFT_EMPTY:
        return TokenInfo(token_id, "LEFT_EMPTY", "note")
    if LEFT_BASE <= token_id < LEFT_BASE + LEFT_COUNT:
        x, y, d = _decode_note_token(token_id, LEFT_BASE)
        return TokenInfo(token_id, f"LEFT({x},{y},d={d})", "note")
    if token_id == RIGHT_EMPTY:
        return TokenInfo(token_id, "RIGHT_EMPTY", "note")
    if RIGHT_BASE <= token_id < RIGHT_BASE + RIGHT_COUNT:
        x, y, d = _decode_note_token(token_id, RIGHT_BASE)
        return TokenInfo(token_id, f"RIGHT({x},{y},d={d})", "note")
    if token_id == BOMB_EMPTY:
        return TokenInfo(token_id, "BOMB_EMPTY", "bomb")
    if BOMB_BASE <= token_id < BOMB_BASE + BOMB_COUNT:
        x, y = _decode_bomb_token(token_id)
        return TokenInfo(token_id, f"BOMB({x},{y})", "bomb")
    return TokenInfo(token_id, f"UNKNOWN_{token_id}", "unknown")
