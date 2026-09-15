"""Rule-based wall/obstacle post-processing.

Obstacles are deliberately NOT part of the token model (see RESEARCH.md's "Scope
decision" and the tokenizer's bomb-only extension) — they have temporal extent
(duration/width/height) rather than being instantaneous like notes/bombs, which
doesn't fit the fixed-slot compound-token grammar without either a much larger
vocabulary or a variable-length sub-sequence. This matches how other Beat Saber
automappers that do support walls (e.g. a from-scratch rebuild surveyed during
planning) handle them too: excluded from the model's token stream entirely and
added afterward by a deterministic rule-based layer.

This module has two halves:
    1. mine_obstacle_stats() — computes real per-difficulty placement statistics
       (density, duration, width, height) from already-processed Parquet data,
       so the heuristic is grounded in how humans actually place walls rather
       than guessed constants.
    2. generate_obstacles() — places walls into the gaps of a generated note
       stream using those statistics, avoiding notes/bombs so nothing becomes
       unplayable.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

from beat_weaver.schemas.normalized import Note, Obstacle

logger = logging.getLogger(__name__)

# Fallback statistics used until mine_obstacle_stats() has been run on real data.
# Values are rough placeholders based on Beat Saber design convention (RESEARCH.md:
# obstacle `_type` 0=full-height/height 5, 1=crouch/height 3) — walls are far rarer
# than notes, so these intentionally err on the sparse side.
DEFAULT_OBSTACLE_STATS: dict[str, dict] = {
    "Easy": {"per_minute": 0.5, "duration_beats": 2.0, "width": 1, "crouch_fraction": 0.05},
    "Normal": {"per_minute": 0.8, "duration_beats": 2.0, "width": 1, "crouch_fraction": 0.1},
    "Hard": {"per_minute": 1.5, "duration_beats": 1.5, "width": 1, "crouch_fraction": 0.15},
    "Expert": {"per_minute": 2.5, "duration_beats": 1.0, "width": 1, "crouch_fraction": 0.2},
    "ExpertPlus": {"per_minute": 3.0, "duration_beats": 1.0, "width": 1, "crouch_fraction": 0.25},
}

FULL_HEIGHT = 5
CROUCH_HEIGHT = 3
MIN_GAP_BEATS = 2.0  # Shortest gap worth filling with a wall — comfortably more
# than typical Expert+ note spacing, so densely-mapped passages get no walls at all.
NOTE_BUFFER_BEATS = 0.25  # Keep walls this far from adjacent notes
MIN_USABLE_BEATS = 0.5  # Minimum wall span after buffering, or skip the gap

# Vanilla obstacle bounds for filtering mined stats. A large fraction of
# BeatSaver's obstacle data comes from mapping-extension/"wall art" maps whose
# width/duration values are way outside normal gameplay (observed: width=127
# and width=0 alone account for ~80% of raw rows in a real corpus — 127 is an
# int8 overflow artifact, not an intentional 127-lane-wide wall; duration_beats
# even goes negative). Excluding these keeps the mined stats representative of
# how walls are actually placed in normal play, matching InfernoSaber's own
# documented practice of excluding maps with "custom modded data" (RESEARCH.md).
_MAX_VANILLA_WIDTH = 4
_MAX_VANILLA_DURATION_BEATS = 64.0


def mine_obstacle_stats(processed_dir: Path) -> dict[str, dict]:
    """Compute real per-difficulty obstacle placement stats from Parquet data.

    Returns a dict shaped like DEFAULT_OBSTACLE_STATS, one entry per difficulty
    found in the data. Call this after `beat-weaver process` has populated
    `data/processed/obstacles_*.parquet` and `notes_*.parquet`; save the result
    (see save_obstacle_stats) so generate_obstacles() can pick it up automatically.
    """
    from beat_weaver.storage.writer import read_notes_parquet, read_obstacles_parquet

    notes_df = read_notes_parquet(processed_dir).to_pandas()
    # Song duration per (song_hash, difficulty, characteristic), used to normalize
    # obstacle counts into a per-minute rate.
    duration_minutes = (
        notes_df.groupby(["song_hash", "difficulty", "characteristic"])["time_seconds"]
        .max() / 60.0
    )

    try:
        obstacles_df = read_obstacles_parquet(processed_dir).to_pandas()
    except FileNotFoundError:
        logger.warning("No obstacles Parquet files in %s — returning defaults", processed_dir)
        return {k: dict(v) for k, v in DEFAULT_OBSTACLE_STATS.items()}

    n_raw = len(obstacles_df)
    obstacles_df = obstacles_df[
        (obstacles_df["width"] >= 1)
        & (obstacles_df["width"] <= _MAX_VANILLA_WIDTH)
        & (obstacles_df["duration_beats"] > 0)
        & (obstacles_df["duration_beats"] <= _MAX_VANILLA_DURATION_BEATS)
    ]
    logger.info(
        "Obstacle stats: kept %d/%d rows after excluding modded/out-of-range values (%.1f%%)",
        len(obstacles_df), n_raw, 100.0 * len(obstacles_df) / n_raw if n_raw else 0.0,
    )

    stats: dict[str, dict] = {}
    for difficulty, group in obstacles_df.groupby("difficulty"):
        if len(group) < 5:
            # Too few clean samples after filtering to trust a per-difficulty
            # estimate — fall back to defaults for this difficulty only.
            continue
        keys = list(zip(group["song_hash"], group["difficulty"], group["characteristic"]))
        total_minutes = sum(duration_minutes.get(k, 0.0) for k in set(keys))
        per_minute = len(group) / total_minutes if total_minutes > 0 else DEFAULT_OBSTACLE_STATS.get(
            difficulty, DEFAULT_OBSTACLE_STATS["Expert"]
        )["per_minute"]

        crouch_fraction = float((group["height"] <= CROUCH_HEIGHT).mean())
        stats[difficulty] = {
            "per_minute": round(float(per_minute), 3),
            "duration_beats": round(float(group["duration_beats"].median()), 3),
            "width": int(group["width"].median()),
            "crouch_fraction": round(crouch_fraction, 3),
        }

    for difficulty, defaults in DEFAULT_OBSTACLE_STATS.items():
        stats.setdefault(difficulty, dict(defaults))

    return stats


def save_obstacle_stats(stats: dict[str, dict], path: Path) -> None:
    """Save mined stats to JSON for generate_obstacles() to load later."""
    Path(path).write_text(json.dumps(stats, indent=2), encoding="utf-8")


def load_obstacle_stats(path: Path) -> dict[str, dict] | None:
    """Load previously mined stats, or None if the file doesn't exist."""
    path = Path(path)
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def _find_gaps(occupied_beats: list[float], end_beat: float) -> list[tuple[float, float]]:
    """Find [start, end) beat ranges with no notes/bombs, sorted largest-first."""
    if not occupied_beats:
        return [(0.0, end_beat)] if end_beat > 0 else []

    sorted_beats = sorted(set(occupied_beats))
    gaps: list[tuple[float, float]] = []

    if sorted_beats[0] > 0:
        gaps.append((0.0, sorted_beats[0]))
    for a, b in zip(sorted_beats, sorted_beats[1:]):
        if b - a >= MIN_GAP_BEATS:
            gaps.append((a, b))
    if end_beat > sorted_beats[-1]:
        gaps.append((sorted_beats[-1], end_beat))

    gaps = [(s, e) for s, e in gaps if e - s >= MIN_GAP_BEATS]
    gaps.sort(key=lambda g: g[1] - g[0], reverse=True)
    return gaps


def generate_obstacles(
    notes: list[Note],
    bombs: list[Note],
    bpm: float,
    difficulty: str,
    stats: dict[str, dict] | None = None,
) -> list[Obstacle]:
    """Place walls into the gaps of a generated note stream.

    Args:
        notes: Generated color notes (color 0/1 — bombs must be passed separately).
        bombs: Generated bomb entries (as returned by decode_tokens with
            include_bombs=True — Note objects with color=3), used only to avoid
            overlapping a wall with a bomb's lane.
        bpm: Song BPM, for beat<->seconds conversion.
        difficulty: Difficulty name, selects placement statistics.
        stats: Mined stats from mine_obstacle_stats()/load_obstacle_stats(); falls
            back to DEFAULT_OBSTACLE_STATS when not provided.

    Returns:
        List of Obstacle objects, sorted by beat. Empty if there are no notes
        or no gaps large enough to place a wall.
    """
    if not notes:
        return []

    diff_stats = (stats or DEFAULT_OBSTACLE_STATS).get(
        difficulty, DEFAULT_OBSTACLE_STATS["Expert"]
    )

    end_beat = max(n.beat for n in notes) + 2.0  # small tail buffer
    duration_minutes = (end_beat * 60.0 / bpm) / 60.0
    target_count = max(0, round(diff_stats["per_minute"] * duration_minutes))
    if target_count == 0:
        return []

    occupied = [n.beat for n in notes] + [b.beat for b in bombs]
    gaps = _find_gaps(occupied, end_beat)

    # Map each occupied beat to the lanes (x) in use near it, for lane avoidance.
    lanes_by_beat: dict[float, set[int]] = {}
    for n in notes:
        lanes_by_beat.setdefault(n.beat, set()).add(n.x)
    for b in bombs:
        lanes_by_beat.setdefault(b.beat, set()).add(b.x)

    def _lanes_near(beat: float, radius: float) -> set[int]:
        used: set[int] = set()
        for b, lanes in lanes_by_beat.items():
            if abs(b - beat) <= radius:
                used |= lanes
        return used

    obstacles: list[Obstacle] = []
    width = max(1, min(4, int(diff_stats["width"])))
    max_duration = diff_stats["duration_beats"]
    crouch_fraction = diff_stats["crouch_fraction"]

    for gap_start, gap_end in gaps:
        if len(obstacles) >= target_count:
            break

        usable_start = gap_start + NOTE_BUFFER_BEATS
        usable_end = gap_end - NOTE_BUFFER_BEATS
        if usable_end - usable_start < MIN_USABLE_BEATS:
            continue

        wall_duration = min(max_duration, usable_end - usable_start)
        wall_beat = usable_start

        # Pick a lane free of notes/bombs immediately before and after the wall.
        blocked_lanes = _lanes_near(gap_start, NOTE_BUFFER_BEATS) | _lanes_near(
            gap_end, NOTE_BUFFER_BEATS
        )
        candidate_lanes = [x for x in range(4 - width + 1) if not (
            set(range(x, x + width)) & blocked_lanes
        )]
        if not candidate_lanes:
            continue
        x = candidate_lanes[len(obstacles) % len(candidate_lanes)]

        height = CROUCH_HEIGHT if (len(obstacles) % 100) / 100.0 < crouch_fraction else FULL_HEIGHT

        obstacles.append(Obstacle(
            beat=wall_beat,
            time_seconds=wall_beat * 60.0 / bpm,
            duration_beats=wall_duration,
            x=x,
            y=0,  # obstacles rise from the floor
            width=width,
            height=height,
        ))

    obstacles.sort(key=lambda o: o.beat)
    return obstacles
