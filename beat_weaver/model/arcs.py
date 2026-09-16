"""Rule-based arc (slider) post-processing.

Like walls (see `beat_weaver.model.obstacles`), arcs are deliberately NOT part of
the token model. InfernoSaber's actual source (`map_creation/gen_sliders.py`,
surveyed while planning this feature — see CLAUDE.md's Open Questions) generates
arcs the same way: after notes are placed, walk consecutive same-color notes and
connect eligible pairs into an arc, using the two notes' own positions/directions
as the arc's head/tail. No model or tokenizer changes are needed.

This module has two halves, mirroring `obstacles.py`:
    1. mine_arc_stats() — computes real per-difficulty placement statistics
       (density, time gap, movement distance) from already-processed Parquet
       data, so the heuristic is grounded in how arcs are actually placed
       rather than guessed constants.
    2. generate_arcs() — connects eligible same-color note pairs in a generated
       note stream using those statistics.
"""

from __future__ import annotations

import json
import logging
import math
from pathlib import Path

from beat_weaver.schemas.normalized import Arc, Note

logger = logging.getLogger(__name__)

# Fallback statistics used until mine_arc_stats() has been run on real data.
# time_gap bounds and movement_minimum follow InfernoSaber's documented
# reference constants (slider_time_gap=[0.5, 12.0]s, slider_movement_minimum=3);
# per_minute values are conservative placeholders — arcs are rarer than walls.
DEFAULT_ARC_STATS: dict[str, dict] = {
    "Easy": {"per_minute": 0.3, "time_gap_min": 0.5, "time_gap_max": 12.0, "movement_min": 3.0, "head_multiplier": 1.0, "tail_multiplier": 1.0},
    "Normal": {"per_minute": 0.5, "time_gap_min": 0.5, "time_gap_max": 12.0, "movement_min": 3.0, "head_multiplier": 1.0, "tail_multiplier": 1.0},
    "Hard": {"per_minute": 1.0, "time_gap_min": 0.5, "time_gap_max": 10.0, "movement_min": 2.5, "head_multiplier": 1.0, "tail_multiplier": 1.0},
    "Expert": {"per_minute": 2.0, "time_gap_min": 0.4, "time_gap_max": 8.0, "movement_min": 2.0, "head_multiplier": 1.0, "tail_multiplier": 1.0},
    "ExpertPlus": {"per_minute": 2.5, "time_gap_min": 0.3, "time_gap_max": 6.0, "movement_min": 1.5, "head_multiplier": 1.0, "tail_multiplier": 1.0},
}

# Standard grid bounds — excludes mapping-extension-positioned arcs from mining,
# matching the same standard-grid filtering used elsewhere in the pipeline
# (dataset.py's bomb/note filtering, obstacles.py's vanilla-width filtering).
_GRID_X = (0, 3)
_GRID_Y = (0, 2)


def mine_arc_stats(processed_dir: Path) -> dict[str, dict]:
    """Compute real per-difficulty arc placement stats from Parquet data.

    Returns a dict shaped like DEFAULT_ARC_STATS, one entry per difficulty
    found in the data. Call this after `beat-weaver process` has populated
    `data/processed/arcs_*.parquet` and `notes_*.parquet`; save the result
    (see save_arc_stats) so generate_arcs() can pick it up automatically.
    """
    from beat_weaver.storage.writer import read_arcs_parquet, read_notes_parquet

    notes_df = read_notes_parquet(processed_dir).to_pandas()
    duration_minutes = (
        notes_df.groupby(["song_hash", "difficulty", "characteristic"])["time_seconds"]
        .max() / 60.0
    )

    try:
        arcs_df = read_arcs_parquet(processed_dir).to_pandas()
    except FileNotFoundError:
        logger.warning("No arcs Parquet files in %s — returning defaults", processed_dir)
        return {k: dict(v) for k, v in DEFAULT_ARC_STATS.items()}

    n_raw = len(arcs_df)
    arcs_df = arcs_df[
        arcs_df["x"].between(*_GRID_X)
        & arcs_df["y"].between(*_GRID_Y)
        & arcs_df["tail_x"].between(*_GRID_X)
        & arcs_df["tail_y"].between(*_GRID_Y)
        & (arcs_df["tail_time_seconds"] > arcs_df["time_seconds"])
    ]
    logger.info(
        "Arc stats: kept %d/%d rows after excluding modded/out-of-range values (%.1f%%)",
        len(arcs_df), n_raw, 100.0 * len(arcs_df) / n_raw if n_raw else 0.0,
    )

    time_gap = arcs_df["tail_time_seconds"] - arcs_df["time_seconds"]
    movement = (
        (arcs_df["tail_x"] - arcs_df["x"]) ** 2 + (arcs_df["tail_y"] - arcs_df["y"]) ** 2
    ) ** 0.5

    stats: dict[str, dict] = {}
    for difficulty, group in arcs_df.groupby("difficulty"):
        if len(group) < 5:
            # Too few clean samples after filtering to trust a per-difficulty
            # estimate — fall back to defaults for this difficulty only.
            continue
        keys = list(zip(group["song_hash"], group["difficulty"], group["characteristic"]))
        total_minutes = sum(duration_minutes.get(k, 0.0) for k in set(keys))
        per_minute = len(group) / total_minutes if total_minutes > 0 else DEFAULT_ARC_STATS.get(
            difficulty, DEFAULT_ARC_STATS["Expert"]
        )["per_minute"]

        gap = time_gap.loc[group.index]
        mv = movement.loc[group.index]
        stats[difficulty] = {
            "per_minute": round(float(per_minute), 3),
            # 10th/90th percentile rather than min/max so a handful of outliers
            # don't blow the eligibility window wide open.
            "time_gap_min": round(float(gap.quantile(0.1)), 3),
            "time_gap_max": round(float(gap.quantile(0.9)), 3),
            "movement_min": round(float(mv.quantile(0.25)), 3),
            "head_multiplier": round(float(group["head_multiplier"].median()), 3),
            "tail_multiplier": round(float(group["tail_multiplier"].median()), 3),
        }

    for difficulty, defaults in DEFAULT_ARC_STATS.items():
        stats.setdefault(difficulty, dict(defaults))

    return stats


def save_arc_stats(stats: dict[str, dict], path: Path) -> None:
    """Save mined stats to JSON for generate_arcs() to load later."""
    Path(path).write_text(json.dumps(stats, indent=2), encoding="utf-8")


def load_arc_stats(path: Path) -> dict[str, dict] | None:
    """Load previously mined stats, or None if the file doesn't exist."""
    path = Path(path)
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def _movement(a: Note, b: Note) -> float:
    return math.hypot(b.x - a.x, b.y - a.y)


def generate_arcs(
    notes: list[Note],
    bpm: float,
    difficulty: str,
    stats: dict[str, dict] | None = None,
) -> list[Arc]:
    """Connect eligible same-color note pairs in a generated note stream into arcs.

    Args:
        notes: Generated color notes (color 0/1 — bombs must not be passed in).
        bpm: Song BPM (unused directly here since Note already carries
            time_seconds, but kept for API symmetry with generate_obstacles).
        difficulty: Difficulty name, selects placement statistics.
        stats: Mined stats from mine_arc_stats()/load_arc_stats(); falls back
            to DEFAULT_ARC_STATS when not provided.

    Returns:
        List of Arc objects, sorted by head beat. Empty if there are fewer
        than 2 notes or no eligible pairs.
    """
    if len(notes) < 2:
        return []

    diff_stats = (stats or DEFAULT_ARC_STATS).get(difficulty, DEFAULT_ARC_STATS["Expert"])

    end_beat = max(n.beat for n in notes)
    duration_minutes = (end_beat * 60.0 / bpm) / 60.0 if bpm > 0 else 0.0
    target_count = max(0, round(diff_stats["per_minute"] * duration_minutes))
    if target_count == 0:
        return []

    gap_min = diff_stats["time_gap_min"]
    gap_max = diff_stats["time_gap_max"]
    movement_min = diff_stats["movement_min"]

    # Consecutive same-color pairs, walking each color's own chronological
    # sub-sequence (matches how arcs connect a note to its next same-color
    # note, not necessarily the very next note in the merged stream).
    by_color: dict[int, list[Note]] = {}
    for n in sorted(notes, key=lambda n: n.beat):
        by_color.setdefault(n.color, []).append(n)

    eligible: list[tuple[Note, Note]] = []
    for color_notes in by_color.values():
        for a, b in zip(color_notes, color_notes[1:]):
            time_gap = b.time_seconds - a.time_seconds
            if not (gap_min <= time_gap <= gap_max):
                continue
            if _movement(a, b) < movement_min:
                continue
            eligible.append((a, b))

    eligible.sort(key=lambda pair: pair[0].beat)
    if not eligible:
        return []

    # Evenly downsample to target_count rather than always taking the
    # earliest eligible pairs, so arcs spread across the whole song.
    if len(eligible) > target_count:
        stride = len(eligible) / target_count
        eligible = [eligible[int(i * stride)] for i in range(target_count)]

    # A note can only be one arc's head or tail — matches how humans map
    # arcs (a note isn't simultaneously slid into two different directions).
    used: set[int] = set()
    arcs: list[Arc] = []
    for a, b in eligible:
        if id(a) in used or id(b) in used:
            continue
        used.add(id(a))
        used.add(id(b))
        arcs.append(Arc(
            beat=a.beat,
            time_seconds=a.time_seconds,
            x=a.x,
            y=a.y,
            color=a.color,
            cut_direction=a.cut_direction,
            head_multiplier=diff_stats["head_multiplier"],
            tail_beat=b.beat,
            tail_time_seconds=b.time_seconds,
            tail_x=b.x,
            tail_y=b.y,
            tail_cut_direction=b.cut_direction,
            tail_multiplier=diff_stats["tail_multiplier"],
            mid_anchor_mode=0,
        ))

    arcs.sort(key=lambda a: a.beat)
    return arcs
