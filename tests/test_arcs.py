"""Tests for rule-based arc (slider) post-processing."""

import json

import pytest

pa = pytest.importorskip("pyarrow")
pq = pytest.importorskip("pyarrow.parquet")

from beat_weaver.model.arcs import (
    DEFAULT_ARC_STATS,
    generate_arcs,
    load_arc_stats,
    mine_arc_stats,
    save_arc_stats,
)
from beat_weaver.schemas.normalized import Note


def _note(beat: float, x: int = 0, y: int = 0, color: int = 0) -> Note:
    return Note(beat=beat, time_seconds=beat * 0.5, x=x, y=y, color=color, cut_direction=0)


def _write_processed_dir(tmp_path):
    """Minimal notes_*.parquet + arcs_*.parquet for mine_arc_stats().

    Includes 6 clean Expert arc rows plus 2 mapping-extension-style outliers
    (out-of-standard-grid x) to verify filtering.
    """
    from beat_weaver.storage.writer import ARCS_SCHEMA, NOTES_SCHEMA

    processed = tmp_path / "processed"
    processed.mkdir()

    notes = {
        "song_hash": ["h1", "h1"],
        "source": ["beatsaver", "beatsaver"],
        "difficulty": ["Expert", "Expert"],
        "characteristic": ["Standard", "Standard"],
        "bpm": [120.0, 120.0],
        "beat": [0.0, 120.0],
        "time_seconds": [0.0, 60.0],  # song is ~60s = 1 minute long
        "x": [0, 1],
        "y": [0, 0],
        "color": [0, 1],
        "cut_direction": [1, 1],
        "angle_offset": [0, 0],
    }
    pq.write_table(pa.table(notes, schema=NOTES_SCHEMA), processed / "notes_0000.parquet")

    n_clean = 6
    arcs = {
        "song_hash": ["h1"] * (n_clean + 2),
        "source": ["beatsaver"] * (n_clean + 2),
        "difficulty": ["Expert"] * (n_clean + 2),
        "characteristic": ["Standard"] * (n_clean + 2),
        "bpm": [120.0] * (n_clean + 2),
        "beat": [float(i * 4) for i in range(n_clean)] + [40.0, 44.0],
        "time_seconds": [float(i) for i in range(n_clean)] + [20.0, 22.0],
        "x": [0, 1, 0, 1, 0, 1] + [127, 0],  # outliers: mapping-extension x
        "y": [0] * (n_clean + 2),
        "color": [0] * (n_clean + 2),
        "cut_direction": [1] * (n_clean + 2),
        "head_multiplier": [1.0] * (n_clean + 2),
        "tail_beat": [float(i * 4 + 1) for i in range(n_clean)] + [41.0, 45.0],
        "tail_time_seconds": [float(i) + 1.0 for i in range(n_clean)] + [21.0, 23.0],
        "tail_x": [2, 3, 2, 3, 2, 3] + [0, 127],
        "tail_y": [1] * (n_clean + 2),
        "tail_cut_direction": [0] * (n_clean + 2),
        "tail_multiplier": [1.0] * (n_clean + 2),
        "mid_anchor_mode": [0] * (n_clean + 2),
    }
    pq.write_table(pa.table(arcs, schema=ARCS_SCHEMA), processed / "arcs_0000.parquet")

    return processed


class TestGenerateArcs:
    def test_empty_notes_returns_empty(self):
        assert generate_arcs([], bpm=120.0, difficulty="Expert") == []

    def test_single_note_returns_empty(self):
        assert generate_arcs([_note(0.0)], bpm=120.0, difficulty="Expert") == []

    def test_connects_eligible_same_color_pair(self):
        """Two same-color notes with a real time gap and movement should connect.

        Beats start at 30 (not 0) so the song is long enough that the default
        Expert per_minute density target rounds up to >= 1 arc.
        """
        notes = [_note(30.0, x=0, y=0, color=0), _note(34.0, x=3, y=2, color=0)]
        arcs = generate_arcs(notes, bpm=120.0, difficulty="Expert")
        assert len(arcs) == 1
        assert arcs[0].x == 0 and arcs[0].y == 0
        assert arcs[0].tail_x == 3 and arcs[0].tail_y == 2

    def test_different_colors_not_connected(self):
        notes = [_note(30.0, color=0), _note(34.0, x=3, y=2, color=1)]
        arcs = generate_arcs(notes, bpm=120.0, difficulty="Expert")
        assert arcs == []

    def test_too_close_together_not_connected(self):
        """Movement below movement_min should not produce an arc."""
        notes = [_note(30.0, x=0, y=0, color=0), _note(34.0, x=0, y=0, color=0)]
        arcs = generate_arcs(notes, bpm=120.0, difficulty="Expert")
        assert arcs == []

    def test_time_gap_out_of_range_not_connected(self):
        """A pair way outside the eligible time-gap window should not connect."""
        notes = [_note(0.0, x=0, y=0, color=0), _note(2000.0, x=3, y=2, color=0)]
        stats = {"Expert": {**DEFAULT_ARC_STATS["Expert"], "time_gap_max": 8.0}}
        arcs = generate_arcs(notes, bpm=120.0, difficulty="Expert", stats=stats)
        assert arcs == []

    def test_note_used_at_most_once(self):
        """A note already claimed as a head/tail should not be reused by another arc."""
        notes = [
            _note(0.0, x=0, y=0, color=0),
            _note(4.0, x=3, y=2, color=0),
            _note(8.0, x=0, y=0, color=0),
        ]
        stats = {"Expert": {**DEFAULT_ARC_STATS["Expert"], "per_minute": 1000.0}}
        arcs = generate_arcs(notes, bpm=120.0, difficulty="Expert", stats=stats)
        used_beats = [b for arc in arcs for b in (arc.beat, arc.tail_beat)]
        assert len(used_beats) == len(set(used_beats))

    def test_respects_target_count_from_density(self):
        """Arc count is bounded by the mined per-minute density."""
        notes = [_note(float(i * 4), x=i % 4, y=0, color=0) for i in range(20)]
        stats = {"Expert": {**DEFAULT_ARC_STATS["Expert"], "per_minute": 1.0}}
        arcs = generate_arcs(notes, bpm=120.0, difficulty="Expert", stats=stats)
        # ~40 beats at 120bpm = 20s = 0.33min -> target ~0 or 1 at per_minute=1.0
        assert len(arcs) <= 1

    def test_unknown_difficulty_falls_back_to_expert_defaults(self):
        notes = [_note(0.0, x=0, y=0, color=0), _note(4.0, x=3, y=2, color=0)]
        arcs = generate_arcs(notes, bpm=120.0, difficulty="NotARealDifficulty")
        assert isinstance(arcs, list)

    def test_sorted_by_beat(self):
        notes = [_note(float(i * 4), x=i % 4, y=0, color=0) for i in range(20)]
        arcs = generate_arcs(notes, bpm=120.0, difficulty="ExpertPlus")
        beats = [a.beat for a in arcs]
        assert beats == sorted(beats)


class TestArcStatsIO:
    def test_save_and_load_round_trip(self, tmp_path):
        path = tmp_path / "stats.json"
        save_arc_stats(DEFAULT_ARC_STATS, path)
        loaded = load_arc_stats(path)
        assert loaded == DEFAULT_ARC_STATS

    def test_load_missing_file_returns_none(self, tmp_path):
        assert load_arc_stats(tmp_path / "does_not_exist.json") is None


class TestMineArcStats:
    def test_returns_json_serializable_values(self, tmp_path):
        """Regression: pandas/pyarrow float32 values must be cast to native
        Python types, or json.dumps (via save_arc_stats) raises TypeError."""
        processed = _write_processed_dir(tmp_path)
        stats = mine_arc_stats(processed)
        json.dumps(stats)  # must not raise

        expert = stats["Expert"]
        assert isinstance(expert["per_minute"], float)
        assert isinstance(expert["time_gap_min"], float)
        assert isinstance(expert["time_gap_max"], float)
        assert isinstance(expert["movement_min"], float)

    def test_missing_difficulties_fall_back_to_defaults(self, tmp_path):
        processed = _write_processed_dir(tmp_path)
        stats = mine_arc_stats(processed)
        assert stats["Easy"] == DEFAULT_ARC_STATS["Easy"]

    def test_excludes_modded_outliers(self, tmp_path):
        """Mapping-extension out-of-grid rows (x=127) must not count toward
        the mined density — only the 6 clean rows should survive filtering,
        out of a 1-minute-long song, giving an exact per_minute of 6.0."""
        processed = _write_processed_dir(tmp_path)
        stats = mine_arc_stats(processed)
        assert stats["Expert"]["per_minute"] == pytest.approx(6.0)

    def test_too_few_clean_samples_falls_back_to_defaults(self, tmp_path):
        from beat_weaver.storage.writer import ARCS_SCHEMA, NOTES_SCHEMA

        processed = tmp_path / "processed"
        processed.mkdir()
        notes = {
            "song_hash": ["h1"], "source": ["beatsaver"], "difficulty": ["Hard"],
            "characteristic": ["Standard"], "bpm": [120.0], "beat": [0.0],
            "time_seconds": [0.0], "x": [0], "y": [0], "color": [0],
            "cut_direction": [1], "angle_offset": [0],
        }
        pq.write_table(pa.table(notes, schema=NOTES_SCHEMA), processed / "notes_0000.parquet")
        # Only 2 arc rows, both outliers -> 0 clean rows survive filtering
        arcs = {
            "song_hash": ["h1", "h1"], "source": ["beatsaver", "beatsaver"],
            "difficulty": ["Hard", "Hard"], "characteristic": ["Standard", "Standard"],
            "bpm": [120.0, 120.0], "beat": [1.0, 2.0], "time_seconds": [0.5, 1.0],
            "x": [127, 0], "y": [0, 0], "color": [0, 0], "cut_direction": [1, 1],
            "head_multiplier": [1.0, 1.0], "tail_beat": [2.0, 3.0],
            "tail_time_seconds": [1.0, 1.5], "tail_x": [0, 127], "tail_y": [1, 1],
            "tail_cut_direction": [0, 0], "tail_multiplier": [1.0, 1.0],
            "mid_anchor_mode": [0, 0],
        }
        pq.write_table(pa.table(arcs, schema=ARCS_SCHEMA), processed / "arcs_0000.parquet")

        stats = mine_arc_stats(processed)
        assert stats["Hard"] == DEFAULT_ARC_STATS["Hard"]

    def test_no_arcs_parquet_returns_defaults(self, tmp_path):
        from beat_weaver.storage.writer import NOTES_SCHEMA

        processed = tmp_path / "processed"
        processed.mkdir()
        notes = {
            "song_hash": ["h1"], "source": ["beatsaver"], "difficulty": ["Expert"],
            "characteristic": ["Standard"], "bpm": [120.0], "beat": [0.0],
            "time_seconds": [0.0], "x": [0], "y": [0], "color": [0],
            "cut_direction": [1], "angle_offset": [0],
        }
        pq.write_table(pa.table(notes, schema=NOTES_SCHEMA), processed / "notes_0000.parquet")

        stats = mine_arc_stats(processed)
        assert stats == DEFAULT_ARC_STATS
