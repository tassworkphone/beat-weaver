"""Tests for rule-based wall/obstacle post-processing."""

import json

import pytest

pa = pytest.importorskip("pyarrow")
pq = pytest.importorskip("pyarrow.parquet")

from beat_weaver.model.obstacles import (
    CROUCH_HEIGHT,
    DEFAULT_OBSTACLE_STATS,
    FULL_HEIGHT,
    _find_gaps,
    generate_obstacles,
    load_obstacle_stats,
    mine_obstacle_stats,
    save_obstacle_stats,
)
from beat_weaver.schemas.normalized import Note


def _note(beat: float, x: int = 0, color: int = 0) -> Note:
    return Note(beat=beat, time_seconds=beat * 0.5, x=x, y=0, color=color, cut_direction=0)


def _write_processed_dir(tmp_path):
    """Minimal notes_*.parquet + obstacles_*.parquet for mine_obstacle_stats().

    Includes 6 clean Expert obstacle rows (3 full-height, 3 crouch, all within
    vanilla width/duration bounds) plus 2 "wall art"/modded-map style outliers
    (width=127 int8-overflow artifact, negative duration) to verify filtering.
    """
    from beat_weaver.storage.writer import NOTES_SCHEMA, OBSTACLES_SCHEMA

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
    obstacles = {
        "song_hash": ["h1"] * n_clean + ["h1", "h1"],
        "source": ["beatsaver"] * (n_clean + 2),
        "difficulty": ["Expert"] * (n_clean + 2),
        "characteristic": ["Standard"] * (n_clean + 2),
        "bpm": [120.0] * (n_clean + 2),
        "beat": [float(i * 2 + 1) for i in range(n_clean)] + [30.0, 31.0],
        "time_seconds": [float(i) for i in range(n_clean)] + [15.0, 15.5],
        "duration_beats": [2.0, 4.0, 2.0, 4.0, 2.0, 4.0] + [-5.0, 3000.0],  # outliers: negative, absurd
        "x": [0, 1, 0, 1, 0, 1] + [0, 0],
        "y": [0] * (n_clean + 2),
        "width": [1, 2, 1, 2, 1, 2] + [127, 0],  # outliers: int8 overflow, zero-width
        "height": [5, 3, 5, 3, 5, 3] + [5, 5],  # 3 full-height, 3 crouch, among the clean rows
    }
    pq.write_table(pa.table(obstacles, schema=OBSTACLES_SCHEMA), processed / "obstacles_0000.parquet")

    return processed


class TestFindGaps:
    def test_no_notes_returns_full_range(self):
        gaps = _find_gaps([], end_beat=10.0)
        assert gaps == [(0.0, 10.0)]

    def test_single_gap_after_last_note(self):
        gaps = _find_gaps([0.0], end_beat=10.0)
        assert (0.0, 10.0) in gaps

    def test_finds_gap_between_notes(self):
        gaps = _find_gaps([0.0, 1.0, 10.0], end_beat=10.0)
        # Largest gap (1.0 -> 10.0) should be first
        assert gaps[0] == (1.0, 10.0)

    def test_small_gaps_excluded(self):
        """Gaps smaller than MIN_GAP_BEATS are dropped."""
        gaps = _find_gaps([0.0, 0.1, 0.2, 0.3], end_beat=0.3)
        assert gaps == []


class TestGenerateObstacles:
    def test_empty_notes_returns_empty(self):
        assert generate_obstacles([], [], bpm=120.0, difficulty="Expert") == []

    def test_dense_notes_no_room_for_walls(self):
        """Notes packed every beat leave no gap during the dense passage itself
        (a wall may still land in the tail buffer after the very last note)."""
        notes = [_note(float(i)) for i in range(40)]
        obstacles = generate_obstacles(notes, [], bpm=120.0, difficulty="Expert")
        last_note_beat = notes[-1].beat
        assert all(obs.beat >= last_note_beat for obs in obstacles)

    def test_places_wall_in_large_gap(self):
        """A long silent gap in an otherwise short song gets at least one wall."""
        notes = [_note(0.0), _note(60.0)]  # 60-beat gap
        obstacles = generate_obstacles(notes, [], bpm=120.0, difficulty="Expert")
        assert len(obstacles) >= 1
        for obs in obstacles:
            assert obs.beat > 0.0
            assert obs.duration_beats > 0
            assert 0 <= obs.x < 4
            assert obs.height in (CROUCH_HEIGHT, FULL_HEIGHT)

    def test_walls_avoid_note_lanes(self):
        """A wall placed near notes shouldn't share their lane."""
        notes = [_note(0.0, x=0), _note(60.0, x=0)]
        obstacles = generate_obstacles(notes, [], bpm=120.0, difficulty="Expert")
        for obs in obstacles:
            occupied_lanes = set(range(obs.x, obs.x + obs.width))
            assert 0 not in occupied_lanes

    def test_respects_target_count_from_density(self):
        """Obstacle count is bounded by available gaps, not an unbounded density target."""
        notes = [_note(0.0), _note(8.0)]  # 8-beat gap, plus a short tail gap after beat 8
        stats = {"Expert": {"per_minute": 100.0, "duration_beats": 0.5, "width": 1, "crouch_fraction": 0.0}}
        obstacles = generate_obstacles(notes, [], bpm=120.0, difficulty="Expert", stats=stats)
        # Only two usable gaps exist (the inter-note gap and the tail gap), even
        # though the density target from per_minute=100 would ask for far more.
        assert len(obstacles) <= 2

    def test_unknown_difficulty_falls_back_to_expert_defaults(self):
        notes = [_note(0.0), _note(60.0)]
        obstacles = generate_obstacles(notes, [], bpm=120.0, difficulty="NotARealDifficulty")
        # Should not raise, and should behave like Expert defaults
        assert isinstance(obstacles, list)

    def test_sorted_by_beat(self):
        notes = [_note(0.0), _note(30.0), _note(60.0), _note(90.0)]
        obstacles = generate_obstacles(notes, [], bpm=120.0, difficulty="ExpertPlus")
        beats = [o.beat for o in obstacles]
        assert beats == sorted(beats)


class TestObstacleStatsIO:
    def test_save_and_load_round_trip(self, tmp_path):
        path = tmp_path / "stats.json"
        save_obstacle_stats(DEFAULT_OBSTACLE_STATS, path)
        loaded = load_obstacle_stats(path)
        assert loaded == DEFAULT_OBSTACLE_STATS

    def test_load_missing_file_returns_none(self, tmp_path):
        assert load_obstacle_stats(tmp_path / "does_not_exist.json") is None


class TestMineObstacleStats:
    def test_returns_json_serializable_values(self, tmp_path):
        """Regression: pandas/pyarrow float32 values must be cast to native
        Python types, or json.dumps (via save_obstacle_stats) raises TypeError."""
        processed = _write_processed_dir(tmp_path)
        stats = mine_obstacle_stats(processed)
        json.dumps(stats)  # must not raise

        expert = stats["Expert"]
        assert isinstance(expert["per_minute"], float)
        assert isinstance(expert["duration_beats"], float)
        assert isinstance(expert["width"], int)
        assert isinstance(expert["crouch_fraction"], float)

    def test_computes_expected_crouch_fraction(self, tmp_path):
        processed = _write_processed_dir(tmp_path)
        stats = mine_obstacle_stats(processed)
        # One full-height (5) and one crouch (3) obstacle -> 50% crouch
        assert stats["Expert"]["crouch_fraction"] == pytest.approx(0.5)

    def test_missing_difficulties_fall_back_to_defaults(self, tmp_path):
        processed = _write_processed_dir(tmp_path)
        stats = mine_obstacle_stats(processed)
        assert stats["Easy"] == DEFAULT_OBSTACLE_STATS["Easy"]

    def test_excludes_modded_outliers(self, tmp_path):
        """The int8-overflow (width=127) and zero-width outlier rows must not
        pollute the mined width/duration stats."""
        processed = _write_processed_dir(tmp_path)
        stats = mine_obstacle_stats(processed)
        assert stats["Expert"]["width"] in (1, 2)  # not 127 or 0
        assert 0 < stats["Expert"]["duration_beats"] <= 4.0  # not -5.0 or 3000.0

    def test_too_few_clean_samples_falls_back_to_defaults(self, tmp_path):
        """A difficulty whose obstacles are all outliers should not report junk stats."""
        from beat_weaver.storage.writer import NOTES_SCHEMA, OBSTACLES_SCHEMA

        processed = tmp_path / "processed"
        processed.mkdir()
        notes = {
            "song_hash": ["h1"], "source": ["beatsaver"], "difficulty": ["Hard"],
            "characteristic": ["Standard"], "bpm": [120.0], "beat": [0.0],
            "time_seconds": [0.0], "x": [0], "y": [0], "color": [0],
            "cut_direction": [1], "angle_offset": [0],
        }
        pq.write_table(pa.table(notes, schema=NOTES_SCHEMA), processed / "notes_0000.parquet")
        # Only 2 obstacle rows, both outliers -> 0 clean rows survive filtering
        obstacles = {
            "song_hash": ["h1", "h1"], "source": ["beatsaver", "beatsaver"],
            "difficulty": ["Hard", "Hard"], "characteristic": ["Standard", "Standard"],
            "bpm": [120.0, 120.0], "beat": [1.0, 2.0], "time_seconds": [0.5, 1.0],
            "duration_beats": [-5.0, 3000.0], "x": [0, 0], "y": [0, 0],
            "width": [127, 0], "height": [5, 5],
        }
        pq.write_table(pa.table(obstacles, schema=OBSTACLES_SCHEMA), processed / "obstacles_0000.parquet")

        stats = mine_obstacle_stats(processed)
        assert stats["Hard"] == DEFAULT_OBSTACLE_STATS["Hard"]

    def test_no_obstacles_parquet_returns_defaults(self, tmp_path):
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

        stats = mine_obstacle_stats(processed)
        assert stats == DEFAULT_OBSTACLE_STATS
