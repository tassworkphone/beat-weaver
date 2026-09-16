"""Tests for process --difficulty filtering.

Regression coverage for physically excluding non-wanted difficulties from
the Parquet output, rather than relying only on a training-time filter
(ModelConfig.min_difficulty/max_difficulty) — the user wanted maps they
never intend to train on excluded from data/processed entirely.
"""
from pathlib import Path

from beat_weaver.cli import _detect_source, _filter_by_difficulty
from beat_weaver.schemas.normalized import DifficultyInfo, NormalizedBeatmap, SongMetadata


def _beatmap(difficulty: str) -> NormalizedBeatmap:
    return NormalizedBeatmap(
        metadata=SongMetadata(source="beatsaver", source_id="x", hash="h"),
        difficulty_info=DifficultyInfo(
            characteristic="Standard", difficulty=difficulty, difficulty_rank=1,
            note_jump_speed=10.0, note_jump_offset=0.0,
        ),
    )


class TestFilterByDifficulty:
    def test_none_keeps_everything(self):
        beatmaps = [_beatmap(d) for d in ("Easy", "Normal", "Hard", "Expert", "ExpertPlus")]
        kept, excluded = _filter_by_difficulty(beatmaps, None)
        assert kept == beatmaps
        assert excluded == 0

    def test_filters_to_single_difficulty(self):
        beatmaps = [_beatmap(d) for d in ("Easy", "Normal", "Hard", "Expert", "ExpertPlus")]
        kept, excluded = _filter_by_difficulty(beatmaps, {"Normal"})
        assert [bm.difficulty_info.difficulty for bm in kept] == ["Normal"]
        assert excluded == 4

    def test_filters_to_multiple_difficulties(self):
        beatmaps = [_beatmap(d) for d in ("Easy", "Normal", "Hard", "Expert", "ExpertPlus")]
        kept, excluded = _filter_by_difficulty(beatmaps, {"Easy", "Normal"})
        assert {bm.difficulty_info.difficulty for bm in kept} == {"Easy", "Normal"}
        assert excluded == 3

    def test_empty_beatmap_list(self):
        kept, excluded = _filter_by_difficulty([], {"Normal"})
        assert kept == []
        assert excluded == 0

    def test_no_matching_difficulty_excludes_all(self):
        beatmaps = [_beatmap(d) for d in ("Easy", "Hard")]
        kept, excluded = _filter_by_difficulty(beatmaps, {"Normal"})
        assert kept == []
        assert excluded == 2


class TestDetectSource:
    """Regression coverage for combining multiple --input directories:
    source detection must work from the full path, not just components
    relative to a single root, so a differently-named folder holding
    copied official maps (e.g. "official_normal_only") still tags correctly.
    """

    def test_official_in_default_layout(self, tmp_path):
        folder = tmp_path / "data" / "raw" / "official" / "somesong"
        folder.mkdir(parents=True)
        assert _detect_source(folder, tmp_path / "data" / "raw") == "official"

    def test_beatsaver_in_default_layout(self, tmp_path):
        folder = tmp_path / "data" / "raw" / "beatsaver" / "somehash"
        folder.mkdir(parents=True)
        assert _detect_source(folder, tmp_path / "data" / "raw") == "beatsaver"

    def test_renamed_official_folder_still_detected(self, tmp_path):
        """A trimmed copy under a differently-named folder (as long as
        "official" still appears in the path) must still tag as official."""
        folder = tmp_path / "data" / "raw" / "official_normal_only" / "somesong"
        folder.mkdir(parents=True)
        assert _detect_source(folder, folder.parent) == "official"

    def test_custom_named_beatsaver_folder_detected(self, tmp_path):
        folder = tmp_path / "data" / "raw" / "beatsaver_normal90" / "somehash"
        folder.mkdir(parents=True)
        assert _detect_source(folder, folder.parent) == "beatsaver"

    def test_unrelated_folder_is_local_custom(self, tmp_path):
        folder = tmp_path / "data" / "raw" / "my_own_maps" / "somesong"
        folder.mkdir(parents=True)
        assert _detect_source(folder, folder.parent) == "local_custom"
