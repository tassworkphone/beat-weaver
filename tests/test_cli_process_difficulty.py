"""Tests for process --difficulty filtering.

Regression coverage for physically excluding non-wanted difficulties from
the Parquet output, rather than relying only on a training-time filter
(ModelConfig.min_difficulty/max_difficulty) — the user wanted maps they
never intend to train on excluded from data/processed entirely.
"""
from beat_weaver.cli import _filter_by_difficulty
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
