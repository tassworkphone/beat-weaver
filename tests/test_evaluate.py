"""Tests for evaluation metrics."""

import pytest

from beat_weaver.model.evaluate import (
    _beat_alignment,
    _notes_per_second,
    _nps_accuracy,
    _onset_f1,
    _parity_violations,
    _pattern_diversity,
    evaluate_map,
    evaluate_standalone,
    mean_map_metrics,
    passes_playability_gate,
    playability_score,
    select_playable_candidate,
    token_accuracy_by_class,
    token_class_name,
)
from beat_weaver.model.tokenizer import (
    BAR,
    BOMB_BASE,
    BOMB_EMPTY,
    LEFT_BASE,
    LEFT_EMPTY,
    PAD,
    POS_BASE,
    RIGHT_BASE,
    RIGHT_EMPTY,
)
from beat_weaver.schemas.normalized import Note


def _note(beat: float, color: int = 0, direction: int = 1, x: int = 0, y: int = 0, bpm: float = 120.0) -> Note:
    return Note(
        beat=beat, time_seconds=beat * 60.0 / bpm,
        x=x, y=y, color=color, cut_direction=direction,
    )


class TestOnsetF1:
    def test_perfect_match(self):
        notes = [_note(0.0), _note(1.0), _note(2.0)]
        f1 = _onset_f1(notes, notes)
        assert f1 == pytest.approx(1.0)

    def test_no_overlap(self):
        gen = [_note(0.0), _note(1.0)]
        ref = [_note(10.0), _note(11.0)]
        f1 = _onset_f1(gen, ref)
        assert f1 == 0.0

    def test_both_empty(self):
        assert _onset_f1([], []) == 1.0

    def test_one_empty(self):
        assert _onset_f1([_note(0.0)], []) == 0.0
        assert _onset_f1([], [_note(0.0)]) == 0.0

    def test_close_match(self):
        gen = [_note(0.0), _note(1.0)]
        ref = [_note(0.01), _note(1.02)]  # within 40ms tolerance
        f1 = _onset_f1(gen, ref)
        assert f1 == pytest.approx(1.0)


class TestNPSAccuracy:
    def test_identical(self):
        notes = [_note(0.0), _note(1.0), _note(2.0)]
        acc = _nps_accuracy(notes, notes)
        assert acc == pytest.approx(1.0)

    def test_double_density(self):
        gen = [_note(i * 0.5) for i in range(6)]
        ref = [_note(i * 1.0) for i in range(3)]
        acc = _nps_accuracy(gen, ref)
        assert acc < 1.0
        assert acc >= 0.0


class TestBeatAlignment:
    def test_perfect_alignment(self):
        """Notes on exact 1/16th grid."""
        notes = [_note(0.0), _note(0.25), _note(0.5), _note(1.0)]
        assert _beat_alignment(notes) == pytest.approx(0.0)

    def test_off_grid(self):
        notes = [_note(0.1)]  # not on 1/16th grid (0.0625 increments)
        alignment = _beat_alignment(notes)
        assert alignment > 0.0

    def test_empty(self):
        assert _beat_alignment([]) == 0.0


class TestParityViolations:
    def test_no_violations(self):
        """Alternating up/down is correct parity."""
        notes = [
            _note(0.0, direction=1),  # Down
            _note(1.0, direction=0),  # Up
            _note(2.0, direction=1),  # Down
        ]
        rate = _parity_violations(notes)
        assert rate == 0.0

    def test_all_same_direction(self):
        """All down = violations after the first."""
        notes = [
            _note(0.0, direction=1),  # Down
            _note(1.0, direction=1),  # Down (violation)
            _note(2.0, direction=1),  # Down (violation)
        ]
        rate = _parity_violations(notes)
        assert rate == pytest.approx(2 / 3)

    def test_any_direction_resets(self):
        """Direction 8 (Any) never causes violations."""
        notes = [
            _note(0.0, direction=1),  # Down
            _note(1.0, direction=8),  # Any — resets
            _note(2.0, direction=1),  # Down — no violation (after Any)
        ]
        rate = _parity_violations(notes)
        assert rate == 0.0

    def test_empty(self):
        assert _parity_violations([]) == 0.0


class TestPatternDiversity:
    def test_all_unique(self):
        notes = [
            _note(i, x=i % 4, y=i % 3, direction=i % 9)
            for i in range(10)
        ]
        diversity = _pattern_diversity(notes)
        assert diversity == 1.0

    def test_all_same(self):
        notes = [_note(float(i), x=0, y=0, direction=1) for i in range(10)]
        diversity = _pattern_diversity(notes)
        # Only 1 unique pattern out of 7 windows
        assert diversity < 0.5

    def test_short_sequence(self):
        notes = [_note(0.0)]
        assert _pattern_diversity(notes) == 1.0


class TestEvaluateMap:
    def test_returns_all_metrics(self):
        gen = [_note(0.0), _note(1.0)]
        ref = [_note(0.0), _note(1.0)]
        result = evaluate_map(gen, ref, bpm=120.0)
        assert "onset_f1" in result
        assert "nps_accuracy" in result
        assert "beat_alignment" in result
        assert "parity_violation_rate" in result
        assert "pattern_diversity" in result
        assert "nps" in result


class TestEvaluateStandalone:
    def test_returns_metrics(self):
        notes = [_note(0.0), _note(1.0), _note(2.0)]
        result = evaluate_standalone(notes, bpm=120.0)
        assert "beat_alignment" in result
        assert "parity_violation_rate" in result
        assert "pattern_diversity" in result
        assert "nps" in result


class TestTokenClassName:
    def test_cube_and_empty(self):
        assert token_class_name(LEFT_BASE) == "cube"
        assert token_class_name(RIGHT_BASE) == "cube"
        assert token_class_name(LEFT_EMPTY) == "cube_empty"
        assert token_class_name(RIGHT_EMPTY) == "cube_empty"

    def test_bombs_and_structure(self):
        assert token_class_name(BOMB_BASE) == "bomb"
        assert token_class_name(BOMB_EMPTY) == "bomb_empty"
        assert token_class_name(BAR) == "structure"
        assert token_class_name(POS_BASE) == "structure"
        assert token_class_name(PAD) == "pad"


class TestTokenAccuracyByClass:
    def test_perfect_overall_and_cube(self):
        targets = [BAR, LEFT_BASE, RIGHT_EMPTY, BOMB_EMPTY]
        preds = list(targets)
        result = token_accuracy_by_class(preds, targets)
        assert result["overall"]["accuracy"] == pytest.approx(1.0)
        assert result["cube"]["accuracy"] == pytest.approx(1.0)
        assert result["cube"]["total"] == 1
        assert result["cube_empty"]["total"] == 1
        assert result["bomb_empty"]["total"] == 1
        assert result["structure"]["total"] == 1
        assert result["bomb"]["total"] == 0
        assert result["bomb"]["accuracy"] is None

    def test_ignores_pad(self):
        targets = [PAD, LEFT_BASE, PAD]
        preds = [BAR, LEFT_BASE, BAR]
        result = token_accuracy_by_class(preds, targets)
        assert result["overall"]["total"] == 1
        assert result["overall"]["accuracy"] == pytest.approx(1.0)

    def test_cube_drop_does_not_hide_in_structure(self):
        """A wrong cube placement with correct BAR/POS should show up in cube acc."""
        targets = [BAR, POS_BASE, LEFT_BASE, RIGHT_EMPTY]
        preds = [BAR, POS_BASE, LEFT_BASE + 1, RIGHT_EMPTY]
        result = token_accuracy_by_class(preds, targets)
        assert result["overall"]["accuracy"] == pytest.approx(0.75)
        assert result["cube"]["accuracy"] == pytest.approx(0.0)
        assert result["structure"]["accuracy"] == pytest.approx(1.0)
        assert result["cube_empty"]["accuracy"] == pytest.approx(1.0)

    def test_length_mismatch_raises(self):
        with pytest.raises(ValueError):
            token_accuracy_by_class([1], [1, 2])


class TestPlayabilityGate:
    def test_empty_fails(self):
        assert passes_playability_gate([], "Normal") is False

    def test_sparse_fails_nps_floor(self):
        notes = [_note(float(i) * 4.0) for i in range(10)]  # very sparse
        assert passes_playability_gate(notes, "Normal") is False

    def test_reasonable_normal_passes(self):
        notes = [_note(i * 0.5, direction=(i % 2)) for i in range(20)]
        assert passes_playability_gate(notes, "Normal") is True

    def test_high_parity_fails(self):
        notes = [_note(i * 0.5, direction=1) for i in range(20)]  # all down
        assert passes_playability_gate(notes, "Normal") is False


class TestPlayabilityScore:
    def test_empty_below_real_map(self):
        real = [_note(i * 0.5, direction=(i % 2)) for i in range(20)]
        assert playability_score([], "Normal") < playability_score(real, "Normal")

    def test_lower_parity_ranks_higher(self):
        good = [_note(i * 0.5, direction=(i % 2)) for i in range(20)]
        bad = [_note(i * 0.5, direction=1) for i in range(20)]
        assert playability_score(good, "Normal") > playability_score(bad, "Normal")


class TestSelectPlayableCandidate:
    def test_prefers_gate_pass_over_higher_failing_score(self):
        empty = []
        good = [_note(i * 0.5, direction=(i % 2)) for i in range(20)]
        empty_metrics = evaluate_standalone(empty, 120.0)
        good_metrics = evaluate_standalone(good, 120.0)
        idx, passed, _score = select_playable_candidate(
            [(empty, empty_metrics), (good, good_metrics)],
            "Normal",
            require_gate=True,
        )
        assert idx == 1
        assert passed is True

    def test_picks_best_score_when_gate_disabled(self):
        a = [_note(i * 0.5, direction=1) for i in range(20)]
        b = [_note(i * 0.5, direction=(i % 2)) for i in range(20)]
        idx, _passed, _score = select_playable_candidate(
            [(a, evaluate_standalone(a, 120.0)), (b, evaluate_standalone(b, 120.0))],
            "Normal",
            require_gate=False,
        )
        assert idx == 1


class TestMeanMapMetrics:
    def test_averages_numeric_fields(self):
        results = [
            {"onset_f1": 0.0, "nps_accuracy": 1.0, "parity_violation_rate": 0.2,
             "pattern_diversity": 1.0, "nps": 2.0, "beat_alignment": 0.0, "song_hash": "a"},
            {"onset_f1": 1.0, "nps_accuracy": 0.0, "parity_violation_rate": 0.4,
             "pattern_diversity": 1.0, "nps": 4.0, "beat_alignment": 0.0, "song_hash": "b"},
        ]
        means = mean_map_metrics(results)
        assert means["onset_f1"] == pytest.approx(0.5)
        assert means["nps_accuracy"] == pytest.approx(0.5)
        assert means["parity_violation_rate"] == pytest.approx(0.3)
        assert means["nps"] == pytest.approx(3.0)
        assert "song_hash" not in means

    def test_empty(self):
        assert mean_map_metrics([]) == {}
