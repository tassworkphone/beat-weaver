"""Post-generation quality metrics for Beat Saber maps."""

from __future__ import annotations

from collections import Counter
from typing import Sequence

from beat_weaver.model.tokenizer import (
    BOMB_BASE,
    BOMB_COUNT,
    BOMB_EMPTY,
    LEFT_BASE,
    LEFT_COUNT,
    LEFT_EMPTY,
    PAD,
    RIGHT_BASE,
    RIGHT_COUNT,
    RIGHT_EMPTY,
)
from beat_weaver.schemas.normalized import Note

# Token-accuracy buckets for the cubes-only diagnostic. "cube" is a real
# LEFT/RIGHT placement (x, y, dir); "cube_empty" is an empty hand slot.
# Structure covers BAR/POS/DIFF/START/END. Bombs are split the same way so
# a 15-point overall drop can be attributed to cubes vs décor vs scaffolding.
TOKEN_ACCURACY_CLASSES = (
    "cube",
    "cube_empty",
    "bomb",
    "bomb_empty",
    "structure",
)

# Cut direction vectors for parity checking
# 0=Up, 1=Down, 2=Left, 3=Right, 4=UpLeft, 5=UpRight, 6=DownLeft, 7=DownRight, 8=Any
_DIR_VECTORS = {
    0: (0, 1),    # Up
    1: (0, -1),   # Down
    2: (-1, 0),   # Left
    3: (1, 0),    # Right
    4: (-1, 1),   # UpLeft
    5: (1, 1),    # UpRight
    6: (-1, -1),  # DownLeft
    7: (1, -1),   # DownRight
    8: (0, 0),    # Any (no specific direction)
}


def _onset_f1(
    generated: list[Note],
    reference: list[Note],
    tolerance_seconds: float = 0.04,
) -> float:
    """Match generated note times to reference within tolerance.

    Returns F1 score (harmonic mean of precision and recall).
    """
    if not generated and not reference:
        return 1.0
    if not generated or not reference:
        return 0.0

    gen_times = sorted(n.time_seconds for n in generated)
    ref_times = sorted(n.time_seconds for n in reference)

    # Greedy matching
    matched_gen = set()
    matched_ref = set()
    ref_idx = 0

    for gi, gt in enumerate(gen_times):
        # Find closest unmatched ref
        best_ri = None
        best_dist = float("inf")
        for ri in range(max(0, ref_idx - 5), min(len(ref_times), ref_idx + 20)):
            if ri in matched_ref:
                continue
            dist = abs(gt - ref_times[ri])
            if dist <= tolerance_seconds and dist < best_dist:
                best_dist = dist
                best_ri = ri
        if best_ri is not None:
            matched_gen.add(gi)
            matched_ref.add(best_ri)
            ref_idx = best_ri

    tp = len(matched_gen)
    precision = tp / len(gen_times) if gen_times else 0.0
    recall = tp / len(ref_times) if ref_times else 0.0

    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


def _nps_accuracy(generated: list[Note], reference: list[Note]) -> float:
    """Compare notes-per-second between generated and reference.

    Returns 1 - |nps_gen - nps_ref| / nps_ref, clamped to [0, 1].
    """
    def _nps(notes: list[Note]) -> float:
        if len(notes) < 2:
            return 0.0
        times = [n.time_seconds for n in notes]
        duration = max(times) - min(times)
        return len(notes) / duration if duration > 0 else 0.0

    nps_gen = _nps(generated)
    nps_ref = _nps(reference)
    if nps_ref == 0:
        return 1.0 if nps_gen == 0 else 0.0
    return max(0.0, 1.0 - abs(nps_gen - nps_ref) / nps_ref)


def _beat_alignment(notes: list[Note], subdivisions_per_beat: int = 16) -> float:
    """Mean distance from each note beat to nearest 1/16th grid position.

    Lower is better. Returns 0.0 for perfectly aligned notes.
    """
    if not notes:
        return 0.0
    grid = 1.0 / subdivisions_per_beat
    distances = []
    for note in notes:
        nearest = round(note.beat / grid) * grid
        distances.append(abs(note.beat - nearest))
    return sum(distances) / len(distances)


def _parity_violations(notes: list[Note]) -> float:
    """Count parity violations per hand.

    Tracks swing state (forehand/backhand) and counts direction reversals
    that violate natural swing patterns.

    Returns fraction of notes that are parity violations.
    """
    if not notes:
        return 0.0

    # Separate by hand
    left_notes = sorted([n for n in notes if n.color == 0], key=lambda n: n.beat)
    right_notes = sorted([n for n in notes if n.color == 1], key=lambda n: n.beat)

    violations = 0
    total = 0

    for hand_notes in [left_notes, right_notes]:
        last_dir = None
        for note in hand_notes:
            if note.cut_direction == 8:  # Any — never a violation
                last_dir = None
                continue
            total += 1
            if last_dir is not None:
                dy_last = _DIR_VECTORS[last_dir][1]
                dy_curr = _DIR_VECTORS[note.cut_direction][1]
                # Parity violation: two consecutive same-vertical-direction swings
                # (e.g., down-down or up-up without alternating)
                if dy_last != 0 and dy_curr != 0 and dy_last == dy_curr:
                    violations += 1
            last_dir = note.cut_direction

    return violations / max(1, total)


def _pattern_diversity(notes: list[Note], window: int = 4) -> float:
    """Fraction of unique n-note subsequences.

    Higher is more diverse. Returns ratio of unique to total windows.
    """
    if len(notes) < window:
        return 1.0

    sorted_notes = sorted(notes, key=lambda n: n.beat)
    patterns: list[tuple] = []
    for i in range(len(sorted_notes) - window + 1):
        pattern = tuple(
            (n.x, n.y, n.color, n.cut_direction)
            for n in sorted_notes[i: i + window]
        )
        patterns.append(pattern)

    if not patterns:
        return 1.0
    return len(set(patterns)) / len(patterns)


def _notes_per_second(notes: list[Note]) -> float:
    """Compute notes per second."""
    if len(notes) < 2:
        return 0.0
    times = [n.time_seconds for n in notes]
    duration = max(times) - min(times)
    return len(notes) / duration if duration > 0 else 0.0


def evaluate_map(
    generated_notes: list[Note],
    reference_notes: list[Note],
    bpm: float,
) -> dict[str, float]:
    """Evaluate generated map against a reference.

    Returns dict with metrics:
        onset_f1, nps_accuracy, beat_alignment, parity_violation_rate,
        pattern_diversity, nps
    """
    return {
        "onset_f1": _onset_f1(generated_notes, reference_notes),
        "nps_accuracy": _nps_accuracy(generated_notes, reference_notes),
        "beat_alignment": _beat_alignment(generated_notes),
        "parity_violation_rate": _parity_violations(generated_notes),
        "pattern_diversity": _pattern_diversity(generated_notes),
        "nps": _notes_per_second(generated_notes),
    }


def token_class_name(token_id: int) -> str:
    """Bucket a token ID for class-wise accuracy.

    PAD is returned as "pad" so callers can skip it. Everything that is not
    a cube, cube-empty, bomb, or bomb-empty slot is "structure".
    """
    if token_id == PAD:
        return "pad"
    if (
        LEFT_BASE <= token_id < LEFT_BASE + LEFT_COUNT
        or RIGHT_BASE <= token_id < RIGHT_BASE + RIGHT_COUNT
    ):
        return "cube"
    if token_id in (LEFT_EMPTY, RIGHT_EMPTY):
        return "cube_empty"
    if BOMB_BASE <= token_id < BOMB_BASE + BOMB_COUNT:
        return "bomb"
    if token_id == BOMB_EMPTY:
        return "bomb_empty"
    return "structure"


def _class_stats(correct: int, total: int) -> dict[str, float | int | None]:
    return {
        "accuracy": (correct / total) if total else None,
        "correct": correct,
        "total": total,
    }


def token_accuracy_by_class(
    preds: Sequence[int],
    targets: Sequence[int],
) -> dict[str, dict[str, float | int | None]]:
    """Teacher-forced token accuracy split by class.

    Ignores PAD targets. ``overall`` matches training's val_token_accuracy
    (every non-PAD token). ``cube`` is the diagnostic that answers whether
    color-note placement (x, y, direction) collapsed independently of bombs.
    """
    if len(preds) != len(targets):
        raise ValueError(
            f"preds and targets must be the same length, got {len(preds)} vs {len(targets)}"
        )

    buckets = {name: {"correct": 0, "total": 0} for name in TOKEN_ACCURACY_CLASSES}
    overall_correct = 0
    overall_total = 0

    for pred, target in zip(preds, targets):
        name = token_class_name(int(target))
        if name == "pad":
            continue
        overall_total += 1
        hit = int(pred) == int(target)
        overall_correct += int(hit)
        bucket = buckets[name]
        bucket["total"] += 1
        bucket["correct"] += int(hit)

    result = {"overall": _class_stats(overall_correct, overall_total)}
    for name in TOKEN_ACCURACY_CLASSES:
        result[name] = _class_stats(buckets[name]["correct"], buckets[name]["total"])
    return result


def _merge_class_stats(
    acc: dict[str, dict[str, int]], extra: dict[str, dict[str, float | int | None]],
) -> None:
    for name, stats in extra.items():
        bucket = acc.setdefault(name, {"correct": 0, "total": 0})
        bucket["correct"] += int(stats["correct"])
        bucket["total"] += int(stats["total"])


def teacher_forced_class_accuracy(model, dataloader, device) -> dict[str, dict[str, float | int | None]]:
    """Run a teacher-forced pass and return token_accuracy_by_class totals.

    Same setup as training validation (input = tokens[:-1], target = tokens[1:],
    ignore PAD), but split by cube / bomb / structure so a drop in overall
    val_acc can be attributed.
    """
    import torch

    from beat_weaver.model.tokenizer import PAD as _PAD

    model.eval()
    merged: dict[str, dict[str, int]] = {}

    with torch.no_grad():
        for mel, mel_mask, tokens, token_mask in dataloader:
            mel = mel.to(device)
            mel_mask = mel_mask.to(device)
            tokens = tokens.to(device)
            token_mask = token_mask.to(device)

            input_tokens = tokens[:, :-1]
            target_tokens = tokens[:, 1:]
            input_mask = token_mask[:, :-1]
            target_mask = token_mask[:, 1:]

            logits = model(mel, input_tokens, mel_mask, input_mask)
            preds = logits.argmax(dim=-1)
            valid = target_mask & (target_tokens != _PAD)

            pred_ids = preds[valid].detach().cpu().tolist()
            target_ids = target_tokens[valid].detach().cpu().tolist()
            _merge_class_stats(merged, token_accuracy_by_class(pred_ids, target_ids))

    result = {}
    for name, bucket in merged.items():
        result[name] = _class_stats(bucket["correct"], bucket["total"])
    # Guarantee a stable key order even if a class never appeared.
    ordered = {}
    for name in ("overall",) + TOKEN_ACCURACY_CLASSES:
        ordered[name] = result.get(name, _class_stats(0, 0))
    return ordered


def mean_map_metrics(results: list[dict]) -> dict[str, float]:
    """Average the numeric evaluate_map fields across a list of per-map dicts."""
    if not results:
        return {}
    keys = (
        "onset_f1",
        "nps_accuracy",
        "beat_alignment",
        "parity_violation_rate",
        "pattern_diversity",
        "nps",
    )
    means: dict[str, float] = {}
    for key in keys:
        values = [float(r[key]) for r in results if key in r]
        if values:
            means[key] = sum(values) / len(values)
    return means


def evaluate_standalone(notes: list[Note], bpm: float) -> dict[str, float]:
    """Evaluate a map without a reference (standalone quality metrics).

    Returns dict with metrics:
        beat_alignment, parity_violation_rate, pattern_diversity, nps
    """
    return {
        "beat_alignment": _beat_alignment(notes),
        "parity_violation_rate": _parity_violations(notes),
        "pattern_diversity": _pattern_diversity(notes),
        "nps": _notes_per_second(notes),
    }


# Soft NPS targets for ranking candidates (customs-like density).
# Floors are the empty-map ship gate, not "must match Expert+".
_NPS_TARGET = {
    "Easy": 1.2,
    "Normal": 2.0,
    "Hard": 3.5,
    "Expert": 5.5,
    "ExpertPlus": 8.0,
}
_NPS_FLOOR = {
    "Easy": 0.4,
    "Normal": 0.6,
    "Hard": 0.8,
    "Expert": 1.0,
    "ExpertPlus": 1.2,
}
_MAX_PARITY_VIOLATION = 0.40
_MIN_NOTES = 8


def passes_playability_gate(
    notes: list[Note],
    difficulty: str = "Expert",
    metrics: dict[str, float] | None = None,
) -> bool:
    """Hard ship gate: not empty, not a wrist-break, enough cubes.

    Onset F1 needs a human reference, so generate-time gating uses NPS
    (empty-map detector) and parity (illegal follow-through) instead.
    """
    if len(notes) < _MIN_NOTES:
        return False
    if metrics is None:
        metrics = evaluate_standalone(notes, bpm=120.0)
    floor = _NPS_FLOOR.get(difficulty, _NPS_FLOOR["Expert"])
    if metrics["nps"] < floor:
        return False
    if metrics["parity_violation_rate"] > _MAX_PARITY_VIOLATION:
        return False
    return True


def playability_score(
    notes: list[Note],
    difficulty: str = "Expert",
    metrics: dict[str, float] | None = None,
) -> float:
    """Higher is better. Used to pick among generate() seeds.

    Combines NPS closeness to a difficulty target, low parity violations,
    and pattern diversity. Empty maps score below every real candidate.
    """
    if len(notes) < 2:
        return -1.0
    if metrics is None:
        metrics = evaluate_standalone(notes, bpm=120.0)
    target = _NPS_TARGET.get(difficulty, _NPS_TARGET["Expert"])
    nps = metrics["nps"]
    nps_term = max(0.0, 1.0 - abs(nps - target) / target)
    parity_term = max(0.0, 1.0 - metrics["parity_violation_rate"])
    diversity_term = metrics["pattern_diversity"]
    return nps_term + parity_term + 0.5 * diversity_term


def select_playable_candidate(
    candidates: list[tuple[list[Note], dict[str, float]]],
    difficulty: str,
    *,
    require_gate: bool = True,
) -> tuple[int, bool, float]:
    """Pick the best candidate. Prefer ones that pass the ship gate.

    ``candidates`` is a list of (notes, metrics) in generation order.
    Returns (index, passed_gate, score) of the winner.
    """
    if not candidates:
        raise ValueError("select_playable_candidate requires at least one candidate")

    scored: list[tuple[int, bool, float]] = []
    for i, (notes, metrics) in enumerate(candidates):
        passed = passes_playability_gate(notes, difficulty, metrics)
        score = playability_score(notes, difficulty, metrics)
        scored.append((i, passed, score))

    passing = [row for row in scored if row[1]]
    pool = passing if (require_gate and passing) else scored
    # Highest score; stable on ties (lowest index).
    winner = max(pool, key=lambda row: (row[2], -row[0]))
    return winner
