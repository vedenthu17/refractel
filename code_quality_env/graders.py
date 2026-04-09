from __future__ import annotations

from dataclasses import dataclass

from .models import GroundTruthFinding, ReviewFinding, TaskSpec


@dataclass(frozen=True)
class GradeResult:
    score: float
    precision: float
    recall: float
    exact_matches: int
    partial_matches: int
    false_positives: int


_SCORE_EPSILON = 1e-6


def _clamp_open_unit_interval(value: float) -> float:
    """Clamp score to the open interval (0, 1)."""
    return max(_SCORE_EPSILON, min(1.0 - _SCORE_EPSILON, value))


def _keyword_overlap_fraction(gt: GroundTruthFinding, pred: ReviewFinding) -> float:
    if not gt.must_include_keywords:
        return 1.0
    text = f"{pred.issue} {pred.suggestion}".lower()
    hits = sum(1 for kw in gt.must_include_keywords if kw.lower() in text)
    return hits / len(gt.must_include_keywords)


def match_finding(
    gt: GroundTruthFinding,
    pred: ReviewFinding,
) -> tuple[bool, bool]:
    if gt.file_path != pred.file_path:
        return False, False
    if gt.finding_type != pred.finding_type:
        return False, False

    line_close = abs(gt.line - pred.line) <= 1
    keyword_fraction = _keyword_overlap_fraction(gt, pred)

    exact = line_close and keyword_fraction >= 0.75
    partial = line_close and keyword_fraction >= 0.25
    return exact, partial


def grade_task(task: TaskSpec, predictions: list[ReviewFinding]) -> GradeResult:
    unmatched = {f.finding_id: f for f in task.ground_truth}
    exact_matches = 0
    partial_matches = 0
    false_positives = 0

    for pred in predictions:
        best_exact_id = None
        best_partial_id = None

        for finding_id, gt in unmatched.items():
            exact, partial = match_finding(gt, pred)
            if exact:
                best_exact_id = finding_id
                break
            if partial and best_partial_id is None:
                best_partial_id = finding_id

        if best_exact_id is not None:
            exact_matches += 1
            unmatched.pop(best_exact_id, None)
            continue

        if best_partial_id is not None:
            partial_matches += 1
            unmatched.pop(best_partial_id, None)
            continue

        false_positives += 1

    weighted_tp = exact_matches + 0.5 * partial_matches
    precision = weighted_tp / max(len(predictions), 1)
    recall = weighted_tp / max(len(task.ground_truth), 1)
    if precision + recall == 0:
        f1 = 0.0
    else:
        f1 = 2 * precision * recall / (precision + recall)

    penalty = min(false_positives * 0.05, 0.4)
    score = _clamp_open_unit_interval(f1 - penalty)

    return GradeResult(
        score=score,
        precision=precision,
        recall=recall,
        exact_matches=exact_matches,
        partial_matches=partial_matches,
        false_positives=false_positives,
    )
