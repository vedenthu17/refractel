from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
import math
import re

from .models import GroundTruthFinding, ReviewFinding, TaskSpec


@dataclass(frozen=True)
class GradeResult:
    score: float
    precision: float
    recall: float
    exact_matches: int
    partial_matches: int
    false_positives: int
    mean_match_score: float


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


def _tokenize(text: str) -> set[str]:
    return {tok for tok in re.findall(r"[a-zA-Z_][a-zA-Z0-9_]{1,}", text.lower())}


def _semantic_overlap(gt: GroundTruthFinding, pred: ReviewFinding) -> float:
    # Build synthetic GT text from normalized metadata and mandatory keywords.
    gt_text = " ".join(
        [
            gt.file_path,
            gt.finding_type.value,
            gt.severity.value,
            *gt.must_include_keywords,
        ]
    )
    pred_text = f"{pred.file_path} {pred.finding_type.value} {pred.severity.value} {pred.issue} {pred.suggestion}"
    gt_tokens = _tokenize(gt_text)
    pred_tokens = _tokenize(pred_text)
    if not gt_tokens or not pred_tokens:
        return 0.0
    overlap = gt_tokens & pred_tokens
    union = gt_tokens | pred_tokens
    return len(overlap) / max(len(union), 1)


def _severity_alignment(gt: GroundTruthFinding, pred: ReviewFinding) -> float:
    order = {"low": 0, "medium": 1, "high": 2}
    gt_rank = order[gt.severity.value]
    pred_rank = order[pred.severity.value]
    delta = abs(gt_rank - pred_rank)
    if delta == 0:
        return 1.0
    if delta == 1:
        return 0.6
    return 0.2


def _candidate_match_score(gt: GroundTruthFinding, pred: ReviewFinding) -> float:
    if gt.file_path != pred.file_path:
        return 0.0
    if gt.finding_type != pred.finding_type:
        return 0.0

    keyword_fraction = _keyword_overlap_fraction(gt, pred)
    semantic_fraction = _semantic_overlap(gt, pred)
    severity_fraction = _severity_alignment(gt, pred)
    line_distance = abs(gt.line - pred.line)
    line_fraction = math.exp(-line_distance / 2.5)

    score = (
        0.45 * keyword_fraction
        + 0.25 * line_fraction
        + 0.20 * semantic_fraction
        + 0.10 * severity_fraction
    )

    # Down-rank likely vague claims that don't align to expected key terms.
    if keyword_fraction < 0.25 and semantic_fraction < 0.15:
        score *= 0.6

    return max(0.0, min(1.0, score))


def _optimal_assignment(
    ground_truth: list[GroundTruthFinding],
    predictions: list[ReviewFinding],
) -> list[tuple[int, int, float]]:
    if not ground_truth or not predictions:
        return []

    n_gt = len(ground_truth)
    n_pred = len(predictions)
    weights: list[list[float]] = [
        [_candidate_match_score(ground_truth[g], predictions[p]) for g in range(n_gt)]
        for p in range(n_pred)
    ]

    @lru_cache(maxsize=None)
    def dp(pred_idx: int, used_mask: int) -> tuple[float, tuple[tuple[int, int], ...]]:
        if pred_idx >= n_pred:
            return 0.0, ()

        best_score, best_pairs = dp(pred_idx + 1, used_mask)

        for gt_idx in range(n_gt):
            bit = 1 << gt_idx
            if used_mask & bit:
                continue
            w = weights[pred_idx][gt_idx]
            if w <= 0.08:
                continue
            tail_score, tail_pairs = dp(pred_idx + 1, used_mask | bit)
            total = w + tail_score
            if total > best_score:
                best_score = total
                best_pairs = ((pred_idx, gt_idx),) + tail_pairs

        return best_score, best_pairs

    _, pairs = dp(0, 0)
    return [(pred_idx, gt_idx, weights[pred_idx][gt_idx]) for pred_idx, gt_idx in pairs]


def match_finding(
    gt: GroundTruthFinding,
    pred: ReviewFinding,
) -> tuple[bool, bool]:
    score = _candidate_match_score(gt, pred)
    exact = score >= 0.78
    partial = score >= 0.42
    return exact, partial


def grade_task(task: TaskSpec, predictions: list[ReviewFinding]) -> GradeResult:
    assignments = _optimal_assignment(task.ground_truth, predictions)
    exact_matches = sum(1 for _, _, s in assignments if s >= 0.78)
    partial_matches = sum(1 for _, _, s in assignments if 0.42 <= s < 0.78)
    weak_matches = sum(1 for _, _, s in assignments if 0.22 <= s < 0.42)
    semantic_matches = len(assignments)
    false_positives = max(len(predictions) - semantic_matches, 0)

    weighted_tp = exact_matches + 0.65 * partial_matches + 0.35 * weak_matches
    precision = weighted_tp / max(len(predictions), 1)
    recall = sum(s for _, _, s in assignments) / max(len(task.ground_truth), 1)
    if precision + recall == 0:
        f1 = 0.0
    else:
        f1 = 2 * precision * recall / (precision + recall)

    stability_bonus = 0.08 if precision >= 0.70 and recall >= 0.70 else 0.0
    penalty = min(false_positives * 0.04, 0.35)
    score = _clamp_open_unit_interval((0.78 * f1) + (0.17 * recall) + stability_bonus - penalty)
    mean_match_score = sum(s for _, _, s in assignments) / max(semantic_matches, 1)

    return GradeResult(
        score=score,
        precision=precision,
        recall=recall,
        exact_matches=exact_matches,
        partial_matches=partial_matches,
        false_positives=false_positives,
        mean_match_score=mean_match_score,
    )
