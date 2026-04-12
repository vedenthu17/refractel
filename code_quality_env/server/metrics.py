from __future__ import annotations

from datetime import datetime, timezone
from statistics import mean

from ..graders import GradeResult, grade_task
from ..models import ReviewFinding, TaskSpec
from ..tasks import build_tasks


def _oracle_predictions(task: TaskSpec) -> list[ReviewFinding]:
    predictions: list[ReviewFinding] = []
    for gt in task.ground_truth:
        keyword_str = " ".join(gt.must_include_keywords) if gt.must_include_keywords else "quality review"
        predictions.append(
            ReviewFinding(
                file_path=gt.file_path,
                line=gt.line,
                finding_type=gt.finding_type,
                severity=gt.severity,
                issue=f"Detected {gt.finding_type.value} concern: {keyword_str}",
                suggestion=f"Suggested fix should address: {keyword_str}",
            )
        )
    return predictions


def _sparse_predictions(task: TaskSpec) -> list[ReviewFinding]:
    predictions: list[ReviewFinding] = []
    for idx, gt in enumerate(task.ground_truth):
        if idx % 2 != 0:
            continue
        keyword_str = " ".join(gt.must_include_keywords[:1]) if gt.must_include_keywords else "maintainability"
        predictions.append(
            ReviewFinding(
                file_path=gt.file_path,
                line=gt.line,
                finding_type=gt.finding_type,
                severity=gt.severity,
                issue=f"Potential {gt.finding_type.value} issue around {keyword_str}",
                suggestion=f"Improve this area with clearer {keyword_str} handling",
            )
        )
    return predictions


def _task_metrics(task: TaskSpec) -> dict:
    oracle_grade: GradeResult = grade_task(task, _oracle_predictions(task))
    sparse_grade: GradeResult = grade_task(task, _sparse_predictions(task))
    return {
        "task_name": task.name,
        "difficulty": task.difficulty,
        "files": len(task.files),
        "ground_truth_findings": len(task.ground_truth),
        "oracle": {
            "score": oracle_grade.score,
            "precision": oracle_grade.precision,
            "recall": oracle_grade.recall,
            "exact_matches": oracle_grade.exact_matches,
            "partial_matches": oracle_grade.partial_matches,
            "false_positives": oracle_grade.false_positives,
            "mean_match_score": oracle_grade.mean_match_score,
        },
        "sparse_baseline": {
            "score": sparse_grade.score,
            "precision": sparse_grade.precision,
            "recall": sparse_grade.recall,
            "exact_matches": sparse_grade.exact_matches,
            "partial_matches": sparse_grade.partial_matches,
            "false_positives": sparse_grade.false_positives,
            "mean_match_score": sparse_grade.mean_match_score,
        },
    }


def build_leaderboard_snapshot() -> dict:
    tasks = build_tasks()
    per_task = [_task_metrics(task) for task in tasks]
    oracle_scores = [float(t["oracle"]["score"]) for t in per_task]
    sparse_scores = [float(t["sparse_baseline"]["score"]) for t in per_task]

    return {
        "benchmark": "code-review-quality-env",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "algorithm": {
            "grader": "semantic_weighted_bipartite_matching_v2",
            "score_bounds": "(0,1)",
        },
        "summary": {
            "task_count": len(per_task),
            "oracle_avg_score": mean(oracle_scores) if oracle_scores else 0.0,
            "sparse_avg_score": mean(sparse_scores) if sparse_scores else 0.0,
            "hard_tasks": sum(1 for t in per_task if t["difficulty"] == "hard"),
        },
        "tasks": per_task,
    }
