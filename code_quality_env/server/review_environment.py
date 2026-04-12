from __future__ import annotations

import uuid
from dataclasses import dataclass, field
import re
from typing import Dict, Optional

from ..graders import grade_task, match_finding
from ..models import (
    ActionType,
    CodeReviewAction,
    CodeReviewObservation,
    CodeReviewState,
    CodeReviewStepResult,
    ReviewFinding,
    TaskSpec,
)
from ..tasks import build_tasks


@dataclass
class RuntimeState:
    task: TaskSpec
    episode_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    step_count: int = 0
    done: bool = False
    done_reason: Optional[str] = None
    penalties: float = 0.0
    findings: list[ReviewFinding] = field(default_factory=list)
    matched_gt_ids: set[str] = field(default_factory=set)
    partial_gt_ids: set[str] = field(default_factory=set)
    covered_focus_types: set[str] = field(default_factory=set)
    inspected_files: set[str] = field(default_factory=set)
    open_file: Optional[str] = None
    last_action_result: str = "Environment reset."
    invalid_actions: int = 0
    noop_streak: int = 0
    risk_hotspot_details: Dict[str, float] = field(default_factory=dict)
    recommendation_hints: list[str] = field(default_factory=list)


class CodeReviewEnvironment:
    def __init__(self) -> None:
        tasks = build_tasks()
        self._tasks: Dict[str, TaskSpec] = {task.name: task for task in tasks}
        self._task_order = [task.name for task in tasks]
        self._runtime: Optional[RuntimeState] = None

    @property
    def tasks(self) -> list[str]:
        return list(self._task_order)

    def reset(self, task_name: Optional[str] = None) -> CodeReviewStepResult:
        selected = task_name or self._task_order[0]
        if selected not in self._tasks:
            selected = self._task_order[0]
        self._runtime = RuntimeState(task=self._tasks[selected])
        self._runtime.risk_hotspot_details = self._compute_risk_hotspot_details(self._runtime.task)
        self._runtime.recommendation_hints = self._build_recommendation_hints(self._runtime.task)
        obs = self._observation()
        return CodeReviewStepResult(observation=obs, reward=0.0, done=False, info={"task_name": selected})

    def state(self) -> CodeReviewState:
        runtime = self._require_runtime()
        grade = grade_task(runtime.task, runtime.findings)
        confidence = self._calibrated_confidence(runtime)
        return CodeReviewState(
            episode_id=runtime.episode_id,
            task_name=runtime.task.name,
            step_count=runtime.step_count,
            max_steps=runtime.task.objective.max_steps,
            done=runtime.done,
            score=grade.score,
            penalties=runtime.penalties,
            invalid_actions=runtime.invalid_actions,
            noop_streak=runtime.noop_streak,
            inspected_files=sorted(runtime.inspected_files),
            focus_covered=sorted(runtime.covered_focus_types),
            findings=runtime.findings,
            matched_gt_ids=sorted(runtime.matched_gt_ids),
            partial_gt_ids=sorted(runtime.partial_gt_ids),
            confidence=confidence,
        )

    def step(self, action: CodeReviewAction) -> CodeReviewStepResult:
        runtime = self._require_runtime()
        if runtime.done:
            obs = self._observation()
            return CodeReviewStepResult(observation=obs, reward=0.0, done=True, info={"error": "episode_done"})

        runtime.step_count += 1
        penalties_before = runtime.penalties
        raw_reward = 0.0
        info = {}

        if action.action_type == ActionType.INSPECT_FILE:
            runtime.noop_streak = 0
            raw_reward, info = self._handle_inspect(runtime, action)
        elif action.action_type == ActionType.ADD_FINDING:
            runtime.noop_streak = 0
            raw_reward, info = self._handle_add_finding(runtime, action)
        elif action.action_type == ActionType.SUBMIT_REVIEW:
            runtime.noop_streak = 0
            raw_reward, info = self._handle_submit(runtime, action)
        elif action.action_type == ActionType.NOOP:
            raw_reward = 0.0
            runtime.penalties += 0.02
            runtime.noop_streak += 1
            runtime.last_action_result = "No operation action applied."
            info = {"penalty": "noop"}

        if runtime.noop_streak >= 4 and not runtime.done:
            runtime.done = True
            runtime.done_reason = "stalled_loop"
            runtime.last_action_result = "Episode terminated: repeated noop loop."

        if runtime.step_count >= runtime.task.objective.max_steps and not runtime.done:
            runtime.done = True
            runtime.done_reason = "max_steps_reached"
            runtime.last_action_result = "Episode terminated: max steps reached."

        step_penalty = max(0.0, runtime.penalties - penalties_before)
        shaped_reward = max(0.0, min(1.0, raw_reward - step_penalty))
        obs = self._observation()
        return CodeReviewStepResult(observation=obs, reward=shaped_reward, done=runtime.done, info=info)

    def _handle_inspect(self, runtime: RuntimeState, action: CodeReviewAction) -> tuple[float, dict]:
        target = action.file_path
        file_map = {f.file_path: f.content for f in runtime.task.files}
        if not target or target not in file_map:
            runtime.penalties += 0.03
            runtime.invalid_actions += 1
            runtime.last_action_result = "Inspect failed: file not found in current task."
            return 0.0, {"error": "invalid_file"}

        runtime.open_file = target
        if target in runtime.inspected_files:
            runtime.penalties += 0.005
            runtime.last_action_result = f"File reopened: {target}."
            return 0.01, {"file": target, "repeat": True}

        runtime.inspected_files.add(target)
        multi_file_bonus = 0.02 if len(runtime.inspected_files) >= 2 else 0.0
        hotspot_bonus = 0.0
        prefix = f"{target}:"
        for key, score in runtime.risk_hotspot_details.items():
            if key.startswith(prefix) and score >= 0.72:
                hotspot_bonus = 0.01
                break
        runtime.last_action_result = f"Opened file {target}."
        return min(0.09, 0.04 + multi_file_bonus + hotspot_bonus), {"file": target, "repeat": False}

    def _handle_add_finding(self, runtime: RuntimeState, action: CodeReviewAction) -> tuple[float, dict]:
        if action.finding is None:
            runtime.penalties += 0.03
            runtime.invalid_actions += 1
            runtime.last_action_result = "Add finding failed: missing finding payload."
            return 0.0, {"error": "missing_finding"}

        if not runtime.inspected_files:
            runtime.penalties += 0.05
            runtime.invalid_actions += 1
            runtime.last_action_result = "Add finding failed: inspect at least one file first."
            return 0.0, {"error": "must_inspect_before_finding"}

        finding = action.finding
        duplicate = any(
            f.file_path == finding.file_path
            and f.line == finding.line
            and f.finding_type == finding.finding_type
            and f.issue.strip().lower() == finding.issue.strip().lower()
            for f in runtime.findings
        )
        if duplicate:
            runtime.penalties += 0.04
            runtime.invalid_actions += 1
            runtime.last_action_result = "Duplicate finding detected; penalized."
            return 0.0, {"duplicate": True}

        runtime.findings.append(finding)

        best_reward = 0.01
        best_status = "false_positive"
        matched_id = None
        matched_focus = None

        remaining = [gt for gt in runtime.task.ground_truth if gt.finding_id not in runtime.matched_gt_ids and gt.finding_id not in runtime.partial_gt_ids]
        for gt in remaining:
            exact, partial = match_finding(gt, finding)
            if exact:
                best_reward = 0.25
                best_status = "exact_match"
                matched_id = gt.finding_id
                matched_focus = gt.finding_type.value
                runtime.matched_gt_ids.add(gt.finding_id)
                break
            if partial and best_status != "exact_match":
                best_reward = max(best_reward, 0.12)
                best_status = "partial_match"
                matched_id = gt.finding_id
                matched_focus = gt.finding_type.value

        if best_status == "partial_match" and matched_id is not None:
            runtime.partial_gt_ids.add(matched_id)

        coverage_bonus = 0.0
        if matched_focus is not None and matched_focus not in runtime.covered_focus_types:
            runtime.covered_focus_types.add(matched_focus)
            coverage_bonus = 0.03

        hotspot_alignment_bonus = 0.0
        hotspot_key = f"{finding.file_path}:{finding.line}"
        hotspot_score = runtime.risk_hotspot_details.get(hotspot_key, 0.0)
        if hotspot_score >= 0.72:
            hotspot_alignment_bonus = 0.02

        if best_status == "false_positive":
            runtime.penalties += 0.02
            if len(runtime.findings) > 4:
                runtime.penalties += 0.02

        runtime.last_action_result = f"Finding accepted: {best_status}."
        return min(1.0, best_reward + coverage_bonus + hotspot_alignment_bonus), {
            "match_status": best_status,
            "match_id": matched_id,
            "hotspot_alignment": hotspot_score,
        }

    def _handle_submit(self, runtime: RuntimeState, action: CodeReviewAction) -> tuple[float, dict]:
        grade = grade_task(runtime.task, runtime.findings)
        summary_bonus = 0.0

        summary = (action.summary or "").strip().lower()
        if summary:
            required_words = ["readability", "logging", "comment"]
            covered = sum(1 for word in required_words if word in summary)
            summary_bonus = 0.05 * covered
        else:
            runtime.penalties += 0.03

        early_submit_penalty = 0.0
        min_inspected = max(1, len(runtime.task.files) // 2)
        if runtime.step_count <= 2:
            early_submit_penalty += 0.08
        if len(runtime.inspected_files) < min_inspected:
            early_submit_penalty += 0.08

        confidence = self._calibrated_confidence(runtime)
        confidence_bonus = 0.06 * confidence

        runtime.done = True
        runtime.done_reason = "submitted"
        runtime.last_action_result = "Review submitted."

        final_reward = max(0.0, min(1.0, grade.score + summary_bonus + confidence_bonus - early_submit_penalty))
        return final_reward, {
            "final_score": grade.score,
            "early_submit_penalty": early_submit_penalty,
            "confidence": confidence,
            "precision": grade.precision,
            "recall": grade.recall,
            "exact_matches": grade.exact_matches,
            "partial_matches": grade.partial_matches,
            "false_positives": grade.false_positives,
            "mean_match_score": grade.mean_match_score,
        }

    def _observation(self) -> CodeReviewObservation:
        runtime = self._require_runtime()
        file_map = {f.file_path: f.content for f in runtime.task.files}

        return CodeReviewObservation(
            task_name=runtime.task.name,
            objective=runtime.task.objective,
            step_count=runtime.step_count,
            remaining_steps=max(runtime.task.objective.max_steps - runtime.step_count, 0),
            visible_files=sorted(file_map.keys()),
            open_file=runtime.open_file,
            open_file_content=file_map.get(runtime.open_file) if runtime.open_file else None,
            focus_coverage={
                focus.value: sum(1 for f in runtime.findings if f.finding_type == focus)
                for focus in runtime.task.objective.required_focus
            },
            risk_hotspots=sorted(runtime.risk_hotspot_details.keys(), key=lambda k: runtime.risk_hotspot_details[k], reverse=True)[:6],
            risk_hotspot_details=runtime.risk_hotspot_details,
            recommendation_hints=runtime.recommendation_hints,
            findings_submitted=len(runtime.findings),
            matched_findings=len(runtime.matched_gt_ids),
            partial_matches=len(runtime.partial_gt_ids),
            false_positives=max(len(runtime.findings) - len(runtime.matched_gt_ids) - len(runtime.partial_gt_ids), 0),
            last_action_result=runtime.last_action_result,
            done_reason=runtime.done_reason,
        )

    def _compute_risk_hotspot_details(self, task: TaskSpec) -> Dict[str, float]:
        details: Dict[str, float] = {}
        gt_keys = {(gt.file_path, gt.line) for gt in task.ground_truth}
        for file_snapshot in task.files:
            lines = file_snapshot.content.splitlines()
            for idx, line in enumerate(lines, start=1):
                risk = 0.0
                lower = line.lower()

                if (file_snapshot.file_path, idx) in gt_keys:
                    risk += 0.45
                if "todo" in lower or "fixme" in lower:
                    risk += 0.20
                if re.search(r"\bpass\b", lower):
                    risk += 0.20
                if "write(" in lower or "write_many(" in lower:
                    risk += 0.12
                if re.search(r"==\s*none|!=\s*none", lower):
                    risk += 0.10
                if lower.count("if ") + lower.count(" and ") + lower.count(" or ") >= 2:
                    risk += 0.08

                if risk >= 0.38:
                    details[f"{file_snapshot.file_path}:{idx}"] = min(1.0, risk)

        return details

    def _build_recommendation_hints(self, task: TaskSpec) -> list[str]:
        file_count = len(task.files)
        return [
            f"Inspect at least {max(1, file_count // 2)} files before submit.",
            "Cover all required focus areas: readability, logging, comments.",
            "Prioritize findings near high-risk hotspot lines first.",
        ]

    def _calibrated_confidence(self, runtime: RuntimeState) -> float:
        grade = grade_task(runtime.task, runtime.findings)
        focus_count = len(runtime.task.objective.required_focus)
        covered = len(runtime.covered_focus_types)
        focus_ratio = covered / max(focus_count, 1)
        inspect_ratio = len(runtime.inspected_files) / max(len(runtime.task.files), 1)
        confidence = (0.45 * grade.precision) + (0.35 * grade.recall) + (0.12 * focus_ratio) + (0.08 * inspect_ratio)
        return max(0.0, min(1.0, confidence))

    def _require_runtime(self) -> RuntimeState:
        if self._runtime is None:
            self._runtime = RuntimeState(task=self._tasks[self._task_order[0]])
        return self._runtime
