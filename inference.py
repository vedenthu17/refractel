"""
Baseline inference script for code-review-quality-env.

Mandatory env vars expected by submission pipeline:
- API_BASE_URL
- MODEL_NAME
- HF_TOKEN
- LOCAL_IMAGE_NAME (optional, for local docker execution)
"""

from __future__ import annotations

import asyncio
import os
import textwrap
from typing import List, Optional

from openai import OpenAI

from code_quality_env.client import CodeReviewEnv
from code_quality_env.models import ActionType, CodeReviewAction, FindingType, ReviewFinding, Severity

API_BASE_URL = os.getenv("API_BASE_URL", "https://router.huggingface.co/v1")
MODEL_NAME = os.getenv("MODEL_NAME", "Qwen/Qwen2.5-72B-Instruct")
HF_TOKEN = os.getenv("HF_TOKEN")
LOCAL_IMAGE_NAME = os.getenv("LOCAL_IMAGE_NAME") or os.getenv("IMAGE_NAME") or "code-review-quality-env:latest"
FALLBACK_ENV_BASE_URL = os.getenv("ENV_BASE_URL", "http://127.0.0.1:7860")

MAX_STEPS = 24
TEMPERATURE = 0.0
MAX_TOKENS = 240
BENCHMARK = "code-review-quality-env"
TASKS = [
    "easy_api_logging_review",
    "medium_batch_job_review",
    "hard_service_refactor_review",
]


def log_start(task: str, env: str, model: str) -> None:
    print(f"[START] task={task} env={env} model={model}", flush=True)


def log_step(step: int, action: str, reward: float, done: bool, error: Optional[str]) -> None:
    error_val = error if error else "null"
    done_val = str(done).lower()
    print(
        f"[STEP] step={step} action={action} reward={reward:.2f} done={done_val} error={error_val}",
        flush=True,
    )


def log_end(success: bool, steps: int, score: float, rewards: List[float]) -> None:
    rewards_str = ",".join(f"{r:.2f}" for r in rewards)
    print(
        f"[END] success={str(success).lower()} steps={steps} score={score:.3f} rewards={rewards_str}",
        flush=True,
    )


def _safe_text(s: str) -> str:
    return s.replace("\n", " ").replace("\r", " ").strip()


def build_system_prompt() -> str:
    return textwrap.dedent(
        """
        You are a code review agent for readability, logging quality, and comments quality.
        Output ONLY JSON with one schema:
        {
          "action_type": "inspect_file" | "add_finding" | "submit_review" | "noop",
          "file_path": "... optional ...",
          "summary": "... optional ...",
          "finding": {
            "file_path": "...",
            "line": <int>,
            "finding_type": "readability" | "logging" | "comments",
            "severity": "low" | "medium" | "high",
            "issue": "...",
            "suggestion": "..."
          }
        }
        Never include markdown fences.
        """
    ).strip()


def build_user_prompt(task: str, obs: dict, step: int, history: list[str]) -> str:
    return textwrap.dedent(
        f"""
        Task: {task}
        Step: {step}
        Objective: {obs['objective']['goal']}
        Visible files: {obs['visible_files']}
        Open file: {obs.get('open_file')}
        Open content:\n{obs.get('open_file_content')}
        Findings submitted: {obs['findings_submitted']}
        Matched findings: {obs['matched_findings']}
        Last action result: {obs['last_action_result']}
        Prior actions: {history[-5:] if history else []}

        Choose the best next action.
        """
    ).strip()


def parse_action(raw_text: str) -> CodeReviewAction:
    import json

    try:
        data = json.loads(raw_text)
    except Exception:
        return CodeReviewAction(action_type=ActionType.NOOP, note="parser_fallback")

    try:
        action_type = ActionType(data.get("action_type", "noop"))
    except Exception:
        action_type = ActionType.NOOP

    if action_type == ActionType.ADD_FINDING and isinstance(data.get("finding"), dict):
        f = data["finding"]
        try:
            finding = ReviewFinding(
                file_path=f.get("file_path", "unknown.py"),
                line=max(1, int(f.get("line", 1))),
                finding_type=FindingType(f.get("finding_type", "readability")),
                severity=Severity(f.get("severity", "low")),
                issue=str(f.get("issue", "Needs improvement."))[:500],
                suggestion=str(f.get("suggestion", "Refactor for clarity."))[:500],
            )
            return CodeReviewAction(action_type=ActionType.ADD_FINDING, finding=finding)
        except Exception:
            return CodeReviewAction(action_type=ActionType.NOOP, note="invalid_finding")

    if action_type == ActionType.INSPECT_FILE:
        return CodeReviewAction(action_type=ActionType.INSPECT_FILE, file_path=str(data.get("file_path") or ""))

    if action_type == ActionType.SUBMIT_REVIEW:
        return CodeReviewAction(
            action_type=ActionType.SUBMIT_REVIEW,
            summary=str(data.get("summary") or "readability logging comments reviewed"),
        )

    return CodeReviewAction(action_type=ActionType.NOOP, note=str(data.get("note") or "noop"))


def get_llm_action(client: OpenAI, task: str, obs: dict, step: int, history: list[str]) -> CodeReviewAction:
    user_prompt = build_user_prompt(task=task, obs=obs, step=step, history=history)
    try:
        completion = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": build_system_prompt()},
                {"role": "user", "content": user_prompt},
            ],
            temperature=TEMPERATURE,
            max_tokens=MAX_TOKENS,
            stream=False,
        )
        text = (completion.choices[0].message.content or "").strip()
        return parse_action(text)
    except Exception:
        if obs.get("open_file") is None and obs.get("visible_files"):
            return CodeReviewAction(action_type=ActionType.INSPECT_FILE, file_path=obs["visible_files"][0])
        if step >= 5:
            return CodeReviewAction(
                action_type=ActionType.SUBMIT_REVIEW,
                summary="readability logging comments reviewed with fallback policy",
            )
        return CodeReviewAction(action_type=ActionType.NOOP, note="model_error")


def _finding_key(action: CodeReviewAction) -> tuple[str, int, str, str] | None:
    if action.action_type != ActionType.ADD_FINDING or action.finding is None:
        return None
    return (
        action.finding.file_path,
        action.finding.line,
        action.finding.finding_type.value,
        action.finding.issue.strip().lower(),
    )


async def run_task(task_name: str, client: OpenAI, env: CodeReviewEnv) -> float:
    rewards: List[float] = []
    history: list[str] = []
    steps_taken = 0
    score = 0.0
    success = False
    seen_finding_keys: set[tuple[str, int, str, str]] = set()
    no_progress_steps = 0
    last_progress = 0

    log_start(task=task_name, env=BENCHMARK, model=MODEL_NAME)
    try:
        result = await env.reset(task_name=task_name)
        obs = result.observation.model_dump()

        for step in range(1, MAX_STEPS + 1):
            if result.done:
                break

            if step >= 5 and no_progress_steps >= 3:
                action = CodeReviewAction(
                    action_type=ActionType.SUBMIT_REVIEW,
                    summary="readability logging comments review completed with prioritized findings",
                )
            else:
                action = get_llm_action(client=client, task=task_name, obs=obs, step=step, history=history)

            # Ensure the agent inspects at least one file before submitting findings.
            if obs.get("open_file") is None and action.action_type in {ActionType.ADD_FINDING, ActionType.SUBMIT_REVIEW}:
                files = obs.get("visible_files") or []
                if files:
                    action = CodeReviewAction(action_type=ActionType.INSPECT_FILE, file_path=files[0])

            # Avoid duplicate findings that trigger penalties.
            finding_key = _finding_key(action)
            if finding_key is not None and finding_key in seen_finding_keys:
                action = CodeReviewAction(
                    action_type=ActionType.SUBMIT_REVIEW,
                    summary="readability logging comments review completed with non-duplicate findings",
                )

            result = await env.step(action)

            reward = float(result.reward or 0.0)
            rewards.append(reward)
            steps_taken = step
            obs = result.observation.model_dump()

            action_str = _safe_text(action.model_dump_json())
            err = result.info.get("error") if isinstance(result.info, dict) else None
            log_step(step=step, action=action_str, reward=reward, done=result.done, error=err)

            finding_key = _finding_key(action)
            if finding_key is not None:
                seen_finding_keys.add(finding_key)

            progress = int(obs.get("matched_findings", 0)) + int(obs.get("partial_matches", 0))
            if progress > last_progress:
                no_progress_steps = 0
                last_progress = progress
            else:
                no_progress_steps += 1

            history.append(f"step={step} action={action.action_type.value} reward={reward:.2f}")

            if result.done:
                break

        st = await env.state()
        score = float(st.get("score", 0.0))
        score = max(0.0, min(1.0, score))
        success = score >= 0.5

    finally:
        log_end(success=success, steps=steps_taken, score=score, rewards=rewards)

    return score


async def main() -> None:
    # Keep OpenAI client usage as required, but do not hard-fail when token is absent in validators.
    client = OpenAI(base_url=API_BASE_URL, api_key=HF_TOKEN or "")

    env: CodeReviewEnv | None = None
    try:
        env = await CodeReviewEnv.from_docker_image(LOCAL_IMAGE_NAME)
    except Exception as exc:
        print(f"[DEBUG] from_docker_image failed: {exc}", flush=True)
        env = CodeReviewEnv(base_url=FALLBACK_ENV_BASE_URL)

    scores: list[float] = []
    try:
        async with env:
            for task in TASKS:
                try:
                    s = await run_task(task, client, env)
                except Exception as task_exc:
                    print(f"[DEBUG] task {task} failed: {task_exc}", flush=True)
                    s = 0.0
                scores.append(s)
    except Exception as env_exc:
        print(f"[DEBUG] environment session failed: {env_exc}", flush=True)

    avg = sum(scores) / max(len(scores), 1)
    print(f"Average score across tasks: {avg:.3f}", flush=True)


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except Exception as exc:
        print(f"[DEBUG] fatal inference error: {exc}", flush=True)
