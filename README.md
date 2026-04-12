---
title: code-review-quality-env
colorFrom: blue
colorTo: indigo
sdk: docker
app_port: 7860
tags:
  - openenv
  - reinforcement-learning
  - code-review
---

# Code Review Quality OpenEnv

A real-world OpenEnv environment for training and evaluating agents on **code review quality work**: readability issues, logging coverage, and comment quality checks in backend Python services.

This is not a toy environment. It simulates a practical workflow software teams run daily during pull-request review.

## Hackathon-Grade Advanced Features

- Semantic weighted grading with multi-factor similarity.
- Optimal bipartite assignment between predicted and ground-truth findings.
- Risk-hotspot mining from code complexity and failure-path patterns.
- Confidence-calibrated final reward and richer environment telemetry.
- Adaptive inference policy with deterministic hard-task plans and UCB-style exploration.

## Why This Environment Is Useful

Human reviewers repeatedly catch:
- readability and maintainability issues,
- missing or weak logging in failure paths,
- low-quality comments and TODO hygiene.

This environment gives agent researchers a deterministic, graded setting to evaluate those capabilities with shaped rewards.

## Real-World Task Modeled

Domain: Code review for production services.

Agent responsibilities:
- inspect files,
- submit structured review findings,
- submit final review summary.

## OpenEnv API Surface

Server endpoints:
- `POST /reset` -> initial observation
- `POST /step` -> observation, reward, done, info
- `GET /state` -> full current episode state
- `GET /tasks` -> task list
- `GET /metrics/leaderboard` -> judge-facing benchmark snapshot (oracle vs sparse baseline)
- `GET /health` -> liveness

Typed Pydantic models:
- `CodeReviewAction`
- `CodeReviewObservation`
- `CodeReviewStepResult`
- `CodeReviewState`

Manifest:
- `openenv.yaml`

## Action Space

`CodeReviewAction` fields:
- `action_type`: one of `inspect_file`, `add_finding`, `submit_review`, `noop`
- `file_path`: used by `inspect_file`
- `finding`: structured finding payload used by `add_finding`
- `summary`: review summary used by `submit_review`
- `note`: optional note for `noop`

`ReviewFinding` fields:
- `file_path`, `line`
- `finding_type` in `{readability, logging, comments}`
- `severity` in `{low, medium, high}`
- `issue`, `suggestion`

## Observation Space

`CodeReviewObservation` includes:
- current `task_name` and `objective`
- `visible_files`
- optional `open_file` and `open_file_content`
- progress counters: findings submitted, matched findings, partial matches, false positives
- `risk_hotspot_details` with per-line risk score
- `recommendation_hints` for strategic action selection
- `last_action_result`
- optional `done_reason`

`CodeReviewState` includes:
- calibrated `confidence` in `[0.0, 1.0]`

## Task Set (Easy -> Hard)

1. `easy_api_logging_review` (easy)
- Single API handler.
- Objective: catch obvious readability, logging, and comment defects.

2. `medium_batch_job_review` (medium)
- ETL sync pipeline + helper module.
- Objective: detect instrumentation gaps and maintainability concerns across files.

3. `hard_service_refactor_review` (hard)
- Multi-step service/report generation path.
- Objective: identify subtle readability refactor opportunities, TODO quality issues, and missing write-time logs.

4. `hard_incident_postmortem_review` (hard)
- Production incident recovery + retry + alert modules.
- Objective: detect weak observability, poor variable naming, and insufficient policy documentation.

5. `hard_distributed_checkout_review` (hard)
- Distributed checkout orchestrator + outbox + billing path.
- Objective: catch idempotency-risk comments, missing failure-path logs, and validation/readability problems.

All tasks have deterministic graders with score in `[0.0, 1.0]`.

## Reward Function

Per-step shaped reward:
- `inspect_file` on new file: positive reward.
- `add_finding` exact GT match: strong reward.
- partial GT match: medium reward.
- hotspot-aligned finding bonus when a finding targets high-risk lines.
- false positives / duplicates / invalid actions / noop loops: penalties.
- `submit_review`: final reward from deterministic grader + summary coverage + confidence bonus.

State score is always clamped to `[0.0, 1.0]`.

Grader internals:
- Match score combines keyword overlap, line-distance decay, semantic token overlap, and severity alignment.
- Matching uses maximum-weight bipartite assignment for global optimality.
- Final score combines weighted F1/recall with bounded false-positive penalties.

## Project Structure

- `code_quality_env/models.py` - typed action/observation/state models
- `code_quality_env/tasks.py` - task definitions and ground truth
- `code_quality_env/graders.py` - deterministic grading logic
- `code_quality_env/server/review_environment.py` - core environment logic
- `code_quality_env/server/app.py` - FastAPI app
- `code_quality_env/client.py` - async client and local Docker launcher
- `openenv.yaml` - environment manifest
- `inference.py` - advanced baseline policy with strict `[START] [STEP] [END]` stdout logs

## Setup

### 1) Local Python install

```bash
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\\Scripts\\activate
pip install -U pip
pip install -e .
```

### 2) Run server locally

```bash
uvicorn code_quality_env.server.app:app --host 0.0.0.0 --port 7860
```

### 3) Quick API test

```bash
curl -X POST http://127.0.0.1:7860/reset -H "Content-Type: application/json" -d '{"task_name":"easy_api_logging_review"}'
```

## Containerized Execution

Build and run:

```bash
docker build -t code-review-quality-env:latest .
docker run --rm -p 7860:7860 code-review-quality-env:latest
```

Health check:

```bash
curl http://127.0.0.1:7860/health
```

Leaderboard metrics snapshot:

```bash
curl http://127.0.0.1:7860/metrics/leaderboard
```

## Hugging Face Space Deployment (Docker)

1. Create a new **Docker Space**.
2. Push this repository to the Space.
3. Ensure Space metadata includes tag `openenv`.
4. Wait for build to finish, then confirm:

```bash
curl -X POST https://<your-space>.hf.space/reset -H "Content-Type: application/json" -d '{}'
```

Expected HTTP code: `200`.

## Baseline Inference Script

Required file: root `inference.py` (included).

Mandatory env vars:
- `API_BASE_URL`
- `MODEL_NAME`
- `API_KEY`
- `LOCAL_IMAGE_NAME` (for local docker inference mode)

Run baseline:

```bash
API_KEY=<token> \
API_BASE_URL=https://router.huggingface.co/v1 \
MODEL_NAME=Qwen/Qwen2.5-72B-Instruct \
LOCAL_IMAGE_NAME=code-review-quality-env:latest \
python inference.py
```

The script emits strict logs:
- `[START] ...`
- `[STEP] ...`
- `[END] ...`

and reports reproducible per-task scores in `[0,1]` (temperature fixed to `0.0`).

Policy highlights in `inference.py`:
- deterministic task-specific finding plans for hard tasks,
- heuristic mining of candidate findings from open file content,
- candidate reranking with risk-hotspot alignment,
- adaptive exploration bonus (UCB-style) over action families.

## Local Dry-Run Scoring (No External LLM Needed)

Run a deterministic benchmark snapshot directly from Python:

```bash
python -c "from code_quality_env.server.metrics import build_leaderboard_snapshot; import json; print(json.dumps(build_leaderboard_snapshot(), indent=2))"
```

This produces per-task and aggregate scores for:
- `oracle` predictions (upper-bound quality),
- `sparse_baseline` predictions (lower-bound quality),
- averaged benchmark summary for quick judge demos.

## Pre-Submission Steps To Pass Initial Requirements

1. Install validator dependencies:
```bash
pip install openenv-core
```

2. Verify Docker build:
```bash
docker build -t code-review-quality-env:latest .
```

3. Validate OpenEnv manifest/spec locally:
```bash
openenv validate
```

4. Run baseline inference end-to-end:
```bash
API_KEY=<token> API_BASE_URL=<endpoint> MODEL_NAME=<model> LOCAL_IMAGE_NAME=code-review-quality-env:latest python inference.py
```

5. Deploy Docker Space on HF and verify `/reset` returns `200`.

6. Run provided `validate-submission.sh` against your Space URL.

## Baseline Score Notes

Expected baseline depends on selected model and endpoint quality. With deterministic temperature (`0.0`) and fixed prompts, score variance should remain low across repeated runs on the same model.
