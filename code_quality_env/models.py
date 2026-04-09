from __future__ import annotations

from enum import Enum
from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel, Field


class ActionType(str, Enum):
    INSPECT_FILE = "inspect_file"
    ADD_FINDING = "add_finding"
    SUBMIT_REVIEW = "submit_review"
    NOOP = "noop"


class FindingType(str, Enum):
    READABILITY = "readability"
    LOGGING = "logging"
    COMMENTS = "comments"

    


class Severity(str, Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


class ReviewFinding(BaseModel):
    file_path: str
    line: int = Field(ge=1)
    finding_type: FindingType
    severity: Severity
    issue: str = Field(min_length=5, max_length=500)
    suggestion: str = Field(min_length=5, max_length=500)


class CodeReviewAction(BaseModel):
    action_type: ActionType
    file_path: Optional[str] = None
    finding: Optional[ReviewFinding] = None
    summary: Optional[str] = None
    note: Optional[str] = None


class FileSnapshot(BaseModel):
    file_path: str
    content: str


class TaskObjective(BaseModel):
    goal: str
    required_focus: List[FindingType]
    max_steps: int = Field(ge=4, le=64)


class CodeReviewObservation(BaseModel):
    task_name: str
    objective: TaskObjective
    step_count: int
    remaining_steps: int
    visible_files: List[str]
    open_file: Optional[str] = None
    open_file_content: Optional[str] = None
    focus_coverage: Dict[str, int] = Field(default_factory=dict)
    risk_hotspots: List[str] = Field(default_factory=list)
    findings_submitted: int
    matched_findings: int
    partial_matches: int
    false_positives: int
    last_action_result: str
    done_reason: Optional[str] = None


class CodeReviewState(BaseModel):
    episode_id: str
    task_name: str
    step_count: int
    max_steps: int
    done: bool
    score: float = Field(ge=0.0, le=1.0)
    penalties: float = Field(ge=0.0)
    invalid_actions: int = Field(ge=0)
    noop_streak: int = Field(ge=0)
    inspected_files: List[str]
    focus_covered: List[str] = Field(default_factory=list)
    findings: List[ReviewFinding]
    matched_gt_ids: List[str]
    partial_gt_ids: List[str]


class CodeReviewStepResult(BaseModel):
    observation: CodeReviewObservation
    reward: float = Field(ge=0.0, le=1.0)
    done: bool
    info: Dict[str, Any] = Field(default_factory=dict)


class GroundTruthFinding(BaseModel):
    finding_id: str
    file_path: str
    line: int
    finding_type: FindingType
    severity: Severity
    must_include_keywords: List[str] = Field(default_factory=list)


class TaskSpec(BaseModel):
    name: str
    difficulty: Literal["easy", "medium", "hard"]
    scenario: str = "code_review"
    objective: TaskObjective
    files: List[FileSnapshot]
    ground_truth: List[GroundTruthFinding]
