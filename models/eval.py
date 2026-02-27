"""
Pydantic models for eval-gated deployments: golden inputs, eval suites, eval runs, deployment status.
"""
from enum import Enum
from datetime import datetime
from typing import Optional, List, Any
from uuid import UUID

from pydantic import BaseModel


class EvalRunStatus(str, Enum):
    running = "running"
    passed = "passed"
    failed = "failed"
    error = "error"


class DeploymentStatus(str, Enum):
    candidate = "candidate"
    evaluating = "evaluating"
    promoted = "promoted"
    failed = "failed"
    rolled_back = "rolled_back"


class GoldenInput(BaseModel):
    id: UUID
    org_id: UUID
    workflow_id: UUID
    name: Optional[str] = None
    input_text: Optional[str] = None
    variables: Optional[dict] = None
    expected_output: Optional[str] = None
    source: str = "manual"
    source_run_id: Optional[UUID] = None
    created_at: datetime
    updated_at: Optional[datetime] = None

    class Config:
        from_attributes = True


class EvalCheck(BaseModel):
    type: str  # "deterministic" | "regression" | "model_graded"
    name: str
    enabled: bool = True
    config: dict = {}


class EvalSuite(BaseModel):
    id: UUID
    org_id: UUID
    workflow_id: UUID
    name: str
    checks: List[EvalCheck]
    enabled: bool = True
    created_at: Optional[datetime] = None
    updated_at: Optional[datetime] = None

    class Config:
        from_attributes = True


class EvalRunResult(BaseModel):
    id: Optional[UUID] = None
    eval_run_id: UUID
    golden_input_id: Optional[UUID] = None
    check_name: str
    check_type: str
    passed: bool
    candidate_output: Optional[str] = None
    candidate_latency_ms: Optional[int] = None
    candidate_cost: Optional[float] = None
    production_output: Optional[str] = None
    production_latency_ms: Optional[int] = None
    production_cost: Optional[float] = None
    latency_delta_pct: Optional[float] = None
    cost_delta_pct: Optional[float] = None
    output_changed: Optional[bool] = None
    score: Optional[float] = None
    grade_reasoning: Optional[str] = None
    failure_reason: Optional[str] = None
    created_at: Optional[datetime] = None


class EvalRun(BaseModel):
    id: UUID
    org_id: UUID
    deployment_id: UUID
    eval_suite_id: Optional[UUID] = None
    status: EvalRunStatus = EvalRunStatus.running
    results: dict = {}
    summary: dict = {}
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    created_at: Optional[datetime] = None

    class Config:
        from_attributes = True


class DeploymentPolicy(BaseModel):
    id: UUID
    org_id: UUID
    rule_type: str
    config: dict = {}
    enabled: bool = True
    created_at: Optional[datetime] = None
    updated_at: Optional[datetime] = None

    class Config:
        from_attributes = True
