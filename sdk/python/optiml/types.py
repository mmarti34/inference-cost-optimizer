"""Response types for the OptiML SDK."""

from __future__ import annotations

from typing import Any, Optional

from pydantic import BaseModel, Field


class WorkflowResponse(BaseModel):
    """Response from executing a workflow endpoint."""

    final_output: Optional[str] = None
    content_type: Optional[str] = None  # "text", "image_url", "audio_url", "embedding"
    total_cost: Optional[float] = None
    total_latency_ms: Optional[int] = Field(None, alias="total_latency")
    run_id: Optional[str] = None
    request_id: Optional[str] = None
    served_version: Optional[int] = None
    experiment_id: Optional[str] = None
    variant_name: Optional[str] = None
    node_results: Optional[list[dict[str, Any]]] = None

    model_config = {"populate_by_name": True}


class StreamEvent(BaseModel):
    """A single event from a streaming workflow execution."""

    event: str  # "token", "node_start", "node_complete", "done", "error"
    data: dict[str, Any] = {}


class FeedbackResponse(BaseModel):
    """Response from submitting feedback/custom metrics."""

    status: str
    message: Optional[str] = None
