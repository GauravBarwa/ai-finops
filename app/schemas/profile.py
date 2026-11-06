"""Evaluation request and persistent model profile contracts."""

from datetime import datetime, timezone
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

TaskType = Literal["summarization", "classification", "extraction", "reasoning"]


class ModelProfile(BaseModel):
    model_config = ConfigDict(allow_inf_nan=False)

    model_id: str
    task_type: TaskType
    quality_score: float = Field(ge=0, le=1)
    median_latency_ms: float = Field(ge=0)
    p95_latency_ms: float = Field(ge=0)
    avg_estimated_cost: float = Field(ge=0)
    reliability_rate: float = Field(ge=0, le=1)
    sample_count: int = Field(ge=1)


class ProfilesResponse(BaseModel):
    profiles: list[ModelProfile] = Field(default_factory=list)
    scorer_version: str = "deterministic-v1"
    dataset_hash: str = ""
    pricing_hash: str = ""
    source_files: list[str] = Field(default_factory=list)
    updated_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    pricing_basis: str = "shadow_commercial_usd"


class EvaluateRequest(BaseModel):
    # Basenames inside results_dir only; never arbitrary filesystem paths.
    run_files: list[str] | None = None
