"""Request analysis and pre-execution estimates, separate from measured model profiles."""

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from app.schemas.profile import TaskType


class PromptMatch(BaseModel):
    task_id: str
    similarity: float = Field(ge=0, le=1)


class PromptAnalysis(BaseModel):
    detected_task_type: TaskType
    effective_task_type: TaskType
    task_type_source: Literal["auto", "explicit"]
    input_tokens_rough: int = Field(ge=1)
    requested_max_output_tokens: int = Field(ge=1)
    output_tokens_hint: int | None = Field(default=None, ge=1)
    signals: list[str] = Field(default_factory=list)
    matched_tasks: list[PromptMatch] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)
    assessment_method: str = "task_profile_fallback"


class CandidateEstimate(BaseModel):
    model_config = ConfigDict(allow_inf_nan=False)

    quality_score: float = Field(ge=0, le=1)
    estimated_cost: float = Field(ge=0)
    budget_estimated_cost: float = Field(ge=0)
    estimated_latency_ms: float = Field(ge=0)
    reliability_rate: float = Field(ge=0, le=1)
    estimated_input_tokens: int = Field(ge=1)
    estimated_output_tokens: int = Field(ge=1)
    supporting_sample_count: int = Field(ge=0)
    matched_task_ids: list[str] = Field(default_factory=list)
    quality_basis: str
