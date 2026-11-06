"""Validated routing policy, constraints, candidate explanations and execution results."""

import math
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from app.schemas.comparison import ModelResult
from app.schemas.profile import ModelProfile, TaskType
from app.schemas.prompt import CandidateEstimate, PromptAnalysis

PolicyName = Literal["cost_first", "balanced", "quality_first", "latency_first"]


class Constraints(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)

    min_quality: float = Field(default=0, ge=0, le=1)
    max_cost: float | None = Field(
        default=None,
        ge=0,
        description="Shadow USD per candidate call. Prompt-aware filtering "
        "uses estimated input plus the maximum output-token budget.",
    )
    max_latency_ms: float | None = Field(
        default=None,
        ge=0,
        description="Prompt-aware: token-scaled latency estimate; "
        "baseline: historical median. Neither guarantees a deadline.",
    )
    min_reliability: float = Field(default=0, ge=0, le=1)


class Weights(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)

    quality: float = Field(default=0.25, ge=0)
    cost: float = Field(default=0.25, ge=0)
    latency: float = Field(default=0.25, ge=0)
    reliability: float = Field(default=0.25, ge=0)

    @model_validator(mode="after")
    def positive_total(self):
        total = sum(self.model_dump().values())
        if not math.isfinite(total) or total <= 0:
            raise ValueError("Weights must have a finite positive total")
        return self


class RouteRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    prompt: str = Field(min_length=1, pattern=r"\S")
    task_type: TaskType | Literal["auto"] = "auto"
    routing_mode: Literal["prompt_aware", "task_average"] = "prompt_aware"
    constraints: Constraints = Field(default_factory=Constraints)
    policy: PolicyName = "balanced"
    weights: Weights | None = None
    execute: bool = False
    max_tokens: int = Field(default=1024, ge=1, le=8192)
    temperature: float = Field(default=0, ge=0, le=2)


class CandidateDecision(BaseModel):
    model_id: str
    passed: bool
    checks: dict[str, bool] = Field(default_factory=dict)
    rejection_reasons: list[str] = Field(default_factory=list)
    profile: ModelProfile | None = None
    estimate: CandidateEstimate | None = None
    score: float | None = None
    normalized_scores: dict[str, float] = Field(default_factory=dict)
    rank: int | None = None


class RouteResponse(BaseModel):
    routing_mode: Literal["prompt_aware", "task_average"] = "prompt_aware"
    prompt_analysis: PromptAnalysis | None = None
    selected_model_id: str | None = None
    executed_model_id: str | None = None
    policy: PolicyName
    weights: Weights
    candidates: list[CandidateDecision]
    ranked_model_ids: list[str] = Field(default_factory=list)
    status: Literal["recommended", "no_candidate", "succeeded", "execution_failed"]
    explanation: str
    response: str = ""
    attempts: list[ModelResult] = Field(default_factory=list)
    fallback_used: bool = False
    total_estimated_cost: float = 0
