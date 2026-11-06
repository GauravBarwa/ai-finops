"""Schemas for the ``POST /compare`` endpoint."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Optional

from pydantic import BaseModel, Field


class CompareRequest(BaseModel):
    """Request body for comparing a prompt across multiple models.

    Attributes
    ----------
    prompt : str
        The prompt / instruction to send to every model.
    task_type : str | None
        Optional workload type label (e.g. ``summarization``, ``classification``).
        Used for filtering and later routing — not enforced at this stage.
    models : list[str] | None
        Optional list of model IDs to compare.  When ``None`` / empty,
        all enabled models in the registry are used.
    max_tokens : int
        Maximum tokens to generate per model call.
    temperature : float
        Sampling temperature for all model calls.
    """

    prompt: str = Field(
        ..., min_length=1, pattern=r"\S", description="The prompt to compare across models"
    )
    task_type: Optional[str] = Field(
        default=None,
        description="Optional workload type (summarization, classification, extraction)",
    )
    models: Optional[list[str]] = Field(
        default=None,
        description="Model IDs to compare. Defaults to all enabled models.",
    )
    max_tokens: int = Field(default=1024, ge=1, le=8192)
    temperature: float = Field(default=0.0, ge=0.0, le=2.0)


class ModelResult(BaseModel):
    """Per-model result within a comparison response."""

    model_id: str
    provider: str
    display_name: str
    response: str = ""
    input_tokens: int = 0
    output_tokens: int = 0
    total_tokens: int = 0
    latency_ms: float = 0.0
    estimated_cost: float = 0.0
    success: bool = True
    error: Optional[str] = None


class CompareResponse(BaseModel):
    """Aggregated comparison response from ``POST /compare``."""

    task_id: str
    prompt: str
    task_type: Optional[str] = None
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    results: list[ModelResult] = Field(default_factory=list)
    total_models: int = 0
    successful: int = 0
    failed: int = 0
