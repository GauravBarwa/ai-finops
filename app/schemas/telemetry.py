"""Telemetry record schema for persistent storage."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Optional

from pydantic import BaseModel, Field


class TelemetryRecord(BaseModel):
    """A single execution record persisted to storage.

    Every model call — whether from a comparison, benchmark, or future
    routing — produces one ``TelemetryRecord``.
    """

    # ── Identifiers ──────────────────────────────────────────────────────
    task_id: str
    model_id: str
    provider: str

    # ── Task context ─────────────────────────────────────────────────────
    task_type: Optional[str] = None
    prompt: str = ""

    # ── Result ───────────────────────────────────────────────────────────
    response: str = ""
    success: bool = True
    error: Optional[str] = None

    # ── Telemetry ────────────────────────────────────────────────────────
    input_tokens: int = 0
    output_tokens: int = 0
    total_tokens: int = 0
    latency_ms: float = 0.0
    estimated_cost: float = 0.0

    # ── Metadata ─────────────────────────────────────────────────────────
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    source: str = "compare"  # "compare" | "benchmark" | "route"
