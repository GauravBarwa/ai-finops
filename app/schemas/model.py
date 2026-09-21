"""Public-facing model information schemas."""

from __future__ import annotations

from pydantic import BaseModel, Field


class ModelInfoResponse(BaseModel):
    """Model metadata exposed through the API.

    Intentionally does not include the LiteLLM model name (internal detail).
    """

    model_id: str
    provider: str
    display_name: str
    context_window: int
    input_price_per_1m_tokens: float
    output_price_per_1m_tokens: float
    capabilities: list[str] = Field(default_factory=list)
    enabled: bool = True


class ModelListResponse(BaseModel):
    """Wrapper for a list of models returned by ``GET /models``."""

    models: list[ModelInfoResponse]
    total: int
