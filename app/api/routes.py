"""FastAPI route definitions for the AI FinOps service.

Endpoints
---------
POST /compare   — Compare a prompt across multiple models
GET  /models    — List all enabled models
GET  /models/{model_id} — Get a single model's info
GET  /health    — Health check
"""

from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, status

from app.core.exceptions import ModelNotFoundError
from app.llm.litellm_client import LiteLLMClient
from app.llm.registry import ModelRegistry
from app.schemas.comparison import CompareRequest, CompareResponse
from app.schemas.model import ModelInfoResponse, ModelListResponse
from app.services.comparison_service import ComparisonService
from app.services.telemetry_service import TelemetryService

router = APIRouter()

# ── Shared singletons (initialised at app startup via dependency injection) ──
_registry: ModelRegistry | None = None
_comparison_service: ComparisonService | None = None
_telemetry_service: TelemetryService | None = None


def init_services(registry: ModelRegistry) -> None:
    """Initialise service singletons.  Called once at app startup."""
    global _registry, _comparison_service, _telemetry_service

    _registry = registry
    llm_client = LiteLLMClient()
    _comparison_service = ComparisonService(llm_client=llm_client, registry=registry)
    _telemetry_service = TelemetryService()


def _get_registry() -> ModelRegistry:
    if _registry is None:
        raise RuntimeError("Services not initialised — call init_services() first")
    return _registry


def _get_comparison_service() -> ComparisonService:
    if _comparison_service is None:
        raise RuntimeError("Services not initialised — call init_services() first")
    return _comparison_service


def _get_telemetry_service() -> TelemetryService:
    if _telemetry_service is None:
        raise RuntimeError("Services not initialised — call init_services() first")
    return _telemetry_service


# ── Health ────────────────────────────────────────────────────────────────────

@router.get("/health", tags=["system"])
async def health_check():
    """Simple health check endpoint."""
    registry = _get_registry()
    return {
        "status": "healthy",
        "service": "ai-finops",
        "models_loaded": len(registry.list_enabled()),
    }


# ── Models ────────────────────────────────────────────────────────────────────

@router.get("/models", response_model=ModelListResponse, tags=["models"])
async def list_models():
    """Return all enabled models from the registry."""
    registry = _get_registry()
    models = registry.list_enabled()
    items = [
        ModelInfoResponse(
            model_id=m.model_id,
            provider=m.provider,
            display_name=m.display_name,
            context_window=m.context_window,
            input_price_per_1m_tokens=m.input_price_per_1m_tokens,
            output_price_per_1m_tokens=m.output_price_per_1m_tokens,
            capabilities=m.capabilities,
            enabled=m.enabled,
        )
        for m in models
    ]
    return ModelListResponse(models=items, total=len(items))


@router.get("/models/{model_id}", response_model=ModelInfoResponse, tags=["models"])
async def get_model(model_id: str):
    """Return info for a single model."""
    registry = _get_registry()
    try:
        m = registry.get(model_id)
    except ModelNotFoundError:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Model not found: {model_id}",
        )
    return ModelInfoResponse(
        model_id=m.model_id,
        provider=m.provider,
        display_name=m.display_name,
        context_window=m.context_window,
        input_price_per_1m_tokens=m.input_price_per_1m_tokens,
        output_price_per_1m_tokens=m.output_price_per_1m_tokens,
        capabilities=m.capabilities,
        enabled=m.enabled,
    )


# ── Compare ───────────────────────────────────────────────────────────────────

@router.post("/compare", response_model=CompareResponse, tags=["compare"])
async def compare_models(request: CompareRequest):
    """Run the same prompt against multiple models and return comparative metrics.

    If ``request.models`` is empty or omitted, all enabled models are used.
    """
    service = _get_comparison_service()
    telemetry = _get_telemetry_service()

    response = await service.compare(request)

    # Persist telemetry asynchronously (fire-and-forget for now)
    telemetry.save_comparison(response)

    return response
