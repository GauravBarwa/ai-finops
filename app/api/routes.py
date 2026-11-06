"""FastAPI endpoints for model metadata, comparison, evaluation, profiles, and routing.

GET /health confirms initialization. GET /models and GET /models/{model_id} expose
registry metadata. POST /compare executes selected models. POST /evaluate scores
saved benchmark runs, GET /profiles reads their profiles, and POST /route recommends
or executes a model using prompt analysis, measured evidence, and constraints.
"""

from __future__ import annotations

from fastapi import APIRouter, HTTPException, status

from app.core.exceptions import BenchmarkError, ModelNotFoundError
from app.llm.interface import LLMClient
from app.llm.provider_client import ProviderRouterClient
from app.llm.registry import ModelRegistry
from app.schemas.comparison import CompareRequest, CompareResponse
from app.schemas.model import ModelInfoResponse, ModelListResponse
from app.schemas.profile import EvaluateRequest, ProfilesResponse
from app.schemas.routing import RouteRequest, RouteResponse
from app.services.comparison_service import ComparisonService
from app.services.evaluation_service import EvaluationService
from app.services.router_service import RouterService
from app.services.telemetry_service import TelemetryService

router = APIRouter()

# ── Shared singletons (initialised at app startup via dependency injection) ──
_registry: ModelRegistry | None = None
_comparison_service: ComparisonService | None = None
_telemetry_service: TelemetryService | None = None
_evaluation_service: EvaluationService | None = None
_router_service: RouterService | None = None


def init_services(
    registry: ModelRegistry,
    llm_client: LLMClient | None = None,
    telemetry_service: TelemetryService | None = None,
    evaluation_service: EvaluationService | None = None,
) -> None:
    """Initialise service singletons.  Called once at app startup."""
    global _registry, _comparison_service, _telemetry_service, _evaluation_service

    global _router_service
    _registry = registry
    llm_client = llm_client or ProviderRouterClient()
    _comparison_service = ComparisonService(llm_client=llm_client, registry=registry)
    _telemetry_service = telemetry_service or TelemetryService()
    _evaluation_service = evaluation_service or EvaluationService(registry=registry)
    _router_service = RouterService(
        llm_client,
        registry,
        _evaluation_service,
        _telemetry_service,
    )


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
            disabled_reason=m.disabled_reason,
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
        disabled_reason=m.disabled_reason,
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


@router.post("/evaluate", response_model=ProfilesResponse, tags=["evaluation"])
async def evaluate_models(request: EvaluateRequest):
    """Score saved benchmarks; no live model calls are made."""
    if _evaluation_service is None:
        raise RuntimeError("Services not initialised")
    try:
        return _evaluation_service.evaluate(request.run_files)
    except (ValueError, OSError, BenchmarkError) as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@router.get("/profiles", response_model=ProfilesResponse, tags=["evaluation"])
async def list_profiles():
    if _evaluation_service is None:
        raise RuntimeError("Services not initialised")
    try:
        return _evaluation_service.load_profiles()
    except (ValueError, OSError) as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc


@router.post("/route", response_model=RouteResponse, tags=["routing"])
async def route_request(request: RouteRequest):
    """Recommend or execute with hard constraints and one ranked fallback."""
    if _router_service is None:
        raise RuntimeError("Services not initialised")
    return await _router_service.route(request)
