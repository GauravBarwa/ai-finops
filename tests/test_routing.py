"""Constraint boundaries, policy tradeoffs, validation and live fallback with fake adapters."""

from unittest.mock import AsyncMock

import pytest
from pydantic import ValidationError

from app.domain.routing import recommend
from app.llm.interface import LLMResponse
from app.llm.registry import ModelInfo, ModelRegistry
from app.schemas.profile import ModelProfile, ProfilesResponse
from app.schemas.routing import Constraints, RouteRequest, Weights
from app.services.evaluation_service import EvaluationService
from app.services.router_service import RouterService
from app.services.telemetry_service import TelemetryService


def models():
    return [
        ModelInfo(
            model_id=name,
            provider="test",
            display_name=name,
            litellm_model_name="test/" + name,
            context_window=4096,
            input_price_per_1m_tokens=1,
            output_price_per_1m_tokens=1,
            capabilities=["classification"],
        )
        for name in ["cheap", "quality", "missing"]
    ]


def profiles():
    return [
        ModelProfile(
            model_id=name,
            task_type="classification",
            quality_score=quality,
            median_latency_ms=latency,
            p95_latency_ms=latency * 2,
            avg_estimated_cost=cost,
            reliability_rate=1,
            sample_count=10,
        )
        for name, quality, latency, cost in [
            ("cheap", 0.8, 100, 0.001),
            ("quality", 0.99, 200, 0.01),
        ]
    ]


@pytest.mark.parametrize(
    "policy,selected",
    [
        ("cost_first", "cheap"),
        ("quality_first", "quality"),
        ("latency_first", "cheap"),
    ],
)
def test_policies(policy, selected):
    result = recommend(
        RouteRequest(prompt="label", task_type="classification", policy=policy),
        models(),
        profiles(),
    )
    assert result.selected_model_id == selected
    assert result.candidates[2].rejection_reasons == ["profile_available: failed"]
    assert result.candidates[2].score is None


def test_hard_constraints_and_disabled():
    request = RouteRequest(
        prompt="label",
        task_type="classification",
        constraints=Constraints(
            min_quality=0.8,
            max_cost=0.001,
            max_latency_ms=100,
            min_reliability=1,
        ),
    )
    result = recommend(request, models(), profiles())
    assert result.ranked_model_ids == ["cheap"]
    assert result.candidates[1].checks["max_cost"] is False
    request.constraints.min_quality = 0.9
    assert recommend(request, models(), profiles()).status == "no_candidate"
    disabled = models()
    disabled[0].enabled = False
    disabled[1].capabilities = []
    assert recommend(request, disabled, profiles()).status == "no_candidate"


def test_ties_and_custom_weights():
    equal = profiles()
    equal[1] = equal[0].model_copy(update={"model_id": "quality"})
    result = recommend(RouteRequest(prompt="label", task_type="classification"), models(), equal)
    assert result.ranked_model_ids == ["cheap", "quality"]
    assert result.candidates[0].score == 1
    request = RouteRequest(
        prompt="label",
        task_type="classification",
        weights=Weights(quality=1, cost=0, latency=0, reliability=0),
    )
    assert recommend(request, models(), profiles()).selected_model_id == "quality"


@pytest.mark.parametrize(
    "kwargs",
    [
        {"constraints": {"min_quality": 2}},
        {"constraints": {"max_cost": -1}},
        {"constraints": {"max_latency_ms": float("inf")}},
        {"weights": dict(quality=0, cost=0, latency=0, reliability=0)},
        {"weights": dict(quality=1e308, cost=1e308)},
        {"policy": "invalid"},
        {"prompt": "   "},
        {"task_type": "unknown"},
    ],
)
def test_validation(kwargs):
    with pytest.raises(ValidationError):
        RouteRequest(**({"prompt": "label", "task_type": "classification"} | kwargs))


@pytest.mark.asyncio
async def test_execution_fallback_and_telemetry(tmp_path):
    evaluation = EvaluationService(tmp_path)
    evaluation.profiles_path.write_text(ProfilesResponse(profiles=profiles()).model_dump_json())
    adapter = AsyncMock()
    adapter.generate.side_effect = [
        RuntimeError("offline"),
        LLMResponse(content="YES", input_tokens=10),
    ]
    telemetry = TelemetryService(tmp_path)
    service = RouterService(adapter, ModelRegistry(models()), evaluation, telemetry)
    request = RouteRequest(
        prompt="label", task_type="classification", policy="cost_first", routing_mode="task_average"
    )
    assert (await service.route(request)).status == "recommended"
    adapter.generate.assert_not_called()
    request.execute = True
    result = await service.route(request)
    assert result.selected_model_id == "cheap"
    assert result.executed_model_id == "quality"
    assert result.fallback_used and result.response == "YES"
    assert len(result.attempts) == 2
    records = telemetry.load_records()
    assert len(records) == 2
    assert all(r.source == "route" for r in records)
    assert result.total_estimated_cost == sum(r.estimated_cost for r in records)


@pytest.mark.asyncio
async def test_all_fail_and_no_eligible_fallback(tmp_path):
    evaluation = EvaluationService(tmp_path)
    evaluation.profiles_path.write_text(ProfilesResponse(profiles=profiles()).model_dump_json())
    adapter = AsyncMock()
    adapter.generate.return_value = LLMResponse(success=False, error="rate limited")
    service = RouterService(
        adapter,
        ModelRegistry(models()),
        evaluation,
        TelemetryService(tmp_path),
    )
    result = await service.route(
        RouteRequest(prompt="label", task_type="classification", execute=True)
    )
    assert result.status == "execution_failed"
    assert adapter.generate.call_count == 2
    adapter.generate.reset_mock()
    result = await service.route(
        RouteRequest(
            prompt="label",
            task_type="classification",
            execute=True,
            constraints=Constraints(min_quality=1),
        )
    )
    assert result.status == "no_candidate"
    adapter.generate.assert_not_called()
