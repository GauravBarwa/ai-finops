"""Offline end-to-end API, benchmark resilience, provider config and KPI validation."""

import json
from unittest.mock import AsyncMock

import pytest
from httpx import ASGITransport, AsyncClient

from app.api import routes
from app.core import config
from app.llm.interface import LLMResponse
from app.llm.litellm_client import LiteLLMClient
from app.llm.registry import ModelInfo, ModelRegistry
from app.main import create_app
from app.schemas.routing import Constraints
from app.schemas.telemetry import TelemetryRecord
from app.services.benchmark_service import BenchmarkService, BenchmarkTask
from app.services.evaluation_service import EvaluationService
from app.services.telemetry_service import TelemetryService
from scripts.evaluate_router_kpi import simulate, write_report


def registry():
    return ModelRegistry(
        [
            ModelInfo(
                model_id=name,
                provider="test",
                display_name=name,
                litellm_model_name="test/" + name,
                context_window=4096,
                input_price_per_1m_tokens=price,
                output_price_per_1m_tokens=price,
                capabilities=["classification"],
            )
            for name, price in [("cheap", 0.1), ("baseline", 1)]
        ]
    )


def dataset():
    tasks = [
        BenchmarkTask("a", "classification", "label A", "YES"),
        BenchmarkTask("b", "classification", "label B", "NO"),
    ]
    records = [
        TelemetryRecord(
            task_id=task.task_id,
            task_type=task.task_type,
            prompt=task.prompt,
            model_id=model,
            provider="test",
            response=task.reference,
            input_tokens=100,
            output_tokens=10,
            total_tokens=110,
            estimated_cost=0,
            latency_ms=latency,
            source="benchmark",
        )
        for task in tasks
        for model, latency in [("cheap", 50), ("baseline", 100)]
    ]
    return tasks, records


@pytest.mark.asyncio
async def test_api_evaluate_profiles_route(tmp_path):
    tasks, records = dataset()
    task_path = tmp_path / "tasks.json"
    task_path.write_text(json.dumps([vars(t) for t in tasks]))
    run_path = tmp_path / "benchmark_test.jsonl"
    BenchmarkService.save_results(records, run_path)
    models = registry()
    evaluation = EvaluationService(tmp_path, task_path, models)
    adapter = AsyncMock()
    adapter.generate.return_value = LLMResponse(content="YES", input_tokens=100, output_tokens=10)
    routes.init_services(models, adapter, TelemetryService(tmp_path), evaluation)
    async with AsyncClient(
        transport=ASGITransport(app=create_app()), base_url="http://test"
    ) as client:
        assert (await client.get("/profiles")).json()["profiles"] == []
        evaluated = await client.post("/evaluate", json={"run_files": [run_path.name]})
        assert evaluated.status_code == 200
        assert len(evaluated.json()["profiles"]) == 2
        assert evaluated.json()["pricing_basis"] == "shadow_commercial_usd"
        assert evaluated.json()["pricing_hash"]
        stored = await client.get("/profiles")
        assert stored.json() == evaluated.json()
        request = dict(prompt="label", task_type="classification", policy="cost_first")
        recommended = await client.post("/route", json=request)
        assert recommended.json()["selected_model_id"] == "cheap"
        adapter.generate.assert_not_called()
        executed = await client.post("/route", json=request | {"execute": True})
        assert executed.json()["status"] == "succeeded"
        assert executed.json()["response"] == "YES"
        rejected = await client.post("/route", json=request | {"constraints": {"max_cost": 0}})
        assert rejected.json()["status"] == "no_candidate"
        invalid = await client.post("/evaluate", json={"run_files": ["../.env"]})
        assert invalid.status_code == 400
        assert (
            await client.post("/route", json=request | {"policy": "unknown"})
        ).status_code == 422
    # Shadow repricing never rewrites source logs.
    assert all(
        json.loads(line)["estimated_cost"] == 0 for line in run_path.read_text().splitlines()
    )


@pytest.mark.asyncio
async def test_benchmark_continues_after_adapter_exception(tmp_path):
    tasks, _ = dataset()
    adapter = AsyncMock()
    adapter.generate.side_effect = [
        RuntimeError("offline"),
        LLMResponse(content="NO", latency_ms=42),
    ]
    service = BenchmarkService(adapter, registry())
    records = await service.run_benchmark(["cheap"], tasks)
    assert len(records) == 2
    assert not records[0].success and records[0].latency_ms >= 0
    assert records[1].success
    saved = service.save_results(records, tmp_path / "benchmark_test.jsonl")
    assert len(saved.read_text().splitlines()) == 2


@pytest.mark.parametrize(
    "name,provider",
    [
        ("groq/llama-3.1-8b-instant", "groq"),
        ("github/gpt-4o-mini", "github"),
        ("gemini/gemini-2.0-flash", "google"),
        ("ollama/llama3.2:latest", "ollama"),
    ],
)
def test_provider_prefix(name, provider):
    assert LiteLLMClient._extract_provider(name) == provider


def test_free_provider_keys_exported(monkeypatch):
    settings = config.Settings(
        _env_file=None,
        gemini_api_key="fake-gemini",
        groq_api_key="fake-groq",
        github_api_key="fake-github",
    )
    for key in ["GEMINI_API_KEY", "GROQ_API_KEY", "GITHUB_API_KEY"]:
        monkeypatch.delenv(key, raising=False)
    monkeypatch.setattr(config, "Settings", lambda: settings)
    config.get_settings.cache_clear()
    try:
        config.get_settings()
        import os

        assert os.environ["GROQ_API_KEY"] == "fake-groq"
        assert os.environ["GITHUB_API_KEY"] == "fake-github"
    finally:
        config.get_settings.cache_clear()


def test_kpi_paired_replay_and_unobserved_models(tmp_path):
    tasks, records = dataset()
    result = simulate(records, tasks, registry(), "baseline")
    assert result["paired_tasks"] == 2
    assert result["cost_reduction_pct"] == pytest.approx(90)
    assert result["quality_retained_pct"] == pytest.approx(100)
    assert result["latency_change_pct"] == pytest.approx(-50)
    path = tmp_path / "report.md"
    write_report(result, path, "baseline", "cost_first", ["benchmark_test.jsonl"])
    assert "90.00%" in path.read_text()
    # Holding out the only task leaves no profile, rather than leaking test outcomes.
    result = simulate(records[:2], tasks[:1], registry(), "baseline")
    assert result["paired_tasks"] == 0
    assert result["skipped_tasks"] == ["a"]
    result = simulate(records, tasks, registry(), "baseline", constraints=Constraints(max_cost=0))
    assert result["paired_tasks"] == 0


def test_registry_shadow_prices_and_task_coverage():
    models = ModelRegistry.from_config().list_enabled()
    assert len(models) == 6
    assert all(m.input_price_per_1m_tokens > 0 and m.output_price_per_1m_tokens > 0 for m in models)
    tasks = BenchmarkService.load_tasks()
    assert 12 <= len(tasks) <= 16
    assert all(t.reference for t in tasks)
    assert {t.task_type for t in tasks} == {
        "classification",
        "summarization",
        "extraction",
        "reasoning",
    }
