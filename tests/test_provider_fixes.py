"""Provider regressions, bounded calls, incremental progress and literal error rendering."""

import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from app.core.config import Settings
from app.demo import build_demo, format_comparison
from app.llm.interface import LLMResponse
from app.llm.litellm_client import LiteLLMClient
from app.llm.registry import ModelInfo, ModelRegistry
from app.schemas.comparison import CompareRequest, CompareResponse, ModelResult
from app.services.comparison_service import ComparisonService


def registry():
    return ModelRegistry(
        [
            ModelInfo(
                model_id=name,
                provider="test",
                display_name=name,
                litellm_model_name="test/" + name,
                context_window=4096,
                input_price_per_1m_tokens=1,
                output_price_per_1m_tokens=1,
            )
            for name in ["fast", "slow"]
        ]
    )


def completion(content="YES", usage=True):
    return SimpleNamespace(
        usage=SimpleNamespace(prompt_tokens=10, completion_tokens=5) if usage else None,
        choices=[SimpleNamespace(message=SimpleNamespace(content=content))],
    )


@pytest.mark.asyncio
async def test_progress_does_not_wait_for_slowest():
    release = asyncio.Event()
    adapter = AsyncMock()

    async def generate(prompt, model, **kwargs):
        if model.endswith("slow"):
            await release.wait()
        return LLMResponse(content="YES", latency_ms=1)

    adapter.generate.side_effect = generate
    stream = ComparisonService(adapter, registry()).stream_compare(CompareRequest(prompt="label"))
    initial = await anext(stream)
    assert initial.results == [] and initial.total_models == 2
    partial = await asyncio.wait_for(anext(stream), 1)
    assert partial.successful == 1 and partial.results[0].model_id == "fast"
    assert partial.failed == 0
    release.set()
    final = await anext(stream)
    assert final.successful == 2
    assert [r.model_id for r in final.results] == ["fast", "slow"]
    assert initial.task_id == partial.task_id == final.task_id
    assert partial.successful == 1  # previously yielded snapshot was not mutated
    await stream.aclose()


@pytest.mark.asyncio
async def test_hung_model_times_out_while_other_model_completes():
    adapter = AsyncMock()
    forever = asyncio.Event()

    async def generate(prompt, model, **kwargs):
        if model.endswith("slow"):
            await forever.wait()
        return LLMResponse(content="YES")

    adapter.generate.side_effect = generate
    result = await ComparisonService(adapter, registry(), timeout_seconds=0.02).compare(
        CompareRequest(prompt="label")
    )
    assert result.successful == result.failed == 1
    assert "timed out" in result.results[1].error
    assert result.results[1].latency_ms >= 10


@pytest.mark.asyncio
async def test_disconnecting_consumer_cancels_provider_tasks():
    cancelled = asyncio.Event()
    adapter = AsyncMock()

    async def generate(prompt, model, **kwargs):
        if model.endswith("slow"):
            try:
                await asyncio.Event().wait()
            finally:
                cancelled.set()
        return LLMResponse(content="YES")

    adapter.generate.side_effect = generate
    stream = ComparisonService(adapter, registry()).stream_compare(CompareRequest(prompt="label"))
    await anext(stream)
    await anext(stream)
    await stream.aclose()
    assert cancelled.is_set()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "model,has_temperature",
    [
        ("gemini/gemini-3.6-flash", False),
        ("groq/openai/gpt-oss-20b", True),
    ],
)
async def test_provider_generation_options(monkeypatch, model, has_temperature):
    settings = Settings(_env_file=None, gemini_api_key="fake-gemini", groq_api_key="fake-groq")
    monkeypatch.setattr("app.llm.litellm_client.get_settings", lambda: settings)
    generate = AsyncMock(return_value=completion())
    monkeypatch.setattr("app.llm.litellm_client.litellm.acompletion", generate)
    result = await LiteLLMClient().generate("label", model, reasoning_effort="low")
    assert result.success and result.total_tokens == 15
    kwargs = generate.call_args.kwargs
    assert ("temperature" in kwargs) == has_temperature
    assert kwargs["num_retries"] == 1
    assert kwargs["reasoning_effort"] == "low"


@pytest.mark.asyncio
async def test_missing_keys_and_retired_provider_fail_without_network(monkeypatch):
    monkeypatch.setattr(
        "app.llm.litellm_client.get_settings",
        lambda: Settings(_env_file=None, gemini_api_key="", groq_api_key=""),
    )
    generate = AsyncMock()
    monkeypatch.setattr("app.llm.litellm_client.litellm.acompletion", generate)
    for model in ["gemini/gemini-3.6-flash", "groq/openai/gpt-oss-20b", "github/gpt-4o-mini"]:
        response = await LiteLLMClient().generate("label", model)
        assert not response.success
        assert "Missing" in response.error or "retired" in response.error
    generate.assert_not_called()


@pytest.mark.asyncio
async def test_no_usage_and_empty_answer(monkeypatch):
    settings = Settings(_env_file=None, groq_api_key="fake-groq")
    monkeypatch.setattr("app.llm.litellm_client.get_settings", lambda: settings)
    generate = AsyncMock(side_effect=[completion(usage=False), completion(content="")])
    monkeypatch.setattr("app.llm.litellm_client.litellm.acompletion", generate)
    result = await LiteLLMClient().generate("label", "groq/openai/gpt-oss-20b")
    assert result.success and result.total_tokens == 0
    result = await LiteLLMClient().generate("label", "groq/openai/gpt-oss-20b")
    assert not result.success and result.total_tokens == 15
    assert "no answer text" in result.error


@pytest.mark.asyncio
async def test_error_credentials_redacted(monkeypatch):
    settings = Settings(_env_file=None, groq_api_key="fake-secret-token")
    monkeypatch.setattr("app.llm.litellm_client.get_settings", lambda: settings)
    monkeypatch.setattr(
        "app.llm.litellm_client.litellm.acompletion",
        AsyncMock(side_effect=RuntimeError("bad key: fake-secret-token")),
    )
    result = await LiteLLMClient().generate("label", "groq/openai/gpt-oss-20b")
    assert "fake-secret-token" not in result.error
    assert "[REDACTED]" in result.error


def test_multiline_errors_cannot_break_results_table():
    error = 'NotFoundError: {\n"message": "missing | model"\n}\n<script>bad</script>'
    result = CompareResponse(
        task_id="a",
        prompt="label",
        total_models=2,
        failed=1,
        results=[
            ModelResult(model_id="m", provider="test", display_name="M", success=False, error=error)
        ],
    )
    summary, rows, details = format_comparison(result)
    assert "1 pending" in summary
    assert len(rows) == 1 and len(rows[0]) == 6
    assert "\n" not in rows[0][-1]
    assert error in details  # full literal text goes to a Textbox, not Markdown


def test_gradio_binds_async_generator_to_comparison():
    demo = build_demo()
    fn = next(fn for fn in demo.fns.values() if fn.fn and fn.fn.__name__ == "_compare")
    import inspect

    assert inspect.isasyncgenfunction(fn.fn)
    assert len(fn.outputs) == 3
    assert fn.outputs[-1].get_config()["type"] == "text"


def test_replacement_registry_and_retired_reason():
    models = ModelRegistry.from_config()
    enabled = models.get_model_ids()
    assert {
        "gemini-3.6-flash",
        "gemini-3.5-flash-lite",
        "groq-gpt-oss-20b",
        "groq-gpt-oss-120b",
    }.issubset(enabled)
    assert not any("llama-3." in mid for mid in enabled)
    github = models.get("github-gpt-4o-mini")
    assert not github.enabled and "retired" in github.disabled_reason
