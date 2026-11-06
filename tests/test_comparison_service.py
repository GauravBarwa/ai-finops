"""Tests for the comparison service with mocked LLM calls."""

from __future__ import annotations

from unittest.mock import AsyncMock

import pytest

from app.llm.interface import LLMClient, LLMResponse
from app.llm.registry import ModelInfo, ModelRegistry
from app.schemas.comparison import CompareRequest
from app.services.comparison_service import ComparisonService


def _make_registry() -> ModelRegistry:
    """Build an in-memory registry with two test models."""
    models = [
        ModelInfo(
            model_id="model-a",
            provider="openai",
            display_name="Model A",
            litellm_model_name="test/model-a",
            context_window=128000,
            input_price_per_1m_tokens=0.15,
            output_price_per_1m_tokens=0.60,
            enabled=True,
        ),
        ModelInfo(
            model_id="model-b",
            provider="google",
            display_name="Model B",
            litellm_model_name="test/model-b",
            context_window=1048576,
            input_price_per_1m_tokens=0.10,
            output_price_per_1m_tokens=0.40,
            enabled=True,
        ),
    ]
    return ModelRegistry(models)


def _make_mock_client(responses: dict[str, LLMResponse]) -> LLMClient:
    """Create a mock LLM client that returns predefined responses per model."""
    client = AsyncMock(spec=LLMClient)

    async def mock_generate(prompt, model, **kwargs):
        return responses.get(model, LLMResponse(
            content="",
            success=False,
            error="Unknown model",
            model=model,
        ))

    client.generate = mock_generate
    return client


class TestComparisonService:
    """Test the comparison service with mocked providers."""

    @pytest.mark.asyncio
    async def test_compare_all_succeed(self):
        registry = _make_registry()
        responses = {
            "test/model-a": LLMResponse(
                content="Response from A",
                input_tokens=100,
                output_tokens=50,
                total_tokens=150,
                latency_ms=500.0,
                model="test/model-a",
                provider="openai",
                success=True,
            ),
            "test/model-b": LLMResponse(
                content="Response from B",
                input_tokens=100,
                output_tokens=40,
                total_tokens=140,
                latency_ms=300.0,
                model="test/model-b",
                provider="google",
                success=True,
            ),
        }
        client = _make_mock_client(responses)
        service = ComparisonService(llm_client=client, registry=registry)

        request = CompareRequest(prompt="Test prompt")
        result = await service.compare(request)

        assert result.total_models == 2
        assert result.successful == 2
        assert result.failed == 0
        assert len(result.results) == 2

        # Verify cost was calculated
        for r in result.results:
            assert r.estimated_cost > 0

    @pytest.mark.asyncio
    async def test_compare_partial_failure(self):
        registry = _make_registry()
        responses = {
            "test/model-a": LLMResponse(
                content="Response from A",
                input_tokens=100,
                output_tokens=50,
                total_tokens=150,
                latency_ms=500.0,
                model="test/model-a",
                provider="openai",
                success=True,
            ),
            "test/model-b": LLMResponse(
                content="",
                input_tokens=0,
                output_tokens=0,
                total_tokens=0,
                latency_ms=5000.0,
                model="test/model-b",
                provider="google",
                success=False,
                error="Timeout",
            ),
        }
        client = _make_mock_client(responses)
        service = ComparisonService(llm_client=client, registry=registry)

        request = CompareRequest(prompt="Test prompt")
        result = await service.compare(request)

        assert result.successful == 1
        assert result.failed == 1

    @pytest.mark.asyncio
    async def test_compare_specific_models(self):
        registry = _make_registry()
        responses = {
            "test/model-a": LLMResponse(
                content="Only A",
                input_tokens=50,
                output_tokens=20,
                total_tokens=70,
                latency_ms=200.0,
                model="test/model-a",
                provider="openai",
                success=True,
            ),
        }
        client = _make_mock_client(responses)
        service = ComparisonService(llm_client=client, registry=registry)

        request = CompareRequest(prompt="Test", models=["model-a"])
        result = await service.compare(request)

        assert result.total_models == 1
        assert result.results[0].model_id == "model-a"

    @pytest.mark.asyncio
    async def test_task_id_is_unique(self):
        registry = _make_registry()
        responses = {
            "test/model-a": LLMResponse(
                content="hi", input_tokens=10, output_tokens=5,
                total_tokens=15, latency_ms=100.0,
                model="test/model-a", provider="openai", success=True,
            ),
        }
        client = _make_mock_client(responses)
        service = ComparisonService(llm_client=client, registry=registry)

        r1 = await service.compare(CompareRequest(prompt="test", models=["model-a"]))
        r2 = await service.compare(CompareRequest(prompt="test", models=["model-a"]))

        assert r1.task_id != r2.task_id
