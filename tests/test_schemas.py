"""Tests for Pydantic request / response schemas."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from app.schemas.comparison import CompareRequest, CompareResponse, ModelResult
from app.schemas.model import ModelInfoResponse, ModelListResponse
from app.schemas.telemetry import TelemetryRecord


class TestCompareRequest:
    """Validate ``CompareRequest`` schema."""

    def test_minimal_request(self):
        req = CompareRequest(prompt="Hello")
        assert req.prompt == "Hello"
        assert req.task_type is None
        assert req.models is None
        assert req.max_tokens == 1024
        assert req.temperature == 0.0

    def test_full_request(self):
        req = CompareRequest(
            prompt="Summarize this",
            task_type="summarization",
            models=["gpt-4o-mini", "gemini-2.0-flash"],
            max_tokens=512,
            temperature=0.5,
        )
        assert req.task_type == "summarization"
        assert len(req.models) == 2

    def test_empty_prompt_rejected(self):
        with pytest.raises(ValidationError):
            CompareRequest(prompt="")

    def test_temperature_bounds(self):
        with pytest.raises(ValidationError):
            CompareRequest(prompt="test", temperature=3.0)

    def test_max_tokens_bounds(self):
        with pytest.raises(ValidationError):
            CompareRequest(prompt="test", max_tokens=0)


class TestModelResult:
    """Validate ``ModelResult`` schema."""

    def test_successful_result(self):
        r = ModelResult(
            model_id="test",
            provider="openai",
            display_name="Test Model",
            response="Hello!",
            input_tokens=10,
            output_tokens=5,
            total_tokens=15,
            latency_ms=100.0,
            estimated_cost=0.001,
        )
        assert r.success is True
        assert r.error is None

    def test_failed_result(self):
        r = ModelResult(
            model_id="test",
            provider="openai",
            display_name="Test Model",
            success=False,
            error="Timeout",
        )
        assert r.success is False
        assert r.error == "Timeout"


class TestCompareResponse:
    """Validate ``CompareResponse`` schema."""

    def test_serialization_roundtrip(self):
        response = CompareResponse(
            task_id="abc-123",
            prompt="test prompt",
            results=[
                ModelResult(
                    model_id="m1",
                    provider="openai",
                    display_name="Model 1",
                    response="hi",
                    latency_ms=50.0,
                )
            ],
            total_models=1,
            successful=1,
            failed=0,
        )
        json_str = response.model_dump_json()
        restored = CompareResponse.model_validate_json(json_str)
        assert restored.task_id == "abc-123"
        assert len(restored.results) == 1
        assert restored.results[0].model_id == "m1"


class TestModelInfoResponse:
    """Validate ``ModelInfoResponse`` schema."""

    def test_basic(self):
        m = ModelInfoResponse(
            model_id="test",
            provider="openai",
            display_name="Test",
            context_window=128000,
            input_price_per_1m_tokens=0.15,
            output_price_per_1m_tokens=0.60,
        )
        assert m.enabled is True
        assert m.capabilities == []


class TestModelListResponse:
    """Validate ``ModelListResponse`` schema."""

    def test_empty_list(self):
        resp = ModelListResponse(models=[], total=0)
        assert resp.total == 0


class TestTelemetryRecord:
    """Validate ``TelemetryRecord`` schema."""

    def test_defaults(self):
        r = TelemetryRecord(
            task_id="t1",
            model_id="m1",
            provider="test",
        )
        assert r.success is True
        assert r.source == "compare"
        assert r.input_tokens == 0

    def test_serialization_roundtrip(self):
        r = TelemetryRecord(
            task_id="t1",
            model_id="m1",
            provider="openai",
            prompt="hello",
            response="world",
            input_tokens=10,
            output_tokens=5,
            total_tokens=15,
            latency_ms=200.0,
            estimated_cost=0.001,
            source="benchmark",
        )
        json_str = r.model_dump_json()
        restored = TelemetryRecord.model_validate_json(json_str)
        assert restored.task_id == "t1"
        assert restored.estimated_cost == 0.001
