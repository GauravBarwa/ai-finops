"""Tests for the FastAPI endpoints with mocked services."""

from __future__ import annotations

from unittest.mock import AsyncMock, patch, MagicMock
from datetime import datetime, timezone

import pytest
from fastapi.testclient import TestClient

from app.llm.registry import ModelInfo, ModelRegistry
from app.main import create_app
from app.api import routes


@pytest.fixture
def test_registry() -> ModelRegistry:
    """Build a small test registry."""
    models = [
        ModelInfo(
            model_id="test-model",
            provider="test",
            display_name="Test Model",
            litellm_model_name="test/test-model",
            context_window=128000,
            input_price_per_1m_tokens=0.10,
            output_price_per_1m_tokens=0.40,
            enabled=True,
        ),
    ]
    return ModelRegistry(models)


@pytest.fixture
def client(test_registry) -> TestClient:
    """Create a test client with mocked services."""
    app = create_app()

    # Manually initialise services with test registry
    routes.init_services(test_registry)

    return TestClient(app)


class TestHealthEndpoint:
    def test_health_returns_200(self, client):
        resp = client.get("/health")
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "healthy"
        assert data["models_loaded"] >= 1


class TestModelsEndpoint:
    def test_list_models(self, client):
        resp = client.get("/models")
        assert resp.status_code == 200
        data = resp.json()
        assert data["total"] >= 1
        assert len(data["models"]) >= 1
        assert data["models"][0]["model_id"] == "test-model"

    def test_get_model_by_id(self, client):
        resp = client.get("/models/test-model")
        assert resp.status_code == 200
        data = resp.json()
        assert data["model_id"] == "test-model"
        assert data["provider"] == "test"

    def test_get_model_not_found(self, client):
        resp = client.get("/models/nonexistent")
        assert resp.status_code == 404


class TestCompareEndpoint:
    def test_compare_validation_error(self, client):
        """Empty prompt should be rejected."""
        resp = client.post("/compare", json={"prompt": ""})
        assert resp.status_code == 422

    def test_compare_missing_prompt(self, client):
        """Missing prompt should be rejected."""
        resp = client.post("/compare", json={})
        assert resp.status_code == 422
