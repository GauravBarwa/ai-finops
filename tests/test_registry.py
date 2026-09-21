"""Tests for the model registry."""

from __future__ import annotations

import tempfile
from pathlib import Path

import pytest
import yaml

from app.core.exceptions import ConfigurationError, ModelNotFoundError
from app.llm.registry import ModelInfo, ModelRegistry


def _write_config(models: list[dict], tmp_path: Path) -> Path:
    """Write a temporary models.yaml and return the path."""
    config_path = tmp_path / "models.yaml"
    with open(config_path, "w") as f:
        yaml.dump({"models": models}, f)
    return config_path


SAMPLE_MODELS = [
    {
        "model_id": "model-a",
        "provider": "openai",
        "display_name": "Model A",
        "litellm_model_name": "gpt-4o-mini",
        "context_window": 128000,
        "input_price_per_1m_tokens": 0.15,
        "output_price_per_1m_tokens": 0.60,
        "capabilities": ["summarization", "classification"],
        "enabled": True,
    },
    {
        "model_id": "model-b",
        "provider": "google",
        "display_name": "Model B",
        "litellm_model_name": "gemini/gemini-2.0-flash",
        "context_window": 1048576,
        "input_price_per_1m_tokens": 0.10,
        "output_price_per_1m_tokens": 0.40,
        "capabilities": ["summarization"],
        "enabled": True,
    },
    {
        "model_id": "model-disabled",
        "provider": "test",
        "display_name": "Disabled Model",
        "litellm_model_name": "test/disabled",
        "context_window": 4096,
        "input_price_per_1m_tokens": 0.0,
        "output_price_per_1m_tokens": 0.0,
        "capabilities": [],
        "enabled": False,
    },
]


class TestModelInfo:
    """Test the ``ModelInfo`` Pydantic model."""

    def test_valid_model(self):
        m = ModelInfo(**SAMPLE_MODELS[0])
        assert m.model_id == "model-a"
        assert m.enabled is True

    def test_default_enabled(self):
        data = SAMPLE_MODELS[0].copy()
        del data["enabled"]
        m = ModelInfo(**data)
        assert m.enabled is True


class TestModelRegistry:
    """Test the ``ModelRegistry`` class."""

    def test_from_config(self, tmp_path):
        config = _write_config(SAMPLE_MODELS, tmp_path)
        registry = ModelRegistry.from_config(config)
        assert len(registry.list_all()) == 3
        assert len(registry.list_enabled()) == 2

    def test_get_existing(self, tmp_path):
        config = _write_config(SAMPLE_MODELS, tmp_path)
        registry = ModelRegistry.from_config(config)
        model = registry.get("model-a")
        assert model.provider == "openai"

    def test_get_missing_raises(self, tmp_path):
        config = _write_config(SAMPLE_MODELS, tmp_path)
        registry = ModelRegistry.from_config(config)
        with pytest.raises(ModelNotFoundError):
            registry.get("nonexistent")

    def test_list_by_capability(self, tmp_path):
        config = _write_config(SAMPLE_MODELS, tmp_path)
        registry = ModelRegistry.from_config(config)
        summarizers = registry.list_by_capability("summarization")
        assert len(summarizers) == 2

        classifiers = registry.list_by_capability("classification")
        assert len(classifiers) == 1

    def test_get_model_ids(self, tmp_path):
        config = _write_config(SAMPLE_MODELS, tmp_path)
        registry = ModelRegistry.from_config(config)
        ids = registry.get_model_ids(enabled_only=True)
        assert "model-a" in ids
        assert "model-b" in ids
        assert "model-disabled" not in ids

    def test_missing_config_raises(self, tmp_path):
        fake_path = tmp_path / "nonexistent.yaml"
        with pytest.raises(ConfigurationError):
            ModelRegistry.from_config(fake_path)

    def test_invalid_config_raises(self, tmp_path):
        config_path = tmp_path / "bad.yaml"
        config_path.write_text("foo: bar")
        with pytest.raises(ConfigurationError):
            ModelRegistry.from_config(config_path)
