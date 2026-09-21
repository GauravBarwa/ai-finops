"""Configuration-driven model registry.

Loads model metadata and pricing from a YAML configuration file.
The registry is the single source of truth for which models are available,
their capabilities, and their estimated pricing.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import yaml
from pydantic import BaseModel, Field

from app.core.config import get_settings
from app.core.exceptions import ConfigurationError, ModelNotFoundError
from app.core.logging import get_logger

logger = get_logger("registry")


class ModelInfo(BaseModel):
    """Static metadata for a single registered model."""

    model_id: str
    provider: str
    display_name: str
    litellm_model_name: str
    context_window: int
    input_price_per_1m_tokens: float
    output_price_per_1m_tokens: float
    capabilities: list[str] = Field(default_factory=list)
    enabled: bool = True


class ModelRegistry:
    """In-memory registry of available models, loaded from YAML config.

    Usage::

        registry = ModelRegistry.from_config()
        model = registry.get("gpt-4o-mini")
        enabled = registry.list_enabled()
    """

    def __init__(self, models: list[ModelInfo]) -> None:
        self._models: dict[str, ModelInfo] = {m.model_id: m for m in models}
        logger.info(
            "Model registry loaded: %d total, %d enabled",
            len(self._models),
            sum(1 for m in self._models.values() if m.enabled),
        )

    # ── Factory ───────────────────────────────────────────────────────────

    @classmethod
    def from_config(cls, path: Optional[Path] = None) -> ModelRegistry:
        """Load models from the YAML configuration file.

        Parameters
        ----------
        path:
            Override path to the YAML file. Defaults to ``Settings.models_config_file``.
        """
        config_path = path or get_settings().models_config_file

        if not config_path.exists():
            raise ConfigurationError(f"Model config not found: {config_path}")

        with open(config_path, "r") as f:
            raw = yaml.safe_load(f)

        if not raw or "models" not in raw:
            raise ConfigurationError(f"Invalid model config (missing 'models' key): {config_path}")

        models = [ModelInfo(**entry) for entry in raw["models"]]
        return cls(models)

    # ── Queries ───────────────────────────────────────────────────────────

    def get(self, model_id: str) -> ModelInfo:
        """Return model info by ID or raise ``ModelNotFoundError``."""
        if model_id not in self._models:
            raise ModelNotFoundError(model_id)
        return self._models[model_id]

    def list_all(self) -> list[ModelInfo]:
        """Return all registered models."""
        return list(self._models.values())

    def list_enabled(self) -> list[ModelInfo]:
        """Return only enabled models."""
        return [m for m in self._models.values() if m.enabled]

    def list_by_capability(self, capability: str) -> list[ModelInfo]:
        """Return enabled models that support the given capability."""
        return [
            m for m in self._models.values()
            if m.enabled and capability in m.capabilities
        ]

    def get_model_ids(self, enabled_only: bool = True) -> list[str]:
        """Return a list of model IDs."""
        models = self.list_enabled() if enabled_only else self.list_all()
        return [m.model_id for m in models]
