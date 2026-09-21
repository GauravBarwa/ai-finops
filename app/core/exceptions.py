"""Custom exception hierarchy for the AI FinOps application."""

from __future__ import annotations


class AIFinOpsError(Exception):
    """Base exception for all AI FinOps errors."""


class LLMProviderError(AIFinOpsError):
    """Raised when an LLM provider call fails (timeout, auth, rate-limit, etc.)."""

    def __init__(self, provider: str, model: str, message: str) -> None:
        self.provider = provider
        self.model = model
        super().__init__(f"[{provider}/{model}] {message}")


class ModelNotFoundError(AIFinOpsError):
    """Raised when a requested model ID does not exist in the registry."""

    def __init__(self, model_id: str) -> None:
        self.model_id = model_id
        super().__init__(f"Model not found in registry: {model_id}")


class BenchmarkError(AIFinOpsError):
    """Raised when a benchmark run encounters an unrecoverable error."""


class ConfigurationError(AIFinOpsError):
    """Raised when configuration is missing or invalid."""
