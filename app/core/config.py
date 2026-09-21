"""Application configuration loaded from environment variables and config files."""

from __future__ import annotations

import os
from pathlib import Path
from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


# ── Project root ──────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent


class Settings(BaseSettings):
    """Central application settings.

    Values are loaded from environment variables and/or a `.env` file located
    at the project root.  Environment variables always take precedence.
    """

    model_config = SettingsConfigDict(
        env_file=str(PROJECT_ROOT / ".env"),
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # ── LLM provider keys ────────────────────────────────────────────────────
    openai_api_key: str = ""
    gemini_api_key: str = ""
    anthropic_api_key: str = ""

    # ── Application ──────────────────────────────────────────────────────────
    app_env: str = "development"
    log_level: str = "INFO"

    # ── Data paths (relative to PROJECT_ROOT) ────────────────────────────────
    data_dir: str = "data"
    results_dir: str = "data/results"
    benchmark_dir: str = "data/benchmark"
    models_config_path: str = "data/models.yaml"

    # ── Server ───────────────────────────────────────────────────────────────
    host: str = "0.0.0.0"
    port: int = 8000

    # ── Derived helpers ──────────────────────────────────────────────────────
    @property
    def data_path(self) -> Path:
        return PROJECT_ROOT / self.data_dir

    @property
    def results_path(self) -> Path:
        return PROJECT_ROOT / self.results_dir

    @property
    def benchmark_path(self) -> Path:
        return PROJECT_ROOT / self.benchmark_dir

    @property
    def models_config_file(self) -> Path:
        return PROJECT_ROOT / self.models_config_path

    def ensure_directories(self) -> None:
        """Create required data directories if they don't exist."""
        self.results_path.mkdir(parents=True, exist_ok=True)
        self.benchmark_path.mkdir(parents=True, exist_ok=True)


@lru_cache
def get_settings() -> Settings:
    """Return a cached singleton of the application settings."""
    settings = Settings()
    settings.ensure_directories()

    # Push provider keys into the environment so LiteLLM can pick them up
    # automatically. Only set if a non-empty value was provided.
    _env_map = {
        "OPENAI_API_KEY": settings.openai_api_key,
        "GEMINI_API_KEY": settings.gemini_api_key,
        "ANTHROPIC_API_KEY": settings.anthropic_api_key,
    }
    for key, value in _env_map.items():
        if value:
            os.environ.setdefault(key, value)

    return settings
