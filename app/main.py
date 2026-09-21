"""FastAPI application factory and startup configuration."""

from __future__ import annotations

from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api.routes import init_services, router
from app.core.config import get_settings
from app.core.logging import setup_logging
from app.llm.registry import ModelRegistry


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application startup / shutdown lifecycle."""
    # ── Startup ──────────────────────────────────────────────────────────
    logger = setup_logging()
    settings = get_settings()

    logger.info("Starting AI FinOps service  env=%s", settings.app_env)

    # Load model registry and initialise services
    registry = ModelRegistry.from_config()
    init_services(registry)

    logger.info("Service ready — %d models loaded", len(registry.list_enabled()))

    yield

    # ── Shutdown ─────────────────────────────────────────────────────────
    logger.info("Shutting down AI FinOps service")


def create_app() -> FastAPI:
    """Build and return the FastAPI application instance."""
    app = FastAPI(
        title="AI FinOps",
        description=(
            "Intelligent LLM Model Selection & Cost Optimization — "
            "compare multiple models on the same workload and capture "
            "quality, cost, latency, and reliability metrics."
        ),
        version="0.1.0",
        lifespan=lifespan,
    )

    # ── CORS (permissive for local dev) ──────────────────────────────────
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # ── Routes ───────────────────────────────────────────────────────────
    app.include_router(router)

    return app


# ── Uvicorn entry point ──────────────────────────────────────────────────────
app = create_app()
