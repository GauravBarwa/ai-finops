"""Structured logging setup for the AI FinOps application."""

from __future__ import annotations

import logging
import sys
from typing import Optional

from app.core.config import get_settings


def setup_logging(level: Optional[str] = None) -> logging.Logger:
    """Configure and return the application root logger.

    Parameters
    ----------
    level:
        Override log level.  Falls back to ``Settings.log_level``.

    Returns
    -------
    logging.Logger
        The configured ``ai_finops`` logger.
    """
    log_level = (level or get_settings().log_level).upper()

    # Structured-ish format — good enough for local dev, parseable in prod
    formatter = logging.Formatter(
        fmt="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(formatter)

    logger = logging.getLogger("ai_finops")
    logger.setLevel(log_level)

    # Avoid duplicate handlers on repeated calls
    if not logger.handlers:
        logger.addHandler(handler)

    # Quiet noisy third-party loggers
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("litellm").setLevel(logging.WARNING)
    logging.getLogger("httpcore").setLevel(logging.WARNING)

    return logger


def get_logger(name: str) -> logging.Logger:
    """Return a child logger under the ``ai_finops`` namespace."""
    return logging.getLogger(f"ai_finops.{name}")
