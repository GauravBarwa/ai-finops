"""LiteLLM-based implementation of the unified ``LLMClient`` interface.

LiteLLM provides a single ``completion()`` / ``acompletion()`` function that
translates to 100+ provider APIs, letting us switch models by changing a
string.
"""

from __future__ import annotations

import os
import time
from types import SimpleNamespace
from typing import Any

# Registry pricing is authoritative; avoid an unrelated network fetch during import.
os.environ.setdefault("LITELLM_LOCAL_MODEL_COST_MAP", "True")

import litellm  # noqa: E402

from app.core.config import get_settings
from app.core.logging import get_logger
from app.llm.interface import LLMClient, LLMResponse

logger = get_logger("litellm_client")

# Suppress LiteLLM's default verbose logging
litellm.suppress_debug_info = True


class LiteLLMClient(LLMClient):
    """Concrete ``LLMClient`` backed by LiteLLM.

    Usage::

        client = LiteLLMClient()
        response = await client.generate(
            prompt="Summarize this text: ...",
            model="gemini/gemini-3.6-flash",
        )
    """

    async def generate(
        self,
        prompt: str,
        model: str,
        *,
        max_tokens: int = 1024,
        temperature: float = 0.0,
        timeout: float = 60.0,
        **kwargs: Any,
    ) -> LLMResponse:
        """Call *model* via LiteLLM and return a unified ``LLMResponse``."""
        start = time.perf_counter()

        try:
            provider = self._extract_provider(model)
            settings = get_settings()
            if provider == "github":
                raise ValueError("GitHub Models retired on 2026-07-30; select another provider")
            key_field = {"google": "gemini_api_key", "groq": "groq_api_key"}.get(provider)
            if key_field and getattr(settings, key_field).strip() in {"", "your-key"}:
                raise ValueError(f"Missing {provider} API key; configure it in your local .env")
            options = {"num_retries": 1, **kwargs}
            # Gemini 3 deprecates sampling controls. Keep them for other providers.
            sampling = (
                {}
                if model.lower().startswith("gemini/gemini-3")
                else {
                    "temperature": temperature,
                }
            )
            response = await litellm.acompletion(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=max_tokens,
                timeout=timeout,
                **sampling,
                **options,
            )
            elapsed_ms = (time.perf_counter() - start) * 1000

            # Extract token usage — LiteLLM normalizes this across providers
            usage = response.usage or SimpleNamespace()
            input_tokens = getattr(usage, "prompt_tokens", 0) or 0
            output_tokens = getattr(usage, "completion_tokens", 0) or 0

            content = response.choices[0].message.content or ""

            logger.debug(
                "OK  model=%s  tokens=%d+%d  latency=%.0fms",
                model,
                input_tokens,
                output_tokens,
                elapsed_ms,
            )

            if not content.strip():
                return LLMResponse(
                    success=False,
                    error=(
                        "Model returned no answer text; "
                        "increase max_tokens or reduce reasoning effort"
                    ),
                    input_tokens=input_tokens,
                    output_tokens=output_tokens,
                    latency_ms=round(elapsed_ms, 2),
                    model=model,
                    provider=provider,
                )
            return LLMResponse(
                content=content,
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                total_tokens=input_tokens + output_tokens,
                latency_ms=round(elapsed_ms, 2),
                model=model,
                provider=self._extract_provider(model),
                success=True,
                error=None,
                raw_response=response,
            )

        except Exception as exc:
            elapsed_ms = (time.perf_counter() - start) * 1000
            error_msg = safe_error(exc)

            logger.warning(
                "FAIL model=%s  error=%s  latency=%.0fms",
                model,
                error_msg,
                elapsed_ms,
            )

            return LLMResponse(
                content="",
                input_tokens=0,
                output_tokens=0,
                total_tokens=0,
                latency_ms=round(elapsed_ms, 2),
                model=model,
                provider=self._extract_provider(model),
                success=False,
                error=error_msg,
            )

    @staticmethod
    def _extract_provider(model: str) -> str:
        """Derive provider name from a LiteLLM model string.

        Examples: ``gemini/gemini-3.6-flash`` → ``google``,
                  ``gpt-4o-mini`` → ``openai``,
                  ``claude-3-haiku-20240307`` → ``anthropic``.
        """
        model_lower = model.lower()
        for provider in ("groq", "github"):
            if model_lower.startswith(provider + "/"):
                return provider
        if model_lower.startswith("gemini/"):
            return "google"
        if model_lower.startswith("ollama/"):
            return "ollama"
        if "claude" in model_lower:
            return "anthropic"
        if "gpt" in model_lower or "o1" in model_lower or "o3" in model_lower:
            return "openai"
        # Fallback: use the prefix before '/' if present
        if "/" in model:
            return model.split("/")[0]
        return "unknown"


def safe_error(exc: Exception) -> str:
    """Never echo a configured credential in a provider error or log message."""
    message = f"{type(exc).__name__}: {exc}"
    settings = get_settings()
    for field in (
        "openai_api_key",
        "gemini_api_key",
        "groq_api_key",
        "github_api_key",
        "anthropic_api_key",
    ):
        value = getattr(settings, field)
        if value and len(value) >= 6:
            message = message.replace(value, "[REDACTED]")
    return message
