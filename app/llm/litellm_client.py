"""LiteLLM-based implementation of the unified ``LLMClient`` interface.

LiteLLM provides a single ``completion()`` / ``acompletion()`` function that
translates to 100+ provider APIs, letting us switch models by changing a
string.
"""

from __future__ import annotations

import time
from typing import Any

import litellm

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
            model="gemini/gemini-2.0-flash",
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
            response = await litellm.acompletion(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=max_tokens,
                temperature=temperature,
                timeout=timeout,
                **kwargs,
            )
            elapsed_ms = (time.perf_counter() - start) * 1000

            # Extract token usage — LiteLLM normalizes this across providers
            usage = response.usage
            input_tokens = usage.prompt_tokens or 0
            output_tokens = usage.completion_tokens or 0

            content = response.choices[0].message.content or ""

            logger.debug(
                "OK  model=%s  tokens=%d+%d  latency=%.0fms",
                model, input_tokens, output_tokens, elapsed_ms,
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
            error_msg = f"{type(exc).__name__}: {exc}"

            logger.warning(
                "FAIL model=%s  error=%s  latency=%.0fms",
                model, error_msg, elapsed_ms,
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

        Examples: ``gemini/gemini-2.0-flash`` → ``google``,
                  ``gpt-4o-mini`` → ``openai``,
                  ``claude-3-haiku-20240307`` → ``anthropic``.
        """
        model_lower = model.lower()
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
