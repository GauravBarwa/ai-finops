"""Native async Ollama implementation of the common LLM client contract.

The project uses Ollama directly for the local prototype. This keeps the
prototype independent of HCL's unavailable gateway and of paid provider keys,
while leaving the rest of the application provider-agnostic.
"""

from __future__ import annotations

import time
from typing import Any, Optional

import httpx

from app.core.config import get_settings
from app.core.logging import get_logger
from app.llm.interface import LLMClient, LLMResponse

logger = get_logger("ollama_client")


class OllamaClient(LLMClient):
    """Call a local or Ollama-hosted model through Ollama's HTTP API."""

    def __init__(
        self,
        base_url: Optional[str] = None,
        default_timeout: Optional[float] = None,
    ) -> None:
        settings = get_settings()
        self._base_url = (base_url or settings.ollama_base_url).rstrip("/")
        self._default_timeout = default_timeout or settings.ollama_timeout_seconds

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
        """Generate one response and normalize Ollama usage metadata."""
        started = time.perf_counter()
        model_name = self._normalise_model_name(model)
        request_timeout = timeout if timeout != 60.0 else self._default_timeout

        payload = {
            "model": model_name,
            "messages": [{"role": "user", "content": prompt}],
            "stream": False,
            "think": False,
            "options": {
                "num_predict": max_tokens,
                "temperature": temperature,
            },
        }
        payload.update(kwargs)

        try:
            async with httpx.AsyncClient(timeout=request_timeout) as client:
                response = await client.post(f"{self._base_url}/api/chat", json=payload)
                response.raise_for_status()
                data = response.json()

            elapsed_ms = (time.perf_counter() - started) * 1000
            message = data.get("message") or {}
            usage = data.get("usage") or {}
            input_tokens = int(
                data.get("prompt_eval_count")
                or usage.get("prompt_tokens")
                or 0
            )
            output_tokens = int(
                data.get("eval_count")
                or usage.get("completion_tokens")
                or 0
            )
            content = message.get("content") or ""

            logger.debug(
                "OK model=%s tokens=%d+%d latency=%.0fms",
                model_name,
                input_tokens,
                output_tokens,
                elapsed_ms,
            )
            return LLMResponse(
                content=content,
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                total_tokens=input_tokens + output_tokens,
                latency_ms=round(elapsed_ms, 2),
                model=model_name,
                provider="ollama",
                success=True,
                raw_response=data,
            )

        except Exception as exc:
            elapsed_ms = (time.perf_counter() - started) * 1000
            error_msg = f"{type(exc).__name__}: {exc}"
            logger.warning(
                "FAIL model=%s error=%s latency=%.0fms",
                model_name,
                error_msg,
                elapsed_ms,
            )
            return LLMResponse(
                model=model_name,
                provider="ollama",
                latency_ms=round(elapsed_ms, 2),
                success=False,
                error=error_msg,
            )

    @staticmethod
    def _normalise_model_name(model: str) -> str:
        """Accept registry values such as ``ollama/llama3.2`` and raw names."""
        return model.removeprefix("ollama/")
