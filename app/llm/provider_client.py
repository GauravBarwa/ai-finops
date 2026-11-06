"""Provider-aware composition of native and optional provider adapters."""

from __future__ import annotations

from typing import Any, Optional

from app.llm.interface import LLMClient, LLMResponse
from app.llm.litellm_client import LiteLLMClient
from app.llm.ollama_client import OllamaClient


class ProviderRouterClient(LLMClient):
    """Route model calls to the adapter indicated by the model name.

    Ollama model strings use the native httpx adapter. Other provider strings
    use LiteLLM. Comparison, benchmarking, and routing share the normalized
    LLMClient contract across the configured providers.
    """

    def __init__(
        self,
        ollama_client: Optional[LLMClient] = None,
        litellm_client: Optional[LLMClient] = None,
    ) -> None:
        self._ollama = ollama_client or OllamaClient()
        self._litellm = litellm_client or LiteLLMClient()

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
        """Delegate to native Ollama or LiteLLM without changing the contract."""
        client = self._ollama if model.lower().startswith("ollama/") else self._litellm
        return await client.generate(
            prompt=prompt,
            model=model,
            max_tokens=max_tokens,
            temperature=temperature,
            timeout=timeout,
            **kwargs,
        )
