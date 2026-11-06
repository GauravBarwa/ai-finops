"""Tests for the native Ollama adapter and provider routing."""

from __future__ import annotations

from unittest.mock import AsyncMock

import pytest

import app.llm.ollama_client as ollama_module
from app.llm.interface import LLMResponse
from app.llm.ollama_client import OllamaClient
from app.llm.provider_client import ProviderRouterClient


class FakeResponse:
    def __init__(self, payload: dict, error: bool = False) -> None:
        self._payload = payload
        self._error = error

    def raise_for_status(self) -> None:
        if self._error:
            raise RuntimeError("Ollama returned an error")

    def json(self) -> dict:
        return self._payload


class FakeAsyncClient:
    def __init__(self, response: FakeResponse) -> None:
        self.response = response
        self.request: dict | None = None

    async def __aenter__(self):
        return self

    async def __aexit__(self, *args):
        return None

    async def post(self, url: str, json: dict):
        self.request = {"url": url, "json": json}
        return self.response


@pytest.mark.asyncio
async def test_ollama_adapter_normalises_response_and_usage(monkeypatch):
    fake_client = FakeAsyncClient(FakeResponse({
        "message": {"role": "assistant", "content": "hello"},
        "prompt_eval_count": 12,
        "eval_count": 7,
    }))
    monkeypatch.setattr(ollama_module.httpx, "AsyncClient", lambda **kwargs: fake_client)

    response = await OllamaClient(base_url="http://ollama.test").generate(
        "Say hello", "ollama/llama3.2:latest", max_tokens=32,
    )

    assert response.success is True
    assert response.content == "hello"
    assert response.input_tokens == 12
    assert response.output_tokens == 7
    assert response.total_tokens == 19
    assert response.latency_ms >= 0
    assert fake_client.request["url"] == "http://ollama.test/api/chat"
    assert fake_client.request["json"]["model"] == "llama3.2:latest"


@pytest.mark.asyncio
async def test_ollama_adapter_returns_failure_telemetry(monkeypatch):
    fake_client = FakeAsyncClient(FakeResponse({}, error=True))
    monkeypatch.setattr(ollama_module.httpx, "AsyncClient", lambda **kwargs: fake_client)

    response = await OllamaClient().generate("Say hello", "ollama/missing")

    assert response.success is False
    assert response.error


@pytest.mark.asyncio
async def test_provider_router_uses_ollama_for_ollama_models():
    ollama = AsyncMock()
    litellm = AsyncMock()
    ollama.generate.return_value = LLMResponse(content="local", model="local")
    litellm.generate.return_value = LLMResponse(content="hosted", model="hosted")
    client = ProviderRouterClient(ollama_client=ollama, litellm_client=litellm)

    local = await client.generate("prompt", "ollama/llama3.2:latest")
    hosted = await client.generate("prompt", "gpt-4o-mini")

    assert local.content == "local"
    assert hosted.content == "hosted"
    ollama.generate.assert_awaited_once()
    litellm.generate.assert_awaited_once()
