"""Comparison service — orchestrates running the same prompt across multiple models."""

from __future__ import annotations

import asyncio
import time
import uuid
from collections.abc import AsyncIterator
from datetime import datetime, timezone

from app.core.config import get_settings
from app.core.logging import get_logger
from app.llm.cost import calculate_cost
from app.llm.interface import LLMClient
from app.llm.litellm_client import safe_error
from app.llm.registry import ModelRegistry
from app.schemas.comparison import CompareRequest, CompareResponse, ModelResult

logger = get_logger("comparison_service")


class ComparisonService:
    """Run one prompt against multiple models and aggregate the results.

    Parameters
    ----------
    llm_client:
        The unified LLM client (e.g. ``LiteLLMClient``).
    registry:
        The model registry for metadata/pricing lookups.
    """

    def __init__(
        self,
        llm_client: LLMClient,
        registry: ModelRegistry,
        timeout_seconds: float | None = None,
    ) -> None:
        self._client = llm_client
        self._registry = registry
        self._timeout = (
            timeout_seconds
            if timeout_seconds is not None
            else get_settings().comparison_timeout_seconds
        )

    async def compare(self, request: CompareRequest) -> CompareResponse:
        """Return the final snapshot; UI callers can consume stream_compare directly."""
        async for snapshot in self.stream_compare(request):
            final = snapshot
        return final

    async def stream_compare(self, request: CompareRequest) -> AsyncIterator[CompareResponse]:
        """Yield progress immediately and each completed model without waiting for the slowest.

        Every snapshot is independent, keeps registry/request order, and shares a task ID.
        Cancelled consumers cancel outstanding provider tasks as well.
        """
        task_id = str(uuid.uuid4())
        timestamp = datetime.now(timezone.utc)
        model_ids = list(
            dict.fromkeys(request.models or self._registry.get_model_ids(enabled_only=True))
        )
        completed: dict[str, ModelResult] = {}

        def snapshot():
            results = [completed[mid] for mid in model_ids if mid in completed]
            return CompareResponse(
                task_id=task_id,
                prompt=request.prompt,
                task_type=request.task_type,
                timestamp=timestamp,
                results=results,
                total_models=len(model_ids),
                successful=sum(r.success for r in results),
                failed=sum(not r.success for r in results),
            )

        yield snapshot()
        pending = [
            asyncio.create_task(
                self._run_single(
                    model_id=mid,
                    prompt=request.prompt,
                    max_tokens=request.max_tokens,
                    temperature=request.temperature,
                )
            )
            for mid in model_ids
        ]
        try:
            for task in asyncio.as_completed(pending):
                result = await task
                completed[result.model_id] = result
                yield snapshot()
        finally:
            for task in pending:
                if not task.done():
                    task.cancel()
            await asyncio.gather(*pending, return_exceptions=True)

    async def _run_single(
        self,
        model_id: str,
        prompt: str,
        max_tokens: int,
        temperature: float,
    ) -> ModelResult:
        """Call a single model and wrap the result in a ``ModelResult``.

        Provider failures are converted into a failed result so comparison of
        the remaining models can still complete.
        """
        try:
            model_info = self._registry.get(model_id)
        except Exception as exc:
            return ModelResult(
                model_id=model_id,
                provider="unknown",
                display_name=model_id,
                success=False,
                error=str(exc),
            )

        if not model_info.enabled:
            return ModelResult(
                model_id=model_id,
                provider=model_info.provider,
                display_name=model_info.display_name,
                success=False,
                error=model_info.disabled_reason or "Model is disabled in the registry",
            )

        started = time.perf_counter()
        try:
            llm_response = await asyncio.wait_for(
                self._client.generate(
                    prompt=prompt,
                    model=model_info.litellm_model_name,
                    max_tokens=max_tokens,
                    temperature=temperature,
                    timeout=self._timeout,
                    **model_info.generation_options,
                ),
                timeout=self._timeout,
            )
        except TimeoutError:
            return ModelResult(
                model_id=model_id,
                provider=model_info.provider,
                display_name=model_info.display_name,
                success=False,
                latency_ms=(time.perf_counter() - started) * 1000,
                error=(
                    f"Request timed out after {self._timeout:g}s; other models can still complete"
                ),
            )
        except Exception as exc:
            logger.warning("Provider exception for model=%s: %s", model_id, safe_error(exc))
            return ModelResult(
                model_id=model_id,
                provider=model_info.provider,
                display_name=model_info.display_name,
                success=False,
                latency_ms=(time.perf_counter() - started) * 1000,
                error=safe_error(exc),
            )

        cost = calculate_cost(
            input_tokens=llm_response.input_tokens,
            output_tokens=llm_response.output_tokens,
            model_info=model_info,
        )

        return ModelResult(
            model_id=model_id,
            provider=model_info.provider,
            display_name=model_info.display_name,
            response=llm_response.content,
            input_tokens=llm_response.input_tokens,
            output_tokens=llm_response.output_tokens,
            total_tokens=llm_response.total_tokens,
            latency_ms=llm_response.latency_ms,
            estimated_cost=cost.total_cost,
            success=llm_response.success,
            error=llm_response.error,
        )
