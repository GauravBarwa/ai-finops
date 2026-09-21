"""Comparison service — orchestrates running the same prompt across multiple models."""

from __future__ import annotations

import asyncio
import uuid
from datetime import datetime, timezone

from app.core.logging import get_logger
from app.llm.cost import calculate_cost
from app.llm.interface import LLMClient
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

    def __init__(self, llm_client: LLMClient, registry: ModelRegistry) -> None:
        self._client = llm_client
        self._registry = registry

    async def compare(self, request: CompareRequest) -> CompareResponse:
        """Execute the comparison for the given request.

        Runs all models concurrently via ``asyncio.gather`` and collects
        unified metrics for each.
        """
        task_id = str(uuid.uuid4())

        # Resolve target models
        if request.models:
            model_ids = request.models
        else:
            model_ids = self._registry.get_model_ids(enabled_only=True)

        logger.info(
            "Starting comparison  task_id=%s  models=%s  task_type=%s",
            task_id, model_ids, request.task_type,
        )

        # Run all models concurrently
        tasks = [
            self._run_single(
                model_id=mid,
                prompt=request.prompt,
                max_tokens=request.max_tokens,
                temperature=request.temperature,
            )
            for mid in model_ids
        ]
        results = await asyncio.gather(*tasks)

        successful = sum(1 for r in results if r.success)
        failed = len(results) - successful

        response = CompareResponse(
            task_id=task_id,
            prompt=request.prompt,
            task_type=request.task_type,
            timestamp=datetime.now(timezone.utc),
            results=results,
            total_models=len(results),
            successful=successful,
            failed=failed,
        )

        logger.info(
            "Comparison complete  task_id=%s  success=%d  failed=%d",
            task_id, successful, failed,
        )

        return response

    async def _run_single(
        self,
        model_id: str,
        prompt: str,
        max_tokens: int,
        temperature: float,
    ) -> ModelResult:
        """Call a single model and wrap the result in a ``ModelResult``."""
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

        llm_response = await self._client.generate(
            prompt=prompt,
            model=model_info.litellm_model_name,
            max_tokens=max_tokens,
            temperature=temperature,
        )

        # Calculate estimated cost
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
