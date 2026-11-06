"""Benchmark service — runs a set of tasks across multiple models and records results."""

from __future__ import annotations

import json
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from app.core.config import get_settings
from app.core.exceptions import BenchmarkError
from app.core.logging import get_logger
from app.llm.cost import calculate_cost
from app.llm.interface import LLMClient, LLMResponse
from app.llm.registry import ModelRegistry
from app.schemas.telemetry import TelemetryRecord
from app.services.statistics import latency_summary

logger = get_logger("benchmark_service")


class BenchmarkTask:
    """A single benchmark task loaded from the dataset."""

    def __init__(
        self,
        task_id: str,
        task_type: str,
        prompt: str,
        reference: Optional[str] = None,
    ) -> None:
        self.task_id = task_id
        self.task_type = task_type
        self.prompt = prompt
        self.reference = reference


class BenchmarkService:
    """Load benchmark tasks and run them across a set of models.

    Parameters
    ----------
    llm_client:
        The unified LLM client.
    registry:
        The model registry.
    """

    def __init__(self, llm_client: LLMClient, registry: ModelRegistry) -> None:
        self._client = llm_client
        self._registry = registry

    @staticmethod
    def load_tasks(path: Optional[Path] = None) -> list[BenchmarkTask]:
        """Load benchmark tasks from a JSON file.

        Parameters
        ----------
        path:
            Path to the tasks JSON file.  Defaults to
            ``data/benchmark/tasks.json``.
        """
        if path is None:
            path = get_settings().benchmark_path / "tasks.json"

        if not path.exists():
            raise BenchmarkError(f"Benchmark tasks file not found: {path}")

        with open(path, "r") as f:
            raw = json.load(f)

        tasks = [
            BenchmarkTask(
                task_id=entry["task_id"],
                task_type=entry["task_type"],
                prompt=entry["prompt"],
                reference=entry.get("reference"),
            )
            for entry in raw
        ]

        logger.info("Loaded %d benchmark tasks from %s", len(tasks), path)
        return tasks

    async def run_benchmark(
        self,
        model_ids: Optional[list[str]] = None,
        tasks: Optional[list[BenchmarkTask]] = None,
        max_tokens: int = 1024,
        temperature: float = 0.0,
    ) -> list[TelemetryRecord]:
        """Run every task against every model and return telemetry records.

        Parameters
        ----------
        model_ids:
            Which models to benchmark.  Defaults to all enabled.
        tasks:
            Which tasks to run.  Defaults to loading from the standard file.
        max_tokens:
            Maximum tokens per call.
        temperature:
            Sampling temperature for all calls.

        Returns
        -------
        list[TelemetryRecord]
            One record per (task, model) combination.
        """
        if tasks is None:
            tasks = self.load_tasks()

        if model_ids is None:
            model_ids = self._registry.get_model_ids(enabled_only=True)

        total = len(tasks) * len(model_ids)
        logger.info(
            "Starting benchmark: %d tasks × %d models = %d calls",
            len(tasks),
            len(model_ids),
            total,
        )

        records: list[TelemetryRecord] = []
        completed = 0

        for task in tasks:
            for model_id in model_ids:
                model_info = self._registry.get(model_id)

                started = time.perf_counter()
                try:
                    llm_response = await self._client.generate(
                        prompt=task.prompt,
                        model=model_info.litellm_model_name,
                        max_tokens=max_tokens,
                        temperature=temperature,
                        **model_info.generation_options,
                    )
                except Exception as exc:
                    llm_response = LLMResponse(
                        success=False,
                        error=f"{type(exc).__name__}: {exc}",
                        latency_ms=(time.perf_counter() - started) * 1000,
                    )

                cost = calculate_cost(
                    input_tokens=llm_response.input_tokens,
                    output_tokens=llm_response.output_tokens,
                    model_info=model_info,
                )

                record = TelemetryRecord(
                    task_id=task.task_id,
                    model_id=model_id,
                    provider=model_info.provider,
                    task_type=task.task_type,
                    prompt=task.prompt,
                    response=llm_response.content,
                    success=llm_response.success,
                    error=llm_response.error,
                    input_tokens=llm_response.input_tokens,
                    output_tokens=llm_response.output_tokens,
                    total_tokens=llm_response.total_tokens,
                    latency_ms=llm_response.latency_ms,
                    estimated_cost=cost.total_cost,
                    source="benchmark",
                )
                records.append(record)

                completed += 1
                logger.info(
                    "  [%d/%d] task=%s model=%s %s latency=%.0fms tokens=%d cost=$%.6f",
                    completed,
                    total,
                    task.task_id,
                    model_id,
                    "OK" if llm_response.success else "FAIL",
                    llm_response.latency_ms,
                    llm_response.total_tokens,
                    cost.total_cost,
                )

        for model_id in model_ids:
            samples = [r for r in records if r.model_id == model_id]
            stats = latency_summary([r.latency_ms for r in samples])
            logger.info(
                "%s avg=%.0fms p50=%.0fms p95=%.0fms total_estimated_cost=$%.8f",
                model_id,
                stats["avg_latency_ms"],
                stats["median_latency_ms"],
                stats["p95_latency_ms"],
                sum(r.estimated_cost for r in samples),
            )
        logger.info("Benchmark complete: %d records", len(records))
        return records

    @staticmethod
    def save_results(
        records: list[TelemetryRecord],
        output_path: Optional[Path] = None,
    ) -> Path:
        """Persist benchmark results as a JSONL file.

        Parameters
        ----------
        records:
            The telemetry records to save.
        output_path:
            Override output file path.  Defaults to
            ``data/results/benchmark_{timestamp}.jsonl``.

        Returns
        -------
        Path
            The path to the saved file.
        """
        if output_path is None:
            ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S_%f")
            output_path = get_settings().results_path / f"benchmark_{ts}.jsonl"

        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, "w") as f:
            for record in records:
                f.write(record.model_dump_json() + "\n")

        logger.info("Saved %d benchmark records to %s", len(records), output_path)
        return output_path
