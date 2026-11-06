"""Evaluate saved benchmarks and atomically persist reproducible model profiles."""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from collections import defaultdict
from pathlib import Path
from statistics import mean

from app.core.config import get_settings
from app.core.exceptions import ModelNotFoundError
from app.evaluation.metrics import SCORER_VERSION, score_output
from app.llm.cost import calculate_cost
from app.llm.registry import ModelRegistry
from app.schemas.profile import ModelProfile, ProfilesResponse
from app.schemas.telemetry import TelemetryRecord
from app.services.benchmark_service import BenchmarkService, BenchmarkTask
from app.services.statistics import latency_summary


class EvaluationService:
    def __init__(
        self,
        results_dir: Path | None = None,
        tasks_path: Path | None = None,
        registry: ModelRegistry | None = None,
    ):
        self.registry = registry
        settings = get_settings()
        self.results_dir = results_dir if results_dir is not None else settings.results_path
        self.tasks_path = tasks_path or settings.benchmark_path / "tasks.json"

    @property
    def profiles_path(self) -> Path:
        return self.results_dir / "model_profiles.json"

    def load_profiles(self) -> ProfilesResponse:
        if not self.profiles_path.exists():
            return ProfilesResponse()
        try:
            return ProfilesResponse.model_validate_json(self.profiles_path.read_text())
        except ValueError as exc:
            raise ValueError(
                "Saved profiles are invalid. Rebuild them with Evaluate saved benchmarks."
            ) from exc

    @staticmethod
    def aggregate(
        records: list[TelemetryRecord],
        tasks: list[BenchmarkTask],
    ) -> list[ModelProfile]:
        golden = {t.task_id: t for t in tasks}
        groups = defaultdict(list)
        for record in records:
            task = golden.get(record.task_id)
            if record.source != "benchmark":
                continue
            if task is None or record.task_type != task.task_type or record.prompt != task.prompt:
                raise ValueError(f"Record does not match golden dataset: {record.task_id}")
            if not task.reference:
                raise ValueError(f"Missing reference: {task.task_id}")
            if record.latency_ms < 0 or record.estimated_cost < 0:
                raise ValueError("Negative benchmark metrics")
            quality = (
                score_output(
                    task.task_type,
                    record.response,
                    task.reference,
                    prompt=task.prompt,
                )
                if record.success
                else 0.0
            )
            groups[(record.model_id, task.task_type)].append((record, quality))
        profiles = []
        for (model_id, task_type), samples in sorted(groups.items()):
            stats = latency_summary([r.latency_ms for r, _ in samples])
            profiles.append(
                ModelProfile(
                    model_id=model_id,
                    task_type=task_type,
                    quality_score=mean(q for _, q in samples),
                    median_latency_ms=stats["median_latency_ms"],
                    p95_latency_ms=stats["p95_latency_ms"],
                    avg_estimated_cost=mean(r.estimated_cost for r, _ in samples),
                    reliability_rate=mean(int(r.success) for r, _ in samples),
                    sample_count=len(samples),
                )
            )
        return profiles

    def load_runs(
        self,
        run_files: list[str] | None = None,
    ) -> tuple[list[TelemetryRecord], list[str]]:
        names = (
            run_files
            if run_files is not None
            else sorted(p.name for p in self.results_dir.glob("benchmark_*.jsonl"))
        )
        if not names:
            raise ValueError("No benchmark runs found; run scripts.run_benchmark first")
        records = []
        for name in sorted(set(names)):
            if (
                Path(name).name != name
                or not name.startswith("benchmark_")
                or not name.endswith(".jsonl")
            ):
                raise ValueError("run_files must be benchmark_*.jsonl basenames")
            path = self.results_dir / name
            if path.is_symlink() or not path.is_file():
                raise ValueError(f"Benchmark file unavailable: {name}")
            records.extend(
                TelemetryRecord.model_validate_json(line)
                for line in path.read_text().splitlines()
                if line.strip()
            )
        return records, sorted(set(names))

    def evaluate(self, run_files: list[str] | None = None) -> ProfilesResponse:
        records, names = self.load_runs(run_files)
        if self.registry is not None:
            records = reprice_records(records, self.registry)
        tasks = BenchmarkService.load_tasks(self.tasks_path)
        profiles = self.aggregate(records, tasks)
        if not profiles:
            raise ValueError("No benchmark records to evaluate")
        result = ProfilesResponse(
            profiles=profiles,
            source_files=names,
            scorer_version=SCORER_VERSION,
            pricing_basis="shadow_commercial_usd" if self.registry else "recorded_estimates",
            pricing_hash=(
                hashlib.sha256(
                    json.dumps(
                        [m.model_dump() for m in self.registry.list_all()], sort_keys=True
                    ).encode()
                ).hexdigest()
                if self.registry
                else ""
            ),
            dataset_hash=hashlib.sha256(self.tasks_path.read_bytes()).hexdigest(),
        )
        self.results_dir.mkdir(parents=True, exist_ok=True)
        fd, temporary = tempfile.mkstemp(dir=self.results_dir, suffix=".tmp")
        try:
            with os.fdopen(fd, "w") as stream:
                stream.write(result.model_dump_json(indent=2))
            os.replace(temporary, self.profiles_path)
        finally:
            Path(temporary).unlink(missing_ok=True)
        return result


def reprice_records(
    records: list[TelemetryRecord],
    registry: ModelRegistry,
) -> list[TelemetryRecord]:
    """Apply current shadow assumptions to saved tokens without changing raw logs."""
    repriced = []
    for record in records:
        try:
            model = registry.get(record.model_id)
        except ModelNotFoundError as exc:
            raise ValueError(
                f"Benchmark references an unregistered model: {record.model_id}. "
                "Select compatible run files or restore its registry metadata."
            ) from exc
        repriced.append(
            record.model_copy(
                update={
                    "estimated_cost": calculate_cost(
                        record.input_tokens,
                        record.output_tokens,
                        model,
                    ).total_cost
                }
            )
        )
    return repriced
