"""Publish validated live verification measurements into the runtime benchmark/profile workflow."""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from pathlib import Path

from app.schemas.comparison import CompareResponse
from app.schemas.profile import ProfilesResponse
from app.schemas.telemetry import TelemetryRecord
from app.services.benchmark_service import BenchmarkService
from app.services.evaluation_service import EvaluationService


def publish_verification(report_path: Path, evaluation: EvaluationService) -> ProfilesResponse:
    """Import raw outcomes, not precomputed scores; retain failures and avoid duplicate imports.

    Publishing is explicit. Re-evaluation then sees the same measurements as the live router.
    Dataset identity is checked before writing. No inference calls are made here.
    """
    raw_text = report_path.read_text()
    report = json.loads(raw_text)
    if not report.get("finished_at") or not report.get("comparisons"):
        raise ValueError("Only a completed verification report can be published")
    records = []
    seen = set()
    for entry in report["comparisons"]:
        response = CompareResponse.model_validate(entry)
        task_id = entry.get("benchmark_task_id")
        if not task_id or response.total_models != len(response.results):
            raise ValueError("Verification contains incomplete task results")
        for result in response.results:
            key = (task_id, result.model_id)
            if key in seen:
                raise ValueError("Duplicate task/model measurements in verification report")
            seen.add(key)
            if evaluation.registry is not None:
                evaluation.registry.get(result.model_id)
            records.append(
                TelemetryRecord(
                    task_id=task_id,
                    task_type=response.task_type,
                    prompt=response.prompt,
                    timestamp=response.timestamp,
                    source="benchmark",
                    **result.model_dump(exclude={"display_name"}),
                )
            )
    # Reject mismatched/edited task data and invalid measurements before publishing.
    evaluation.aggregate(records, BenchmarkService.load_tasks(evaluation.tasks_path))
    digest = hashlib.sha256(raw_text.encode()).hexdigest()[:16]
    output = evaluation.results_dir / f"benchmark_verified_{digest}.jsonl"
    evaluation.results_dir.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(dir=evaluation.results_dir, suffix=".tmp")
    try:
        with os.fdopen(fd, "w") as stream:
            for record in records:
                stream.write(record.model_dump_json() + "\n")
        os.replace(temporary, output)
    finally:
        Path(temporary).unlink(missing_ok=True)
    # Use the normal source selection, so UI re-evaluation cannot lose cloud coverage.
    return evaluation.evaluate()
