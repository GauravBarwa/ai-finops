"""Live verification across enabled models and all golden tasks; no credentials in output.

Run: python -m scripts.verify_providers
This makes 16 x 6 inference calls with a pause between task groups for free-tier quotas.
Use --publish-profiles to publish validated measurements into active model profiles.
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

from app.core.config import get_settings
from app.evaluation.metrics import score_output
from app.llm.provider_client import ProviderRouterClient
from app.llm.registry import ModelRegistry
from app.schemas.comparison import CompareRequest
from app.services.benchmark_service import BenchmarkService
from app.services.comparison_service import ComparisonService
from app.services.evaluation_service import EvaluationService
from app.services.profile_publication import publish_verification


async def verify(args):
    settings = get_settings()
    registry = ModelRegistry.from_config()
    service = ComparisonService(ProviderRouterClient(), registry)
    tasks = BenchmarkService.load_tasks()
    if args.task_ids:
        tasks = [t for t in tasks if t.task_id in args.task_ids]
        if not tasks:
            raise ValueError("No matching task IDs")
    report = {
        "started_at": datetime.now(timezone.utc).isoformat(),
        "registry_sha256": hashlib.sha256(settings.models_config_file.read_bytes()).hexdigest(),
        "enabled_models": registry.get_model_ids(),
        "disabled_models": {
            m.model_id: m.disabled_reason or "Disabled in registry"
            for m in registry.list_all()
            if not m.enabled
        },
        "generation_max_tokens": args.max_tokens,
        "comparisons": [],
    }
    output = args.output or settings.results_path / "provider_verification.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    for index, task in enumerate(tasks):
        result = await service.compare(
            CompareRequest(
                prompt=task.prompt,
                task_type=task.task_type,
                max_tokens=args.max_tokens,
            )
        )
        entry = result.model_dump(mode="json")
        entry["benchmark_task_id"] = task.task_id
        entry["quality_scores"] = {
            r.model_id: score_output(task.task_type, r.response, task.reference, prompt=task.prompt)
            if r.success
            else 0
            for r in result.results
        }
        report["comparisons"].append(entry)
        output.write_text(json.dumps(report, indent=2))
        print(
            f"[{index + 1}/{len(tasks)}] {task.task_id}: "
            f"{result.successful} succeeded, {result.failed} failed",
            flush=True,
        )
        for result in result.results:
            if not result.success:
                print(f"  {result.model_id}: {result.error}", flush=True)
        if index < len(tasks) - 1:
            await asyncio.sleep(args.pause_seconds)
    counts = defaultdict(lambda: {"succeeded": 0, "failed": 0})
    for comparison in report["comparisons"]:
        for result in comparison["results"]:
            counts[result["model_id"]]["succeeded" if result["success"] else "failed"] += 1
    report["finished_at"] = datetime.now(timezone.utc).isoformat()
    report["model_counts"] = dict(counts)
    report["total_succeeded"] = sum(c["succeeded"] for c in counts.values())
    report["total_failed"] = sum(c["failed"] for c in counts.values())
    output.write_text(json.dumps(report, indent=2))
    print(json.dumps(report["model_counts"], indent=2), flush=True)
    print(f"Proof saved to {output}", flush=True)
    if args.publish_profiles:
        profiles = publish_verification(output, EvaluationService(registry=registry))
        print(f"Published {len(profiles.profiles)} active profiles", flush=True)
    return 1 if report["total_failed"] else 0


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--publish-profiles", action="store_true")
    parser.add_argument("--task-ids", nargs="+", default=None)
    parser.add_argument("--max-tokens", type=int, default=1024)
    parser.add_argument("--pause-seconds", type=float, default=15)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()
    if not 1 <= args.max_tokens <= 8192 or args.pause_seconds < 0:
        parser.error("max-tokens must be 1..8192; pause-seconds must be nonnegative")
    raise SystemExit(asyncio.run(verify(args)))


if __name__ == "__main__":
    main()
