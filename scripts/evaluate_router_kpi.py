"""Offline, leave-one-task-out replay using saved responses and current shadow prices.

python -m scripts.evaluate_router_kpi --policy cost_first --min-quality 0.5
No live model calls. Skips and coverage are explicit; no fabricated outcomes.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path
from statistics import mean, median

from app.domain.routing import recommend
from app.evaluation.metrics import score_output
from app.llm.registry import ModelRegistry
from app.schemas.routing import Constraints, RouteRequest
from app.schemas.telemetry import TelemetryRecord
from app.services.benchmark_service import BenchmarkService, BenchmarkTask
from app.services.evaluation_service import EvaluationService, reprice_records


def simulate(
    records: list[TelemetryRecord],
    tasks: list[BenchmarkTask],
    registry: ModelRegistry,
    baseline: str,
    policy: str = "cost_first",
    constraints: Constraints | None = None,
) -> dict:
    """Each held-out task is routed from profiles of every other task.

    Costs, quality and latency are paired over the exact same held-out tasks.
    Multiple trials for a task/model are averaged, including failed trials.
    There is no simulated live fallback because retry outcomes are not observed.
    """
    registry.get(baseline)
    records = reprice_records(records, registry)
    # Validate dataset identity before constructing any folds.
    EvaluationService.aggregate(records, tasks)
    by_task = defaultdict(list)
    for record in records:
        if record.source == "benchmark":
            by_task[record.task_id].append(record)
    golden = {t.task_id: t for t in tasks}
    pairs, skipped = [], []
    for task_id, samples in sorted(by_task.items()):
        task = golden[task_id]
        training = [r for r in records if r.task_id != task_id]
        profiles = EvaluationService.aggregate(training, tasks)
        decision = recommend(
            RouteRequest(
                prompt=task.prompt,
                task_type=task.task_type,
                routing_mode="task_average",
                policy=policy,
                constraints=constraints or Constraints(),
            ),
            registry.list_all(),
            profiles,
        )
        selected = decision.selected_model_id
        baseline_samples = [r for r in samples if r.model_id == baseline]
        routed_samples = [r for r in samples if r.model_id == selected]
        if not baseline_samples or not routed_samples:
            skipped.append(task_id)
            continue

        def metrics(group):
            return {
                "cost": mean(r.estimated_cost for r in group),
                "latency": mean(r.latency_ms for r in group),
                "quality": mean(
                    score_output(task.task_type, r.response, task.reference, prompt=task.prompt)
                    if r.success
                    else 0
                    for r in group
                ),
                "reliability": mean(int(r.success) for r in group),
            }

        pairs.append(
            {
                "task_id": task_id,
                "selected": selected,
                "baseline": metrics(baseline_samples),
                "router": metrics(routed_samples),
            }
        )
    result = {
        "constraints": (constraints or Constraints()).model_dump(),
        "paired_tasks": len(pairs),
        "observed_tasks": len(by_task),
        "dataset_tasks": len(tasks),
        "skipped_tasks": skipped,
        "pairs": pairs,
    }
    if not pairs:
        return result
    base_cost = sum(p["baseline"]["cost"] for p in pairs)
    route_cost = sum(p["router"]["cost"] for p in pairs)
    base_quality = mean(p["baseline"]["quality"] for p in pairs)
    route_quality = mean(p["router"]["quality"] for p in pairs)
    base_latency = median(p["baseline"]["latency"] for p in pairs)
    route_latency = median(p["router"]["latency"] for p in pairs)
    result.update(
        baseline_estimated_cost=base_cost,
        router_estimated_cost=route_cost,
        cost_reduction_pct=100 * (1 - route_cost / base_cost) if base_cost else None,
        quality_retained_pct=100 * route_quality / base_quality if base_quality else None,
        baseline_quality=base_quality,
        router_quality=route_quality,
        baseline_median_latency_ms=base_latency,
        router_median_latency_ms=route_latency,
        latency_change_pct=100 * (route_latency / base_latency - 1) if base_latency else None,
        baseline_reliability=mean(p["baseline"]["reliability"] for p in pairs),
        router_reliability=mean(p["router"]["reliability"] for p in pairs),
    )
    return result


def write_report(
    result: dict, output: Path, baseline: str, policy: str, sources: list[str]
) -> None:
    lines = [
        "# AI FinOps — Offline Router KPI Replay",
        "",
        f"Baseline: `{baseline}`. Policy: `{policy}`.",
        f"Constraints: `{result['constraints']}`.",
        "",
        "Costs use Method A shadow commercial USD pricing on recorded token counts. "
        "They do not represent billed spend. No provider requests were made.",
        "",
        "Method: leave one task out; build routing profiles from other tasks, then "
        "compare saved baseline and selected-model outcomes on the held-out task. "
        "Failed trials contribute zero quality and remain in latency and reliability. "
        "Only paired tasks contribute to KPI denominators; no live fallback is simulated.",
        "",
        f"Paired tasks: {result['paired_tasks']} / {result['observed_tasks']} observed / "
        f"{result['dataset_tasks']} in the current dataset.",
        "",
        "Sources: " + ", ".join(f"`{s}`" for s in sources),
        "",
    ]
    if result["paired_tasks"]:
        for key, label in [
            ("cost_reduction_pct", "Estimated cost reduction"),
            ("quality_retained_pct", "Quality retained"),
            ("latency_change_pct", "Median latency change (negative is faster)"),
        ]:
            value = result[key]
            lines.append(
                f"- {label}: {value:.2f}%"
                if value is not None
                else f"- {label}: unavailable (zero baseline denominator)"
            )
        lines += [
            "",
            f"Baseline quality: {result['baseline_quality']:.3f}; "
            f"router quality: {result['router_quality']:.3f}.",
            f"Baseline estimated cost: ${result['baseline_estimated_cost']:.8f}; "
            f"router estimated cost: ${result['router_estimated_cost']:.8f}.",
            f"Baseline median latency: {result['baseline_median_latency_ms']:.1f}ms; "
            f"router median latency: {result['router_median_latency_ms']:.1f}ms.",
            f"Baseline reliability: {result['baseline_reliability']:.3f}; "
            f"router reliability: {result['router_reliability']:.3f}.",
        ]
    else:
        lines += [
            "No paired evaluation is possible. Collect more tasks per task type "
            "for both the baseline and candidate models before reporting KPIs."
        ]
    lines += [
        "",
        "Skipped tasks: " + (", ".join(result["skipped_tasks"]) or "none"),
        "",
        "Limitations: deterministic lexical quality is a proxy; small historical "
        "samples, cold starts, model availability, and provider quotas limit "
        "generalization. Historical profile constraints do not guarantee the "
        "latency, quality or cost of a new request. Unobserved models are never "
        "assigned synthetic results.",
        "",
    ]
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("\n".join(lines))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", default="gpt-oss-120b-cloud")
    parser.add_argument(
        "--policy",
        choices=["cost_first", "balanced", "quality_first", "latency_first"],
        default="cost_first",
    )
    parser.add_argument("--min-quality", type=float, default=0.5)
    parser.add_argument("--results-dir", type=Path, default=None)
    parser.add_argument("--output", type=Path, default=Path("docs/kpi_report.md"))
    args = parser.parse_args()
    service = EvaluationService(args.results_dir)
    records, sources = service.load_runs()
    registry = ModelRegistry.from_config()
    result = simulate(
        records,
        BenchmarkService.load_tasks(),
        registry,
        args.baseline,
        args.policy,
        Constraints(min_quality=args.min_quality),
    )
    write_report(result, args.output, args.baseline, args.policy, sources)
    print(f"Wrote {args.output}: {result['paired_tasks']} paired tasks")


if __name__ == "__main__":
    main()
