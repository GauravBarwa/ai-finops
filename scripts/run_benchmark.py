#!/usr/bin/env python3
"""CLI script to run the benchmark suite.

Usage::

    # Run against all enabled models
    python -m scripts.run_benchmark

    # Run against specific models
    python -m scripts.run_benchmark --models llama3.2-local gpt-oss-120b-cloud

    # Custom benchmark file
    python -m scripts.run_benchmark --tasks data/benchmark/tasks.json
"""

from __future__ import annotations

import argparse
import asyncio
import sys
from pathlib import Path

# Ensure project root is on the path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.core.logging import setup_logging
from app.llm.provider_client import ProviderRouterClient
from app.llm.registry import ModelRegistry
from app.services.benchmark_service import BenchmarkService
from app.services.statistics import latency_summary


async def main(args: argparse.Namespace) -> None:
    logger = setup_logging()

    logger.info("=" * 60)
    logger.info("AI FinOps — Benchmark Runner")
    logger.info("=" * 60)

    # Load registry
    registry = ModelRegistry.from_config()

    # Resolve models
    if args.models:
        model_ids = args.models
        logger.info("Target models: %s", model_ids)
    else:
        model_ids = registry.get_model_ids(enabled_only=True)
        logger.info("Using all enabled models: %s", model_ids)

    # Load tasks
    tasks_path = Path(args.tasks) if args.tasks else None
    client = ProviderRouterClient()
    service = BenchmarkService(llm_client=client, registry=registry)

    tasks = service.load_tasks(tasks_path)
    logger.info("Loaded %d tasks", len(tasks))

    # Run benchmark
    records = await service.run_benchmark(
        model_ids=model_ids,
        tasks=tasks,
        max_tokens=args.max_tokens,
        temperature=args.temperature,
    )

    # Save results
    output = service.save_results(records)

    # Print summary
    logger.info("")
    logger.info("=" * 60)
    logger.info("BENCHMARK SUMMARY")
    logger.info("=" * 60)

    # Group by model
    from collections import defaultdict
    model_stats: dict[str, dict] = defaultdict(lambda: {
        "calls": 0, "success": 0, "total_tokens": 0,
        "total_cost": 0.0, "total_latency": 0.0, "latencies": [],
    })

    for r in records:
        s = model_stats[r.model_id]
        s["calls"] += 1
        s["success"] += int(r.success)
        s["total_tokens"] += r.total_tokens
        s["total_cost"] += r.estimated_cost
        s["total_latency"] += r.latency_ms
        s["latencies"].append(r.latency_ms)

    for model_id, stats in model_stats.items():
        avg_latency = stats["total_latency"] / max(stats["calls"], 1)
        percentiles = latency_summary(stats["latencies"])
        logger.info(
            "  %-25s  calls=%d  success=%d  tokens=%d  "
            "est_cost=$%.6f  avg_latency=%.0fms p50=%.0fms p95=%.0fms",
            model_id,
            stats["calls"],
            stats["success"],
            stats["total_tokens"],
            stats["total_cost"],
            avg_latency,
            percentiles["median_latency_ms"],
            percentiles["p95_latency_ms"],
        )

    logger.info("")
    logger.info("Results saved to: %s", output)
    logger.info("=" * 60)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="AI FinOps Benchmark Runner")
    parser.add_argument(
        "--models", nargs="+", default=None,
        help="Model IDs to benchmark (default: all enabled)",
    )
    parser.add_argument(
        "--tasks", default=None,
        help="Path to benchmark tasks JSON file",
    )
    parser.add_argument(
        "--max-tokens", type=int, default=1024,
        help="Max tokens per call (default: 1024)",
    )
    parser.add_argument(
        "--temperature", type=float, default=0.0,
        help="Sampling temperature (default: 0.0)",
    )
    return parser.parse_args()


if __name__ == "__main__":
    asyncio.run(main(parse_args()))
