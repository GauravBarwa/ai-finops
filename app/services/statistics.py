"""Shared latency statistics (linear interpolation, including cold starts)."""

from statistics import mean, median


def percentile(values: list[float], fraction: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    position = (len(ordered) - 1) * fraction
    lo = int(position)
    hi = min(lo + 1, len(ordered) - 1)
    return ordered[lo] + (ordered[hi] - ordered[lo]) * (position - lo)


def latency_summary(values: list[float]) -> dict[str, float]:
    return {
        "avg_latency_ms": mean(values) if values else 0.0,
        "median_latency_ms": median(values) if values else 0.0,
        "p95_latency_ms": percentile(values, 0.95),
    }
