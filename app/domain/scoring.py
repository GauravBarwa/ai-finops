"""Min-max utility normalization among passing candidates, deterministic ties."""

from app.schemas.routing import CandidateDecision, Weights


def rank_candidates(candidates: list[CandidateDecision], weights: Weights) -> list[str]:
    passing = [c for c in candidates if c.passed and c.profile is not None]
    if not passing:
        return []
    fields = {
        "quality": "quality_score",
        "cost": "avg_estimated_cost",
        "latency": "median_latency_ms",
        "reliability": "reliability_rate",
    }
    total_weight = sum(weights.model_dump().values())
    for metric, field in fields.items():
        estimate_fields = {
            "quality": "quality_score",
            "cost": "estimated_cost",
            "latency": "estimated_latency_ms",
            "reliability": "reliability_rate",
        }
        values = [
            getattr(c.estimate, estimate_fields[metric])
            if c.estimate is not None
            else getattr(c.profile, field)
            for c in passing
        ]
        lo, hi = min(values), max(values)
        for candidate, value in zip(passing, values):
            utility = (value - lo) / (hi - lo) if hi != lo else 1.0
            if hi != lo and metric in {"cost", "latency"}:
                utility = 1 - utility
            candidate.normalized_scores[metric] = utility
    for candidate in passing:
        candidate.score = (
            sum(
                getattr(weights, metric) * value
                for metric, value in candidate.normalized_scores.items()
            )
            / total_weight
        )
    ordered = sorted(passing, key=lambda c: (-(c.score or 0), c.model_id))
    for rank, candidate in enumerate(ordered, 1):
        candidate.rank = rank
    return [c.model_id for c in ordered]
