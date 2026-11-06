"""Configurable nonnegative weight presets; callers may override via RouteRequest."""

from app.schemas.routing import PolicyName, Weights

PRESETS: dict[PolicyName, Weights] = {
    "balanced": Weights(),
    "cost_first": Weights(cost=0.65, quality=0.2, latency=0.1, reliability=0.05),
    "quality_first": Weights(quality=0.65, cost=0.1, latency=0.1, reliability=0.15),
    "latency_first": Weights(latency=0.65, quality=0.2, cost=0.1, reliability=0.05),
}


def policy_weights(policy: PolicyName, custom: Weights | None = None) -> Weights:
    return (custom or PRESETS[policy]).model_copy()
