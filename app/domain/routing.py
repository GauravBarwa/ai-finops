"""Hard constraints and weighted ranking over explicit measured or prompt-estimated metrics."""

from app.domain.policies import policy_weights
from app.domain.prompt_analysis import analyze_prompt
from app.domain.scoring import rank_candidates
from app.llm.registry import ModelInfo
from app.schemas.profile import ModelProfile
from app.schemas.prompt import CandidateEstimate, PromptAnalysis
from app.schemas.routing import CandidateDecision, RouteRequest, RouteResponse


def recommend(
    request: RouteRequest,
    models: list[ModelInfo],
    profiles: list[ModelProfile],
    *,
    analysis: PromptAnalysis | None = None,
    estimates: dict[str, CandidateEstimate] | None = None,
) -> RouteResponse:
    analysis = analysis or analyze_prompt(request.prompt, request.task_type, request.max_tokens)
    task_type = analysis.effective_task_type
    estimates = estimates or {}
    profile_map = {(p.model_id, p.task_type): p for p in profiles}
    candidates = []
    constraints = request.constraints
    for model in models:
        profile = profile_map.get((model.model_id, task_type))
        estimate = estimates.get(model.model_id) if request.routing_mode == "prompt_aware" else None
        input_tokens = estimate.estimated_input_tokens if estimate else analysis.input_tokens_rough
        checks = {
            "enabled": model.enabled,
            "task_supported": task_type in model.capabilities,
            "profile_available": profile is not None,
            "context_window": input_tokens + request.max_tokens <= model.context_window,
        }
        details = {
            "context_window": f"estimated input {input_tokens} + output budget "
            f"{request.max_tokens} exceeds {model.context_window} tokens"
        }
        if profile is not None:
            quality = estimate.quality_score if estimate else profile.quality_score
            cost = estimate.budget_estimated_cost if estimate else profile.avg_estimated_cost
            latency = estimate.estimated_latency_ms if estimate else profile.median_latency_ms
            reliability = estimate.reliability_rate if estimate else profile.reliability_rate
            checks.update(
                min_quality=quality >= constraints.min_quality,
                max_cost=constraints.max_cost is None or cost <= constraints.max_cost,
                max_latency_ms=constraints.max_latency_ms is None
                or latency <= constraints.max_latency_ms,
                min_reliability=reliability >= constraints.min_reliability,
            )
            basis = "estimated" if estimate else "measured"
            cost_basis = "estimated using maximum output budget" if estimate else "measured average"
            details.update(
                {
                    "min_quality": (
                        f"{basis} {quality:.3f} < required {constraints.min_quality:.3f}"
                    ),
                    "max_cost": f"{cost_basis} ${cost:.8f} exceeds limit ${constraints.max_cost}",
                    "max_latency_ms": (
                        f"{basis} {latency:.0f}ms exceeds limit {constraints.max_latency_ms}ms"
                    ),
                    "min_reliability": f"{basis} {reliability:.3f} < required "
                    f"{constraints.min_reliability:.3f}",
                }
            )
        reasons = [
            f"{name}: failed" + (f" ({details[name]})" if name in details else "")
            for name, passed in checks.items()
            if not passed
        ]
        candidates.append(
            CandidateDecision(
                model_id=model.model_id,
                passed=all(checks.values()),
                checks=checks,
                rejection_reasons=reasons,
                profile=profile,
                estimate=estimate,
            )
        )
    weights = policy_weights(request.policy, request.weights)
    ranked = rank_candidates(candidates, weights)
    if not ranked:
        explanation = no_candidate_explanation(request, candidates, task_type)
    elif estimates and request.routing_mode == "prompt_aware":
        explanation = (
            f"{ranked[0]} has the highest weighted utility for this prompt using "
            f"{task_type} evidence, request token estimates and your constraints. "
            "Cost filtering uses the maximum output-token budget; ranking uses expected "
            "cost. Quality and latency are predictions, not new measurements."
        )
    else:
        explanation = (
            f"{ranked[0]} has the highest weighted utility among eligible models. "
            "Performance metrics use task averages; the prompt is checked for task "
            "intent and context size. See assessment notes for evidence limitations."
        )
    return RouteResponse(
        routing_mode=request.routing_mode,
        prompt_analysis=analysis,
        selected_model_id=ranked[0] if ranked else None,
        policy=request.policy,
        weights=weights,
        candidates=candidates,
        ranked_model_ids=ranked,
        status="recommended" if ranked else "no_candidate",
        explanation=explanation,
    )


def no_candidate_explanation(
    request: RouteRequest,
    candidates: list[CandidateDecision],
    task_type: str,
) -> str:
    measured = [
        c
        for c in candidates
        if c.profile is not None and c.checks["enabled"] and c.checks["task_supported"]
    ]
    if not measured:
        return (
            f"No measured profiles are available for enabled {task_type} models. "
            "Run benchmarks and use Model Profiles > Evaluate saved benchmarks. "
            "No provider was called."
        )
    message = "No model satisfies all the current constraints. "
    best = max(
        c.estimate.quality_score if c.estimate else c.profile.quality_score for c in measured
    )
    if request.constraints.min_quality > best:
        basis = "estimated" if any(c.estimate for c in measured) else "measured"
        message += (
            f"Best {basis} {task_type} quality is {best:.3f}; your minimum is "
            f"{request.constraints.min_quality:.3f}. Lower the minimum explicitly or "
            "collect improved benchmark results. "
        )
    else:
        message += "Inspect prompt estimates and rejection reasons below, then adjust your limits. "
    return message + "Constraints were not relaxed and no provider was called."
