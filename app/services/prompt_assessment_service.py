"""Build prompt-conditioned estimates from validated, saved benchmark observations."""

from __future__ import annotations

import math
from collections import defaultdict
from statistics import mean, median

from app.domain.prompt_analysis import match_examples, rough_tokens
from app.evaluation.metrics import score_output
from app.llm.cost import calculate_cost
from app.llm.registry import ModelInfo
from app.schemas.profile import ModelProfile
from app.schemas.prompt import CandidateEstimate, PromptAnalysis
from app.schemas.telemetry import TelemetryRecord
from app.services.benchmark_service import BenchmarkTask


def assess_candidates(
    analysis: PromptAnalysis,
    prompt: str,
    models: list[ModelInfo],
    profiles: list[ModelProfile],
    records: list[TelemetryRecord],
    tasks: list[BenchmarkTask],
) -> dict[str, CandidateEstimate]:
    golden = {t.task_id: t for t in tasks if t.task_type == analysis.effective_task_type}
    # Only benchmark records tied to the exact current golden input are evidence.
    samples = [
        r
        for r in records
        if r.source == "benchmark"
        and r.task_id in golden
        and r.task_type == analysis.effective_task_type
        and r.prompt == golden[r.task_id].prompt
    ]
    available = {r.task_id for r in samples}
    analysis.matched_tasks = match_examples(
        prompt, {key: t.prompt for key, t in golden.items() if key in available}
    )
    analysis.assessment_method = (
        "lexical_neighbors_and_token_estimates"
        if analysis.matched_tasks
        else "token_estimates_and_task_prior"
    )
    if not analysis.matched_tasks:
        analysis.warnings.append(
            "No close benchmark topic match; quality/reliability use task "
            "averages. Cost/context use this prompt size. Latency scales with token usage "
            "where measured usage is available; otherwise it keeps the historical median."
        )
    else:
        analysis.warnings.append(
            "Topic similarity and blended quality are heuristics, not "
            "measurements of this new answer or its semantic correctness."
        )
    profile_map = {(p.model_id, p.task_type): p for p in profiles}
    estimates = {}
    for model in models:
        profile = profile_map.get((model.model_id, analysis.effective_task_type))
        observed = [r for r in samples if r.model_id == model.model_id]
        if profile is None:
            continue
        by_task = defaultdict(list)
        for record in observed:
            by_task[record.task_id].append(record)
        # Do not replace an unavailable model's evidence with another model's measurements.
        matches = [m for m in analysis.matched_tasks if m.task_id in by_task]
        evidence = [r for m in matches for r in by_task[m.task_id]] if matches else observed
        quality, reliability = profile.quality_score, profile.reliability_rate
        basis = "task_average_fallback" if observed else "task_average_without_raw_evidence"
        latency = median(r.latency_ms for r in evidence) if evidence else profile.median_latency_ms
        if matches:
            total = sum(m.similarity for m in matches)
            neighbor_quality = (
                sum(
                    m.similarity
                    * mean(
                        score_output(
                            analysis.effective_task_type,
                            r.response,
                            golden[r.task_id].reference or "",
                            prompt=r.prompt,
                        )
                        if r.success
                        else 0
                        for r in by_task[m.task_id]
                    )
                    for m in matches
                )
                / total
            )
            neighbor_reliability = (
                sum(
                    m.similarity * mean(int(r.success) for r in by_task[m.task_id]) for m in matches
                )
                / total
            )
            # Shrink weak lexical matches toward the broad measured task prior.
            blend = max(m.similarity for m in matches)
            quality = blend * neighbor_quality + (1 - blend) * quality
            reliability = blend * neighbor_reliability + (1 - blend) * reliability
            basis = "similarity_weighted_benchmark_blend"
        successful = [r for r in evidence if r.success and r.input_tokens > 0]
        if not successful:
            successful = [r for r in observed if r.success and r.input_tokens > 0]
        ratio = (
            median(r.input_tokens / rough_tokens(r.prompt) for r in successful) if successful else 1
        )
        input_tokens = max(1, math.ceil(analysis.input_tokens_rough * ratio))
        observed_output = (
            math.ceil(mean(r.output_tokens for r in successful))
            if successful
            else analysis.requested_max_output_tokens
        )
        output_tokens = max(
            1,
            min(
                analysis.requested_max_output_tokens, analysis.output_tokens_hint or observed_output
            ),
        )
        cost = calculate_cost(input_tokens, output_tokens, model).total_cost
        budget_cost = calculate_cost(
            input_tokens, analysis.requested_max_output_tokens, model
        ).total_cost
        baseline_tokens = (
            mean(r.input_tokens + r.output_tokens for r in successful)
            if successful
            else analysis.input_tokens_rough + output_tokens
        )
        # Token-linear latency is an explicit planning heuristic; no artificial quality penalties.
        if successful:
            latency *= (input_tokens + output_tokens) / max(1, baseline_tokens)
        estimates[model.model_id] = CandidateEstimate(
            quality_score=max(0, min(1, quality)),
            estimated_cost=cost,
            budget_estimated_cost=budget_cost,
            estimated_latency_ms=max(0, latency),
            reliability_rate=max(0, min(1, reliability)),
            estimated_input_tokens=input_tokens,
            estimated_output_tokens=output_tokens,
            supporting_sample_count=len(evidence),
            matched_task_ids=[m.task_id for m in matches],
            quality_basis=basis,
        )
    return estimates
