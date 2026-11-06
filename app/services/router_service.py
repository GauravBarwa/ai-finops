"""Recommend from offline profiles; execute primary and at most one eligible fallback."""

from app.core.exceptions import BenchmarkError
from app.domain.prompt_analysis import analyze_prompt
from app.domain.routing import recommend
from app.llm.interface import LLMClient
from app.llm.registry import ModelRegistry
from app.schemas.comparison import CompareRequest
from app.schemas.routing import RouteRequest, RouteResponse
from app.services.benchmark_service import BenchmarkService
from app.services.comparison_service import ComparisonService
from app.services.evaluation_service import EvaluationService
from app.services.prompt_assessment_service import assess_candidates
from app.services.telemetry_service import TelemetryService


class RouterService:
    def __init__(
        self,
        llm_client: LLMClient,
        registry: ModelRegistry,
        evaluation_service: EvaluationService,
        telemetry_service: TelemetryService,
    ):
        self.registry = registry
        self.evaluation = evaluation_service
        self.telemetry = telemetry_service
        self.comparison = ComparisonService(llm_client, registry)

    async def route(self, request: RouteRequest) -> RouteResponse:
        analysis = analyze_prompt(request.prompt, request.task_type, request.max_tokens)
        try:
            snapshot = self.evaluation.load_profiles()
            profiles = snapshot.profiles
        except (ValueError, OSError) as exc:
            decision = recommend(request, self.registry.list_all(), [], analysis=analysis)
            decision.explanation = f"Profiles could not be loaded: {exc} No model was called."
            return decision
        estimates = {}
        if request.routing_mode == "prompt_aware" and profiles:
            try:
                records, _ = self.evaluation.load_runs(snapshot.source_files or None)
                tasks = BenchmarkService.load_tasks(self.evaluation.tasks_path)
                self.evaluation.aggregate(records, tasks)  # validate golden identity/metrics
                estimates = assess_candidates(
                    analysis,
                    request.prompt,
                    self.registry.list_all(),
                    profiles,
                    records,
                    tasks,
                )
            except (ValueError, OSError, BenchmarkError) as exc:
                analysis.warnings.append(
                    f"Detailed benchmark evidence is unavailable ({exc}); quality/reliability "
                    "and latency use task profiles. Request cost/context still use prompt size."
                )
                estimates = assess_candidates(
                    analysis,
                    request.prompt,
                    self.registry.list_all(),
                    profiles,
                    [],
                    [],
                )
        elif request.routing_mode == "task_average":
            analysis.assessment_method = "task_average_baseline"
        decision = recommend(
            request, self.registry.list_all(), profiles, analysis=analysis, estimates=estimates
        )
        if not request.execute or not decision.ranked_model_ids:
            return decision
        for model_id in decision.ranked_model_ids[:2]:
            result = await self.comparison.compare(
                CompareRequest(
                    prompt=request.prompt,
                    task_type=analysis.effective_task_type,
                    models=[model_id],
                    max_tokens=request.max_tokens,
                    temperature=request.temperature,
                )
            )
            self.telemetry.save_comparison(result, source="route")
            attempt = result.results[0]
            decision.attempts.append(attempt)
            decision.total_estimated_cost += attempt.estimated_cost
            decision.fallback_used = len(decision.attempts) > 1
            if attempt.success:
                decision.executed_model_id = model_id
                decision.response = attempt.response
                decision.status = "succeeded"
                if decision.fallback_used:
                    decision.explanation += " Primary failed; second-ranked candidate succeeded."
                return decision
        decision.status = "execution_failed"
        decision.explanation += " All attempted eligible candidates failed."
        return decision
