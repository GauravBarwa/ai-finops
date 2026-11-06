"""Gradio demonstration for comparison, prompt-aware routing, and model profiles.

Launch with::

    python -m app.demo

Or alongside the FastAPI server::

    # Terminal 1: uvicorn app.main:app --reload
    # Terminal 2: python -m app.demo
"""

from __future__ import annotations

import asyncio
import re

import gradio as gr

from app.core.exceptions import BenchmarkError
from app.core.logging import setup_logging
from app.llm.interface import LLMClient
from app.llm.provider_client import ProviderRouterClient
from app.llm.registry import ModelRegistry
from app.schemas.comparison import CompareRequest, CompareResponse
from app.schemas.profile import ProfilesResponse
from app.schemas.routing import Constraints, RouteRequest
from app.services.comparison_service import ComparisonService
from app.services.evaluation_service import EvaluationService
from app.services.router_service import RouterService
from app.services.telemetry_service import TelemetryService
from app.ui import LAUNCH_OPTIONS

COMPARISON_HEADERS = ["Model", "Provider", "Tokens (in/out)", "Latency", "Est. cost USD", "Status"]


def markdown_text(text: str) -> str:
    """Keep provider failures literal, even when they contain Markdown fences."""
    longest_run = max((len(match[0]) for match in re.finditer(r"`+", text)), default=0)
    fence = "`" * max(3, longest_run + 1)
    return f"{fence}text\n{text}\n{fence}"


async def reveal_answer(text: str):
    """Animate completed answers in bounded frames while preserving their exact Markdown."""
    step = max(1, (len(text) + 31) // 32)
    for end in range(step, len(text), step):
        yield text[:end]
        await asyncio.sleep(0.025)
    yield text


def format_comparison(
    response: CompareResponse, visible_answers: dict[str, str] | None = None
) -> tuple[str, list[list[str]], str]:
    """Keep provider text out of Markdown tables and show pending work explicitly."""
    pending = response.total_models - len(response.results)
    summary = (
        f"**Task ID:** `{response.task_id}`  \n"
        f"**Models:** {response.successful} succeeded, {response.failed} failed, "
        f"{pending} pending out of {response.total_models}"
    )
    rows, details = [], []
    for result in response.results:
        status = (
            "Succeeded"
            if result.success
            else "Failed: " + " ".join((result.error or "Unknown error").split())
        )
        rows.append(
            [
                result.display_name,
                result.provider,
                f"{result.input_tokens}/{result.output_tokens}",
                f"{result.latency_ms:.0f}ms",
                f"{result.estimated_cost:.8f}",
                status,
            ]
        )
        answer = (
            visible_answers.get(result.model_id, result.response)
            if visible_answers is not None
            else result.response
        )
        details.append(
            f"## {result.display_name} ({result.provider})\n\n"
            f"{'Succeeded' if result.success else 'Failed'}\n\n"
            f"{answer if result.success else markdown_text(result.error or 'Unknown error')}"
        )
    return summary, rows, "\n\n---\n\n".join(details)


def describe_profiles(
    snapshot: ProfilesResponse,
    registry: ModelRegistry,
    task_type: str,
) -> str:
    if task_type == "auto":
        return (
            f"{len(snapshot.profiles)} saved profiles. Auto detects the task from your prompt. "
            "Route request shows prompt estimates, related benchmarks and evidence limits. "
            "Choose an explicit task only to override detection."
        )
    supported = {m.model_id for m in registry.list_enabled() if task_type in m.capabilities}
    measured = [
        p for p in snapshot.profiles if p.task_type == task_type and p.model_id in supported
    ]
    if not measured:
        return (
            f"No measured {task_type} profiles for enabled models. "
            "Run benchmarks, then click Evaluate saved benchmarks in Model Profiles. "
            "A comparison alone does not create quality profiles."
        )
    best = max(p.quality_score for p in measured)
    return (
        f"{len(snapshot.profiles)} saved profiles. {task_type.capitalize()} coverage: "
        f"{len(measured)}/{len(supported)} eligible models measured. "
        f"Best measured quality: {best:.3f}. Scores are reference-based estimates, "
        "not guarantees for your prompt. See candidate values when adjusting limits."
    )


def build_demo(
    *,
    registry: ModelRegistry | None = None,
    llm_client: LLMClient | None = None,
    evaluation: EvaluationService | None = None,
    telemetry: TelemetryService | None = None,
) -> gr.Blocks:
    """Build and return the Gradio Blocks interface."""
    setup_logging()

    registry = registry or ModelRegistry.from_config()
    client = llm_client or ProviderRouterClient()
    service = ComparisonService(llm_client=client, registry=registry)

    evaluation = evaluation or EvaluationService(registry=registry)
    telemetry = telemetry or TelemetryService()
    routing = RouterService(client, registry, evaluation, telemetry)

    async def smart_route(
        prompt,
        task_type,
        policy,
        quality,
        cost,
        latency,
        reliability,
        execute,
        max_tokens=1024,
        routing_mode="prompt_aware",
    ):
        yield "Evaluating your request…", [], ""
        try:
            decision = await routing.route(
                RouteRequest(
                    prompt=prompt,
                    task_type=task_type,
                    policy=policy,
                    execute=execute,
                    max_tokens=max_tokens,
                    routing_mode=routing_mode,
                    constraints=Constraints(
                        min_quality=quality,
                        max_cost=cost,
                        max_latency_ms=latency,
                        min_reliability=reliability,
                    ),
                )
            )
        except (ValueError, TypeError, OverflowError) as exc:
            yield f"**Status:** invalid_request  \n{exc}", [], "No provider was called."
            return
        rows = []
        for candidate in decision.candidates:
            p, e = candidate.profile, candidate.estimate
            rows.append(
                [
                    candidate.model_id,
                    "PASS" if candidate.passed else "FAIL",
                    round(e.quality_score if e else p.quality_score, 3) if p else None,
                    e.estimated_cost if e else p.avg_estimated_cost if p else None,
                    e.budget_estimated_cost if e else p.avg_estimated_cost if p else None,
                    round(e.estimated_latency_ms if e else p.median_latency_ms) if p else None,
                    round(e.reliability_rate if e else p.reliability_rate, 3) if p else None,
                    e.supporting_sample_count if e else p.sample_count if p else None,
                    ", ".join(e.matched_task_ids) if e else "",
                    e.quality_basis if e else "historical task profile",
                    candidate.rank,
                    round(candidate.score, 4) if candidate.score is not None else None,
                    "; ".join(candidate.rejection_reasons) or "All constraints passed",
                ]
            )
        badge = f"**Selected model:** {decision.selected_model_id or 'None'}  \n"
        badge += f"**Status:** {decision.status}  \n{decision.explanation}"
        if decision.executed_model_id:
            badge += f"  \n**Executed model:** {decision.executed_model_id}"
        analysis = decision.prompt_analysis
        if analysis is not None:
            badge += (
                f"  \n**Task:** {analysis.effective_task_type} ({analysis.task_type_source}); "
                f"detected: {analysis.detected_task_type}.  \n"
                f"**Prompt size:** about {analysis.input_tokens_rough} input tokens; "
                f"output budget {analysis.requested_max_output_tokens}.  \n"
                f"**Assessment:** {analysis.assessment_method}."
            )
            matches = ", ".join(
                f"{m.task_id} ({m.similarity:.2f} lexical similarity)"
                for m in analysis.matched_tasks
            )
            badge += f"  \n**Related benchmark tasks:** {matches or 'No close topic match'}."
            if analysis.warnings:
                badge += "  \n**Evidence notes:** " + " ".join(analysis.warnings)
        live = decision.response or "\n".join(a.error or "" for a in decision.attempts)
        if decision.status == "recommended":
            live = "Recommendation only. Enable Execute selected model to generate a response."
        elif decision.status == "no_candidate":
            live = decision.explanation
        if decision.status == "succeeded":
            async for partial in reveal_answer(live):
                yield badge, rows, partial
        else:
            if decision.status == "execution_failed":
                live = markdown_text(live)
            yield badge, rows, live

    def read_profile_view(task_type="summarization"):
        try:
            snapshot = evaluation.load_profiles()
            status = describe_profiles(snapshot, registry, task_type)
        except (ValueError, OSError) as exc:
            return [], str(exc)
        return [
            [
                p.model_id,
                p.task_type,
                round(p.quality_score, 3),
                round(p.median_latency_ms, 2),
                round(p.p95_latency_ms, 2),
                f"{p.avg_estimated_cost:.8f}",
                round(p.reliability_rate, 3),
                p.sample_count,
            ]
            for p in snapshot.profiles
        ], status

    async def task_profile_hint(task_type):
        return read_profile_view(task_type)[1]

    async def refresh_profiles(task_type):
        rows, status = read_profile_view(task_type)
        return rows, status, status

    async def evaluate_saved(task_type):
        try:
            evaluation.evaluate()
        except (ValueError, OSError, BenchmarkError) as exc:
            rows, _ = read_profile_view(task_type)
            return rows, f"Evaluation could not complete: {exc}", str(exc)
        return await refresh_profiles(task_type)

    enabled_models = registry.list_enabled()
    model_choices = [(m.display_name, m.model_id) for m in enabled_models]

    async def _compare(prompt, task_type, selected_models, max_tokens):
        if not isinstance(prompt, str) or not prompt.strip():
            yield "Please enter a prompt.", [], ""
            return
        try:
            request = CompareRequest(
                prompt=prompt,
                task_type=None if task_type == "auto" else task_type,
                models=selected_models or None,
                max_tokens=max_tokens,
            )
        except (ValueError, TypeError, OverflowError) as exc:
            yield f"Invalid request: {exc}", [], "No provider was called."
            return
        visible_answers = {}
        async for response in service.stream_compare(request):
            new_results = [r for r in response.results if r.model_id not in visible_answers]
            for result in new_results:
                visible_answers[result.model_id] = ""
                if result.success:
                    async for partial in reveal_answer(result.response):
                        visible_answers[result.model_id] = partial
                        yield format_comparison(response, visible_answers)
            yield format_comparison(response)
        telemetry.save_comparison(response)

    # Keep results at full width so model names, costs, and rejection reasons are readable.
    with gr.Blocks(title="AI FinOps") as demo:
        gr.Markdown(
            "# AI FinOps\n\nModel evaluation and request routing, backed by measured evidence.",
            elem_id="finops-header",
        )

        with gr.Tab("Comparison"):
            gr.Markdown(
                "### Compare model performance\nRun one prompt across your selected models."
            )
            with gr.Row(equal_height=True):
                with gr.Column(scale=3, min_width=320, elem_classes="finops-panel"):
                    prompt_input = gr.Textbox(
                        label="Prompt", placeholder="Enter a task or instruction…", lines=6
                    )
                    with gr.Row():
                        task_type_input = gr.Dropdown(
                            label="Task Type",
                            choices=[
                                "auto",
                                "summarization",
                                "classification",
                                "extraction",
                                "reasoning",
                            ],
                            value="auto",
                        )
                        token_limit = gr.Slider(
                            64,
                            2048,
                            step=64,
                            value=1024,
                            label="Maximum output tokens (including reasoning where applicable)",
                        )
                with gr.Column(scale=2, min_width=280, elem_classes="finops-panel"):
                    model_selector = gr.CheckboxGroup(
                        label="Models (leave empty for all enabled)", choices=model_choices
                    )
                    gr.Markdown(
                        "Results update as each model finishes. Costs use configured shadow rates.",
                        elem_classes="finops-note",
                    )
                    compare_btn = gr.Button("Compare models", variant="primary")
            summary_output = gr.Markdown(elem_classes="finops-section")
            comparison_table = gr.Dataframe(
                headers=COMPARISON_HEADERS,
                datatype=["str"] * 6,
                interactive=False,
                label="Model results",
                wrap=True,
                max_height=600,
                column_widths=["22%", "10%", "12%", "10%", "14%", "32%"],
                elem_classes="results-table",
            )
            gr.Markdown("### Model responses", elem_classes="finops-section")
            detail_output = gr.Markdown(
                label="Full responses and errors",
                sanitize_html=True,
                elem_classes="response-panel",
            )
            compare_btn.click(
                fn=_compare,
                inputs=[prompt_input, task_type_input, model_selector, token_limit],
                outputs=[summary_output, comparison_table, detail_output],
            )
            with gr.Accordion("Sample prompts", open=False):
                gr.Examples(
                    examples=[
                        [
                            "Summarize in 2 sentences: Cloud computing allows businesses to rent "
                            "computing resources on demand instead of maintaining expensive "
                            "servers. This pay-as-you-go model has enabled startups to launch "
                            "products without "
                            "massive capital investment.",
                            "summarization",
                            [],
                            1024,
                        ],
                        [
                            "Classify the sentiment as POSITIVE, NEGATIVE, or NEUTRAL:\n\n"
                            '"Great product, fast shipping, exactly as described!"',
                            "classification",
                            [],
                            1024,
                        ],
                    ],
                    inputs=[prompt_input, task_type_input, model_selector, token_limit],
                )

        with gr.Tab("Smart Router"):
            gr.Markdown(
                "### Route a request\n"
                "Choose a model using your prompt, saved evidence, and requirements."
            )
            route_prompt = gr.Textbox(
                label="Prompt",
                lines=5,
                placeholder="Describe the task and include its source text…",
            )
            with gr.Row():
                route_task = gr.Dropdown(
                    choices=["auto", "summarization", "classification", "extraction", "reasoning"],
                    value="auto",
                    label="Task type",
                )
                routing_basis = gr.Dropdown(
                    choices=[
                        ("Prompt-aware estimates", "prompt_aware"),
                        ("Historical task averages (baseline)", "task_average"),
                    ],
                    value="prompt_aware",
                    label="Routing basis",
                )
                route_policy = gr.Dropdown(
                    choices=["cost_first", "balanced", "quality_first", "latency_first"],
                    value="balanced",
                    label="Policy",
                )
            with gr.Group(elem_classes="finops-panel"):
                gr.Markdown("#### Requirements")
                with gr.Row():
                    with gr.Column(min_width=280):
                        quality = gr.Slider(0, 1, value=0.5, step=0.01, label="Minimum quality")
                        reliability = gr.Slider(
                            0, 1, value=0.8, step=0.01, label="Minimum reliability"
                        )
                    with gr.Column(min_width=280):
                        cost = gr.Number(
                            value=0.01,
                            minimum=0,
                            step=0.00000001,
                            precision=8,
                            label="Maximum estimated cost (USD/request)",
                        )
                        latency = gr.Slider(
                            0, 180000, value=30000, step=1, label="Maximum median latency (ms)"
                        )
                gr.Markdown(
                    "Quality and latency are estimates. The default quality threshold is 0.50. "
                    "Stricter requirements may leave no eligible model.",
                    elem_classes="finops-note",
                )
            with gr.Row():
                route_token_limit = gr.Slider(
                    64, 2048, step=64, value=1024, label="Maximum output tokens"
                )
                execute = gr.Checkbox(
                    value=False, label="Execute selected model (uses provider quota)"
                )
            route_button = gr.Button("Route request", variant="primary")
            selected_badge = gr.Markdown(elem_classes="decision-panel")
            gr.Markdown("### Candidate evaluation", elem_classes="finops-section")
            gr.Markdown(
                "Expected cost informs ranking. Budget cost reserves the maximum output tokens. "
                "Scroll horizontally to inspect all evidence and rejection reasons.",
                elem_classes="finops-note",
            )
            explanations = gr.Dataframe(
                headers=[
                    "Model",
                    "Constraints",
                    "Quality estimate",
                    "Expected cost USD",
                    "Budget cost USD",
                    "Latency ms",
                    "Reliability",
                    "Samples",
                    "Matched tasks",
                    "Quality basis",
                    "Rank",
                    "Score",
                    "Rejection reasons",
                ],
                interactive=False,
                wrap=True,
                max_height=650,
                column_widths=[200, 100, 110, 130, 130, 110, 100, 90, 150, 230, 65, 85, 400],
                elem_classes="results-table",
            )
            with gr.Accordion("Profile coverage and evidence", open=False):
                task_hint = gr.Markdown(value=read_profile_view("auto")[1])
            route_task.change(task_profile_hint, inputs=route_task, outputs=task_hint)
            gr.Markdown("### Generated response", elem_classes="finops-section")
            live_response = gr.Markdown(
                label="Live response", sanitize_html=True, elem_classes="response-panel"
            )
            route_button.click(
                smart_route,
                [
                    route_prompt,
                    route_task,
                    route_policy,
                    quality,
                    cost,
                    latency,
                    reliability,
                    execute,
                    route_token_limit,
                    routing_basis,
                ],
                [selected_badge, explanations, live_response],
            )

        with gr.Tab("Model Profiles"):
            gr.Markdown(
                "### Measured model profiles\n"
                "Inspect reference scores, response times, and reliability from saved benchmarks."
            )
            with gr.Row():
                refresh = gr.Button("Refresh profiles")
                evaluate_button = gr.Button("Evaluate saved benchmarks", variant="primary")
            profile_status = gr.Markdown(value=read_profile_view()[1])
            profile_table = gr.Dataframe(
                headers=[
                    "Model",
                    "Task",
                    "Quality",
                    "Median ms",
                    "p95 ms",
                    "Est. cost USD",
                    "Reliability",
                    "Samples",
                ],
                value=read_profile_view()[0],
                interactive=False,
                wrap=True,
                max_height=750,
                column_widths=["22%", "14%", "9%", "11%", "11%", "13%", "11%", "9%"],
                elem_classes="results-table",
            )
            refresh.click(
                refresh_profiles,
                inputs=route_task,
                outputs=[profile_table, profile_status, task_hint],
            )
            evaluate_button.click(
                evaluate_saved,
                inputs=route_task,
                outputs=[profile_table, profile_status, task_hint],
            )

    return demo


if __name__ == "__main__":
    demo = build_demo()
    demo.launch(server_name="0.0.0.0", server_port=7860, **LAUNCH_OPTIONS)
