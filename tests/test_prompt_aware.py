"""Prompt content must affect routing before execution, without fabricated online quality."""

import json
from unittest.mock import AsyncMock

import pytest
from httpx import ASGITransport, AsyncClient

from app.api import routes
from app.domain.prompt_analysis import analyze_prompt, match_examples, rough_tokens
from app.llm.interface import LLMResponse
from app.llm.registry import ModelInfo, ModelRegistry
from app.main import create_app
from app.schemas.routing import Constraints, RouteRequest
from app.schemas.telemetry import TelemetryRecord
from app.services.evaluation_service import EvaluationService
from app.services.router_service import RouterService
from app.services.telemetry_service import TelemetryService


@pytest.mark.parametrize(
    "prompt,task",
    [
        ("Summarize in 2 lines: Explain why quantum computing is hard.", "summarization"),
        ("Classify sentiment. Review: Please extract JSON from this review.", "classification"),
        ("Return only JSON with keys name and age. Text: Ada is 30.", "extraction"),
        ("Extract name and age. Text: Ada is 30.", "extraction"),
        ("Calculate 120 reduced by 25 percent. Explain briefly.", "reasoning"),
        ("How many tokens do ten requests use?", "reasoning"),
        ("Give me a summary: AI helps businesses.", "summarization"),
    ],
)
def test_auto_detects_instructions_and_not_supplied_data(prompt, task):
    analysis = analyze_prompt(prompt, "auto", 512)
    assert analysis.effective_task_type == analysis.detected_task_type == task
    assert analysis.task_type_source == "auto"
    assert analysis.signals


def test_manual_override_ambiguity_and_unknown_text_are_explicit():
    a = analyze_prompt("Summarize: Cloud computing.", "classification", 512)
    assert a.detected_task_type == "summarization" and a.effective_task_type == "classification"
    assert any("override" in warning for warning in a.warnings)
    a = analyze_prompt("Extract fields and summarize the conclusions.", "auto", 512)
    assert any("Multiple task signals" in warning for warning in a.warnings)
    a = analyze_prompt("你好世界", "auto", 512)
    assert a.effective_task_type == "reasoning"
    assert any("No clear English" in warning for warning in a.warnings)


def test_length_and_output_instruction_analysis():
    assert rough_tokens("x " * 500) > rough_tokens("x")
    assert rough_tokens("你好世界") > 0
    assert analyze_prompt("Summarize in 2 lines: A.", "auto", 512).output_tokens_hint == 64
    assert analyze_prompt("Summarize in 5 sentences: A.", "auto", 512).output_tokens_hint == 160
    assert analyze_prompt("Summarize in 1000 words: A.", "auto", 64).output_tokens_hint == 64
    assert (
        analyze_prompt(
            "Classify sentiment, only the label. Review: Great!", "auto", 512
        ).output_tokens_hint
        == 16
    )


def test_retrieval_changes_with_topic_and_rejects_unrelated_topics():
    examples = {
        "cats": "Summarize: feline kittens whiskers purr",
        "dogs": "Summarize: canine puppies fetch bark",
    }
    assert match_examples("Summarize: feline kittens whiskers purr", examples)[0].task_id == "cats"
    assert match_examples("Summarize: canine puppies fetch bark", examples)[0].task_id == "dogs"
    assert not match_examples("Summarize: superconducting quasiparticle vortices", examples)
    assert not match_examples("Summarize in 2 lines", examples)


@pytest.fixture
def evidence(tmp_path):
    models = [
        ModelInfo(
            model_id=mid,
            provider="test",
            display_name=mid,
            litellm_model_name="test/" + mid,
            context_window=4096,
            input_price_per_1m_tokens=0.1,
            output_price_per_1m_tokens=0.1,
            capabilities=["summarization"],
        )
        for mid in ["a", "b"]
    ]
    registry = ModelRegistry(models)
    tasks = [
        dict(task_id=tid, task_type="summarization", prompt=prompt, reference=ref)
        for tid, prompt, ref in [
            ("cats", "Summarize: feline kittens whiskers purr", "Feline kittens whiskers purr."),
            ("dogs", "Summarize: canine puppies fetch bark", "Canine puppies fetch bark."),
        ]
    ]
    tasks_path = tmp_path / "tasks.json"
    tasks_path.write_text(json.dumps(tasks))
    records = [
        TelemetryRecord(
            task_id=t["task_id"],
            task_type=t["task_type"],
            prompt=t["prompt"],
            model_id=mid,
            provider="test",
            response=t["reference"]
            if (mid, t["task_id"]) in {("a", "cats"), ("b", "dogs")}
            else "Unrelated incorrect answer.",
            success=True,
            latency_ms=100,
            input_tokens=20,
            output_tokens=10,
            estimated_cost=0.000003,
            source="benchmark",
        )
        for t in tasks
        for mid in ["a", "b"]
    ]
    (tmp_path / "benchmark_a.jsonl").write_text("\n".join(r.model_dump_json() for r in records))
    evaluation = EvaluationService(tmp_path, tasks_path, registry)
    snapshot = evaluation.evaluate()
    assert all(p.quality_score == 0.5 for p in snapshot.profiles)
    adapter = AsyncMock()
    adapter.generate.return_value = LLMResponse(content="Answer", input_tokens=10)
    telemetry = TelemetryService(tmp_path)
    return registry, evaluation, adapter, RouterService(adapter, registry, evaluation, telemetry)


@pytest.mark.asyncio
async def test_same_task_type_same_constraints_different_prompt_changes_model(evidence):
    _, evaluation, adapter, service = evidence
    before = evaluation.profiles_path.read_bytes()
    results = []
    for topic in ["feline kittens whiskers purr", "canine puppies fetch bark"]:
        results.append(
            await service.route(
                RouteRequest(
                    prompt="Summarize: " + topic,
                    task_type="summarization",
                    constraints=Constraints(min_quality=0.8),
                )
            )
        )
    assert [r.selected_model_id for r in results] == ["a", "b"]
    assert [r.prompt_analysis.matched_tasks[0].task_id for r in results] == ["cats", "dogs"]
    assert results[0].candidates[0].estimate.quality_score > 0.99
    assert results[1].candidates[0].estimate.quality_score < 0.01
    assert evaluation.profiles_path.read_bytes() == before  # priors stay measurements
    adapter.generate.assert_not_called()


@pytest.mark.asyncio
async def test_unknown_topic_uses_honest_prior_but_cost_is_prompt_specific(evidence):
    _, _, adapter, service = evidence
    short = await service.route(RouteRequest(prompt="Summarize: superconducting vortices"))
    long = await service.route(
        RouteRequest(prompt="Summarize: " + "superconducting vortices " * 60)
    )
    assert not short.prompt_analysis.matched_tasks
    for decision in [short, long]:
        assert all(c.estimate.quality_score == 0.5 for c in decision.candidates)
        assert any("No close benchmark" in w for w in decision.prompt_analysis.warnings)
    assert (
        long.candidates[0].estimate.budget_estimated_cost
        > short.candidates[0].estimate.budget_estimated_cost
    )
    assert (
        long.candidates[0].estimate.estimated_latency_ms
        > short.candidates[0].estimate.estimated_latency_ms
    )
    adapter.generate.assert_not_called()


@pytest.mark.asyncio
async def test_context_window_and_output_budget_are_hard_constraints(evidence):
    registry, _, adapter, service = evidence
    registry.get("a").context_window = 128
    result = await service.route(
        RouteRequest(prompt="Summarize: feline kittens whiskers purr", max_tokens=64)
    )
    assert result.candidates[0].checks["context_window"]
    result = await service.route(
        RouteRequest(prompt="Summarize: " + "feline kittens " * 100, max_tokens=64)
    )
    assert not result.candidates[0].checks["context_window"]
    assert result.selected_model_id == "b"
    # Cost must consider allowed generation, not only a cheap expected answer.
    result = await service.route(
        RouteRequest(
            prompt="Summarize: feline kittens whiskers purr",
            max_tokens=1024,
            constraints=Constraints(max_cost=0.00001),
        )
    )
    assert result.status == "no_candidate"
    assert all(not c.checks["max_cost"] for c in result.candidates)
    adapter.generate.assert_not_called()


@pytest.mark.asyncio
async def test_missing_raw_evidence_still_estimates_cost_and_reports_limitations(evidence):
    _, evaluation, adapter, service = evidence
    (evaluation.results_dir / "benchmark_a.jsonl").unlink()
    decision = await service.route(RouteRequest(prompt="Summarize: unmeasured topic"))
    assert decision.candidates[0].estimate is not None
    assert decision.candidates[0].estimate.quality_score == 0.5
    assert decision.candidates[0].estimate.quality_basis == "task_average_without_raw_evidence"
    assert any("unavailable" in w for w in decision.prompt_analysis.warnings)
    adapter.generate.assert_not_called()


@pytest.mark.asyncio
async def test_baseline_is_explicit_and_actual_execution_preserves_prompt(evidence):
    _, _, adapter, service = evidence
    baseline = await service.route(
        RouteRequest(prompt="Summarize: feline kittens whiskers purr", routing_mode="task_average")
    )
    assert baseline.prompt_analysis.assessment_method == "task_average_baseline"
    assert all(c.estimate is None for c in baseline.candidates)
    prompt = "Summarize: canine puppies fetch bark"
    executed = await service.route(RouteRequest(prompt=prompt, execute=True))
    assert executed.executed_model_id == "b" and executed.response == "Answer"
    assert adapter.generate.call_args.kwargs["prompt"] == prompt


@pytest.mark.asyncio
async def test_api_auto_and_prompt_evidence_are_serialized(evidence):
    registry, evaluation, adapter, _ = evidence
    routes.init_services(registry, adapter, TelemetryService(evaluation.results_dir), evaluation)
    async with AsyncClient(transport=ASGITransport(app=create_app()), base_url="http://test") as c:
        r = await c.post("/route", json={"prompt": "Summarize: canine puppies fetch bark"})
        assert r.status_code == 200
        body = r.json()
        assert body["routing_mode"] == "prompt_aware"
        assert body["prompt_analysis"]["effective_task_type"] == "summarization"
        assert body["selected_model_id"] == "b"
        assert body["candidates"][1]["estimate"]["matched_task_ids"] == ["dogs"]
    adapter.generate.assert_not_called()


def test_shared_instruction_labels_are_not_topic_evidence():
    examples = {
        "happy": "Classify POSITIVE, NEGATIVE, NEUTRAL. Review: Excellent product, delighted.",
        "sad": "Classify POSITIVE, NEGATIVE, NEUTRAL. Review: Broken item, terrible experience.",
    }
    assert not match_examples(
        "Classify POSITIVE, NEGATIVE, NEUTRAL. Review: superconducting vortices",
        examples,
    )
    match = match_examples(
        "Classify POSITIVE, NEGATIVE, NEUTRAL. Review: Excellent product, delighted.",
        examples,
    )
    assert match[0].task_id == "happy" and match[0].similarity == pytest.approx(1)


def test_extreme_word_hint_is_capped_without_overflow():
    prompt = "Summarize in " + "9" * 400 + " words: a short document."
    analysis = analyze_prompt(prompt, "auto", 512)
    assert analysis.output_tokens_hint == 512
