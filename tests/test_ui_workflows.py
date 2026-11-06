"""Exercise all three Gradio callbacks, profile lifecycle, and execution failures."""

import json
from unittest.mock import AsyncMock

import pytest

from app.demo import build_demo
from app.demo_presets import COMPARISON_PRESETS, ROUTING_PRESETS
from app.llm.interface import LLMResponse
from app.llm.registry import ModelInfo, ModelRegistry
from app.schemas.profile import ModelProfile, ProfilesResponse
from app.schemas.telemetry import TelemetryRecord
from app.services.evaluation_service import EvaluationService
from app.services.telemetry_service import TelemetryService

TASKS = ["summarization", "classification", "extraction", "reasoning"]
POLICIES = ["cost_first", "balanced", "quality_first", "latency_first"]


@pytest.fixture
def ui(tmp_path):
    registry = ModelRegistry(
        [
            ModelInfo(
                model_id=mid,
                provider="test",
                display_name=mid,
                litellm_model_name="test/" + mid,
                context_window=4096,
                input_price_per_1m_tokens=1 if mid == "cheap" else 2,
                output_price_per_1m_tokens=1 if mid == "cheap" else 2,
                capabilities=TASKS,
            )
            for mid in ["cheap", "fast"]
        ]
    )
    profiles = [
        ModelProfile(
            model_id=mid,
            task_type=task,
            quality_score=0.6 if mid == "cheap" else 0.65,
            median_latency_ms=100 if mid == "cheap" else 50,
            p95_latency_ms=200,
            avg_estimated_cost=0.001 if mid == "cheap" else 0.002,
            reliability_rate=1,
            sample_count=2,
        )
        for mid in ["cheap", "fast"]
        for task in TASKS
    ]
    references = ["Cloud costs fell. Reliability improved.", "YES", '{"name":"Ada"}', "4 hours"]
    tasks = [
        dict(task_id=task, task_type=task, prompt=task, reference=ref)
        for task, ref in zip(TASKS, references)
    ]
    tasks_path = tmp_path / "tasks.json"
    tasks_path.write_text(json.dumps(tasks))
    records = [
        TelemetryRecord(
            task_id=t["task_id"],
            task_type=t["task_type"],
            prompt=t["prompt"],
            response=t["reference"],
            model_id=mid,
            provider="test",
            latency_ms=100,
            input_tokens=10,
            output_tokens=10,
            estimated_cost=0.001,
            source="benchmark",
        )
        for t in tasks
        for mid in ["cheap", "fast"]
    ]
    (tmp_path / "benchmark_a.jsonl").write_text("\n".join(r.model_dump_json() for r in records))
    evaluation = EvaluationService(tmp_path, tasks_path, registry)
    evaluation.profiles_path.write_text(ProfilesResponse(profiles=profiles).model_dump_json())
    client = AsyncMock()
    client.generate.return_value = LLMResponse(
        content="A real-looking test answer.", input_tokens=10
    )
    telemetry = TelemetryService(tmp_path)
    demo = build_demo(
        registry=registry, llm_client=client, evaluation=evaluation, telemetry=telemetry
    )
    functions = {f.fn.__name__: f.fn for f in demo.fns.values() if f.fn}
    route_stream = functions["smart_route"]

    async def route_final(*args, **kwargs):
        outputs = [output async for output in route_stream(*args, **kwargs)]
        return outputs[-1]

    functions["smart_route"] = route_final
    return demo, functions, evaluation, client, telemetry


@pytest.mark.asyncio
@pytest.mark.parametrize("task", TASKS)
@pytest.mark.parametrize("policy", POLICIES)
async def test_all_task_policy_combinations(ui, task, policy):
    _, f, _, client, _ = ui
    badge, rows, live = await f["smart_route"](
        "Test prompt", task, policy, 0.5, 0.01, 30000, 0.8, False
    )
    assert "**Status:** recommended" in badge
    assert len(rows) == 2 and all(len(row) == 13 for row in rows)
    assert all(row[1] == "PASS" for row in rows)
    assert "Recommendation only" in live
    client.generate.assert_not_called()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "limits,reason",
    [
        ((0.7, 0.01, 30000, 0.8), "min_quality"),
        ((0.5, 0, 30000, 0.8), "max_cost"),
        ((0.5, 0.01, 0, 0.8), "max_latency_ms"),
    ],
)
async def test_strict_constraints_explain_and_do_not_execute(ui, limits, reason):
    _, f, _, client, _ = ui
    badge, rows, live = await f["smart_route"](
        "Summary", "summarization", "balanced", *limits, True
    )
    assert "**Status:** no_candidate" in badge
    assert all(reason in row[-1] for row in rows)
    assert all("estimated" in row[-1] for row in rows)
    assert "No provider was called" in live or "no provider was called" in live
    if reason == "min_quality":
        assert "0.650" in live and "0.700" in live
    client.generate.assert_not_called()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "field,value",
    [
        (0, ""),
        (0, "  "),
        (0, None),
        (1, "unknown"),
        (2, "unknown"),
        (3, -1),
        (3, 2),
        (4, -1),
        (4, float("nan")),
        (5, float("inf")),
        (6, 2),
        (8, 0),
        (8, 512.5),
        (8, None),
    ],
)
async def test_invalid_router_inputs_leave_no_stale_response(ui, field, value):
    _, f, _, client, _ = ui
    args = ["Prompt", "summarization", "balanced", 0.5, 0.01, 30000, 0.8, False, 1024]
    args[field] = value
    badge, rows, live = await f["smart_route"](*args)
    assert "invalid_request" in badge and rows == []
    assert live == "No provider was called."
    client.generate.assert_not_called()


@pytest.mark.asyncio
async def test_missing_profiles_and_evaluate_repair(ui):
    _, f, evaluation, client, _ = ui
    evaluation.profiles_path.unlink()
    rows, status, hint = await f["refresh_profiles"]("summarization")
    assert rows == [] and "No measured" in status and hint == status
    badge, _, _ = await f["smart_route"](
        "Summary", "summarization", "balanced", 0.5, 0.01, 30000, 0.8, True
    )
    assert "no_candidate" in badge and "No measured profiles" in badge
    client.generate.assert_not_called()
    rows, status, _ = await f["evaluate_saved"]("summarization")
    assert len(rows) == 8 and "2/2" in status
    badge, _, _ = await f["smart_route"](
        "Summary", "summarization", "balanced", 0.7, 0.01, 30000, 0.8, False
    )
    assert "**Status:** recommended" in badge
    assert len((await f["refresh_profiles"]("reasoning"))[0]) == 8


@pytest.mark.asyncio
async def test_corrupt_profiles_do_not_crash_any_tab_and_can_be_rebuilt(ui):
    _, f, evaluation, client, _ = ui
    evaluation.profiles_path.write_text("{invalid")
    rows, status, _ = await f["refresh_profiles"]("summarization")
    assert rows == [] and "invalid" in status
    badge, _, _ = await f["smart_route"](
        "Summary", "summarization", "balanced", 0.5, 0.01, 30000, 0.8, True
    )
    assert "no_candidate" in badge and "Profiles could not be loaded" in badge
    client.generate.assert_not_called()
    outputs = [r async for r in f["_compare"]("Prompt", "auto", ["cheap"], 1024)]
    assert "1 succeeded" in outputs[-1][0]
    rows, status, _ = await f["evaluate_saved"]("summarization")
    assert len(rows) == 8 and "invalid" not in status


@pytest.mark.asyncio
async def test_evaluation_missing_or_bad_runs_retains_previous_snapshot(ui):
    _, f, evaluation, _, _ = ui
    before = evaluation.profiles_path.read_bytes()
    run = evaluation.results_dir / "benchmark_a.jsonl"
    run.unlink()
    rows, status, _ = await f["evaluate_saved"]("summarization")
    assert rows and "No benchmark runs" in status
    assert evaluation.profiles_path.read_bytes() == before
    run.write_text("bad json")
    rows, status, _ = await f["evaluate_saved"]("summarization")
    assert rows and "could not complete" in status
    assert evaluation.profiles_path.read_bytes() == before


@pytest.mark.asyncio
async def test_live_route_fallback_and_failed_attempts(ui):
    _, f, _, client, telemetry = ui
    client.generate.side_effect = [
        RuntimeError("primary unavailable"),
        LLMResponse(content="Answer"),
    ]
    badge, _, live = await f["smart_route"](
        "Summary", "summarization", "cost_first", 0.5, 0.01, 30000, 0.8, True
    )
    assert "**Status:** succeeded" in badge and "**Executed model:** fast" in badge
    assert live == "Answer"
    assert len(telemetry.load_records()) == 2
    client.generate.side_effect = [RuntimeError("first failed"), RuntimeError("second failed")]
    badge, _, live = await f["smart_route"](
        "Summary", "summarization", "cost_first", 0.5, 0.01, 30000, 0.8, True
    )
    assert "**Status:** execution_failed" in badge
    assert "first failed" in live and "second failed" in live


@pytest.mark.asyncio
async def test_route_reveals_markdown_without_changing_answer(ui):
    demo, _, _, client, _ = ui
    answer = "## Summary\n\n- **Lower cost**\n- Better reliability\n\n```python\nprint('ok')\n```"
    client.generate.return_value = LLMResponse(content=answer)
    route = next(f.fn for f in demo.fns.values() if f.fn and f.fn.__name__ == "smart_route")
    outputs = [
        output
        async for output in route(
            "Summarize cloud computing.", "auto", "balanced", 0.5, 0.005, 30000, 0.8, True
        )
    ]
    assert len(outputs) > 2
    assert outputs[-1][2] == answer
    assert all(answer.startswith(output[2]) for output in outputs)
    assert all(len(a[2]) < len(b[2]) for a, b in zip(outputs, outputs[1:]))


@pytest.mark.asyncio
async def test_comparison_reveals_markdown_and_keeps_full_error(ui):
    _, functions, _, client, _ = ui
    answer = "## Findings\n\n**Reliable** and inexpensive."
    error = "Provider failed | " + "Long diagnostic message. " * 20 + "```\nDetails"

    async def generate(prompt, model, **kwargs):
        if model.endswith("fast"):
            return LLMResponse(success=False, error=error)
        return LLMResponse(content=answer)

    client.generate.side_effect = generate
    outputs = [item async for item in functions["_compare"]("Prompt", "auto", [], 1024)]
    assert len(outputs) > 3
    assert answer in outputs[-1][2] and error in outputs[-1][2]
    assert " ".join(error.split()) in outputs[-1][1][1][-1]


@pytest.mark.asyncio
@pytest.mark.parametrize("prompt", ["", "   ", None])
async def test_empty_comparison_inputs(ui, prompt):
    _, f, _, client, _ = ui
    outputs = [r async for r in f["_compare"](prompt, "auto", [], 1024)]
    assert outputs == [("Please enter a prompt.", [], "")]
    client.generate.assert_not_called()


@pytest.mark.asyncio
async def test_comparison_errors_selected_models_and_repeat(ui):
    _, f, _, client, _ = ui

    async def generate(prompt, model, **kwargs):
        if model.endswith("fast"):
            return LLMResponse(success=False, error='503 | overloaded\n{"retry":true}')
        return LLMResponse(content="Answer")

    client.generate.side_effect = generate
    outputs = [r async for r in f["_compare"]("Prompt", "auto", [], 1024)]
    assert "1 succeeded, 1 failed, 0 pending out of 2" in outputs[-1][0]
    assert len(outputs[-1][1]) == 2 and "503 | overloaded\n" in outputs[-1][2]
    outputs = [r async for r in f["_compare"]("Prompt", "auto", ["cheap"], 1024)]
    assert "1 succeeded, 0 failed, 0 pending out of 1" in outputs[-1][0]
    outputs = [r async for r in f["_compare"]("Prompt", "auto", ["unknown"], 1024)]
    assert "0 succeeded, 1 failed" in outputs[-1][0]
    outputs = [r async for r in f["_compare"]("Prompt", "auto", [], None)]
    assert "Invalid request" in outputs[-1][0] and outputs[-1][1] == []


def test_ui_defaults_and_complete_presets(ui):
    demo, functions, _, client, _ = ui
    config = demo.config
    quality = next(c for c in config["components"] if c["props"].get("label") == "Minimum quality")
    assert quality["props"]["value"] == 0.5
    for name, (description, request) in COMPARISON_PRESETS.items():
        loaded = functions["load_comparison_preset"](name)
        assert loaded[:4] == (
            request.prompt, request.task_type, request.models or [], request.max_tokens
        )
        assert loaded[4:] == (description, "", [], "")
    for name, (description, request) in ROUTING_PRESETS.items():
        loaded = functions["load_route_preset"](name)
        assert loaded[:10] == (
            request.prompt, request.task_type, request.policy, request.constraints.min_quality,
            request.constraints.max_cost, request.constraints.max_latency_ms,
            request.constraints.min_reliability, False, request.max_tokens, request.routing_mode,
        )
        assert loaded[10:] == (description, "", [], "")
    client.generate.assert_not_called()
    assert len([c for c in config["components"] if c["type"] == "tabitem"]) == 3


@pytest.mark.parametrize("callback", ["load_comparison_preset", "load_route_preset"])
def test_clearing_preset_preserves_edited_controls(ui, callback):
    _, functions, _, client, _ = ui
    assert all(value == {"__type__": "update"} for value in functions[callback](None))
    client.generate.assert_not_called()


@pytest.mark.parametrize("budget", [0, 0.00000001, 0.0001, 0.005, 0.01, 1.25])
def test_cost_field_preserves_fractional_dollar_budgets(ui, budget):
    demo, _, _, _, _ = ui
    cost = next(
        component
        for component in demo.config["components"]
        if component["props"].get("label") == "Maximum estimated cost (USD/request)"
    )
    assert cost["props"]["step"] == 0.00000001
    assert cost["props"]["precision"] == 8
    assert demo.blocks[cost["id"]].preprocess(budget) == budget
    writers = [event for event in demo.config["dependencies"] if cost["id"] in event["outputs"]]
    assert len(writers) == 1
    assert demo.fns[writers[0]["id"]].fn.__name__ == "load_route_preset"


@pytest.mark.asyncio
async def test_custom_cost_limit_applies_after_prompt_edits(ui):
    _, functions, _, client, _ = ui
    for prompt in ["Summarize cloud computing.", "Summarize the benefits of cloud computing."]:
        for budget, expected_status in [(0.005, "recommended"), (0.00000001, "no_candidate")]:
            badge, rows, _ = await functions["smart_route"](
                prompt, "summarization", "balanced", 0.5, budget, 30000, 0.8, False
            )
            assert f"**Status:** {expected_status}" in badge
            if expected_status == "no_candidate":
                assert all("max_cost" in row[-1] for row in rows)
    client.generate.assert_not_called()


@pytest.mark.asyncio
async def test_missing_golden_dataset_has_actionable_profile_error(ui):
    _, f, evaluation, _, _ = ui
    original = evaluation.profiles_path.read_bytes()
    evaluation.tasks_path.unlink()
    rows, status, _ = await f["evaluate_saved"]("summarization")
    assert rows and "Benchmark tasks file not found" in status
    assert evaluation.profiles_path.read_bytes() == original


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "raw",
    [
        "{}",
        "[null]",
        "[]",
        '[{"task_id":"missing_fields"}]',
        '[{"task_id":"a","task_type":"unknown","prompt":"Hello"}]',
        '[{"task_id":"a","task_type":"summarization","prompt":"  "}]',
        json.dumps([
            dict(task_id="a", task_type="summarization", prompt="Summarize", reference="A")
        ] * 2),
    ],
)
async def test_malformed_dataset_preserves_profiles_and_allows_routing(ui, raw):
    _, functions, evaluation, client, _ = ui
    original = evaluation.profiles_path.read_bytes()
    evaluation.tasks_path.write_text(raw)
    rows, status, _ = await functions["evaluate_saved"]("summarization")
    assert rows and "Evaluation could not complete" in status
    assert evaluation.profiles_path.read_bytes() == original
    badge, candidates, live = await functions["smart_route"](
        "Summarize cloud computing.", "auto", "balanced", 0.5, 0.005, 30000, 0.8, False
    )
    assert "**Status:** recommended" in badge
    assert "Detailed benchmark evidence is unavailable" in badge
    assert len(candidates) == 2 and "Recommendation only" in live
    client.generate.assert_not_called()
