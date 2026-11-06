"""Imported verification must feed normal evaluation without fabricating or double-counting data."""

import json

import pytest

from app.llm.registry import ModelInfo, ModelRegistry
from app.schemas.comparison import CompareResponse, ModelResult
from app.schemas.telemetry import TelemetryRecord
from app.services.evaluation_service import EvaluationService
from app.services.profile_publication import publish_verification


@pytest.fixture
def publication(tmp_path):
    tasks = [
        {
            "task_id": "a",
            "task_type": "summarization",
            "prompt": "Summarize",
            "reference": "Cloud costs fell.",
        }
    ]
    task_path = tmp_path / "tasks.json"
    task_path.write_text(json.dumps(tasks))
    registry = ModelRegistry(
        [
            ModelInfo(
                model_id=mid,
                provider="test",
                display_name=mid,
                litellm_model_name="test/" + mid,
                context_window=4096,
                input_price_per_1m_tokens=0.1,
                output_price_per_1m_tokens=0.2,
                capabilities=["summarization"],
            )
            for mid in ["local", "cloud"]
        ]
    )
    old = TelemetryRecord(
        task_id="a",
        task_type="summarization",
        prompt="Summarize",
        response="Cloud costs fell.",
        model_id="local",
        provider="test",
        latency_ms=100,
        input_tokens=10,
        output_tokens=5,
        source="benchmark",
    )
    (tmp_path / "benchmark_old.jsonl").write_text(old.model_dump_json())
    evaluation = EvaluationService(tmp_path, task_path, registry)
    evaluation.evaluate()
    response = CompareResponse(
        task_id="runtime_uuid",
        prompt="Summarize",
        task_type="summarization",
        total_models=2,
        successful=1,
        failed=1,
        results=[
            ModelResult(
                model_id="local",
                provider="test",
                display_name="Local",
                response="Cloud costs fell.",
                latency_ms=80,
                input_tokens=10,
                output_tokens=5,
            ),
            ModelResult(
                model_id="cloud",
                provider="test",
                display_name="Cloud",
                success=False,
                error="503 overloaded",
                latency_ms=1000,
            ),
        ],
    )
    entry = response.model_dump(mode="json") | {"benchmark_task_id": "a"}
    # Imported scores must be computed from raw outcomes rather than trusting this field.
    entry["quality_scores"] = {"cloud": 1.0}
    report = {"finished_at": response.timestamp.isoformat(), "comparisons": [entry]}
    report_path = tmp_path / "report.json"
    report_path.write_text(json.dumps(report))
    return evaluation, report_path, report


def test_publish_is_idempotent_and_survives_regular_ui_evaluation(publication):
    evaluation, path, _ = publication
    snapshot = publish_verification(path, evaluation)
    assert len(snapshot.profiles) == 2
    cloud = next(p for p in snapshot.profiles if p.model_id == "cloud")
    assert cloud.quality_score == cloud.reliability_rate == 0
    assert cloud.sample_count == 1
    local = next(p for p in snapshot.profiles if p.model_id == "local")
    assert local.sample_count == 2 and local.quality_score == 1
    assert len(snapshot.source_files) == 2
    assert publish_verification(path, evaluation).profiles == snapshot.profiles
    assert evaluation.evaluate().profiles == snapshot.profiles
    assert len(list(evaluation.results_dir.glob("benchmark_verified_*.jsonl"))) == 1


@pytest.mark.parametrize("corruption", ["unfinished", "wrong_prompt", "duplicate", "incomplete"])
def test_bad_import_cannot_replace_last_good_profiles(publication, corruption):
    evaluation, path, report = publication
    original = evaluation.profiles_path.read_bytes()
    if corruption == "unfinished":
        report.pop("finished_at")
    elif corruption == "wrong_prompt":
        report["comparisons"][0]["prompt"] = "Edited task"
    elif corruption == "duplicate":
        report["comparisons"].append(report["comparisons"][0])
    else:
        report["comparisons"][0]["total_models"] = 99
    path.write_text(json.dumps(report))
    with pytest.raises(ValueError):
        publish_verification(path, evaluation)
    assert evaluation.profiles_path.read_bytes() == original
    assert not list(evaluation.results_dir.glob("benchmark_verified_*.jsonl"))


def test_unknown_run_model_gives_actionable_error(publication):
    evaluation, _, _ = publication
    record = TelemetryRecord(
        task_id="a",
        task_type="summarization",
        prompt="Summarize",
        model_id="removed-model",
        provider="test",
        source="benchmark",
    )
    (evaluation.results_dir / "benchmark_bad.jsonl").write_text(record.model_dump_json())
    original = evaluation.profiles_path.read_bytes()
    with pytest.raises(ValueError, match="unregistered model"):
        evaluation.evaluate()
    assert evaluation.profiles_path.read_bytes() == original
