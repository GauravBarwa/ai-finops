"""Quality, percentiles and profile persistence without provider calls."""

import json

import pytest

from app.evaluation.metrics import extraction_score, parse_json_object, score_output
from app.schemas.telemetry import TelemetryRecord
from app.services.benchmark_service import BenchmarkTask
from app.services.evaluation_service import EvaluationService
from app.services.statistics import latency_summary


def test_classification():
    assert score_output("classification", " Positive.\n", "POSITIVE") == 1
    assert score_output("classification", "not positive", "POSITIVE") == 0


def test_wrapped_nested_json():
    ref = '{"name":"Ada", "data":{"year":2025}}'
    assert extraction_score(ref, ref) == 1
    assert extraction_score(f"Here it is:\n```json\n{ref}\n```", ref) == 0.95
    assert extraction_score('{"name":"Ada"}', ref) == 0.5
    assert extraction_score("invalid", ref) == 0
    assert parse_json_object('{"name":"a}b"}')[0] == {"name": "a}b"}


def test_reference_constraints_and_reasoning():
    ref = "Cloud costs fell. Reliability stayed high."
    assert score_output("summarization", ref, ref, prompt="Use 2 sentences") == 1
    assert score_output("summarization", ref, ref, prompt="Use 3 sentences") == 0.9
    assert score_output("reasoning", "The cost is 80", "The cost is 90") == 0
    assert score_output("reasoning", "", "90") == 0


def test_percentiles_include_cold_start():
    stats = latency_summary([29600, 254, 300])
    assert stats["median_latency_ms"] == 300
    assert stats["p95_latency_ms"] == pytest.approx(26670)
    assert latency_summary([])["p95_latency_ms"] == 0
    assert latency_summary([42])["p95_latency_ms"] == 42


def test_profiles_failures_and_persistence(tmp_path):
    tasks_path = tmp_path / "tasks.json"
    task = dict(task_id="a", task_type="classification", prompt="label?", reference="YES")
    tasks_path.write_text(json.dumps([task]))
    records = [
        TelemetryRecord(
            task_id="a",
            model_id="m",
            provider="test",
            task_type="classification",
            prompt="label?",
            response="YES",
            latency_ms=latency,
            estimated_cost=0.001,
            success=success,
            source="benchmark",
        )
        for latency, success in [(100, True), (1000, False)]
    ]
    (tmp_path / "benchmark_a.jsonl").write_text("\n".join(r.model_dump_json() for r in records))
    service = EvaluationService(tmp_path, tasks_path)
    result = service.evaluate()
    profile = result.profiles[0]
    assert profile.quality_score == 0.5
    assert profile.reliability_rate == 0.5
    assert profile.sample_count == 2
    assert profile.median_latency_ms == 550
    assert profile.avg_estimated_cost == 0.001
    assert service.load_profiles() == result
    assert service.evaluate().profiles == result.profiles  # rebuild, no double counting
    with pytest.raises(ValueError, match="basenames"):
        service.evaluate(["../benchmark_a.jsonl"])
    records[0].prompt = "changed"
    with pytest.raises(ValueError, match="golden"):
        service.aggregate(records, [BenchmarkTask(**task)])


def test_no_profiles_without_data(tmp_path):
    service = EvaluationService(tmp_path)
    assert not service.load_profiles().profiles
    with pytest.raises(ValueError, match="No benchmark"):
        service.evaluate()
