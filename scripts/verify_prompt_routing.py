"""Prove prompt-aware behavior through Gradio HTTP, without new offline benchmarks.

python -m scripts.verify_prompt_routing --live
Live option executes four small prompts; recommendation checks never call providers.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

from gradio_client import Client

from app.core.config import get_settings
from app.demo import build_demo
from app.services.benchmark_service import BenchmarkService
from app.ui import LAUNCH_OPTIONS
from scripts.verify_ui import TASK_PROMPTS


def verify(args):
    output = args.output or get_settings().results_path / "prompt_routing_verification.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    demo = build_demo()
    proof = []
    try:
        demo.launch(
            server_name="127.0.0.1", server_port=args.port, prevent_thread_lock=True, quiet=True,
            **LAUNCH_OPTIONS,
        )
        client = Client(f"http://127.0.0.1:{args.port}/", verbose=False)

        def check(
            name,
            prompt,
            *,
            mode="prompt_aware",
            task="auto",
            execute=False,
            quality=0.5,
            max_tokens=1024,
            cost=0.01,
            reliability=0.8,
        ):
            job = client.submit(
                prompt,
                task,
                "quality_first",
                quality,
                cost,
                180000,
                reliability,
                execute,
                max_tokens,
                mode,
                api_name="/smart_route",
            )
            result = job.result(timeout=150)
            proof.append({"case": name, "result": result, "passed": False})
            output.write_text(json.dumps(proof, indent=2))
            return result

        def passed():
            proof[-1]["passed"] = True
            output.write_text(json.dumps(proof, indent=2))
            print("PASS", proof[-1]["case"], flush=True)

        summaries = [t for t in BenchmarkService.load_tasks() if t.task_type == "summarization"]
        chosen = {"task_average": [], "prompt_aware": []}
        for mode in chosen:
            for task in summaries:
                result = check(f"{mode}_{task.task_id}", task.prompt, mode=mode)
                assert "**Status:** recommended" in result[0]
                assert "**Task:** summarization (auto)" in result[0]
                chosen[mode].append(re.search(r"Selected model:\*\* (\S+)", result[0])[1])
                if mode == "prompt_aware":
                    assert task.task_id in result[0]
                passed()
        assert len(set(chosen["task_average"])) == 1
        assert len(set(chosen["prompt_aware"])) > 1
        print("Same task, policy and constraints:", json.dumps(chosen), flush=True)
        for task, prompt in TASK_PROMPTS.items():
            result = check("detect_" + task, prompt)
            assert f"**Task:** {task} (auto)" in result[0]
            assert "**Status:** recommended" in result[0]
            passed()
        short = check("short_unknown_topic", "Summarize: superconducting vortices", quality=0)
        assert "No close benchmark topic match" in short[0]
        passed()
        long = check(
            "long_same_topic", "Summarize: " + "superconducting vortices " * 100, quality=0
        )
        short_rows = {r[0]: r for r in short[1]["data"]}
        long_rows = {r[0]: r for r in long[1]["data"]}
        assert long_rows["groq-gpt-oss-20b"][4] > short_rows["groq-gpt-oss-20b"][4]
        assert long_rows["groq-gpt-oss-20b"][5] > short_rows["groq-gpt-oss-20b"][5]
        passed()
        small_budget = check(
            "small_output_budget", "Summarize in 2 lines: cloud servers", quality=0, max_tokens=64
        )
        passed()
        large_budget = check(
            "large_output_budget", "Summarize in 2 lines: cloud servers", quality=0, max_tokens=1024
        )

        def row(result):
            return next(r for r in result[1]["data"] if r[0] == "groq-gpt-oss-20b")

        assert row(large_budget)[4] > row(small_budget)[4]
        passed()
        result = check(
            "explicit_override_warning",
            "Summarize in 2 lines: cloud servers",
            task="classification",
            quality=0,
        )
        assert "**Task:** classification (explicit)" in result[0]
        assert "Prompt suggests summarization" in result[0]
        passed()
        result = check("ambiguous_prompt", "Hello there, help me", quality=0)
        assert "No clear English task" in result[0]
        passed()
        # Large request is never executed. Only context/cost analysis is being tested.
        result = check(
            "context_limit",
            "Summarize: " + "unfamiliartext " * 80000,
            quality=0,
            reliability=0,
            cost=10,
            max_tokens=64,
        )
        rows = {r[0]: r for r in result[1]["data"]}
        for model in (
            "llama3.2-local",
            "gpt-oss-120b-cloud",
            "groq-gpt-oss-20b",
            "groq-gpt-oss-120b",
        ):
            assert "context_window: failed" in rows[model][-1]
        passed()
        if args.live:
            for task, prompt in TASK_PROMPTS.items():
                # Use balanced for live calls; quality-first probes above are estimates only.
                job = client.submit(
                    prompt,
                    "auto",
                    "balanced",
                    0.5,
                    0.01,
                    30000,
                    0.8,
                    True,
                    1024,
                    "prompt_aware",
                    api_name="/smart_route",
                )
                result = job.result(timeout=150)
                proof.append({"case": "live_auto_" + task, "passed": False, "result": result})
                assert "**Status:** succeeded" in result[0] and result[2].strip()
                assert f"**Task:** {task} (auto)" in result[0]
                passed()
        print(f"{len(proof)} prompt-routing cases passed. Proof: {output}", flush=True)
    finally:
        output.write_text(json.dumps(proof, indent=2))
        demo.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--live", action="store_true")
    parser.add_argument("--port", type=int, default=7861)
    parser.add_argument("--output", type=Path)
    verify(parser.parse_args())


if __name__ == "__main__":
    main()
