"""Exercise every Gradio tab through its real HTTP API in one repeated client session.

python -m scripts.verify_ui --live
Starts a temporary server on 127.0.0.1:7861 and closes it afterward. --live makes
provider calls; other cases use profiles, evaluation and validation only. Profile
re-evaluation uses the active benchmark files, exactly like clicking the UI button.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from gradio_client import Client

from app.core.config import get_settings
from app.demo import build_demo
from app.ui import LAUNCH_OPTIONS

USER_PROMPT = (
    "Summarize in 2 lines: An LLM (Large Language Model) Engineer is a specialized "
    "software and AI professional who builds, deploys, and maintains production-ready "
    "applications powered by language models like GPT, Claude, or LLaMA.\n"
    "Unlike traditional Machine Learning (ML) engineers—who train complex neural networks "
    "from scratch—LLM Engineers typically do not build foundation models. Instead, they "
    "act as bridge-builders who take existing models and connect them reliably to "
    "real-world business software, data, and workflows"
)
TASK_PROMPTS = {
    "summarization": USER_PROMPT,
    "classification": "Classify as POSITIVE, NEGATIVE or NEUTRAL. Reply only with the label. "
    "Review: Great product, fast shipping!",
    "extraction": "Return only JSON with keys company and year. Text: Acme launched in 2025.",
    "reasoning": "Each request uses 200 input tokens and 100 output tokens. "
    "How many total tokens do 10 requests use? Explain briefly.",
}


def verify(args):
    proof = []
    demo = build_demo()
    output = args.output or get_settings().results_path / "full_ui_verification.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    try:
        demo.launch(
            server_name="127.0.0.1", server_port=args.port, prevent_thread_lock=True, quiet=True,
            **LAUNCH_OPTIONS,
        )
        client = Client(f"http://127.0.0.1:{args.port}/", verbose=False)

        def check(name, api_name, arguments, assertion):
            job = client.submit(*arguments, api_name=api_name)
            result = job.result(timeout=150)
            assertion(result)
            entry = {
                "case": name,
                "api_name": api_name,
                "updates": len(job.outputs()),
                "passed": True,
                "result": result,
            }
            proof.append(entry)
            output.write_text(json.dumps(proof, indent=2))
            print(f"PASS {name}", flush=True)
            return result

        def require(condition, message):
            if not condition:
                raise AssertionError(message)

        for task in TASK_PROMPTS:
            check(
                "profiles_hint_" + task,
                "/task_profile_hint",
                [task],
                lambda r: require("Best measured quality" in r, "Missing task profiles"),
            )
        refreshed = check(
            "profiles_refresh_before",
            "/refresh_profiles",
            ["summarization"],
            lambda r: require(len(r[0]["data"]) >= 24, "Missing active profiles"),
        )
        count = len(refreshed[0]["data"])
        check(
            "profiles_evaluate",
            "/evaluate_saved",
            ["summarization"],
            lambda r: require(len(r[0]["data"]) == count, "Evaluation lost model coverage"),
        )
        check(
            "profiles_refresh_after",
            "/refresh_profiles",
            ["reasoning"],
            lambda r: require(len(r[0]["data"]) == count, "Refresh lost model coverage"),
        )
        for task, prompt in TASK_PROMPTS.items():
            for policy in ["cost_first", "balanced", "quality_first", "latency_first"]:
                check(
                    f"recommend_{task}_{policy}",
                    "/smart_route",
                    [prompt, task, policy, 0.5, 0.01, 30000, 0.8, False, 1024],
                    lambda r: require(
                        "**Status:** recommended" in r[0]
                        and len(r[1]["data"]) == 9
                        and "Recommendation only" in r[2],
                        "Recommendation failed",
                    ),
                )
        for name, quality, cost, latency in [
            ("strict_quality", 1, 0.01, 30000),
            ("strict_cost", 0.5, 0, 30000),
            ("strict_latency", 0.5, 0.01, 0),
        ]:
            check(
                name,
                "/smart_route",
                [USER_PROMPT, "summarization", "balanced", quality, cost, latency, 0.8, True, 1024],
                lambda r: require(
                    "no_candidate" in r[0] and "No model satisfies" in r[2],
                    "Strict constraints were bypassed",
                ),
            )
        check(
            "summary_original_quality_070",
            "/smart_route",
            [USER_PROMPT, "summarization", "balanced", 0.7, 0.01, 30000, 0.8, False, 1024],
            lambda r: require(
                "no_candidate" in r[0] and "Best estimated summarization quality" in r[0],
                "Missing actionable rejection",
            ),
        )
        check(
            "router_empty_prompt",
            "/smart_route",
            ["   ", "summarization", "balanced", 0.5, 0.01, 30000, 0.8, True, 1024],
            lambda r: require(
                "invalid_request" in r[0] and not r[1]["data"],
                "Invalid router input retained old output",
            ),
        )
        check(
            "compare_empty_prompt",
            "/_compare",
            ["", "auto", [], 1024],
            lambda r: require(
                "Please enter a prompt" in r[0] and not r[1]["data"],
                "Missing comparison input validation",
            ),
        )
        # Exercises example selection, including the token limit that was missing previously.
        if any(fn.fn and fn.fn.__name__ == "load_example" for fn in demo.fns.values()):
            for index in range(2):
                check(
                    f"comparison_example_{index}",
                    "/load_example",
                    [index],
                    lambda r: require(
                        len(r) == 4
                        and (r[-1].get("value") if isinstance(r[-1], dict) else r[-1]) == 1024,
                        "Example failed to populate all controls",
                    ),
                )
        if args.live:
            check(
                "compare_all_live",
                "/_compare",
                [TASK_PROMPTS["classification"], "classification", [], 1024],
                lambda r: require(
                    "0 pending out of 6" in r[0] and len(r[1]["data"]) == 6,
                    "Comparison never completed",
                ),
            )
            check(
                "compare_repeat_selected_live",
                "/_compare",
                [
                    TASK_PROMPTS["classification"],
                    "classification",
                    ["llama3.2-local", "gpt-oss-120b-cloud"],
                    1024,
                ],
                lambda r: require("2 succeeded, 0 failed" in r[0], "Repeated comparison failed"),
            )
            for task, prompt in TASK_PROMPTS.items():
                check(
                    "execute_" + task,
                    "/smart_route",
                    [prompt, task, "balanced", 0.5, 0.01, 30000, 0.8, True, 1024],
                    lambda r: require(
                        "**Status:** succeeded" in r[0] and bool(r[2].strip()),
                        "Routed live execution failed",
                    ),
                )
        print(f"{len(proof)} UI cases passed. Proof: {output}", flush=True)
    except Exception as exc:
        proof.append(
            {
                "case": "verification_failure",
                "passed": False,
                "error": f"{type(exc).__name__}: {exc}",
            }
        )
        output.write_text(json.dumps(proof, indent=2))
        raise
    finally:
        demo.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--live", action="store_true")
    parser.add_argument("--port", type=int, default=7861)
    parser.add_argument("--output", type=Path)
    verify(parser.parse_args())


if __name__ == "__main__":
    main()
