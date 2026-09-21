"""Minimal Gradio demo for multi-model comparison.

Launch with::

    python -m app.demo

Or alongside the FastAPI server::

    # Terminal 1: uvicorn app.main:app --reload
    # Terminal 2: python -m app.demo
"""

from __future__ import annotations

import asyncio
from typing import Any

import gradio as gr

from app.core.config import get_settings
from app.core.logging import setup_logging
from app.llm.litellm_client import LiteLLMClient
from app.llm.registry import ModelRegistry
from app.schemas.comparison import CompareRequest
from app.services.comparison_service import ComparisonService


def build_demo() -> gr.Blocks:
    """Build and return the Gradio Blocks interface."""
    setup_logging()
    settings = get_settings()

    registry = ModelRegistry.from_config()
    client = LiteLLMClient()
    service = ComparisonService(llm_client=client, registry=registry)

    enabled_models = registry.list_enabled()
    model_choices = [(m.display_name, m.model_id) for m in enabled_models]

    async def _compare(
        prompt: str,
        task_type: str,
        selected_models: list[str],
    ) -> tuple[str, str]:
        """Run comparison and format results."""
        if not prompt.strip():
            return "⚠️ Please enter a prompt.", ""

        models = selected_models if selected_models else None
        task = task_type if task_type != "auto" else None

        request = CompareRequest(
            prompt=prompt,
            task_type=task,
            models=models,
        )

        response = await service.compare(request)

        # Build results table
        headers = [
            "Model", "Provider", "Tokens (in/out)", "Latency",
            "Est. Cost", "Status",
        ]
        rows: list[list[str]] = []
        details: list[str] = []

        for r in response.results:
            rows.append([
                r.display_name,
                r.provider,
                f"{r.input_tokens}/{r.output_tokens}",
                f"{r.latency_ms:.0f}ms",
                f"${r.estimated_cost:.6f}",
                "✅" if r.success else f"❌ {r.error}",
            ])

            # Full response details
            status_icon = "✅" if r.success else "❌"
            details.append(
                f"### {status_icon} {r.display_name} ({r.provider})\n"
                f"**Tokens:** {r.input_tokens} in / {r.output_tokens} out "
                f"({r.total_tokens} total)  \n"
                f"**Latency:** {r.latency_ms:.0f}ms  \n"
                f"**Est. Cost:** ${r.estimated_cost:.6f}  \n\n"
                f"**Response:**\n\n{r.response if r.success else r.error}\n\n---\n"
            )

        # Summary markdown
        summary = (
            f"**Task ID:** `{response.task_id}`  \n"
            f"**Models:** {response.successful} succeeded, "
            f"{response.failed} failed out of {response.total_models}\n\n"
        )

        # Table markdown
        table_md = "| " + " | ".join(headers) + " |\n"
        table_md += "| " + " | ".join(["---"] * len(headers)) + " |\n"
        for row in rows:
            table_md += "| " + " | ".join(row) + " |\n"

        summary_md = summary + table_md
        detail_md = "\n".join(details)

        return summary_md, detail_md

    def compare_sync(prompt: str, task_type: str, selected_models: list[str]):
        """Sync wrapper for the async compare function."""
        return asyncio.run(_compare(prompt, task_type, selected_models))

    # ── Build the UI ─────────────────────────────────────────────────────
    with gr.Blocks(
        title="AI FinOps — Model Comparison",
        theme=gr.themes.Soft(),
    ) as demo:
        gr.Markdown(
            "# 🔬 AI FinOps — Multi-Model Comparison\n"
            "Compare the same prompt across multiple LLMs. "
            "See tokens, latency, and estimated cost side-by-side."
        )

        with gr.Row():
            with gr.Column(scale=2):
                prompt_input = gr.Textbox(
                    label="Prompt",
                    placeholder="Enter your prompt here...",
                    lines=5,
                )
                task_type_input = gr.Dropdown(
                    label="Task Type",
                    choices=["auto", "summarization", "classification", "extraction"],
                    value="auto",
                )
                model_selector = gr.CheckboxGroup(
                    label="Models (leave empty for all enabled)",
                    choices=model_choices,
                )
                compare_btn = gr.Button("🚀 Compare Models", variant="primary")

            with gr.Column(scale=3):
                summary_output = gr.Markdown(label="Summary")
                detail_output = gr.Markdown(label="Detailed Results")

        compare_btn.click(
            fn=compare_sync,
            inputs=[prompt_input, task_type_input, model_selector],
            outputs=[summary_output, detail_output],
        )

        # Example prompts
        gr.Examples(
            examples=[
                [
                    "Summarize in 2 sentences: Cloud computing allows businesses to rent "
                    "computing resources on demand instead of maintaining expensive servers. "
                    "This pay-as-you-go model has enabled startups to launch products without "
                    "massive capital investment.",
                    "summarization",
                    [],
                ],
                [
                    "Classify the sentiment as POSITIVE, NEGATIVE, or NEUTRAL:\n\n"
                    "\"Great product, fast shipping, exactly as described!\"",
                    "classification",
                    [],
                ],
            ],
            inputs=[prompt_input, task_type_input, model_selector],
        )

    return demo


if __name__ == "__main__":
    demo = build_demo()
    demo.launch(server_name="0.0.0.0", server_port=7860)
