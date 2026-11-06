# Mentor and peer demonstration guide

This walkthrough explains the current prompt-aware prototype. Model choices and numeric
values depend on the measurements loaded in the demonstration environment.

## Opening explanation

> Different AI models have different quality, speed, reliability, and cost characteristics.
> My project measures them on shared sample tasks, then selects a model for a new prompt
> using those observations and the user's requirements. It shows why candidates passed
> or failed and can execute the selected model with one fallback.

Use the delivery-service analogy if needed. A delivery must meet basic requirements,
then the customer can prioritize price, speed, or service quality among eligible options.

The project integrates existing models through a common interface. Its two main workflows
are offline measurement and runtime selection.

```text
Golden tasks -> model runs -> scoring -> saved model profiles
New prompt -> analysis -> hard requirements -> ranking -> optional execution
```

## Preparation

1. Activate the project environment and keep credentials outside the shared screen.

   ```bash
   cd ai-finops
   source .venv/bin/activate
   export GRADIO_ANALYTICS_ENABLED=False
   ```

2. Confirm access to the exact models you plan to call. For local Ollama, inspect the
   running service and installed names.

   ```bash
   ollama list
   curl http://127.0.0.1:11434/api/tags
   ```

3. Prepare profiles before the meeting. For one installed local model, run the benchmark
   and evaluate it with the API. For the full configured experiment, use the provider
   verifier with publication.

   ```bash
   python -m scripts.verify_providers --publish-profiles
   ```

   The full command makes 96 calls and can take several minutes. Retain the results
   directory so you can demonstrate recommendation-only routing during an outage.

4. Start the API in one terminal and Gradio in another activated terminal.

   ```bash
   uvicorn app.main:app --host 127.0.0.1 --port 8000
   ```

   ```bash
   python -m app.demo
   ```

   Open <http://127.0.0.1:8000/docs> and <http://127.0.0.1:7860>.

5. In Model Profiles, refresh and confirm coverage. The full historical experiment
   created 24 model/task profiles. A fresh clone needs locally generated measurements.

6. Rehearse a live response and the strict-constraint cases below. Disclose any local
   warm-up because cold-start latency can differ from later calls.

## A 12–15 minute walkthrough

### Minutes 0–2 — Problem and architecture

Use the opening explanation. Give the example of a company labeling customer reviews.
Its task requires enough quality and acceptable response time while keeping token usage
within a planning budget. Explain how offline profiles support runtime recommendations.

### Minutes 2–4 — Comparison

Open Comparison. Select the verified models explicitly. An empty model selection calls
all enabled registry entries.

Use the following prompt.

```text
Classify as POSITIVE, NEGATIVE or NEUTRAL. Reply only with the label.
Review: The product arrived on time and works perfectly. I am very happy.
```

Choose classification and click Compare Models. Point out the incremental
results, token usage, latency, shadow cost, and full response/error area.

Explain that this view captures execution measurements. Quality evaluation uses golden
reference tasks in the separate offline workflow.

### Minutes 4–6 — Model Profiles

Refresh the profile table. Explain that a profile belongs to one model and one task type.
Show quality, median latency, p95, average estimated cost, reliability, and sample count.
Use the loaded values in your explanation and mention the sample size.

Explain the distinction between operational call success and reference-answer quality.
A successful call can still produce a low-scoring answer. Re-evaluation includes failed
calls in reliability and assigns them zero quality.

### Minutes 6–10 — Prompt-aware selection

Open Smart Router. Use `auto` task detection and `Prompt-aware estimates`. Start with
quality 0.50, cost 0.01 USD, latency 30000ms, and reliability 0.80. Keep execution off
while demonstrating the decision process.

Use this prompt.

```text
Summarize in 2 lines: An LLM Engineer builds, deploys, and maintains applications
powered by existing language models. The role connects models to business software,
data, and workflows while handling reliability and deployment requirements.
```

Click Route request. Explain the detected/effective task, input-size estimate, output
budget, related benchmark IDs, and uncertainty notes. Inspect the candidate table's
quality basis, expected cost, budget cost, latency estimate, passing checks, and rank.

Then keep the settings fixed and try a different topic.

```text
Summarize in 2 lines: Quantum computers use qubits and superposition. Potential
applications include cryptography and drug discovery. Practical systems still face
qubit stability and error-correction challenges.
```

Point out changed benchmark matches and estimates. The winning model can change when
those estimates alter eligibility or utility. A model can also remain preferred for
both requests when it continues to score highest.

To demonstrate the historical baseline, choose `Historical task averages (baseline)`.
Compare the candidate metrics across the two prompts. Return to prompt-aware mode
before continuing.

Change the policy between cost-first and latency-first, then try a zero cost limit or
zero latency limit. Show the explicit rejection reasons and the no-candidate explanation.
A higher quality requirement can also leave the available set empty.

### Minutes 10–11 — One live routed answer

Restore the prepared demo limits and enable Execute selected model. Submit the short
summarization prompt. Show the selected model, executed model, and generated answer.
If the primary fails, inspect the fallback explanation and attempt details in Swagger.

For a dependable fallback demonstration, run the regression case with simulated provider
failure and identify it as a test.

```bash
pytest -q tests/test_routing.py -k 'fallback or all_fail'
```

### Minutes 11–13 — Business evidence

Open the saved KPI report. Explain its fixed GPT-OSS baseline and historical cost-first
replay. It records 95.96% lower shadow estimated cost and 108.93% reference-quality
retention across nine observed tasks. Median task latency increased from roughly
0.84 seconds to 11.97 seconds.

Suggested explanation follows.

> In this small historical experiment, the cost-first policy reduced shadow token cost
> while retaining the reference-based quality score. The latency increased considerably.
> An interactive application would need tighter latency requirements or a different
> preference policy for this workload.

Explain that quality retained is a ratio against the baseline score, which can exceed
100%. Replaying a new experiment can produce different values. The saved report uses
the task-average baseline and records its coverage and assumptions.
[KPI report](kpi_report.md)

### Minutes 13–15 — Engineering and next steps

Show `app/llm`, `app/domain`, `app/services`, and `tests` briefly. Explain the common adapter
contract, deterministic selection, schema validation, stored evidence, and regression tests.

The next validation work is a larger independent dataset, stronger semantic evaluation,
provider availability management, and a deployment design matching the mentor's use case.

## Explain the metrics

- A token is a unit of text accounting. Input tokens describe the prompt and output
  tokens describe generated content, with provider-specific accounting rules.
- Shadow cost applies configured USD input/output rates per million tokens. For
  1,000 input and 200 output tokens, the configured local Llama rates yield $0.000072.
- Median latency describes the middle observed call. p95 interpolates the slow end
  of the recorded sample and is sensitive to small sample counts.
- Quality reflects the deterministic scorer. Its interpretation depends on task type
  and reference coverage.
- Reliability reflects the fraction of calls the adapters reported as successful.
- Prompt-aware estimates use the current text and benchmark evidence. Actual output
  quality and elapsed time require execution and evaluation.

## Likely questions

### How does the router use my prompt?

It detects English instruction intent, estimates size/output/context requirements,
retrieves lexically related benchmark examples, and conditions candidate estimates
on those observations. The response exposes the evidence and fallback behavior.

### Does every different prompt choose a different model?

The same model can remain optimal. What changes is the analysis, applicable evidence,
estimates, and potentially the eligible set or final ranking.

### How does it know answer quality before generation?

It predicts quality from related measured outcomes and task priors. Correctness checks
of the actual new response need generated output and a suitable evaluation method.

### Why use shadow rates with free/local models?

They provide a cost axis for testing selection while using the intended free-tier/local
workflow. Real account billing and local infrastructure cost require separate records.

### Why can an enabled model be rejected?

Its task capability, profile evidence, context capacity, or a requested performance
requirement can fail. Enabled status describes registry configuration.

### What happens during a provider outage?

Live execution can try the second-ranked eligible model once. The response reports
attempts and their errors when the available execution path fails.

### What would be needed for production?

The local MVP needs an authentication and deployment design, approved telemetry handling,
more representative evaluations, profile freshness controls, and stronger storage
coordination for multi-process operation.

## Recovery during the meeting

- For an empty profile table, check `RESULTS_DIR`, collect measurements, and evaluate them.
- For no-candidate results, inspect the measured/predicted values and requirements.
- For 429 or 503 responses, show the provider outcome and use prepared recommendation
  examples while access or quota recovers.
- For stale controls, restart the Gradio process after code/configuration updates.
- For internet failure, use saved profiles and the recommendation-only examples.
- For a short meeting, show profiles, one prompt-aware decision, a changed constraint,
  and the KPI tradeoff in five minutes.

The configuration guide includes commands for all preparation and troubleshooting steps.
[Configuration](configuration.md)
