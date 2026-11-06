# Routing and evaluation methodology

AI FinOps uses offline measurements to make a pre-execution selection. Prompt-aware
routing conditions those measurements on the incoming text, while the `task_average`
mode provides a historical-performance baseline.

## Prompt analysis

`RouteRequest.task_type` defaults to `auto`. The analyzer examines the instruction
prefix for English cues associated with summarization, classification, extraction,
and reasoning. It stops at explicit data boundaries such as `Text:` or `Review:`.
The first detected instruction determines the automatic type. Multiple signals and
an uncertain broad reasoning fallback appear in the response warnings.

An explicit task type takes precedence. If the detected instruction suggests another
type, the response records the mismatch so the caller can verify the override.

The initial token estimate takes the larger of UTF-8 byte length divided by three
and the lexeme count multiplied by 1.3, then rounds upward. Where measured usage is
available, each model's input estimate is calibrated by the median ratio of observed
input tokens to the same rough estimator on benchmark prompts.

Word, sentence, and line requests supply output hints. The current heuristic allows
1.5 tokens per requested word and 32 tokens per sentence/line, bounded by `max_tokens`.
Classification instructions requesting only a label use a 16-token hint. Expected
output otherwise uses successful benchmark output usage or the requested budget.

These rules provide explainable estimates with limited language and instruction
coverage. Exact provider tokenization, formatting, hidden reasoning usage, and actual
response length remain execution-dependent.

## Related benchmark evidence

The assessment service reads the source files associated with the profile snapshot
and validates their golden task identities. For the effective task type, it builds
TF-IDF vectors from source/question content. Common instructions, label scaffolding,
stop words, and numeric-only terms are removed during lexical matching.

Cosine similarity selects up to three tasks with similarity at least 0.20. Retained
matches must also reach 80% of the best match's similarity. Matching task IDs and their
similarities appear in `prompt_analysis.matched_tasks`.

For each model, outcomes within a matching task are averaged before applying task
similarity weights. Quality and reliability predictions blend those neighbor outcomes
with the historical model/task prior.

```text
blend = highest retained lexical similarity
predicted_quality = blend * neighbor_quality + (1 - blend) * task_quality
predicted_reliability = blend * neighbor_reliability + (1 - blend) * task_reliability
```

A weak or absent lexical match leads to a visible task-prior fallback. If raw evidence
is unavailable, the response explains the missing source and retains the historical
performance prior. Available request text still supports cost and context estimation.

Similarity reflects word overlap in this small dataset. Quality predictions inherit
the benchmark scorer's limitations. Correctness assessment of a generated answer needs
observed output and a suitable evaluation method.

## Request-specific cost and latency

Every candidate estimate keeps two shadow cost values.

- `estimated_cost` uses estimated input and expected output tokens for ranking.
- `budget_estimated_cost` uses estimated input and the maximum allowed output tokens
  for the cost constraint.

```text
shadow_cost = input_tokens / 1,000,000 * input_rate
            + output_tokens / 1,000,000 * output_rate
```

Model rates are fixed experimental USD assumptions in the registry. The arithmetic
also prices local/free-tier usage to make cost tradeoffs observable. Local hardware,
energy, and account billing require separate accounting.

Latency begins with the median of the applicable observed calls. Where successful
usage observations exist, it scales by the ratio of requested estimated total tokens
to average observed total tokens. Missing usage retains the historical median.
This linear heuristic approximates workload size while provider load, cold starts,
network conditions, and fixed overhead can change actual elapsed time.

`ModelProfile` preserves measured task statistics. `CandidateEstimate` separately
records request predictions, token estimates, evidence sample counts, matched tasks,
and quality basis. This separation lets clients inspect the prediction and its prior.

## Stage 1 requirements

Candidates must satisfy enabled status, declared task support, profile availability,
context capacity, minimum quality, cost limit, latency limit, and minimum reliability.
Context checks reserve estimated input tokens plus the maximum output budget.

Prompt-aware mode applies its candidate estimates. The historical mode applies task
quality, mean cost, median latency, and observed reliability. Requirements are checked
before policy ranking. Each failed check carries an explicit rejection reason.

A structurally valid request can return `no_candidate` when its requirements exceed
the available evidence. The response identifies the limiting values and possible
recovery through adjusted requirements or stronger measurements.

## Stage 2 policies

Passing candidates receive min-max utility values for quality, cost, latency, and
reliability. Higher quality/reliability and lower cost/latency receive higher utility.
A constant metric assigns equal utility to all passing candidates. The weighted sum
is divided by the total weight. Model ID resolves exact final score ties.

- `cost_first` uses cost 0.65, quality 0.20, latency 0.10, and reliability 0.05.
- `balanced` uses 0.25 for every metric.
- `quality_first` uses quality 0.65, reliability 0.15, cost 0.10, and latency 0.10.
- `latency_first` uses latency 0.65, quality 0.20, cost 0.10, and reliability 0.05.

Custom API weights override the preset and must have a positive total. These policies
combine preferences, so the same model can remain preferred across several prompts or
policies when the eligible set and measurements favor it.

## Execution

Recommendation mode performs local analysis and returns estimates. With execution
enabled, the service calls the primary model, then tries the second-ranked eligible
model once after a primary failure. Both attempts retain telemetry, and the response
sums their observed estimated costs. Budget filtering applies per candidate attempt.

A candidate's historical reliability describes the observed experiment. Current account
quotas and service availability are determined during actual calls and can change the
execution outcome.

## Offline quality scoring

The golden dataset contains four tasks per type and 16 tasks in total.

- Classification uses normalized label equality.
- Extraction parses strict JSON first and then objects within wrappers. It compares
  expected key/value pairs, reduces precision for additional fields, and multiplies
  wrapped-format accuracy by 0.95 when strict JSON is requested.
- Summarization and reasoning use reference-word coverage and precision. Supported
  sentence/word constraints apply a small penalty for length/format disagreement.
- Reasoning also requires the reference numbers to appear in the output.
- Failed calls receive zero quality.

Reference overlap favors particular wording. An optional judge callback exists in the
metric function, and the current service uses the deterministic scorer. Broader semantic
assessment requires additional rubrics, larger datasets, and independent validation.

## Statistical profiles and KPI replay

Evaluation records mean quality, mean shadow cost, adapter success fraction, sample
count, median latency, and p95 latency. Percentile interpolation uses the sorted sample
position `(sample_count - 1) * 0.95`. Cold starts and failed calls remain in the data.
Small sample p95 and reliability estimates have substantial uncertainty.

The KPI replay holds one task out, builds profiles from the remaining tasks, and compares
observed selected-model and fixed-baseline outcomes on the held-out task. Its current
implementation explicitly uses `task_average`. The report includes paired coverage,
cost reduction, retained reference quality, and median latency change. The saved historical
results and sources are recorded separately. [KPI report](kpi_report.md)

## Verification observations

On 5 November 2026, Gradio HTTP checks held the task type, `quality_first` policy,
and constraints fixed across four stored summarization prompts. Historical mode chose
Groq GPT-OSS 20B for all four. Prompt-aware mode chose Gemini 3.5 Flash-Lite for the AI
passage, Gemini 3.6 Flash for the cloud/quantum passages, and local Llama for the short
server-cost passage.

These known examples demonstrate content-conditioned selection. The regression suite
also uses two models with equal global quality but different observed topic strengths.
Larger held-out evaluation is needed to assess generalization to new workloads.

Reproduce the focused prepared-experiment checks after collecting the full model set.

```bash
GRADIO_ANALYTICS_ENABLED=False python -m scripts.verify_prompt_routing
# Also execute four automatically detected task types
GRADIO_ANALYTICS_ENABLED=False python -m scripts.verify_prompt_routing --live
```

The raw result is written to the local `data/results/prompt_routing_verification.json`.
The verification report records the historical outcomes and publication checks.
[UI verification](ui_verification.md)
