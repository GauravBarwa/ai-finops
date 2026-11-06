# API reference

Start the service with Uvicorn from the repository root.

```bash
uvicorn app.main:app --host 127.0.0.1 --port 8000
```

Swagger UI is available at `/docs`, ReDoc at `/redoc`, and the generated machine-readable
contract at `/openapi.json`. Pydantic schemas define the authoritative field types and
validation rules.

## `GET /health`

Returns `status`, `service`, and `models_loaded`. A healthy response confirms application
startup and registry initialization. Provider catalog/inference checks supply separate
access evidence.

## `GET /models`

Lists enabled model metadata in a `models` array with `total`. Each entry contains
its ID, provider, display name, context window, shadow rates, capabilities, enabled
flag, and optional disabled reason.

## `GET /models/{model_id}`

Returns metadata for a known registry ID, including a disabled entry when explicitly
requested. An unknown ID returns HTTP 404. The registry holds internal provider model strings. Public metadata exposes
the stable registry identity and model characteristics.

## `POST /compare`

Executes one prompt across the selected registry IDs.

```json
{
  "prompt": "Classify as POSITIVE, NEGATIVE or NEUTRAL. Reply only with the label. Review: Excellent service!",
  "task_type": "classification",
  "models": ["llama3.2-local", "groq-gpt-oss-20b"],
  "max_tokens": 512,
  "temperature": 0
}
```

- `prompt` is a required string containing at least one non-whitespace character.
- `task_type` is an optional free-form workload label for comparison telemetry.
- `models` defaults to all enabled registry IDs when omitted, null, or empty.
- `max_tokens` defaults to 1024 and accepts integers from 1 through 8192.
- `temperature` defaults to 0 and accepts values from 0 through 2. Provider adapters
  can omit unsupported controls.

The response includes `task_id`, input context, timestamp, total/success/failure counts,
and a `results` array. Each result includes the model/provider identity, response,
input/output/total tokens, latency, estimated shadow cost, success flag, and optional
error text.

Provider failures, unknown model IDs, and disabled models become failed result entries.
Other candidates can complete, and the aggregate request returns HTTP 200. Structurally
invalid payloads return HTTP 422. The HTTP endpoint returns the final comparison.
The Gradio interface consumes incremental service snapshots for progress display.

## `POST /evaluate`

Scores saved benchmark files and atomically publishes a profile snapshot.

```json
{}
```

To choose specific runs, supply their basenames.

```json
{
  "run_files": ["benchmark_20261106_120000_000000.jsonl"]
}
```

The default searches for `benchmark_*.jsonl` under the configured results directory.
Names must identify benchmark files in that directory. Evaluation validates task ID,
task type, and exact golden prompt, and applies current registry shadow rates to
recorded token counts. It returns a `ProfilesResponse`.

Missing runs, invalid data, incompatible model metadata, or an unavailable golden
dataset produce HTTP 400 with an actionable detail. Failed evaluation preserves the
last completed profile snapshot. Repeating evaluation rebuilds the selected dataset.

## `GET /profiles`

Returns the saved `ProfilesResponse`. Its `profiles` list contains one record per
model/task pair with the following fields.

- `model_id` and `task_type` identify the measurement group.
- `quality_score` is the mean reference-scoring result in the range 0 through 1.
- `median_latency_ms` and `p95_latency_ms` include the observed calls in that group.
- `avg_estimated_cost` is mean shadow USD per call.
- `reliability_rate` is the fraction of calls whose adapter reported success.
- `sample_count` records the number of included observations.

Snapshot metadata includes scorer version, dataset hash, pricing hash, source filenames,
update timestamp, and pricing basis. An absent snapshot returns an empty profile list.
An invalid or unreadable saved snapshot produces HTTP 503 with recovery guidance.

## `POST /route`

Analyzes the request, filters candidates, ranks eligible models, and optionally executes.

```json
{
  "prompt": "Summarize in 2 lines: Cloud services provide resources on demand and let businesses scale with usage.",
  "task_type": "auto",
  "routing_mode": "prompt_aware",
  "policy": "balanced",
  "constraints": {
    "min_quality": 0.5,
    "max_cost": 0.01,
    "max_latency_ms": 30000,
    "min_reliability": 0.8
  },
  "execute": false,
  "max_tokens": 512,
  "temperature": 0
}
```

### Request fields

- `prompt` is required and must contain a non-whitespace character.
- `task_type` defaults to `auto`. Explicit values are `summarization`, `classification`,
  `extraction`, and `reasoning`.
- `routing_mode` defaults to `prompt_aware`. `task_average` selects historical
  performance metrics for a baseline.
- `policy` defaults to `balanced`. Other presets are `cost_first`, `quality_first`,
  and `latency_first`.
- `constraints` defaults to minimum quality 0, minimum reliability 0, and unbounded
  cost/latency. Quality and reliability bounds accept values from 0 through 1.
  `max_cost` and `max_latency_ms` accept nonnegative finite values or null.
- `weights` optionally overrides the preset through nonnegative finite `quality`,
  `cost`, `latency`, and `reliability` values. At least one weight must be positive.
- `execute` defaults to false.
- `max_tokens` defaults to 1024 with an accepted integer range of 1 through 8192.
- `temperature` defaults to 0 with an accepted range of 0 through 2.

The API's unconstrained defaults differ from the Gradio demonstration, whose visible
controls set quality 0.50, cost 0.01, latency 30000ms, and reliability 0.80. Unknown
routing fields, invalid modes, and out-of-range values return HTTP 422.

### Response fields

- `routing_mode` records the selected metric basis.
- `prompt_analysis` records detected/effective task types, override source, rough
  input tokens, requested output budget, output hint, signals, matched tasks,
  warnings, and assessment method.
- `candidates` contains every registered model and its checks, reasons, historical
  profile, optional request estimate, normalized utility values, score, and rank.
- `ranked_model_ids` orders the passing models.
- `selected_model_id` identifies the original primary recommendation.
- `executed_model_id` identifies the successful executor when execution succeeds.
- `status` records `recommended`, `no_candidate`, `succeeded`, or `execution_failed`.
- `explanation` describes the selection or failed requirements.
- `response` contains generated text after successful execution.
- `attempts` records execution results in attempt order.
- `fallback_used` records whether the second-ranked candidate was attempted.
- `total_estimated_cost` sums the observed estimated cost of execution attempts.
  Recommendation-only requests leave this executed-cost field at zero. Candidate
  estimates carry the predicted request costs.

A candidate estimate includes quality/reliability predictions, expected and budget
cost, estimated latency, input/output tokens, supporting sample count, matched task
IDs, and quality basis. Historical measurements stay in the separate `profile` field.

### Constraint and execution behavior

Prompt-aware cost filtering uses estimated input plus the maximum output-token budget.
Ranking uses expected cost. Latency filtering uses a request-scaled estimate when token
observations support it. The baseline mode uses mean cost and median latency from the
historical profile. Context checks reserve input plus the requested output budget.

A valid request with no eligible model returns HTTP 200 and `status=no_candidate`.
The explanation and candidate checks describe recovery options. A profile-loading
failure also returns an explained no-candidate decision.

When execution is enabled, the primary candidate is attempted first. A failed attempt
can trigger one call to the second-ranked eligible candidate. If every attempted
candidate fails, the response returns `execution_failed` with attempt details. The
fallback may add cost to the total across attempts.

Cost and latency constraints are enforced against the selected estimate basis.
The methodology explains uncertainty, word matching, and sampling coverage.
[Routing methodology](prompt_routing.md)

## Local data and access

Comparison and routed execution persist prompts, responses, and execution metrics in
JSONL telemetry. Profile evaluation reads local benchmark evidence. Deployments need
an authentication and approved data-handling design appropriate to their users.
The bundled API configuration targets local development.
