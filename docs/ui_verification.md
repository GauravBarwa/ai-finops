# UI and publication verification

## Publication checks

The final documentation pass on 6 November 2026 reran the offline regression suite,
Ruff, and diff whitespace checks. The suite recorded **147 passing tests**.
The isolated Python wheel build also passed. Internal documentation links, JSON
examples, OpenAPI defaults, and all six CLI help entry points were checked.
The regression suite covers provider normalization, schemas, comparison, cost, profile evaluation and
publication, constraints, prompt-aware estimates, routing fallback, API contracts,
and the actual Gradio callback functions.

The Dockerfile's package metadata copy includes the README. Docker runtime
validation remains open because the publication environment lacks a Docker executable.
Container setup commands are documented for an environment with Docker available.
[Configuration](configuration.md)

## Final repository review

The mentor-sharing review on 6 November 2026 reran all **169 tests** in an isolated
copy of the GitHub checkout with the review changes applied. That copy contained
no `.env`, profiles, or raw benchmark measurements. Application startup, API metadata
and validation, missing-profile/run guidance, and callbacks for all three UI tabs
passed. All six CLI help entry points and an isolated Python wheel build passed.
Ruff, documentation links, JSON examples, and diff whitespace checks passed.

The review also checked tracked generated-state exclusions and common credential
patterns in Git history. The new project walkthrough explains the components and
selection logic, while the README and API/methodology descriptions reflect the
latest regression coverage and validation rules. Provider inference was covered
by the earlier checks described below. Docker runtime verification remains open
because Docker is unavailable in this environment.
[Project walkthrough](project_walkthrough.md)

## Historical Gradio HTTP checks

Live checks on 5 November 2026 used temporary local Gradio servers and repeated
`gradio_client` submissions to the real app callbacks. The temporary processes were
closed after each run.

The prompt-aware implementation completed **37 broader workflows** across all three tabs.

- Model Profiles supplied hints for all four task types, refreshed measurements,
  rebuilt the active snapshot, and retained profile coverage after another refresh.
- Smart Router recommended across four task types and four policy presets, exercised
  strict quality/cost/latency limits, rejected blank input, and executed all four types.
- Comparison handled empty input, both example controls, all enabled models, and a
  repeated selected-model submission in the same session.

A later focused run completed **23 prompt-routing workflows**.

- Four known summarization inputs were checked in both baseline and prompt-aware modes.
- All four task types were automatically detected.
- Short/long versions of one topic changed request cost and latency estimates.
- Output-token budget changes affected budget cost.
- Explicit task override and uncertain intent were reported.
- Oversized inputs exercised context-window rejection.
- Four automatically detected task types produced live routed answers.

The focused check held the summarization policy and constraints fixed. Historical mode
selected Groq GPT-OSS 20B for all four known examples. Prompt-aware mode selected three
different models using their matching-case evidence.
[Routing methodology](prompt_routing.md)

The broader comparison completed with five successes and one Gemini 3.6 HTTP 429
free-tier daily-quota response. Its check verified result completion and error display.
The focused live router executions succeeded through their eligible execution paths.

A connected browser surface was unavailable during those checks. Verification used
HTTP workflows and component wiring. Screenshot/layout inspection remains an additional
manual check for the presentation environment.

## Regression coverage

The preset expansion recorded **173 passing tests** and **84 Gradio HTTP workflow
checks**. The checks load all 11 comparison presets and all 22 routing presets,
validate every populated setting, clear earlier outputs, and submit every routing
setup. Routing preset selection resets execution to off. Clearing a selector preserves
edited controls. Switching a preset cancels the corresponding in-progress UI generation.

An additional recommendation-only replay against the local measured profiles confirmed
three distinct selections across the four prompt-aware summaries, a shared selection
for their historical equivalents, policy-driven selection changes, strict zero-cost
rejection, and rejection when increasing the reserved output budget under a fixed cost
limit. These are replays of known benchmark inputs and depend on the active profiles.
The demo guide records the checked selections and suggested presentation order.
[Preset demonstration sequence](demo_guide.md#preset-demonstration-sequence)

The subsequent UI refresh recorded **171 passing tests** and **31 Gradio HTTP workflow
checks**. Answer components render sanitized Markdown, retain complete final content,
and reveal successful responses in bounded incremental frames. Provider errors remain
literal fenced text, including embedded Markdown fences and diagnostics longer than
180 characters. Table cells wrap and profile metrics use consistent display precision.
The typing reveal begins after a provider response completes. Its animation time is
separate from the recorded provider latency.

Browser checks used simulated Markdown answers and long quota errors. They inspected
all three tabs, light/dark colors, a narrow viewport, profile formatting, and page/table
overflow. The refreshed UI targets Gradio 6.28 within major version 6, matching the
verification environment.

The follow-up bug audit on 6 November 2026 recorded **169 passing tests** and
**31 Gradio HTTP workflow checks** across all three tabs. Ruff and diff whitespace
checks passed. The fixes cover fractional-dollar budget input, millisecond latency
steps, whitespace-only comparison API prompts, blank Ollama answers, malformed or
duplicate benchmark tasks, disabled benchmark models, and overflowing custom weights.
Failed dataset evaluation retains the last valid profiles, which remain available
for routing with an explicit evidence warning.

Browser checks verified a $0.005 budget and a 502 ms latency limit after prompt edits,
tab switches, profile refresh, and repeated routing. Comparison checks used simulated
provider responses to exercise partial failure and multiline error display. A separate
live call to the configured local Llama 3.2 model returned a nonempty answer with
consistent token totals. Blank-answer handling uses simulated responses for regression
coverage. The original reported budget reset did not reproduce in the test browser,
but the cost field now has explicit eight-decimal precision and a fractional step.

The suite exercises the following failure and recovery paths.

- Adapter errors, provider prefixes, missing keys, retired providers, empty cloud
  answers, absent token usage, and credential redaction.
- Incremental snapshots, cancellation, per-model deadlines, partial failures, unknown
  model IDs, repeated calls, and literal multiline error content.
- Reference scoring, wrapped JSON, failed observations, percentile interpolation,
  data identity, repeated evaluation, and invalid source filenames.
- Idempotent verification publication and rejection of edited, incomplete, or duplicate
  input. Failed evaluation preserves the prior profile snapshot.
- Missing/corrupt profiles, missing/bad benchmark files, missing golden datasets,
  invalid routing fields, and empty/whitespace/null UI prompts.
- All task/policy combinations, hard constraints, fallback success/failure, prompt
  override warnings, unknown topics, context limits, output budgets, and API serialization.
- Content-conditioned model selection using equal global priors and different measured
  topic strengths, plus a baseline comparison and preservation of the original profiles.

Provider failure regressions use fake adapters. The historical live runs supply
separate evidence for the configured accounts and observed quota state.

## Reproduce

Run the offline checks from an activated environment.

```bash
GRADIO_ANALYTICS_ENABLED=False pytest -q
ruff check .
git diff --check
```

Prepare the full six-model profile experiment before running the UI workflow scripts.
They expect the configured demonstration dataset and model coverage.

```bash
python -m scripts.verify_providers --publish-profiles
GRADIO_ANALYTICS_ENABLED=False python -m scripts.verify_ui
GRADIO_ANALYTICS_ENABLED=False python -m scripts.verify_prompt_routing
```

Append `--live` to a UI verifier to execute its inference cases. Use `--port` to select
an available temporary-server port. A different experiment can change model selections,
quality estimates, and whether prepared constraint scenarios remain feasible.

The scripts save raw proof under the local results directory. The original files used
`full_ui_verification.json`, `prompt_aware_ui_checks.json`, and
`prompt_routing_verification.json`. These outputs are gitignored and are regenerated
per environment. The test commands and committed observation summaries support review
from a fresh clone.

The demonstration guide describes a manual presentation walkthrough.
[Demo guide](demo_guide.md)
