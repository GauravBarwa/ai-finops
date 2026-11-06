# Historical provider verification

## Experiment record

The provider migration and live checks were performed on 5 November 2026 using the
configured free-tier accounts and local Ollama service. Model access and available
quota reflect that experiment. Fresh installations should run their own checks before
collecting profiles for a demonstration.

The active registry used Gemini 3.6 Flash, Gemini 3.5 Flash-Lite, Groq GPT-OSS 20B,
Groq GPT-OSS 120B, local Llama 3.2, and GPT-OSS 120B through Ollama Cloud.

## Catalog and migration findings

Gemini 2.0 calls returned retirement errors. Gemini 3.8 appeared in the account catalog
but timed out repeatedly during LiteLLM and direct API probes. A successful Gemini 3.6
probe supported using that identifier in the recorded experiment. Google's model and
migration documentation describes the available model families.
[Google model documentation](https://aistudio.google.com/docs/models)

The account's Groq catalog omitted the configured Llama 3.1/3.3 entries. GPT-OSS 20B and
120B were listed and answered synthetic inference probes. The replacement IDs received
their own registry identities and subsequent measurements.
[Groq supported models](https://console.groq.com/docs/models)

GitHub documented full retirement of GitHub Models on 30 July 2026. Its legacy entry
was disabled with that reason, and explicit adapter requests return retirement guidance.
[GitHub Models documentation](https://docs.github.com/en/github-models)

## Complete live workload

Six enabled models ran all 16 golden tasks, producing 96 call outcomes.

- Gemini 3.6 Flash succeeded on 15 calls and failed on one.
- Gemini 3.5 Flash-Lite succeeded on 14 calls and failed on two.
- GPT-OSS 120B through Ollama Cloud succeeded on all 16 calls.
- Local Llama 3.2 succeeded on all 16 calls.
- Groq GPT-OSS 20B succeeded on all 16 calls.
- Groq GPT-OSS 120B succeeded on all 16 calls.

The total was 93 successes and three failures. All three failures were Gemini 503
service-overload responses after the bounded retry. Separate rechecks recovered two
of those task/model pairs, while one still returned 503. The original failed outcomes
were retained when the measurements were published into active profiles.

This run verifies observed access and nonempty answer text. The report also recorded
separate deterministic reference quality scores for the outcomes.

## Comparison and adapter behavior

The repairs added incremental Gradio snapshots, literal provider-text rendering,
per-model deadlines, early missing-key handling, empty-answer handling, credential
redaction, and configured cloud reasoning effort. Automated regressions cover those
paths. Later UI checks also exercised quota exhaustion and repeated submissions.
[UI verification](ui_verification.md)

The completed verification measurements were published into normal benchmark files
and evaluated into 24 model/task profiles. This makes them available to both runtime
routing and the UI evaluation button.

## Reproduce

After configuring the providers, run the complete collection and publication command.

```bash
python -m scripts.verify_providers --publish-profiles
```

A smaller probe can select a golden task ID.

```bash
python -m scripts.verify_providers \
  --task-ids cls_001 \
  --output data/results/quick_check.json
```

The verifier exits nonzero if any call fails. It saves a local JSON report containing
request context, outcomes, quality scores, registry hash, timestamps, and per-model
counts. The default output is `data/results/provider_verification.json`.

A completed local report can be published later.

```bash
python -m scripts.publish_profiles \
  --report data/results/provider_verification.json
```

Raw reports, logs, and profiles are generated local artifacts and are gitignored.
The committed report records the observed totals and commands for a new experiment.
Setup and account troubleshooting are documented with the environment configuration.
[Configuration](configuration.md)
