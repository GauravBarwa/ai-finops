# Configuration and operations

## Environment setup

Create a Python 3.11+ virtual environment and install the project with the development
extra when running the checks.

The refreshed UI uses Gradio 6.28 or later within major version 6. Reinstall the project
dependencies when updating an older environment. Presentation settings live in
`app/ui.py` and are supplied at launch by the UI module and verification scripts.

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[dev]'
[ -f .env ] || cp .env.example .env
```

Settings load from the project-root `.env` and environment variables. Environment
variables take precedence. Paths resolve relative to the project root unless an
absolute path is supplied. Restart the API, Gradio, or CLI process after configuration
changes because settings and registry objects are initialized per process.

### Provider settings

- `GEMINI_API_KEY` supplies the configured Google Gemini models. The default is empty.
- `GROQ_API_KEY` supplies the configured Groq models. The default is empty.
- `GITHUB_API_KEY` is retained for legacy configuration compatibility. Its registry
  entry carries the recorded retirement reason and is disabled.
- `OPENAI_API_KEY` and `ANTHROPIC_API_KEY` are optional adapter settings for explicitly
  enabled integrations. Their bundled registry entries are disabled.
- `OLLAMA_BASE_URL` defaults to `http://127.0.0.1:11434`.
- `OLLAMA_TIMEOUT_SECONDS` defaults to `180` for the native adapter.
- `COMPARISON_TIMEOUT_SECONDS` defaults to `60` and accepts values above zero through
  600. Comparison and router execution enforce this outer deadline, including retry
  time, even when the native adapter has a larger configured timeout.

Use free-tier accounts and local inference for the intended project workflow. Requests
consume the account quota available to the configured key. The provider console is the
source for account billing and quota controls.

### Application and storage settings

- `APP_ENV` defaults to `development` and labels application logs.
- `LOG_LEVEL` defaults to `INFO`.
- `DATA_DIR` defaults to `data`.
- `RESULTS_DIR` defaults to `data/results` and stores benchmark runs, telemetry,
  verification output, and `model_profiles.json`.
- `BENCHMARK_DIR` defaults to `data/benchmark` and supplies `tasks.json`.
- `MODELS_CONFIG_PATH` defaults to `data/models.yaml`.
- `HOST` and `PORT` default to `0.0.0.0` and `8000` in Settings. Use Uvicorn command-line
  flags to control its binding. The Gradio module launches on port 7860.

`GRADIO_ANALYTICS_ENABLED=False` disables Gradio analytics for the documented checks.
`LITELLM_LOCAL_MODEL_COST_MAP=True` uses LiteLLM's bundled metadata. The adapter sets
that default to avoid an import-time metadata download. Shadow cost calculations
always use the repository's model registry.

## Configured models

The following active IDs reflect the registry validated on 5 November 2026.
Provider catalogs and account access can change.

- `llama3.2-local` uses `ollama/llama3.2:latest`.
- `gpt-oss-120b-cloud` uses `ollama/gpt-oss:120b-cloud` and requires Ollama cloud access.
- `gemini-3.6-flash` uses `gemini/gemini-3.6-flash`.
- `gemini-3.5-flash-lite` uses `gemini/gemini-3.5-flash-lite`.
- `groq-gpt-oss-20b` uses `groq/openai/gpt-oss-20b`.
- `groq-gpt-oss-120b` uses `groq/openai/gpt-oss-120b`.

Each YAML entry specifies its capabilities, context window, enabled state, and shadow
input/output rates per million tokens. Cloud generation options currently request low
reasoning effort. Gemini 3 requests omit deprecated sampling controls in the adapter.
A measured profile supports routing only when the model also declares that task capability.

The earlier provider checks recorded Gemini 2.0 retirement, absent Groq Llama access,
and GitHub Models retirement. Preserve model identity when replacing entries so that
an earlier model's measurements keep their original attribution.
[Provider verification](provider_verification.md)

## Local Ollama workflow

Install Ollama, start its service, and confirm the model name used by the registry.

```bash
ollama pull llama3.2:latest
ollama list
curl http://127.0.0.1:11434/api/tags
python -m scripts.run_benchmark --models llama3.2-local
```

With the API running, evaluate the saved files.

```bash
curl -X POST http://127.0.0.1:8000/evaluate \
  -H 'Content-Type: application/json' -d '{}'
curl http://127.0.0.1:8000/profiles
```

One local model produces four model/task profiles when all 16 tasks run. Several
profiles are sufficient to use the router, although policy demonstrations become more
informative with multiple measured models. In Comparison, select the intended models
explicitly. An empty selection calls all enabled entries.

## Multi-provider collection and publication

The benchmark runner accepts selected model IDs, a custom task file, generation limits,
and temperature.

```bash
python -m scripts.run_benchmark \
  --models llama3.2-local groq-gpt-oss-20b \
  --max-tokens 1024 --temperature 0
```

`POST /evaluate` then rebuilds profiles from the normal benchmark files. Its optional
`run_files` list selects a controlled experiment using benchmark basenames.

The provider verifier runs the configured enabled set and inserts pauses between task
groups. With the bundled registry, a complete run makes 96 calls.

```bash
python -m scripts.verify_providers --publish-profiles
```

Without `--publish-profiles`, the verifier saves a JSON report. A completed report can
be published later into ordinary benchmark files and active profiles.

```bash
python -m scripts.publish_profiles \
  --report data/results/provider_verification.json
```

Publication validates golden task identity, recomputes quality from raw outcomes, retains
failures, and uses a content-derived filename for repeated imports. The UI evaluation
button subsequently sees the same evidence. Generated artifacts are local and gitignored.

Use a separate `RESULTS_DIR` for experiments with different datasets. Retain raw runs,
matching task definitions, model configuration, and the profile snapshot together when
backing up measurements.

## Verification commands

Offline regression checks use fake provider responses where execution is required.

```bash
GRADIO_ANALYTICS_ENABLED=False pytest -q
ruff check .
```

After collecting full profiles, exercise the three Gradio tabs through a temporary
HTTP server on port 7861.

```bash
GRADIO_ANALYTICS_ENABLED=False python -m scripts.verify_ui
GRADIO_ANALYTICS_ENABLED=False python -m scripts.verify_prompt_routing
```

These scripts expect the full six-model measurements used by the demonstration.
Their assertions cover prepared task/constraint scenarios. A smaller or newer experiment
can produce different eligible sets and selections. Use the unit suite and direct UI/API
inspection while preparing that experiment. The `--port` argument selects a free port.
Adding `--live` also executes inference calls.

## Containers

The Dockerfile packages the FastAPI service. Build from the repository root.

```bash
docker build -t ai-finops .
docker run --rm --env-file .env \
  -p 127.0.0.1:8000:8000 \
  -v "$PWD/data/results:/app/data/results" \
  ai-finops
```

Supply an Ollama URL reachable from the container when using native models. For a Linux
host with an Ollama service on the host loopback interface, host networking allows the
container to use the configured loopback URL.

```bash
docker run --rm --network host --env-file .env \
  -v "$PWD/data/results:/app/data/results" \
  ai-finops uvicorn app.main:app --host 127.0.0.1 --port 8000
```

Persist the results mount to retain measurement history and profiles. Public deployment
needs an authentication and network-access design appropriate to the environment.
The publication validation environment had Python 3.12 and lacked a Docker executable,
so container runtime validation remains an open check.

## Troubleshooting

### The router returns `no_candidate`

Read each candidate's checks. `profile_available` means that the selected task needs
measurements and evaluation. A failed quality, cost, latency, or reliability check
shows the applicable value and requirement. Large inputs or output budgets can fail
the context-window check. Adjust limits explicitly or collect stronger evidence.

In prompt-aware mode, the budget cost can exceed expected cost because filtering reserves
the maximum allowed output. The historical task-average mode uses the stored mean cost.

### Recommendation appears with an empty generated-answer area

Recommendation-only mode produces the selection and explanatory estimates. Enable
execution to call the selected model and obtain an answer.

### Cloud models are listed but calls fail

Check the account's model catalog, key configuration, permissions, and quota. An enabled
registry entry records configuration. Live provider access is checked separately. HTTP
429 indicates quota/rate limiting and 503 indicates provider overload. Execution can
attempt the second-ranked eligible candidate after a primary failure.

### Profiles disappear after evaluation

Ensure the relevant measurements were published as ordinary `benchmark_*.jsonl` files
in the active `RESULTS_DIR`. Rebuild from that directory. API and Gradio processes must
use the same storage and configuration when sharing experiments.

### Detailed benchmark evidence is unavailable

Keep the benchmark files named in `source_files` with the profile snapshot. Routing
can fall back to historical priors and reports the missing evidence in its analysis.
Re-evaluate after restoring compatible source files.

### The UI still uses old settings

Stop the Python process and restart it. A browser refresh reconnects to the running
process. It reloads the client view while the Python process retains initialized settings.

### Port 8000, 7860, or 7861 is occupied

Choose a different Uvicorn port or a verifier `--port`. The standard Gradio module uses
7860, so stop an earlier instance before launching another through that entry point.
