# Contributing

## Development setup

Use Python 3.11+ and install the development dependencies in a virtual environment.

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[dev]'
[ -f .env ] || cp .env.example .env
```

Use synthetic or approved public task data for tests and demonstrations. Credentials
belong in the local environment or `.env`. Generated telemetry contains prompt/response
content and stays in ignored local storage. Review staged files before sharing changes.

## Before opening a pull request

```bash
GRADIO_ANALYTICS_ENABLED=False pytest -q
ruff check .
git diff --check
```

Existing tests exercise the service and UI contracts with fake adapters. Add a focused
regression test when changing behavior that those tests would miss. Validate live
provider behavior separately when changing adapters or configured provider identifiers.

Describe the user-visible behavior, the reason for the change, the checks run, and any
verification limitations in the pull request. Keep changes focused and update the
relevant setup, API, or methodology document when behavior changes.

## Project conventions

- Keep provider payloads and normalization inside `app/llm` adapters.
- Keep network/storage orchestration in services and pure selection logic in `app/domain`.
- Use Pydantic request/response contracts and preserve documented caller behavior.
- Keep model availability, generation options, and shadow pricing in the registry.
- Preserve model IDs when interpreting historical measurements. Replacement models
  need their own measured evidence.
- Store measured profiles separately from request predictions and expose uncertainty.
- Preserve failures when evaluating reliability and quality.
- Keep hard requirements ahead of weighted optimization and provide rejection reasons.
- Use atomic replacement for profile publication and preserve the last completed
  snapshot when rebuilding fails.

The current stack uses Python, FastAPI, Pydantic, httpx, LiteLLM, Gradio, and file storage.
Changes that expand runtime dependencies should explain the behavior they enable.

## Live checks and reproducibility

```bash
python -m scripts.verify_providers --publish-profiles
GRADIO_ANALYTICS_ENABLED=False python -m scripts.verify_ui --live
GRADIO_ANALYTICS_ENABLED=False python -m scripts.verify_prompt_routing --live
```

These commands consume provider quota and need working models and prepared profiles.
The UI workflow scripts use the full six-model demonstration experiment. Record the
registry, dataset, source runs, scorer version, dependency versions, and date alongside
results. Update dated reports with the actual observed coverage and failures.

The historical validation environment used Python 3.12, LiteLLM 1.101.0, and Gradio 6.28.0.
Dependency ranges in `pyproject.toml` permit newer resolutions. A fresh environment
should run the regression suite and the provider checks relevant to its intended use.

## Documentation and local artifacts

The README links the architecture, configuration, API, methodology, demo, and historical
reports. Keep GitHub links within that committed documentation. Raw verification outputs,
benchmark JSONL, and profiles live under `data/results` and are generated per environment.
Refer to those local artifacts with paths and reproduction commands.

The project declares the MIT license in its package metadata and repository license.
[License](LICENSE)
