# AI FinOps — Intelligent LLM Model Selection & Cost Optimization

A model-agnostic service that compares multiple LLMs on the same workload and captures quality, cost, latency, and reliability metrics — laying the foundation for intelligent routing.

## 🎯 What This Does

Given a prompt, this service:

1. Sends it to **multiple LLM models** through a unified interface
2. Captures **tokens, latency, and estimated cost** for each
3. Returns a **side-by-side comparison** via API or Gradio UI
4. **Persists telemetry** for reproducible analysis

```
Prompt → Model A, Model B, Model C → Comparative Metrics → Stored Results
```

## 🏗️ Architecture

```
FastAPI (/compare, /models, /health)
   ↓
ComparisonService (concurrent model execution)
   ↓
LiteLLM (unified provider abstraction)
   ↓
┌─────────────┬──────────────┬──────────────┐
│ Gemini 2.0  │ GPT-4o Mini  │ Claude Haiku │
│  Flash      │              │              │
└─────────────┴──────────────┴──────────────┘
   ↓
TelemetryService → JSONL files
```

## 🚀 Quick Start

### Prerequisites

- Python 3.11+
- API keys for at least one provider (Gemini, OpenAI, or Anthropic)

### Setup

```bash
# Clone and enter project
cd ai-finops

# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # Linux/Mac
# .venv\Scripts\activate   # Windows

# Install dependencies
pip install -e ".[dev]"

# Configure API keys
cp .env.example .env
# Edit .env with your actual API keys
```

### Run the API Server

```bash
uvicorn app.main:app --reload
```

Visit http://localhost:8000/docs for the interactive API docs.

### Run the Gradio Demo

```bash
python -m app.demo
```

Visit http://localhost:7860 for the comparison UI.

### Run the Benchmark

```bash
# All enabled models
python -m scripts.run_benchmark

# Specific models
python -m scripts.run_benchmark --models gemini-2.0-flash gpt-4o-mini
```

### Run Tests

```bash
pytest tests/ -v
```

## 📡 API Endpoints

| Method | Path | Description |
|--------|------|-------------|
| `GET` | `/health` | Health check |
| `GET` | `/models` | List all enabled models |
| `GET` | `/models/{model_id}` | Get single model info |
| `POST` | `/compare` | Compare prompt across models |

### Example: Compare Models

```bash
curl -X POST http://localhost:8000/compare \
  -H "Content-Type: application/json" \
  -d '{
    "prompt": "Summarize: Cloud computing allows businesses to rent computing resources on demand.",
    "task_type": "summarization"
  }'
```

## 📁 Project Structure

```
ai-finops/
├── app/
│   ├── api/           # FastAPI routes
│   ├── core/          # Config, logging, exceptions
│   ├── llm/           # LLM interface, LiteLLM client, registry, cost
│   ├── schemas/       # Pydantic request/response models
│   ├── services/      # Comparison, telemetry, benchmark services
│   ├── main.py        # FastAPI app factory
│   └── demo.py        # Gradio demo
├── data/
│   ├── benchmark/     # Curated benchmark tasks
│   ├── models.yaml    # Model registry configuration
│   └── results/       # Telemetry output (gitignored)
├── scripts/           # CLI utilities
├── tests/             # Automated tests
├── docs/              # Documentation
├── .env.example       # Environment variable template
├── pyproject.toml     # Project dependencies
└── Dockerfile         # Container build
```

## ⚙️ Configuration

### Models

Models are configured in `data/models.yaml`. Add, remove, or modify models without changing code:

```yaml
models:
  - model_id: "gemini-2.0-flash"
    provider: "google"
    display_name: "Gemini 2.0 Flash"
    litellm_model_name: "gemini/gemini-2.0-flash"
    input_price_per_1m_tokens: 0.10
    output_price_per_1m_tokens: 0.40
    enabled: true
```

### Environment Variables

See `.env.example` for all configuration options.

## 🔒 Security

- API keys loaded from environment variables — never committed
- Only public/synthetic data used for benchmarks
- Telemetry files are gitignored
- No confidential data processed without explicit approval

## 📌 Current Phase

This is the **Pre-HCL-Alignment Phase** — a minimal technical proof. The foundation is ready for:

- Intelligent routing (constraint-based model selection)
- Quality evaluation (LLM-as-judge, deterministic metrics)
- Full benchmark suite
- Enterprise integration with HCL infrastructure

## License

MIT
