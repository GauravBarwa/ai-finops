# AI FinOps — Architecture Overview

This document describes the architecture of the Pre-HCL-Alignment prototype, designed as the foundation for future intelligent routing.

## System Architecture

```
┌──────────────────────────────────────────────────────────┐
│                      Clients                              │
│  ┌──────────┐  ┌────────────┐  ┌───────────────────────┐│
│  │   curl   │  │ Gradio UI  │  │  Benchmark Runner     ││
│  └────┬─────┘  └─────┬──────┘  └──────────┬────────────┘│
└───────┼───────────────┼────────────────────┼─────────────┘
        │               │                    │
        ▼               ▼                    ▼
┌──────────────────────────────────────────────────────────┐
│                    FastAPI Service                        │
│  ┌──────────┐  ┌────────────┐  ┌───────────────────────┐│
│  │ /health  │  │  /models   │  │      /compare         ││
│  └──────────┘  └────────────┘  └──────────┬────────────┘│
└───────────────────────────────────────────┼──────────────┘
                                            │
                                            ▼
┌──────────────────────────────────────────────────────────┐
│                  ComparisonService                        │
│  Orchestrates concurrent model execution + cost calc      │
└───────────────────────┬──────────────────────────────────┘
                        │
                        ▼
┌──────────────────────────────────────────────────────────┐
│              LLM Provider Abstraction                     │
│  ┌───────────────────────────────────────────────────┐   │
│  │              LiteLLM Client                       │   │
│  │   (unified interface for all providers)           │   │
│  └────────────┬──────────┬──────────┬────────────────┘   │
└───────────────┼──────────┼──────────┼────────────────────┘
                │          │          │
       ┌────────┘    ┌─────┘    ┌─────┘
       ▼             ▼          ▼
   ┌────────┐   ┌────────┐ ┌────────┐
   │ Gemini │   │ OpenAI │ │Claude  │
   │  API   │   │  API   │ │  API   │
   └────────┘   └────────┘ └────────┘
```

## Data Flow

### Comparison Request

```
1. Client sends POST /compare with prompt
2. ComparisonService resolves target models from registry
3. Same prompt sent to all models concurrently (asyncio.gather)
4. For each model response:
   a. Capture latency (wall-clock)
   b. Extract token counts
   c. Calculate estimated cost from registry pricing
5. Aggregate results into CompareResponse
6. Save telemetry records to JSONL
7. Return response to client
```

### Benchmark Run

```
1. Load tasks from data/benchmark/tasks.json
2. For each (task × model) combination:
   a. Send task prompt to model
   b. Capture telemetry
   c. Calculate cost
3. Save all records to benchmark_{timestamp}.jsonl
4. Print summary statistics
```

## Key Design Decisions

| Decision | Rationale |
|----------|-----------|
| **LiteLLM** for provider abstraction | Single interface for 100+ models, avoids coupling to any provider |
| **YAML** for model registry | Easy to add/remove models without code changes |
| **JSONL** for telemetry storage | Simple, append-only, grep-friendly — upgrade to DB when justified |
| **Configuration-driven pricing** | Prices change frequently; never hard-coded |
| **Async concurrent execution** | Compare N models in parallel, not sequentially |
| **Abstract LLMClient interface** | Business logic never touches provider SDKs directly |

## Extension Points

The architecture is designed for future extension:

| Future Feature | Extension Point |
|---------------|----------------|
| Intelligent routing | Add `RoutingService` using model profiles + constraints |
| Quality evaluation | Add scoring logic in `evaluation/` package |
| LLM-as-judge | Extend `LLMClient` usage in evaluation pipeline |
| PostgreSQL storage | Replace `TelemetryService` file backend |
| Redis caching | Add cache layer in `ComparisonService` |
| HCL AI Gateway | Replace `LiteLLMClient` with HCL adapter |

## Component Dependencies

```
app/api/routes.py
    → app/services/comparison_service.py
        → app/llm/litellm_client.py (LLMClient interface)
        → app/llm/registry.py (model metadata)
        → app/llm/cost.py (pricing calculation)
    → app/services/telemetry_service.py
        → app/schemas/telemetry.py

app/core/config.py ← used by all components
app/core/exceptions.py ← used by llm, services
```
