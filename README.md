# langchain-nim-router

Select the best NVIDIA NIM model for LangChain based on requested capabilities and live runtime history.

## What this package is

A standalone, reusable Python package that answers one question: **"Which NIM model should I use right now?"**

Given a set of requirements (tools, structured output, vision, reasoning, speed, quality), it:

1. Discovers available NVIDIA NIM models via `ChatNVIDIA.get_available_models()`
2. Filters by capabilities and availability
3. Scores candidates based on runtime stats and priority
4. Returns a configured `ChatNVIDIA` instance ready to use

## What this package is not

- Not an agent framework
- Not tied to any specific application (Z-Apply, DeepAgents, LangGraph, etc.)
- Not a proxy server
- Not a replacement for LiteLLM or any routing layer

## Installation

```bash
pip install langchain-nim-router
```

## Quickstart

```python
from nim_router import NimRouter

router = NimRouter()

llm = await router.get(
    tools=True,
    structured=True,
    priority="fast",
)

response = await llm.ainvoke("Hello, how are you?")
```

## Configuration

Configure via environment variables or programmatic config:

### Environment Variables

```bash
# Only use these models (comma-separated)
export NIM_ROUTER_MODEL_POOL="meta/llama-3.3-70b-instruct,meta/llama-3.1-8b-instruct"

# Exclude specific models (comma-separated)
export NIM_ROUTER_EXCLUDED_MODELS="some/deprecated-model"

# Default requests per minute per model
export NIM_ROUTER_DEFAULT_RPM=30

# Per-model RPM limits (JSON)
export NIM_ROUTER_MODEL_RPM_JSON='{"meta/llama-3.3-70b-instruct": 20, "some/slow-model": 5}'

# Capability overrides (JSON)
export NIM_ROUTER_CAPABILITIES_JSON='{"meta/llama-3.3-70b-instruct": {"tools": true, "structured": true, "vision": false, "reasoning": false}}'

# Quality hints (JSON)
export NIM_ROUTER_QUALITY_HINTS_JSON='{"meta/llama-3.3-70b-instruct": 0.95, "meta/llama-3.1-8b-instruct": 0.65}'

# Request timeout in seconds
export NIM_ROUTER_TIMEOUT_SECONDS=60.0
```

### Programmatic Config

```python
from nim_router import NimRouter
from nim_router.config import RouterConfig

config = RouterConfig(
    model_pool=["meta/llama-3.3-70b-instruct", "meta/llama-3.1-8b-instruct"],
    excluded_models=["some/deprecated-model"],
    default_rpm=30,
    model_rpm={"some/slow-model": 5},
    capabilities_overrides={
        "meta/llama-3.3-70b-instruct": {
            "tools": True,
            "structured": True,
            "vision": False,
            "reasoning": False,
        }
    },
    quality_hints={
        "meta/llama-3.3-70b-instruct": 0.95,
        "meta/llama-3.1-8b-instruct": 0.65,
    },
    timeout_seconds=60.0,
)

router = NimRouter(config=config)
```

## Capability Overrides

If a model's capabilities are not automatically detected, you can override them:

```json
{
  "meta/llama-3.3-70b-instruct": {
    "tools": true,
    "structured": true,
    "vision": false,
    "reasoning": false
  },
  "some/vision-model": {
    "tools": false,
    "structured": false,
    "vision": true,
    "reasoning": false
  }
}
```

## RPM Overrides

Control per-model rate limits:

```json
{
  "meta/llama-3.3-70b-instruct": 20,
  "some/slow-model": 5
}
```

## Runtime Stats

The router tracks per-model statistics:

- **calls**: Total invocations
- **successes/failures**: Success/failure counts
- **rate_limits**: Number of 429 errors
- **avg_latency**: Average response time
- **avg_tokens_per_second**: Average generation speed
- **structured_success_rate**: Success rate for structured output
- **tool_success_rate**: Success rate for tool calls
- **vision_success_rate**: Success rate for vision tasks

Stats are kept in memory by default. Optionally persist to JSON:

```python
router = NimRouter(stats_path="nim_router_stats.json")
```

## Rate Limiting and Cooldowns

- **Rate limiting**: Per-model RPM tracking. If a model exceeds its RPM limit, it's temporarily excluded.
- **Cooldowns**: 429/rate-limit errors cool down a model for 30 seconds. HTTP errors cool down for 10 seconds.
- **Banning**: 404/model-not-found errors permanently ban a model for the process lifetime.

```python
# Manual control
router.ban_model("some/bad-model")
router.cooldown_model("some/slow-model", 60.0)
```

## Using with Your Application

This package is designed to be imported and used by any application:

```python
from nim_router import NimRouter

router = NimRouter()

# Get a fast model with tool support
llm = await router.get(tools=True, priority="fast")
response = await llm.ainvoke("Use the weather tool to check today's forecast")

# Get a model for structured output
llm = await router.get(structured=True, priority="balanced")
response = await llm.ainvoke("Return a JSON object with name and age")

# Record results for better future selections
router.record_success(llm.model, latency=1.2, tokens_per_second=45.0)
# or on failure
router.record_failure(llm.model, kind="rate_limit")
```

The router does not perform LLM calls itself. It only selects and configures the model.

## API Reference

### `NimRouter`

```python
router = NimRouter(config=None, stats_path=None, **config_overrides)
```

**Methods:**

- `await router.get(**kwargs)` → `ChatNVIDIA`: Pick a model and return configured instance
- `await router.pick(**kwargs)` → `ModelInfo`: Pick a model and return metadata
- `router.record_success(model_id, **kwargs)`: Record successful call
- `router.record_failure(model_id, error=None, kind=None)`: Record failed call
- `router.ban_model(model_id)`: Ban a model for this process
- `router.cooldown_model(model_id, seconds)`: Cool down a model
- `router.stats()` → `dict`: Get runtime stats snapshot
- `await router.fast_tools_model(**kwargs)` → `ChatNVIDIA`: Fast model with tools
- `await router.structured_model(**kwargs)` → `ChatNVIDIA`: Model for structured output
- `await router.vision_model(**kwargs)` → `ChatNVIDIA`: Model with vision
- `await router.reasoning_model(**kwargs)` → `ChatNVIDIA`: Model with reasoning

### Priority Modes

- **`fast`**: Prefer low latency and high tokens/second
- **`quality`**: Prefer high quality hint and success rate
- **`balanced`**: Blend speed, reliability, and quality
