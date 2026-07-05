# langchain-nim-router

Select the best NVIDIA NIM model for LangChain based on requested capabilities and live runtime history.

## What this is

A small, importable Python package that answers one question: **"Which NIM model should I use right now?"**

Given requirements (tools, structured output, vision, reasoning, speed vs quality), it:

1. Discovers available NVIDIA NIM models via `ChatNVIDIA.get_available_models()`
2. Filters by capabilities and availability (cooldowns, bans, RPM limits)
3. Scores candidates based on runtime stats and priority
4. Returns a **real `ChatNVIDIA` instance** — not a wrapper

Tracking is handled via LangChain callbacks, so it works through any composition: `with_structured_output`, `bind_tools`, LCEL pipes, LangGraph, and `astream_events`.

## Installation

```bash
pip install langchain-nim-router
```

## Quickstart

### Selection mode — just get a model

```python
from nim_router import NimRouter

router = NimRouter()
llm = await router.get(tools=True, structured=True, priority="fast")
response = await llm.ainvoke("Hello, how are you?")
```

`get()` returns a genuine `ChatNVIDIA`. Use it anywhere LangChain expects a `BaseChatModel`.

### One-shot tracked mode — select + invoke + auto-track

```python
response = await router.ainvoke(
    [{"role": "user", "content": "Hello"}],
    tools=True,
    structured=True,
    priority="fast",
)
```

`ainvoke()` selects a model, invokes it, and records latency/errors via a LangChain callback. Failures automatically trigger cooldowns or bans.

### Advanced composed chain mode — model + callback

```python
from pydantic import BaseModel

class MySchema(BaseModel):
    name: str
    age: int

selection = await router.select(tools=True, structured=True)

chain = prompt | selection.llm.with_structured_output(MySchema)
result = await chain.ainvoke(
    input,
    config={"callbacks": [selection.callback]},
)
```

`select()` returns a `ModelSelection` with the LLM, model info, and a tracking callback. Pass the callback in `config={"callbacks": [...]}` — it propagates through any LangChain composition automatically.

### Manual overrides

```python
router.ban_model("bad/model")
router.cooldown_model("slow/model", 60)
```

## API Reference

### `NimRouter`

```python
router = NimRouter(config=None, stats_path=None, **config_overrides)
```

**Core methods:**

| Method | Returns | Description |
|--------|---------|-------------|
| `await router.pick(**caps)` | `ModelInfo` | Pure selection — no LLM, no tracking |
| `await router.get(**caps)` | `ChatNVIDIA` | Select and return a real LangChain model |
| `await router.select(**caps)` | `ModelSelection` | Select + LLM + tracking callback |
| `await router.ainvoke(messages, **caps)` | `Any` | One-shot: select + invoke + auto-track |
| `router.tracker_for(model_id)` | `TrackingCallback` | Create a callback bound to a model |

**Capability parameters** (shared by all methods above):

- `tools=False` — require tool/function calling
- `structured=False` — require structured output
- `vision=False` — require image understanding
- `reasoning=False` — require reasoning/thinking
- `priority="balanced"` — `"fast"`, `"quality"`, or `"balanced"`

**Admin methods:**

- `router.ban_model(model_id)` — permanently ban a model
- `router.cooldown_model(model_id, seconds)` — temporarily cool down
- `router.record_success(model_id, ...)` — manual success recording
- `router.record_failure(model_id, error=...)` — manual failure recording
- `router.stats()` — snapshot of all runtime stats

**Convenience helpers:**

- `await router.fast_tools_model()` — fast model with tools
- `await router.structured_model()` — model for structured output
- `await router.vision_model()` — model with vision
- `await router.reasoning_model()` — model with reasoning

### `ModelSelection`

```python
@dataclass
class ModelSelection:
    info: ModelInfo        # model metadata
    llm: Any              # real ChatNVIDIA
    callback: TrackingCallback  # LangChain callback for auto-tracking
```

### `TrackingCallback`

A LangChain `BaseCallbackHandler` that records latency, token usage, and errors back to the router. Created via `router.tracker_for()` or `router.select()`.

## Configuration

### Environment Variables

```bash
# Only use these models (comma-separated)
NIM_ROUTER_MODEL_POOL="meta/llama-3.3-70b-instruct,meta/llama-3.1-8b-instruct"

# Exclude specific models (comma-separated)
NIM_ROUTER_EXCLUDED_MODELS="some/deprecated-model"

# Default requests per minute per model
NIM_ROUTER_DEFAULT_RPM=30

# Per-model RPM limits (JSON)
NIM_ROUTER_MODEL_RPM_JSON='{"meta/llama-3.3-70b-instruct": 20}'

# Capability overrides (JSON)
NIM_ROUTER_CAPABILITIES_JSON='{"model-id": {"tools": true, "structured": true}}'

# Quality hints (JSON)
NIM_ROUTER_QUALITY_HINTS_JSON='{"model-id": 0.95}'

# Request timeout in seconds
NIM_ROUTER_TIMEOUT_SECONDS=120.0

# Allow override-only models not discovered from API (default: false)
NIM_ROUTER_ALLOW_UNDISCOVERED=1
```

### Programmatic Config

```python
from nim_router import NimRouter
from nim_router.config import RouterConfig

config = RouterConfig(
    model_pool=["meta/llama-3.3-70b-instruct"],
    excluded_models=["some/deprecated-model"],
    default_rpm=30,
    quality_hints={"meta/llama-3.3-70b-instruct": 0.95},
    timeout_seconds=60.0,
    allow_undiscovered_models=False,
)
router = NimRouter(config=config)
```

## How tracking works

The router uses LangChain's callback system — no wrappers, no monkey-patching:

1. `ainvoke()` and `select()` create a `TrackingCallback` bound to the chosen model
2. The callback fires on `on_chat_model_start` (marks rate-limit slot) and `on_llm_end` / `on_llm_error` (records stats)
3. Stats feed back into scoring, so future picks prefer models with better history
4. Errors trigger automatic cooldowns (rate-limit, timeout, HTTP) or bans (404)

Callbacks propagate through any LangChain composition — `with_structured_output()`, `bind_tools()`, LCEL `|` pipes, and agent chains — without any extra code.

## Rate limiting and cooldowns

- **RPM**: Per-model requests-per-minute tracking. Excluded when limit hit.
- **Rate-limit (429)**: 30-second cooldown
- **Timeout**: 20-second cooldown
- **HTTP error**: 10-second cooldown
- **Model not found (404)**: permanent ban for process lifetime

## Runtime stats

Tracked per model: `calls`, `successes`, `failures`, `rate_limits`, `avg_latency`, `avg_tokens_per_second`, `structured_success_rate`, `tool_success_rate`, `vision_success_rate`.

Persist to JSON:

```python
router = NimRouter(stats_path="nim_router_stats.json")
```

## Priority modes

- **`fast`**: Prefer low latency and high tokens/second
- **`quality`**: Prefer high quality hint and success rate
- **`balanced`**: Blend speed, reliability, and quality

## What this is not

- Not an agent framework
- Not a proxy server
- Not a LiteLLM replacement
- Not tied to any specific application
