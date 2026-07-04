from __future__ import annotations

from nim_router.limiter import RateLimiter
from nim_router.schemas import ModelCapabilities, ModelInfo, ModelRuntimeStats
from nim_router.stats import StatsStore


def filter_candidates(
    models: list[ModelInfo],
    required: ModelCapabilities,
    limiter: RateLimiter,
    stats_store: StatsStore,
) -> list[ModelInfo]:
    """Filter models by capability requirements and availability."""
    result: list[ModelInfo] = []
    for model in models:
        # Skip deprecated
        if model.deprecated:
            continue

        # Skip banned
        if stats_store.is_banned(model.id):
            continue

        # Skip cooling down
        if stats_store.is_cooling_down(model.id):
            continue

        # Skip rate-limited
        if not limiter.is_available(model.id):
            continue

        # Check capabilities
        if not _capabilities_satisfy(required, model.capabilities):
            continue

        result.append(model)
    return result


def _capabilities_satisfy(required: ModelCapabilities, provided: ModelCapabilities) -> bool:
    if required.tools and not provided.tools:
        return False
    if required.structured and not provided.structured:
        return False
    if required.vision and not provided.vision:
        return False
    if required.reasoning and not provided.reasoning:
        return False
    return True


def score_models(
    candidates: list[ModelInfo],
    stats_store: StatsStore,
    priority: str = "balanced",
    *,
    required: ModelCapabilities | None = None,
) -> list[tuple[ModelInfo, float]]:
    """Score and rank candidate models. Returns sorted (model, score) tuples.

    When *required* is provided the score is penalised by the model's
    capability-specific success rate for each requested capability.
    """
    scored: list[tuple[ModelInfo, float]] = []
    for model in candidates:
        stats = stats_store.get_stats(model.id)
        score = _compute_score(model, stats, priority, required=required)
        scored.append((model, score))
    scored.sort(key=lambda x: x[1], reverse=True)
    return scored


def _compute_score(
    model: ModelInfo,
    stats: ModelRuntimeStats,
    priority: str,
    *,
    required: ModelCapabilities | None = None,
) -> float:
    # Get normalized components
    success = stats.success_rate
    latency = _normalize_latency(stats.avg_latency)
    tok_speed = _normalize_tok_speed(stats.avg_tokens_per_second)
    quality = model.quality_hint

    if priority == "fast":
        base = (
            0.40 * tok_speed
            + 0.30 * (1.0 - latency)
            + 0.20 * success
            + 0.10 * quality
        )
    elif priority == "quality":
        base = (
            0.40 * quality
            + 0.30 * success
            + 0.15 * tok_speed
            + 0.15 * (1.0 - latency)
        )
    else:  # balanced
        base = (
            0.25 * quality
            + 0.25 * success
            + 0.25 * tok_speed
            + 0.25 * (1.0 - latency)
        )

    # Apply capability-specific penalties
    if required is not None:
        if required.structured:
            base *= stats.structured_success_rate
        if required.tools:
            base *= stats.tool_success_rate
        if required.vision:
            base *= stats.vision_success_rate

    return base


def _normalize_latency(latency: float | None) -> float:
    """Normalize latency to 0-1 range. Lower is better.

    Default 2.0s is neutral (0.5). 0.5s is fast (0.9). 5s+ is slow (0.1).
    """
    if latency is None:
        return 0.5
    # Clamp and normalize: 0.3s -> 0.95, 2s -> 0.5, 5s -> 0.1
    clamped = max(0.3, min(latency, 5.0))
    return 1.0 - (clamped - 0.3) / 4.7 * 0.85


def _normalize_tok_speed(tok_speed: float | None) -> float:
    """Normalize tokens/second to 0-1 range. Higher is better.

    Default 30 tok/s is neutral (0.5). 100+ is fast (0.9). 5 is slow (0.1).
    """
    if tok_speed is None:
        return 0.5
    clamped = max(5.0, min(tok_speed, 100.0))
    return 0.1 + (clamped - 5.0) / 95.0 * 0.8
