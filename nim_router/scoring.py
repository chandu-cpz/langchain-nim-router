from __future__ import annotations

from collections.abc import Collection

from nim_router.capabilities import capabilities_satisfy
from nim_router.limiter import RateLimiter
from nim_router.schemas import ModelCapabilities, ModelInfo, ModelRuntimeStats
from nim_router.stats import StatsStore


def filter_candidates(
    models: list[ModelInfo],
    required: ModelCapabilities,
    limiter: RateLimiter,
    stats_store: StatsStore,
    *,
    excluded_model_ids: Collection[str] = (),
) -> list[ModelInfo]:
    """Filter models by capability requirements and availability."""
    result: list[ModelInfo] = []
    for model in models:
        if model.id in excluded_model_ids:
            continue

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
        if not capabilities_satisfy(required, model.capabilities):
            continue

        result.append(model)
    return result


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


def prioritize_initial_exploration(
    candidates: list[ModelInfo],
    limiter: RateLimiter,
    stats_store: StatsStore,
    *,
    attempts_per_model: int,
) -> list[ModelInfo]:
    """Prefer eligible models not yet tried in this router's lifetime.

    Runtime stats and limiter state belong to one ``NimRouter`` instance. In
    the usual application integration that is one run, so this gives each
    eligible model a deterministic first attempt before normal score-based
    reuse begins. A request reservation counts as an attempt immediately so
    concurrent leases cannot repeatedly choose the same untried model.
    """
    if attempts_per_model <= 0:
        return candidates

    untried = [
        model
        for model in candidates
        if stats_store.get_stats(model.id).calls < attempts_per_model
        and len(limiter.get_state(model.id).recent_request_timestamps) < attempts_per_model
    ]
    return untried or candidates


def scheduled_exploration_candidate(
    candidates: list[ModelInfo],
    stats_store: StatsStore,
) -> ModelInfo:
    """Choose the least-observed eligible model for a scheduled probe."""
    return min(
        candidates,
        key=lambda model: (
            stats_store.get_stats(model.id).calls,
            stats_store.get_stats(model.id).last_used_at or 0.0,
            model.id,
        ),
    )


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
            + 0.30 * latency
            + 0.20 * success
            + 0.10 * quality
        )
    elif priority == "quality":
        base = (
            0.40 * quality
            + 0.30 * success
            + 0.15 * tok_speed
            + 0.15 * latency
        )
    else:  # balanced
        base = (
            0.25 * quality
            + 0.25 * success
            + 0.25 * tok_speed
            + 0.25 * latency
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
