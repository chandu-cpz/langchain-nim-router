from __future__ import annotations

import pytest

from nim_router.config import RouterConfig
from nim_router.limiter import RateLimiter
from nim_router.schemas import ModelCapabilities, ModelInfo
from nim_router.scoring import filter_candidates, score_models
from nim_router.stats import StatsStore


def _make_model(
    model_id: str,
    tools: bool = False,
    structured: bool = False,
    vision: bool = False,
    reasoning: bool = False,
    quality_hint: float = 0.5,
    deprecated: bool = False,
) -> ModelInfo:
    return ModelInfo(
        id=model_id,
        capabilities=ModelCapabilities(
            tools=tools, structured=structured, vision=vision, reasoning=reasoning
        ),
        quality_hint=quality_hint,
        deprecated=deprecated,
    )


def test_filter_by_tools():
    models = [
        _make_model("a", tools=True),
        _make_model("b", tools=False),
    ]
    required = ModelCapabilities(tools=True)
    config = RouterConfig()
    limiter = RateLimiter(config)
    stats = StatsStore()

    result = filter_candidates(models, required, limiter, stats)
    assert len(result) == 1
    assert result[0].id == "a"


def test_filter_by_vision():
    models = [
        _make_model("a", vision=True),
        _make_model("b", vision=False),
    ]
    required = ModelCapabilities(vision=True)
    config = RouterConfig()
    limiter = RateLimiter(config)
    stats = StatsStore()

    result = filter_candidates(models, required, limiter, stats)
    assert len(result) == 1
    assert result[0].id == "a"


def test_filter_excludes_deprecated():
    models = [
        _make_model("a", tools=True, deprecated=True),
        _make_model("b", tools=True),
    ]
    required = ModelCapabilities(tools=True)
    config = RouterConfig()
    limiter = RateLimiter(config)
    stats = StatsStore()

    result = filter_candidates(models, required, limiter, stats)
    assert len(result) == 1
    assert result[0].id == "b"


def test_filter_excludes_banned():
    models = [
        _make_model("a", tools=True),
        _make_model("b", tools=True),
    ]
    required = ModelCapabilities(tools=True)
    config = RouterConfig()
    limiter = RateLimiter(config)
    stats = StatsStore()
    stats.ban_model("a")

    result = filter_candidates(models, required, limiter, stats)
    assert len(result) == 1
    assert result[0].id == "b"


def test_filter_excludes_rate_limited():
    models = [
        _make_model("a", tools=True),
        _make_model("b", tools=True),
    ]
    required = ModelCapabilities(tools=True)
    config = RouterConfig(default_rpm=1)
    limiter = RateLimiter(config)
    stats = StatsStore()
    # Fill up the RPM for model a
    limiter.mark_request("a")
    limiter.mark_request("a")

    result = filter_candidates(models, required, limiter, stats)
    assert len(result) == 1
    assert result[0].id == "b"


def test_filter_excludes_cooling_down():
    models = [
        _make_model("a", tools=True),
        _make_model("b", tools=True),
    ]
    required = ModelCapabilities(tools=True)
    config = RouterConfig()
    limiter = RateLimiter(config)
    stats = StatsStore()
    stats.cooldown_model("a", 60.0)

    result = filter_candidates(models, required, limiter, stats)
    assert len(result) == 1
    assert result[0].id == "b"


def test_filter_impossible_capability():
    models = [
        _make_model("a", tools=True, vision=False),
    ]
    required = ModelCapabilities(vision=True)
    config = RouterConfig()
    limiter = RateLimiter(config)
    stats = StatsStore()

    result = filter_candidates(models, required, limiter, stats)
    assert len(result) == 0


def test_scoring_fast_prefers_speed():
    models = [
        _make_model("a", tools=True, quality_hint=0.9),
        _make_model("b", tools=True, quality_hint=0.3),
    ]
    stats = StatsStore()
    stats.record_success("a", latency=3.0, tokens_per_second=20.0)
    stats.record_success("b", latency=0.5, tokens_per_second=80.0)

    scored = score_models(models, stats, priority="fast")
    assert scored[0][0].id == "b"


def test_scoring_quality_prefers_quality():
    models = [
        _make_model("a", tools=True, quality_hint=0.9),
        _make_model("b", tools=True, quality_hint=0.3),
    ]
    stats = StatsStore()
    stats.record_success("a", latency=3.0, tokens_per_second=20.0)
    stats.record_success("b", latency=0.5, tokens_per_second=80.0)

    scored = score_models(models, stats, priority="quality")
    assert scored[0][0].id == "a"


def test_scoring_balanced_blends():
    models = [
        _make_model("a", tools=True, quality_hint=0.9),
        _make_model("b", tools=True, quality_hint=0.3),
    ]
    stats = StatsStore()
    stats.record_success("a", latency=3.0, tokens_per_second=20.0)
    stats.record_success("b", latency=0.5, tokens_per_second=80.0)

    scored = score_models(models, stats, priority="balanced")
    # Both should get reasonable scores; just check ordering is deterministic
    assert len(scored) == 2
    assert scored[0][1] >= scored[1][1]


def test_new_model_gets_neutral_defaults():
    models = [
        _make_model("a", tools=True),
    ]
    stats = StatsStore()
    scored = score_models(models, stats, priority="balanced")
    # New model should get neutral 0.5 score components, resulting in ~0.5 total
    assert len(scored) == 1
    score = scored[0][1]
    assert 0.4 <= score <= 0.6


def test_capability_penalty_structured():
    """Model with bad structured success rate loses when structured is required."""
    models = [
        _make_model("good", tools=True, structured=True, quality_hint=0.5),
        _make_model("bad", tools=True, structured=True, quality_hint=0.5),
    ]
    stats = StatsStore()
    # "good" has 100% structured success
    for _ in range(5):
        stats.record_success("good", latency=1.0, tokens_per_second=30.0, structured=True)
    # "bad" has 0% structured success
    for _ in range(5):
        stats.record_failure("bad", structured=True)

    required = ModelCapabilities(structured=True)
    scored = score_models(models, stats, priority="balanced", required=required)
    assert scored[0][0].id == "good"


def test_capability_penalty_tools():
    """Model with bad tool success rate loses when tools are required."""
    models = [
        _make_model("good", tools=True, quality_hint=0.5),
        _make_model("bad", tools=True, quality_hint=0.5),
    ]
    stats = StatsStore()
    for _ in range(5):
        stats.record_success("good", latency=1.0, tokens_per_second=30.0, tools=True)
    for _ in range(5):
        stats.record_failure("bad", tools=True)

    required = ModelCapabilities(tools=True)
    scored = score_models(models, stats, priority="balanced", required=required)
    assert scored[0][0].id == "good"


def test_no_penalty_without_required():
    """Without required caps, capability-specific stats don't affect scoring."""
    models = [
        _make_model("a", tools=True, structured=True, quality_hint=0.5),
        _make_model("b", tools=True, structured=True, quality_hint=0.5),
    ]
    stats = StatsStore()
    # Both have identical latency and tps, same overall success
    # but "a" has bad structured success rate
    stats.record_success("a", latency=1.0, tokens_per_second=30.0, structured=False)
    stats.record_success("b", latency=1.0, tokens_per_second=30.0, structured=True)

    scored = score_models(models, stats, priority="balanced")
    # Without required=, structured stats are ignored; identical models → tied scores
    assert scored[0][1] == pytest.approx(scored[1][1], abs=1e-6)
