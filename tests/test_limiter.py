from __future__ import annotations

import time

from nim_router.config import RouterConfig
from nim_router.limiter import RateLimiter


def test_model_becomes_unavailable_after_rpm_limit():
    config = RouterConfig(default_rpm=2)
    limiter = RateLimiter(config)

    assert limiter.is_available("model-a") is True
    limiter.mark_request("model-a")
    assert limiter.is_available("model-a") is True
    limiter.mark_request("model-a")
    assert limiter.is_available("model-a") is False


def test_different_models_independent():
    config = RouterConfig(default_rpm=1)
    limiter = RateLimiter(config)

    limiter.mark_request("model-a")
    assert limiter.is_available("model-a") is False
    assert limiter.is_available("model-b") is True


def test_cooldown_blocks_model():
    config = RouterConfig(default_rpm=100)
    limiter = RateLimiter(config)

    limiter.cooldown("model-a", 60.0)
    assert limiter.is_available("model-a") is False


def test_cooldown_clear():
    config = RouterConfig(default_rpm=100)
    limiter = RateLimiter(config)

    limiter.cooldown("model-a", 0.0)
    # Even 0 seconds cooldown should pass since cooldown_until <= now
    time.sleep(0.01)
    assert limiter.is_available("model-a") is True


def test_rate_limited_count_increments():
    config = RouterConfig(default_rpm=1)
    limiter = RateLimiter(config)

    limiter.mark_rate_limited("model-a")
    limiter.mark_rate_limited("model-a")
    state = limiter.get_state("model-a")
    assert state.rate_limited_count == 2


def test_custom_rpm_per_model():
    config = RouterConfig(
        default_rpm=10,
        model_rpm={"slow-model": 3},
    )
    limiter = RateLimiter(config)

    for _ in range(3):
        limiter.mark_request("slow-model")
    assert limiter.is_available("slow-model") is False

    # fast-model still uses default
    for _ in range(3):
        limiter.mark_request("fast-model")
    assert limiter.is_available("fast-model") is True


def test_snapshot_returns_all_states():
    config = RouterConfig(default_rpm=10)
    limiter = RateLimiter(config)

    limiter.mark_request("a")
    limiter.mark_request("b")
    snap = limiter.snapshot()
    assert "a" in snap
    assert "b" in snap


def test_clear_cooldown():
    config = RouterConfig(default_rpm=100)
    limiter = RateLimiter(config)

    limiter.cooldown("model-a", 60.0)
    assert limiter.is_available("model-a") is False

    limiter.clear_cooldown("model-a")
    assert limiter.is_available("model-a") is True
