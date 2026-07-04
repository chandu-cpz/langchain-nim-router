"""Exhaustive tests for model discovery, registry filtering, and scoring edge cases."""
from __future__ import annotations

import json
import os
import time
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from nim_router.config import RouterConfig
from nim_router.errors import ErrorKind
from nim_router.limiter import RateLimiter
from nim_router.registry import ModelRegistry
from nim_router.schemas import ModelCapabilities, ModelInfo, ModelRuntimeStats
from nim_router.scoring import (
    _compute_score,
    _normalize_latency,
    _normalize_tok_speed,
    filter_candidates,
    score_models,
)
from nim_router.stats import StatsStore


# =============================================================================
# MODEL DISCOVERY FILTERING
# =============================================================================


class TestRegistryDiscoveryFiltering:
    """Test registry.py model filtering logic."""

    @pytest.mark.asyncio
    async def test_filters_non_chat_client(self):
        config = RouterConfig()
        registry = ModelRegistry(config)
        fake_models = [
            MagicMock(id="model-a", model_type="chat", client="ChatNVIDIA",
                      supports_tools=False, supports_structured_output=False,
                      supports_thinking=False, deprecated=False,
                      base_model=None, aliases=None),
            MagicMock(id="model-b", model_type="chat", client="OtherClient",
                      supports_tools=False, supports_structured_output=False,
                      supports_thinking=False, deprecated=False,
                      base_model=None, aliases=None),
        ]
        with patch("nim_router.registry._get_chat_nvidia_cls") as mock_cls:
            mock_cls.return_value.get_available_models = MagicMock(return_value=fake_models)
            models = await registry.discover()
        ids = [m.id for m in models]
        assert "model-a" in ids
        assert "model-b" not in ids

    @pytest.mark.asyncio
    async def test_filters_non_chat_model_types(self):
        config = RouterConfig()
        registry = ModelRegistry(config)
        fake_models = [
            MagicMock(id="chat-model", model_type="chat", client="ChatNVIDIA",
                      supports_tools=False, supports_structured_output=False,
                      supports_thinking=False, deprecated=False,
                      base_model=None, aliases=None),
            MagicMock(id="vlm-model", model_type="vlm", client="ChatNVIDIA",
                      supports_tools=False, supports_structured_output=False,
                      supports_thinking=False, deprecated=False,
                      base_model=None, aliases=None),
            MagicMock(id="nv-vlm-model", model_type="nv-vlm", client="ChatNVIDIA",
                      supports_tools=False, supports_structured_output=False,
                      supports_thinking=False, deprecated=False,
                      base_model=None, aliases=None),
            MagicMock(id="embedding-model", model_type="embedding", client="ChatNVIDIA",
                      supports_tools=False, supports_structured_output=False,
                      supports_thinking=False, deprecated=False,
                      base_model=None, aliases=None),
            MagicMock(id="rerank-model", model_type="rerank", client="ChatNVIDIA",
                      supports_tools=False, supports_structured_output=False,
                      supports_thinking=False, deprecated=False,
                      base_model=None, aliases=None),
            MagicMock(id="completion-model", model_type="completion", client="ChatNVIDIA",
                      supports_tools=False, supports_structured_output=False,
                      supports_thinking=False, deprecated=False,
                      base_model=None, aliases=None),
        ]
        with patch("nim_router.registry._get_chat_nvidia_cls") as mock_cls:
            mock_cls.return_value.get_available_models = MagicMock(return_value=fake_models)
            models = await registry.discover()
        ids = [m.id for m in models]
        assert "chat-model" in ids
        assert "vlm-model" in ids
        assert "nv-vlm-model" in ids
        assert "embedding-model" not in ids
        assert "rerank-model" not in ids
        assert "completion-model" not in ids

    @pytest.mark.asyncio
    async def test_allows_models_with_no_client_type(self):
        config = RouterConfig()
        registry = ModelRegistry(config)
        fake_models = [
            MagicMock(id="unknown-client", model_type="chat", client=None,
                      supports_tools=False, supports_structured_output=False,
                      supports_thinking=False, deprecated=False,
                      base_model=None, aliases=None),
        ]
        with patch("nim_router.registry._get_chat_nvidia_cls") as mock_cls:
            mock_cls.return_value.get_available_models = MagicMock(return_value=fake_models)
            models = await registry.discover()
        assert len(models) == 1
        assert models[0].id == "unknown-client"

    @pytest.mark.asyncio
    async def test_allows_models_with_no_model_type(self):
        config = RouterConfig()
        registry = ModelRegistry(config)
        fake_models = [
            MagicMock(id="no-type", model_type=None, client="ChatNVIDIA",
                      supports_tools=False, supports_structured_output=False,
                      supports_thinking=False, deprecated=False,
                      base_model=None, aliases=None),
        ]
        with patch("nim_router.registry._get_chat_nvidia_cls") as mock_cls:
            mock_cls.return_value.get_available_models = MagicMock(return_value=fake_models)
            models = await registry.discover()
        assert len(models) == 1
        assert models[0].id == "no-type"

    @pytest.mark.asyncio
    async def test_skips_models_with_no_id(self):
        config = RouterConfig()
        registry = ModelRegistry(config)
        fake_models = [
            MagicMock(id=None, model_type="chat", client="ChatNVIDIA",
                      supports_tools=False, supports_structured_output=False,
                      supports_thinking=False, deprecated=False,
                      base_model=None, aliases=None),
            MagicMock(id="", model_type="chat", client="ChatNVIDIA",
                      supports_tools=False, supports_structured_output=False,
                      supports_thinking=False, deprecated=False,
                      base_model=None, aliases=None),
            MagicMock(id=123, model_type="chat", client="ChatNVIDIA",
                      supports_tools=False, supports_structured_output=False,
                      supports_thinking=False, deprecated=False,
                      base_model=None, aliases=None),
            MagicMock(id="valid-model", model_type="chat", client="ChatNVIDIA",
                      supports_tools=False, supports_structured_output=False,
                      supports_thinking=False, deprecated=False,
                      base_model=None, aliases=None),
        ]
        with patch("nim_router.registry._get_chat_nvidia_cls") as mock_cls:
            mock_cls.return_value.get_available_models = MagicMock(return_value=fake_models)
            models = await registry.discover()
        assert len(models) == 1
        assert models[0].id == "valid-model"

    @pytest.mark.asyncio
    async def test_marks_deprecated_models(self):
        config = RouterConfig()
        registry = ModelRegistry(config)
        fake_models = [
            MagicMock(id="old-model", model_type="chat", client="ChatNVIDIA",
                      supports_tools=False, supports_structured_output=False,
                      supports_thinking=False, deprecated=True,
                      base_model=None, aliases=None),
            MagicMock(id="new-model", model_type="chat", client="ChatNVIDIA",
                      supports_tools=False, supports_structured_output=False,
                      supports_thinking=False, deprecated=False,
                      base_model=None, aliases=None),
        ]
        with patch("nim_router.registry._get_chat_nvidia_cls") as mock_cls:
            mock_cls.return_value.get_available_models = MagicMock(return_value=fake_models)
            models = await registry.discover()
        old = next(m for m in models if m.id == "old-model")
        new = next(m for m in models if m.id == "new-model")
        assert old.deprecated is True
        assert new.deprecated is False

    @pytest.mark.asyncio
    async def test_pool_restriction(self):
        config = RouterConfig(model_pool=["model-a", "model-c"])
        registry = ModelRegistry(config)
        fake_models = [
            MagicMock(id="model-a", model_type="chat", client="ChatNVIDIA",
                      supports_tools=False, supports_structured_output=False,
                      supports_thinking=False, deprecated=False,
                      base_model=None, aliases=None),
            MagicMock(id="model-b", model_type="chat", client="ChatNVIDIA",
                      supports_tools=False, supports_structured_output=False,
                      supports_thinking=False, deprecated=False,
                      base_model=None, aliases=None),
            MagicMock(id="model-c", model_type="chat", client="ChatNVIDIA",
                      supports_tools=False, supports_structured_output=False,
                      supports_thinking=False, deprecated=False,
                      base_model=None, aliases=None),
        ]
        with patch("nim_router.registry._get_chat_nvidia_cls") as mock_cls:
            mock_cls.return_value.get_available_models = MagicMock(return_value=fake_models)
            models = await registry.discover()
        ids = [m.id for m in models]
        assert "model-a" in ids
        assert "model-c" in ids
        assert "model-b" not in ids

    @pytest.mark.asyncio
    async def test_exclusion(self):
        config = RouterConfig(excluded_models=["bad-model"])
        registry = ModelRegistry(config)
        fake_models = [
            MagicMock(id="good-model", model_type="chat", client="ChatNVIDIA",
                      supports_tools=False, supports_structured_output=False,
                      supports_thinking=False, deprecated=False,
                      base_model=None, aliases=None),
            MagicMock(id="bad-model", model_type="chat", client="ChatNVIDIA",
                      supports_tools=False, supports_structured_output=False,
                      supports_thinking=False, deprecated=False,
                      base_model=None, aliases=None),
        ]
        with patch("nim_router.registry._get_chat_nvidia_cls") as mock_cls:
            mock_cls.return_value.get_available_models = MagicMock(return_value=fake_models)
            models = await registry.discover()
        ids = [m.id for m in models]
        assert "good-model" in ids
        assert "bad-model" not in ids

    @pytest.mark.asyncio
    async def test_capability_overrides_from_config(self):
        config = RouterConfig(
            capabilities_overrides={"custom-model": {"tools": True, "reasoning": True}}
        )
        registry = ModelRegistry(config)
        fake_models = [
            MagicMock(id="custom-model", model_type="chat", client="ChatNVIDIA",
                      supports_tools=False, supports_structured_output=False,
                      supports_thinking=False, deprecated=False,
                      base_model=None, aliases=None),
        ]
        with patch("nim_router.registry._get_chat_nvidia_cls") as mock_cls:
            mock_cls.return_value.get_available_models = MagicMock(return_value=fake_models)
            models = await registry.discover()
        custom = next(m for m in models if m.id == "custom-model")
        assert custom.capabilities.tools is True
        assert custom.capabilities.reasoning is True

    @pytest.mark.asyncio
    async def test_quality_hints_applied(self):
        config = RouterConfig(quality_hints={"model-a": 0.95, "model-b": 0.1})
        registry = ModelRegistry(config)
        fake_models = [
            MagicMock(id="model-a", model_type="chat", client="ChatNVIDIA",
                      supports_tools=False, supports_structured_output=False,
                      supports_thinking=False, deprecated=False,
                      base_model=None, aliases=None),
            MagicMock(id="model-b", model_type="chat", client="ChatNVIDIA",
                      supports_tools=False, supports_structured_output=False,
                      supports_thinking=False, deprecated=False,
                      base_model=None, aliases=None),
        ]
        with patch("nim_router.registry._get_chat_nvidia_cls") as mock_cls:
            mock_cls.return_value.get_available_models = MagicMock(return_value=fake_models)
            models = await registry.discover()
        a = next(m for m in models if m.id == "model-a")
        b = next(m for m in models if m.id == "model-b")
        assert a.quality_hint == 0.95
        assert b.quality_hint == 0.1

    @pytest.mark.asyncio
    async def test_metadata_preserved(self):
        config = RouterConfig()
        registry = ModelRegistry(config)
        fake_models = [
            MagicMock(id="model-a", model_type="chat", client="ChatNVIDIA",
                      supports_tools=False, supports_structured_output=False,
                      supports_thinking=False, deprecated=False,
                      base_model="llama-3.3-70b", aliases=["alias1", "alias2"]),
        ]
        with patch("nim_router.registry._get_chat_nvidia_cls") as mock_cls:
            mock_cls.return_value.get_available_models = MagicMock(return_value=fake_models)
            models = await registry.discover()
        m = models[0]
        assert m.metadata["base_model"] == "llama-3.3-70b"
        assert m.metadata["aliases"] == ["alias1", "alias2"]

    @pytest.mark.asyncio
    async def test_discover_called_only_once(self):
        config = RouterConfig()
        registry = ModelRegistry(config)
        fake_models = [
            MagicMock(id="model-a", model_type="chat", client="ChatNVIDIA",
                      supports_tools=False, supports_structured_output=False,
                      supports_thinking=False, deprecated=False,
                      base_model=None, aliases=None),
        ]
        with patch("nim_router.registry._get_chat_nvidia_cls") as mock_cls:
            mock_cls.return_value.get_available_models = MagicMock(return_value=fake_models)
            await registry.ensure_loaded()
            await registry.ensure_loaded()
        mock_cls.return_value.get_available_models.assert_called_once()

    @pytest.mark.asyncio
    async def test_api_failure_raises_model_discovery_error(self):
        config = RouterConfig()
        registry = ModelRegistry(config)
        with patch("nim_router.registry._get_chat_nvidia_cls") as mock_cls:
            mock_cls.return_value.get_available_models = MagicMock(
                side_effect=RuntimeError("API down")
            )
            with pytest.raises(Exception, match="Failed to discover models"):
                await registry.discover()

    def test_get_model_returns_none(self):
        config = RouterConfig()
        registry = ModelRegistry(config)
        assert registry.get_model("unknown") is None

    def test_get_model_returns_model(self):
        config = RouterConfig()
        registry = ModelRegistry(config)
        registry._models = [ModelInfo(id="model-a"), ModelInfo(id="model-b")]
        result = registry.get_model("model-a")
        assert result is not None
        assert result.id == "model-a"


# =============================================================================
# SCORING NORMALIZATION EDGE CASES
# =============================================================================


class TestScoringNormalization:
    """Test normalization functions for edge cases.

    Actual formulas:
        latency:  1.0 - (clamped - 0.3) / 4.7 * 0.85
        tok_speed: 0.1 + (clamped - 5.0) / 95.0 * 0.8
    """

    def test_normalize_latency_none(self):
        assert _normalize_latency(None) == 0.5

    def test_normalize_latency_zero(self):
        # clamped to 0.3 -> 1.0 - 0/4.7*0.85 = 1.0
        assert _normalize_latency(0.0) == pytest.approx(1.0)

    def test_normalize_latency_03(self):
        # clamped to 0.3 -> 1.0 - 0 = 1.0
        assert _normalize_latency(0.3) == pytest.approx(1.0)

    def test_normalize_latency_20(self):
        # clamped to 2.0 -> 1.0 - (1.7/4.7)*0.85 = 1.0 - 0.3074 = 0.6926
        assert _normalize_latency(2.0) == pytest.approx(0.6926, abs=0.01)

    def test_normalize_latency_50(self):
        # clamped to 5.0 -> 1.0 - (4.7/4.7)*0.85 = 0.15
        assert _normalize_latency(5.0) == pytest.approx(0.15, abs=0.01)

    def test_normalize_latency_very_slow(self):
        # clamped to 5.0 -> 0.15
        assert _normalize_latency(10.0) == pytest.approx(0.15, abs=0.01)

    def test_normalize_latency_clamped_high(self):
        assert _normalize_latency(100.0) == pytest.approx(0.15, abs=0.01)

    def test_normalize_tok_speed_none(self):
        assert _normalize_tok_speed(None) == 0.5

    def test_normalize_tok_speed_zero(self):
        # clamped to 5.0 -> 0.1 + 0/95*0.8 = 0.1
        assert _normalize_tok_speed(0.0) == pytest.approx(0.1)

    def test_normalize_tok_speed_5(self):
        # clamped to 5.0 -> 0.1
        assert _normalize_tok_speed(5.0) == pytest.approx(0.1)

    def test_normalize_tok_speed_30(self):
        # clamped to 30 -> 0.1 + (25/95)*0.8 = 0.1 + 0.2105 = 0.3105
        assert _normalize_tok_speed(30.0) == pytest.approx(0.3105, abs=0.01)

    def test_normalize_tok_speed_100(self):
        # clamped to 100 -> 0.1 + (95/95)*0.8 = 0.9
        assert _normalize_tok_speed(100.0) == pytest.approx(0.9)

    def test_normalize_tok_speed_very_fast(self):
        # clamped to 100 -> 0.9
        assert _normalize_tok_speed(200.0) == pytest.approx(0.9)


# =============================================================================
# COMPUTE SCORE WITH DIFFERENT PRIORITIES
# =============================================================================


class TestComputeScore:
    """Test _compute_score with various stats combinations."""

    def test_new_model_balanced(self):
        model = ModelInfo(id="new", quality_hint=0.5)
        stats = ModelRuntimeStats()
        score = _compute_score(model, stats, "balanced")
        # 0.25*0.5 + 0.25*0.8 + 0.25*0.5 + 0.25*0.5 = 0.575
        assert score == pytest.approx(0.575, abs=0.01)

    def test_new_model_fast(self):
        model = ModelInfo(id="new", quality_hint=0.5)
        stats = ModelRuntimeStats()
        score = _compute_score(model, stats, "fast")
        # 0.40*0.5 + 0.30*0.5 + 0.20*0.8 + 0.10*0.5 = 0.56
        assert score == pytest.approx(0.56, abs=0.01)

    def test_new_model_quality(self):
        model = ModelInfo(id="new", quality_hint=0.5)
        stats = ModelRuntimeStats()
        score = _compute_score(model, stats, "quality")
        # 0.40*0.5 + 0.30*0.8 + 0.15*0.5 + 0.15*0.5 = 0.59
        assert score == pytest.approx(0.59, abs=0.01)

    def test_fast_model_beats_slow_model_in_fast_priority(self):
        fast = ModelInfo(id="fast", quality_hint=0.5)
        slow = ModelInfo(id="slow", quality_hint=0.5)
        fast_stats = ModelRuntimeStats(calls=10, successes=10, avg_latency=0.3, avg_tokens_per_second=100)
        slow_stats = ModelRuntimeStats(calls=10, successes=10, avg_latency=5.0, avg_tokens_per_second=5)
        assert _compute_score(fast, fast_stats, "fast") > _compute_score(slow, slow_stats, "fast")

    def test_quality_hint_matters_in_quality_priority(self):
        high_q = ModelInfo(id="high", quality_hint=0.9)
        low_q = ModelInfo(id="low", quality_hint=0.1)
        same_stats = ModelRuntimeStats(calls=10, successes=10, avg_latency=2.0, avg_tokens_per_second=30)
        assert _compute_score(high_q, same_stats, "quality") > _compute_score(low_q, same_stats, "quality")

    def test_success_rate_matters(self):
        good = ModelInfo(id="good", quality_hint=0.5)
        bad = ModelInfo(id="bad", quality_hint=0.5)
        good_stats = ModelRuntimeStats(calls=10, successes=9)
        bad_stats = ModelRuntimeStats(calls=10, successes=1)
        assert _compute_score(good, good_stats, "balanced") > _compute_score(bad, bad_stats, "balanced")


# =============================================================================
# STATS STORE PERSISTENCE
# =============================================================================


class TestStatsStorePersistence:
    def test_save_and_load(self, tmp_path):
        path = tmp_path / "stats.json"
        store1 = StatsStore(stats_path=str(path))
        store1.record_success("model-a", latency=1.5, tokens_per_second=45.0)
        store1.record_failure("model-b", kind="http_error")
        store2 = StatsStore(stats_path=str(path))
        assert store2.get_stats("model-a").calls == 1
        assert store2.get_stats("model-a").avg_latency == 1.5
        assert store2.get_stats("model-b").failures == 1
        assert store2.get_stats("model-b").http_errors == 1

    def test_load_corrupt_file(self, tmp_path):
        path = tmp_path / "stats.json"
        path.write_text("not valid json {{{")
        store = StatsStore(stats_path=str(path))
        assert store.get_stats("model-a").calls == 0

    def test_load_nonexistent_file(self, tmp_path):
        path = tmp_path / "nonexistent.json"
        store = StatsStore(stats_path=str(path))
        assert store.get_stats("model-a").calls == 0

    def test_no_persistence_when_no_path(self):
        store = StatsStore()
        store.record_success("model-a")
        assert store.get_stats("model-a").calls == 1

    def test_ban_persists(self, tmp_path):
        path = tmp_path / "stats.json"
        store1 = StatsStore(stats_path=str(path))
        store1.ban_model("model-a")
        store2 = StatsStore(stats_path=str(path))
        assert store2.is_banned("model-a") is True

    def test_cooldown_persists(self, tmp_path):
        path = tmp_path / "stats.json"
        store1 = StatsStore(stats_path=str(path))
        store1.cooldown_model("model-a", 300.0)
        store2 = StatsStore(stats_path=str(path))
        assert store2.is_cooling_down("model-a") is True


# =============================================================================
# EMA CALCULATION
# =============================================================================


class TestEMACalculation:
    def test_first_value(self):
        from nim_router.stats import _update_avg
        assert _update_avg(None, 10.0) == 10.0

    def test_subsequent_values(self):
        from nim_router.stats import _update_avg
        result = _update_avg(10.0, 20.0)
        expected = 10.0 * 0.7 + 20.0 * 0.3
        assert result == pytest.approx(expected)

    def test_many_updates_converge(self):
        from nim_router.stats import _update_avg
        current = 50.0
        for _ in range(100):
            current = _update_avg(current, 100.0)
        assert current > 90.0

    def test_stable_value(self):
        from nim_router.stats import _update_avg
        assert _update_avg(50.0, 50.0) == pytest.approx(50.0)


# =============================================================================
# RATE LIMITER TIMESTAMP CLEANUP
# =============================================================================


class TestLimiterTimestampCleanup:
    def test_old_timestamps_are_removed(self):
        config = RouterConfig(default_rpm=5)
        limiter = RateLimiter(config)
        state = limiter.get_state("model-a")
        state.recent_request_timestamps = [
            time.monotonic() - 120,
            time.monotonic() - 90,
            time.monotonic() - 10,
        ]
        assert limiter.is_available("model-a") is True
        assert len(state.recent_request_timestamps) == 1


# =============================================================================
# ALL ERROR KINDS IN RECORD_FAILURE
# =============================================================================


class TestRecordFailureAllKinds:
    @pytest.fixture
    def router(self):
        from nim_router.router import NimRouter
        config = RouterConfig()
        mock_cls = MagicMock()
        with patch("nim_router.registry._get_chat_nvidia_cls", return_value=mock_cls):
            return NimRouter(config=config)

    def test_rate_limit_error(self, router):
        router.registry._loaded = True
        router.registry._models = [ModelInfo(id="m1")]
        router.record_failure("m1", kind=ErrorKind.RATE_LIMIT)
        assert router.stats_store.is_cooling_down("m1")
        assert not router.stats_store.is_banned("m1")

    def test_http_error(self, router):
        router.registry._loaded = True
        router.registry._models = [ModelInfo(id="m1")]
        router.record_failure("m1", kind=ErrorKind.HTTP_ERROR)
        assert router.stats_store.is_cooling_down("m1")
        assert not router.stats_store.is_banned("m1")

    def test_model_not_found_error(self, router):
        router.registry._loaded = True
        router.registry._models = [ModelInfo(id="m1")]
        router.record_failure("m1", kind=ErrorKind.MODEL_NOT_FOUND)
        assert router.stats_store.is_banned("m1")

    def test_timeout_error(self, router):
        router.registry._loaded = True
        router.registry._models = [ModelInfo(id="m1")]
        router.record_failure("m1", kind=ErrorKind.TIMEOUT)
        assert not router.stats_store.is_cooling_down("m1")
        assert not router.stats_store.is_banned("m1")

    def test_generic_error(self, router):
        router.registry._loaded = True
        router.registry._models = [ModelInfo(id="m1")]
        router.record_failure("m1", kind=ErrorKind.GENERIC)
        assert not router.stats_store.is_cooling_down("m1")
        assert not router.stats_store.is_banned("m1")

    def test_structured_output_failure(self, router):
        router.registry._loaded = True
        router.registry._models = [ModelInfo(id="m1")]
        router.record_failure("m1", kind=ErrorKind.STRUCTURED_OUTPUT_FAILURE, structured=True)
        assert router.stats_store.get_stats("m1").structured_failures == 1

    def test_tool_call_failure(self, router):
        router.registry._loaded = True
        router.registry._models = [ModelInfo(id="m1")]
        router.record_failure("m1", kind=ErrorKind.TOOL_CALL_FAILURE, tools=True)
        assert router.stats_store.get_stats("m1").tool_failures == 1

    def test_vision_failure(self, router):
        router.registry._loaded = True
        router.registry._models = [ModelInfo(id="m1")]
        router.record_failure("m1", kind=ErrorKind.VISION_FAILURE, vision=True)
        assert router.stats_store.get_stats("m1").vision_failures == 1

    def test_string_kind_converted(self, router):
        router.registry._loaded = True
        router.registry._models = [ModelInfo(id="m1")]
        router.record_failure("m1", kind="rate_limit")
        assert router.stats_store.is_cooling_down("m1")

    def test_auto_classify_from_exception(self, router):
        router.registry._loaded = True
        router.registry._models = [ModelInfo(id="m1")]
        router.record_failure("m1", error=Exception("rate limit exceeded"))
        assert router.stats_store.is_cooling_down("m1")


# =============================================================================
# RECORD SUCCESS FULL
# =============================================================================


class TestRecordSuccessFull:
    def test_record_with_all_params(self):
        store = StatsStore()
        store.record_success("m1", latency=1.0, tokens_per_second=50.0,
                             time_to_first_token=0.2, structured=True, tools=True, vision=True)
        stats = store.get_stats("m1")
        assert stats.calls == 1
        assert stats.successes == 1
        assert stats.avg_latency == 1.0
        assert stats.avg_tokens_per_second == 50.0
        assert stats.time_to_first_token == 0.2
        assert stats.structured_successes == 1
        assert stats.tool_successes == 1
        assert stats.vision_successes == 1

    def test_record_without_optional_params(self):
        store = StatsStore()
        store.record_success("m1")
        stats = store.get_stats("m1")
        assert stats.calls == 1
        assert stats.successes == 1
        assert stats.avg_latency is None

    def test_multiple_successes_update_ema(self):
        store = StatsStore()
        store.record_success("m1", latency=1.0)
        store.record_success("m1", latency=3.0)
        # EMA: 1.0*0.7 + 3.0*0.3 = 1.6
        assert store.get_stats("m1").avg_latency == pytest.approx(1.6, abs=0.01)

    def test_record_failure_with_kind(self):
        store = StatsStore()
        store.record_failure("m1", kind="rate_limit")
        stats = store.get_stats("m1")
        assert stats.calls == 1
        assert stats.failures == 1
        assert stats.rate_limits == 1

    def test_record_failure_http_error(self):
        store = StatsStore()
        store.record_failure("m1", kind="http_error")
        assert store.get_stats("m1").http_errors == 1

    def test_record_failure_no_kind(self):
        store = StatsStore()
        store.record_failure("m1")
        stats = store.get_stats("m1")
        assert stats.calls == 1
        assert stats.failures == 1
        assert stats.rate_limits == 0
        assert stats.http_errors == 0


# =============================================================================
# CONFIG EDGE CASES
# =============================================================================


class TestConfigEdgeCases:
    def test_empty_csv_list(self):
        with patch.dict(os.environ, {"NIM_ROUTER_MODEL_POOL": ""}, clear=False):
            for key in list(os.environ):
                if key.startswith("NIM_ROUTER_") and key != "NIM_ROUTER_MODEL_POOL":
                    del os.environ[key]
            config = RouterConfig.from_env()
            assert config.model_pool == []

    def test_whitespace_csv_list(self):
        with patch.dict(os.environ, {"NIM_ROUTER_MODEL_POOL": "  ,  ,  "}, clear=False):
            for key in list(os.environ):
                if key.startswith("NIM_ROUTER_") and key != "NIM_ROUTER_MODEL_POOL":
                    del os.environ[key]
            config = RouterConfig.from_env()
            assert config.model_pool == []

    def test_negative_rpm(self):
        with patch.dict(os.environ, {"NIM_ROUTER_DEFAULT_RPM": "-5"}, clear=False):
            for key in list(os.environ):
                if key.startswith("NIM_ROUTER_") and key != "NIM_ROUTER_DEFAULT_RPM":
                    del os.environ[key]
            config = RouterConfig.from_env()
            assert config.default_rpm == -5

    def test_zero_timeout(self):
        with patch.dict(os.environ, {"NIM_ROUTER_TIMEOUT_SECONDS": "0"}, clear=False):
            for key in list(os.environ):
                if key.startswith("NIM_ROUTER_") and key != "NIM_ROUTER_TIMEOUT_SECONDS":
                    del os.environ[key]
            config = RouterConfig.from_env()
            assert config.timeout_seconds == 0.0

    def test_float_rpm_falls_back(self):
        with patch.dict(os.environ, {"NIM_ROUTER_DEFAULT_RPM": "30.5"}, clear=False):
            for key in list(os.environ):
                if key.startswith("NIM_ROUTER_") and key != "NIM_ROUTER_DEFAULT_RPM":
                    del os.environ[key]
            config = RouterConfig.from_env()
            assert config.default_rpm == 30

    def test_programmatic_config_overrides(self):
        config = RouterConfig(model_pool=["a", "b"], default_rpm=50)
        assert config.model_pool == ["a", "b"]
        assert config.default_rpm == 50

    def test_stats_path_config(self):
        config = RouterConfig(stats_path="/tmp/stats.json")
        assert config.stats_path == "/tmp/stats.json"


# =============================================================================
# CAPABILITIES SATISFY
# =============================================================================


class TestCapabilitiesSatisfy:
    def test_empty_required_satisfies_anything(self):
        from nim_router.capabilities import capabilities_satisfy
        assert capabilities_satisfy(ModelCapabilities(), ModelCapabilities(tools=True, vision=True))

    def test_tools_required_but_not_provided(self):
        from nim_router.capabilities import capabilities_satisfy
        assert not capabilities_satisfy(ModelCapabilities(tools=True), ModelCapabilities(tools=False))

    def test_structured_required_but_not_provided(self):
        from nim_router.capabilities import capabilities_satisfy
        assert not capabilities_satisfy(ModelCapabilities(structured=True), ModelCapabilities(structured=False))

    def test_vision_required_but_not_provided(self):
        from nim_router.capabilities import capabilities_satisfy
        assert not capabilities_satisfy(ModelCapabilities(vision=True), ModelCapabilities(vision=False))

    def test_reasoning_required_but_not_provided(self):
        from nim_router.capabilities import capabilities_satisfy
        assert not capabilities_satisfy(ModelCapabilities(reasoning=True), ModelCapabilities(reasoning=False))

    def test_all_capabilities_satisfied(self):
        from nim_router.capabilities import capabilities_satisfy
        req = ModelCapabilities(tools=True, structured=True, vision=True, reasoning=True)
        prov = ModelCapabilities(tools=True, structured=True, vision=True, reasoning=True)
        assert capabilities_satisfy(req, prov)

    def test_extra_provided_capabilities_ok(self):
        from nim_router.capabilities import capabilities_satisfy
        assert capabilities_satisfy(ModelCapabilities(tools=True),
                                    ModelCapabilities(tools=True, vision=True, reasoning=True))
