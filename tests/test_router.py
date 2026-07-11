from __future__ import annotations

import json
import os
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from nim_router.callbacks import TrackingCallback
from nim_router.config import RouterConfig
from nim_router.errors import NoUsableModelError
from nim_router.router import NimRouter, _merge_config_with_callback
from nim_router.schemas import ModelCapabilities, ModelInfo, ModelSelection


@pytest.fixture
def mock_chat_nvidia_cls():
    """Provide a mock ChatNVIDIA class."""
    mock_cls = MagicMock()
    return mock_cls


@pytest.fixture
def router_with_mock(mock_chat_nvidia_cls):
    """Create a NimRouter with mocked ChatNVIDIA."""
    config = RouterConfig()
    with patch("nim_router.registry._get_chat_nvidia_cls", return_value=mock_chat_nvidia_cls):
        r = NimRouter(config=config)
        return r


def _make_fake_models():
    """Return fake model data for tests."""
    from tests.conftest import FAKE_MODELS

    return FAKE_MODELS


# ── pick ─────────────────────────────────────────────────────────────


class TestPick:
    @pytest.mark.asyncio
    async def test_pick_reports_initial_exploration_for_fully_untried_pool(
        self, router_with_mock, caplog
    ):
        router_with_mock.registry._loaded = True
        router_with_mock.registry._models = _build_model_infos()
        router_with_mock.stats_store.claim_exploration(
            router_with_mock.config.exploration_interval_seconds
        )

        with caplog.at_level("INFO", logger="nim_router.router"):
            await router_with_mock.pick(tools=True)

        assert "candidates=3, exploring=True" in caplog.text

    @pytest.mark.asyncio
    async def test_pick_tools_model(self, router_with_mock):
        router_with_mock.registry._loaded = True
        router_with_mock.registry._models = _build_model_infos()
        result = await router_with_mock.pick(tools=True)
        assert result.capabilities.tools is True

    @pytest.mark.asyncio
    async def test_pick_vision_model(self, router_with_mock):
        router_with_mock.registry._loaded = True
        router_with_mock.registry._models = _build_model_infos()
        result = await router_with_mock.pick(vision=True)
        assert result.capabilities.vision is True

    @pytest.mark.asyncio
    async def test_pick_reasoning_model(self, router_with_mock):
        router_with_mock.registry._loaded = True
        router_with_mock.registry._models = _build_model_infos()
        result = await router_with_mock.pick(reasoning=True)
        assert result.capabilities.reasoning is True

    @pytest.mark.asyncio
    async def test_pick_impossible_raises(self, router_with_mock):
        router_with_mock.registry._loaded = True
        router_with_mock.registry._models = _build_model_infos()
        with pytest.raises(NoUsableModelError):
            await router_with_mock.pick(tools=True, vision=True, reasoning=True)

    @pytest.mark.asyncio
    async def test_pick_excludes_banned(self, router_with_mock):
        router_with_mock.registry._loaded = True
        router_with_mock.registry._models = _build_model_infos()
        router_with_mock.ban_model("meta/llama-3.3-70b-instruct")
        router_with_mock.ban_model("meta/llama-3.1-8b-instruct")
        result = await router_with_mock.pick(tools=True)
        assert result.id == "nvidia/llama-3.1-nemotron-70b-instruct"

    @pytest.mark.asyncio
    async def test_pick_fast_prefers_speed(self, router_with_mock):
        router_with_mock.registry._loaded = True
        models = _build_model_infos()
        router_with_mock.registry._models = models

        router_with_mock.record_success(
            "meta/llama-3.1-8b-instruct", latency=0.5, tokens_per_second=80.0
        )
        router_with_mock.record_success(
            "meta/llama-3.3-70b-instruct", latency=3.0, tokens_per_second=20.0
        )
        router_with_mock.record_success(
            "nvidia/llama-3.1-nemotron-70b-instruct", latency=4.0, tokens_per_second=15.0
        )

        result = await router_with_mock.pick(tools=True, priority="fast")
        assert result.id == "meta/llama-3.1-8b-instruct"


# ── get ──────────────────────────────────────────────────────────────


class TestGet:
    @pytest.mark.asyncio
    async def test_get_returns_raw(self, router_with_mock):
        router_with_mock.registry._loaded = True
        router_with_mock.registry._models = _build_model_infos()

        with patch("nim_router.router.create_chat_nvidia") as mock_create:
            mock_llm = MagicMock()
            mock_create.return_value = mock_llm
            result = await router_with_mock.get(
                tools=True, temperature=0.7, top_p=0.9, max_completion_tokens=1024
            )
            assert result is mock_llm
            call_args = mock_create.call_args
            assert call_args[1]["model_id"] is not None
            assert call_args[1]["temperature"] == 0.7
            assert call_args[1]["top_p"] == 0.9
            assert call_args[1]["max_completion_tokens"] == 1024


# ── select ───────────────────────────────────────────────────────────


class TestSelect:
    @pytest.mark.asyncio
    async def test_select_returns_model_selection(self, router_with_mock):
        router_with_mock.registry._loaded = True
        router_with_mock.registry._models = _build_model_infos()

        with patch("nim_router.router.create_chat_nvidia") as mock_create:
            mock_llm = MagicMock()
            mock_create.return_value = mock_llm
            sel = await router_with_mock.select(tools=True, structured=True)

            assert isinstance(sel, ModelSelection)
            assert sel.info.capabilities.tools is True
            assert sel.llm is mock_llm
            assert isinstance(sel.callback, TrackingCallback)
            assert sel.callback.model_id == sel.info.id
            assert sel.callback._tools is True
            assert sel.callback._structured is True


# ── lease ────────────────────────────────────────────────────────────


class TestLease:
    @pytest.mark.asyncio
    async def test_lease_returns_model_selection(self, router_with_mock):
        router_with_mock.registry._loaded = True
        router_with_mock.registry._models = _build_model_infos()

        with patch("nim_router.router.create_chat_nvidia") as mock_create:
            mock_llm = MagicMock()
            mock_create.return_value = mock_llm
            sel = await router_with_mock.lease(tools=True, structured=True)

            assert isinstance(sel, ModelSelection)
            assert sel.info.capabilities.tools is True
            assert sel.llm is mock_llm
            assert isinstance(sel.callback, TrackingCallback)
            # lease uses mark_request_on_start=False so it never double-marks.
            assert sel.callback._mark_request_on_start is False

    @pytest.mark.asyncio
    async def test_lease_reserves_rpm_slot(self, router_with_mock):
        router_with_mock.registry._loaded = True
        router_with_mock.registry._models = _build_model_infos()

        with patch("nim_router.router.create_chat_nvidia", return_value=MagicMock()):
            sel = await router_with_mock.lease(tools=True)

        state = router_with_mock.limiter.get_state(sel.info.id)
        assert len(state.recent_request_timestamps) == 1

    @pytest.mark.asyncio
    async def test_lease_does_not_invoke_model(self, router_with_mock):
        router_with_mock.registry._loaded = True
        router_with_mock.registry._models = _build_model_infos()

        with patch("nim_router.router.create_chat_nvidia") as mock_create:
            mock_create.return_value = MagicMock()
            sel = await router_with_mock.lease(tools=True)

        # Lease must not call ainvoke/stream on the produced LLM.
        assert sel.llm.ainvoke.call_count == 0
        assert sel.llm.astream.call_count == 0


# ── tracker_for ──────────────────────────────────────────────────────


class TestTrackerFor:
    def test_tracker_for_string_id(self, router_with_mock):
        cb = router_with_mock.tracker_for("model-a", tools=True)
        assert isinstance(cb, TrackingCallback)
        assert cb.model_id == "model-a"
        assert cb._tools is True

    def test_tracker_for_model_info(self, router_with_mock):
        info = ModelInfo(id="model-b", capabilities=ModelCapabilities(vision=True))
        cb = router_with_mock.tracker_for(info, vision=True, reasoning=True)
        assert cb.model_id == "model-b"
        assert cb._vision is True
        assert cb._reasoning is True


# ── TrackingCallback ─────────────────────────────────────────────────


class TestTrackingCallback:
    @pytest.mark.asyncio
    async def test_callback_marks_limiter_on_start(self, router_with_mock):
        router_with_mock.registry._loaded = True
        router_with_mock.registry._models = _build_model_infos()

        cb = router_with_mock.tracker_for("model-a", tools=True)
        from uuid import uuid4
        run_id = uuid4()
        cb.on_chat_model_start({}, [], run_id=run_id)

        state = router_with_mock.limiter.get_state("model-a")
        assert len(state.recent_request_timestamps) == 1

    @pytest.mark.asyncio
    async def test_callback_marks_limiter_once_per_run(self, router_with_mock):
        cb = router_with_mock.tracker_for("model-a")
        from uuid import uuid4
        run_id = uuid4()
        cb.on_chat_model_start({}, [], run_id=run_id)
        cb.on_llm_start({}, [], run_id=run_id)

        state = router_with_mock.limiter.get_state("model-a")
        assert len(state.recent_request_timestamps) == 1

    @pytest.mark.asyncio
    async def test_callback_records_success_on_end(self, router_with_mock):
        router_with_mock.registry._loaded = True
        router_with_mock.registry._models = _build_model_infos()

        cb = router_with_mock.tracker_for("model-a", tools=True)
        from uuid import uuid4

        from langchain_core.outputs import Generation, LLMResult
        run_id = uuid4()
        cb.on_chat_model_start({}, [], run_id=run_id)

        # Simulate LLMResult with token usage
        llm_result = LLMResult(
            generations=[[Generation(text="hi")]],
            llm_output={"token_usage": {"completion_tokens": 50}},
        )
        cb.on_llm_end(llm_result, run_id=run_id)

        stats = router_with_mock.stats_store.get_stats("model-a")
        assert stats.successes >= 1
        assert stats.avg_latency is not None
        assert stats.avg_tokens_per_second is not None

    @pytest.mark.asyncio
    async def test_callback_records_failure_on_error(self, router_with_mock):
        router_with_mock.registry._loaded = True
        router_with_mock.registry._models = _build_model_infos()

        cb = router_with_mock.tracker_for("model-a")
        from uuid import uuid4
        run_id = uuid4()
        cb.on_chat_model_start({}, [], run_id=run_id)
        cb.on_llm_error(Exception("429 too many requests"), run_id=run_id)

        stats = router_with_mock.stats_store.get_stats("model-a")
        assert stats.failures >= 1
        assert router_with_mock.stats_store.is_cooling_down("model-a")

    @pytest.mark.asyncio
    async def test_callback_bans_on_404(self, router_with_mock):
        router_with_mock.registry._loaded = True
        router_with_mock.registry._models = _build_model_infos()

        cb = router_with_mock.tracker_for("model-a")
        from uuid import uuid4
        run_id = uuid4()
        cb.on_chat_model_start({}, [], run_id=run_id)
        cb.on_llm_error(Exception("model not found"), run_id=run_id)

        assert router_with_mock.stats_store.is_banned("model-a")


# ── ainvoke ──────────────────────────────────────────────────────────


class TestAinvoke:
    @pytest.mark.asyncio
    async def test_ainvoke_passes_callback_in_config(self, router_with_mock):
        router_with_mock.registry._loaded = True
        router_with_mock.registry._models = _build_model_infos()

        with patch("nim_router.router.create_chat_nvidia") as mock_create:
            mock_llm = MagicMock()
            mock_result = MagicMock()
            mock_llm.ainvoke = AsyncMock(return_value=mock_result)
            mock_create.return_value = mock_llm

            result = await router_with_mock.ainvoke(
                [{"role": "user", "content": "hi"}], tools=True
            )
            assert result is mock_result
            # Check that ainvoke passed config with callbacks
            call_kwargs = mock_llm.ainvoke.call_args
            config = call_kwargs[1]["config"]
            assert "callbacks" in config
            assert any(isinstance(cb, TrackingCallback) for cb in config["callbacks"])
            assert "tags" in config
            assert any("nim-router" in t for t in config["tags"])
            assert "metadata" in config
            assert config["metadata"]["nim_router_tools"] is True

    @pytest.mark.asyncio
    async def test_ainvoke_preserves_caller_callbacks(self, router_with_mock):
        router_with_mock.registry._loaded = True
        router_with_mock.registry._models = _build_model_infos()

        with patch("nim_router.router.create_chat_nvidia") as mock_create:
            mock_llm = MagicMock()
            mock_result = MagicMock()
            mock_llm.ainvoke = AsyncMock(return_value=mock_result)
            mock_create.return_value = mock_llm

            caller_cb = MagicMock()
            await router_with_mock.ainvoke(
                [{"role": "user", "content": "hi"}],
                tools=True,
                config={"callbacks": [caller_cb]},
            )
            config = mock_llm.ainvoke.call_args[1]["config"]
            cbs = config["callbacks"]
            assert caller_cb in cbs
            assert any(isinstance(cb, TrackingCallback) for cb in cbs)

    @pytest.mark.asyncio
    async def test_ainvoke_does_not_mutate_caller_config(self, router_with_mock):
        router_with_mock.registry._loaded = True
        router_with_mock.registry._models = _build_model_infos()

        with patch("nim_router.router.create_chat_nvidia") as mock_create:
            mock_llm = MagicMock()
            mock_llm.ainvoke = AsyncMock(return_value=MagicMock())
            mock_create.return_value = mock_llm

            original_config = {"callbacks": [MagicMock()], "tags": ["user-tag"]}
            await router_with_mock.ainvoke(
                [{"role": "user", "content": "hi"}],
                tools=True,
                config=original_config,
            )
            # Original config should not be mutated
            assert len(original_config["callbacks"]) == 1
            assert original_config["tags"] == ["user-tag"]


# ── config merge ─────────────────────────────────────────────────────


class TestConfigMerge:
    def test_merge_creates_callbacks(self):
        cb = MagicMock()
        info = ModelInfo(id="m1")
        caps = ModelCapabilities(tools=True)
        result = _merge_config_with_callback(None, cb, info, caps, "fast")
        assert cb in result["callbacks"]
        assert "nim-router" in result["tags"]
        assert result["metadata"]["nim_router_model_id"] == "m1"

    def test_merge_appends_to_existing_callbacks(self):
        cb = MagicMock()
        existing_cb = MagicMock()
        info = ModelInfo(id="m1")
        caps = ModelCapabilities()
        result = _merge_config_with_callback(
            {"callbacks": [existing_cb]}, cb, info, caps, "balanced"
        )
        assert existing_cb in result["callbacks"]
        assert cb in result["callbacks"]

    def test_merge_preserves_existing_tags(self):
        cb = MagicMock()
        info = ModelInfo(id="m1")
        caps = ModelCapabilities()
        result = _merge_config_with_callback(
            {"tags": ["user-tag"]}, cb, info, caps, "balanced"
        )
        assert "user-tag" in result["tags"]
        assert "nim-router" in result["tags"]

    def test_merge_preserves_existing_metadata(self):
        cb = MagicMock()
        info = ModelInfo(id="m1")
        caps = ModelCapabilities()
        result = _merge_config_with_callback(
            {"metadata": {"key": "val"}}, cb, info, caps, "balanced"
        )
        assert result["metadata"]["key"] == "val"
        assert result["metadata"]["nim_router_model_id"] == "m1"


# ── record failure / cooldown / ban ─────────────────────────────────


class TestRecordFailure:
    def test_rate_limit_cools_down(self, router_with_mock):
        router_with_mock.registry._loaded = True
        router_with_mock.registry._models = _build_model_infos()
        error = Exception("rate limit exceeded")
        router_with_mock.record_failure("meta/llama-3.3-70b-instruct", error=error)
        assert router_with_mock.stats_store.is_cooling_down("meta/llama-3.3-70b-instruct")

    def test_404_bans_model(self, router_with_mock):
        router_with_mock.registry._loaded = True
        router_with_mock.registry._models = _build_model_infos()
        error = Exception("model not found")
        router_with_mock.record_failure("meta/llama-3.3-70b-instruct", error=error)
        assert router_with_mock.stats_store.is_banned("meta/llama-3.3-70b-instruct")

    def test_rate_limit_does_not_ban(self, router_with_mock):
        router_with_mock.registry._loaded = True
        router_with_mock.registry._models = _build_model_infos()
        error = Exception("429 too many requests")
        router_with_mock.record_failure("meta/llama-3.3-70b-instruct", error=error)
        assert not router_with_mock.stats_store.is_banned("meta/llama-3.3-70b-instruct")
        assert router_with_mock.stats_store.is_cooling_down("meta/llama-3.3-70b-instruct")

    def test_429_with_status_code_cooldowns(self, router_with_mock):
        router_with_mock.registry._loaded = True
        router_with_mock.registry._models = _build_model_infos()

        class FakeHTTPError(Exception):
            status_code = 429

        error = FakeHTTPError("rate limit")
        router_with_mock.record_failure("meta/llama-3.3-70b-instruct", error=error)
        assert not router_with_mock.stats_store.is_banned("meta/llama-3.3-70b-instruct")


class TestFailureFallback:
    @pytest.mark.asyncio
    async def test_fallback_after_rate_limit(self, router_with_mock):
        router_with_mock.registry._loaded = True
        router_with_mock.registry._models = _build_model_infos()

        first = await router_with_mock.pick(tools=True)
        first_id = first.id
        router_with_mock.record_failure(first_id, error=Exception("429 too many requests"))

        second = await router_with_mock.pick(tools=True)
        assert second.id != first_id


class TestTimeoutCooldown:
    def test_timeout_causes_cooldown(self, router_with_mock):
        router_with_mock.registry._loaded = True
        router_with_mock.registry._models = _build_model_infos()
        router_with_mock.record_failure("meta/llama-3.3-70b-instruct", kind="timeout")
        assert router_with_mock.stats_store.is_cooling_down("meta/llama-3.3-70b-instruct")
        assert not router_with_mock.stats_store.is_banned("meta/llama-3.3-70b-instruct")

    @pytest.mark.asyncio
    async def test_timeout_excludes_model_from_pick(self, router_with_mock):
        router_with_mock.registry._loaded = True
        router_with_mock.registry._models = _build_model_infos()

        first = await router_with_mock.pick(tools=True)
        router_with_mock.record_failure(first.id, kind="timeout")

        second = await router_with_mock.pick(tools=True)
        assert second.id != first.id


# ── record success ───────────────────────────────────────────────────


class TestRecordSuccess:
    def test_record_success_updates_stats(self, router_with_mock):
        router_with_mock.record_success(
            "model-a", latency=1.5, tokens_per_second=45.0, structured=True, tools=True,
        )
        stats = router_with_mock.stats_store.get_stats("model-a")
        assert stats.calls == 1
        assert stats.successes == 1
        assert stats.structured_successes == 1
        assert stats.tool_successes == 1
        assert stats.avg_latency is not None
        assert stats.avg_tokens_per_second is not None


# ── ban / cooldown / stats ───────────────────────────────────────────


class TestBanAndCooldown:
    def test_ban_model(self, router_with_mock):
        router_with_mock.ban_model("model-a")
        assert router_with_mock.stats_store.is_banned("model-a")

    def test_cooldown_model(self, router_with_mock):
        router_with_mock.cooldown_model("model-a", 30.0)
        assert router_with_mock.stats_store.is_cooling_down("model-a")

    def test_stats(self, router_with_mock):
        router_with_mock.record_success("model-a")
        result = router_with_mock.stats()
        assert "model-a" in result


# ── convenience helpers ─────────────────────────────────────────────


class TestConvenienceHelpers:
    @pytest.mark.asyncio
    async def test_fast_tools_model(self, router_with_mock):
        router_with_mock.registry._loaded = True
        router_with_mock.registry._models = _build_model_infos()

        with patch("nim_router.router.create_chat_nvidia") as mock_create:
            mock_create.return_value = MagicMock()
            await router_with_mock.fast_tools_model()
            mock_create.assert_called_once()

    @pytest.mark.asyncio
    async def test_structured_model(self, router_with_mock):
        router_with_mock.registry._loaded = True
        router_with_mock.registry._models = _build_model_infos()

        with patch("nim_router.router.create_chat_nvidia") as mock_create:
            mock_create.return_value = MagicMock()
            await router_with_mock.structured_model()
            mock_create.assert_called_once()

    @pytest.mark.asyncio
    async def test_vision_model(self, router_with_mock):
        router_with_mock.registry._loaded = True
        router_with_mock.registry._models = _build_model_infos()

        with patch("nim_router.router.create_chat_nvidia") as mock_create:
            mock_create.return_value = MagicMock()
            await router_with_mock.vision_model()
            mock_create.assert_called_once()

    @pytest.mark.asyncio
    async def test_reasoning_model(self, router_with_mock):
        router_with_mock.registry._loaded = True
        router_with_mock.registry._models = _build_model_infos()

        with patch("nim_router.router.create_chat_nvidia") as mock_create:
            mock_create.return_value = MagicMock()
            await router_with_mock.reasoning_model()
            mock_create.assert_called_once()


# ── env config ───────────────────────────────────────────────────────


class TestEnvConfig:
    def test_model_pool_restriction(self):
        env = {"NIM_ROUTER_MODEL_POOL": "model-a,model-b"}
        with patch.dict(os.environ, env, clear=False):
            for key in list(os.environ):
                if key.startswith("NIM_ROUTER_") and key not in env:
                    del os.environ[key]
            config = RouterConfig.from_env()
            assert config.model_pool == ["model-a", "model-b"]

    def test_excluded_models(self):
        env = {"NIM_ROUTER_EXCLUDED_MODELS": "bad-model"}
        with patch.dict(os.environ, env, clear=False):
            for key in list(os.environ):
                if key.startswith("NIM_ROUTER_") and key not in env:
                    del os.environ[key]
            config = RouterConfig.from_env()
            assert "bad-model" in config.excluded_models

    def test_capability_overrides(self):
        caps = {
            "model-a": {"tools": True, "structured": False, "vision": False, "reasoning": False}
        }
        env = {"NIM_ROUTER_CAPABILITIES_JSON": json.dumps(caps)}
        with patch.dict(os.environ, env, clear=False):
            for key in list(os.environ):
                if key.startswith("NIM_ROUTER_") and key not in env:
                    del os.environ[key]
            config = RouterConfig.from_env()
            assert config.capabilities_overrides["model-a"]["tools"] is True

    def test_quality_hints(self):
        hints = {"model-a": 0.95}
        env = {"NIM_ROUTER_QUALITY_HINTS_JSON": json.dumps(hints)}
        with patch.dict(os.environ, env, clear=False):
            for key in list(os.environ):
                if key.startswith("NIM_ROUTER_") and key not in env:
                    del os.environ[key]
            config = RouterConfig.from_env()
            assert config.quality_hints["model-a"] == 0.95


# ── allow_undiscovered ───────────────────────────────────────────────


class TestAllowUndiscovered:
    @pytest.mark.asyncio
    async def test_override_only_model_not_added_by_default(self, mock_chat_nvidia_cls):
        config = RouterConfig(
            capabilities_overrides={"phantom/model": {"tools": True}},
            allow_undiscovered_models=False,
        )
        with patch("nim_router.registry._get_chat_nvidia_cls", return_value=mock_chat_nvidia_cls):
            r = NimRouter(config=config)
            await r.registry.ensure_loaded()
            assert not any(m.id == "phantom/model" for m in r.registry.models)

    @pytest.mark.asyncio
    async def test_override_only_model_added_when_enabled(self, mock_chat_nvidia_cls):
        config = RouterConfig(
            capabilities_overrides={"phantom/model": {"tools": True}},
            allow_undiscovered_models=True,
        )
        with patch("nim_router.registry._get_chat_nvidia_cls", return_value=mock_chat_nvidia_cls):
            r = NimRouter(config=config)
            await r.registry.ensure_loaded()
            assert any(m.id == "phantom/model" for m in r.registry.models)


# ── helpers ──────────────────────────────────────────────────────────


def _build_model_infos() -> list[ModelInfo]:
    """Build ModelInfo objects matching the fake models in conftest."""
    return [
        ModelInfo(
            id="meta/llama-3.3-70b-instruct",
            capabilities={"tools": True, "structured": True, "vision": False, "reasoning": False},
            quality_hint=0.5,
        ),
        ModelInfo(
            id="meta/llama-3.1-8b-instruct",
            capabilities={"tools": True, "structured": True, "vision": False, "reasoning": False},
            quality_hint=0.5,
        ),
        ModelInfo(
            id="nvidia/llama-3.1-nemotron-70b-instruct",
            capabilities={"tools": True, "structured": True, "vision": False, "reasoning": False},
            quality_hint=0.5,
        ),
        ModelInfo(
            id="meta/llama-3.2-11b-vision-instruct",
            capabilities={"tools": False, "structured": False, "vision": True, "reasoning": False},
            quality_hint=0.5,
        ),
        ModelInfo(
            id="nvidia/nemotron-3-nano-30b-a3b",
            capabilities={"tools": False, "structured": True, "vision": False, "reasoning": True},
            quality_hint=0.5,
        ),
    ]
