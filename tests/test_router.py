from __future__ import annotations

import json
import os
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from nim_router.config import RouterConfig
from nim_router.errors import NoUsableModelError
from nim_router.router import NimRouter
from nim_router.schemas import ModelInfo


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


class TestPick:
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
        # Ban two tools-capable models, one remains
        router_with_mock.ban_model("meta/llama-3.3-70b-instruct")
        router_with_mock.ban_model("meta/llama-3.1-8b-instruct")
        result = await router_with_mock.pick(tools=True)
        assert result.id == "nvidia/llama-3.1-nemotron-70b-instruct"

    @pytest.mark.asyncio
    async def test_pick_fast_prefers_speed(self, router_with_mock):
        router_with_mock.registry._loaded = True
        models = _build_model_infos()
        router_with_mock.registry._models = models

        # Give 8b model fast stats, 70b model slow stats, nemotron neutral
        router_with_mock.record_success(
            "meta/llama-3.1-8b-instruct", latency=0.5, tokens_per_second=80.0
        )
        router_with_mock.record_success(
            "meta/llama-3.3-70b-instruct", latency=3.0, tokens_per_second=20.0
        )
        # Give nemotron slow stats too so it doesn't win on neutral defaults
        router_with_mock.record_success(
            "nvidia/llama-3.1-nemotron-70b-instruct", latency=4.0, tokens_per_second=15.0
        )

        result = await router_with_mock.pick(tools=True, priority="fast")
        # The faster model should be preferred
        assert result.id == "meta/llama-3.1-8b-instruct"


class TestRecordFailure:
    def test_rate_limit_cools_down(self, router_with_mock):
        router_with_mock.registry._loaded = True
        router_with_mock.registry._models = _build_model_infos()

        error = Exception("rate limit exceeded")
        router_with_mock.record_failure("meta/llama-3.3-70b-instruct", error=error)

        # Model should be cooling down
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

        # Should cooldown, not ban
        assert not router_with_mock.stats_store.is_banned("meta/llama-3.3-70b-instruct")
        assert router_with_mock.stats_store.is_cooling_down("meta/llama-3.3-70b-instruct")

    def test_429_bans_model(self, router_with_mock):
        router_with_mock.registry._loaded = True
        router_with_mock.registry._models = _build_model_infos()

        class FakeHTTPError(Exception):
            status_code = 429

        error = FakeHTTPError("rate limit")
        router_with_mock.record_failure("meta/llama-3.3-70b-instruct", error=error)

        # 429 with status_code should not ban
        assert not router_with_mock.stats_store.is_banned("meta/llama-3.3-70b-instruct")


class TestFailureFallback:
    @pytest.mark.asyncio
    async def test_fallback_after_rate_limit(self, router_with_mock):
        router_with_mock.registry._loaded = True
        router_with_mock.registry._models = _build_model_infos()

        # First pick
        first = await router_with_mock.pick(tools=True)
        first_id = first.id

        # Record 429 failure
        router_with_mock.record_failure(first_id, error=Exception("429 too many requests"))

        # Next pick should exclude the rate-limited model
        second = await router_with_mock.pick(tools=True)
        assert second.id != first_id


class TestRecordSuccess:
    def test_record_success_updates_stats(self, router_with_mock):
        router_with_mock.record_success(
            "model-a",
            latency=1.5,
            tokens_per_second=45.0,
            structured=True,
            tools=True,
        )
        stats = router_with_mock.stats_store.get_stats("model-a")
        assert stats.calls == 1
        assert stats.successes == 1
        assert stats.structured_successes == 1
        assert stats.tool_successes == 1
        assert stats.avg_latency is not None
        assert stats.avg_tokens_per_second is not None


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


class TestGetRaw:
    @pytest.mark.asyncio
    async def test_get_returns_bare_llm(self, router_with_mock):
        router_with_mock.registry._loaded = True
        router_with_mock.registry._models = _build_model_infos()

        with patch("nim_router.router.create_chat_nvidia") as mock_create:
            mock_llm = MagicMock()
            mock_create.return_value = mock_llm
            result = await router_with_mock.get(tools=True)
            assert result is mock_llm


class TestAinvoke:
    @pytest.mark.asyncio
    async def test_ainvoke_auto_records_success(self, router_with_mock):
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
            # Stats should have been recorded
            model_id = mock_create.call_args[1]["model_id"]
            stats = router_with_mock.stats_store.get_stats(model_id)
            assert stats.calls >= 1
            assert stats.successes >= 1

    @pytest.mark.asyncio
    async def test_ainvoke_auto_records_failure(self, router_with_mock):
        router_with_mock.registry._loaded = True
        router_with_mock.registry._models = _build_model_infos()

        with patch("nim_router.router.create_chat_nvidia") as mock_create:
            mock_llm = MagicMock()
            mock_llm.ainvoke = AsyncMock(
                side_effect=Exception("rate limit exceeded")
            )
            mock_create.return_value = mock_llm

            with pytest.raises(Exception, match="rate limit"):
                await router_with_mock.ainvoke(
                    [{"role": "user", "content": "hi"}], tools=True
                )
            model_id = mock_create.call_args[1]["model_id"]
            stats = router_with_mock.stats_store.get_stats(model_id)
            assert stats.failures >= 1
            # Should also have triggered a cooldown
            assert router_with_mock.stats_store.is_cooling_down(model_id)


class TestTrackedLLM:
    @pytest.mark.asyncio
    async def test_tracked_ainvoke_records_success(self, router_with_mock):
        router_with_mock.registry._loaded = True
        router_with_mock.registry._models = _build_model_infos()

        with patch("nim_router.router.create_chat_nvidia") as mock_create:
            mock_llm = MagicMock()
            mock_result = MagicMock()
            mock_llm.ainvoke = AsyncMock(return_value=mock_result)
            mock_create.return_value = mock_llm

            tracked = await router_with_mock.get_tracked(tools=True)
            result = await tracked.ainvoke([{"role": "user", "content": "hi"}])
            assert result is mock_result
            stats = router_with_mock.stats_store.get_stats(tracked.model_id)
            assert stats.successes >= 1

    @pytest.mark.asyncio
    async def test_tracked_ainvoke_records_failure(self, router_with_mock):
        router_with_mock.registry._loaded = True
        router_with_mock.registry._models = _build_model_infos()

        with patch("nim_router.router.create_chat_nvidia") as mock_create:
            mock_llm = MagicMock()
            mock_llm.ainvoke = AsyncMock(side_effect=Exception("429"))
            mock_create.return_value = mock_llm

            tracked = await router_with_mock.get_tracked(tools=True)
            with pytest.raises(Exception, match="429"):
                await tracked.ainvoke([{"role": "user", "content": "hi"}])
            stats = router_with_mock.stats_store.get_stats(tracked.model_id)
            assert stats.failures >= 1

    @pytest.mark.asyncio
    async def test_tracked_delegates_attributes(self, router_with_mock):
        router_with_mock.registry._loaded = True
        router_with_mock.registry._models = _build_model_infos()

        with patch("nim_router.router.create_chat_nvidia") as mock_create:
            mock_llm = MagicMock()
            mock_llm.custom_attr = "hello"
            mock_create.return_value = mock_llm

            tracked = await router_with_mock.get_tracked(tools=True)
            assert tracked.custom_attr == "hello"

    @pytest.mark.asyncio
    async def test_tracked_with_structured_output_wraps(self, router_with_mock):
        router_with_mock.registry._loaded = True
        router_with_mock.registry._models = _build_model_infos()

        with patch("nim_router.router.create_chat_nvidia") as mock_create:
            mock_llm = MagicMock()
            mock_structured = MagicMock()
            mock_structured.ainvoke = AsyncMock(return_value="ok")
            mock_llm.with_structured_output.return_value = mock_structured
            mock_create.return_value = mock_llm

            tracked = await router_with_mock.get_tracked(tools=True)
            result = tracked.with_structured_output({"type": "object"})
            from nim_router.router import TrackedLLM
            assert isinstance(result, TrackedLLM)
            assert result._structured is True
            # Invoke through the structured wrapper still tracks
            await result.ainvoke("hi")
            stats = router_with_mock.stats_store.get_stats(tracked.model_id)
            assert stats.successes >= 1

    @pytest.mark.asyncio
    async def test_tracked_bind_tools_wraps(self, router_with_mock):
        router_with_mock.registry._loaded = True
        router_with_mock.registry._models = _build_model_infos()

        with patch("nim_router.router.create_chat_nvidia") as mock_create:
            mock_llm = MagicMock()
            mock_bound = MagicMock()
            mock_bound.ainvoke = AsyncMock(return_value="ok")
            mock_llm.bind_tools.return_value = mock_bound
            mock_create.return_value = mock_llm

            tracked = await router_with_mock.get_tracked(tools=True)
            result = tracked.bind_tools([{"type": "function"}])
            from nim_router.router import TrackedLLM
            assert isinstance(result, TrackedLLM)
            assert result._tools is True

    @pytest.mark.asyncio
    async def test_tracked_invoke_marks_rate_limit(self, router_with_mock):
        router_with_mock.registry._loaded = True
        router_with_mock.registry._models = _build_model_infos()

        with patch("nim_router.router.create_chat_nvidia") as mock_create:
            mock_llm = MagicMock()
            mock_llm.invoke.return_value = "ok"
            mock_create.return_value = mock_llm

            tracked = await router_with_mock.get_tracked(tools=True)
            # pick() should NOT have marked a request
            state = router_with_mock.limiter.get_state(tracked.model_id)
            assert len(state.recent_request_timestamps) == 0
            # invoke() should mark a request
            tracked.invoke("hi")
            state = router_with_mock.limiter.get_state(tracked.model_id)
            assert len(state.recent_request_timestamps) == 1


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
            # get() returns bare ChatNVIDIA
            assert result is mock_llm
            call_args = mock_create.call_args
            assert call_args[1]["model_id"] is not None
            assert call_args[1]["temperature"] == 0.7
            assert call_args[1]["top_p"] == 0.9
            assert call_args[1]["max_completion_tokens"] == 1024

    @pytest.mark.asyncio
    async def test_get_tracked_returns_tracked(self, router_with_mock):
        router_with_mock.registry._loaded = True
        router_with_mock.registry._models = _build_model_infos()

        with patch("nim_router.router.create_chat_nvidia") as mock_create:
            mock_llm = MagicMock()
            mock_create.return_value = mock_llm
            result = await router_with_mock.get_tracked(tools=True)
            from nim_router.router import TrackedLLM
            assert isinstance(result, TrackedLLM)
            assert result.llm is mock_llm


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
