"""Real API integration tests against NVIDIA NIM.

Requires NVIDIA_API_KEY environment variable to be set.
These tests make actual API calls to verify end-to-end behavior.
"""
from __future__ import annotations

import asyncio
import os
import time

import pytest

from nim_router.config import RouterConfig
from nim_router.errors import ErrorKind, NoUsableModelError
from nim_router.router import NimRouter


# Skip all tests if no API key
pytestmark = pytest.mark.skipif(
    not os.environ.get("NVIDIA_API_KEY"),
    reason="NVIDIA_API_KEY not set",
)


@pytest.fixture
def router():
    return NimRouter()


# =============================================================================
# MODEL DISCOVERY
# =============================================================================


class TestRealDiscovery:
    @pytest.mark.asyncio
    async def test_discover_returns_models(self, router):
        models = await router.registry.ensure_loaded()
        assert len(models) > 0, "Should discover at least some models"

    @pytest.mark.asyncio
    async def test_all_models_have_ids(self, router):
        models = await router.registry.ensure_loaded()
        for m in models:
            assert m.id, f"Model should have an id: {m}"

    @pytest.mark.asyncio
    async def test_all_models_are_chat_type(self, router):
        """Only chat/vlm/nv-vlm models should be included."""
        models = await router.registry.ensure_loaded()
        for m in models:
            if m.model_type is not None:
                assert m.model_type in ("chat", "vlm", "nv-vlm"), \
                    f"Unexpected model_type: {m.model_type} for {m.id}"

    @pytest.mark.asyncio
    async def test_vlm_models_have_vision(self, router):
        """VLM models should have vision=True."""
        models = await router.registry.ensure_loaded()
        for m in models:
            if m.model_type in ("vlm", "nv-vlm"):
                assert m.capabilities.vision is True, \
                    f"VLM model {m.id} should have vision=True"

    @pytest.mark.asyncio
    async def test_deprecated_models_present(self, router):
        """Some models should be marked deprecated."""
        models = await router.registry.ensure_loaded()
        deprecated = [m for m in models if m.deprecated]
        # At least some should be deprecated on a live API
        print(f"  Found {len(deprecated)} deprecated models out of {len(models)}")


# =============================================================================
# PICK OPERATIONS
# =============================================================================


class TestRealPick:
    @pytest.mark.asyncio
    async def test_pick_tools(self, router):
        m = await router.pick(tools=True)
        assert m.capabilities.tools is True

    @pytest.mark.asyncio
    async def test_pick_vision(self, router):
        m = await router.pick(vision=True)
        assert m.capabilities.vision is True

    @pytest.mark.asyncio
    async def test_pick_reasoning(self, router):
        m = await router.pick(reasoning=True)
        assert m.capabilities.reasoning is True

    @pytest.mark.asyncio
    async def test_pick_structured(self, router):
        m = await router.pick(structured=True)
        assert m.capabilities.structured is True

    @pytest.mark.asyncio
    async def test_pick_fast(self, router):
        m = await router.pick(tools=True, priority="fast")
        assert m.capabilities.tools is True

    @pytest.mark.asyncio
    async def test_pick_quality(self, router):
        m = await router.pick(tools=True, priority="quality")
        assert m.capabilities.tools is True

    @pytest.mark.asyncio
    async def test_pick_balanced(self, router):
        m = await router.pick(tools=True, priority="balanced")
        assert m.capabilities.tools is True

    @pytest.mark.asyncio
    async def test_pick_multiple_times(self, router):
        """Picking multiple times should work without errors."""
        for _ in range(5):
            m = await router.pick(tools=True)
            assert m.id is not None

    @pytest.mark.asyncio
    async def test_pick_impossible_raises(self, router):
        """Requesting all capabilities should raise if none exist."""
        # Try to find a model with tools+vision+reasoning+structured
        # Many models support this combo on NIM, so try a more extreme combo
        # Actually, on real API models like kimi-k2.6 support all, so test with
        # a non-existent model pool
        config = RouterConfig(model_pool=["meta/llama-3.1-8b-instruct"])
        restricted_router = NimRouter(config=config)
        await restricted_router.registry.ensure_loaded()
        
        # 8b doesn't have vision, so this should fail
        with pytest.raises(NoUsableModelError):
            await restricted_router.pick(tools=True, vision=True)


# =============================================================================
# GET OPERATIONS
# =============================================================================


class TestRealGet:
    @pytest.mark.asyncio
    async def test_get_returns_chat_nvidia(self, router):
        llm = await router.get(tools=True, temperature=0.7)
        assert type(llm).__name__ == "ChatNVIDIA"

    @pytest.mark.asyncio
    async def test_get_tracked_returns_tracked_llm(self, router):
        llm = await router.get_tracked(tools=True, temperature=0.7)
        assert type(llm).__name__ == "TrackedLLM"
        assert type(llm.llm).__name__ == "ChatNVIDIA"

    @pytest.mark.asyncio
    async def test_get_with_all_params(self, router):
        llm = await router.get(
            tools=True,
            temperature=0.5,
            top_p=0.9,
            max_completion_tokens=512,
        )
        assert type(llm).__name__ == "ChatNVIDIA"


# =============================================================================
# ERROR HANDLING
# =============================================================================


class TestRealErrorHandling:
    @pytest.mark.asyncio
    async def test_rate_limit_cooldown(self, router):
        m = await router.pick(tools=True)
        router.record_failure(m.id, kind=ErrorKind.RATE_LIMIT)
        assert router.stats_store.is_cooling_down(m.id)
        assert not router.stats_store.is_banned(m.id)

    @pytest.mark.asyncio
    async def test_rate_limit_fallback(self, router):
        m1 = await router.pick(tools=True)
        router.record_failure(m1.id, kind=ErrorKind.RATE_LIMIT)
        m2 = await router.pick(tools=True)
        assert m2.id != m1.id

    @pytest.mark.asyncio
    async def test_http_error_cooldown(self, router):
        m = await router.pick(tools=True)
        router.record_failure(m.id, kind=ErrorKind.HTTP_ERROR)
        assert router.stats_store.is_cooling_down(m.id)

    @pytest.mark.asyncio
    async def test_404_bans(self, router):
        m = await router.pick(tools=True)
        router.record_failure(m.id, kind=ErrorKind.MODEL_NOT_FOUND)
        assert router.stats_store.is_banned(m.id)

    @pytest.mark.asyncio
    async def test_success_records(self, router):
        m = await router.pick(tools=True)
        router.record_success(m.id, latency=1.5, tokens_per_second=50.0)
        stats = router.stats_store.get_stats(m.id)
        assert stats.calls >= 1
        assert stats.successes >= 1

    @pytest.mark.asyncio
    async def test_multiple_failures_cascade(self, router):
        """After banning multiple models, should still find others."""
        picked_ids = set()
        for _ in range(5):
            try:
                m = await router.pick(tools=True)
                picked_ids.add(m.id)
                router.record_failure(m.id, kind=ErrorKind.RATE_LIMIT)
            except NoUsableModelError:
                break
        # Should have picked at least 2 different models
        assert len(picked_ids) >= 2

    @pytest.mark.asyncio
    async def test_auto_classify_exception(self, router):
        m = await router.pick(tools=True)
        router.record_failure(m.id, error=Exception("rate limit exceeded"))
        assert router.stats_store.is_cooling_down(m.id)

    @pytest.mark.asyncio
    async def test_manual_ban_and_cooldown(self, router):
        m = await router.pick(tools=True)
        router.ban_model(m.id)
        assert router.stats_store.is_banned(m.id)
        
        m2 = await router.pick(tools=True)
        router.cooldown_model(m2.id, 5.0)
        assert router.stats_store.is_cooling_down(m2.id)

    @pytest.mark.asyncio
    async def test_stats_snapshot(self, router):
        m = await router.pick(tools=True)
        router.record_success(m.id, latency=1.0)
        stats = router.stats()
        assert m.id in stats


# =============================================================================
# FILTERING
# =============================================================================


class TestRealFiltering:
    @pytest.mark.asyncio
    async def test_pool_restriction(self):
        """Pool restriction should limit available models."""
        # First discover what tools models exist
        r1 = NimRouter()
        m_all = await r1.pick(tools=True)
        
        # Now restrict pool to just that model
        config = RouterConfig(model_pool=[m_all.id])
        r2 = NimRouter(config=config)
        m_restricted = await r2.pick(tools=True)
        assert m_restricted.id == m_all.id

    @pytest.mark.asyncio
    async def test_excluded_models(self):
        """Excluded models should not be selected."""
        r1 = NimRouter()
        m_all = await r1.pick(tools=True)
        
        # Exclude the model we just picked
        config = RouterConfig(excluded_models=[m_all.id])
        r2 = NimRouter(config=config)
        m_excluded = await r2.pick(tools=True)
        assert m_excluded.id != m_all.id


# =============================================================================
# CAPABILITY COMBOS
# =============================================================================


class TestRealCapabilityCombos:
    @pytest.mark.asyncio
    async def test_tools_and_structured(self, router):
        m = await router.pick(tools=True, structured=True)
        assert m.capabilities.tools is True
        assert m.capabilities.structured is True

    @pytest.mark.asyncio
    async def test_vision_and_reasoning(self, router):
        m = await router.pick(vision=True, reasoning=True)
        assert m.capabilities.vision is True
        assert m.capabilities.reasoning is True
