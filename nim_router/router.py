from __future__ import annotations

import asyncio
import logging
from typing import Any, Literal

from nim_router.callbacks import TrackingCallback
from nim_router.capabilities import capabilities_satisfy as _capabilities_satisfy
from nim_router.client import create_chat_nvidia
from nim_router.config import RouterConfig
from nim_router.errors import ErrorKind, NoUsableModelError, classify_error
from nim_router.limiter import RateLimiter
from nim_router.registry import ModelRegistry
from nim_router.schemas import ModelCapabilities, ModelInfo, ModelSelection
from nim_router.scoring import filter_candidates, prioritize_initial_exploration, score_models
from nim_router.stats import StatsStore

logger = logging.getLogger(__name__)

# Default cooldown periods
_RATE_LIMIT_COOLDOWN = 30.0
_HTTP_ERROR_COOLDOWN = 10.0
_TIMEOUT_COOLDOWN = 20.0
_MODEL_NOT_FOUND_BAN = True


class NimRouter:
    """Select the best NVIDIA NIM model based on capabilities and runtime history.

    Core methods:

    * ``pick(...)`` — pure selection, returns ``ModelInfo``.
    * ``get(...)`` — select and return a real ``ChatNVIDIA``.
    * ``select(...)`` — select, create LLM *and* tracking callback.
    * ``ainvoke(...)`` — one-shot: select + invoke + auto-track.
    """

    def __init__(
        self,
        config: RouterConfig | None = None,
        stats_path: str | None = None,
        **config_overrides: Any,
    ) -> None:
        if config is not None:
            self.config = config
        else:
            self.config = RouterConfig.from_env()

        for key, val in config_overrides.items():
            if hasattr(self.config, key):
                setattr(self.config, key, val)

        if stats_path is not None:
            self.config.stats_path = stats_path

        self.registry = ModelRegistry(self.config)
        self.limiter = RateLimiter(self.config)
        self.stats_store = StatsStore(stats_path=self.config.stats_path)
        self._pick_lock = asyncio.Lock()

    # ── Selection ────────────────────────────────────────────────────

    async def pick(
        self,
        *,
        tools: bool = False,
        structured: bool = False,
        vision: bool = False,
        reasoning: bool = False,
        priority: Literal["fast", "quality", "balanced"] = "balanced",
    ) -> ModelInfo:
        """Select the best model and return its metadata.

        Pure selection — no rate-limit reservation, no LLM creation.
        Use :meth:`select` or :meth:`ainvoke` for tracked invocations.
        """
        return await self._pick(
            tools=tools, structured=structured, vision=vision,
            reasoning=reasoning, priority=priority, reserve=False,
        )

    async def _pick(
        self,
        *,
        tools: bool,
        structured: bool,
        vision: bool,
        reasoning: bool,
        priority: Literal["fast", "quality", "balanced"],
        reserve: bool,
    ) -> ModelInfo:
        """Internal pick with optional atomic rate-limit reservation.

        When *reserve* is True the winning model's RPM slot is marked
        inside the pick lock so concurrent ``ainvoke()`` calls cannot
        all select the same model before any callback fires.
        """
        async with self._pick_lock:
            models = await self.registry.ensure_loaded()

            required = ModelCapabilities(
                tools=tools,
                structured=structured,
                vision=vision,
                reasoning=reasoning,
            )

            candidates = filter_candidates(
                models, required, self.limiter, self.stats_store
            )

            if not candidates:
                reasons = self._build_exclusion_reasons(models, required)
                req_dict = required.model_dump()
                raise NoUsableModelError(
                    f"No usable model found for capabilities: {req_dict}. "
                    f"Exclusion reasons: {reasons}",
                    required_capabilities=req_dict,
                    excluded_reasons=reasons,
                )

            exploration_candidates = prioritize_initial_exploration(
                candidates,
                self.limiter,
                self.stats_store,
                attempts_per_model=self.config.initial_exploration_attempts,
            )
            scored = score_models(
                exploration_candidates, self.stats_store, priority, required=required
            )
            best = scored[0][0]

            if reserve:
                self.limiter.mark_request(best.id)

            logger.info(
                "Selected model %s (priority=%s, tools=%s, structured=%s, vision=%s, "
                "reasoning=%s, candidates=%d, exploring=%s)",
                best.id,
                priority,
                tools,
                structured,
                vision,
                reasoning,
                len(candidates),
                len(exploration_candidates) < len(candidates),
            )

            return best

    async def get(
        self,
        *,
        tools: bool = False,
        structured: bool = False,
        vision: bool = False,
        reasoning: bool = False,
        priority: Literal["fast", "quality", "balanced"] = "balanced",
        max_completion_tokens: int | None = None,
        temperature: float | None = None,
        top_p: float | None = None,
        model_kwargs: dict[str, Any] | None = None,
    ) -> Any:
        """Pick a model and return a real ``ChatNVIDIA`` instance.

        The returned object is a genuine LangChain ``BaseChatModel`` —
        works with ``with_structured_output``, ``bind_tools``, LCEL
        pipes, LangGraph, ``astream_events``, and ``isinstance`` checks.

        No automatic tracking — use :meth:`select` or :meth:`ainvoke`
        if you need stats.
        """
        info = await self.pick(
            tools=tools,
            structured=structured,
            vision=vision,
            reasoning=reasoning,
            priority=priority,
        )
        return create_chat_nvidia(
            model_id=info.id,
            temperature=temperature,
            top_p=top_p,
            max_completion_tokens=max_completion_tokens,
            timeout=self.config.timeout_seconds,
            patch_timeout=self.config.patch_timeout,
            model_kwargs=model_kwargs,
        )

    async def select(
        self,
        *,
        tools: bool = False,
        structured: bool = False,
        vision: bool = False,
        reasoning: bool = False,
        priority: Literal["fast", "quality", "balanced"] = "balanced",
        max_completion_tokens: int | None = None,
        temperature: float | None = None,
        top_p: float | None = None,
        model_kwargs: dict[str, Any] | None = None,
    ) -> ModelSelection:
        """Pick a model, create LLM *and* tracking callback.

        Returns a :class:`ModelSelection` bundling:
        * ``info`` — ``ModelInfo`` metadata
        * ``llm`` — real ``ChatNVIDIA``
        * ``callback`` — ``TrackingCallback`` for automatic stats

        Pass ``selection.callback`` in LangChain's ``config={"callbacks": [...]}``
        when invoking the LLM (or a chain built from it).
        """
        info = await self.pick(
            tools=tools,
            structured=structured,
            vision=vision,
            reasoning=reasoning,
            priority=priority,
        )
        llm = create_chat_nvidia(
            model_id=info.id,
            temperature=temperature,
            top_p=top_p,
            max_completion_tokens=max_completion_tokens,
            timeout=self.config.timeout_seconds,
            patch_timeout=self.config.patch_timeout,
            model_kwargs=model_kwargs,
        )
        callback = self.tracker_for(
            info,
            tools=tools,
            structured=structured,
            vision=vision,
            reasoning=reasoning,
        )
        return ModelSelection(info=info, llm=llm, callback=callback)

    async def lease(
        self,
        *,
        tools: bool = False,
        structured: bool = False,
        vision: bool = False,
        reasoning: bool = False,
        priority: Literal["fast", "quality", "balanced"] = "balanced",
        max_completion_tokens: int | None = None,
        temperature: float | None = None,
        top_p: float | None = None,
        model_kwargs: dict[str, Any] | None = None,
    ) -> ModelSelection:
        """Select a model and create LLM + tracking callback *without* invoking.

        Returns a :class:`ModelSelection` for use by callers (e.g. middleware)
        that drive the actual model call themselves and want to record
        success/failure explicitly.

        Differences from :meth:`select`:

        * Reserves a rate-limit slot atomically inside the pick lock
          (``reserve=True``) so concurrent leases distribute across models.
        * The returned callback uses ``mark_request_on_start=False`` — the
          slot is already reserved, so the callback must not mark again.
        * Does **not** invoke the model and does **not** release RPM slots
          afterward. RPM accounting is request-window based; failed requests
          still count.
        """
        info = await self._pick(
            tools=tools,
            structured=structured,
            vision=vision,
            reasoning=reasoning,
            priority=priority,
            reserve=True,
        )
        llm = create_chat_nvidia(
            model_id=info.id,
            temperature=temperature,
            top_p=top_p,
            max_completion_tokens=max_completion_tokens,
            timeout=self.config.timeout_seconds,
            patch_timeout=self.config.patch_timeout,
            model_kwargs=model_kwargs,
        )
        callback = TrackingCallback(
            self,
            info.id,
            tools=tools,
            structured=structured,
            vision=vision,
            reasoning=reasoning,
            mark_request_on_start=False,
        )
        return ModelSelection(info=info, llm=llm, callback=callback)

    def tracker_for(
        self,
        model: str | ModelInfo,
        *,
        tools: bool = False,
        structured: bool = False,
        vision: bool = False,
        reasoning: bool = False,
    ) -> TrackingCallback:
        """Return a :class:`TrackingCallback` bound to a specific model.

        *model* can be a model ID string or a ``ModelInfo`` object.
        """
        model_id = model if isinstance(model, str) else model.id
        return TrackingCallback(
            self,
            model_id,
            tools=tools,
            structured=structured,
            vision=vision,
            reasoning=reasoning,
        )

    # ── One-shot invocation ──────────────────────────────────────────

    async def ainvoke(
        self,
        messages: list[Any],
        *,
        tools: bool = False,
        structured: bool = False,
        vision: bool = False,
        reasoning: bool = False,
        priority: Literal["fast", "quality", "balanced"] = "balanced",
        config: dict[str, Any] | None = None,
        max_completion_tokens: int | None = None,
        temperature: float | None = None,
        top_p: float | None = None,
        model_kwargs: dict[str, Any] | None = None,
    ) -> Any:
        """One-shot: select, invoke, and auto-track.

        * Reserves a rate-limit slot atomically inside the pick lock so
          concurrent ``ainvoke()`` calls distribute across models.
        * Creates a callback that does NOT re-mark the request on start
          (the slot is already reserved).
        * Merges caller config with the callback, tags, and metadata.
        * Invokes the LLM.  Stats recording is handled by the callback.
        """
        required = ModelCapabilities(
            tools=tools, structured=structured, vision=vision, reasoning=reasoning,
        )
        info = await self._pick(
            tools=tools, structured=structured, vision=vision,
            reasoning=reasoning, priority=priority, reserve=True,
        )
        llm = create_chat_nvidia(
            model_id=info.id,
            temperature=temperature,
            top_p=top_p,
            max_completion_tokens=max_completion_tokens,
            timeout=self.config.timeout_seconds,
            patch_timeout=self.config.patch_timeout,
            model_kwargs=model_kwargs,
        )
        callback = TrackingCallback(
            self, info.id,
            tools=tools, structured=structured,
            vision=vision, reasoning=reasoning,
            mark_request_on_start=False,
        )
        merged = _merge_config_with_callback(config, callback, info, required, priority)
        return await llm.ainvoke(messages, config=merged)

    # ── Manual recording ─────────────────────────────────────────────

    def record_success(
        self,
        model_id: str,
        latency: float | None = None,
        tokens_per_second: float | None = None,
        time_to_first_token: float | None = None,
        structured: bool | None = None,
        tools: bool | None = None,
        vision: bool | None = None,
    ) -> None:
        """Record a successful call for the given model."""
        self.stats_store.record_success(
            model_id,
            latency=latency,
            tokens_per_second=tokens_per_second,
            time_to_first_token=time_to_first_token,
            structured=structured,
            tools=tools,
            vision=vision,
        )

    def record_failure(
        self,
        model_id: str,
        error: BaseException | None = None,
        kind: ErrorKind | str | None = None,
        structured: bool | None = None,
        tools: bool | None = None,
        vision: bool | None = None,
    ) -> None:
        """Record a failure and apply appropriate cooldown/ban.

        This is the single place that classifies errors and applies
        automatic cooldowns/bans.  Called by the tracking callback on
        ``on_llm_error``.
        """
        if isinstance(error, BaseException) and kind is None:
            kind = classify_error(error)
        elif isinstance(kind, str):
            kind = ErrorKind(kind)

        self.stats_store.record_failure(
            model_id,
            error=error,
            kind=kind.value if kind else None,
            structured=structured,
            tools=tools,
            vision=vision,
        )

        if kind == ErrorKind.RATE_LIMIT:
            self.limiter.mark_rate_limited(model_id)
            self.limiter.cooldown(model_id, _RATE_LIMIT_COOLDOWN)
            self.stats_store.cooldown_model(model_id, _RATE_LIMIT_COOLDOWN)
        elif kind == ErrorKind.MODEL_NOT_FOUND:
            self.stats_store.ban_model(model_id)
            logger.warning("Model %s banned (model not found)", model_id)
        elif kind == ErrorKind.TIMEOUT:
            self.limiter.cooldown(model_id, _TIMEOUT_COOLDOWN)
            self.stats_store.cooldown_model(model_id, _TIMEOUT_COOLDOWN)
        elif kind == ErrorKind.HTTP_ERROR:
            self.limiter.cooldown(model_id, _HTTP_ERROR_COOLDOWN)
            self.stats_store.cooldown_model(model_id, _HTTP_ERROR_COOLDOWN)

    # ── Admin overrides ──────────────────────────────────────────────

    def ban_model(self, model_id: str) -> None:
        """Manually ban a model for this process."""
        self.stats_store.ban_model(model_id)

    def cooldown_model(self, model_id: str, seconds: float) -> None:
        """Manually cool down a model for the given duration."""
        self.limiter.cooldown(model_id, seconds)
        self.stats_store.cooldown_model(model_id, seconds)

    def stats(self) -> dict[str, Any]:
        """Return a snapshot of all runtime stats."""
        raw = self.stats_store.snapshot()
        return {k: v.model_dump() for k, v in raw.items()}

    # ── Convenience helpers ──────────────────────────────────────────

    async def fast_tools_model(self, **kwargs: Any) -> Any:
        """Get a fast model with tool support."""
        return await self.get(tools=True, priority="fast", **kwargs)

    async def structured_model(self, **kwargs: Any) -> Any:
        """Get a model optimized for structured output."""
        return await self.get(structured=True, priority="balanced", **kwargs)

    async def vision_model(self, **kwargs: Any) -> Any:
        """Get a model with vision support."""
        return await self.get(vision=True, priority="balanced", **kwargs)

    async def reasoning_model(self, **kwargs: Any) -> Any:
        """Get a model with reasoning/thinking support."""
        return await self.get(reasoning=True, priority="quality", **kwargs)

    # ── Internal ─────────────────────────────────────────────────────

    def _build_exclusion_reasons(
        self, models: list[ModelInfo], required: ModelCapabilities
    ) -> dict[str, list[str]]:
        reasons: dict[str, list[str]] = {}
        for model in models:
            excluded: list[str] = []
            if model.deprecated:
                excluded.append("deprecated")
            if self.stats_store.is_banned(model.id):
                excluded.append("banned")
            if self.stats_store.is_cooling_down(model.id):
                excluded.append("cooling_down")
            if not self.limiter.is_available(model.id):
                excluded.append("rate_limited")
            if not _capabilities_satisfy(required, model.capabilities):
                missing = []
                if required.tools and not model.capabilities.tools:
                    missing.append("tools")
                if required.structured and not model.capabilities.structured:
                    missing.append("structured")
                if required.vision and not model.capabilities.vision:
                    missing.append("vision")
                if required.reasoning and not model.capabilities.reasoning:
                    missing.append("reasoning")
                excluded.append(f"missing_capabilities: {missing}")
            if excluded:
                reasons[model.id] = excluded
        return reasons


def _merge_config_with_callback(
    config: dict[str, Any] | None,
    callback: TrackingCallback,
    model_info: ModelInfo,
    required: ModelCapabilities,
    priority: str,
) -> dict[str, Any]:
    """Build a LangChain RunnableConfig dict that includes the tracking
    callback, tags, and metadata — without mutating the caller's config.

    Uses shallow copy so callback objects, locks, and clients are not
    cloned (deepcopy can break those).
    """
    merged: dict[str, Any] = dict(config) if config else {}

    # ── callbacks ────────────────────────────────────────────────────
    existing = merged.get("callbacks")
    if existing is None:
        merged["callbacks"] = [callback]
    elif isinstance(existing, list):
        merged["callbacks"] = list(existing) + [callback]
    else:
        merged["callbacks"] = [existing, callback]

    # ── tags ─────────────────────────────────────────────────────────
    tags = [
        "nim-router",
        f"nim-model:{model_info.id}",
    ]
    if required.tools:
        tags.append("nim-tools")
    if required.structured:
        tags.append("nim-structured")
    if required.vision:
        tags.append("nim-vision")
    if required.reasoning:
        tags.append("nim-reasoning")
    tags.append(f"nim-priority:{priority}")

    existing_tags = merged.get("tags")
    if isinstance(existing_tags, list):
        merged["tags"] = list(existing_tags) + tags
    else:
        merged["tags"] = tags

    # ── metadata ─────────────────────────────────────────────────────
    meta = {
        "nim_router_model_id": model_info.id,
        "nim_router_priority": priority,
        "nim_router_tools": required.tools,
        "nim_router_structured": required.structured,
        "nim_router_vision": required.vision,
        "nim_router_reasoning": required.reasoning,
    }
    existing_meta = merged.get("metadata")
    if isinstance(existing_meta, dict):
        merged["metadata"] = {**existing_meta, **meta}
    else:
        merged["metadata"] = meta

    return merged
