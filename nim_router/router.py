from __future__ import annotations

import asyncio
import logging
import time
from typing import Any, Literal

from nim_router.client import create_chat_nvidia
from nim_router.config import RouterConfig
from nim_router.errors import ErrorKind, NoUsableModelError, classify_error
from nim_router.limiter import RateLimiter
from nim_router.registry import ModelRegistry
from nim_router.schemas import ModelCapabilities, ModelInfo
from nim_router.scoring import filter_candidates, score_models
from nim_router.stats import StatsStore

logger = logging.getLogger(__name__)

# Default cooldown periods
_RATE_LIMIT_COOLDOWN = 30.0
_HTTP_ERROR_COOLDOWN = 10.0
_TIMEOUT_COOLDOWN = 20.0
_MODEL_NOT_FOUND_BAN = True


class NimRouter:
    """Select the best NVIDIA NIM model based on capabilities and runtime history."""

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
        """Pick a model and return a bare ChatNVIDIA instance.

        Returns the real langchain ChatNVIDIA object so it works everywhere
        LangChain expects a ``BaseChatModel`` (LangGraph, ``with_structured_output``,
        ``bind_tools``, ``|`` pipeline composition, etc.).

        Use :meth:`get_tracked` if you want automatic latency/error recording.
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
            model_kwargs=model_kwargs,
        )

    async def get_tracked(
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
    ) -> TrackedLLM:
        """Pick a model and return a TrackedLLM that auto-records every call.

        The returned wrapper records latency and errors on each
        ``invoke()`` / ``ainvoke()`` call.  It also reserves a rate-limit
        slot per actual invocation rather than at pick-time.
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
            model_kwargs=model_kwargs,
        )
        return TrackedLLM(
            llm=llm,
            router=self,
            model_id=info.id,
            structured=structured,
            tools=tools,
            vision=vision,
        )

    async def ainvoke(
        self,
        messages: list[Any],
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
        """Pick a model, invoke it, and auto-record success/failure.

        This is the recommended one-shot API: it reserves a rate-limit
        slot, measures latency, records stats, and applies cooldowns on
        failure — all in a single call.
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
            model_kwargs=model_kwargs,
        )
        self.limiter.mark_request(info.id)
        t0 = time.monotonic()
        try:
            result = await llm.ainvoke(messages)
            latency = time.monotonic() - t0
            self.record_success(
                info.id,
                latency=latency,
                structured=structured,
                tools=tools,
                vision=vision,
            )
            return result
        except Exception as exc:
            self.record_failure(
                info.id,
                error=exc,
                structured=structured,
                tools=tools,
                vision=vision,
            )
            raise

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

        Acquires an asyncio lock so concurrent callers don't double-reserve
        the same rate-limited model.

        Note: ``pick()`` does NOT reserve a rate-limit slot.  Reservation
        happens at invoke-time (inside ``TrackedLLM`` / ``ainvoke()``) so
        that unused picks don't waste quota.
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

            scored = score_models(
                candidates, self.stats_store, priority, required=required
            )
            return scored[0][0]

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
        """Record a failure and apply appropriate cooldown/ban."""
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


class TrackedLLM:
    """Wrapper around a ChatNVIDIA instance that auto-records success/failure.

    Each ``invoke()`` / ``ainvoke()`` call:
    1. Reserves a rate-limit slot (so repeated calls on the same handle
       each count against RPM).
    2. Measures wall-clock latency.
    3. Records success or failure back to the router's stats store.

    Composition methods (``with_structured_output``, ``bind_tools``) return
    new ``TrackedLLM`` wrappers so that tracking survives LangChain
    composition chains.
    """

    def __init__(
        self,
        llm: Any,
        router: NimRouter,
        model_id: str,
        *,
        structured: bool = False,
        tools: bool = False,
        vision: bool = False,
    ) -> None:
        self._llm = llm
        self._router = router
        self._model_id = model_id
        self._structured = structured
        self._tools = tools
        self._vision = vision

    @property
    def llm(self) -> Any:
        """Access the underlying ChatNVIDIA instance."""
        return self._llm

    @property
    def model_id(self) -> str:
        return self._model_id

    def invoke(self, messages: Any, **kwargs: Any) -> Any:
        """Synchronous invoke with auto-tracking."""
        self._router.limiter.mark_request(self._model_id)
        t0 = time.monotonic()
        try:
            result = self._llm.invoke(messages, **kwargs)
            self._router.record_success(
                self._model_id,
                latency=time.monotonic() - t0,
                structured=self._structured,
                tools=self._tools,
                vision=self._vision,
            )
            return result
        except Exception as exc:
            self._router.record_failure(
                self._model_id,
                error=exc,
                structured=self._structured,
                tools=self._tools,
                vision=self._vision,
            )
            raise

    async def ainvoke(self, messages: Any, **kwargs: Any) -> Any:
        """Async invoke with auto-tracking."""
        self._router.limiter.mark_request(self._model_id)
        t0 = time.monotonic()
        try:
            result = await self._llm.ainvoke(messages, **kwargs)
            self._router.record_success(
                self._model_id,
                latency=time.monotonic() - t0,
                structured=self._structured,
                tools=self._tools,
                vision=self._vision,
            )
            return result
        except Exception as exc:
            self._router.record_failure(
                self._model_id,
                error=exc,
                structured=self._structured,
                tools=self._tools,
                vision=self._vision,
            )
            raise

    # ── Composition wrappers ─────────────────────────────────────────
    # These ensure tracking survives LangChain chains like:
    #   llm = await router.get_tracked(structured=True)
    #   chain = llm.with_structured_output(MySchema)
    #   result = await chain.ainvoke(...)  # still tracked

    def with_structured_output(self, schema: Any, **kwargs: Any) -> TrackedLLM:
        """Return a tracked wrapper around ``llm.with_structured_output()``."""
        inner = self._llm.with_structured_output(schema, **kwargs)
        return TrackedLLM(
            llm=inner,
            router=self._router,
            model_id=self._model_id,
            structured=True,
            tools=self._tools,
            vision=self._vision,
        )

    def bind_tools(self, tools: Any, **kwargs: Any) -> TrackedLLM:
        """Return a tracked wrapper around ``llm.bind_tools()``."""
        inner = self._llm.bind_tools(tools, **kwargs)
        return TrackedLLM(
            llm=inner,
            router=self._router,
            model_id=self._model_id,
            structured=self._structured,
            tools=True,
            vision=self._vision,
        )

    def with_retry(self, **kwargs: Any) -> TrackedLLM:
        """Return a tracked wrapper around ``llm.with_retry()``."""
        inner = self._llm.with_retry(**kwargs)
        return TrackedLLM(
            llm=inner,
            router=self._router,
            model_id=self._model_id,
            structured=self._structured,
            tools=self._tools,
            vision=self._vision,
        )

    def with_config(self, **kwargs: Any) -> TrackedLLM:
        """Return a tracked wrapper around ``llm.with_config()``."""
        inner = self._llm.with_config(**kwargs)
        return TrackedLLM(
            llm=inner,
            router=self._router,
            model_id=self._model_id,
            structured=self._structured,
            tools=self._tools,
            vision=self._vision,
        )

    def __or__(self, other: Any) -> Any:
        """Support ``tracked_llm | prompt`` pipeline composition.

        Returns the raw composed runnable — tracking is lost for piped
        chains.  Use ``ainvoke()`` directly when you need stats.
        """
        return self._llm | other

    def __getattr__(self, name: str) -> Any:
        """Delegate attribute access to the underlying LLM."""
        return getattr(self._llm, name)
