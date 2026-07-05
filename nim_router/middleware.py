"""Optional middleware helper for LangChain agent integration.

This module provides convenience functions for wiring the router into
LangChain agent frameworks that support model-call middleware / hooks.

Usage::

    from nim_router.middleware import create_nim_model_middleware

    middleware = create_nim_model_middleware(
        router,
        tools=True,
        structured=True,
        priority="fast",
    )

    # Pass middleware to your agent harness
    agent = create_agent(..., model_middleware=middleware)

The middleware selects a fresh model at each call site, so it adapts to
rate limits, cooldowns, and bans automatically.

Note: LangChain docs warn that pre-bound models are not supported with
structured output in dynamic model middleware.  The middleware selects
a raw ``ChatNVIDIA`` and lets the agent/harness handle binding.
"""
from __future__ import annotations

import logging
from collections.abc import Awaitable, Callable
from typing import Any

logger = logging.getLogger(__name__)


def create_nim_model_middleware(
    router: Any,
    *,
    tools: bool = False,
    structured: bool = False,
    vision: bool = False,
    reasoning: bool = False,
    priority: str = "balanced",
) -> Callable[..., Awaitable[Any]]:
    """Return an async middleware that picks a fresh NIM model per call.

    The returned callable accepts a request-like object and a ``handler``
    coroutine.  It selects a model via the router, overrides
    ``request.model`` with the selected LLM, and delegates to the handler.

    This is designed for agent frameworks that support a
    ``@wrap_model_call`` or similar hook pattern.

    Example hook signature the middleware expects::

        async def handler(request):
            ...

    The request object is expected to have a mutable ``.model`` attribute
    or a ``.override(model=...)`` method — whichever the framework uses.

    If neither pattern is found the middleware logs a warning and calls
    the handler unchanged.
    """

    async def _middleware(
        request: Any,
        handler: Callable[..., Awaitable[Any]],
    ) -> Any:
        try:
            llm = await router.get(
                tools=tools,
                structured=structured,
                vision=vision,
                reasoning=reasoning,
                priority=priority,
            )
        except Exception:
            logger.warning(
                "NIM router failed to select a model; "
                "falling through to handler with original request.",
                exc_info=True,
            )
            return await handler(request)

        # Try common override patterns
        if hasattr(request, "override") and callable(request.override):
            overridden = request.override(model=llm)
            return await handler(overridden)

        if hasattr(request, "model"):
            try:
                original = request.model
                request.model = llm
                result = await handler(request)
                request.model = original
                return result
            except Exception:
                request.model = original
                raise

        logger.warning(
            "Request object has no .model or .override(); "
            "passing request unchanged to handler."
        )
        return await handler(request)

    return _middleware


def create_nim_model_middleware_lc(
    router: Any,
    *,
    tools: bool = False,
    structured: bool = False,
    vision: bool = False,
    reasoning: bool = False,
    priority: str = "balanced",
) -> Callable[..., Awaitable[Any]]:
    """Return a LangChain-native middleware using ``@wrap_model_call``.

    This uses LangChain's official ``wrap_model_call`` decorator (available
    in ``langchain.agents.middleware``) which properly integrates with
    LangChain's model-calling infrastructure including structured output
    bindings.

    Usage::

        from nim_router.middleware import create_nim_model_middleware_lc

        middleware = create_nim_model_middleware_lc(
            router,
            tools=True,
            structured=True,
            priority="fast",
        )

        # Pass to LangGraph/LangChain agent that supports model middleware
        # e.g., create_react_agent(..., model=..., model_call_middleware=middleware)

    Requires ``langchain`` package (not just ``langchain-core``).
    """
    try:
        from langchain.agents.middleware import wrap_model_call
    except ImportError as e:
        raise ImportError(
            "create_nim_model_middleware_lc requires the 'langchain' package. "
            "Install with: pip install langchain"
        ) from e

    @wrap_model_call
    async def _middleware(request: Any, handler: Callable[..., Awaitable[Any]]) -> Any:
        # router.select returns ModelSelection with llm and callback
        selection = await router.select(
            tools=tools,
            structured=structured,
            vision=vision,
            reasoning=reasoning,
            priority=priority,
        )
        # Use request.override(model=...) which is the official pattern
        # Also inject the tracking callback via config
        overridden = request.override(model=selection.llm)
        if hasattr(overridden, "config") and isinstance(overridden.config, dict):
            callbacks = overridden.config.get("callbacks", [])
            callbacks.append(selection.callback)
            overridden.config["callbacks"] = callbacks
        return await handler(overridden)

    return _middleware
