from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger(__name__)

DEFAULT_TIMEOUT = 300


def create_chat_nvidia(
    model_id: str,
    temperature: float | None = None,
    top_p: float | None = None,
    max_completion_tokens: int | None = None,
    timeout: float | None = None,
    patch_timeout: bool = True,
    model_kwargs: dict[str, Any] | None = None,
    **extra: Any,
) -> Any:
    """Create a configured ChatNVIDIA instance with optional timeout patching.

    When *patch_timeout* is True (the default) and the expected private
    attributes exist, the async client is patched so requests cannot hang
    forever.  If internals have changed the patch is skipped and a warning
    is logged — the LLM still works, just without the timeout guard.
    """
    try:
        import aiohttp  # noqa: F811
    except ImportError:
        aiohttp = None  # type: ignore[assignment]

    from langchain_nvidia_ai_endpoints import ChatNVIDIA

    timeout_seconds = timeout if timeout is not None else DEFAULT_TIMEOUT

    kwargs: dict[str, Any] = {"model": model_id}

    if temperature is not None:
        kwargs["temperature"] = temperature
    if top_p is not None:
        kwargs["top_p"] = top_p
    if max_completion_tokens is not None:
        kwargs["max_completion_tokens"] = max_completion_tokens

    # Merge extra model_kwargs
    if model_kwargs:
        kwargs.setdefault("model_kwargs", {}).update(model_kwargs)

    # Merge any extra keyword arguments
    kwargs.update(extra)

    llm = ChatNVIDIA(**kwargs)

    if not patch_timeout:
        return llm

    # Guard: only patch if internals exist
    if not hasattr(llm, "_client") or not hasattr(llm, "_async_client"):
        logger.warning(
            "ChatNVIDIA private attributes (_client, _async_client) not found; "
            "skipping timeout patch. Timeout protection disabled."
        )
        return llm

    # Patch timeout on both clients (controls 202 polling timeout)
    for client in (llm._client, llm._async_client):
        client.timeout = timeout_seconds

    # Patch async session factory with real aiohttp timeout
    if aiohttp is not None and hasattr(llm._async_client, "_build_ssl_context"):
        try:

            def _async_session() -> Any:
                connector = aiohttp.TCPConnector(
                    ssl=llm._async_client._build_ssl_context()
                )
                client_timeout = aiohttp.ClientTimeout(total=timeout_seconds)
                return aiohttp.ClientSession(
                    connector=connector, timeout=client_timeout
                )

            llm._async_client.get_async_session_fn = _async_session
        except Exception:
            logger.warning(
                "Failed to patch async session timeout; "
                "requests may hang without a timeout.",
                exc_info=True,
            )
    elif aiohttp is None:
        logger.warning(
            "aiohttp not installed; async session timeout patch skipped. "
            "Install with: pip install aiohttp"
        )
    else:
        logger.warning(
            "ChatNVIDIA._async_client._build_ssl_context not found; "
            "async session timeout patch skipped."
        )

    return llm
