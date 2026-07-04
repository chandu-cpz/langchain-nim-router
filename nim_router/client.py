from __future__ import annotations

import logging
from typing import Any

import aiohttp

logger = logging.getLogger(__name__)

DEFAULT_TIMEOUT = 300


def create_chat_nvidia(
    model_id: str,
    temperature: float | None = None,
    top_p: float | None = None,
    max_completion_tokens: int | None = None,
    timeout: float | None = None,
    model_kwargs: dict[str, Any] | None = None,
    **extra: Any,
) -> Any:
    """Create a configured ChatNVIDIA instance with proper timeout handling.

    The langchain-nvidia-ai-endpoints library creates aiohttp.ClientSession
    with NO timeout by default — requests can hang forever. This patches
    the async client to use a real ClientTimeout.
    """
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

    # Patch timeout on both clients (controls 202 polling timeout)
    for client in (llm._client, llm._async_client):
        client.timeout = timeout_seconds

    # Patch async session factory with real aiohttp timeout
    def _async_session():
        connector = aiohttp.TCPConnector(ssl=llm._async_client._build_ssl_context())
        client_timeout = aiohttp.ClientTimeout(total=timeout_seconds)
        return aiohttp.ClientSession(connector=connector, timeout=client_timeout)

    llm._async_client.get_async_session_fn = _async_session

    return llm
