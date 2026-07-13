"""LangChain callback handler for automatic NIM model tracking.

Records latency, token usage, and errors back to the router's stats store.
Fires on standard LangChain callback hooks so tracking survives any
composition (``with_structured_output``, ``bind_tools``, LCEL pipes, etc.).
"""
from __future__ import annotations

import logging
import time
from typing import Any
from uuid import UUID

from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.outputs import LLMResult

logger = logging.getLogger(__name__)


class TrackingCallback(BaseCallbackHandler):
    """Records per-model latency, success/failure, and token usage.

    Instances are created via :meth:`NimRouter.tracker_for` or
    :meth:`NimRouter.select` — not constructed directly.

    .. note::

       ``structured_success_rate`` / ``tool_success_rate`` /
       ``vision_success_rate`` record whether the **LLM call** succeeded
       while selected for that capability — not whether downstream schema
       parsing or tool execution succeeded.  A structured success means
       the API returned without error; a ``ValidationError`` in
       ``with_structured_output`` is *not* captured here.  Use explicit
       app-level ``record_failure(kind="structured_output_failure")`` for
       parser/tool correctness tracking.
    """

    def __init__(
        self,
        router: Any,
        model_id: str,
        *,
        tools: bool = False,
        structured: bool = False,
        vision: bool = False,
        reasoning: bool = False,
        mark_request_on_start: bool = True,
        pre_reserved_requests: int = 0,
    ) -> None:
        super().__init__()
        self._router = router
        self._model_id = model_id
        self._tools = tools
        self._structured = structured
        self._vision = vision
        self._reasoning = reasoning
        self._mark_request_on_start = mark_request_on_start
        self._pre_reserved_requests = pre_reserved_requests
        # run_id → monotonic start time
        self._starts: dict[UUID, float] = {}

    @property
    def model_id(self) -> str:
        return self._model_id

    # ── start hooks ──────────────────────────────────────────────────
    # Chat models may emit on_chat_model_start OR on_llm_start.
    # We handle both but guard against double-counting per run_id.

    def on_chat_model_start(
        self,
        serialized: dict[str, Any],
        messages: list[list[Any]],
        *,
        run_id: UUID | None = None,
        **kwargs: Any,
    ) -> None:
        self._record_start(run_id)

    def on_llm_start(
        self,
        serialized: dict[str, Any],
        prompts: list[str],
        *,
        run_id: UUID | None = None,
        **kwargs: Any,
    ) -> None:
        self._record_start(run_id)

    def _record_start(self, run_id: UUID | None) -> None:
        if run_id is None or run_id in self._starts:
            return
        self._starts[run_id] = time.monotonic()
        if self._pre_reserved_requests > 0:
            self._pre_reserved_requests -= 1
        elif self._mark_request_on_start:
            try:
                self._router.limiter.mark_request(self._model_id)
            except Exception:
                logger.debug("Failed to mark rate-limit request", exc_info=True)

    # ── end hook ─────────────────────────────────────────────────────

    def on_llm_end(
        self,
        response: LLMResult,
        *,
        run_id: UUID | None = None,
        **kwargs: Any,
    ) -> None:
        if run_id is None:
            return
        start = self._starts.pop(run_id, None)
        if start is None:
            return

        latency = time.monotonic() - start
        tokens_out, tokens_per_sec = _extract_token_usage(response, latency)

        try:
            self._router.record_success(
                self._model_id,
                latency=latency,
                tokens_per_second=tokens_per_sec,
                structured=self._structured,
                tools=self._tools,
                vision=self._vision,
            )
        except Exception:
            logger.debug("Failed to record success stats", exc_info=True)

    # ── error hook ───────────────────────────────────────────────────

    def on_llm_error(
        self,
        error: BaseException,
        *,
        run_id: UUID | None = None,
        **kwargs: Any,
    ) -> None:
        if run_id is None:
            return
        self._starts.pop(run_id, None)

        try:
            self._router.record_failure(
                self._model_id,
                error=error,
                structured=self._structured,
                tools=self._tools,
                vision=self._vision,
            )
        except Exception:
            logger.debug("Failed to record failure stats", exc_info=True)


def _extract_token_usage(
    response: LLMResult,
    latency: float,
) -> tuple[int | None, float | None]:
    """Try to pull output token count from an LLMResult.

    LangChain stores token info in several places depending on the
    provider.  We check them all defensively.

    Returns (output_tokens, tokens_per_second) — either may be None.
    """
    output_tokens: int | None = None

    # 1. LLMResult.llm_output → token_usage / usage_metadata
    llm_out = response.llm_output or {}
    usage = llm_out.get("token_usage") or llm_out.get("usage_metadata") or {}
    if isinstance(usage, dict):
        output_tokens = _int_or_none(
            usage.get("completion_tokens")
            or usage.get("output_tokens")
            or usage.get("completion_token_count")
        )

    # 2. First generation → message → usage_metadata (AIMessage style)
    if output_tokens is None:
        for gen_list in response.generations:
            for gen in gen_list:
                msg = getattr(gen, "message", None)
                if msg is None:
                    continue
                um = getattr(msg, "usage_metadata", None)
                if isinstance(um, dict):
                    output_tokens = _int_or_none(um.get("output_tokens"))
                    if output_tokens is not None:
                        break
                # response_metadata fallback
                rm = getattr(msg, "response_metadata", None)
                if isinstance(rm, dict):
                    inner_usage = rm.get("token_usage") or rm.get("usage") or {}
                    if isinstance(inner_usage, dict):
                        output_tokens = _int_or_none(
                            inner_usage.get("completion_tokens")
                            or inner_usage.get("output_tokens")
                        )
                        if output_tokens is not None:
                            break
            if output_tokens is not None:
                break

    tok_per_sec: float | None = None
    if output_tokens is not None and latency > 0:
        tok_per_sec = output_tokens / latency

    return output_tokens, tok_per_sec


def _int_or_none(val: Any) -> int | None:
    if val is None:
        return None
    try:
        return int(val)
    except (TypeError, ValueError):
        return None
