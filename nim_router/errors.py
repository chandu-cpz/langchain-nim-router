from __future__ import annotations

from enum import Enum


class NimRouterError(Exception):
    """Base error for nim_router."""


class NoUsableModelError(NimRouterError):
    """No model satisfies the requested capabilities and constraints."""

    def __init__(
        self,
        message: str,
        required_capabilities: dict[str, bool] | None = None,
        excluded_reasons: dict[str, list[str]] | None = None,
    ) -> None:
        self.required_capabilities = required_capabilities or {}
        self.excluded_reasons = excluded_reasons or {}
        super().__init__(message)


class ModelDiscoveryError(NimRouterError):
    """Failed to discover or list available models."""


class ErrorKind(str, Enum):
    RATE_LIMIT = "rate_limit"
    HTTP_ERROR = "http_error"
    TIMEOUT = "timeout"
    STRUCTURED_OUTPUT_FAILURE = "structured_output_failure"
    TOOL_CALL_FAILURE = "tool_call_failure"
    VISION_FAILURE = "vision_failure"
    MODEL_NOT_FOUND = "model_not_found"
    NETWORK = "network"
    GENERIC = "generic"


def classify_error(error: BaseException) -> ErrorKind:
    """Classify an exception into an ErrorKind.

    Priority: status code range FIRST, then message patterns.
    This prevents a message containing "429" inside a 500 error from
    being misclassified as RATE_LIMIT, and prevents 408/504 with
    "timeout" in the body from skipping the HTTP_ERROR cooldown.
    """
    msg = str(error).lower()

    # Extract status code (int or string) — must handle both
    status = None
    if isinstance(error, Exception):
        raw = getattr(error, "status_code", None) or getattr(error, "code", None)
        if raw is not None:
            try:
                status = int(raw)
            except (TypeError, ValueError):
                status = None

    # ── 1. Status code range check (HIGHEST PRIORITY) ──────────────
    #    408/504 with "timeout" body → HTTP_ERROR (10s cooldown), NOT TIMEOUT
    #    500 with "429" body → HTTP_ERROR, NOT RATE_LIMIT
    if status is not None:
        if status == 429:
            return ErrorKind.RATE_LIMIT
        if status in {404, 410}:
            return ErrorKind.MODEL_NOT_FOUND
        if 400 <= status < 600:
            return ErrorKind.HTTP_ERROR

    # ── 2. Timeout detection (native Python type) ──────────────────
    if isinstance(error, (TimeoutError, ConnectionError)):
        return ErrorKind.TIMEOUT
    # Check class name for custom timeout types
    if type(error).__name__ in ("TimeoutError", "AsyncTimeoutError"):
        return ErrorKind.TIMEOUT

    network_error_names = {
        "ConnectError",
        "ConnectTimeout",
        "ClientConnectorError",
        "ClientConnectionError",
        "NameResolutionError",
    }
    current: BaseException | None = error
    while current is not None:
        if type(current).__name__ in network_error_names:
            return ErrorKind.NETWORK
        current = current.__cause__ or current.__context__

    # ── 3. Message-based fallbacks ─────────────────────────────────
    #    Only when no status code was available
    if "rate limit" in msg or "too many requests" in msg:
        return ErrorKind.RATE_LIMIT
    # Catch "rate-limit", "rate_limit", "ratelimit" etc.
    if "ratelimit" in msg or "rate-limit" in msg or "rate_limit" in msg:
        return ErrorKind.RATE_LIMIT
    # Catch "429" embedded in message text (e.g. "error 429")
    if "429" in msg:
        return ErrorKind.RATE_LIMIT

    if "timeout" in msg or "timed out" in msg:
        return ErrorKind.TIMEOUT

    if "name resolution" in msg or "cannot connect to host" in msg:
        return ErrorKind.NETWORK

    if "model not found" in msg or "endpoint not found" in msg:
        return ErrorKind.MODEL_NOT_FOUND
    # Catch "[404]" or "404" embedded in message text
    if "404" in msg or "410" in msg or "gone" in msg:
        return ErrorKind.MODEL_NOT_FOUND

    return ErrorKind.GENERIC
