from __future__ import annotations

import threading
import time
from collections import defaultdict

from nim_router.config import RouterConfig
from nim_router.schemas import RateLimitState


class RateLimiter:
    """In-memory per-model RPM rate limiter."""

    def __init__(self, config: RouterConfig) -> None:
        self._config = config
        self._states: dict[str, RateLimitState] = {}
        self._locks: dict[str, threading.Lock] = defaultdict(threading.Lock)

    def _get_state(self, model_id: str) -> RateLimitState:
        if model_id not in self._states:
            rpm = self._config.model_rpm.get(model_id, self._config.default_rpm)
            self._states[model_id] = RateLimitState(rpm_limit=rpm)
        return self._states[model_id]

    def is_available(self, model_id: str) -> bool:
        """Check if a model is available (not rate-limited or cooling down)."""
        state = self._get_state(model_id)
        now = time.monotonic()

        # Check cooldown
        if state.cooldown_until is not None and now < state.cooldown_until:
            return False

        # Clean old timestamps (older than 60s)
        cutoff = now - 60.0
        state.recent_request_timestamps = [
            ts for ts in state.recent_request_timestamps if ts > cutoff
        ]

        # Check RPM
        if len(state.recent_request_timestamps) >= state.rpm_limit:
            return False

        return True

    def mark_request(self, model_id: str) -> None:
        """Record a request timestamp for the model (thread-safe)."""
        with self._locks[model_id]:
            state = self._get_state(model_id)
            now = time.monotonic()
            # Clean old timestamps first
            state.recent_request_timestamps = [
                ts for ts in state.recent_request_timestamps if now - ts < 60.0
            ]
            state.recent_request_timestamps.append(now)

    def mark_rate_limited(self, model_id: str) -> None:
        """Record that a model returned a rate-limit error."""
        state = self._get_state(model_id)
        state.rate_limited_count += 1

    def cooldown(self, model_id: str, seconds: float) -> None:
        """Set a cooldown period for the model."""
        state = self._get_state(model_id)
        new_cooldown = time.monotonic() + seconds
        # Only extend cooldown, never shorten an existing one
        if state.cooldown_until is None or state.cooldown_until < new_cooldown:
            state.cooldown_until = new_cooldown

    def clear_cooldown(self, model_id: str) -> None:
        """Clear cooldown for a model."""
        state = self._get_state(model_id)
        state.cooldown_until = None

    def get_state(self, model_id: str) -> RateLimitState:
        """Get the current rate-limit state for a model."""
        return self._get_state(model_id)

    def snapshot(self) -> dict[str, RateLimitState]:
        """Return a snapshot of all rate-limit states."""
        return dict(self._states)
