from __future__ import annotations

import json
import logging
import time
from pathlib import Path

from nim_router.schemas import ModelRuntimeStats

logger = logging.getLogger(__name__)

# Incremental moving average factor
_EMA_ALPHA = 0.3


class StatsStore:
    """In-memory per-model runtime statistics with optional JSON persistence."""

    def __init__(self, stats_path: str | None = None) -> None:
        self._stats: dict[str, ModelRuntimeStats] = {}
        self._stats_path = Path(stats_path) if stats_path else None
        if self._stats_path and self._stats_path.exists():
            self._load()

    def _load(self) -> None:
        if not self._stats_path or not self._stats_path.exists():
            return
        try:
            raw = json.loads(self._stats_path.read_text())
            for model_id, data in raw.items():
                self._stats[model_id] = ModelRuntimeStats(**data)
        except Exception:
            logger.warning("Failed to load stats from %s", self._stats_path)

    def _save(self) -> None:
        if not self._stats_path:
            return
        try:
            self._stats_path.parent.mkdir(parents=True, exist_ok=True)
            data = {k: v.model_dump() for k, v in self._stats.items()}
            # Atomic write: write to temp file then rename
            tmp_path = self._stats_path.with_suffix(".tmp")
            tmp_path.write_text(json.dumps(data, indent=2))
            tmp_path.replace(self._stats_path)
        except Exception:
            logger.warning("Failed to save stats to %s", self._stats_path)

    def _get(self, model_id: str) -> ModelRuntimeStats:
        if model_id not in self._stats:
            self._stats[model_id] = ModelRuntimeStats()
        return self._stats[model_id]

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
        stats = self._get(model_id)
        stats.calls += 1
        stats.successes += 1
        stats.last_used_at = time.time()

        if latency is not None:
            stats.avg_latency = _update_avg(stats.avg_latency, latency)
        if tokens_per_second is not None:
            stats.avg_tokens_per_second = _update_avg(
                stats.avg_tokens_per_second, tokens_per_second
            )
        if time_to_first_token is not None:
            stats.time_to_first_token = _update_avg(
                stats.time_to_first_token, time_to_first_token
            )

        if structured is True:
            stats.structured_successes += 1
        if tools is True:
            stats.tool_successes += 1
        if vision is True:
            stats.vision_successes += 1

        self._save()

    def record_failure(
        self,
        model_id: str,
        error: BaseException | None = None,
        kind: str | None = None,
        structured: bool | None = None,
        tools: bool | None = None,
        vision: bool | None = None,
    ) -> None:
        stats = self._get(model_id)
        stats.calls += 1
        stats.failures += 1
        stats.last_used_at = time.time()

        if kind == "rate_limit":
            stats.rate_limits += 1
        elif kind == "http_error":
            stats.http_errors += 1

        if structured is True:
            stats.structured_failures += 1
        if tools is True:
            stats.tool_failures += 1
        if vision is True:
            stats.vision_failures += 1

        self._save()

    def ban_model(self, model_id: str) -> None:
        stats = self._get(model_id)
        stats.banned = True
        self._save()

    def cooldown_model(self, model_id: str, seconds: float) -> None:
        stats = self._get(model_id)
        new_cooldown = time.time() + seconds
        # Only extend cooldown, never shorten an existing one
        if stats.cooldown_until is None or stats.cooldown_until < new_cooldown:
            stats.cooldown_until = new_cooldown
        self._save()

    def get_stats(self, model_id: str) -> ModelRuntimeStats:
        return self._get(model_id)

    def snapshot(self) -> dict[str, ModelRuntimeStats]:
        return dict(self._stats)

    def is_banned(self, model_id: str) -> bool:
        return self._stats.get(model_id, ModelRuntimeStats()).banned

    def is_cooling_down(self, model_id: str) -> bool:
        stats = self._stats.get(model_id)
        if stats is None or stats.cooldown_until is None:
            return False
        return time.time() < stats.cooldown_until


def _update_avg(current: float | None, new_value: float) -> float:
    if current is None:
        return new_value
    return current * (1 - _EMA_ALPHA) + new_value * _EMA_ALPHA
