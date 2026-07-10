from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from pydantic import BaseModel, Field


class ModelCapabilities(BaseModel):
    tools: bool = False
    structured: bool = False
    vision: bool = False
    reasoning: bool = False


class ModelInfo(BaseModel):
    id: str
    capabilities: ModelCapabilities = Field(default_factory=ModelCapabilities)
    quality_hint: float = 0.5
    deprecated: bool = False
    model_type: str | None = None
    metadata: dict[str, object] = Field(default_factory=dict)


if TYPE_CHECKING:
    from langchain_core.language_models.chat_models import BaseChatModel

    from nim_router.callbacks import TrackingCallback


@dataclass
class ModelSelection:
    """Bundles the chosen model metadata, a real ChatNVIDIA LLM, and a
    :class:`~nim_router.callbacks.TrackingCallback` for automatic stats.

    Returned by :meth:`NimRouter.select`.
    """

    info: ModelInfo
    llm: BaseChatModel
    callback: TrackingCallback


class RateLimitState(BaseModel):
    rpm_limit: int = 30
    recent_request_timestamps: list[float] = Field(default_factory=list)
    cooldown_until: float | None = None
    rate_limited_count: int = 0


class ModelRuntimeStats(BaseModel):
    calls: int = 0
    successes: int = 0
    failures: int = 0
    rate_limits: int = 0
    http_errors: int = 0
    avg_latency: float | None = None
    avg_tokens_per_second: float | None = None
    time_to_first_token: float | None = None
    structured_successes: int = 0
    structured_failures: int = 0
    tool_successes: int = 0
    tool_failures: int = 0
    vision_successes: int = 0
    vision_failures: int = 0
    last_used_at: float | None = None
    cooldown_until: float | None = None
    banned: bool = False

    @property
    def success_rate(self) -> float:
        if self.calls == 0:
            return 0.80
        return self.successes / self.calls

    @property
    def structured_success_rate(self) -> float:
        total = self.structured_successes + self.structured_failures
        if total == 0:
            return 0.80
        return self.structured_successes / total

    @property
    def tool_success_rate(self) -> float:
        total = self.tool_successes + self.tool_failures
        if total == 0:
            return 0.80
        return self.tool_successes / total

    @property
    def vision_success_rate(self) -> float:
        total = self.vision_successes + self.vision_failures
        if total == 0:
            return 0.80
        return self.vision_successes / total
