from __future__ import annotations

import json
import os
from typing import Any

from pydantic import BaseModel, Field


class RouterConfig(BaseModel):
    model_pool: list[str] = Field(default_factory=list)
    excluded_models: list[str] = Field(default_factory=list)
    default_rpm: int = 30
    model_rpm: dict[str, int] = Field(default_factory=dict)
    capabilities_overrides: dict[str, dict[str, bool]] = Field(default_factory=dict)
    quality_hints: dict[str, float] = Field(default_factory=dict)
    timeout_seconds: float = 120.0
    stats_path: str | None = None
    allow_undiscovered_models: bool = False
    patch_timeout: bool = True
    initial_exploration_attempts: int = 1
    exploration_interval_seconds: float = 900.0

    @classmethod
    def from_env(cls) -> RouterConfig:
        """Load configuration from environment variables."""
        return cls(
            model_pool=_parse_csv_list("NIM_ROUTER_MODEL_POOL"),
            excluded_models=_parse_csv_list("NIM_ROUTER_EXCLUDED_MODELS"),
            default_rpm=_parse_int("NIM_ROUTER_DEFAULT_RPM", 30),
            model_rpm=_parse_json("NIM_ROUTER_MODEL_RPM_JSON", {}),
            capabilities_overrides=_parse_json("NIM_ROUTER_CAPABILITIES_JSON", {}),
            quality_hints=_parse_json("NIM_ROUTER_QUALITY_HINTS_JSON", {}),
            timeout_seconds=_parse_float("NIM_ROUTER_TIMEOUT_SECONDS", 120.0),
            stats_path=os.environ.get("NIM_ROUTER_STATS_PATH"),
            allow_undiscovered_models=_parse_bool("NIM_ROUTER_ALLOW_UNDISCOVERED", False),
            patch_timeout=_parse_bool("NIM_ROUTER_PATCH_TIMEOUT", True),
            initial_exploration_attempts=_parse_int(
                "NIM_ROUTER_INITIAL_EXPLORATION_ATTEMPTS", 1
            ),
            exploration_interval_seconds=_parse_float(
                "NIM_ROUTER_EXPLORATION_INTERVAL_SECONDS", 900.0
            ),
        )


def _parse_csv_list(env_var: str) -> list[str]:
    raw = os.environ.get(env_var, "").strip()
    if not raw:
        return []
    return [item.strip() for item in raw.split(",") if item.strip()]


def _parse_int(env_var: str, default: int) -> int:
    raw = os.environ.get(env_var)
    if raw is None or raw.strip() == "":
        return default
    try:
        return int(raw.strip())
    except ValueError:
        return default


def _parse_float(env_var: str, default: float) -> float:
    raw = os.environ.get(env_var)
    if raw is None or raw.strip() == "":
        return default
    try:
        return float(raw.strip())
    except ValueError:
        return default


def _parse_json(env_var: str, default: Any) -> Any:
    raw = os.environ.get(env_var)
    if raw is None or raw.strip() == "":
        return default
    try:
        return json.loads(raw)
    except (json.JSONDecodeError, TypeError):
        return default


def _parse_bool(env_var: str, default: bool = False) -> bool:
    """Parse a boolean environment variable.

    Accepts "1", "true", "yes", "on" (case-insensitive) as True.
    Any other value (including "0", "false", "no", "off", "") is False.
    """
    raw = os.environ.get(env_var)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}
