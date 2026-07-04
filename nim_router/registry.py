from __future__ import annotations

import logging
from typing import Any

from nim_router.capabilities import infer_capabilities
from nim_router.config import RouterConfig
from nim_router.errors import ModelDiscoveryError
from nim_router.schemas import ModelCapabilities, ModelInfo

logger = logging.getLogger(__name__)

# Lazy import to keep import time fast
_chat_nvidia_cls: Any = None


def _merge_capabilities(
    raw_model: Any,
    model_id: str,
    overrides: dict[str, bool],
) -> ModelCapabilities:
    """Build capabilities with three-tier priority.

    1. Raw API metadata (lowest — may be incomplete)
    2. Built-in profile (fills gaps where API says False)
    3. Env / programmatic overrides (highest)

    A capability is True if ANY source says True.
    """
    # Layer 1: raw metadata
    api_caps = infer_capabilities(raw_model)

    # Layer 2: built-in profile
    profile = DEFAULT_MODEL_PROFILES.get(model_id)

    result = ModelCapabilities()
    for field_name in ("tools", "structured", "vision", "reasoning"):
        api_val = getattr(api_caps, field_name, False)
        profile_val = bool(profile.get(field_name, False)) if profile else False
        override_val = overrides.get(field_name) if field_name in overrides else None

        if override_val is not None:
            # Env/programmatic override wins outright
            setattr(result, field_name, bool(override_val))
        else:
            # API metadata OR built-in profile (either True → True)
            setattr(result, field_name, api_val or profile_val)

    return result

# Built-in profiles for well-known NIM models.
# quality: 0-1 hint used by scoring when no runtime history exists.
# speed_hint: informational only (scoring derives speed from real latency).
# Env overrides (NIM_ROUTER_QUALITY_HINTS_JSON) always win over these.
DEFAULT_MODEL_PROFILES: dict[str, dict[str, Any]] = {
    "openai/gpt-oss-120b": {
        "quality": 0.90,
        "tools": True,
        "structured": True,
        "vision": False,
        "reasoning": True,
    },
    "meta/llama-3.3-70b-instruct": {
        "quality": 0.85,
        "tools": True,
        "structured": True,
        "vision": False,
        "reasoning": False,
    },
    "meta/llama-3.1-70b-instruct": {
        "quality": 0.82,
        "tools": True,
        "structured": True,
        "vision": False,
        "reasoning": False,
    },
    "meta/llama-3.1-8b-instruct": {
        "quality": 0.70,
        "tools": True,
        "structured": True,
        "vision": False,
        "reasoning": False,
    },
    "nvidia/llama-3.1-nemotron-70b-instruct": {
        "quality": 0.88,
        "tools": True,
        "structured": True,
        "vision": False,
        "reasoning": False,
    },
    "nvidia/nemotron-3-nano-30b-a3b": {
        "quality": 0.78,
        "tools": False,
        "structured": True,
        "vision": False,
        "reasoning": True,
    },
    "meta/llama-3.2-11b-vision-instruct": {
        "quality": 0.72,
        "tools": False,
        "structured": False,
        "vision": True,
        "reasoning": False,
    },
    "meta/llama-3.2-90b-vision-instruct": {
        "quality": 0.82,
        "tools": False,
        "structured": False,
        "vision": True,
        "reasoning": False,
    },
    "mistralai/mistral-large-2-instruct": {
        "quality": 0.87,
        "tools": True,
        "structured": True,
        "vision": False,
        "reasoning": False,
    },
    "google/gemma-2-27b-it": {
        "quality": 0.78,
        "tools": True,
        "structured": True,
        "vision": False,
        "reasoning": False,
    },
}


def _get_chat_nvidia_cls() -> Any:
    global _chat_nvidia_cls  # noqa: PLW0603
    if _chat_nvidia_cls is None:
        try:
            from langchain_nvidia_ai_endpoints import ChatNVIDIA

            _chat_nvidia_cls = ChatNVIDIA
        except ImportError as exc:
            raise ModelDiscoveryError(
                "langchain-nvidia-ai-endpoints is not installed. "
                "Install it with: pip install langchain-nvidia-ai-endpoints"
            ) from exc
    return _chat_nvidia_cls


class ModelRegistry:
    """Discovers and manages available NVIDIA NIM models."""

    def __init__(self, config: RouterConfig) -> None:
        self.config = config
        self._models: list[ModelInfo] = []
        self._loaded = False

    @property
    def models(self) -> list[ModelInfo]:
        return self._models

    async def ensure_loaded(self) -> list[ModelInfo]:
        """Ensure model discovery has been performed."""
        if not self._loaded:
            await self.discover()
        return self._models

    async def discover(self) -> list[ModelInfo]:
        """Discover available models from NVIDIA API."""
        chat_nvidia_cls = _get_chat_nvidia_cls()

        try:
            raw_models = chat_nvidia_cls.get_available_models()
        except Exception as exc:
            raise ModelDiscoveryError(
                f"Failed to discover models: {exc}"
            ) from exc

        models: list[ModelInfo] = []
        for raw in raw_models:
            model_id = getattr(raw, "id", None)
            if not model_id or not isinstance(model_id, str):
                continue

            # Only chat-capable models
            model_type = getattr(raw, "model_type", None)
            client_type = getattr(raw, "client", None)
            if client_type and client_type != "ChatNVIDIA":
                continue
            if model_type and model_type not in ("chat", "vlm", "nv-vlm"):
                continue

            # Deprecated models
            deprecated = bool(getattr(raw, "deprecated", False))

            # Build capabilities: raw API metadata → built-in profile → env override
            cap_overrides = self.config.capabilities_overrides.get(model_id, {})
            capabilities = _merge_capabilities(raw, model_id, cap_overrides)

            # Quality hint: env override > built-in profile > neutral default
            profile = DEFAULT_MODEL_PROFILES.get(model_id, {})
            quality_hint = self.config.quality_hints.get(
                model_id, profile.get("quality", 0.5)
            )

            info = ModelInfo(
                id=model_id,
                capabilities=capabilities,
                quality_hint=quality_hint,
                deprecated=deprecated,
                model_type=model_type,
                metadata={
                    "base_model": getattr(raw, "base_model", None),
                    "aliases": getattr(raw, "aliases", None),
                },
            )
            models.append(info)

        # Apply pool restriction
        if self.config.model_pool:
            pool_set = set(self.config.model_pool)
            models = [m for m in models if m.id in pool_set]

        # Apply exclusions
        excluded = set(self.config.excluded_models)
        if excluded:
            models = [m for m in models if m.id not in excluded]

        # Apply capability overrides for models not discovered from API.
        # Only appended when allow_undiscovered_models is True — prevents
        # typos in config from silently creating phantom models that 404.
        if self.config.allow_undiscovered_models:
            for model_id, overrides in self.config.capabilities_overrides.items():
                if any(m.id == model_id for m in models):
                    continue
                capabilities = infer_capabilities({}, overrides)
                profile = DEFAULT_MODEL_PROFILES.get(model_id, {})
                quality_hint = self.config.quality_hints.get(
                    model_id, profile.get("quality", 0.5)
                )
                models.append(
                    ModelInfo(
                        id=model_id,
                        capabilities=capabilities,
                        quality_hint=quality_hint,
                    )
                )

        # Apply quality hints for models not yet seen
        for model_id, hint in self.config.quality_hints.items():
            for m in models:
                if m.id == model_id:
                    m.quality_hint = hint
                    break

        self._models = models
        self._loaded = True
        logger.info("Discovered %d models", len(models))
        return models

    def get_model(self, model_id: str) -> ModelInfo | None:
        """Look up a model by ID."""
        for m in self._models:
            if m.id == model_id:
                return m
        return None
