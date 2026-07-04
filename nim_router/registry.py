from __future__ import annotations

import logging
from typing import Any

from nim_router.capabilities import infer_capabilities
from nim_router.config import RouterConfig
from nim_router.errors import ModelDiscoveryError
from nim_router.schemas import ModelInfo

logger = logging.getLogger(__name__)

# Lazy import to keep import time fast
_chat_nvidia_cls: Any = None


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

            # Build capabilities
            cap_overrides = self.config.capabilities_overrides.get(model_id, {})
            capabilities = infer_capabilities(raw, cap_overrides)

            # Quality hint
            quality_hint = self.config.quality_hints.get(model_id, 0.5)

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

        # Apply capability overrides for models not discovered from API
        for model_id, overrides in self.config.capabilities_overrides.items():
            if any(m.id == model_id for m in models):
                continue
            capabilities = infer_capabilities({}, overrides)
            quality_hint = self.config.quality_hints.get(model_id, 0.5)
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
