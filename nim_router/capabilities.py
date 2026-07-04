from __future__ import annotations

from typing import Any

from nim_router.schemas import ModelCapabilities


def infer_capabilities(
    model_data: dict[str, Any] | Any,
    overrides: dict[str, bool] | None = None,
) -> ModelCapabilities:
    """Infer capabilities from model metadata, applying manual overrides.

    Unknown capabilities default to False. Overrides take precedence.
    """
    caps = ModelCapabilities()

    # Try to read from a Model object or dict
    if isinstance(model_data, dict):
        caps.tools = bool(model_data.get("supports_tools", False))
        caps.structured = bool(model_data.get("supports_structured_output", False))
        caps.reasoning = bool(model_data.get("supports_thinking", False))
        caps.vision = _infer_vision(model_data)
    else:
        # langchain_nvidia_ai_endpoints Model object
        caps.tools = bool(getattr(model_data, "supports_tools", False))
        caps.structured = bool(getattr(model_data, "supports_structured_output", False))
        caps.reasoning = bool(getattr(model_data, "supports_thinking", False))
        caps.vision = _infer_vision_model(model_data)

    # Apply manual overrides
    if overrides:
        for key, val in overrides.items():
            if hasattr(caps, key):
                setattr(caps, key, bool(val))

    return caps


def _infer_vision(model_data: dict[str, Any]) -> bool:
    model_type = model_data.get("model_type", "")
    if model_type in ("vlm", "nv-vlm"):
        return True
    return False


def _infer_vision_model(model_data: Any) -> bool:
    model_type = getattr(model_data, "model_type", None)
    if model_type in ("vlm", "nv-vlm"):
        return True
    return False


def capabilities_satisfy(
    required: ModelCapabilities,
    provided: ModelCapabilities,
) -> bool:
    """Check if provided capabilities satisfy required capabilities."""
    if required.tools and not provided.tools:
        return False
    if required.structured and not provided.structured:
        return False
    if required.vision and not provided.vision:
        return False
    if required.reasoning and not provided.reasoning:
        return False
    return True
