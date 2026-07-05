"""langchain-nim-router: Select the best NVIDIA NIM model for LangChain."""

from nim_router.callbacks import TrackingCallback
from nim_router.errors import (
    ErrorKind,
    ModelDiscoveryError,
    NimRouterError,
    NoUsableModelError,
    classify_error,
)
from nim_router.router import NimRouter
from nim_router.schemas import (
    ModelCapabilities,
    ModelInfo,
    ModelRuntimeStats,
    ModelSelection,
    RateLimitState,
)

__all__ = [
    "NimRouter",
    "TrackingCallback",
    "ModelSelection",
    "NimRouterError",
    "NoUsableModelError",
    "ModelDiscoveryError",
    "ErrorKind",
    "classify_error",
    "ModelCapabilities",
    "ModelInfo",
    "ModelRuntimeStats",
    "RateLimitState",
]

__version__ = "0.1.0"
