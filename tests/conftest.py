from __future__ import annotations

from dataclasses import dataclass
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from nim_router.config import RouterConfig
from nim_router.router import NimRouter


@dataclass
class FakeModel:
    """Minimal stand-in for langchain_nvidia_ai_endpoints.Model."""

    id: str
    model_type: str = "chat"
    client: str = "ChatNVIDIA"
    supports_tools: bool = False
    supports_structured_output: bool = False
    supports_thinking: bool = False
    deprecated: bool = False
    base_model: str | None = None
    aliases: list[str] | None = None
    endpoint: str | None = None


FAKE_MODELS: list[FakeModel] = [
    FakeModel(
        id="meta/llama-3.3-70b-instruct",
        supports_tools=True,
        supports_structured_output=True,
    ),
    FakeModel(
        id="meta/llama-3.1-8b-instruct",
        supports_tools=True,
        supports_structured_output=True,
    ),
    FakeModel(
        id="nvidia/llama-3.1-nemotron-70b-instruct",
        supports_tools=True,
        supports_structured_output=True,
    ),
    FakeModel(
        id="meta/llama-3.2-11b-vision-instruct",
        model_type="vlm",
        supports_tools=False,
        supports_structured_output=False,
    ),
    FakeModel(
        id="nvidia/nemotron-3-nano-30b-a3b",
        supports_tools=False,
        supports_structured_output=True,
        supports_thinking=True,
    ),
]


def _fake_get_available_models(**kwargs: Any) -> list[FakeModel]:
    return list(FAKE_MODELS)


@pytest.fixture
def mock_chat_nvidia():
    """Monkeypatch ChatNVIDIA.get_available_models to return fake models."""
    mock_cls = MagicMock()
    mock_cls.get_available_models = staticmethod(_fake_get_available_models)
    with patch("nim_router.registry._get_chat_nvidia_cls", return_value=mock_cls):
        yield mock_cls


@pytest.fixture
def config():
    return RouterConfig()


@pytest.fixture
def router(mock_chat_nvidia, config):
    return NimRouter(config=config)
