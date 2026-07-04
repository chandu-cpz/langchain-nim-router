from __future__ import annotations

import json
import os
from unittest.mock import patch

from nim_router.config import RouterConfig


def test_default_config():
    config = RouterConfig()
    assert config.model_pool == []
    assert config.excluded_models == []
    assert config.default_rpm == 30
    assert config.timeout_seconds == 120.0


def test_from_env_empty():
    with patch.dict(os.environ, {}, clear=True):
        config = RouterConfig.from_env()
        assert config.model_pool == []
        assert config.excluded_models == []
        assert config.default_rpm == 30


def test_model_pool_parsing():
    env = {"NIM_ROUTER_MODEL_POOL": "model-a, model-b ,model-c"}
    with patch.dict(os.environ, env, clear=False):
        # Remove other vars to avoid interference
        for key in list(os.environ):
            if key.startswith("NIM_ROUTER_") and key not in env:
                del os.environ[key]
        config = RouterConfig.from_env()
        assert config.model_pool == ["model-a", "model-b", "model-c"]


def test_excluded_models_parsing():
    env = {"NIM_ROUTER_EXCLUDED_MODELS": "bad-model,old-model"}
    with patch.dict(os.environ, env, clear=False):
        for key in list(os.environ):
            if key.startswith("NIM_ROUTER_") and key not in env:
                del os.environ[key]
        config = RouterConfig.from_env()
        assert config.excluded_models == ["bad-model", "old-model"]


def test_default_rpm_parsing():
    env = {"NIM_ROUTER_DEFAULT_RPM": "50"}
    with patch.dict(os.environ, env, clear=False):
        for key in list(os.environ):
            if key.startswith("NIM_ROUTER_") and key not in env:
                del os.environ[key]
        config = RouterConfig.from_env()
        assert config.default_rpm == 50


def test_model_rpm_json_parsing():
    rpm_data = {"model-a": 10, "model-b": 5}
    env = {"NIM_ROUTER_MODEL_RPM_JSON": json.dumps(rpm_data)}
    with patch.dict(os.environ, env, clear=False):
        for key in list(os.environ):
            if key.startswith("NIM_ROUTER_") and key not in env:
                del os.environ[key]
        config = RouterConfig.from_env()
        assert config.model_rpm == rpm_data


def test_capabilities_json_parsing():
    caps = {
        "model-a": {"tools": True, "structured": True, "vision": False, "reasoning": False},
        "model-b": {"tools": False, "structured": False, "vision": True, "reasoning": False},
    }
    env = {"NIM_ROUTER_CAPABILITIES_JSON": json.dumps(caps)}
    with patch.dict(os.environ, env, clear=False):
        for key in list(os.environ):
            if key.startswith("NIM_ROUTER_") and key not in env:
                del os.environ[key]
        config = RouterConfig.from_env()
        assert config.capabilities_overrides == caps


def test_quality_hints_json_parsing():
    hints = {"model-a": 0.95, "model-b": 0.65}
    env = {"NIM_ROUTER_QUALITY_HINTS_JSON": json.dumps(hints)}
    with patch.dict(os.environ, env, clear=False):
        for key in list(os.environ):
            if key.startswith("NIM_ROUTER_") and key not in env:
                del os.environ[key]
        config = RouterConfig.from_env()
        assert config.quality_hints == hints


def test_timeout_parsing():
    env = {"NIM_ROUTER_TIMEOUT_SECONDS": "60.5"}
    with patch.dict(os.environ, env, clear=False):
        for key in list(os.environ):
            if key.startswith("NIM_ROUTER_") and key not in env:
                del os.environ[key]
        config = RouterConfig.from_env()
        assert config.timeout_seconds == 60.5


def test_invalid_json_falls_back_to_default():
    env = {"NIM_ROUTER_MODEL_RPM_JSON": "not-valid-json"}
    with patch.dict(os.environ, env, clear=False):
        for key in list(os.environ):
            if key.startswith("NIM_ROUTER_") and key not in env:
                del os.environ[key]
        config = RouterConfig.from_env()
        assert config.model_rpm == {}


def test_invalid_int_falls_back_to_default():
    env = {"NIM_ROUTER_DEFAULT_RPM": "not-a-number"}
    with patch.dict(os.environ, env, clear=False):
        for key in list(os.environ):
            if key.startswith("NIM_ROUTER_") and key not in env:
                del os.environ[key]
        config = RouterConfig.from_env()
        assert config.default_rpm == 30
