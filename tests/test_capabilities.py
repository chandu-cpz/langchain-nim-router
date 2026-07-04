from __future__ import annotations

from nim_router.capabilities import infer_capabilities


def test_infer_from_dict_with_tools():
    data = {"supports_tools": True, "supports_structured_output": True}
    caps = infer_capabilities(data)
    assert caps.tools is True
    assert caps.structured is True
    assert caps.vision is False
    assert caps.reasoning is False


def test_infer_vision_from_model_type():
    data = {"model_type": "vlm"}
    caps = infer_capabilities(data)
    assert caps.vision is True


def test_infer_nv_vlm():
    data = {"model_type": "nv-vlm"}
    caps = infer_capabilities(data)
    assert caps.vision is True


def test_infer_thinking():
    data = {"supports_thinking": True}
    caps = infer_capabilities(data)
    assert caps.reasoning is True


def test_overrides_take_precedence():
    data = {"supports_tools": False}
    overrides = {"tools": True}
    caps = infer_capabilities(data, overrides)
    assert caps.tools is True


def test_overrides_disable():
    data = {"supports_tools": True, "supports_structured_output": True}
    overrides = {"tools": False}
    caps = infer_capabilities(data, overrides)
    assert caps.tools is False
    assert caps.structured is True


def test_empty_dict():
    caps = infer_capabilities({})
    assert caps.tools is False
    assert caps.structured is False
    assert caps.vision is False
    assert caps.reasoning is False


def test_infer_from_model_object():
    class FakeModel:
        supports_tools = True
        supports_structured_output = False
        supports_thinking = True
        model_type = "chat"

    caps = infer_capabilities(FakeModel())
    assert caps.tools is True
    assert caps.structured is False
    assert caps.reasoning is True
    assert caps.vision is False


def test_infer_vision_from_model_object():
    class FakeModel:
        supports_tools = False
        supports_structured_output = False
        supports_thinking = False
        model_type = "vlm"

    caps = infer_capabilities(FakeModel())
    assert caps.vision is True
