"""Tests for shared DashScope chat endpoint routing."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from app.config import Settings
from app.dashscope_chat import generate_chat_text, resolve_chat_endpoint


def test_resolve_chat_endpoint_uses_builtin_model_routes() -> None:
    """Known model families should route without network probing."""
    settings = Settings(
        dashscope_api_key="key",
        dashscope_base_url="https://dashscope.aliyuncs.com/api/v1",
    )

    assert resolve_chat_endpoint("qwen-plus", settings) == "generation"
    assert resolve_chat_endpoint("qwen3-max", settings) == "generation"
    assert resolve_chat_endpoint("qwen3.6-max-preview", settings) == "generation"
    assert resolve_chat_endpoint("qwen3.6-plus", settings) == "multimodal"
    assert resolve_chat_endpoint("qwen3.6-flash-2026-04-16", settings) == "multimodal"
    assert resolve_chat_endpoint("qwen-vl-plus", settings) == "multimodal"


def test_resolve_chat_endpoint_allows_config_override() -> None:
    """Exact and wildcard config should override built-in routes."""
    settings = Settings(
        dashscope_api_key="key",
        dashscope_base_url="https://dashscope.aliyuncs.com/api/v1",
        dashscope_model_endpoints={
            "qwen3.6-*": "compatible",
            "qwen3.6-plus": "multimodal",
            "qwen-plus": "multimodal",
        },
    )

    assert resolve_chat_endpoint("qwen3.6-plus", settings) == "multimodal"
    assert resolve_chat_endpoint("qwen3.6-flash", settings) == "compatible"
    assert resolve_chat_endpoint("qwen-plus", settings) == "multimodal"


def test_resolve_chat_endpoint_rejects_unknown_config_value() -> None:
    """Bad endpoint config should fail before issuing a model request."""
    settings = Settings(
        dashscope_api_key="key",
        dashscope_base_url="https://dashscope.aliyuncs.com/api/v1",
        dashscope_model_endpoints={"qwen-test": "bad"},
    )

    with pytest.raises(ValueError, match="Unsupported DashScope endpoint"):
        resolve_chat_endpoint("qwen-test", settings)


def test_generate_chat_text_uses_multimodal_route_without_generation_probe(
    monkeypatch,
) -> None:
    """Qwen3.6 should call native multimodal directly, not try Generation first."""
    multimodal_calls = []

    def unexpected_generation_call(**kwargs):
        raise AssertionError("Generation.call should not be probed for qwen3.6-plus.")

    def unexpected_post(*args, **kwargs):
        raise AssertionError("Compatible endpoint should not be used by default.")

    def fake_multimodal_call(**kwargs):
        multimodal_calls.append(kwargs)
        return SimpleNamespace(
            status_code=200,
            output={"choices": [{"message": {"content": [{"text": "OK"}]}}]},
        )

    monkeypatch.setattr(
        "app.dashscope_chat.Generation.call", unexpected_generation_call
    )
    monkeypatch.setattr(
        "app.dashscope_chat.MultiModalConversation.call", fake_multimodal_call
    )
    monkeypatch.setattr("app.dashscope_chat.requests.post", unexpected_post)
    settings = Settings(
        dashscope_api_key="key",
        dashscope_base_url="https://dashscope.aliyuncs.com/api/v1",
    )

    result = generate_chat_text(
        settings=settings,
        model="qwen3.6-plus",
        messages=[{"role": "user", "content": "只回复 OK"}],
        request_timeout=120,
        temperature=0.0,
        enable_thinking=False,
    )

    assert result == "OK"
    assert multimodal_calls[0]["messages"] == [
        {"role": "user", "content": [{"text": "只回复 OK"}]}
    ]
    assert multimodal_calls[0]["enable_thinking"] is False


def test_generate_chat_text_uses_compatible_route_when_configured(monkeypatch) -> None:
    """OpenAI-compatible can be selected explicitly for a model."""
    compatible_calls = []

    class FakeResponse:
        status_code = 200
        text = ""

        def json(self):
            return {"choices": [{"message": {"content": "OK"}}]}

    def fake_post(url, **kwargs):
        compatible_calls.append((url, kwargs))
        return FakeResponse()

    def unexpected_generation_call(**kwargs):
        raise AssertionError("Generation.call should not run for compatible route.")

    def unexpected_multimodal_call(**kwargs):
        raise AssertionError(
            "MultiModalConversation.call should not run for compatible route."
        )

    monkeypatch.setattr(
        "app.dashscope_chat.Generation.call", unexpected_generation_call
    )
    monkeypatch.setattr(
        "app.dashscope_chat.MultiModalConversation.call", unexpected_multimodal_call
    )
    monkeypatch.setattr("app.dashscope_chat.requests.post", fake_post)
    settings = Settings(
        dashscope_api_key="key",
        dashscope_base_url="https://dashscope.aliyuncs.com/api/v1",
        dashscope_model_endpoints={"qwen3.6-plus": "compatible"},
    )

    result = generate_chat_text(
        settings=settings,
        model="qwen3.6-plus",
        messages=[{"role": "user", "content": "只回复 OK"}],
        request_timeout=120,
        temperature=0.0,
        enable_thinking=False,
    )

    assert result == "OK"
    assert (
        compatible_calls[0][0]
        == "https://dashscope.aliyuncs.com/compatible-mode/v1/chat/completions"
    )
    assert compatible_calls[0][1]["json"] == {
        "model": "qwen3.6-plus",
        "messages": [{"role": "user", "content": "只回复 OK"}],
        "temperature": 0.0,
        "enable_thinking": False,
    }
