"""Shared DashScope chat routing for text-like model calls."""

from __future__ import annotations

import fnmatch
from typing import Any, Literal

import dashscope
import requests
from dashscope import Generation, MultiModalConversation

from app.config import Settings

DashScopeChatEndpoint = Literal["generation", "multimodal", "compatible"]

DASHSCOPE_COMPATIBLE_CHAT_COMPLETIONS_PATH = "/compatible-mode/v1/chat/completions"
VALID_DASHSCOPE_CHAT_ENDPOINTS: set[str] = {"generation", "multimodal", "compatible"}


def generate_chat_text(
    *,
    settings: Settings,
    model: str,
    messages: list[dict[str, str]],
    request_timeout: int,
    temperature: float,
    enable_thinking: bool | None = None,
) -> str:
    """Generate assistant text through the configured endpoint for a model."""
    _configure_dashscope(settings)
    endpoint = resolve_chat_endpoint(model, settings)
    if endpoint == "generation":
        response = Generation.call(
            model=model,
            api_key=settings.dashscope_api_key,
            messages=messages,
            result_format="message",
            request_timeout=request_timeout,
            temperature=temperature,
            **_thinking_kwargs(enable_thinking),
        )
        _raise_for_generation_error(response)
        return _extract_response_text(response)
    if endpoint == "multimodal":
        response = MultiModalConversation.call(
            model=model,
            api_key=settings.dashscope_api_key,
            messages=_multimodal_messages(messages),
            result_format="message",
            request_timeout=request_timeout,
            temperature=temperature,
            **_thinking_kwargs(enable_thinking),
        )
        _raise_for_generation_error(response)
        return _extract_response_text(response)
    return _generate_text_compatible(
        settings=settings,
        model=model,
        messages=messages,
        request_timeout=request_timeout,
        temperature=temperature,
        enable_thinking=enable_thinking,
    )


def resolve_chat_endpoint(model: str, settings: Settings) -> DashScopeChatEndpoint:
    """Resolve the model endpoint from config overrides or built-in routes."""
    configured_endpoint = _configured_endpoint(model, settings.dashscope_model_endpoints or {})
    if configured_endpoint:
        return configured_endpoint
    return "multimodal" if _looks_multimodal_chat_model(model) else "generation"


def _configured_endpoint(model: str, routes: dict[str, str]) -> DashScopeChatEndpoint | None:
    """Return an endpoint from exact or wildcard route config."""
    exact_endpoint = routes.get(model)
    if exact_endpoint is not None:
        return _validate_endpoint(model, exact_endpoint)
    for pattern, endpoint in routes.items():
        if pattern != model and fnmatch.fnmatchcase(model, pattern):
            return _validate_endpoint(pattern, endpoint)
    return None


def _validate_endpoint(pattern: str, endpoint: str) -> DashScopeChatEndpoint:
    """Validate a configured endpoint value."""
    normalized_endpoint = endpoint.strip().lower()
    if normalized_endpoint not in VALID_DASHSCOPE_CHAT_ENDPOINTS:
        supported = ", ".join(sorted(VALID_DASHSCOPE_CHAT_ENDPOINTS))
        raise ValueError(f"Unsupported DashScope endpoint {endpoint!r} for {pattern!r}; supported: {supported}")
    return normalized_endpoint  # type: ignore[return-value]


def _looks_multimodal_chat_model(model: str) -> bool:
    """Return whether official model naming indicates multimodal-generation."""
    lowered = model.lower()
    if lowered.startswith(
        (
            "qwen3.6-plus",
            "qwen3.6-flash",
            "qwen3.6-35b-a3b",
            "qwen3.5-plus",
            "qwen3.5-flash",
            "qwen3.5-omni",
            "qwen3-vl-",
            "qwen-vl-",
            "qvq-",
        )
    ):
        return True
    return "-vl-" in lowered or "-omni" in lowered


def _configure_dashscope(settings: Settings) -> None:
    """Set DashScope API key and optional base URL."""
    dashscope.api_key = settings.dashscope_api_key
    if settings.dashscope_base_url:
        for attr in ("base_http_api_url", "base_url"):
            if hasattr(dashscope, attr):
                setattr(dashscope, attr, settings.dashscope_base_url)


def _thinking_kwargs(enable_thinking: bool | None) -> dict[str, bool]:
    """Return optional thinking-mode kwargs."""
    return {} if enable_thinking is None else {"enable_thinking": enable_thinking}


def _multimodal_messages(messages: list[dict[str, str]]) -> list[dict[str, object]]:
    """Convert text-generation messages to DashScope multimodal message parts."""
    return [{"role": item["role"], "content": [{"text": item["content"]}]} for item in messages]


def _generate_text_compatible(
    *,
    settings: Settings,
    model: str,
    messages: list[dict[str, str]],
    request_timeout: int,
    temperature: float,
    enable_thinking: bool | None,
) -> str:
    """Call DashScope's OpenAI-compatible chat completion endpoint."""
    body: dict[str, Any] = {
        "model": model,
        "messages": messages,
        "temperature": temperature,
    }
    if enable_thinking is not None:
        body["enable_thinking"] = enable_thinking
    response = requests.post(
        compatible_chat_completions_url(settings.dashscope_base_url),
        headers={
            "Authorization": f"Bearer {settings.dashscope_api_key}",
            "Content-Type": "application/json",
        },
        json=body,
        timeout=request_timeout,
    )
    payload = _compatible_json_payload(response)
    if response.status_code >= 400:
        error = payload.get("error") if isinstance(payload, dict) else None
        raise RuntimeError(f"DashScope compatible generation failed: HTTP {response.status_code} {error or payload}")
    choices = payload.get("choices") if isinstance(payload, dict) else None
    if not choices:
        raise RuntimeError("DashScope compatible generation response did not contain choices.")
    content = ((choices[0] or {}).get("message") or {}).get("content")
    if not content:
        raise RuntimeError("DashScope compatible generation response did not contain generated text.")
    return str(content)


def compatible_chat_completions_url(base_url: str | None) -> str:
    """Return the OpenAI-compatible chat completions URL for a DashScope base URL."""
    root = (base_url or "https://dashscope.aliyuncs.com/api/v1").rstrip("/")
    if root.endswith("/chat/completions"):
        return root
    if root.endswith("/compatible-mode/v1"):
        return f"{root}/chat/completions"
    if root.endswith("/api/v1"):
        return f"{root[:-len('/api/v1')]}{DASHSCOPE_COMPATIBLE_CHAT_COMPLETIONS_PATH}"
    return f"{root}{DASHSCOPE_COMPATIBLE_CHAT_COMPLETIONS_PATH}"


def _compatible_json_payload(response: requests.Response) -> dict[str, Any]:
    """Decode a DashScope-compatible JSON response."""
    try:
        payload = response.json()
    except ValueError as exc:
        raise RuntimeError(f"DashScope compatible generation returned non-JSON: {response.text[:200]}") from exc
    if not isinstance(payload, dict):
        raise RuntimeError("DashScope compatible generation returned non-object JSON.")
    return payload


def _raise_for_generation_error(response: Any) -> None:
    """Raise when DashScope returns an error response."""
    status_code = getattr(response, "status_code", None)
    if status_code and int(status_code) >= 400:
        message = getattr(response, "message", None) or getattr(response, "code", None) or response
        raise RuntimeError(f"DashScope generation failed: HTTP {status_code} {message}")


def _extract_response_text(response: Any) -> str:
    """Extract text from common DashScope response shapes."""
    output = _field(response, "output")
    text = _field(output, "text")
    if text:
        return str(text)
    choices = _field(output, "choices") or []
    if choices:
        message = _field(choices[0], "message")
        content = _field(message, "content")
        if content:
            return _message_content_text(content)
    raise RuntimeError("DashScope generation response did not contain generated text.")


def _message_content_text(content: Any) -> str:
    """Extract assistant text from string or multimodal content parts."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        chunks = []
        for item in content:
            text = _field(item, "text")
            if text:
                chunks.append(str(text))
        if chunks:
            return "".join(chunks)
    return str(content)


def _field(obj: Any, key: str) -> Any:
    """Read a key from dict-like or attribute-like response objects."""
    if isinstance(obj, dict):
        return obj.get(key)
    return getattr(obj, key, None)
