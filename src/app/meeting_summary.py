"""DashScope meeting summarization helpers."""

from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass
from typing import Any

import dashscope
from dashscope import Generation

from app.config import Settings
from app.models import TranscriptResult
from app.postprocess import render_speaker_text
from app.utils import retry

MAX_SUMMARY_TRANSCRIPT_CHARS = 24_000


@dataclass(frozen=True, slots=True)
class MeetingSummary:
    """Structured meeting summary generated from a transcript."""

    title: str
    summary: str
    topics: list[str]
    action_items: list[str]
    model: str

    def to_dict(self) -> dict[str, object]:
        """Return a JSON-ready dictionary."""
        return asdict(self)


def generate_meeting_summary(
    result: TranscriptResult,
    *,
    settings: Settings,
    model: str | None,
) -> MeetingSummary:
    """
    Generate a meeting title and summary with DashScope.

    Args:
        result: Normalized transcript.
        settings: Runtime DashScope settings.
        model: Optional model override.

    Returns:
        Structured meeting summary.
    """
    resolved_model = model or settings.dashscope_summary_model
    prompt = _build_summary_prompt(result)
    _configure_dashscope(settings)

    def _call() -> Any:
        response = Generation.call(
            model=resolved_model,
            api_key=settings.dashscope_api_key,
            messages=[
                {"role": "system", "content": _system_prompt()},
                {"role": "user", "content": prompt},
            ],
            result_format="message",
            temperature=0.2,
        )
        _raise_for_generation_error(response)
        return response

    content = _extract_generation_text(retry(_call, attempts=3, delay_seconds=1.0))
    return _parse_summary_text(content, model=resolved_model)


def render_meeting_summary_markdown(summary: MeetingSummary) -> str:
    """
    Render a structured meeting summary as Markdown.

    Args:
        summary: Structured summary.

    Returns:
        Markdown text.
    """
    lines = [
        f"# {summary.title}",
        "",
        "## 摘要",
        summary.summary or "无",
        "",
        "## 议题",
    ]
    lines.extend(_bullet_lines(summary.topics))
    lines.extend(["", "## 待办"])
    lines.extend(_bullet_lines(summary.action_items))
    lines.extend(["", f"模型：{summary.model}", ""])
    return "\n".join(lines)


def _configure_dashscope(settings: Settings) -> None:
    """
    Set DashScope API key and optional base URL.

    Args:
        settings: Runtime settings.

    Returns:
        None.
    """
    dashscope.api_key = settings.dashscope_api_key
    if settings.dashscope_base_url:
        for attr in ("base_http_api_url", "base_url"):
            if hasattr(dashscope, attr):
                setattr(dashscope, attr, settings.dashscope_base_url)


def _system_prompt() -> str:
    """
    Return the fixed summarization system prompt.

    Returns:
        System prompt.
    """
    return "你是专业会议纪要助手。只输出 JSON，不要输出 Markdown，不要解释。"


def _build_summary_prompt(result: TranscriptResult) -> str:
    """
    Build the user prompt from transcript text.

    Args:
        result: Normalized transcript.

    Returns:
        Prompt text.
    """
    transcript = render_speaker_text(result).strip() or result.full_text.strip()
    transcript = _truncate_transcript(transcript)
    return (
        "请根据下面的会议转写生成结构化结果。\n"
        "要求：\n"
        "1. title 使用 8 到 28 个中文字符，概括会议核心主题，不要写日期。\n"
        "2. summary 用 2 到 5 句话说明会议结论和讨论重点。\n"
        "3. topics 是 3 到 8 个要点。\n"
        "4. action_items 是明确待办；没有待办时返回空数组。\n"
        "5. 只返回 JSON，字段为 title, summary, topics, action_items。\n\n"
        f"会议转写：\n{transcript}"
    )


def _truncate_transcript(transcript: str) -> str:
    """
    Bound transcript length before sending it to a text model.

    Args:
        transcript: Rendered transcript.

    Returns:
        Truncated transcript with head and tail context when needed.
    """
    if len(transcript) <= MAX_SUMMARY_TRANSCRIPT_CHARS:
        return transcript
    half = MAX_SUMMARY_TRANSCRIPT_CHARS // 2
    return (
        transcript[:half]
        + "\n\n[中间内容过长，已截断]\n\n"
        + transcript[-half:]
    )


def _raise_for_generation_error(response: Any) -> None:
    """
    Raise when DashScope returns a failed response.

    Args:
        response: DashScope response object.

    Returns:
        None.
    """
    status_code = getattr(response, "status_code", None)
    if status_code and int(status_code) >= 400:
        message = getattr(response, "message", None) or getattr(response, "code", None) or response
        raise RuntimeError(f"DashScope summary failed: HTTP {status_code} {message}")


def _extract_generation_text(response: Any) -> str:
    """
    Extract text from common DashScope generation response shapes.

    Args:
        response: DashScope generation response.

    Returns:
        Generated text.
    """
    output = getattr(response, "output", None)
    text = _field(output, "text")
    if text:
        return str(text)
    choices = _field(output, "choices") or []
    if choices:
        message = _field(choices[0], "message")
        content = _field(message, "content")
        if content:
            return str(content)
    raise RuntimeError("DashScope summary response did not contain generated text.")


def _parse_summary_text(text: str, *, model: str) -> MeetingSummary:
    """
    Parse a model JSON response into a summary.

    Args:
        text: Raw model response.
        model: Model used for generation.

    Returns:
        Structured summary.
    """
    payload = _load_summary_json(text)
    title = _clean_title(str(payload.get("title") or "会议总结"))
    summary = str(payload.get("summary") or "").strip()
    return MeetingSummary(
        title=title,
        summary=summary,
        topics=_string_list(payload.get("topics")),
        action_items=_string_list(payload.get("action_items")),
        model=model,
    )


def _load_summary_json(text: str) -> dict[str, Any]:
    """
    Load JSON from raw model output, including fenced JSON.

    Args:
        text: Raw model output.

    Returns:
        Parsed object.
    """
    cleaned = text.strip()
    if cleaned.startswith("```"):
        cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned)
        cleaned = re.sub(r"\s*```$", "", cleaned)
    try:
        payload = json.loads(cleaned)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", cleaned, flags=re.S)
        if match is None:
            raise RuntimeError(f"DashScope summary response was not JSON: {text}") from None
        payload = json.loads(match.group(0))
    if not isinstance(payload, dict):
        raise RuntimeError("DashScope summary response JSON must be an object.")
    return payload


def _clean_title(title: str) -> str:
    """
    Normalize a generated title for project metadata.

    Args:
        title: Raw title.

    Returns:
        Single-line title.
    """
    return re.sub(r"\s+", " ", title).strip(" #")[:80] or "会议总结"


def _string_list(value: object) -> list[str]:
    """
    Convert a model field into a list of strings.

    Args:
        value: Raw field.

    Returns:
        Clean string list.
    """
    if not isinstance(value, list):
        return []
    return [str(item).strip() for item in value if str(item).strip()]


def _bullet_lines(items: list[str]) -> list[str]:
    """
    Render a list as Markdown bullets.

    Args:
        items: Items to render.

    Returns:
        Markdown bullet lines.
    """
    return [f"- {item}" for item in items] if items else ["- 无"]


def _field(value: Any, key: str) -> Any:
    """
    Read a field from a dict-like or object-like value.

    Args:
        value: Source value.
        key: Field key.

    Returns:
        Field value or None.
    """
    if isinstance(value, dict):
        return value.get(key)
    return getattr(value, key, None)
