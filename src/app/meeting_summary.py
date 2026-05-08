"""DashScope meeting memory-index helpers."""

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
    """Lightweight meeting memory index generated from a transcript."""

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
    Generate a lightweight meeting memory index with DashScope.

    Args:
        result: Normalized transcript.
        settings: Runtime DashScope settings.
        model: Optional model override.

    Returns:
        Lightweight meeting memory index.
    """
    resolved_model = model or settings.dashscope_summary_model
    _configure_dashscope(settings)
    memory = _parse_summary_text(
        _call_generation(
            model=resolved_model,
            settings=settings,
            system_prompt=_memory_system_prompt(),
            prompt=_build_memory_prompt(result),
        ),
        model=resolved_model,
    )
    title = _parse_title_text(
        _call_generation(
            model=resolved_model,
            settings=settings,
            system_prompt=_title_system_prompt(),
            prompt=_build_title_prompt(memory),
        )
    )
    return MeetingSummary(title, memory.summary, memory.topics, [], resolved_model)


def render_meeting_summary_markdown(summary: MeetingSummary) -> str:
    """
    Render a lightweight meeting memory index as Markdown.

    Args:
        summary: Lightweight memory index.

    Returns:
        Markdown text.
    """
    lines = [
        f"# {summary.title}",
        "",
        "## 回忆提示",
        summary.summary or "无",
        "",
        "## 关键词",
    ]
    lines.extend(_bullet_lines(summary.topics))
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


def _call_generation(*, model: str, settings: Settings, system_prompt: str, prompt: str) -> str:
    """
    Call DashScope text generation and return message content.

    Args:
        model: DashScope model name.
        settings: Runtime DashScope settings.
        system_prompt: System prompt.
        prompt: User prompt.

    Returns:
        Generated message content.
    """

    def _call() -> Any:
        response = Generation.call(
            model=model,
            api_key=settings.dashscope_api_key,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt},
            ],
            result_format="message",
            temperature=0.2,
        )
        _raise_for_generation_error(response)
        return response

    return _extract_generation_text(retry(_call, attempts=3, delay_seconds=1.0))


def _memory_system_prompt() -> str:
    """
    Return the fixed memory-index system prompt.

    Returns:
        System prompt.
    """
    return "你是会议回忆索引助手。只输出 JSON，不要输出 Markdown，不要解释。"


def _title_system_prompt() -> str:
    """
    Return the fixed title-generation system prompt.

    Returns:
        System prompt.
    """
    return "你是会议标题助手。只输出 JSON，不要输出 Markdown，不要解释。"


def _build_memory_prompt(result: TranscriptResult) -> str:
    """
    Build the memory-index prompt from transcript text.

    Args:
        result: Normalized transcript.

    Returns:
        Prompt text.
    """
    transcript = render_speaker_text(result).strip() or result.full_text.strip()
    transcript = _truncate_transcript(transcript)
    return (
        "请根据下面的会议转写生成一个很短的回忆索引。\n"
        "要求：\n"
        "1. summary 只写 1 到 2 句话，目标是让人快速想起这是哪一场会议。\n"
        "2. 不要写正式纪要，不要写待办事项，不要扩展结论。\n"
        "3. topics 是 3 到 6 个短关键词或场景词，用于检索和回忆。\n"
        "4. 只返回 JSON，字段为 summary, topics。\n\n"
        f"会议转写：\n{transcript}"
    )


def _build_title_prompt(summary: MeetingSummary) -> str:
    """
    Build a compact title prompt from the generated memory index.

    Args:
        summary: Lightweight memory index without the final title.

    Returns:
        Prompt text.
    """
    topics = "、".join(summary.topics) if summary.topics else "无"
    return (
        "请根据下面的会议回忆索引生成一个短标题。\n"
        "要求：\n"
        "1. title 使用 8 到 28 个中文字符，概括这场会议，方便在项目列表里识别。\n"
        "2. 不要写日期，不要写“会议总结”，不要写待办。\n"
        "3. 只返回 JSON，字段为 title。\n\n"
        f"回忆提示：{summary.summary}\n"
        f"关键词：{topics}"
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
    return transcript[:half] + "\n\n[中间内容过长，已截断]\n\n" + transcript[-half:]


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
    Parse a model JSON response into a lightweight memory index.

    Args:
        text: Raw model response.
        model: Model used for generation.

    Returns:
        Lightweight memory index.
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


def _parse_title_text(text: str) -> str:
    """
    Parse a generated title response.

    Args:
        text: Raw model response.

    Returns:
        Clean title.
    """
    try:
        payload = _load_summary_json(text)
    except RuntimeError:
        return _clean_title(text)
    return _clean_title(str(payload.get("title") or text))


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
