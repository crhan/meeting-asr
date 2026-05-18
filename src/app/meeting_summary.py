"""DashScope meeting memory-index helpers.

Produces title, recall summary, and discriminating keywords in a single
LLM round-trip against the full meeting transcript.
"""

from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass
from typing import Any

from app.config import Settings
from app.dashscope_chat import generate_chat_text
from app.models import TranscriptResult
from app.postprocess import render_speaker_text
from app.utils import retry

# qwen-plus base model has a 131k-token context window. An 80k-character
# Chinese meeting transcript is roughly 50k tokens, well inside that budget.
# Guard against pathological inputs (multi-hour transcripts that drift past
# the model's input cap) with a soft cliff rather than chopping average
# meetings into pieces.
TRANSCRIPT_HARD_LIMIT_CHARS = 200_000
KEYWORD_COUNT = 5
KEYWORD_MIN_CHARS = 3
KEYWORD_MAX_CHARS = 12
TITLE_MAX_CHARS = 80


@dataclass(frozen=True, slots=True)
class MeetingSummary:
    """Lightweight meeting memory index generated from a transcript."""

    title: str
    summary: str
    keywords: list[str]
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
    Generate a meeting memory index with one DashScope call.

    Args:
        result: Normalized transcript.
        settings: Runtime DashScope settings.
        model: Optional model override.

    Returns:
        Lightweight meeting memory index.
    """
    resolved_model = model or settings.dashscope_summary_model
    raw = _call_generation(
        model=resolved_model,
        settings=settings,
        system_prompt=_memory_system_prompt(),
        prompt=_build_memory_prompt(result),
    )
    return _parse_summary_text(raw, model=resolved_model)


def render_meeting_summary_markdown(summary: MeetingSummary) -> str:
    """
    Render a meeting memory index as Markdown.

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
    lines.extend(_bullet_lines(summary.keywords))
    lines.extend(["", f"模型：{summary.model}", ""])
    return "\n".join(lines)


def _call_generation(*, model: str, settings: Settings, system_prompt: str, prompt: str) -> str:
    """Call DashScope text generation and return message content."""

    def _call() -> str:
        return generate_chat_text(
            settings=settings,
            model=model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt},
            ],
            request_timeout=120,
            temperature=0.2,
            enable_thinking=False,
        )

    return retry(_call, attempts=3, delay_seconds=1.0)


def _memory_system_prompt() -> str:
    """Return the unified meeting memory-index system prompt."""
    return (
        "你是会议回忆索引助手。"
        "你的任务是从会议转写里直接产出标题、回忆提示、关键词。"
        "所有产出必须忠实于转写，不允许编造转写里没出现的具体数字、版本号、命令名、人名、卡单号。"
        "只输出 JSON，不要输出 Markdown 包裹、不要解释。"
    )


def _build_memory_prompt(result: TranscriptResult) -> str:
    """Build the unified memory-index prompt with the full transcript."""
    transcript = render_speaker_text(result).strip() or result.full_text.strip()
    transcript = _bound_transcript(transcript)
    return (
        "请根据下面这场会议的真实转写，输出 JSON：{\"title\": \"...\", \"summary\": \"...\", \"keywords\": [\"...\", ...]}。\n"
        "\n"
        "三个字段都必须严格满足以下规则：\n"
        "\n"
        "[title]\n"
        "1. 8 到 28 个中文字符，概括这场会议的核心讨论。\n"
        "2. 不要写日期、不要写 '会议总结' '会议纪要' '总结' 这种词、不要写待办。\n"
        "3. 倾向于使用一个能让你 1 个月后想起'这是哪场会议'的具体说法。\n"
        "\n"
        "[summary]\n"
        "1. 1 到 2 句中文，目标是让人快速想起这是哪一场会议、讨论了什么、得出了什么。\n"
        "2. 不要写正式纪要、不要扩写结论、不要列待办。\n"
        "\n"
        f"[keywords]\n"
        f"1. 恰好 {KEYWORD_COUNT} 个关键字，用于在项目列表里区分这场会议和其他同主题会议。\n"
        f"2. 每个关键字 {KEYWORD_MIN_CHARS} 到 {KEYWORD_MAX_CHARS} 个中文字符或词，禁止整句、禁止冒号、禁止省略号。\n"
        "3. **必须直接来源于转写**：人名、项目名、产品名、代号、命令名、技术术语、转写里出现过的数字、卡单号、里程碑、决议。\n"
        "4. **严禁编造**：转写里没出现过的数字、版本号、SLA、卡单号、命令名、工具名都不能写。宁可没有数字也不要造假。\n"
        "5. **严禁泛词单独成词**：'AI诊断'、'评测体系'、'维修流程'、'团队协同'、'飞轮'、'AI'、'诊断'、'评测'、'优化' 单独都不允许，要带上限定语让它落到这场会议（例如 '飞轮POC本地跑通'、'诊断准确率30%'、'诊断站点A3A6A8'）。\n"
        "6. **忽略文件名里的乱码**：不要把那种几十位的十六进制 hash 写成关键字。\n"
        "7. 关键字之间要互相补充，覆盖会议里 4-5 个不同的具体讨论对象，不要 5 个都是同一件事的变体。\n"
        "\n"
        "=== 会议转写 ===\n"
        f"{transcript}\n"
        "=== 转写结束 ==="
    )


def _bound_transcript(transcript: str) -> str:
    """Apply only the hard upper-bound cliff; do not split typical meetings."""
    if len(transcript) <= TRANSCRIPT_HARD_LIMIT_CHARS:
        return transcript
    # Multi-hour pathological case: keep head and tail, mark drop explicitly.
    half = TRANSCRIPT_HARD_LIMIT_CHARS // 2
    return transcript[:half] + "\n\n[中段超长已省略]\n\n" + transcript[-half:]


def _parse_summary_text(text: str, *, model: str) -> MeetingSummary:
    """Parse the unified JSON response into a MeetingSummary."""
    payload = _load_summary_json(text)
    title = _clean_title(str(payload.get("title") or "会议总结"))
    summary = str(payload.get("summary") or "").strip()
    keywords = _normalize_keywords(payload.get("keywords"))
    return MeetingSummary(title=title, summary=summary, keywords=keywords, model=model)


def _load_summary_json(text: str) -> dict[str, Any]:
    """Load JSON from raw model output, including fenced JSON."""
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
    """Normalize a generated title for project metadata."""
    return re.sub(r"\s+", " ", title).strip(" #")[:TITLE_MAX_CHARS] or "会议总结"


def _normalize_keywords(value: object) -> list[str]:
    """Trim, dedupe, length-clamp and cap keyword count."""
    if not isinstance(value, list):
        return []
    out: list[str] = []
    seen: set[str] = set()
    for item in value:
        text = re.sub(r"\s+", "", str(item)).strip(" #'\"`，。：:、·…—-")
        if not text:
            continue
        if len(text) > KEYWORD_MAX_CHARS:
            text = text[:KEYWORD_MAX_CHARS]
        if len(text) < KEYWORD_MIN_CHARS:
            continue
        if text in seen:
            continue
        seen.add(text)
        out.append(text)
        if len(out) >= KEYWORD_COUNT:
            break
    return out


def _bullet_lines(items: list[str]) -> list[str]:
    """Render a list as Markdown bullets."""
    return [f"- {item}" for item in items] if items else ["- 无"]

