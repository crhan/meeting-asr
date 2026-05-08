"""Tests for DashScope meeting memory-index parsing and rendering."""

from __future__ import annotations

from types import SimpleNamespace

from app.config import Settings
from app.meeting_summary import _parse_summary_text, generate_meeting_summary, render_meeting_summary_markdown
from app.models import SentenceSegment, TranscriptResult


def test_parse_summary_text_accepts_fenced_json() -> None:
    """Model output may come back as fenced JSON despite prompt instructions."""
    summary = _parse_summary_text(
        """```json
{"title":"AI 转型讨论","summary":"讨论团队 AI 转型路径。","topics":["目标","落地"],"action_items":["补充方案"]}
```""",
        model="qwen-test",
    )

    assert summary.title == "AI 转型讨论"
    assert summary.summary == "讨论团队 AI 转型路径。"
    assert summary.topics == ["目标", "落地"]
    assert summary.action_items == ["补充方案"]
    assert summary.model == "qwen-test"


def test_render_meeting_summary_markdown_is_memory_oriented() -> None:
    """Markdown output should be a memory cue, not a formal action-item summary."""
    summary = _parse_summary_text(
        '{"title":"例会复盘","summary":"完成状态同步。","topics":[],"action_items":[]}',
        model="qwen-test",
    )

    rendered = render_meeting_summary_markdown(summary)

    assert "# 例会复盘" in rendered
    assert "## 回忆提示\n完成状态同步。" in rendered
    assert "## 关键词\n- 无" in rendered
    assert "待办" not in rendered


def test_generate_meeting_summary_derives_title_from_short_memory(monkeypatch) -> None:
    """Title generation should use the short memory index, not the full transcript."""
    calls = []

    def fake_call(**kwargs):
        calls.append(kwargs)
        if len(calls) == 1:
            content = '{"summary":"讨论维修样板间目标对齐和后续处理节奏。","topics":["维修样板间","目标对齐"]}'
        else:
            prompt = kwargs["messages"][1]["content"]
            assert "会议转写" not in prompt
            assert "很长的转写正文" not in prompt
            assert "维修样板间" in prompt
            content = '{"title":"维修样板间目标对齐"}'
        return SimpleNamespace(status_code=200, output={"choices": [{"message": {"content": content}}]})

    monkeypatch.setattr("app.meeting_summary.Generation.call", fake_call)
    result = TranscriptResult(
        "很长的转写正文",
        [SentenceSegment(0, 1000, "很长的转写正文", 0, 1)],
        [0],
    )

    summary = generate_meeting_summary(
        result,
        settings=Settings(dashscope_api_key="key", dashscope_base_url=None, dashscope_summary_model="qwen-test"),
        model=None,
    )

    assert len(calls) == 2
    assert calls[0]["model"] == "qwen-test"
    assert calls[1]["model"] == "qwen-test"
    assert summary.title == "维修样板间目标对齐"
    assert summary.summary == "讨论维修样板间目标对齐和后续处理节奏。"
    assert summary.topics == ["维修样板间", "目标对齐"]
    assert summary.action_items == []
