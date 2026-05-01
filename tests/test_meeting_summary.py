"""Tests for DashScope meeting summary parsing and rendering."""

from __future__ import annotations

from app.meeting_summary import _parse_summary_text, render_meeting_summary_markdown


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


def test_render_meeting_summary_markdown_handles_empty_lists() -> None:
    """Markdown output should remain useful when the model finds no action items."""
    summary = _parse_summary_text(
        '{"title":"例会复盘","summary":"完成状态同步。","topics":[],"action_items":[]}',
        model="qwen-test",
    )

    rendered = render_meeting_summary_markdown(summary)

    assert "# 例会复盘" in rendered
    assert "## 议题\n- 无" in rendered
    assert "## 待办\n- 无" in rendered
