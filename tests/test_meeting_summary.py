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
{"title":"AI 转型讨论","summary":"讨论团队 AI 转型路径。","keywords":["飞轮POC本地跑通","518里程碑","诊断准确率30%","LMVK知识图谱","钉钉工作形态"]}
```""",
        model="qwen-test",
    )

    assert summary.title == "AI 转型讨论"
    assert summary.summary == "讨论团队 AI 转型路径。"
    assert summary.keywords == [
        "飞轮POC本地跑通",
        "518里程碑",
        "诊断准确率30%",
        "LMVK知识图谱",
        "钉钉工作形态",
    ]
    assert summary.model == "qwen-test"


def test_parse_summary_text_caps_to_five_keywords_and_strips_short() -> None:
    """Keyword normalization should cap to 5 entries, drop too-short tokens, and dedupe."""
    summary = _parse_summary_text(
        '{"title":"测试","summary":"测试","keywords":["a","飞轮POC","518里程碑","飞轮POC","诊断准确率30%","真实卡单#76","额外条目"]}',
        model="qwen-test",
    )

    # "a" is too short; duplicate "飞轮POC" is dropped; result is capped at 5.
    assert summary.keywords == ["飞轮POC", "518里程碑", "诊断准确率30%", "真实卡单#76", "额外条目"]


def test_render_meeting_summary_markdown_uses_keywords() -> None:
    """Markdown rendering should surface keywords under 关键词 section."""
    summary = _parse_summary_text(
        '{"title":"例会复盘","summary":"完成状态同步。","keywords":["飞轮POC","518里程碑"]}',
        model="qwen-test",
    )

    rendered = render_meeting_summary_markdown(summary)

    assert "# 例会复盘" in rendered
    assert "## 回忆提示\n完成状态同步。" in rendered
    assert "## 关键词\n- 飞轮POC\n- 518里程碑" in rendered
    assert "待办" not in rendered


def test_generate_meeting_summary_uses_single_full_transcript_call(monkeypatch) -> None:
    """Title + summary + keywords come from one call that sees the full transcript."""
    calls = []

    def fake_call(**kwargs):
        calls.append(kwargs)
        prompt = kwargs["messages"][1]["content"]
        # The full transcript must be in the prompt so title generation is grounded.
        assert "会议转写" in prompt
        assert "很长的转写正文" in prompt
        return SimpleNamespace(
            status_code=200,
            output={
                "choices": [
                    {
                        "message": {
                            "content": (
                                '{"title":"维修样板间目标对齐",'
                                '"summary":"讨论维修样板间目标对齐和后续处理节奏。",'
                                '"keywords":["飞轮POC本地跑通","518里程碑","诊断准确率30%","钉钉工作形态","LMVK知识图谱"]}'
                            )
                        }
                    }
                ]
            },
        )

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

    # Title is no longer derived from a second hop; one call covers everything.
    assert len(calls) == 1
    assert calls[0]["model"] == "qwen-test"
    assert summary.title == "维修样板间目标对齐"
    assert summary.summary == "讨论维修样板间目标对齐和后续处理节奏。"
    assert summary.keywords == [
        "飞轮POC本地跑通",
        "518里程碑",
        "诊断准确率30%",
        "钉钉工作形态",
        "LMVK知识图谱",
    ]
