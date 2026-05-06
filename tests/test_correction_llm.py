"""Tests for DashScope-assisted vocabulary correction proposals."""

from __future__ import annotations

from types import SimpleNamespace

from app.config import Settings
from app.correction_llm import (
    DASHSCOPE_TEXT_REQUEST_TIMEOUT_SECONDS,
    LlmCorrectionCandidate,
    LlmCorrectionSample,
    infer_vocabulary_replacements,
    propose_transcript_polish,
    propose_vocabulary_corrections,
)


def test_propose_vocabulary_corrections_parses_dashscope_json(monkeypatch) -> None:
    """DashScope correction helper should return validated candidate edits."""
    calls = {}

    def fake_call(**kwargs):
        calls.update(kwargs)
        content = '{"understanding":"艾赛应为 iSee","corrections":[{"id":"c1","corrected_text":"我们看 iSee 系统。"}]}'
        return SimpleNamespace(status_code=200, output={"choices": [{"message": {"content": content}}]})

    monkeypatch.setattr("app.correction_llm.Generation.call", fake_call)
    settings = Settings(dashscope_api_key="key", dashscope_base_url="https://dashscope.example.com")

    result = propose_vocabulary_corrections(
        samples=[
            LlmCorrectionSample(
                original_text="我们看艾赛系统。",
                corrected_text="我们看 iSee 系统。",
                replacements=[{"wrong_text": "艾赛", "corrected_text": "iSee"}],
            )
        ],
        candidates=[LlmCorrectionCandidate("c1", 1, "敬悦", "我们看艾赛系统。")],
        settings=settings,
        model="qwen-test",
    )

    assert calls["model"] == "qwen-test"
    assert calls["request_timeout"] == DASHSCOPE_TEXT_REQUEST_TIMEOUT_SECONDS
    assert result.understanding == "艾赛应为 iSee"
    assert result.corrected_text_by_id == {"c1": "我们看 iSee 系统。"}


def test_infer_vocabulary_replacements_parses_term_level_rules(monkeypatch) -> None:
    """DashScope replacement inference should parse corrected Chinese term boundaries."""

    calls = {}

    def fake_call(**kwargs):
        calls.update(kwargs)
        content = (
            '{"replacements":[{"wrong_text":"云原声","corrected_text":"云原生",'
            '"left_context":"建设","right_context":"平台","reason":"系统术语"}]}'
        )
        return SimpleNamespace(status_code=200, output={"choices": [{"message": {"content": content}}]})

    monkeypatch.setattr("app.correction_llm.Generation.call", fake_call)
    settings = Settings(dashscope_api_key="key", dashscope_base_url=None)

    rules = infer_vocabulary_replacements(
        samples=[
            LlmCorrectionSample(
                original_text="我们建设云原声平台。",
                corrected_text="我们建设云原生平台。",
                replacements=[{"wrong_text": "声", "corrected_text": "生"}],
            )
        ],
        settings=settings,
        model="qwen-test",
    )

    assert len(rules) == 1
    assert calls["request_timeout"] == DASHSCOPE_TEXT_REQUEST_TIMEOUT_SECONDS
    assert rules[0].wrong_text == "云原声"
    assert rules[0].corrected_text == "云原生"


def test_propose_transcript_polish_parses_dashscope_json(monkeypatch) -> None:
    """Transcript polish helper should allow safe wording and word-order fixes."""
    calls = {}

    def fake_call(**kwargs):
        calls.update(kwargs)
        content = (
            '{"understanding":"修复入参/出参语序",'
            '"corrections":[{"id":"c1","corrected_text":"入参和出参需要记录起来。"}]}'
        )
        return SimpleNamespace(status_code=200, output={"choices": [{"message": {"content": content}}]})

    monkeypatch.setattr("app.correction_llm.Generation.call", fake_call)
    settings = Settings(dashscope_api_key="key", dashscope_base_url=None)

    result = propose_transcript_polish(
        candidates=[
            LlmCorrectionCandidate(
                "c1",
                813,
                "米汤",
                "这个入参的时候输出什么，然后出参的时候输出什么，就类有点类似于出参跟入参记录起来",
            )
        ],
        settings=settings,
        model="qwen-test",
    )

    prompt = calls["messages"][1]["content"]
    assert calls["request_timeout"] == DASHSCOPE_TEXT_REQUEST_TIMEOUT_SECONDS
    assert "语序断裂" in prompt
    assert "不要总结、扩写" in prompt
    assert result.understanding == "修复入参/出参语序"
    assert result.corrected_text_by_id == {"c1": "入参和出参需要记录起来。"}
