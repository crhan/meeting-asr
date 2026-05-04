"""Tests for DashScope-assisted vocabulary correction proposals."""

from __future__ import annotations

from types import SimpleNamespace

from app.config import Settings
from app.correction_llm import (
    LlmCorrectionCandidate,
    LlmCorrectionSample,
    infer_vocabulary_replacements,
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
    assert result.understanding == "艾赛应为 iSee"
    assert result.corrected_text_by_id == {"c1": "我们看 iSee 系统。"}


def test_infer_vocabulary_replacements_parses_term_level_rules(monkeypatch) -> None:
    """DashScope replacement inference should parse corrected Chinese term boundaries."""

    def fake_call(**kwargs):
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
    assert rules[0].wrong_text == "云原声"
    assert rules[0].corrected_text == "云原生"
