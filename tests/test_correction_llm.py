"""Tests for DashScope-assisted vocabulary correction proposals."""

from __future__ import annotations

from types import SimpleNamespace

from app.config import Settings
from app.correction_llm import LlmCorrectionCandidate, LlmCorrectionSample, propose_vocabulary_corrections


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
