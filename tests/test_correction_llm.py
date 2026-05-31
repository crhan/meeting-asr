"""Tests for DashScope-assisted vocabulary correction proposals."""

from __future__ import annotations

import pytest
import requests

from app.config import Settings
from app.correction_llm import (
    DASHSCOPE_TEXT_REQUEST_TIMEOUT_SECONDS,
    LlmCorrectionCandidate,
    LlmCorrectionSample,
    _build_polish_strict_prompt,
    infer_vocabulary_replacements,
    propose_transcript_polish,
    propose_transcript_polish_strict,
    propose_vocabulary_corrections,
)


def test_polish_strict_prompt_injects_lexicon_disambiguations() -> None:
    """Ambiguous-term guidance from the lexicon must reach the strict polish prompt."""
    candidates = [
        LlmCorrectionCandidate("c0", 0, "speaker", "把IC这边维修平台跟一下。")
    ]
    guidance = "指 iSee 平台时改成 iSee；指个人贡献者(IC)角色时保持原样"

    prompt = _build_polish_strict_prompt(candidates, [("IC", guidance)])

    assert "术语消歧" in prompt
    assert "「IC」" in prompt
    assert guidance in prompt


def test_polish_strict_prompt_has_no_hardcoded_business_terms() -> None:
    """No business platform name may be baked into the prompt; it comes from config."""
    prompt = _build_polish_strict_prompt(
        [LlmCorrectionCandidate("c0", 0, "speaker", "测试句子。")]
    )

    assert "术语消歧" not in prompt
    # iSee is our business data and must only enter via lexicon-driven guidance.
    assert "iSee" not in prompt


def test_propose_vocabulary_corrections_parses_dashscope_json(monkeypatch) -> None:
    """DashScope correction helper should return validated candidate edits."""
    calls = {}

    def fake_call(**kwargs):
        calls.update(kwargs)
        return '{"understanding":"艾赛应为 iSee","corrections":[{"id":"c1","corrected_text":"我们看 iSee 系统。"}]}'

    monkeypatch.setattr("app.correction_llm.generate_chat_text", fake_call)
    settings = Settings(
        dashscope_api_key="key", dashscope_base_url="https://dashscope.example.com"
    )

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
    assert calls["enable_thinking"] is False
    assert result.understanding == "艾赛应为 iSee"
    assert result.corrected_text_by_id == {"c1": "我们看 iSee 系统。"}


def test_infer_vocabulary_replacements_parses_term_level_rules(monkeypatch) -> None:
    """DashScope replacement inference should parse corrected Chinese term boundaries."""

    calls = {}

    def fake_call(**kwargs):
        calls.update(kwargs)
        return (
            '{"replacements":[{"wrong_text":"云原声","corrected_text":"云原生",'
            '"left_context":"建设","right_context":"平台","reason":"系统术语"}]}'
        )

    monkeypatch.setattr("app.correction_llm.generate_chat_text", fake_call)
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
        return (
            '{"understanding":"修复入参/出参语序",'
            '"corrections":[{"id":"c1","corrected_text":"入参和出参需要记录起来。"}]}'
        )

    monkeypatch.setattr("app.correction_llm.generate_chat_text", fake_call)
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


def test_propose_transcript_polish_does_not_retry_read_timeout(monkeypatch) -> None:
    """Transcript polish should fail fast because project run can continue without it."""
    calls = 0

    def fake_call(**kwargs):
        nonlocal calls
        calls += 1
        raise requests.Timeout("read timeout")

    monkeypatch.setattr("app.correction_llm.generate_chat_text", fake_call)
    settings = Settings(dashscope_api_key="key", dashscope_base_url=None)

    with pytest.raises(
        RuntimeError, match="Operation failed after 1 retryable attempts"
    ):
        propose_transcript_polish(
            candidates=[
                LlmCorrectionCandidate("c1", 1, "米汤", "需要记录入参和出参。")
            ],
            settings=settings,
            model="qwen-test",
        )

    assert calls == 1


def _polish_strict_candidates(count: int) -> list[LlmCorrectionCandidate]:
    """Build N strict-polish candidates with stable ids c0..c{N-1}."""
    return [
        LlmCorrectionCandidate(f"c{index}", index, "speaker", f"待修正句子{index}。")
        for index in range(count)
    ]


def test_load_json_object_tolerates_unescaped_control_chars(monkeypatch) -> None:
    """qwen sometimes emits a literal newline inside a string value; strict=False parses it."""

    # Literal \n inside corrected_text is the most common malformed-JSON we see.
    content = (
        '{"understanding":"修字",'
        '"corrections":[{"id":"c0","corrected_text":"第一行\n第二行","change_type":"typo","reason":"x"}]}'
    )
    monkeypatch.setattr("app.correction_llm.generate_chat_text", lambda **_: content)
    settings = Settings(dashscope_api_key="key", dashscope_base_url=None)

    result = propose_transcript_polish_strict(
        candidates=_polish_strict_candidates(1),
        settings=settings,
        model="qwen-test",
    )

    assert [item.candidate_id for item in result.items] == ["c0"]
    assert result.items[0].corrected_text == "第一行\n第二行"


def test_load_json_object_salvages_items_when_response_is_truncated(
    monkeypatch,
) -> None:
    """Truncated mid-array response should still surface every complete item."""

    # Two complete items, then a half-written third item (mid-key). Real
    # symptom is `Expecting ',' delimiter: line N column 6` at max_tokens.
    content = (
        '{"understanding":"修字","corrections":['
        '{"id":"c0","corrected_text":"OK0","change_type":"typo","reason":"r0"},'
        '{"id":"c1","corrected_text":"OK1","change_type":"typo","reason":"r1"},'
        '{"id":"c2","corrected_text":"OK'
    )
    monkeypatch.setattr("app.correction_llm.generate_chat_text", lambda **_: content)
    settings = Settings(dashscope_api_key="key", dashscope_base_url=None)

    result = propose_transcript_polish_strict(
        candidates=_polish_strict_candidates(3),
        settings=settings,
        model="qwen-test",
    )

    # Two intact items kept, half-written third dropped — better than losing the batch.
    assert {item.candidate_id for item in result.items} == {"c0", "c1"}
    assert {item.corrected_text for item in result.items} == {"OK0", "OK1"}


def test_load_json_object_salvages_around_one_bad_item(monkeypatch) -> None:
    """A single item with unescaped quote should not poison the surrounding items."""

    # Middle item has an unescaped " inside corrected_text — JSON parse fails
    # at that record but the others are intact JSON objects.
    bad_inner = (
        '{"id":"c1","corrected_text":"含"号的句子","change_type":"typo","reason":"r1"}'
    )
    content = (
        '{"understanding":"修字","corrections":['
        '{"id":"c0","corrected_text":"OK0","change_type":"typo","reason":"r0"},'
        f"{bad_inner},"
        '{"id":"c2","corrected_text":"OK2","change_type":"typo","reason":"r2"}]}'
    )
    monkeypatch.setattr("app.correction_llm.generate_chat_text", lambda **_: content)
    settings = Settings(dashscope_api_key="key", dashscope_base_url=None)

    result = propose_transcript_polish_strict(
        candidates=_polish_strict_candidates(3),
        settings=settings,
        model="qwen-test",
    )

    assert {item.candidate_id for item in result.items} == {"c0", "c2"}


def test_load_json_object_raises_when_no_json_present(monkeypatch) -> None:
    """If the model returns prose with no usable JSON object, surface a clear error."""

    monkeypatch.setattr(
        "app.correction_llm.generate_chat_text",
        lambda **_: "Sorry, I cannot help with that.",
    )
    settings = Settings(dashscope_api_key="key", dashscope_base_url=None)

    with pytest.raises(RuntimeError, match="was not JSON"):
        propose_transcript_polish_strict(
            candidates=_polish_strict_candidates(1),
            settings=settings,
            model="qwen-test",
        )
