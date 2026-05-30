"""Direct unit tests for the deterministic Polish guard.

These lock the guard branches that the JSONL eval set
(``evals/polish_cases.jsonl``) cannot reach. The offline eval runs the guard with
an EMPTY vocabulary so it stays deterministic and independent of the local
lexicon; the vocabulary-aware ASCII restoration (底码 -> Dima) only flips from
"reject" to "accept" when the live lexicon supplies the term, so it is exercised
here with an explicit synthetic vocab instead of the real lexicon database.

The full two-way scoreboard (``evals/polish_scoreboard.py``) regresses the guard
against the on-machine reviewed gold, which holds sensitive meeting transcripts
and is gitignored — it cannot run in CI. The cases below lock the same
recover/hold behaviors with invented text only, no real transcript content.
"""

from __future__ import annotations

from app.models import SentenceSegment
from app.transcript_corrections import (
    _ascii_hallucination_check,
    _is_destutter_only,
    _polish_guard,
)


def _sentences(original: str) -> list[SentenceSegment]:
    """Minimal neighbor context for a guard call with no borrow risk."""
    return [
        SentenceSegment(0, 1000, "", None, 0),
        SentenceSegment(1000, 2000, original, None, 1),
        SentenceSegment(2000, 3000, "", None, 2),
    ]


# --- vocabulary-aware ASCII restoration (the core of this change) --------------


def test_ascii_restoration_to_known_term_accepted() -> None:
    """A homophone restored to a term that IS in the lexicon is a real fix."""
    assert (
        _ascii_hallucination_check("我们跑法兰克任务", "我们跑 Flink 任务", frozenset({"flink"}))
        is None
    )


def test_ascii_restoration_rejected_without_vocab() -> None:
    """Same restoration with no lexicon support stays a fabrication (the default)."""
    assert (
        _ascii_hallucination_check("我们跑法兰克任务", "我们跑 Flink 任务", frozenset())
        == "ascii_hallucination"
    )


def test_ascii_known_term_cannot_smuggle_a_fabrication() -> None:
    """All-or-nothing whitelist: one known term can't usher in an unknown one."""
    assert (
        _ascii_hallucination_check(
            "我们跑法兰克任务", "我们跑 Flink Spark 任务", frozenset({"flink"})
        )
        == "ascii_hallucination"
    )


def test_ascii_resegmentation_accepted_regardless_of_vocab() -> None:
    """Identical ASCII characters, only spacing changed -> not a hallucination."""
    assert _ascii_hallucination_check("用 Open AI 接口", "用 OpenAI 接口", frozenset()) is None


def test_ascii_typo_fix_accepted() -> None:
    """A 1-edit fix of an existing ASCII token (CRI -> CLI) is allowed."""
    assert _ascii_hallucination_check("用 CRI 工具", "用 CLI 工具", frozenset()) is None


# --- de-stutter skeleton: adjacent repeats collapse, distributed deletes don't --


def test_destutter_collapses_adjacent_repeats() -> None:
    """Adjacent stutter (Chinese phrase, English token) reduces to the same skeleton."""
    assert _is_destutter_only("这个方案可以可以。", "这个方案可以。")
    assert _is_destutter_only("这个接口返回 truetrue。", "这个接口返回 true。")


def test_destutter_does_not_mask_distributed_deletion() -> None:
    """Deleting a repeat from a DISTINCT position is not stutter — must not be masked."""
    assert not _is_destutter_only("我觉得这样我觉得那样。", "我觉得这样那样。")


# --- vocab injection end-to-end through _polish_guard --------------------------


def test_polish_guard_accepts_known_restoration_only_with_vocab() -> None:
    """The vocab argument is what flips a known ASCII restoration to accept."""
    original, proposed = "我们跑法兰克任务", "我们跑 Flink 任务"
    sentences = _sentences(original)
    assert _polish_guard(1, sentences, original, proposed, frozenset({"flink"})) is None
    assert (
        _polish_guard(1, sentences, original, proposed, frozenset())
        == "ascii_hallucination"
    )
