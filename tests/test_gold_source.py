"""Lock the gold-source classification that separates independent from circular gold.

The whole de-circularization rests on one invariant: every gold row is tagged with
a source, and exactly the destutter / despace / ascii_vocab sources are "circular"
(decided by the guard's own functions) while audio / panel / codex are independent.
If that mapping or the precedence order silently changes, the scoreboard's
"独立金标 / 循环金标" split becomes wrong without any test failing — so pin it here.

These tests use only pure functions with synthetic inputs (no lexicon DB, no local
gold files), so they are deterministic and CI-safe.
"""

from __future__ import annotations

import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from evals.assemble_gold import (  # noqa: E402
    GOLD_SOURCE_INDEPENDENT,
    GOLD_SOURCE_LABEL,
    _classify_gold,
)


def test_source_independence_map_is_exhaustive_and_correct() -> None:
    """The 7 sources split into exactly 4 independent + 3 circular, no more no less."""
    assert set(GOLD_SOURCE_INDEPENDENT) == {
        "audio_human", "panel", "codex_keep", "codex_reject",
        "destutter", "despace", "ascii_vocab",
    }
    independent = {k for k, v in GOLD_SOURCE_INDEPENDENT.items() if v}
    circular = {k for k, v in GOLD_SOURCE_INDEPENDENT.items() if not v}
    assert independent == {"audio_human", "panel", "codex_keep", "codex_reject"}
    # The circular set is precisely the guard-derived rules; this is the load-bearing
    # claim of the whole refactor — changing it must break this test.
    assert circular == {"destutter", "despace", "ascii_vocab"}
    # every source has a human label
    assert set(GOLD_SOURCE_LABEL) == set(GOLD_SOURCE_INDEPENDENT)


def _classify(row: dict, *, overrides=None, panel=None, vocab=None):
    """Thin wrapper passing synthetic precedence inputs to _classify_gold."""
    key = (row["original_text"], row["proposed_text"])
    return _classify_gold(
        row, key, overrides=overrides or {}, panel=panel or {}, vocab=vocab or set()
    )


def test_audio_override_wins_over_everything() -> None:
    """A human audio ruling beats destutter/panel/codex for the same pair."""
    # 可以可以 -> 可以 is destutter-only, but an audio override must still win.
    row = {"original_text": "可以可以", "proposed_text": "可以", "_kind": "reject"}
    ov = {("可以可以", "可以"): "reject"}
    gold, source = _classify(row, overrides=ov)
    assert (gold, source) == ("reject", "audio_human")
    assert GOLD_SOURCE_INDEPENDENT[source] is True


def test_destutter_is_circular_keep() -> None:
    """Adjacent-stutter collapse classifies as the circular destutter source."""
    row = {"original_text": "就是就是这样", "proposed_text": "就是这样", "_kind": "reject"}
    gold, source = _classify(row)
    assert (gold, source) == ("keep", "destutter")
    assert GOLD_SOURCE_INDEPENDENT[source] is False


def test_despace_is_circular_keep() -> None:
    """Ascii re-segmentation (plus other edits) classifies as circular despace.

    A PURE despace (Open AI -> OpenAI with nothing else changed) is shadowed by
    destutter, which strips spaces first and would match the skeleton — that is
    correct, both are circular keeps. The despace branch only fires when the row
    ALSO changed something destutter doesn't normalize (here, a trailing word),
    so the skeletons differ and destutter declines. This mirrors all 133 real
    despace rows, none of which are destutter-only.
    """
    row = {
        "original_text": "我们再 case by case 看",
        "proposed_text": "我们再 casebycase 看一下",
        "_kind": "reject",
    }
    # sanity: this is genuinely despace-but-not-destutter, like the real data
    from app.transcript_corrections import _is_destutter_only

    assert not _is_destutter_only(row["original_text"], row["proposed_text"])
    gold, source = _classify(row)
    assert (gold, source) == ("keep", "despace")
    assert GOLD_SOURCE_INDEPENDENT[source] is False


def test_ascii_vocab_circular_uses_passed_vocab() -> None:
    """A restored ascii term in vocab -> circular keep; out of vocab -> circular reject."""
    row = {
        "original_text": "用底码部署", "proposed_text": "用 Dima 部署",
        "_kind": "reject", "category": "ascii_hallucination",
    }
    keep_gold, keep_src = _classify(row, vocab={"dima"})
    assert (keep_gold, keep_src) == ("keep", "ascii_vocab")
    reject_gold, reject_src = _classify(row, vocab=set())
    assert (reject_gold, reject_src) == ("reject", "ascii_vocab")
    assert GOLD_SOURCE_INDEPENDENT["ascii_vocab"] is False


def test_panel_is_independent() -> None:
    """A non-ascii contested pair resolved by the panel is independent gold."""
    row = {"original_text": "他说要上线", "proposed_text": "他说今天要上线", "_kind": "reject"}
    panel = {("他说要上线", "他说今天要上线"): "reject"}
    gold, source = _classify(row, panel=panel)
    assert (gold, source) == ("reject", "panel")
    assert GOLD_SOURCE_INDEPENDENT[source] is True


def test_codex_fallback_is_independent() -> None:
    """With no higher-priority signal, codex_verdict decides and is independent."""
    # A real word deletion (其实) that destutter does NOT normalize, so the row
    # falls through to codex. '啊'/'吧'-style fillers are in the destutter noise
    # set and would be caught earlier — that is why this uses a content word.
    keep_row = {
        "original_text": "他其实想说这个", "proposed_text": "他想说这个",
        "_kind": "accept", "codex_verdict": "keep",
    }
    from app.transcript_corrections import _is_destutter_only

    assert not _is_destutter_only(keep_row["original_text"], keep_row["proposed_text"])
    gold, source = _classify(keep_row)
    assert (gold, source) == ("keep", "codex_keep")
    assert GOLD_SOURCE_INDEPENDENT[source] is True

    reject_row = {
        "original_text": "他说要上线", "proposed_text": "完全不同的另一句话内容",
        "_kind": "reject", "codex_verdict": "reject",
    }
    gold, source = _classify(reject_row)
    assert (gold, source) == ("reject", "codex_reject")
    assert GOLD_SOURCE_INDEPENDENT[source] is True


def test_classification_is_deterministic() -> None:
    """Same input -> same (gold, source) across repeated calls (no hidden state)."""
    row = {"original_text": "就是就是这样", "proposed_text": "就是这样", "_kind": "reject"}
    results = {_classify(row) for _ in range(5)}
    assert results == {("keep", "destutter")}
