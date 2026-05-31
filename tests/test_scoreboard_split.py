"""Lock the scoreboard's independent/circular split gating (Codex review P2).

The split columns describe how ``gold_verdict`` was decided, so they must appear
ONLY when ``gold_verdict`` is the field being scored. Scoring the weaker
``codex_verdict`` (the default) must fall back to a plain single-column report
instead of reusing ``gold_verdict``'s independence tags to split a different
label's counts — otherwise the columns are mislabeled and, on an untagged
reviewed file, every row would be reported as independent, hiding the very
circular rows this whole change exists to expose.

guard_decision is monkeypatched to a per-row literal so these tests need no
lexicon DB and stay deterministic / CI-safe.
"""

from __future__ import annotations

import contextlib
import io
import sys
from pathlib import Path

import pytest

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from evals import polish_scoreboard as sb  # noqa: E402


def _run(rows: list[dict], gold_field: str, monkeypatch: pytest.MonkeyPatch) -> str:
    """Run the scoreboard over rows with a literal guard, capturing stdout."""
    monkeypatch.setattr(sb, "guard_decision", lambda row: row["_guard"])
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        sb.run(rows, gold_field)
    return buf.getvalue()


def _reject_rows() -> list[dict]:
    """Two over-rejections recovered by the guard: one independent, one circular."""
    return [
        {"_kind": "reject", "gold_verdict": "keep", "gold_independent": True, "_guard": "keep"},
        {"_kind": "reject", "gold_verdict": "keep", "gold_independent": False, "_guard": "keep"},
    ]


def test_split_emitted_for_gold_verdict(monkeypatch: pytest.MonkeyPatch) -> None:
    """Scoring gold_verdict shows both the independent and circular columns."""
    out = _run(_reject_rows(), "gold_verdict", monkeypatch)
    assert "独立金标 救回" in out
    assert "循环金标 救回" in out
    # one recovered row is independent (1/1), one is circular (1/1)
    assert out.count("1/1 (100.0%)") >= 2


def test_no_split_when_scoring_codex_verdict(monkeypatch: pytest.MonkeyPatch) -> None:
    """Scoring codex_verdict must NOT split, even if rows carry gold_independent."""
    rows = [{**r, "codex_verdict": r["gold_verdict"]} for r in _reject_rows()]
    out = _run(rows, "codex_verdict", monkeypatch)
    assert "独立金标" not in out
    assert "循环金标" not in out
    # the plain aggregate still counts BOTH rows — nothing hidden, just not split
    assert "救回 2/2 (100.0%)" in out
    # and it warns that a weaker field was chosen while gold_verdict exists
    assert "⚠" in out and "gold_verdict" in out


def test_untagged_gold_verdict_counts_as_independent(monkeypatch: pytest.MonkeyPatch) -> None:
    """A gold_verdict row with no source tag is counted independent and flagged."""
    rows = [{"_kind": "reject", "gold_verdict": "keep", "_guard": "keep"}]
    out = _run(rows, "gold_verdict", monkeypatch)
    assert "无 gold_source 标记" in out
    assert "独立金标 救回" in out and "1/1 (100.0%)" in out  # counted independent
    assert "0/0 (n/a)" in out  # circular column is empty, not fabricated
