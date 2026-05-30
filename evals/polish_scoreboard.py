"""Score the Polish guard against the reviewed gold standard (two-way metrics).

The eval set has two subsets with DIFFERENT sampling rates, so a single global
rate is misleading — we report each subset on its own terms:

  * 被拒集 (reject set, FULL census of what the baseline guard rejected): the
    gold split tells us how many rejects were over-rejections (gold=keep). A good
    guard change RECOVERS these (re-decides keep) WITHOUT also flipping the
    genuinely-dangerous rejects (gold=reject) — that would manufacture new misses.
  * 放行集 (accept set, SAMPLE of what the baseline guard kept): the gold split
    tells us how many accepts were missed dangers (gold=reject). A good guard
    change CATCHES these (re-decides reject) without disturbing correct accepts.

Run before and after a guard edit; compare 救回/守住/抓住/误伤. Uses the
worktree's guard via PYTHONPATH so it reflects in-progress edits:

    PYTHONPATH=src .venv/bin/python evals/polish_scoreboard.py
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

from app.lexicon_store import list_lexicon_known_texts
from app.models import SentenceSegment
from app.transcript_corrections import _is_change_type_allowed, _polish_guard

LOCAL = Path(__file__).resolve().parent / "local"
REVIEWED = LOCAL / "polish_reviewed.jsonl"
# Local known-term set, so the board's guard whitelists verified ASCII
# restorations exactly as the vocabulary-aware gold does.
GUARD_VOCAB = list_lexicon_known_texts()


def guard_decision(row: dict) -> str:
    """Replay the current guard on one row; return 'keep' or 'reject'."""
    original = row["original_text"]
    proposed = row["proposed_text"]
    sentences = [
        SentenceSegment(0, 1000, row.get("previous_text", ""), None, 0),
        SentenceSegment(1000, 2000, original, None, 1),
        SentenceSegment(2000, 3000, row.get("next_text", ""), None, 2),
    ]
    if not _is_change_type_allowed(row.get("change_type", "")):
        return "reject"
    verdict = _polish_guard(1, sentences, original, proposed, GUARD_VOCAB)
    return "reject" if verdict else "keep"


def gold_of(row: dict, gold_field: str) -> str | None:
    """Resolve the gold verdict to 'keep'/'reject', or None if absent."""
    value = row.get(gold_field) or row.get("codex_verdict")
    return value if value in {"keep", "reject"} else None


def main() -> None:
    """Replay the current guard over the reviewed set and report per-subset metrics."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--gold-field", default="codex_verdict")
    parser.add_argument(
        "--reviewed",
        type=Path,
        default=REVIEWED,
        help="Reviewed JSONL with a gold field. Defaults to polish_reviewed.jsonl.",
    )
    args = parser.parse_args()

    rows = [
        json.loads(line)
        for line in args.reviewed.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    # subset -> (gold, guard) -> count
    cells: dict[str, Counter] = {"reject": Counter(), "accept": Counter()}
    skipped = 0
    for row in rows:
        gold = gold_of(row, args.gold_field)
        if gold is None:
            skipped += 1
            continue
        kind = row.get("_kind", "reject")
        cells[kind][(gold, guard_decision(row))] += 1

    _report_reject(cells["reject"])
    _report_accept(cells["accept"])
    if skipped:
        print(f"\n(跳过无 gold {skipped} 条)")


def _report_reject(cell: Counter) -> None:
    """Report the reject census: over-rejection recovered vs dangerous holds kept."""
    over_total = cell[("keep", "reject")] + cell[("keep", "keep")]   # gold=keep
    recovered = cell[("keep", "keep")]                                # now kept = saved
    danger_total = cell[("reject", "reject")] + cell[("reject", "keep")]  # gold=reject
    held = cell[("reject", "reject")]                                 # still rejected = safe
    print("=" * 60)
    print(f"被拒评测集 (全量普查, {over_total + danger_total} 条)")
    print("=" * 60)
    if over_total:
        print(f"  误杀 (gold=keep)   {over_total:5d}  -> 当前 guard 救回 {recovered} "
              f"({recovered / over_total * 100:.1f}%)  ↑越高越好")
    if danger_total:
        leaked = danger_total - held
        print(f"  正确拦截 (gold=reject) {danger_total:5d}  -> 当前 guard 守住 {held} "
              f"({held / danger_total * 100:.1f}%)  必须保持 100%")
        if leaked:
            print(f"    ⚠ 被错误救回成漏放: {leaked}")


def _report_accept(cell: Counter) -> None:
    """Report the accept sample: correct accepts vs missed dangers caught."""
    correct_total = cell[("keep", "keep")] + cell[("keep", "reject")]   # gold=keep
    kept_ok = cell[("keep", "keep")]
    miss_total = cell[("reject", "keep")] + cell[("reject", "reject")]   # gold=reject
    caught = cell[("reject", "reject")]
    print("\n" + "=" * 60)
    print(f"放行评测集 (抽样, {correct_total + miss_total} 条)")
    print("=" * 60)
    if correct_total:
        hurt = correct_total - kept_ok
        print(f"  正确放行 (gold=keep)   {correct_total:5d}  -> 当前 guard 仍放行 {kept_ok} "
              f"({kept_ok / correct_total * 100:.1f}%)  改 guard 别误伤")
        if hurt:
            print(f"    ⚠ 被新规则误伤: {hurt}")
    if miss_total:
        print(f"  漏放 (gold=reject)     {miss_total:5d}  -> 当前 guard 抓住 {caught} "
              f"({caught / miss_total * 100:.1f}%)  ↑越高越好")


if __name__ == "__main__":
    main()
