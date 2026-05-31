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
    """Resolve the requested gold field to 'keep'/'reject', or None if absent.

    No silent fallback to codex_verdict: if you ask for gold_verdict, you get
    gold_verdict or nothing. The old ``or row.get("codex_verdict")`` quietly
    downgraded rows whose authoritative verdict was missing, mixing two signals
    in one census. The caller already warns when the weaker field is selected.
    """
    value = row.get(gold_field)
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
    if args.gold_field != "gold_verdict" and any("gold_verdict" in r for r in rows):
        print(
            f"⚠ 数据里有 gold_verdict,但你用了 --gold-field {args.gold_field}。"
            "权威字段是 gold_verdict;当前结果含更弱/更循环的信号。\n"
        )
    # subset -> (gold, guard, independent) -> count. The independent flag (from
    # assemble_gold's gold_source) lets us separate genuinely-independent agreement
    # from rows whose gold was computed by the guard's own rules (destutter / despace
    # / ascii-vocab), where guard == gold is near-tautological.
    cells: dict[str, Counter] = {"reject": Counter(), "accept": Counter()}
    skipped = 0
    missing_source = 0
    for row in rows:
        gold = gold_of(row, args.gold_field)
        if gold is None:
            skipped += 1
            continue
        independent = row.get("gold_independent")
        if independent is None:
            missing_source += 1
            independent = True  # untagged gold -> count as independent, never hide it
        kind = row.get("_kind", "reject")
        cells[kind][(gold, guard_decision(row), bool(independent))] += 1

    _report_reject(cells["reject"])
    _report_accept(cells["accept"])
    if skipped:
        print(f"\n(跳过无 gold {skipped} 条)")
    if missing_source:
        print(
            f"(无 gold_source 标记 {missing_source} 条,按独立计入;"
            "重跑 `python -m evals.assemble_gold` 可补全独立/循环拆分)"
        )


def _pct(num: int, den: int) -> str:
    """Format num/den (pct), or n/a when the denominator is empty."""
    return f"{num}/{den} ({num / den * 100:.1f}%)" if den else f"{num}/0 (n/a)"


def _split(cell: Counter, gold: str, guard: str) -> tuple[int, int]:
    """Return (independent, circular) counts for one (gold, guard) outcome."""
    return cell[(gold, guard, True)], cell[(gold, guard, False)]


def _split_line(label: str, num_i: int, num_c: int, den_i: int, den_c: int) -> None:
    """Print one metric split into independent (real) vs circular (self-confirming)."""
    print(f"       └─ 独立金标 {label} {_pct(num_i, den_i):>18}   ← 真实信号")
    print(f"          循环金标 {label} {_pct(num_c, den_c):>18}   ← destutter/despace/ascii 自我确认")


def _report_reject(cell: Counter) -> None:
    """Report the reject census: over-rejection recovered vs dangerous holds kept."""
    rec_i, rec_c = _split(cell, "keep", "keep")          # gold=keep, now kept = saved
    okr_i, okr_c = _split(cell, "keep", "reject")        # gold=keep, still rejected
    over_i, over_c = rec_i + okr_i, rec_c + okr_c
    over_total, recovered = over_i + over_c, rec_i + rec_c
    held_i, held_c = _split(cell, "reject", "reject")    # gold=reject, still rejected = safe
    leak_i, leak_c = _split(cell, "reject", "keep")      # gold=reject, wrongly recovered
    danger_i, danger_c = held_i + leak_i, held_c + leak_c
    danger_total, held = danger_i + danger_c, held_i + held_c
    print("=" * 64)
    print(f"被拒评测集 (全量普查, {over_total + danger_total} 条)")
    print("=" * 64)
    if over_total:
        print(f"  误杀 (gold=keep)   {over_total:5d}  -> guard 救回 {_pct(recovered, over_total)}  ↑越高越好")
        _split_line("救回", rec_i, rec_c, over_i, over_c)
    if danger_total:
        leaked = danger_total - held
        print(f"  正确拦截 (gold=reject) {danger_total:5d}  -> guard 守住 {_pct(held, danger_total)}  独立部分必须 100%")
        _split_line("守住", held_i, held_c, danger_i, danger_c)
        if leaked:
            print(f"    ⚠ 被错误救回成漏放: {leaked} (独立 {leak_i} / 循环 {leak_c})")


def _report_accept(cell: Counter) -> None:
    """Report the accept sample: correct accepts vs missed dangers caught."""
    keep_i, keep_c = _split(cell, "keep", "keep")        # gold=keep, still kept = correct
    hurt_i, hurt_c = _split(cell, "keep", "reject")      # gold=keep, wrongly rejected
    correct_i, correct_c = keep_i + hurt_i, keep_c + hurt_c
    correct_total, kept_ok = correct_i + correct_c, keep_i + keep_c
    miss_i, miss_c = _split(cell, "reject", "keep")      # gold=reject, still let through
    caught_i, caught_c = _split(cell, "reject", "reject")  # gold=reject, now caught
    mt_i, mt_c = miss_i + caught_i, miss_c + caught_c
    miss_total, caught = mt_i + mt_c, caught_i + caught_c
    print("\n" + "=" * 64)
    print(f"放行评测集 (抽样, {correct_total + miss_total} 条)")
    print("=" * 64)
    if correct_total:
        hurt = correct_total - kept_ok
        print(f"  正确放行 (gold=keep)   {correct_total:5d}  -> guard 仍放行 {_pct(kept_ok, correct_total)}  改 guard 别误伤")
        _split_line("仍放行", keep_i, keep_c, correct_i, correct_c)
        if hurt:
            print(f"    ⚠ 被新规则误伤: {hurt} (独立 {hurt_i} / 循环 {hurt_c})")
    if miss_total:
        print(f"  漏放 (gold=reject)     {miss_total:5d}  -> guard 抓住 {_pct(caught, miss_total)}  ↑越高越好")
        _split_line("抓住", caught_i, caught_c, mt_i, mt_c)


if __name__ == "__main__":
    main()
