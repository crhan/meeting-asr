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

When scoring the authoritative ``gold_verdict`` field, each metric is further
split into an INDEPENDENT column (gold decided by audio / panel / codex) and a
CIRCULAR column (gold decided by the guard's own destutter / despace / ascii
rules, where guard == gold is near-tautological). The split is meaningful only
for ``gold_verdict``; scoring any other field gets the plain single-column report.

Run before and after a guard edit; compare 救回/守住/抓住/误伤:

    uv run python -m evals.polish_scoreboard \\
        --reviewed evals/local/polish_reviewed_gold.jsonl --gold-field gold_verdict
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

from app.lexicon_store import list_lexicon_known_texts
from app.models import SentenceSegment
from app.residual_noise import residual_noise
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


def run(rows: list[dict], gold_field: str) -> None:
    """Score ``rows`` on ``gold_field`` and print the two-subset report.

    The independent/circular split describes how ``gold_verdict`` was decided, so
    it is emitted ONLY when ``gold_field`` is ``gold_verdict``. Scoring any other
    field (the weaker ``codex_verdict`` default, or an untagged reviewed file)
    gets the original single-column report — reusing ``gold_verdict``'s
    independence tags to split a different label's counts would mislabel the
    columns (Codex review P2). When the split is off, ``gold_independent`` is
    ignored entirely.
    """
    split = gold_field == "gold_verdict"
    if not split and any("gold_verdict" in row for row in rows):
        print(
            f"⚠ 数据里有 gold_verdict,但你用了 --gold-field {gold_field}。"
            "权威字段是 gold_verdict;当前为单栏(不按独立/循环拆分),"
            "且含更弱/更循环的信号。\n"
        )

    # subset -> (gold, guard, independent|None) -> count. ``independent`` is the
    # bool tag from assemble_gold only when scoring gold_verdict, else None.
    cells: dict[str, Counter] = {"reject": Counter(), "accept": Counter()}
    skipped = 0
    untagged = 0
    for row in rows:
        gold = gold_of(row, gold_field)
        if gold is None:
            skipped += 1
            continue
        independent: bool | None = None
        if split:
            tag = row.get("gold_independent")
            if tag is None:
                untagged += 1
                tag = True  # missing tag -> count as real signal, never fake-circular
            independent = bool(tag)
        kind = row.get("_kind", "reject")
        cells[kind][(gold, guard_decision(row), independent)] += 1

    _report_reject(cells["reject"], split=split)
    _report_accept(cells["accept"], split=split)
    _report_residual(rows, gold_field)
    if skipped:
        print(f"\n(跳过无 gold {skipped} 条)")
    if untagged:
        print(
            f"(gold_verdict 评分但 {untagged} 行无 gold_source 标记,按独立计入;"
            "重跑 `python -m evals.assemble_gold` 补全)"
        )


def _report_residual(rows: list[dict], gold_field: str) -> None:
    """Report residual noise left IN the polish outputs — the inverse dimension.

    Orthogonal to the keep/reject guard axis: a polish can be correctly accepted
    yet still leave a filler / uncollapsed repeat in. Detector lives in
    ``app.residual_noise`` (verified ~100% precision). For the per-row report run
    ``python -m evals.scan_residual_noise``.
    """
    changed = [
        row
        for row in rows
        if row.get("proposed_text") and row["proposed_text"] != row.get("original_text")
    ]
    if not changed:
        return
    dirty = [(row, residual_noise(row["proposed_text"])) for row in changed]
    dirty = [(row, reasons) for row, reasons in dirty if reasons]
    kept_dirty = sum(
        1 for row, _ in dirty if (row.get(gold_field) or row.get("gold_verdict")) == "keep"
    )
    print("\n" + "=" * 64)
    print(f"残留脏字维度 (polish 漏删,扫 proposed 输出, {len(changed)} 改写行)")
    print("=" * 64)
    print(f"  仍含脏字   {len(dirty):5d}  ({len(dirty) / len(changed) * 100:.1f}%)  ↓越低越好")
    print(f"  其中 gold=keep {kept_dirty:5d}  ← 评测放行了仍脏的 polish")
    by_reason: Counter = Counter()
    for _, reasons in dirty:
        for code in reasons:
            by_reason[code.split(":")[0]] += 1
    if by_reason:
        breakdown = "  ".join(f"{k} {v}" for k, v in by_reason.most_common())
        print(f"  类型: {breakdown}")


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
    run(rows, args.gold_field)


def _total(cell: Counter, gold: str, guard: str) -> int:
    """Total count for one (gold, guard), summed over every independence tag."""
    return sum(v for (g, gd, _ind), v in cell.items() if g == gold and gd == guard)


def _split(cell: Counter, gold: str, guard: str) -> tuple[int, int]:
    """Return (independent, circular) counts for one (gold, guard) outcome."""
    return cell[(gold, guard, True)], cell[(gold, guard, False)]


def _pct(num: int, den: int) -> str:
    """Format num/den (pct), or n/a when the denominator is empty."""
    return f"{num}/{den} ({num / den * 100:.1f}%)" if den else f"{num}/0 (n/a)"


def _split_line(label: str, num_i: int, num_c: int, den_i: int, den_c: int) -> None:
    """Print one metric split into independent (real) vs circular (self-confirming)."""
    print(f"       └─ 独立金标 {label} {_pct(num_i, den_i):>18}   ← 真实信号")
    print(f"          循环金标 {label} {_pct(num_c, den_c):>18}   ← destutter/despace/ascii 自我确认")


def _report_reject(cell: Counter, *, split: bool) -> None:
    """Report the reject census: over-rejection recovered vs dangerous holds kept."""
    over_total = _total(cell, "keep", "reject") + _total(cell, "keep", "keep")
    recovered = _total(cell, "keep", "keep")
    danger_total = _total(cell, "reject", "reject") + _total(cell, "reject", "keep")
    held = _total(cell, "reject", "reject")
    print("=" * 64)
    print(f"被拒评测集 (全量普查, {over_total + danger_total} 条)")
    print("=" * 64)
    if over_total:
        print(f"  误杀 (gold=keep)   {over_total:5d}  -> guard 救回 {_pct(recovered, over_total)}  ↑越高越好")
        if split:
            rec_i, rec_c = _split(cell, "keep", "keep")
            okr_i, okr_c = _split(cell, "keep", "reject")
            _split_line("救回", rec_i, rec_c, rec_i + okr_i, rec_c + okr_c)
    if danger_total:
        leaked = danger_total - held
        gate = "独立部分必须 100%" if split else "必须保持 100%"
        print(f"  正确拦截 (gold=reject) {danger_total:5d}  -> guard 守住 {_pct(held, danger_total)}  {gate}")
        if split:
            held_i, held_c = _split(cell, "reject", "reject")
            leak_i, leak_c = _split(cell, "reject", "keep")
            _split_line("守住", held_i, held_c, held_i + leak_i, held_c + leak_c)
            if leaked:
                print(f"    ⚠ 被错误救回成漏放: {leaked} (独立 {leak_i} / 循环 {leak_c})")
        elif leaked:
            print(f"    ⚠ 被错误救回成漏放: {leaked}")


def _report_accept(cell: Counter, *, split: bool) -> None:
    """Report the accept sample: correct accepts vs missed dangers caught."""
    correct_total = _total(cell, "keep", "keep") + _total(cell, "keep", "reject")
    kept_ok = _total(cell, "keep", "keep")
    miss_total = _total(cell, "reject", "keep") + _total(cell, "reject", "reject")
    caught = _total(cell, "reject", "reject")
    print("\n" + "=" * 64)
    print(f"放行评测集 (抽样, {correct_total + miss_total} 条)")
    print("=" * 64)
    if correct_total:
        hurt = correct_total - kept_ok
        print(f"  正确放行 (gold=keep)   {correct_total:5d}  -> guard 仍放行 {_pct(kept_ok, correct_total)}  改 guard 别误伤")
        if split:
            keep_i, keep_c = _split(cell, "keep", "keep")
            hurt_i, hurt_c = _split(cell, "keep", "reject")
            _split_line("仍放行", keep_i, keep_c, keep_i + hurt_i, keep_c + hurt_c)
            if hurt:
                print(f"    ⚠ 被新规则误伤: {hurt} (独立 {hurt_i} / 循环 {hurt_c})")
        elif hurt:
            print(f"    ⚠ 被新规则误伤: {hurt}")
    if miss_total:
        print(f"  漏放 (gold=reject)     {miss_total:5d}  -> guard 抓住 {_pct(caught, miss_total)}  ↑越高越好")
        if split:
            miss_i, miss_c = _split(cell, "reject", "keep")
            caught_i, caught_c = _split(cell, "reject", "reject")
            _split_line("抓住", caught_i, caught_c, miss_i + caught_i, miss_c + caught_c)


if __name__ == "__main__":
    main()
