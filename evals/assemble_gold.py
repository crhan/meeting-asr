"""Assemble the final Polish gold standard from Linus's adjudication.

Combines three independent signals into one ``gold_verdict`` per reviewed row,
following Linus's direction rulings (all option (a), the conservative/faithful
column):

  方向1 split 边界            -> reject (majority-of-3 default)
  方向2 protected 删除         -> 相邻重复 keep / 分布式删 reject (already in panel)
  方向3 英文去空格/断词        -> keep   (override: despace-equal ascii)
  方向4 纯删除删掉实义         -> reject (already auto=reject)
  方向5 ascii 术语/人名还原     -> reject (override: ban term restoration)

For the contested/high-risk 508 the verdict is the reconciled panel result with
the two overrides; for the rest of the 2096 we trust codex (it agrees with the
panel on the undisputed keeps, and the 200 structural rejects sampled 80-100%).

Writes ``polish_reviewed_gold.jsonl`` (adds ``gold_verdict``) and prints the
gold distribution. Run:

    PYTHONPATH=src .venv/bin/python evals/assemble_gold.py
"""

from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from build_review_file import _collect  # noqa: E402  (local tool import)

LOCAL = Path(__file__).resolve().parent / "local"
REVIEWED = LOCAL / "polish_reviewed.jsonl"
GOLD_OUT = LOCAL / "polish_reviewed_gold.jsonl"


def _reconcile(case: dict) -> str:
    """Final verdict for one contested/high-risk case under Linus's rulings."""
    # 方向3: english spacing/segmentation only (chars identical) -> keep.
    if case["despace_eq"]:
        return "keep"
    # 方向5: ascii term/name restoration (garbled sound -> plausible term) -> reject,
    # even where both blind lenses kept it. Faithful over readable.
    if case["category"] == "ascii_hallucination" and case["tier"] != "T3_both_reject":
        return "reject"
    # 方向1/2/4: majority-of-3 (both lenses keep -> keep; else reject).
    return case["auto"]


def main() -> None:
    """Assemble gold_verdict for every reviewed row and write the gold file."""
    cases = _collect()
    # (original, proposed) -> reconciled verdict for the 508 contested/high-risk
    contested: dict[tuple[str, str], str] = {}
    flips = Counter()
    for c in cases:
        verdict = _reconcile(c)
        contested[(c["orig"], c["prop"])] = verdict
        if verdict != c["auto"]:
            flips[f"{c['auto']}->{verdict}"] += 1

    rows = [
        json.loads(line)
        for line in REVIEWED.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    source = Counter()
    dist = Counter()
    for row in rows:
        key = (row["original_text"], row["proposed_text"])
        if key in contested:
            gold = contested[key]
            source["contested(508 重裁)"] += 1
        elif row.get("codex_verdict") == "keep":
            gold = "keep"  # 761 我+codex 一致 keep + 627 codex 单方 keep
            source["codex keep(直接采纳)"] += 1
        else:
            gold = "reject"  # 200 结构类 codex 单方 reject（抽样 80-100% 可信）
            source["codex reject 结构类(信 codex)"] += 1
        row["gold_verdict"] = gold
        dist[(row.get("_kind", "reject"), gold)] += 1

    with GOLD_OUT.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")

    print(f"写出 {GOLD_OUT}  ({len(rows)} 行)")
    print("\n[gold 来源]")
    for k, v in source.most_common():
        print(f"  {k:28s}{v:5d}")
    print("\n[相对三方多数票的覆盖翻转]")
    for k, v in flips.most_common():
        print(f"  {k:16s}{v:5d}")
    print("\n[最终 gold 分布 (按子集)]")
    for kind in ("reject", "accept"):
        keep = dist[(kind, "keep")]
        rej = dist[(kind, "reject")]
        name = "被拒集" if kind == "reject" else "放行集"
        print(f"  {name}: gold=keep {keep} / gold=reject {rej} (共 {keep + rej})")


if __name__ == "__main__":
    main()
