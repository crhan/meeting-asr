"""Field-report harness: scan the reviewed gold for residual noise in polish output.

The guard and the whole gold effort target one failure mode — polish deleting
too much (real content). The opposite failure is invisible to all of it: polish
that left noise IN — a stray filler (嗯/呃), an uncollapsed phrase repeat (就是
就是), a multi-char run (对对对). A row whose output still carries such noise is a
gold=keep (no content lost) yet still dirty, so a gold=keep that is still dirty
means the review accepted a half-cleaned polish — exactly "没有认真看 polish 的结果
里是否还有脏字".

The detector itself lives in ``app.residual_noise`` (verified ~100% precision; the
eval-polish CI gate and the scoreboard share that one implementation). This module
is just the report around it: it scans every rewritten ``proposed_text`` and writes
the flagged rows to evals/local/residual_noise_flagged.jsonl for review.

Reads evals/local/. Run:
    uv run python -m evals.scan_residual_noise
    uv run python -m evals.scan_residual_noise --reviewed evals/local/polish_reviewed_gold.jsonl --show 30
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

from app.residual_noise import residual_noise

from evals._log import log

LOCAL = Path(__file__).resolve().parent / "local"
GOLD = LOCAL / "polish_reviewed_gold.jsonl"
OUT = LOCAL / "residual_noise_flagged.jsonl"


def main() -> None:
    """Scan polish outputs for residual noise and report kept-but-dirty rows."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--reviewed", type=Path, default=GOLD)
    parser.add_argument("--show", type=int, default=25, help="How many examples to print.")
    args = parser.parse_args()

    rows = [
        json.loads(line)
        for line in args.reviewed.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    flagged: list[dict] = []
    by_reason: Counter = Counter()
    kept_dirty = 0
    changed_rows = 0
    for row in rows:
        proposed = row.get("proposed_text", "")
        if not proposed or proposed == row.get("original_text"):
            continue  # only judge rows the polish actually rewrote
        changed_rows += 1
        reasons = residual_noise(proposed)
        if not reasons:
            continue
        gold = row.get("gold_verdict") or row.get("codex_verdict")
        flagged.append(
            {
                "source": row.get("source", ""),
                "gold_verdict": gold,
                "gold_source": row.get("gold_source"),
                "reasons": reasons,
                "original_text": row.get("original_text", ""),
                "proposed_text": proposed,
            }
        )
        for reason in reasons:
            by_reason[reason.split(":")[0]] += 1
        if gold == "keep":
            kept_dirty += 1

    OUT.write_text(
        "\n".join(json.dumps(r, ensure_ascii=False) for r in flagged) + "\n",
        encoding="utf-8",
    )

    print("=" * 66)
    print("Polish 残留脏字扫描 (扫的是 polish 输出 proposed_text)")
    print("=" * 66)
    print(f"  改写过的行         {changed_rows:5d}")
    print(f"  仍含脏字           {len(flagged):5d}  ({len(flagged) / max(1, changed_rows) * 100:.1f}%)")
    print(f"  其中 gold=keep     {kept_dirty:5d}  ← 评测放行了仍脏的 polish(本次重点)")
    print("\n  [按脏字类型]")
    for reason, n in by_reason.most_common():
        label = {
            "filler": "残留语气词 嗯/呃",
            "triple": "3+ 连字未折叠 (对对对)",
            "chunk_dup": "词块重复未折叠 (就是就是)",
        }.get(reason, reason)
        print(f"     {reason:10s}{n:5d}  {label}")

    dirty_keep = [r for r in flagged if r["gold_verdict"] == "keep"]
    print(f"\n  [gold=keep 但仍脏的样例 (前 {min(args.show, len(dirty_keep))})]")
    for r in dirty_keep[: args.show]:
        print(f"     [{';'.join(r['reasons'])}]")
        print(f"       {r['proposed_text'][:70]}")
    print(f"\n写出 {OUT}  ({len(flagged)} 行)")
    log.info(
        "scan_done",
        changed=changed_rows,
        flagged=len(flagged),
        kept_dirty=kept_dirty,
        by_reason=dict(by_reason),
    )


if __name__ == "__main__":
    main()
