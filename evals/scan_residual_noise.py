"""Scan polish OUTPUTS for residual ASR noise the polish failed to remove.

The guard and the whole gold effort target one failure mode — polish deleting
too much (real content). The opposite failure is invisible to all of it: polish
that left noise IN — a stray filler (嗯/呃/额), an uncollapsed stutter (在在,
对对对), a duplicated phrase. A row like "都是一两个 case 在在命中" is a "keep"
in the gold (no content lost) yet the output is still dirty. So a gold=keep that
still carries residual noise means the review accepted a half-cleaned polish —
exactly "没有认真看 polish 的结果里是否还有脏字".

This scans every proposed_text (the polish output) in the reviewed set and flags
residual noise, with detectors tuned for precision (a report to act on, not an
auto-fixer). The headline is flagged ∩ gold=keep: kept polishes that are still
dirty. Flagged rows are written to evals/local/residual_noise_flagged.jsonl.

Reads evals/local/. Run:
    uv run python -m evals.scan_residual_noise
    uv run python -m evals.scan_residual_noise --reviewed evals/local/polish_reviewed_gold.jsonl --show 30
"""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from pathlib import Path

from evals._log import log

LOCAL = Path(__file__).resolve().parent / "local"
GOLD = LOCAL / "polish_reviewed_gold.jsonl"
OUT = LOCAL / "residual_noise_flagged.jsonl"

# Pure fillers that a clean polish should never leave behind. 嗯/呃/额 are
# unambiguous noise; 啊/哦/呀/吧/呢/嘛 double as legitimate sentence particles, so
# they are deliberately EXCLUDED to keep the scan high-precision.
_FILLER_RE = re.compile(r"[嗯呃额]")

# Three or more identical CJK chars in a row — almost always uncollapsed stutter
# (对对对, 是是是). Two-in-a-row is handled separately with a legit-word stoplist.
_TRIPLE_RE = re.compile(r"([一-鿿])\1{2,}")

# An adjacent repeat of a 2-4 char CJK chunk (就是就是, 可以可以, 这个这个) left in
# the OUTPUT. Single-char doubles use the allowlist below instead.
_CHUNK_DUP_RE = re.compile(r"([一-鿿]{2,4})\1")

# Single-char doubling is flagged by an ALLOWLIST of stutter-prone function words
# and pronouns, NOT a stoplist of legit reduplications. Chinese reduplication is
# open-ended and common (看看/常常/刚刚/慢慢/天天/个个...), so a stoplist over-fires
# (~91% of outputs). But these particular chars — pronouns, particles, copulas,
# common verbs/adverbs that do NOT legitimately reduplicate — are almost always
# uncollapsed stutter when doubled (在在, 的的, 我我, 是是, 就就, 不不). Closed set
# => high precision.
_STUTTER_DOUBLE = set("在的了我你他她它是就都也还把被这那但而没不要会能和与之其该")
_SINGLE_DUP_RE = re.compile(r"([一-鿿])\1")


def residual_noise(text: str) -> list[str]:
    """Return reason codes for residual noise in one polish output (empty if clean)."""
    reasons: list[str] = []
    fillers = _FILLER_RE.findall(text)
    if fillers:
        reasons.append(f"filler:{''.join(sorted(set(fillers)))}")
    triples = _TRIPLE_RE.findall(text)
    if triples:
        reasons.append(f"triple:{','.join(sorted(set(triples)))}")
    chunks = {m.group(1) for m in _CHUNK_DUP_RE.finditer(text)}
    if chunks:
        reasons.append(f"chunk_dup:{','.join(sorted(chunks))}")
    singles = {
        m.group(1)
        for m in _SINGLE_DUP_RE.finditer(text)
        if m.group(1) in _STUTTER_DOUBLE
    }
    if singles:
        reasons.append(f"char_dup:{','.join(sorted(singles))}")
    return reasons


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
        rec = {
            "source": row.get("source", ""),
            "gold_verdict": gold,
            "gold_source": row.get("gold_source"),
            "reasons": reasons,
            "original_text": row.get("original_text", ""),
            "proposed_text": proposed,
        }
        flagged.append(rec)
        for r in reasons:
            by_reason[r.split(":")[0]] += 1
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
            "filler": "残留语气词 嗯/呃/额",
            "triple": "3+ 连字未折叠 (对对对)",
            "chunk_dup": "词块重复未折叠 (就是就是)",
            "char_dup": "虚词/代词叠字未折叠 (在在/的的/我我)",
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
