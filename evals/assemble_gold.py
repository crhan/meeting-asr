"""Assemble the final Polish gold standard from Linus's adjudication.

Combines three independent signals into one ``gold_verdict`` per reviewed row,
following Linus's direction rulings:

  方向1 split 边界            -> reject (majority-of-3 default)
  方向2 protected 删除         -> 相邻重复 keep / 分布式删 reject (already in panel)
  方向3 英文去空格/断词        -> keep   (override: despace-equal ascii)
  方向4 纯删除删掉实义         -> reject (already auto=reject)
  方向5 ascii 术语/人名还原     -> VOCABULARY-aware: restored to a term that is in
        the lexicon (底码->Dima, Dima is a hotword) is a legit homophone fix -> keep;
        restored to an unknown/unverifiable token (武一->WuYi) -> reject (fabrication).

方向5 used to be a blanket "ban restoration" stopgap from the no-vocabulary era.
Now that the lexicon carries the authoritative person/system names, a restoration
to a known term is correct, so the gold trusts the vocabulary as ground truth.
The ascii rule is applied GLOBALLY to every ascii reject row (not only the 508),
which also fixes the earlier bug where codex-kept ascii rows skipped the override.

For non-ascii contested/high-risk the verdict is the reconciled panel (majority-
of-3); for the rest of the 2096 we trust codex.

Writes ``polish_reviewed_gold.jsonl`` (adds ``gold_verdict``). Run:

    PYTHONPATH=src .venv/bin/python evals/assemble_gold.py
"""

from __future__ import annotations

import json
import re
import sqlite3
from collections import Counter
from pathlib import Path

from app.lexicon_store import default_lexicon_db_path
from app.transcript_corrections import _is_destutter_only

from evals.build_review_file import _collect

LOCAL = Path(__file__).resolve().parent / "local"
REVIEWED = LOCAL / "polish_reviewed.jsonl"
GOLD_OUT = LOCAL / "polish_reviewed_gold.jsonl"
HUMAN_OVERRIDES = LOCAL / "protected_gold_overrides.jsonl"
_ASCII_RE = re.compile(r"[A-Za-z0-9]+")


def _load_vocab() -> set[str]:
    """Lowercased set of every known lexicon term (canonical + alias)."""
    db = default_lexicon_db_path()
    if not db.exists():
        return set()
    with sqlite3.connect(str(db)) as con:
        vocab = {r[0].lower() for r in con.execute("SELECT canonical FROM terms")}
        vocab |= {r[0].lower() for r in con.execute("SELECT alias FROM aliases")}
    return vocab


def _is_despace(original: str, proposed: str) -> bool:
    """True if ascii content is identical and only spacing/segmentation changed."""
    o, p = _ASCII_RE.findall(original), _ASCII_RE.findall(proposed)
    return "".join(o).lower() == "".join(p).lower() != "" and o != p


def _ascii_introduced(original: str, proposed: str) -> list[str]:
    """Ascii tokens present in proposed but absent from original."""
    orig = set(_ASCII_RE.findall(original))
    return [t for t in _ASCII_RE.findall(proposed) if t not in orig]


def _ascii_gold(original: str, proposed: str, vocab: set[str]) -> str:
    """Vocabulary-aware verdict for one ascii_hallucination row."""
    if _is_despace(original, proposed):
        return "keep"  # 方向3: pure re-segmentation, chars unchanged
    introduced = _ascii_introduced(original, proposed)
    # 方向5 (vocab-aware): every newly introduced ascii token must be a known
    # lexicon term, otherwise the restoration is unverifiable -> fabrication.
    if introduced and all(token.lower() in vocab for token in introduced):
        return "keep"
    return "reject"


# Each gold source maps to whether its verdict is INDEPENDENT of the guard's own
# logic. The circular sources reuse the very functions / lexicon the guard uses
# (_is_destutter_only, the ascii re-segmentation exemption, the vocab whitelist),
# so on those rows guard == gold is near-tautological — it measures self-consistency,
# not correctness. The scoreboard must report independent and circular agreement
# separately so the headline "救回" number is not inflated by definitional matches.
GOLD_SOURCE_INDEPENDENT: dict[str, bool] = {
    "audio_human": True,   # 人工听原音频裁定 — 终极真值
    "panel": True,         # 508 争议盲面板多数票
    "codex_keep": True,    # codex 直接采纳 keep
    "codex_reject": True,  # codex 结构类 reject(抽样 80-100% 可信)
    "destutter": False,    # 去口吃 — guard 的 _is_destutter_only 顶层早放行,循环
    "despace": False,      # 去空格 — guard ascii 检查的去空格豁免,循环
    "ascii_vocab": False,  # ascii 词表重判 — guard 的 vocab 白名单 + despace,循环
}

GOLD_SOURCE_LABEL: dict[str, str] = {
    "audio_human": "人工音频裁定",
    "panel": "contested(508 面板)",
    "codex_keep": "codex keep(直接采纳)",
    "codex_reject": "codex reject 结构类(信 codex)",
    "destutter": "去口吃(destutter)",
    "despace": "去空格(方向3)",
    "ascii_vocab": "ascii 词表重判",
}


def _classify_gold(
    row: dict,
    key: tuple[str, str],
    *,
    overrides: dict[tuple[str, str], str],
    panel: dict[tuple[str, str], str],
    vocab: set[str],
) -> tuple[str, str]:
    """Resolve (gold_verdict, gold_source) for one row via the precedence chain.

    The order is fixed: human audio > destutter > despace > ascii-vocab >
    508-panel > codex. The returned source key feeds GOLD_SOURCE_INDEPENDENT so
    downstream scoring can separate genuinely-independent gold from gold that
    merely re-runs the guard's own deterministic rules.
    """
    original, proposed = key
    if key in overrides:
        return overrides[key], "audio_human"
    if _is_destutter_only(original, proposed):
        return "keep", "destutter"
    if _is_despace(original, proposed):
        return "keep", "despace"
    if row.get("_kind") == "reject" and row.get("category") == "ascii_hallucination":
        return _ascii_gold(original, proposed, vocab), "ascii_vocab"
    if key in panel:
        return panel[key], "panel"
    if row.get("codex_verdict") == "keep":
        return "keep", "codex_keep"
    return "reject", "codex_reject"


def main() -> None:
    """Assemble gold_verdict for every reviewed row and write the gold file."""
    vocab = _load_vocab()
    # (original, proposed) -> panel verdict (majority-of-3) for the 508 contested.
    panel: dict[tuple[str, str], str] = {
        (c["orig"], c["prop"]): c["auto"] for c in _collect()
    }
    # Highest-priority human rulings: rows the user adjudicated from the source
    # AUDIO (evals/audio_verify.py). These override codex/panel — the recording is
    # ground truth. Currently: 14 protected-deletion rejects the user confirmed are
    # stutter/substring artifacts (codex over-rejected), all -> keep.
    overrides: dict[tuple[str, str], str] = {}
    if HUMAN_OVERRIDES.exists():
        for line in HUMAN_OVERRIDES.read_text(encoding="utf-8").splitlines():
            if line.strip():
                o = json.loads(line)
                overrides[(o["original_text"], o["proposed_text"])] = o["gold_verdict"]

    rows = [
        json.loads(line)
        for line in REVIEWED.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    source = Counter()
    dist = Counter()
    for row in rows:
        key = (row["original_text"], row["proposed_text"])
        gold, gold_source = _classify_gold(
            row, key, overrides=overrides, panel=panel, vocab=vocab
        )
        independent = GOLD_SOURCE_INDEPENDENT[gold_source]
        row["gold_verdict"] = gold
        row["gold_source"] = gold_source
        row["gold_independent"] = independent
        source[gold_source] += 1
        dist[(row.get("_kind", "reject"), gold)] += 1

    with GOLD_OUT.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")

    print(f"写出 {GOLD_OUT}  ({len(rows)} 行)")
    print("\n[gold 来源] (independent = 与 guard 逻辑无关的真实信号)")
    indep_total = sum(v for k, v in source.items() if GOLD_SOURCE_INDEPENDENT[k])
    circ_total = sum(v for k, v in source.items() if not GOLD_SOURCE_INDEPENDENT[k])
    for group, want in (("独立金标", True), ("循环金标", False)):
        members = [(k, v) for k, v in source.most_common() if GOLD_SOURCE_INDEPENDENT[k] == want]
        subtotal = indep_total if want else circ_total
        print(f"  -- {group} 小计 {subtotal} --")
        for k, v in members:
            print(f"     {GOLD_SOURCE_LABEL[k]:28s}{v:5d}  [{k}]")
    pct = circ_total / max(1, len(rows)) * 100
    print(f"  => 独立 {indep_total} / 循环 {circ_total} ({pct:.1f}% 的 gold 由 guard 自身规则判定)")
    print(f"\n词表规模(canonical+alias): {len(vocab)}")
    print("\n[最终 gold 分布 (按子集)]")
    for kind in ("reject", "accept"):
        keep = dist[(kind, "keep")]
        rej = dist[(kind, "reject")]
        name = "被拒集" if kind == "reject" else "放行集"
        print(f"  {name}: gold=keep {keep} / gold=reject {rej} (共 {keep + rej})")


if __name__ == "__main__":
    main()
