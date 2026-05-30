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
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from build_review_file import _collect  # noqa: E402  (local tool import)
from app.lexicon_store import default_lexicon_db_path  # noqa: E402

LOCAL = Path(__file__).resolve().parent / "local"
REVIEWED = LOCAL / "polish_reviewed.jsonl"
GOLD_OUT = LOCAL / "polish_reviewed_gold.jsonl"
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


def main() -> None:
    """Assemble gold_verdict for every reviewed row and write the gold file."""
    vocab = _load_vocab()
    # (original, proposed) -> panel verdict (majority-of-3) for the 508 contested.
    panel: dict[tuple[str, str], str] = {
        (c["orig"], c["prop"]): c["auto"] for c in _collect()
    }

    rows = [
        json.loads(line)
        for line in REVIEWED.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    source = Counter()
    dist = Counter()
    for row in rows:
        original, proposed = row["original_text"], row["proposed_text"]
        key = (original, proposed)
        if _is_despace(original, proposed):
            gold = "keep"
            source["去空格(方向3)"] += 1
        elif row.get("_kind") == "reject" and row.get("category") == "ascii_hallucination":
            gold = _ascii_gold(original, proposed, vocab)
            source[f"ascii 词表重判->{gold}"] += 1
        elif key in panel:
            gold = panel[key]  # non-ascii contested -> majority-of-3
            source["contested(508 面板)"] += 1
        elif row.get("codex_verdict") == "keep":
            gold = "keep"
            source["codex keep(直接采纳)"] += 1
        else:
            gold = "reject"  # structural codex reject (sampled 80-100% 可信)
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
    print(f"\n词表规模(canonical+alias): {len(vocab)}")
    print("\n[最终 gold 分布 (按子集)]")
    for kind in ("reject", "accept"):
        keep = dist[(kind, "keep")]
        rej = dist[(kind, "reject")]
        name = "被拒集" if kind == "reject" else "放行集"
        print(f"  {name}: gold=keep {keep} / gold=reject {rej} (共 {keep + rej})")


if __name__ == "__main__":
    main()
