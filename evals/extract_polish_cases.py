"""Extract a Polish evaluation set from real on-machine sidecars.

Reads every ``polish_strict_meta_*.json`` under the local projects dir, replays
each (original, proposed, change_type) triple through the CURRENT production
guard, and emits two JSONL datasets under ``evals/local/`` (git-ignored):

  * ``polish_reject_cases.jsonl`` — EVERY triple the current guard rejects.
    These drive the false-reject (over-rejection) diagnosis.
  * ``polish_accept_sample.jsonl`` — a stratified sample of kept triples, with
    every suspected over-acceptance (negation flip / introduced number) kept in
    full. These drive the false-accept (missed-danger) diagnosis.

Each row carries a PROGRAMMATIC initial label (``initial_label``) plus a
confidence. The label is a first pass only — anything uncertain is marked
``needs_review`` so the downstream codex cross-check is the real arbiter, never
this script's single-sided judgement. In particular we deliberately do NOT try
to be clever about Chinese-vs-Arabic numeral equivalence here (that exact trap
produces false positives); introduced-number cases are demoted to needs_review.

Nothing here is committed: the output dir is git-ignored. Run with::

    uv run --no-project python evals/extract_polish_cases.py
"""

from __future__ import annotations

import json
import os
import re
from collections import Counter
from pathlib import Path

from app.models import SentenceSegment
from app.transcript_corrections import (
    _POLISH_PROTECTED_WORDS,
    _is_change_type_allowed,
    _polish_guard,
)

PROJECTS_DIR = (
    Path(os.environ.get("XDG_DATA_HOME", Path.home() / ".local" / "share"))
    / "meeting-asr"
    / "projects"
)
OUT_DIR = Path(__file__).resolve().parent / "local"
REJECT_OUT = OUT_DIR / "polish_reject_cases.jsonl"
ACCEPT_OUT = OUT_DIR / "polish_accept_sample.jsonl"

# Negation markers: inserting one of these can flip meaning (可以 -> 不可以).
_NEGATION_RE = re.compile(r"[不没無无未别勿非]")
# Arabic digit runs, used only as a coarse "introduced a number" signal.
_DIGIT_RE = re.compile(r"\d+(?:\.\d+)?")
# Deterministic 1-in-N sampling stride for ordinary kept cases (the control set).
_ACCEPT_SAMPLE_STRIDE = 40


def is_subsequence(small: str, big: str) -> bool:
    """Return True if ``small`` is ``big`` with only characters deleted (order kept).

    A pure deletion cannot borrow neighbour text, cannot hallucinate ASCII, and
    cannot grow length — so rejecting it under those guards is by construction a
    false reject. This predicate is the load-bearing distinction of the whole set.
    """
    iterator = iter(big)
    return all(char in iterator for char in small)


def inserts_negation(original: str, proposed: str) -> bool:
    """Return True if ``proposed`` contains more negation markers than ``original``.

    More negations on a non-deletion edit is the clearest missed-danger signal:
    the model rewrote attitude (可以 -> 不可以) and the current guard lets it pass.
    """
    return len(_NEGATION_RE.findall(proposed)) > len(_NEGATION_RE.findall(original))


def introduces_arabic_number(original: str, proposed: str) -> bool:
    """Return True if ``proposed`` has an Arabic number run absent from ``original``.

    Coarse on purpose: Chinese<->Arabic numeral equivalence is NOT handled here
    because a naive check false-positives (`'30' in '百分之三十'` is False). Hits
    are demoted to needs_review for codex, not auto-labelled as fabrication.
    """
    return bool(set(_DIGIT_RE.findall(proposed)) - set(_DIGIT_RE.findall(original)))


def deleted_protected_word(verdict: str) -> str:
    """Pull the protected word out of a ``protected_word_deleted:<word>`` verdict.

    The guard returns a single-colon code like ``protected_word_deleted:可以``.
    """
    parts = verdict.split(":", 1)
    return parts[1] if len(parts) >= 2 else ""


def label_reject(original: str, proposed: str, verdict: str) -> tuple[str, str, str]:
    """Initial-label a rejected triple: (label, confidence, reason).

    The question for a reject is "did the guard reject correctly, or over-reject?".
    """
    pure_deletion = is_subsequence(proposed, original)
    reason_code = verdict.split(":", 1)[0]
    if reason_code == "protected_word_deleted":
        word = deleted_protected_word(verdict)
        if pure_deletion and word and proposed.count(word) >= 1:
            return (
                "should_keep",
                "high",
                f"纯删除去重复，保护词「{word}」去重后仍残留，属误杀",
            )
        return (
            "needs_review",
            "low",
            f"保护词「{word}」可能被真删除或改写，交 codex 判定",
        )
    if pure_deletion:
        return (
            "should_keep",
            "high",
            f"纯删除（只删不加），不可能借句/造词/超长，{reason_code} 属误杀",
        )
    return (
        "needs_review",
        "low",
        f"改写型（非子序列），{reason_code} 可能拦对也可能拦错，交 codex 判定",
    )


def label_accept(original: str, proposed: str) -> tuple[str, str, str, str]:
    """Initial-label a kept triple: (label, confidence, reason, category).

    The question for a kept is "did the guard accept correctly, or miss danger?".
    A non-"control" category flags a row that must always be exported (never
    sampled out). Both heuristics below are CHARACTER-LEVEL and demonstrably
    over-fire on ASR noise (办法->没办法 adds 没 but is a legit fix, not a flip;
    splitting digits with a comma looks like a new number but is not), so we do
    NOT auto-label them should_reject — they go to codex for the semantic call.
    """
    if inserts_negation(original, proposed):
        return (
            "needs_review",
            "low",
            "否定词增多：可能是危险翻转（我就说→我就不说），也可能是合法 ASR 纠错"
            "（办法→没办法），字符启发式分不清，交 codex 判定",
            "suspect_negation",
        )
    if introduces_arabic_number(original, proposed):
        return (
            "needs_review",
            "low",
            "引入原文没有的数字：可能编造，也可能是中文数字转写（O三→O3）或加分隔符，"
            "交 codex 判定",
            "suspect_number",
        )
    return ("should_keep", "low", "guard 正常放行，抽样作正确放行对照", "control")


def build_sentences(rows: list[dict]) -> list[SentenceSegment]:
    """Rebuild the per-sidecar sentence list in candidate order for neighbour lookup."""
    return [
        SentenceSegment(
            int(row.get("begin_time_ms") or index * 1000),
            int(row.get("end_time_ms") or index * 1000 + 500),
            str(row.get("original_text", "")),
            None,
            index,
        )
        for index, row in enumerate(rows)
    ]


def neighbour_texts(rows: list[dict], index: int) -> tuple[str, str]:
    """Return (previous_text, next_text) for cross-sentence context in a case."""
    prev = str(rows[index - 1].get("original_text", "")).strip() if index > 0 else ""
    nxt = (
        str(rows[index + 1].get("original_text", "")).strip()
        if index + 1 < len(rows)
        else ""
    )
    return prev, nxt


def main() -> None:
    """Scan all sidecars, replay the current guard, label, dedup, and write JSONL."""
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    sidecars = sorted(PROJECTS_DIR.glob("p-*/tmp/corrections/polish_strict_meta_*.json"))

    seen: set[tuple[str, str]] = set()
    reject_rows: list[dict] = []
    accept_rows: list[dict] = []
    label_stats: Counter = Counter()
    kept_seen = 0

    for path in sidecars:
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        project = path.parts[path.parts.index("projects") + 1]
        rows = payload.get("items", [])
        sentences = build_sentences(rows)
        for index, row in enumerate(rows):
            original = str(row.get("original_text", "")).strip()
            proposed = str(row.get("proposed_text", "")).strip()
            change_type = str(row.get("change_type", ""))
            if not proposed or proposed == original:
                continue
            dedup_key = (original, proposed)
            if dedup_key in seen:
                continue
            seen.add(dedup_key)

            if not _is_change_type_allowed(change_type):
                verdict = "reject_unknown_type"
            else:
                verdict = _polish_guard(index, sentences, original, proposed)
            prev_text, next_text = neighbour_texts(rows, index)

            if verdict is not None:
                label, confidence, reason = label_reject(original, proposed, verdict)
                label_stats[f"reject/{label}"] += 1
                reject_rows.append(
                    {
                        "id": f"rej_{len(reject_rows):04d}",
                        "category": verdict.split(":", 1)[0],
                        "original_text": original,
                        "proposed_text": proposed,
                        "change_type": change_type,
                        "previous_text": prev_text,
                        "next_text": next_text,
                        "guard_decision": f"reject:{verdict}",
                        "is_subsequence": is_subsequence(proposed, original),
                        "initial_label": label,
                        "label_confidence": confidence,
                        "label_reason": reason,
                        "source": project,
                    }
                )
                continue

            # Kept by the current guard: keep all suspects, sample the rest.
            label, confidence, reason, accept_category = label_accept(original, proposed)
            suspected = accept_category != "control"
            if not suspected and kept_seen % _ACCEPT_SAMPLE_STRIDE != 0:
                kept_seen += 1
                continue
            kept_seen += 1
            label_stats[f"accept/{label}"] += 1
            accept_rows.append(
                {
                    "id": f"acc_{len(accept_rows):04d}",
                    "category": accept_category,
                    "original_text": original,
                    "proposed_text": proposed,
                    "change_type": change_type,
                    "previous_text": prev_text,
                    "next_text": next_text,
                    "guard_decision": "kept",
                    "is_subsequence": is_subsequence(proposed, original),
                    "initial_label": label,
                    "label_confidence": confidence,
                    "label_reason": reason,
                    "source": project,
                }
            )

    _write_jsonl(REJECT_OUT, reject_rows)
    _write_jsonl(ACCEPT_OUT, accept_rows)
    _print_summary(reject_rows, accept_rows, label_stats)


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    """Write rows as one JSON object per line (UTF-8, human-readable)."""
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def _print_summary(
    reject_rows: list[dict], accept_rows: list[dict], label_stats: Counter
) -> None:
    """Print dataset sizes and the initial-label distribution for inspection."""
    print("=" * 64)
    print(f"被拒 case (全量): {len(reject_rows)}  -> {REJECT_OUT}")
    print(f"放行 case (抽样): {len(accept_rows)}  -> {ACCEPT_OUT}")
    print("=" * 64)
    print("\n[初标分布]")
    for key, count in sorted(label_stats.items()):
        print(f"  {key:32s} {count:5d}")
    needs = sum(v for k, v in label_stats.items() if k.endswith("needs_review"))
    print(f"\n需 codex 重点复核 (needs_review): {needs}")
    print("（should_keep/should_reject 也会全量过 codex，needs_review 是最不确定的）")


if __name__ == "__main__":
    main()
