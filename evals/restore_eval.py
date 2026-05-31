"""Model-capability eval: semantic anomaly detection + context restoration.

Tests a paradigm that does NOT rely on enumerated wrong_text -> right mappings.
The model is given only the authoritative lexicon vocabulary (canonical term +
its context signature) and must, per sentence, detect a phonetically-odd /
context-incongruent token and restore it to a vocabulary entry ONLY when sound
and context both support it — otherwise leave it untouched. ASR misrecognition
forms are open-ended (the same proper noun mis-hears many ways), so a vocabulary
+ context approach is more robust than chasing every wrong spelling.

Why this is an eval, not a one-off: the model's restoration ability (recall) and
its restraint (not over-restoring normal words — precision) are model-dependent,
and so is its tolerance for long context (attention dilution). Re-run on every
model swap to track capability quantitatively, and to find the chunk-size /
overlap knee where a given model starts to dilute.

Metrics, per (chunk_size, overlap, vocab) config:
  * recall        — restored ASR misrecognitions / total recall cases.
  * false_restore — unchanged sentences the model wrongly edited (over-reach).
  * role_leak     — ambiguous-sense probes wrongly restored (e.g. role "IC"
                    turned into platform "iSee"); needs evals/local probes.

Ground truth is the real raw -> corrected diff on-machine; meeting text and the
scoreboard are git-ignored under evals/local/. The `:.` puts the repo root on
the path so `import evals` resolves. Run:

    PYTHONPATH=src:. .venv/bin/python evals/restore_eval.py qwen3.7-max \
        p-ad0faa7dfee0b9a2 --window 150

Optional ambiguous-sense probes (real terms, git-ignored) live in
evals/local/restore_probes.jsonl as one JSON object per line:
    {"id": "...", "text": "...一线IC...", "restore_to": "iSee"}
where `restore_to` is the form the model must NOT produce (role-sense kept).
"""

from __future__ import annotations

import argparse
import json
import re
from datetime import datetime
from pathlib import Path

from evals._log import log
from app.config import load_settings
from app.correction_llm import generate_chat_text
from app.lexicon_store import (
    get_lexicon_term,
    list_lexicon_disambiguations,
    list_lexicon_terms,
)

PROJ = Path.home() / ".local" / "share" / "meeting-asr" / "projects"
OUT = Path(__file__).resolve().parent / "local"
# Bump when the prompt below changes; scoreboard rows are only comparable across
# models within the same PROMPT_VERSION.
PROMPT_VERSION = 1

ASCII_TOKEN = re.compile(r"[A-Za-z0-9][A-Za-z0-9_+.#-]*")

RESTORE_SYSTEM = (
    "你是会议转写的【专名还原】助手。转写来自语音识别，口齿不清/收音差/模型误差会把"
    "领域专名识别成读音相近的错词，错误形态不固定（同一个专名可能错成多种词）。"
    "给你一份【权威词库】——本组织真实存在的专名及其用途/语境。"
    "任务：找出句中『语境违和或读音可疑』的词，**当且仅当**同时满足"
    "（a）它读音接近词库里某个专名，且（b）该专名在当前上下文里讲得通，才还原成该专名；"
    "否则一律保持原样。绝不无中生有，绝不把正常词/常见英文词改成专名，不确定就不改。"
    "只对 role=core 的句子判断，context 句仅供上下文。只输出 JSON，不要解释。"
)


def load_sents(path: Path) -> list[tuple[int, str]]:
    """Load (sentence_id, text) pairs from a sentences json file."""
    data = json.loads(path.read_text(encoding="utf-8"))
    rows = data if isinstance(data, list) else data.get("sentences", [])
    return [
        (int(r["sentence_id"]), r.get("text", "")) for r in rows if "sentence_id" in r
    ]


def token_present(word: str, text: str) -> bool:
    """Whether `word` occurs in `text` (token boundaries for ASCII terms)."""
    if ASCII_TOKEN.fullmatch(word):
        pattern = rf"(?<![A-Za-z0-9_+.#-]){re.escape(word)}(?![A-Za-z0-9_+.#-])"
        return re.search(pattern, text) is not None
    return word in text


def ascii_canonicals() -> list[str]:
    """Lexicon canonical terms that contain Latin letters (ASR mangles these)."""
    return [
        t.canonical
        for t in list_lexicon_terms(limit=500)
        if re.search(r"[A-Za-z]", t.canonical)
    ]


def build_vocab_block(canon: list[str]) -> str:
    """Authoritative vocabulary + context signature, all sourced from the lexicon."""
    guide = {d.canonical: d.guidance for d in list_lexicon_disambiguations()}
    lines = []
    for term in canon:
        try:
            detail = get_lexicon_term(term, context_limit=0)
            desc, cat = detail.term.description or "", detail.term.category or ""
        except Exception:
            desc, cat = "", ""
        signature = guide.get(term, "") or desc
        tag = f"（{cat}）" if cat and cat != "unknown" else ""
        lines.append(f"- {term}{tag}：{signature}" if signature else f"- {term}{tag}")
    return "\n".join(lines)


def build_prompt(
    chunk: list[tuple[str, str, str]], vocab_block: str, use_vocab: bool
) -> str:
    """Build a restoration prompt for one chunk of (id, role, text) rows."""
    payload = [{"id": sid, "role": role, "text": text} for sid, role, text in chunk]
    header = (
        f"【权威词库（音近+语境匹配才可还原）】\n{vocab_block}\n\n"
        if use_vocab
        else "【无词库，仅凭通用常识判断专名误识别】\n\n"
    )
    return (
        header
        + "【待还原句子】role=core 需你判断，context 仅上下文：\n"
        + json.dumps(payload, ensure_ascii=False)
        + '\n\n输出：{"restorations":[{"id":"<core id>","text":"<还原后整句>",'
        '"word":"<还原成的专名>","reason":"<音近+语境依据>"}]}'
        "，只返回真正做了还原的 core 句，没有就空数组。"
    )


def call_restore(chunk, vocab_block, use_vocab, settings, model) -> dict[str, str]:
    """Run one restoration call; return {sentence_id: restored_text}."""
    prompt = build_prompt(chunk, vocab_block, use_vocab)
    content = generate_chat_text(
        settings=settings,
        model=model,
        messages=[
            {"role": "system", "content": RESTORE_SYSTEM},
            {"role": "user", "content": prompt},
        ],
        request_timeout=240,
        temperature=0.0,
        enable_thinking=False,
    )
    match = re.search(r"\{.*\}", content, re.DOTALL)
    if not match:
        return {}
    try:
        obj = json.loads(match.group(0))
    except json.JSONDecodeError:
        return {}
    return {str(r.get("id")): r.get("text", "") for r in obj.get("restorations", [])}


def slice_with_overlap(items, size, overlap):
    """Chunks of `size` core items, padded with `overlap` context items each side."""
    chunks = []
    start = 0
    n = len(items)
    while start < n:
        core_end = min(start + size, n)
        lo, hi = max(0, start - overlap), min(n, core_end + overlap)
        chunk = []
        for i in range(lo, hi):
            role = "core" if start <= i < core_end else "context"
            chunk.append((str(items[i][0]), role, items[i][1]))
        chunks.append(chunk)
        start = core_end
    return chunks


def densest_window(rows, recall_sids, window):
    """Return the contiguous window of `window` rows covering the most recall cases."""
    if len(rows) <= window:
        return rows
    best_start, best = 0, -1
    for start in range(0, len(rows) - window + 1):
        ids = {rows[i][0] for i in range(start, start + window)}
        hit = len(recall_sids & ids)
        if hit > best:
            best, best_start = hit, start
    return rows[best_start : best_start + window]


def extract_project(proj: str, canon: list[str], window: int):
    """Extract a windowed test set + ground truth from one on-machine project."""
    raw = load_sents(PROJ / proj / "asr" / "sentences.json")
    cor = dict(load_sents(PROJ / proj / "asr" / "sentences_corrected.json"))
    raw_by_id = dict(raw)
    recall_all = [
        (sid, w)
        for sid, ctext in cor.items()
        for w in canon
        if token_present(w, ctext) and not token_present(w, raw_by_id.get(sid, ""))
    ]
    win = densest_window(raw, {sid for sid, _ in recall_all}, window)
    win_ids = {sid for sid, _ in win}
    recall = [(sid, w) for sid, w in recall_all if sid in win_ids]
    unchanged = [sid for sid, t in win if cor.get(sid, t) == t]
    return win, raw_by_id, recall, unchanged


def load_role_probes() -> list[dict]:
    """Optional ambiguous-sense probes from evals/local (real terms, git-ignored)."""
    path = OUT / "restore_probes.jsonl"
    if not path.exists():
        return []
    probes = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            probes.append(json.loads(line))
    return probes


def score(outputs, recall, raw_by_id, unchanged, probes):
    """Return (recall_hits, false_restores, role_leaks)."""
    rec = sum(
        1
        for sid, w in recall
        if token_present(w, outputs.get(str(sid), raw_by_id.get(sid, "")))
    )
    false_restore = sum(
        1
        for sid in unchanged
        if str(sid) in outputs and outputs[str(sid)] != raw_by_id.get(sid, "")
    )
    role_leak = sum(
        1
        for p in probes
        if token_present(p["restore_to"], outputs.get(str(p["id"]), p["text"]))
    )
    return rec, false_restore, role_leak


def run_config(items, vocab_block, size, overlap, use_vocab, settings, model):
    """Run one (size, overlap, vocab) config over the items; return merged outputs."""
    merged = {}
    chunks = slice_with_overlap(items, size, overlap)
    for chunk in chunks:
        try:
            merged.update(call_restore(chunk, vocab_block, use_vocab, settings, model))
        except Exception as exc:
            log.warning("chunk_error", size=size, overlap=overlap, err=str(exc)[:120])
    return merged, len(chunks)


def parse_args():
    ap = argparse.ArgumentParser(description="Restoration capability eval.")
    ap.add_argument("model", help="DashScope text model id, e.g. qwen3.7-max")
    ap.add_argument(
        "projects", nargs="+", help="On-machine project ids with corrected output"
    )
    ap.add_argument(
        "--window", type=int, default=150, help="Contiguous sentences per project"
    )
    ap.add_argument(
        "--chunks", default="0,30,10", help="Comma chunk sizes; 0 = whole window"
    )
    ap.add_argument(
        "--overlaps", default="0,6", help="Comma overlaps to test (chunked configs)"
    )
    ap.add_argument(
        "--no-control", action="store_true", help="Skip the NO-VOCAB control run"
    )
    return ap.parse_args()


def main():
    args = parse_args()
    settings = load_settings(require_oss=False, require_dashscope=True)
    canon = ascii_canonicals()
    vocab_block = build_vocab_block(canon)
    probes = load_role_probes()
    chunk_sizes = [int(x) for x in args.chunks.split(",")]
    overlaps = [int(x) for x in args.overlaps.split(",")]

    # Aggregate the ground-truth test set across the requested projects.
    sets = []
    for proj in args.projects:
        win, raw_by_id, recall, unchanged = extract_project(proj, canon, args.window)
        items = [(sid, t) for sid, t in win] + [(p["id"], p["text"]) for p in probes]
        sets.append(
            (
                proj,
                items,
                raw_by_id | {p["id"]: p["text"] for p in probes},
                recall,
                unchanged,
            )
        )
        log.info(
            "project_set",
            proj=proj,
            window=len(win),
            recall=len(recall),
            unchanged=len(unchanged),
        )

    total_recall = sum(len(s[3]) for s in sets)
    configs = []
    for size in chunk_sizes:
        for overlap in overlaps if size else [0]:
            configs.append(
                (
                    f"{'full' if not size else f'chunk{size}'}/ov{overlap}",
                    size,
                    overlap,
                    True,
                )
            )
    if not args.no_control:
        configs.append(("chunk30/ov6 NO-VOCAB", 30, 6, False))

    print(
        f"\nmodel={args.model}  prompt_v={PROMPT_VERSION}  "
        f"projects={len(sets)}  recall_cases={total_recall}  "
        f"unchanged={sum(len(s[4]) for s in sets)}  role_probes={len(probes)}\n"
    )
    print(
        f"{'config':24} {'calls':>5} {'recall':>10} {'false_restore':>13} {'role_leak':>9}"
    )

    rows = []
    for name, size, overlap, use_vocab in configs:
        rec_t = fr_t = rl_t = calls_t = 0
        for _proj, items, raw_by_id, recall, unchanged in sets:
            window_size = size if size else len(items)
            merged, calls = run_config(
                items,
                vocab_block,
                window_size,
                overlap,
                use_vocab,
                settings,
                args.model,
            )
            rec, fr, rl = score(merged, recall, raw_by_id, unchanged, probes)
            rec_t += rec
            fr_t += fr
            rl_t += rl
            calls_t += calls
        print(
            f"{name:24} {calls_t:>5} {f'{rec_t}/{total_recall}':>10} {fr_t:>13} {rl_t:>9}"
        )
        rows.append(
            {
                "config": name,
                "calls": calls_t,
                "recall": rec_t,
                "recall_total": total_recall,
                "false_restore": fr_t,
                "role_leak": rl_t,
            }
        )

    OUT.mkdir(parents=True, exist_ok=True)
    board = OUT / "restore_scoreboard.jsonl"
    entry = {
        "model": args.model,
        "prompt_version": PROMPT_VERSION,
        "stamp": datetime.now().strftime("%Y-%m-%d %H:%M"),
        "window": args.window,
        "projects": args.projects,
        "rows": rows,
    }
    with board.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(entry, ensure_ascii=False) + "\n")
    print(f"\nscoreboard appended -> {board}")


if __name__ == "__main__":
    main()
