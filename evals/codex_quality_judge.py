"""Blind A/B quality judging of two Polish models by codex (cross-source).

Loads the divergences from model_compare, samples the buckets where quality (not
just guard-compatibility) decides — both models edited differently, or one edited
where the other left it — and asks codex which candidate is the better FAITHFUL
polish. A/B order is hidden per case (deterministic by index parity) so codex
cannot tell which model is which; we map the verdict back afterwards and report
win rates by true model.

Per the project rule, codex (an independent model) is the arbiter, never my own
single-sided read. Real meeting text; output git-ignored. Run:
    uv run --no-project python evals/codex_quality_judge.py
"""

from __future__ import annotations

import json
import random
import subprocess
import tempfile
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

LOCAL = Path(__file__).resolve().parent / "local"
DIVERGE = LOCAL / "model_compare_qwen3_7-max.jsonl"
OUT = LOCAL / "codex_quality_verdicts.jsonl"
BATCH = 15
WORKERS = 6
CODEX_TIMEOUT_S = 600

# How many to sample per divergence bucket (quality-relevant ones only).
SAMPLE = {
    "both_accept_differ": 70,
    "base_no_change__chal_accept": 30,
    "base_accept__chal_no_change": 20,
}

SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "required": ["results"],
    "properties": {
        "results": {
            "type": "array",
            "items": {
                "type": "object",
                "additionalProperties": False,
                "required": ["id", "better", "reason"],
                "properties": {
                    "id": {"type": "string"},
                    "better": {"type": "string", "enum": ["A", "B", "tie"]},
                    "reason": {"type": "string"},
                },
            },
        }
    },
}

PROMPT_HEAD = """\
你在评测中文会议 ASR 转写的 polish(润色)质量。每条给你 original(语音识别原文，
含口吃/语气词/同音错字)和两个候选润色 A、B。判定哪个是更好的"严格忠实润色"。

好的润色标准(按重要性):
1. 忠实:不改变原意，不编造原文没有的内容/术语/数字，不从上下文借句子。
   - 把术语错字改成"另一个不同的东西"(如 case库->Skill库、test库->Skill库)是编造，扣分。
   - 把中文名音译成拼音/英文(武一->WuYi)且无依据，是编造，扣分。
2. 干净:去掉口吃、重复、语气词、明显 ASR 噪音，补必要标点/中英文空格。
3. 适度:只删噪音、只改明显错字；过度压缩到丢信息、或重写句子结构，扣分。

A 和 B 谁更好就选谁；若质量基本相当(或都只是忠实保留原文)选 tie。
只依据文本判断，不知道也不要猜哪个来自哪个模型。
"""


def load_sample() -> list[dict]:
    """Sample quality-relevant divergences, balanced across buckets and projects."""
    rows = [json.loads(l) for l in DIVERGE.read_text(encoding="utf-8").splitlines() if l.strip()]
    by_kind: dict[str, list[dict]] = {}
    for r in rows:
        by_kind.setdefault(r["kind"], []).append(r)
    rng = random.Random(42)
    picked: list[dict] = []
    for kind, n in SAMPLE.items():
        pool = by_kind.get(kind, [])
        picked.extend(rng.sample(pool, min(n, len(pool))))
    return picked


def to_ab(row: dict, seq: int) -> dict:
    """Assign 3.6/3.7 proposals to A/B by parity (hidden order); keep the mapping."""
    p36 = row["qwen36_proposed"] or row["original"]
    p37 = row["challenger_proposed"] or row["original"]
    if seq % 2 == 0:
        a_text, b_text, a_model = p36, p37, "qwen3.6-plus"
    else:
        a_text, b_text, a_model = p37, p36, "qwen3.7-max"
    cid = f"{row['project']}_{row['idx']}"
    return {
        "id": cid, "original": row["original"], "A": a_text, "B": b_text,
        "_a_model": a_model, "_kind": row["kind"],
    }


def build_prompt(batch: list[dict]) -> str:
    """Render one batch into the codex prompt."""
    items = [{"id": c["id"], "original": c["original"], "A": c["A"], "B": c["B"]} for c in batch]
    return PROMPT_HEAD + "\n\ncases:\n" + json.dumps(items, ensure_ascii=False)


def call_codex(prompt: str) -> dict[str, dict]:
    """Invoke codex on one batch; return {id: {better, reason}}."""
    with tempfile.TemporaryDirectory() as tmp:
        schema_path = Path(tmp) / "schema.json"
        out_path = Path(tmp) / "out.json"
        schema_path.write_text(json.dumps(SCHEMA), encoding="utf-8")
        proc = subprocess.run(
            ["codex", "exec", "--skip-git-repo-check", "--ephemeral",
             "-o", str(out_path), "--output-schema", str(schema_path), "-"],
            input=prompt, text=True, capture_output=True, timeout=CODEX_TIMEOUT_S,
        )
        if not out_path.exists():
            raise RuntimeError(f"codex no output: {proc.stderr[-300:]}")
        results = json.loads(out_path.read_text(encoding="utf-8"))["results"]
    return {item["id"]: item for item in results}


def judge_batch(batch: list[dict]) -> list[dict]:
    """Judge one batch with one retry; map verdicts back to true models."""
    for _ in range(2):
        try:
            verdicts = call_codex(build_prompt(batch))
            out = []
            for c in batch:
                v = verdicts.get(c["id"])
                if not v:
                    continue
                better_model = _winner(c, v["better"])
                out.append({**{k: c[k] for k in ("id", "original", "A", "B", "_a_model", "_kind")},
                            "better": v["better"], "winner_model": better_model, "reason": v["reason"]})
            return out
        except (subprocess.TimeoutExpired, RuntimeError, json.JSONDecodeError):
            continue
    return []


def _winner(case: dict, better: str) -> str:
    """Map an A/B/tie verdict to the true model that won (or 'tie')."""
    if better == "tie":
        return "tie"
    a_model = case["_a_model"]
    b_model = "qwen3.7-max" if a_model == "qwen3.6-plus" else "qwen3.6-plus"
    return a_model if better == "A" else b_model


def main() -> None:
    """Sample, judge blind via codex, and report win rates by true model."""
    sample = load_sample()
    cases = [to_ab(r, i) for i, r in enumerate(sample)]
    batches = [cases[i:i + BATCH] for i in range(0, len(cases), BATCH)]
    print(f"盲判 {len(cases)} 句，{len(batches)} 批，codex 跨源仲裁\n")

    verdicts: list[dict] = []
    with ThreadPoolExecutor(max_workers=WORKERS) as ex:
        futs = [ex.submit(judge_batch, b) for b in batches]
        for fut in as_completed(futs):
            verdicts.extend(fut.result())

    with OUT.open("w", encoding="utf-8") as h:
        for v in verdicts:
            h.write(json.dumps(v, ensure_ascii=False) + "\n")

    overall = Counter(v["winner_model"] for v in verdicts)
    print("=" * 60)
    print(f"质量盲判结果 (codex)  共判定 {len(verdicts)} 句")
    print("=" * 60)
    for k in ("qwen3.7-max", "qwen3.6-plus", "tie"):
        c = overall.get(k, 0)
        pct = c / len(verdicts) * 100 if verdicts else 0
        print(f"  {k:14s}{c:5d}  ({pct:.1f}%)")
    print("\n[按分歧类型]")
    for kind in SAMPLE:
        sub = [v for v in verdicts if v["_kind"] == kind]
        if not sub:
            continue
        cc = Counter(v["winner_model"] for v in sub)
        print(f"  {kind:30s} 3.7 {cc.get('qwen3.7-max',0)} / 3.6 {cc.get('qwen3.6-plus',0)} / tie {cc.get('tie',0)}  (n={len(sub)})")
    print(f"\n明细: {OUT}")


if __name__ == "__main__":
    main()
