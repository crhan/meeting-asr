"""Light Polish model comparison on real on-machine projects.

Baseline = qwen3.6-plus, whose proposals already live in each project's strict
polish sidecar. Challenger = a model id passed on the CLI (e.g. qwen3.7-max),
re-run live on the SAME original sentences. BOTH proposal sets are replayed
through the current production guard (with the live lexicon vocab), so the
comparison isolates the effect of the polish MODEL on:

  * reject rate  — fraction of the model's proposed changes the guard rejects
                   (the user's pain: "Polish 老被拒"). Lower with equal quality
                   = more guard-compatible proposals.
  * accepted     — proposed changes that survive the guard to reach the user.

Every sentence where the two models diverge is dumped to evals/local/ for
spot-check + codex quality judgement — accept rate alone can't say which edit is
better (a model that edits less trivially gets rejected less).

Real meeting text: the dump is git-ignored. Run:
    PYTHONPATH=src .venv/bin/python evals/model_compare.py qwen3.7-max p-69ca1d7502ebec7d p-625dc6de7e0c96c4
"""

from __future__ import annotations

import json
import os
import sys
import threading
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import app.dashscope_chat as dc

from evals._log import log
from app.config import load_settings
from app.correction_llm import (
    LlmCorrectionCandidate,
    propose_transcript_polish_strict,
)
from app.lexicon_store import list_lexicon_known_texts
from app.models import SentenceSegment
from app.transcript_corrections import (
    _POLISH_STRICT_BATCH_SIZE,
    _is_change_type_allowed,
    _polish_guard,
    _strict_polish_request_timeout,
)

PROJ = Path.home() / ".local" / "share" / "meeting-asr" / "projects"
OUT = Path(__file__).resolve().parent / "local"
VOCAB = list_lexicon_known_texts()

# STATED ASSUMPTION — DashScope public list, ¥/1k tokens, for the challenger only.
RATE_RMB_PER_1K = {"in": 0.0024, "out": 0.0096}  # qwen3.7-max (max tier)

# Concurrency is tunable via MC_WORKERS env; backoff absorbs any rate-limits.
MAX_WORKERS = int(os.environ.get("MC_WORKERS", "24"))
MAX_RETRIES = 4
_BACKOFF_S = (2, 5, 12)  # waits before retries 2/3/4

# Meter the challenger's real token usage so the full run also reports its cost.
_usage = [0, 0, 0]  # input_tokens, output_tokens, calls
_meter_lock = threading.Lock()


def _meter(orig):
    """Wrap a DashScope .call to accumulate challenger token usage."""
    def wrapper(*args, **kwargs):
        resp = orig(*args, **kwargs)
        u = getattr(resp, "usage", None)

        def field(key: str) -> int:
            if u is None:
                return 0
            if isinstance(u, dict):
                return int(u.get(key, 0) or 0)
            return int(getattr(u, key, 0) or 0)

        with _meter_lock:
            _usage[0] += field("input_tokens")
            _usage[1] += field("output_tokens")
            _usage[2] += 1
        return resp
    return wrapper


dc.Generation.call = _meter(dc.Generation.call)
dc.MultiModalConversation.call = _meter(dc.MultiModalConversation.call)


def latest_sidecar(proj: str) -> dict | None:
    """Return the most recent strict-polish sidecar payload for a project."""
    sidecars = sorted((PROJ / proj).glob("tmp/corrections/polish_strict_meta_*.json"))
    if not sidecars:
        return None
    return json.loads(sidecars[-1].read_text(encoding="utf-8"))


def build_sentences(items: list[dict]) -> list[SentenceSegment]:
    """Rebuild the sentence list (for cross-sentence borrow neighbour lookup)."""
    return [
        SentenceSegment(i * 1000, i * 1000 + 500, str(it.get("original_text", "")), None, i)
        for i, it in enumerate(items)
    ]


def guard_verdict(idx: int, sentences: list[SentenceSegment], original: str,
                  proposed: str, change_type: str) -> str:
    """Replay the production guard on one (original, proposed) pair."""
    if not proposed or proposed == original:
        return "no_change"
    if not _is_change_type_allowed(change_type):
        return "reject:type"
    v = _polish_guard(idx, sentences, original, proposed, VOCAB)
    return f"reject:{v}" if v is not None else "accept"


def load_projects(projects: list[str]) -> list[dict]:
    """Load each project's latest sidecar + sentence context; skip ones with none."""
    loaded = []
    for proj in projects:
        payload = latest_sidecar(proj)
        if not payload:
            log.info("skip_no_sidecar", proj=proj)
            continue
        items = payload.get("items", [])
        loaded.append({"proj": proj, "items": items, "sentences": build_sentences(items)})
    return loaded


def _make_batches(loaded: list[dict]) -> list[list[LlmCorrectionCandidate]]:
    """Flatten ALL projects into globally-tagged candidate batches.

    candidate_id is ``"<proj_idx>:<sentence_idx>"`` so one global pool can run
    every project's batches together, yet each proposal still keys back to its
    project. This is the fix for the per-project pool that capped concurrency at
    a single (usually small) project's batch count.
    """
    batches: list[list[LlmCorrectionCandidate]] = []
    for pi, p in enumerate(loaded):
        cands = [
            LlmCorrectionCandidate(candidate_id=f"{pi}:{i}", sentence_id=i,
                                   speaker_name="S", text=str(it.get("original_text", "")))
            for i, it in enumerate(p["items"])
        ]
        for j in range(0, len(cands), _POLISH_STRICT_BATCH_SIZE):
            batches.append(cands[j:j + _POLISH_STRICT_BATCH_SIZE])
    return batches


def run_all(batches: list[list[LlmCorrectionCandidate]], model: str) -> dict[str, object]:
    """Run the challenger over EVERY batch through ONE global pool; id -> item.

    A single ThreadPoolExecutor over all projects' batches keeps MAX_WORKERS busy
    end-to-end (one drain at the very end) instead of refilling/draining per
    project, so the realized concurrency actually approaches MAX_WORKERS.
    """
    settings = load_settings(require_oss=False, require_dashscope=True)
    timeout = _strict_polish_request_timeout(model)
    proposals: dict[str, object] = {}
    failures = 0
    done = 0
    total = len(batches)
    t0 = time.perf_counter()

    def one(batch: list[LlmCorrectionCandidate]) -> list:
        last: Exception | None = None
        for attempt in range(MAX_RETRIES):
            try:
                return propose_transcript_polish_strict(
                    candidates=batch, settings=settings, model=model, request_timeout=timeout,
                ).items
            except Exception as exc:  # noqa: BLE001
                last = exc
                if attempt < MAX_RETRIES - 1:
                    time.sleep(_BACKOFF_S[attempt])
        raise last if last else RuntimeError("batch failed")

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as ex:
        futs = [ex.submit(one, b) for b in batches]
        for fut in as_completed(futs):
            done += 1
            try:
                for item in fut.result():
                    proposals[item.candidate_id] = item
            except Exception as exc:  # noqa: BLE001 — one bad batch must not abort
                failures += 1
                log.warning("batch_failed", err=type(exc).__name__, msg=str(exc)[:80])
            if done % 25 == 0 or done == total:
                el = time.perf_counter() - t0
                cost = _usage[0] / 1000 * RATE_RMB_PER_1K["in"] + _usage[1] / 1000 * RATE_RMB_PER_1K["out"]
                sps = _usage[2] * _POLISH_STRICT_BATCH_SIZE / el if el else 0
                eta = (total - done) / (done / el) if el and done else 0
                log.info("polish_progress", batches=f"{done}/{total}", sps=round(sps, 1),
                         calls=_usage[2], cost=round(cost, 2), eta_min=round(eta / 60))
    if failures:
        log.warning("batches_failed", count=failures, model=model, tries=MAX_RETRIES)
    return proposals


def compare_one(p: dict, pi: int, proposals: dict[str, object]) -> dict:
    """CPU-only guard comparison for one project, using the global proposals."""
    proj, items, sentences = p["proj"], p["items"], p["sentences"]
    tally = {"base": Counter(), "chal": Counter()}
    reasons = {"base": Counter(), "chal": Counter()}
    diverge: list[dict] = []
    for i, it in enumerate(items):
        original = str(it.get("original_text", ""))
        p_base = str(it.get("proposed_text", "") or "")
        ct_base = str(it.get("change_type", ""))
        ch = proposals.get(f"{pi}:{i}")
        p_chal = str(getattr(ch, "corrected_text", "") or "")
        ct_chal = str(getattr(ch, "change_type", "") or "")

        v_base = guard_verdict(i, sentences, original, p_base, ct_base)
        v_chal = guard_verdict(i, sentences, original, p_chal, ct_chal)
        tally["base"][_bucket(v_base)] += 1
        tally["chal"][_bucket(v_chal)] += 1
        if v_base.startswith("reject:"):
            reasons["base"][v_base.split(":", 1)[1].split(":", 1)[0]] += 1
        if v_chal.startswith("reject:"):
            reasons["chal"][v_chal.split(":", 1)[1].split(":", 1)[0]] += 1

        if p_base != p_chal:
            diverge.append({
                "project": proj, "idx": i, "original": original,
                "qwen36_proposed": p_base, "qwen36_guard": v_base, "qwen36_ct": ct_base,
                "challenger_proposed": p_chal, "challenger_guard": v_chal, "challenger_ct": ct_chal,
                "kind": _divergence_kind(v_base, v_chal),
            })
    return {"proj": proj, "n": len(items), "tally": tally, "reasons": reasons, "diverge": diverge}


def _bucket(verdict: str) -> str:
    """Collapse a verdict into accept / reject / no_change."""
    if verdict == "accept":
        return "accept"
    if verdict == "no_change":
        return "no_change"
    return "reject"


def _divergence_kind(v_base: str, v_chal: str) -> str:
    """Label how the two models diverge on one sentence."""
    b, c = _bucket(v_base), _bucket(v_chal)
    if b == c:
        return f"both_{b}_differ" if b != "no_change" else "one_changed"
    return f"base_{b}__chal_{c}"


def main() -> None:
    """Run the comparison over the given projects and print + dump results."""
    if len(sys.argv) < 3:
        print("usage: model_compare.py <challenger_model> <proj> [<proj> ...]")
        raise SystemExit(2)
    model = sys.argv[1]
    projects = sys.argv[2:]
    OUT.mkdir(parents=True, exist_ok=True)
    t0 = time.perf_counter()

    loaded = load_projects(projects)
    batches = _make_batches(loaded)
    total_n = sum(len(p["items"]) for p in loaded)
    log.info("start", projects=len(loaded), sentences=total_n,
             batches=len(batches), workers=MAX_WORKERS, retries=MAX_RETRIES)

    # The slow part: one global pool runs every project's batches together.
    proposals = run_all(batches, model)

    # The fast part: replay the guard per project on the collected proposals.
    all_diverge: list[dict] = []
    agg = {"base": Counter(), "chal": Counter()}
    agg_reasons = {"base": Counter(), "chal": Counter()}
    for pi, p in enumerate(loaded):
        res = compare_one(p, pi, proposals)
        all_diverge.extend(res["diverge"])
        for side in ("base", "chal"):
            agg[side] += res["tally"][side]
            agg_reasons[side] += res["reasons"][side]
    elapsed = time.perf_counter() - t0

    out_path = OUT / f"model_compare_{model.replace('.', '_').replace('/', '_')}.jsonl"
    with out_path.open("w", encoding="utf-8") as h:
        for row in all_diverge:
            h.write(json.dumps(row, ensure_ascii=False) + "\n")

    _report(model, agg, agg_reasons, all_diverge, out_path)
    _report_cost(model, total_n, elapsed)


def _report_cost(model: str, n: int, elapsed: float) -> None:
    """Report the metered cost + wall-clock of THIS evaluation run (challenger only)."""
    cost = _usage[0] / 1000 * RATE_RMB_PER_1K["in"] + _usage[1] / 1000 * RATE_RMB_PER_1K["out"]
    print("\n" + "=" * 66)
    print(f"本次全量评测开销  (challenger={model} 实跑 {n} 句)")
    print("=" * 66)
    print(f"  墙钟 {elapsed:.0f}s ({elapsed/60:.1f} 分) | API 调用 {_usage[2]} 次")
    print(f"  token: 输入 {_usage[0]} / 输出 {_usage[1]}")
    print(f"  DashScope 费用(假设价 in {RATE_RMB_PER_1K['in']} out {RATE_RMB_PER_1K['out']} ¥/1k): ¥{cost:.2f}")
    print("  注：baseline qwen3.6-plus 复用历史 sidecar，不计入本次开销；codex 判定费另计。")


def _report(model, agg, agg_reasons, diverge, out_path) -> None:
    """Print the per-model accept/reject summary and divergence breakdown."""
    print("\n" + "=" * 66)
    print(f"模型对比汇总  baseline=qwen3.6-plus  challenger={model}")
    print("=" * 66)
    for side, name in (("base", "qwen3.6-plus"), ("chal", model)):
        t = agg[side]
        proposed = t["accept"] + t["reject"]
        rej_rate = (t["reject"] / proposed * 100) if proposed else 0.0
        acc_rate = (t["accept"] / proposed * 100) if proposed else 0.0
        print(f"\n[{name}]")
        print(f"  提改 {proposed}  (放行 {t['accept']} / 被拒 {t['reject']} / 未改 {t['no_change']})")
        print(f"  放行率 {acc_rate:.1f}%   拒绝率 {rej_rate:.1f}%")
        if agg_reasons[side]:
            print(f"  拒因: {dict(agg_reasons[side].most_common())}")
    kinds = Counter(d["kind"] for d in diverge)
    print(f"\n[两模型分歧 {len(diverge)} 句] 分类:")
    for k, v in kinds.most_common():
        print(f"  {k:28s}{v}")
    print(f"\n分歧明细已写出: {out_path}")
    print("  关注 base_reject__chal_accept (3.7 把好编辑做进去了) 与")
    print("       base_accept__chal_reject (3.7 反而触发护栏)")


if __name__ == "__main__":
    main()
