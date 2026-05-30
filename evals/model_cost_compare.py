"""Time + token + cost comparison for two Polish models on real sentences.

Runs BOTH models fresh on the SAME capped sentence sample under identical
concurrency, monkeypatching the DashScope client to capture real input/output
token usage (the production path discards it), and times each run with a
monotonic clock. Cost is tokens x a STATED public-list rate — the code has no
text-model pricing table, so the rates here are assumptions to confirm, while
tokens and wall-clock are hard measured data.

Real meeting text: nothing is written to git. Run:
    PYTHONPATH=src .venv/bin/python evals/model_cost_compare.py <projA> [cap]
"""

from __future__ import annotations

import sys
import threading
import time
from pathlib import Path

import app.dashscope_chat as dc
from app.correction_llm import LlmCorrectionCandidate, propose_transcript_polish_strict
from app.config import load_settings
from app.transcript_corrections import (
    _POLISH_STRICT_BATCH_SIZE,
    _strict_polish_request_timeout,
)
from concurrent.futures import ThreadPoolExecutor, as_completed
import json

PROJ = Path.home() / ".local" / "share" / "meeting-asr" / "projects"

# STATED ASSUMPTIONS — DashScope public list price, ¥/1k tokens. Confirm/replace.
# plus tier ~ qwen-plus; max tier ~ qwen-max. Tiered context pricing ignored.
RATE_RMB_PER_1K = {
    "qwen3.6-plus": {"in": 0.0008, "out": 0.002},
    "qwen3.7-max": {"in": 0.0024, "out": 0.0096},
}

_usage: dict[str, list[int]] = {}
_endpoint: dict[str, str] = {}
_lock = threading.Lock()


def _meter(orig, tag: str):
    """Wrap a DashScope .call to accumulate per-model token usage + endpoint."""
    def wrapper(*args, **kwargs):
        resp = orig(*args, **kwargs)
        model = kwargs.get("model", "?")
        u = getattr(resp, "usage", None)

        def field(key: str) -> int:
            if u is None:
                return 0
            if isinstance(u, dict):
                return int(u.get(key, 0) or 0)
            return int(getattr(u, key, 0) or 0)

        with _lock:
            rec = _usage.setdefault(model, [0, 0, 0])
            rec[0] += field("input_tokens")
            rec[1] += field("output_tokens")
            rec[2] += 1
            _endpoint[model] = tag
        return resp
    return wrapper


# qwen3.6-plus routes to the multimodal endpoint, qwen3.7-max to generation — meter both.
dc.Generation.call = _meter(dc.Generation.call, "generation")
dc.MultiModalConversation.call = _meter(dc.MultiModalConversation.call, "multimodal")


def latest_items(proj: str) -> list[dict]:
    """Return the original sentences from a project's latest polish sidecar."""
    sc = sorted((PROJ / proj).glob("tmp/corrections/polish_strict_meta_*.json"))
    return json.loads(sc[-1].read_text(encoding="utf-8")).get("items", []) if sc else []


def run_model(items: list[dict], model: str) -> float:
    """Run strict polish on the items with a model; return wall-clock seconds."""
    settings = load_settings(require_oss=False, require_dashscope=True)
    timeout = _strict_polish_request_timeout(model)
    cands = [
        LlmCorrectionCandidate(candidate_id=f"c{i}", sentence_id=i, speaker_name="S",
                               text=str(it.get("original_text", "")))
        for i, it in enumerate(items)
    ]
    batches = [cands[i:i + _POLISH_STRICT_BATCH_SIZE]
               for i in range(0, len(cands), _POLISH_STRICT_BATCH_SIZE)]
    t0 = time.perf_counter()

    def one(b):
        try:
            return propose_transcript_polish_strict(
                candidates=b, settings=settings, model=model, request_timeout=timeout)
        except Exception as exc:  # noqa: BLE001
            print(f"  ! batch failed ({model}): {type(exc).__name__}: {str(exc)[:70]}")
            return None

    with ThreadPoolExecutor(max_workers=12) as ex:
        list(as_completed([ex.submit(one, b) for b in batches]))
    return time.perf_counter() - t0


def main() -> None:
    """Run both models on a capped sample and report time/tokens/cost."""
    proj = sys.argv[1]
    cap = int(sys.argv[2]) if len(sys.argv) > 2 else 300
    items = latest_items(proj)[:cap]
    n = len(items)
    print(f"项目 {proj} | 取前 {n} 句 | 并发 12 | 两模型先后跑\n")

    timings: dict[str, float] = {}
    for model in ("qwen3.6-plus", "qwen3.7-max"):
        print(f"跑 {model} ...")
        timings[model] = run_model(items, model)
        print(f"  完成 {model}: {timings[model]:.1f}s")

    print("\n" + "=" * 74)
    print(f"时间 / Token / 成本对比  (n={n} 句)")
    print("=" * 74)
    hdr = f"{'model':14s}{'墙钟s':>8s}{'句/s':>7s}{'in_tok':>9s}{'out_tok':>9s}{'¥(估)':>9s}{'¥/百句':>9s}"
    print(hdr)
    for model in ("qwen3.6-plus", "qwen3.7-max"):
        secs = timings[model]
        rec = _usage.get(model, [0, 0, 0])
        rate = RATE_RMB_PER_1K[model]
        cost = rec[0] / 1000 * rate["in"] + rec[1] / 1000 * rate["out"]
        per100 = cost / n * 100 if n else 0
        sps = n / secs if secs else 0
        ep = _endpoint.get(model, "?")
        print(f"{model:14s}{secs:8.1f}{sps:7.1f}{rec[0]:9d}{rec[1]:9d}{cost:9.4f}{per100:9.4f}  [{ep}]")

    print("\n[费率假设 ¥/1k token，公开档位价，待你确认]")
    for m, r in RATE_RMB_PER_1K.items():
        print(f"  {m:14s} in {r['in']}  out {r['out']}")
    print("\n注：token 与墙钟为实测；成本随费率假设缩放。")


if __name__ == "__main__":
    main()
