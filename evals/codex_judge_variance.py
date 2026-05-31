"""Measure the codex quality judge's variance: self-consistency + position bias.

The model switch to qwen3.7-max rode on "3.7 胜 63%", a SINGLE codex run. codex
is a non-deterministic LLM, so a single run has unknown variance — 63% could be
solid or could be noise. Before a 3.4x-cost decision leans on it, the judge
itself needs an error bar. This re-judges an already-judged subset two ways:

  * repeat: same cases, same A/B order, judged again. Agreement with the first
    run is the run-to-run self-consistency. <~90% means the headline is shaky.
  * flip:   same cases, A and B swapped. A position-unbiased judge flips its
    letter but keeps the same WINNING MODEL. Disagreement on the winning model
    is pure position bias (independent of the edit-more bias E already found).

Reuses codex_quality_judge's call_codex/build_prompt so the prompt is identical
to the production judge. Reads codex_quality_verdicts.jsonl as run-1 (no re-cost
for that). The re-judge IS outward-facing (codex calls) so it is opt-in:

    uv run python -m evals.codex_judge_variance --sample 100 --mode both
    uv run python -m evals.codex_judge_variance --dry-run        # plan only, no calls
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from evals._log import log

LOCAL = Path(__file__).resolve().parent / "local"
RUN1 = LOCAL / "codex_quality_verdicts.jsonl"
OUT = LOCAL / "codex_variance.json"
BATCH = 15
WORKERS = 6


def _load_run1() -> list[dict]:
    """Load the first judging pass; each row already carries A/B and the winner."""
    if not RUN1.exists():
        return []
    return [json.loads(line) for line in RUN1.read_text(encoding="utf-8").splitlines() if line.strip()]


def _seeded_sample(rows: list[dict], n: int, seed: int) -> list[dict]:
    """Deterministic subset of already-judged rows to re-judge."""
    import random

    rng = random.Random(seed)
    pool = list(rows)
    rng.shuffle(pool)
    return pool[: min(n, len(pool))]


def _winner_from_letter(row: dict, better: str) -> str:
    """Map an A/B/tie letter to the true model, given this row's A-model."""
    if better == "tie":
        return "tie"
    a_model = row["_a_model"]
    b_model = "qwen3.7-max" if a_model == "qwen3.6-plus" else "qwen3.6-plus"
    return a_model if better == "A" else b_model


def _rejudge(cases: list[dict], *, flip: bool) -> dict[str, str]:
    """Re-judge cases via codex; return {id: winner_model}. flip swaps A/B."""
    from evals.codex_quality_judge import build_prompt, call_codex

    prompts = []
    for c in cases:
        a, b = (c["B"], c["A"]) if flip else (c["A"], c["B"])
        prompts.append({"id": c["id"], "original": c["original"], "A": a, "B": b})
    batches = [prompts[i : i + BATCH] for i in range(0, len(prompts), BATCH)]
    by_id_row = {c["id"]: c for c in cases}
    winners: dict[str, str] = {}

    def judge(batch: list[dict]) -> dict[str, str]:
        for _ in range(2):
            try:
                verdicts = call_codex(build_prompt(batch))
                out: dict[str, str] = {}
                for item in batch:
                    v = verdicts.get(item["id"])
                    if not v:
                        continue
                    letter = v["better"]
                    # When flipped, A and B were swapped in the prompt, so a letter
                    # 'A' now points at the model that was originally B. Undo it
                    # before mapping to the true model, so winner is comparable.
                    if flip and letter in ("A", "B"):
                        letter = "B" if letter == "A" else "A"
                    out[item["id"]] = _winner_from_letter(by_id_row[item["id"]], letter)
                return out
            except Exception as exc:  # noqa: BLE001 - judge batch is best-effort
                log.warning("rejudge_batch_failed", err=str(exc)[:120])
        return {}

    with ThreadPoolExecutor(max_workers=WORKERS) as ex:
        futs = [ex.submit(judge, b) for b in batches]
        for fut in as_completed(futs):
            winners.update(fut.result())
    return winners


def _agreement(run1: dict[str, str], run2: dict[str, str]) -> tuple[int, int, Counter]:
    """Count winning-model agreement between two passes over the shared ids."""
    shared = set(run1) & set(run2)
    agree = sum(1 for cid in shared if run1[cid] == run2[cid])
    flips = Counter((run1[cid], run2[cid]) for cid in shared if run1[cid] != run2[cid])
    return agree, len(shared), flips


def _winrate(winners: dict[str, str]) -> dict[str, float]:
    """Win-rate by model over a verdict map."""
    c = Counter(winners.values())
    n = len(winners) or 1
    return {m: round(v / n * 100, 1) for m, v in c.most_common()}


def main() -> None:
    """Re-judge a subset to report self-consistency and position bias."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--sample", type=int, default=100)
    parser.add_argument("--seed", type=int, default=11)
    parser.add_argument("--mode", choices=["repeat", "flip", "both"], default="both")
    parser.add_argument("--dry-run", action="store_true", help="Plan only; no codex calls.")
    args = parser.parse_args()

    rows = _load_run1()
    if not rows:
        log.error("no_run1", path=str(RUN1), hint="先跑 codex_quality_judge 产出第一遍")
        return
    sample = _seeded_sample(rows, args.sample, args.seed)
    run1 = {r["id"]: r["winner_model"] for r in sample}
    run1_rate = _winrate(run1)
    log.info("variance_plan", run1_cases=len(rows), resampled=len(sample),
             mode=args.mode, run1_winrate=run1_rate)
    if args.dry_run:
        log.info("dry_run", note="加 --mode 真跑 codex(外部调用);此处仅打印计划")
        return

    report: dict[str, object] = {"sample": len(sample), "run1_winrate": run1_rate}

    if args.mode in ("repeat", "both"):
        run2 = _rejudge(sample, flip=False)
        agree, shared, flips = _agreement(run1, run2)
        report["repeat"] = {
            "shared": shared,
            "self_consistency_pct": round(agree / max(1, shared) * 100, 1),
            "run2_winrate": _winrate(run2),
            "winner_flips": {f"{a}->{b}": n for (a, b), n in flips.items()},
        }
        log.info("repeat_done", **report["repeat"])

    if args.mode in ("flip", "both"):
        run3 = _rejudge(sample, flip=True)
        agree, shared, flips = _agreement(run1, run3)
        report["flip"] = {
            "shared": shared,
            "position_consistency_pct": round(agree / max(1, shared) * 100, 1),
            "flipped_winrate": _winrate(run3),
            "winner_flips": {f"{a}->{b}": n for (a, b), n in flips.items()},
            "note": "position_consistency 低 = 换 A/B 位置就改判 = 位置偏置",
        }
        log.info("flip_done", **report["flip"])

    OUT.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    log.info("variance_written", out=str(OUT))


if __name__ == "__main__":
    main()
