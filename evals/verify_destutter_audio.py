"""Independently spot-check the destutter→keep gold assumption against audio.

The gold and the guard share one load-bearing axiom: a proposal whose de-stuttered
skeleton equals the original's skeleton (可以可以 -> 可以, 嗯嗯是是 -> 是) is
*harmless noise removal* and is always KEEP. assemble_gold tags 569 rows this way
(gold_source="destutter") and the guard accepts them up front — so on these rows
guard == gold by construction. That makes the axiom the single biggest piece of
circular gold (it drives 399 of the 518 "recovered" rows).

An assertion that load-bearing should be VERIFIED, not just stated. This samples
destutter→keep rows, cuts the real audio slice for each, and builds the same
interactive page audio_verify uses, so the user can listen and confirm each is
genuinely stutter/filler and not a real deletion. If the user rules 0/N are real
deletions, the axiom graduates from "asserted" to "audio-verified"; any real
deletion found is a counterexample that breaks _is_destutter_only and must be
fixed in the guard.

Audio + page land under evals/local/verify_destutter/ (git-ignored). Run:
    uv run python -m evals.verify_destutter_audio            # default 40
    uv run python -m evals.verify_destutter_audio --sample 60 --seed 7
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from app.lexicon_store import list_lexicon_correction_rules
from app.transcript_corrections import _destutter, _is_destutter_only

from evals._log import log
from evals.audio_verify import GOLD, build, cut_clip, sidecar_times

OUTDIR = Path(__file__).resolve().parent / "local" / "verify_destutter"
TITLE = "destutter→keep 抽检"
INTRO = "（gold/guard 都断言这些只是去口吃，确认没删实义）"


def _seeded_sample(rows: list[dict], n: int, seed: int) -> list[dict]:
    """Deterministic sample without Math.random — index by a seeded shuffle.

    random.Random is fine in a standalone eval script (only Workflow JS forbids
    it); we keep it seeded so the same audit is reproducible.
    """
    import random

    rng = random.Random(seed)
    pool = list(rows)
    rng.shuffle(pool)
    return pool[:n]


def _note(row: dict) -> str:
    """Show WHAT collapsed, so the listener knows exactly what to check for."""
    o, p = row["original_text"], row["proposed_text"]
    skeleton = _destutter(o)
    removed = len(o) - len(p)
    return f"骨架「{skeleton}」不变，去掉 {removed} 字噪音；确认这些字是口吃/重复而非实义"


def main() -> None:
    """Sample destutter→keep gold rows, cut audio, write the verification page."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--sample", type=int, default=40)
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args()

    OUTDIR.mkdir(parents=True, exist_ok=True)
    (OUTDIR / "clips").mkdir(exist_ok=True)
    rows = [
        json.loads(line)
        for line in GOLD.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    destutter = [
        r
        for r in rows
        if r.get("gold_source") == "destutter"
        and _is_destutter_only(r["original_text"], r["proposed_text"])
    ]
    log.info("destutter_pool", total=len(destutter), sampling=args.sample)
    cases = _seeded_sample(destutter, args.sample, args.seed)

    times_cache: dict[str, dict] = {}
    cut = 0
    for i, c in enumerate(cases):
        proj = c["source"]
        times_cache.setdefault(proj, sidecar_times(proj))
        c["_note"] = _note(c)
        match = times_cache[proj].get(c["original_text"].strip())
        if not match:
            log.warning("no_timestamp", i=i, proj=proj)
            continue
        clip_name = f"{i:02d}_{proj}.mp3"
        if cut_clip(proj, match[0], match[1], OUTDIR / "clips" / clip_name):
            c["_clip"] = clip_name
            cut += 1
        else:
            log.warning("cut_failed", i=i, proj=proj)

    # Match audio_verify: replay current lexicon rules on display text so the
    # page shows what the live transcript shows (timestamp matching used raw text).
    rules = list_lexicon_correction_rules()
    from app.transcript_corrections import _apply_rules_to_text

    for c in cases:
        c["original_text"] = _apply_rules_to_text(c["original_text"], rules)
        c["proposed_text"] = _apply_rules_to_text(c["proposed_text"], rules)

    page = build(cases, title=TITLE, intro=INTRO)
    (OUTDIR / "verify.html").write_text(page, encoding="utf-8")
    log.info(
        "written",
        cases=len(cases),
        clips=cut,
        page=str(OUTDIR / "verify.html"),
        hint="听完导出 JSON，0 条真删则 destutter 公理通过音频验证",
    )


if __name__ == "__main__":
    main()
