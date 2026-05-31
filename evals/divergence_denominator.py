"""Honest denominators for the qwen3.6 vs qwen3.7 quality comparison.

The headline "3.7 胜 63%" is NOT a global quality win rate. It is computed by
codex_quality_judge over a SAMPLE of the DIVERGENCE buckets that are quality-
relevant. This script re-derives, from the already-collected artifacts (no model
or codex calls), exactly what that 63% is a fraction OF:

  * total sentences compared            (model_compare_run full_compare_run.log)
  * divergence sentences                (model_compare_<chal>.jsonl row count)
  * per-bucket divergence counts        (the 'kind' field)
  * which buckets the quality judge sampled, and at what coverage
  * the quality win rate WITH its true denominator spelled out

It also surfaces the asymmetry between the two opposite-direction buckets
(base_no_change__chal_accept vs base_accept__chal_no_change): if the judge
favors whichever model edited MORE, the sampling ratio between these buckets
biases the headline. Run:

    uv run python -m evals.divergence_denominator
"""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

LOCAL = Path(__file__).resolve().parent / "local"
DIVERGE = LOCAL / "model_compare_qwen3_7-max.jsonl"
VERDICTS = LOCAL / "codex_quality_verdicts.jsonl"

# Buckets the quality judge treats as quality-relevant (codex_quality_judge.SAMPLE).
# The rest (one_changed, both_reject_differ, base_*_reject, ...) are guard-relevant
# or trivial and are NOT quality-judged, so they are outside the 63% denominator.
QUALITY_BUCKETS = (
    "both_accept_differ",
    "base_no_change__chal_accept",
    "base_accept__chal_no_change",
)
# Opposite-direction pair: a judge that prefers the model which edited MORE will
# systematically split these two in opposite directions; their sampling ratio then
# tilts the headline. We report them side by side.
DIRECTIONAL_PAIR = ("base_no_change__chal_accept", "base_accept__chal_no_change")


def _load_jsonl(path: Path) -> list[dict]:
    """Load a JSONL file into a list of dicts (empty if absent)."""
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def main() -> None:
    """Re-derive the honest denominator behind the quality win rate."""
    diverge = _load_jsonl(DIVERGE)
    verdicts = _load_jsonl(VERDICTS)
    buckets = Counter(row.get("kind", "?") for row in diverge)
    divergence_n = len(diverge)

    print("=" * 70)
    print("分母透明化:  qwen3.6-plus vs qwen3.7-max 质量胜率到底是谁的几分之几")
    print("=" * 70)
    # total_n is recorded only in the run log; surface it if the file is around.
    total_n = _scrape_total_n()
    if total_n:
        print(f"全量对比句数 (total)      {total_n:6d}")
        print(f"两模型分歧句数 (diverge)  {divergence_n:6d}  "
              f"({divergence_n / total_n * 100:.1f}% 的句子两模型结果不同)")
        print(f"两模型一致句数            {total_n - divergence_n:6d}  "
              f"(质量相同,不进任何胜率计算)")
    else:
        print(f"两模型分歧句数 (diverge)  {divergence_n:6d}  "
              "(全量 total 见 full_compare_run.log;此处文件缺失)")

    print("\n[分歧按桶分布]")
    quality_div = 0
    for kind, n in buckets.most_common():
        tag = "  ← 质量裁判覆盖" if kind in QUALITY_BUCKETS else ""
        if kind in QUALITY_BUCKETS:
            quality_div += n
        print(f"  {kind:32s}{n:6d}{tag}")

    print("\n[质量裁判实际覆盖]")
    judged = Counter(v.get("_kind", "?") for v in verdicts)
    judged_total = len(verdicts)
    print(f"  质量相关桶分歧合计   {quality_div:6d}")
    print(f"  codex 实判          {judged_total:6d}  "
          f"(覆盖质量相关分歧的 {judged_total / max(1, quality_div) * 100:.1f}%, "
          f"占全部分歧的 {judged_total / max(1, divergence_n) * 100:.1f}%)")
    for kind in QUALITY_BUCKETS:
        pool = buckets.get(kind, 0)
        seen = judged.get(kind, 0)
        cov = seen / pool * 100 if pool else 0.0
        print(f"    {kind:32s} 判 {seen:4d} / 池 {pool:5d}  ({cov:.1f}%)")

    _report_winrate(verdicts, judged)
    _report_directional_bias(verdicts, buckets)


def _report_winrate(verdicts: list[dict], judged: Counter) -> None:
    """Print the quality win rate WITH its true denominator stated."""
    wins = Counter(v.get("winner_model", "?") for v in verdicts)
    n = len(verdicts) or 1
    print("\n[质量胜率 — 诚实表述]")
    print(f"  分母 = 被 codex 盲判的 {len(verdicts)} 条质量相关分歧 (非全量,非全部分歧)")
    for model, c in wins.most_common():
        print(f"    {model:16s}{c:5d}  ({c / n * 100:.1f}%)")
    print('  正确读法: "在两模型分歧且属质量相关桶、且被抽中的样本里,3.7 胜 X%",'
          "\n           不是 \"3.7 在全部会议句子上质量胜 X%\"。")


def _report_directional_bias(verdicts: list[dict], buckets: Counter) -> None:
    """Surface the more-edits-wins asymmetry across the opposite-direction pair."""
    more, less = DIRECTIONAL_PAIR  # chal edited more / chal edited less
    by_bucket: dict[str, Counter] = {more: Counter(), less: Counter()}
    for v in verdicts:
        k = v.get("_kind")
        if k in by_bucket:
            by_bucket[k][v.get("winner_model", "?")] += 1
    print("\n[方向偏置自检]  两个方向相反的桶,谁多改 codex 就偏向谁?")
    for kind in (more, less):
        c = by_bucket[kind]
        chal = c.get("qwen3.7-max", 0)
        base = c.get("qwen3.6-plus", 0)
        tot = chal + base + c.get("tie", 0) or 1
        edited = "3.7 多改" if kind == more else "3.7 少改(3.6 多改)"
        print(f"  {kind:32s} [{edited}]  3.7 胜 {chal} / 3.6 胜 {base}  "
              f"(3.7 {chal / tot * 100:.0f}%)  采样池 {buckets.get(kind, 0)}")
    print("  若两行的胜方一致偏向'多改的一方',说明 codex 有 edit-more 偏好;")
    print("  此时两桶的采样配比会直接抬高或压低头条胜率,需在配比上对齐或加权。")


def _scrape_total_n() -> int:
    """Best-effort: read total_n from the full compare run log if present."""
    for name in ("full_compare_run.log", "model_compare_run.log"):
        path = LOCAL / name
        if not path.exists():
            continue
        # The summary line reads "challenger=... 实跑 NNNNN 句" or "累计 NNNNN句/...".
        import re
        text = path.read_text(encoding="utf-8")
        matches = re.findall(r"实跑\s*(\d+)\s*句", text) or re.findall(r"累计\s*(\d+)\s*句", text)
        if matches:
            return int(matches[-1])
    return 0


if __name__ == "__main__":
    main()
