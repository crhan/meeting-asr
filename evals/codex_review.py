"""Second-pass review of Polish eval labels by codex (an independent model).

The user's rule: every eval-set label must be cross-checked by codex so my
single-sided judgement cannot silently corrupt the gold standard. This driver
batches the extracted cases, asks codex (GPT-5) to independently decide keep vs
reject for each, and records where codex DISAGREES with my initial label. The
disagreements go to the human (Linus) for the final call.

Designed for a long run: batches are reviewed CONCURRENTLY, each finished batch
is appended to the output immediately (so a crash loses nothing), and a re-run
SKIPS already-reviewed ids (resume). codex needs network, so run with the
sandbox disabled:

    uv run --no-project python evals/codex_review.py --limit 60   # validate first
    uv run --no-project python evals/codex_review.py              # full / resume
"""

from __future__ import annotations

import argparse
import json
import subprocess
import tempfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

LOCAL = Path(__file__).resolve().parent / "local"
REVIEWED_OUT = LOCAL / "polish_reviewed.jsonl"
DISAGREE_OUT = LOCAL / "polish_disagreements.jsonl"
BATCH_SIZE = 30
WORKERS = 8
CODEX_TIMEOUT_S = 420

OUTPUT_SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "required": ["results"],
    "properties": {
        "results": {
            "type": "array",
            "items": {
                "type": "object",
                "additionalProperties": False,
                "required": ["id", "verdict", "reason"],
                "properties": {
                    "id": {"type": "string"},
                    "verdict": {"type": "string", "enum": ["keep", "reject"]},
                    "reason": {"type": "string"},
                },
            },
        }
    },
}

PROMPT_HEAD = """\
你是中文会议转写 Polish（文本润色）质量审查专家，正在独立复核另一个系统的标注，\
目标是抓出它的误判，不要附和。

Polish 的目标：清理 ASR 噪声（口吃重复、语气词、重启片段、强调重复）、修正错别字/\
术语/大小写。铁律：
- 不得改变语义（尤其插入否定如「我就说→我就不说」属严重错误）
- 不得编造原文没有的内容（数字、版本号、人名、卡单号）
- 不得从前句/后句搬运内容到本句
- 不得删除实质信息；态度/确认/决策词（我觉得/可能/对吧/同意 等）不能删，
  但删除「重复出现」的同一个词（去口吃，如 可以可以→可以）不算删语义，应采纳
- 英文连写拆分（casebycase→case by case）、中文数字转写（O三→O3）属合法修正

下面每条含：原文 original、提议 proposed、变更类型 change_type、前句 prev、后句 next。
对每条独立判断 proposed 相对 original 该不该被采纳：
- "keep"：正确清理噪声/修对错别字，语义不变 → 应采纳
- "reject"：改变语义/编造/搬运邻句/删实质信息/过度改写 → 应拒绝

只输出 JSON 对象 {"results":[{"id","verdict":"keep"|"reject","reason":"≤25字中文理由"}]}。
"""


def load_cases(limit: int | None) -> list[dict]:
    """Load reject (full) + accept (sample) cases, tagging each with its kind."""
    rows: list[dict] = []
    for kind, name in (
        ("reject", "polish_reject_cases.jsonl"),
        ("accept", "polish_accept_sample.jsonl"),
    ):
        for line in (LOCAL / name).read_text(encoding="utf-8").splitlines():
            if line.strip():
                row = json.loads(line)
                row["_kind"] = kind
                rows.append(row)
    return rows[:limit] if limit else rows


def load_done_ids() -> set[str]:
    """Return ids already present in the output, so a re-run resumes."""
    if not REVIEWED_OUT.exists():
        return set()
    return {
        json.loads(line)["id"]
        for line in REVIEWED_OUT.read_text(encoding="utf-8").splitlines()
        if line.strip()
    }


def build_prompt(batch: list[dict]) -> str:
    """Render one batch of cases into the codex prompt (neighbours truncated)."""
    items = [
        {
            "id": row["id"],
            "original": row["original_text"],
            "proposed": row["proposed_text"],
            "change_type": row["change_type"],
            "prev": row["previous_text"][:40],
            "next": row["next_text"][:40],
        }
        for row in batch
    ]
    return PROMPT_HEAD + "\n\ncases:\n" + json.dumps(items, ensure_ascii=False)


def call_codex(prompt: str) -> dict[str, dict]:
    """Invoke codex on one batch; return {id: {verdict, reason}}."""
    with tempfile.TemporaryDirectory() as tmp:
        schema_path = Path(tmp) / "schema.json"
        out_path = Path(tmp) / "out.json"
        schema_path.write_text(json.dumps(OUTPUT_SCHEMA), encoding="utf-8")
        proc = subprocess.run(
            [
                "codex", "exec", "--skip-git-repo-check", "--ephemeral",
                "-o", str(out_path), "--output-schema", str(schema_path), "-",
            ],
            input=prompt,
            text=True,
            capture_output=True,
            timeout=CODEX_TIMEOUT_S,
        )
        if not out_path.exists():
            raise RuntimeError(f"codex produced no output: {proc.stderr[-300:]}")
        results = json.loads(out_path.read_text(encoding="utf-8"))["results"]
    return {item["id"]: item for item in results}


def review_batch(batch: list[dict]) -> list[dict]:
    """Review one batch (one retry on failure); return rows with codex verdicts."""
    last_exc: Exception | None = None
    for _ in range(2):
        try:
            verdicts = call_codex(build_prompt(batch))
            reviewed = []
            for row in batch:
                verdict = verdicts.get(row["id"])
                if verdict is None:
                    continue  # missing id: leave unreviewed so resume retries it
                row["codex_verdict"] = verdict["verdict"]
                row["codex_reason"] = verdict["reason"]
                reviewed.append(row)
            return reviewed
        except (subprocess.TimeoutExpired, RuntimeError, json.JSONDecodeError) as exc:
            last_exc = exc
    raise last_exc if last_exc else RuntimeError("unknown batch failure")


def expected_verdict(initial_label: str) -> str | None:
    """Map my initial label to the verdict it implies (None for needs_review)."""
    return {"should_keep": "keep", "should_reject": "reject"}.get(initial_label)


def main() -> None:
    """Concurrently review all pending cases, appending each finished batch."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()

    cases = load_cases(args.limit)
    done = load_done_ids()
    pending = [case for case in cases if case["id"] not in done]
    batches = [pending[i : i + BATCH_SIZE] for i in range(0, len(pending), BATCH_SIZE)]
    print(f"待审 {len(pending)} 条（已审 {len(done)}），{len(batches)} 批，{WORKERS} 并发")

    written = failed = 0
    with REVIEWED_OUT.open("a", encoding="utf-8") as out:
        with ThreadPoolExecutor(max_workers=WORKERS) as executor:
            futures = [executor.submit(review_batch, batch) for batch in batches]
            for future in as_completed(futures):
                try:
                    rows = future.result()
                except Exception as exc:  # noqa: BLE001 - log and continue/resume
                    failed += 1
                    print(f"  批失败（将于续跑重试）: {exc}")
                    continue
                for row in rows:
                    out.write(json.dumps(row, ensure_ascii=False) + "\n")
                out.flush()
                written += len(rows)
                print(f"  进度 {written}/{len(pending)}")

    _finalize(failed)


def _finalize(failed_batches: int) -> None:
    """Recompute disagreements over the full reviewed file and print stats."""
    reviewed = [
        json.loads(line)
        for line in REVIEWED_OUT.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    disagreements = []
    agree = needs_review = 0
    for row in reviewed:
        expect = expected_verdict(row["initial_label"])
        if expect is None:
            needs_review += 1
            disagreements.append(row)
        elif row.get("codex_verdict") == expect:
            agree += 1
        else:
            disagreements.append(row)

    with DISAGREE_OUT.open("w", encoding="utf-8") as handle:
        for row in disagreements:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")

    print("\n" + "=" * 60)
    print(f"已复核: {len(reviewed)}   本次失败批次: {failed_batches}")
    print(f"  我有初判且 codex 一致: {agree}")
    print(f"  needs_review 由 codex 定夺: {needs_review}")
    print(f"  分歧+待定 (上报人工): {len(disagreements)} -> {DISAGREE_OUT.name}")
    print("=" * 60)


if __name__ == "__main__":
    main()
