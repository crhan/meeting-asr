"""Measure the summary pipeline before/after the single-call refactor.

For each real project under PROJECTS_DIR this script:

1. Runs the OLD logic — two LLM calls with a 24,000-char head+tail truncation
   for the memory step, then a separate title call driven only by the
   distilled summary text. This replicates the pre-refactor behavior locally
   so we can compare apples to apples.
2. Runs the NEW logic via ``app.meeting_summary.generate_meeting_summary``,
   which is one LLM call seeing the entire transcript.

It then prints per-project metrics: number of LLM calls, total input
characters sent across all calls, wall time, and the actual title /
summary / keywords each variant produced. Output is purely read-only — no
project artifacts are modified.
"""

from __future__ import annotations

import json
import re
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from app.config import load_settings  # noqa: E402
from app.meeting_summary import (  # noqa: E402
    _call_generation,
    _configure_dashscope,
    _load_summary_json,
    generate_meeting_summary,
)
from app.models import SentenceSegment, TranscriptResult  # noqa: E402
from app.postprocess import render_speaker_text  # noqa: E402

PROJECTS_DIR = Path.home() / ".local/share/meeting-asr/projects"

# Old constants reproduced verbatim so we can measure the legacy path.
OLD_MAX_SUMMARY_TRANSCRIPT_CHARS = 24_000


@dataclass
class CallStat:
    """One LLM round-trip's vital stats."""

    input_chars: int
    wall_ms: int


def _old_truncate(transcript: str) -> str:
    """Replicate the pre-refactor head+tail cut at 24k chars."""
    if len(transcript) <= OLD_MAX_SUMMARY_TRANSCRIPT_CHARS:
        return transcript
    half = OLD_MAX_SUMMARY_TRANSCRIPT_CHARS // 2
    return transcript[:half] + "\n\n[中间内容过长，已截断]\n\n" + transcript[-half:]


def _old_memory_prompt(result: TranscriptResult) -> str:
    """Reproduce the legacy memory-step prompt (no transcript wrapper marker)."""
    transcript = render_speaker_text(result).strip() or result.full_text.strip()
    transcript = _old_truncate(transcript)
    return (
        "请根据下面的会议转写生成一个很短的回忆索引。\n"
        "要求：\n"
        "1. summary 只写 1 到 2 句话，目标是让人快速想起这是哪一场会议。\n"
        "2. 不要写正式纪要，不要写待办事项，不要扩展结论。\n"
        "3. topics 是 3 到 6 个短关键词或场景词，用于检索和回忆。\n"
        "4. 只返回 JSON，字段为 summary, topics。\n\n"
        f"会议转写：\n{transcript}"
    )


def _old_title_prompt(memory_summary: str, topics: list[str]) -> str:
    """Reproduce the legacy second-call title prompt."""
    topics_text = "、".join(topics) if topics else "无"
    return (
        "请根据下面的会议回忆索引生成一个短标题。\n"
        "要求：\n"
        "1. title 使用 8 到 28 个中文字符，概括这场会议，方便在项目列表里识别。\n"
        "2. 不要写日期，不要写“会议总结”，不要写待办。\n"
        "3. 只返回 JSON，字段为 title。\n\n"
        f"回忆提示：{memory_summary}\n"
        f"关键词：{topics_text}"
    )


def _run_old(
    result: TranscriptResult, settings
) -> tuple[list[CallStat], dict[str, Any]]:
    """Run the legacy two-call summary pipeline and capture metrics."""
    calls: list[CallStat] = []
    memory_prompt = _old_memory_prompt(result)
    start = time.perf_counter()
    raw_memory = _call_generation(
        model=settings.dashscope_summary_model,
        settings=settings,
        system_prompt="你是会议回忆索引助手。只输出 JSON，不要输出 Markdown，不要解释。",
        prompt=memory_prompt,
    )
    calls.append(
        CallStat(len(memory_prompt), int((time.perf_counter() - start) * 1000))
    )

    try:
        memory_payload = _load_summary_json(raw_memory)
    except RuntimeError:
        memory_payload = {}
    memory_summary = str(memory_payload.get("summary") or "").strip()
    topics_raw = memory_payload.get("topics") or []
    topics = [str(item).strip() for item in topics_raw if str(item).strip()]

    title_prompt = _old_title_prompt(memory_summary, topics)
    start = time.perf_counter()
    raw_title = _call_generation(
        model=settings.dashscope_summary_model,
        settings=settings,
        system_prompt="你是会议标题助手。只输出 JSON，不要输出 Markdown，不要解释。",
        prompt=title_prompt,
    )
    calls.append(CallStat(len(title_prompt), int((time.perf_counter() - start) * 1000)))

    title = ""
    try:
        title_payload = _load_summary_json(raw_title)
        title = str(title_payload.get("title") or "").strip()
    except RuntimeError:
        title = raw_title.strip()
    title = re.sub(r"\s+", " ", title).strip(" #")[:80]

    return calls, {"title": title, "summary": memory_summary, "topics": topics}


def _run_new(
    result: TranscriptResult, settings
) -> tuple[list[CallStat], dict[str, Any]]:
    """Run the refactored single-call summary pipeline and capture metrics."""
    transcript = render_speaker_text(result).strip() or result.full_text.strip()
    # The new prompt sends the full transcript; we count what would actually flow.
    from app.meeting_summary import (
        _build_memory_prompt,
    )  # local import keeps measurement explicit

    prompt = _build_memory_prompt(result)
    start = time.perf_counter()
    summary = generate_meeting_summary(result, settings=settings, model=None)
    elapsed = int((time.perf_counter() - start) * 1000)
    stats = [CallStat(len(prompt), elapsed)]
    return stats, {
        "title": summary.title,
        "summary": summary.summary,
        "keywords": summary.keywords,
        "transcript_chars": len(transcript),
    }


def _load_transcript(project_dir: Path) -> TranscriptResult | None:
    """Load a TranscriptResult from a project's sentences.json."""
    path = project_dir / "asr" / "sentences.json"
    if not path.exists():
        return None
    payload = json.loads(path.read_text(encoding="utf-8"))
    sentences = []
    for entry in payload.get("sentences") or []:
        sentences.append(
            SentenceSegment(
                begin_time_ms=int(entry.get("begin_time_ms") or 0),
                end_time_ms=int(entry.get("end_time_ms") or 0),
                text=str(entry.get("text") or ""),
                speaker_id=int(entry.get("speaker_id") or 0),
                sentence_id=int(entry.get("sentence_id") or 0),
            )
        )
    detected = [int(item) for item in payload.get("detected_speakers") or []]
    full_text = str(payload.get("full_text") or "")
    return TranscriptResult(
        full_text=full_text, sentences=sentences, detected_speakers=detected
    )


def _iter_projects(projects_dir: Path, limit: int | None = None) -> list[Path]:
    """Return project directories newest-first up to an optional limit."""
    paths = [
        p
        for p in projects_dir.iterdir()
        if p.is_dir() and (p / "project.json").exists()
    ]

    def sort_key(path: Path) -> str:
        try:
            payload = json.loads((path / "project.json").read_text(encoding="utf-8"))
        except Exception:
            return ""
        source = payload.get("source") or {}
        return str(payload.get("meeting_time") or source.get("meeting_time") or "")

    paths.sort(key=sort_key, reverse=True)
    return paths if limit is None else paths[:limit]


def main() -> int:
    """Run both pipelines side-by-side and print a comparison report."""
    limit = int(sys.argv[1]) if len(sys.argv) > 1 else 5
    settings = load_settings(require_oss=False)
    _configure_dashscope(settings)

    rows = []
    for project_dir in _iter_projects(PROJECTS_DIR, limit=limit):
        project_id = project_dir.name
        result = _load_transcript(project_dir)
        if result is None:
            continue

        old_calls, old_out = _run_old(result, settings)
        new_calls, new_out = _run_new(result, settings)

        rows.append(
            {
                "project_id": project_id,
                "old_calls": old_calls,
                "old_out": old_out,
                "new_calls": new_calls,
                "new_out": new_out,
            }
        )

        print(f"\n=== {project_id} ===")
        print(
            f"  OLD calls={len(old_calls)} "
            f"input_chars={sum(c.input_chars for c in old_calls)} "
            f"wall_ms={sum(c.wall_ms for c in old_calls)}"
        )
        print(f"      title    : {old_out['title']}")
        print(f"      summary  : {old_out['summary']}")
        print(f"      topics   : {old_out['topics']}")
        print(
            f"  NEW calls={len(new_calls)} "
            f"input_chars={sum(c.input_chars for c in new_calls)} "
            f"wall_ms={sum(c.wall_ms for c in new_calls)} "
            f"transcript_chars={new_out['transcript_chars']}"
        )
        print(f"      title    : {new_out['title']}")
        print(f"      summary  : {new_out['summary']}")
        print(f"      keywords : {new_out['keywords']}")

    print("\n=== Aggregate ===")
    total_old_calls = sum(len(r["old_calls"]) for r in rows)
    total_new_calls = sum(len(r["new_calls"]) for r in rows)
    total_old_chars = sum(sum(c.input_chars for c in r["old_calls"]) for r in rows)
    total_new_chars = sum(sum(c.input_chars for c in r["new_calls"]) for r in rows)
    total_old_ms = sum(sum(c.wall_ms for c in r["old_calls"]) for r in rows)
    total_new_ms = sum(sum(c.wall_ms for c in r["new_calls"]) for r in rows)
    print(f"  projects  : {len(rows)}")
    print(
        f"  calls     : OLD={total_old_calls}  NEW={total_new_calls}  Δ={total_new_calls - total_old_calls}"
    )
    print(
        f"  input chr : OLD={total_old_chars:,}  NEW={total_new_chars:,}  "
        f"Δ={total_new_chars - total_old_chars:+,}"
    )
    print(
        f"  wall ms   : OLD={total_old_ms:,}  NEW={total_new_ms:,}  "
        f"Δ={total_new_ms - total_old_ms:+,}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
