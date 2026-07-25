"""Per-sample overlap detection: which stored rows carry another voice."""

from __future__ import annotations

import json
from pathlib import Path

from app.voiceprint_models import VoiceprintSampleRow
from app.voiceprint_sample_overlap import overlapped_sample_ids


def _row(sample_id: int, *, begin_ms: int, end_ms: int, project: str) -> VoiceprintSampleRow:
    """Build a stored sample row with only the fields the check reads."""
    return VoiceprintSampleRow(
        sample_id=sample_id,
        public_id=f"s{sample_id}",
        speaker_id=1,
        speaker_public_id="p1",
        speaker_name="Someone",
        project_id=project,
        project_speaker_id=0,
        clip_path=Path("/nonexistent.wav"),
        clip_rel_path="clips/nonexistent.wav",
        clip_sha256="0" * 64,
        source_begin_time_ms=begin_ms,
        source_end_time_ms=end_ms,
        transcript_text="",
    )


def _write_project(root: Path, project_id: str, sentences: list[dict]) -> None:
    """Write only the transcript file the overlap check reads."""
    asr_dir = root / project_id / "asr"
    asr_dir.mkdir(parents=True)
    (asr_dir / "sentences.json").write_text(
        json.dumps(
            {
                "full_text": "".join(item["text"] for item in sentences),
                "sentences": sentences,
                "detected_speakers": sorted({s["speaker_id"] for s in sentences}),
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )


def test_only_the_crowded_sample_is_flagged(tmp_path: Path) -> None:
    """Two samples of the same person, one clean and one over another speaker."""
    projects_dir = tmp_path / "projects"
    _write_project(
        projects_dir,
        "p-mixed",
        [
            # Clean: the next speaker is a full 30s away.
            {"begin_time_ms": 0, "end_time_ms": 8_000, "text": "甲", "speaker_id": 0},
            # Crowded: the other speaker starts 100ms after this one stops.
            {
                "begin_time_ms": 60_000,
                "end_time_ms": 68_000,
                "text": "甲",
                "speaker_id": 0,
            },
            {
                "begin_time_ms": 68_100,
                "end_time_ms": 72_000,
                "text": "乙",
                "speaker_id": 1,
            },
        ],
    )
    rows = [
        _row(1, begin_ms=0, end_ms=8_000, project="p-mixed"),
        _row(2, begin_ms=60_000, end_ms=68_000, project="p-mixed"),
    ]

    assert overlapped_sample_ids(rows, projects_dir) == frozenset({2})


def test_missing_projects_directory_returns_none_not_empty(tmp_path: Path) -> None:
    """"Not checked" and "checked and clean" must stay distinguishable.

    Both would render as a clean library, which is the exact failure this whole
    check exists to prevent, so the two answers cannot share a value.
    """
    rows = [_row(1, begin_ms=0, end_ms=8_000, project="p-missing")]

    assert overlapped_sample_ids(rows, None) is None
    # A readable projects dir with no transcript for that project is a real
    # "nothing found", which is an empty set rather than None.
    assert overlapped_sample_ids(rows, tmp_path) == frozenset()


def test_unreadable_transcript_does_not_fail_the_whole_check(tmp_path: Path) -> None:
    """One broken project must not take the other projects' answers down."""
    projects_dir = tmp_path / "projects"
    (projects_dir / "p-broken" / "asr").mkdir(parents=True)
    (projects_dir / "p-broken" / "asr" / "sentences.json").write_text(
        "{not json", encoding="utf-8"
    )
    _write_project(
        projects_dir,
        "p-ok",
        [
            {"begin_time_ms": 0, "end_time_ms": 8_000, "text": "甲", "speaker_id": 0},
            {"begin_time_ms": 8_100, "end_time_ms": 9_000, "text": "乙", "speaker_id": 1},
        ],
    )
    rows = [
        _row(1, begin_ms=0, end_ms=8_000, project="p-broken"),
        _row(2, begin_ms=0, end_ms=8_000, project="p-ok"),
    ]

    assert overlapped_sample_ids(rows, projects_dir) == frozenset({2})
