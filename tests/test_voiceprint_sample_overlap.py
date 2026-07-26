"""Per-sample overlap detection: which stored rows carry another voice."""

from __future__ import annotations

import json
from pathlib import Path

from app.voiceprint_models import VoiceprintSampleRow
from app.voiceprint_sample_overlap import (
    check_project_sources,
    check_sample_overlap,
    inspect_sample_sources,
)


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


def _write_project(
    root: Path,
    project_id: str,
    sentences: list[dict],
    *,
    dir_name: str | None = None,
    transcript: str = "sentences.json",
) -> Path:
    """Write the project files a reference resolver and the check both need.

    ``project.json`` is not optional decoration: project ids are resolved by
    scanning manifests, so a directory without one is invisible to every ref
    resolver -- including the review route this check has to agree with.
    """
    project_dir = root / (dir_name or project_id)
    asr_dir = project_dir / "asr"
    asr_dir.mkdir(parents=True)
    created_at = "2026-05-11T12:00:00+08:00"
    (project_dir / "project.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "project_id": project_id,
                "title": "会议",
                "created_at": created_at,
                "updated_at": created_at,
                "status": "corrected",
                "source": {
                    "path": f"source/{project_id}.mp3",
                    "filename": f"{project_id}.mp3",
                    "size_bytes": 1,
                    "mtime": created_at,
                    "meeting_time": created_at,
                },
                "audio": {"duration_seconds": 3600.0},
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    (asr_dir / transcript).write_text(
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
    return project_dir


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

    assert check_sample_overlap(rows, projects_dir) == {1: False, 2: True}


def test_unchecked_samples_are_absent_rather_than_clean(tmp_path: Path) -> None:
    """"Not checked" and "checked and clean" must stay distinguishable.

    Both render as a sample with no warning, which is the exact failure this
    whole check exists to prevent, so the two answers cannot share a value.
    A verdict is therefore only recorded for samples actually examined.
    """
    rows = [_row(1, begin_ms=0, end_ms=8_000, project="p-missing")]

    assert check_sample_overlap(rows, None) == {}
    # A readable projects dir that holds no transcript for this project is
    # still "unknown", not "clean": the sample was never examined.
    assert check_sample_overlap(rows, tmp_path) == {}


def test_unreadable_transcript_does_not_fail_the_whole_check(tmp_path: Path) -> None:
    """One broken project must not take the other projects' answers down."""
    projects_dir = tmp_path / "projects"
    broken = _write_project(
        projects_dir,
        "p-broken",
        [{"begin_time_ms": 0, "end_time_ms": 8_000, "text": "甲", "speaker_id": 0}],
    )
    # A resolvable project whose transcript is corrupt -- not merely a directory
    # the resolver never sees.
    (broken / "asr" / "sentences.json").write_text("{not json", encoding="utf-8")
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

    verdicts = check_sample_overlap(rows, projects_dir)

    # The readable project answers for its sample; the broken one stays absent
    # instead of being reported clean.
    assert verdicts == {2: True}
    assert 1 not in verdicts


def test_deleted_source_project_is_reported_as_gone(tmp_path: Path) -> None:
    """A sample outlives its project, and the UI has to be able to say so.

    Clips live in the library, so deleting a project leaves its samples
    matching normally -- but the review page has nothing left to show, and
    linking there lands on its "nothing to review" error.
    """
    projects_dir = tmp_path / "projects"
    _write_project(
        projects_dir,
        "p-ok",
        [{"begin_time_ms": 0, "end_time_ms": 8_000, "text": "甲", "speaker_id": 0}],
    )
    rows = [
        _row(1, begin_ms=0, end_ms=8_000, project="p-ok"),
        _row(2, begin_ms=0, end_ms=8_000, project="p-deleted"),
    ]

    facts = inspect_sample_sources(rows, projects_dir)

    assert facts.available == {"p-ok": True, "p-deleted": False}
    # The gone project's sample has no overlap verdict either -- and the two
    # answers are carried separately so a caller can say *why* it is unchecked.
    assert facts.overlap == {1: False}


def test_availability_is_unknown_without_a_projects_dir(tmp_path: Path) -> None:
    """No projects dir means nothing was established, not "everything is gone".

    Reporting False here would put a "source deleted" marker on every sample in
    the library the moment the projects directory is unavailable.
    """
    rows = [_row(1, begin_ms=0, end_ms=8_000, project="p-any")]

    assert inspect_sample_sources(rows, None).available == {}
    assert check_project_sources(["p-any"], None) == {}


def test_project_sources_answer_without_samples_in_hand(tmp_path: Path) -> None:
    """The quality report knows project ids, not sample rows."""
    projects_dir = tmp_path / "projects"
    _write_project(
        projects_dir,
        "p-ok",
        [{"begin_time_ms": 0, "end_time_ms": 8_000, "text": "甲", "speaker_id": 0}],
    )

    assert check_project_sources(["p-ok", "p-gone", "p-ok"], projects_dir) == {
        "p-ok": True,
        "p-gone": False,
    }


def test_a_project_whose_directory_is_not_its_id_is_still_reachable(
    tmp_path: Path,
) -> None:
    """Project ids are resolved through manifests, not by joining a path.

    A project created with an explicit ``--project-dir`` keeps the directory
    name it was given while its content-addressed id lives in the manifest.
    Probing ``projects_dir / project_id`` would call it deleted, strip a
    working review link, and have library health demand a recapture that
    nothing is wrong with.
    """
    projects_dir = tmp_path / "projects"
    _write_project(
        projects_dir,
        "p-abc",
        [{"begin_time_ms": 0, "end_time_ms": 8_000, "text": "甲", "speaker_id": 0}],
        dir_name="my-custom-name",
    )
    rows = [_row(1, begin_ms=0, end_ms=8_000, project="p-abc")]

    assert inspect_sample_sources(rows, projects_dir).available == {"p-abc": True}
    assert check_project_sources(["p-abc"], projects_dir) == {"p-abc": True}


def test_a_corrected_transcript_alone_still_opens_for_review(tmp_path: Path) -> None:
    """Review prefers the corrected transcript, so availability must too.

    Reading only the raw file would mark a fully reviewable project deleted the
    moment its raw transcript is missing.
    """
    projects_dir = tmp_path / "projects"
    _write_project(
        projects_dir,
        "p-corrected",
        [
            {"begin_time_ms": 0, "end_time_ms": 8_000, "text": "甲", "speaker_id": 0},
            {"begin_time_ms": 8_100, "end_time_ms": 9_000, "text": "乙", "speaker_id": 1},
        ],
        transcript="sentences_corrected.json",
    )
    rows = [_row(1, begin_ms=0, end_ms=8_000, project="p-corrected")]

    facts = inspect_sample_sources(rows, projects_dir)

    assert facts.available == {"p-corrected": True}
    # And the overlap answer comes from that same corrected view, which is the
    # one whose speaker attribution a human actually confirmed.
    assert facts.overlap == {1: True}


def test_a_transcript_with_no_speaker_lines_cannot_be_reviewed(tmp_path: Path) -> None:
    """Review refuses a transcript with nothing attributed to a speaker.

    The file parses, so a "does it load" test would call this reachable and the
    UI would link to a page that raises.
    """
    projects_dir = tmp_path / "projects"
    _write_project(
        projects_dir,
        "p-empty",
        [{"begin_time_ms": 0, "end_time_ms": 8_000, "text": "", "speaker_id": None}],
    )

    assert check_project_sources(["p-empty"], projects_dir) == {"p-empty": False}
