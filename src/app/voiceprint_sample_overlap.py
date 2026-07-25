"""Which stored samples were recorded while someone else was talking.

A voiceprint is only as clean as the audio behind it, and the store records
nothing about how crowded that audio was. The answer lives in the source
project's transcript, so this joins back to it.

The unit here is the *sample*, deliberately. Reporting only a per-person count
("3 samples are contaminated") names a problem without pointing at anything the
operator can act on -- they open the person's sample list and every row looks
identical. Flagging the rows is what makes the finding fixable.

Read-only; nothing here mutates the store or the projects.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterable
from pathlib import Path

from app.models import SentenceSegment
from app.speaker_labeling import load_transcript_result
from app.voiceprint_models import VoiceprintSampleRow
from app.voiceprint_segment_selection import has_overlap_risk


def check_sample_overlap(
    samples: Iterable[VoiceprintSampleRow],
    projects_dir: Path | None,
) -> dict[int, bool]:
    """
    Return per-sample overlap verdicts for the samples that could be checked.

    The result is deliberately *partial*. A sample is absent when its source
    transcript could not be read -- the project was deleted, is unreadable, or
    lives outside ``projects_dir`` because it was created with an explicit
    ``--project-dir``. Absent means "unknown", and callers must carry that
    through instead of defaulting it to False: rendering an unchecked sample as
    clean is the exact failure this check exists to catch, and a per-project
    read failure would otherwise silently whitewash every sample from it.

    Args:
        samples: Sample rows to check, from any number of projects.
        projects_dir: Projects parent directory, or None when unavailable.

    Returns:
        Sample id to "overlaps another speaker". Empty when no projects
        directory was given, so nothing is claimed to have been checked.
    """
    if projects_dir is None:
        return {}
    by_project: dict[str, list[VoiceprintSampleRow]] = defaultdict(list)
    for row in samples:
        by_project[row.project_id].append(row)
    verdicts: dict[int, bool] = {}
    for project_id, rows in by_project.items():
        segments = _project_segments(projects_dir / project_id)
        if segments is None:
            continue
        for row in rows:
            verdicts[row.sample_id] = has_overlap_risk(_probe_segment(row), segments)
    return verdicts


def _probe_segment(row: VoiceprintSampleRow) -> SentenceSegment:
    """Represent a stored sample as the transcript segment it was cut from."""
    return SentenceSegment(
        begin_time_ms=row.source_begin_time_ms,
        end_time_ms=row.source_end_time_ms,
        text="",
        speaker_id=row.project_speaker_id,
    )


def _project_segments(project_dir: Path) -> list[SentenceSegment] | None:
    """Load a project's normalized transcript, or None when unreadable."""
    path = project_dir / "asr" / "sentences.json"
    if not path.is_file():
        return None
    try:
        return load_transcript_result(path).sentences
    except Exception:  # noqa: BLE001 - one unreadable project must not fail the report
        return None
