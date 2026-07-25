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


def overlapped_sample_ids(
    samples: Iterable[VoiceprintSampleRow],
    projects_dir: Path | None,
) -> frozenset[int] | None:
    """
    Return the ids of samples whose source audio overlaps another speaker.

    Args:
        samples: Sample rows to check, from any number of projects.
        projects_dir: Projects parent directory, or None when unavailable.

    Returns:
        Ids of the overlapping samples, or None when the check could not run.
        None and an empty set mean different things and callers must keep them
        apart: "not checked" must never be presented as "checked and clean".
    """
    if projects_dir is None:
        return None
    by_project: dict[str, list[VoiceprintSampleRow]] = defaultdict(list)
    for row in samples:
        by_project[row.project_id].append(row)
    flagged: set[int] = set()
    for project_id, rows in by_project.items():
        segments = _project_segments(projects_dir / project_id)
        if segments is None:
            continue
        for row in rows:
            if has_overlap_risk(_probe_segment(row), segments):
                flagged.add(row.sample_id)
    return frozenset(flagged)


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
