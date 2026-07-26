"""What a stored sample's source project can still tell us about it.

A voiceprint is only as clean as the audio behind it, and the store records
nothing about how crowded that audio was. The answer lives in the source
project's transcript, so this joins back to it -- and the same join answers a
second question the store cannot: whether that project is still there at all.
Clips live in the library and survive a project's deletion, so a sample keeps
matching long after the recording it was cut from is gone; what dies with the
project is the ability to hear the sample in context or to check it for
overlap.

The unit here is the *sample*, deliberately. Reporting only a per-person count
("3 samples are contaminated") names a problem without pointing at anything the
operator can act on -- they open the person's sample list and every row looks
identical. Flagging the rows is what makes the finding fixable.

Read-only; nothing here mutates the store or the projects.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

from app.models import SentenceSegment
from app.speaker_labeling import load_transcript_result
from app.voiceprint_models import VoiceprintSampleRow
from app.voiceprint_segment_selection import has_overlap_risk


@dataclass(frozen=True, slots=True)
class SampleSourceFacts:
    """Everything one pass over the source projects established.

    Both maps are *partial* in the same way and for the same reason: an absent
    key means "not checked", never "checked and fine". Callers must carry that
    distinction to their own output.
    """

    overlap: dict[int, bool]
    """Sample id to "another speaker holds the floor next to this clip"."""

    available: dict[str, bool]
    """Project id to "still openable for review"."""


def inspect_sample_sources(
    samples: Iterable[VoiceprintSampleRow],
    projects_dir: Path | None,
) -> SampleSourceFacts:
    """
    Read each sample's source project once and report what it yielded.

    Callers that need both facts must come through here rather than calling the
    two narrow helpers in turn, which would parse every transcript twice.

    Args:
        samples: Sample rows to check, from any number of projects.
        projects_dir: Projects parent directory, or None when unavailable.

    Returns:
        Overlap verdicts and per-project availability. Both empty when no
        projects directory was given, so nothing is claimed to have been
        checked.
    """
    if projects_dir is None:
        return SampleSourceFacts(overlap={}, available={})
    by_project: dict[str, list[VoiceprintSampleRow]] = defaultdict(list)
    for row in samples:
        by_project[row.project_id].append(row)
    overlap: dict[int, bool] = {}
    available: dict[str, bool] = {}
    for project_id, rows in by_project.items():
        segments = _project_segments(projects_dir / project_id)
        available[project_id] = segments is not None
        if segments is None:
            continue
        for row in rows:
            overlap[row.sample_id] = has_overlap_risk(_probe_segment(row), segments)
    return SampleSourceFacts(overlap=overlap, available=available)


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
    return inspect_sample_sources(samples, projects_dir).overlap


def check_project_sources(
    project_ids: Iterable[str],
    projects_dir: Path | None,
) -> dict[str, bool]:
    """
    Return whether each project can still be opened for speaker review.

    The rule is deliberately the same one the review page itself lives by --
    a normalized transcript under ``projects_dir`` -- so a caller can decide
    whether linking there will land on the page or on its "nothing to review"
    error.

    Args:
        project_ids: Project ids to check.
        projects_dir: Projects parent directory, or None when unavailable.

    Returns:
        Project id to availability. Empty when no projects directory was given.
    """
    if projects_dir is None:
        return {}
    return {
        project_id: _project_segments(projects_dir / project_id) is not None
        for project_id in dict.fromkeys(project_ids)
    }


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
