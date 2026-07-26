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

from app.core.project_refs import list_projects
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
    dirs = _project_dirs(projects_dir)
    overlap: dict[int, bool] = {}
    available: dict[str, bool] = {}
    for project_id, rows in by_project.items():
        segments = _review_segments(dirs.get(project_id))
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
    dirs = _project_dirs(projects_dir)
    return {
        project_id: _review_segments(dirs.get(project_id)) is not None
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


def _project_dirs(projects_dir: Path) -> dict[str, Path]:
    """Map project id to directory the way a project reference is resolved.

    Emphatically *not* ``projects_dir / project_id``. A project created with an
    explicit ``--project-dir`` keeps the directory name it was given while its
    content-addressed id lives in the manifest, and every ref resolver finds it
    by scanning manifests instead. Probing the id as a path would report such a
    project deleted while its review page opens perfectly well -- and here that
    mistake is not silent: it strips a working link and tells library health to
    warn that the person needs recapturing.

    ``restrict_to_projects_dir`` matches what the web resolves refs with, so a
    project the review route refuses to open is not advertised as reachable.
    """
    try:
        listing = list_projects(projects_dir, restrict_to_projects_dir=True)
    except OSError:
        return {}
    return {item.project_id: item.project_dir for item in listing.projects}


def _review_segments(project_dir: Path | None) -> list[SentenceSegment] | None:
    """Load the transcript a human would review, or None when there is none.

    Mirrors what the speaker-review session loads, for two reasons that pull
    the same way. Availability has to match it or the UI links to a page that
    cannot open (and hides one that can): the corrected transcript wins when
    present, and review needs at least one speaker-attributed line.

    Overlap wants the same file for a different reason -- the question is who
    else was talking next to this clip, and a reassignment made during review
    is precisely the more accurate answer. Low-information lines are kept for
    the same reason: a neighbour's "嗯嗯对" is filtered out of the transcript
    but was still recorded into the clip, and dropping it would hide exactly
    the crosstalk this check exists to find.
    """
    if project_dir is None:
        return None
    asr_dir = project_dir / "asr"
    corrected = asr_dir / "sentences_corrected.json"
    path = corrected if corrected.is_file() else asr_dir / "sentences.json"
    if not path.is_file():
        return None
    try:
        segments = load_transcript_result(path, include_low_information=True).sentences
    except Exception:  # noqa: BLE001 - one unreadable project must not fail the report
        return None
    # Review refuses a transcript with no speaker-attributed line, so this is
    # part of "can it be opened". It only gates the answer -- every segment is
    # still returned, because a stretch of speech nobody was attributed to was
    # recorded into the clip just the same.
    if not any(item.speaker_id is not None and item.text.strip() for item in segments):
        return None
    return segments
