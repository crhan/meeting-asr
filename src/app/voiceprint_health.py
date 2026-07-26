"""Voiceprint library health diagnostics.

``voiceprint quality`` scores individual samples against their person centroid
(outlier hunting). This module answers the person-level question instead: can
each person in the library be *trusted* for matching? Health folds together
coverage (sample count / total speech seconds / project diversity), embedding
completeness for the active model key, clip file integrity, cluster cohesion
(reusing quality diagnostics), and separation from the closest other person.
It is strictly read-only.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path

from app.speaker_pipeline_params import DEFAULT_MATCH_THRESHOLD
from app.voiceprint_embedding import resolve_voiceprint_embedding_options
from app.voiceprint_models import VoiceprintSampleRow, VoiceprintSpeakerRow
from app.voiceprint_quality import (
    VOICEPRINT_MATCHING_SAMPLE_STATUSES,
    VoiceprintQualityPerson,
    analyze_voiceprint_quality,
)
from app.voiceprint_store import (
    get_default_voiceprint_store_dir,
    get_voiceprint_db_path,
    list_all_voiceprint_samples,
    list_embedded_sample_ids,
    list_voiceprint_speakers,
    resolve_in_store_clip_path,
)

HEALTH_LEVEL_OK = "ok"
HEALTH_LEVEL_WARNING = "warning"
HEALTH_LEVEL_CRITICAL = "critical"
_LEVEL_SEVERITY = {
    HEALTH_LEVEL_OK: 0,
    HEALTH_LEVEL_WARNING: 1,
    HEALTH_LEVEL_CRITICAL: 2,
}


@dataclass(frozen=True, slots=True)
class VoiceprintHealthParams:
    """Thresholds for voiceprint library health checks."""

    min_matching_samples: int = 3
    min_matching_seconds: float = 20.0
    min_projects: int = 2
    separation_warning_score: float = 0.65
    separation_critical_score: float = DEFAULT_MATCH_THRESHOLD


@dataclass(frozen=True, slots=True)
class VoiceprintHealthCheck:
    """One health check result for a person."""

    key: str
    level: str
    detail: str
    action: str | None = None


@dataclass(frozen=True, slots=True)
class VoiceprintHealthPerson:
    """Health diagnostics for one voiceprint person."""

    speaker_id: int
    speaker_public_id: str
    speaker_name: str
    level: str
    sample_count: int
    matching_sample_count: int
    matching_seconds: float
    project_count: int
    missing_embedding_count: int
    missing_clip_count: int
    mean_score: float | None
    suspicious_count: int
    critical_count: int
    nearest_name: str | None
    nearest_score: float | None
    checks: tuple[VoiceprintHealthCheck, ...] = field(default=())

    @property
    def issues(self) -> tuple[VoiceprintHealthCheck, ...]:
        """Return checks that need attention."""
        return tuple(check for check in self.checks if check.level != HEALTH_LEVEL_OK)


@dataclass(frozen=True, slots=True)
class VoiceprintHealthReport:
    """Health diagnostics for the voiceprint library."""

    db_path: Path
    store_dir: Path
    model: str
    people: tuple[VoiceprintHealthPerson, ...]

    @property
    def sample_count(self) -> int:
        """Return total sample count across people."""
        return sum(person.sample_count for person in self.people)

    @property
    def matching_sample_count(self) -> int:
        """Return total matching sample count across people."""
        return sum(person.matching_sample_count for person in self.people)

    @property
    def ok_count(self) -> int:
        """Return number of healthy people."""
        return sum(1 for person in self.people if person.level == HEALTH_LEVEL_OK)

    @property
    def warning_count(self) -> int:
        """Return number of people with warnings."""
        return sum(1 for person in self.people if person.level == HEALTH_LEVEL_WARNING)

    @property
    def critical_count(self) -> int:
        """Return number of people with critical issues."""
        return sum(1 for person in self.people if person.level == HEALTH_LEVEL_CRITICAL)


def analyze_voiceprint_health(
    *,
    store_dir: Path | None = None,
    speaker: str | None = None,
    provider: str | None = None,
    model: str | None = None,
    params: VoiceprintHealthParams | None = None,
) -> VoiceprintHealthReport:
    """
    Analyze person-level health of the voiceprint library.

    Args:
        store_dir: Optional voiceprint store directory.
        speaker: Optional person name or public id filter.
        provider: Optional embedding provider override.
        model: Optional embedding model key.
        params: Optional health thresholds.

    Returns:
        Voiceprint health report. Read-only; never mutates the store.
    """
    resolved_params = params or VoiceprintHealthParams()
    _provider, resolved_model = resolve_voiceprint_embedding_options(
        provider=provider, model=model
    )
    resolved_store_dir = (
        store_dir.expanduser() if store_dir else get_default_voiceprint_store_dir()
    )
    db_path = get_voiceprint_db_path(store_dir)
    quality = analyze_voiceprint_quality(
        store_dir=store_dir, speaker=speaker, model=resolved_model
    )
    quality_by_id = {person.speaker_id: person for person in quality.people}
    samples_by_id: dict[int, list[VoiceprintSampleRow]] = defaultdict(list)
    for row in list_all_voiceprint_samples(db_path):
        samples_by_id[row.speaker_id].append(row)
    embedded_ids = list_embedded_sample_ids(resolved_model, db_path)
    people = tuple(
        _person_health(
            speaker_row,
            samples_by_id.get(speaker_row.speaker_id, []),
            quality_by_id.get(speaker_row.speaker_id),
            embedded_ids=embedded_ids,
            store_dir=resolved_store_dir,
            model=resolved_model,
            params=resolved_params,
        )
        for speaker_row in list_voiceprint_speakers(db_path)
        if _speaker_selected(speaker_row, speaker)
    )
    return VoiceprintHealthReport(
        db_path,
        resolved_store_dir,
        resolved_model,
        tuple(sorted(people, key=_person_sort_key)),
    )


def _speaker_selected(row: VoiceprintSpeakerRow, speaker: str | None) -> bool:
    """Apply the optional person name or public id filter."""
    if not speaker:
        return True
    normalized = speaker.strip().casefold()
    return normalized in {row.public_id.casefold(), row.name.casefold()}


def _person_health(
    speaker_row: VoiceprintSpeakerRow,
    samples: list[VoiceprintSampleRow],
    quality_person: VoiceprintQualityPerson | None,
    *,
    embedded_ids: set[int],
    store_dir: Path,
    model: str,
    params: VoiceprintHealthParams,
) -> VoiceprintHealthPerson:
    """Build health diagnostics for one person."""
    matching = [
        row
        for row in samples
        if row.sample_status in VOICEPRINT_MATCHING_SAMPLE_STATUSES
    ]
    matching_seconds = (
        sum(
            max(0, row.source_end_time_ms - row.source_begin_time_ms)
            for row in matching
        )
        / 1000.0
    )
    project_ids = {row.project_id for row in matching}
    missing_embedding = [row for row in matching if row.sample_id not in embedded_ids]
    missing_clip = [row for row in matching if _clip_missing(row, store_dir)]
    dead = [row for row in missing_embedding if _clip_missing(row, store_dir)]
    checks = (
        _coverage_checks(matching, matching_seconds, samples, params)
        + _diversity_checks(matching, project_ids, params)
        + _embedding_checks(matching, missing_embedding, dead, model)
        + _clip_checks(missing_clip, dead)
        + _cohesion_checks(matching, quality_person)
        + _separation_checks(quality_person, params)
    )
    nearest = (
        quality_person.closest_people[0]
        if quality_person and quality_person.closest_people
        else None
    )
    return VoiceprintHealthPerson(
        speaker_id=speaker_row.speaker_id,
        speaker_public_id=speaker_row.public_id,
        speaker_name=speaker_row.name,
        level=_worst_level(checks),
        sample_count=len(samples),
        matching_sample_count=len(matching),
        matching_seconds=matching_seconds,
        project_count=len(project_ids),
        missing_embedding_count=len(missing_embedding),
        missing_clip_count=len(missing_clip),
        mean_score=quality_person.mean_score if quality_person else None,
        suspicious_count=quality_person.suspicious_count if quality_person else 0,
        critical_count=quality_person.critical_count if quality_person else 0,
        nearest_name=nearest.speaker_name if nearest else None,
        nearest_score=nearest.score if nearest else None,
        checks=tuple(checks),
    )


def _coverage_checks(
    matching: list[VoiceprintSampleRow],
    matching_seconds: float,
    samples: list[VoiceprintSampleRow],
    params: VoiceprintHealthParams,
) -> list[VoiceprintHealthCheck]:
    """Check matching sample count and total speech duration."""
    if not matching:
        detail = (
            "no samples stored"
            if not samples
            else f"no matching samples ({len(samples)} stored, none active)"
        )
        return [
            VoiceprintHealthCheck(
                "coverage",
                HEALTH_LEVEL_CRITICAL,
                detail,
                "meeting-asr voiceprint review",
            )
        ]
    checks: list[VoiceprintHealthCheck] = []
    if len(matching) < params.min_matching_samples:
        checks.append(
            VoiceprintHealthCheck(
                "coverage",
                HEALTH_LEVEL_WARNING,
                (
                    f"only {len(matching)} matching sample(s)"
                    f" (recommend >= {params.min_matching_samples})"
                ),
                "meeting-asr voiceprint review",
            )
        )
    if matching_seconds < params.min_matching_seconds:
        checks.append(
            VoiceprintHealthCheck(
                "duration",
                HEALTH_LEVEL_WARNING,
                (
                    f"matching speech totals {matching_seconds:.1f}s"
                    f" (recommend >= {params.min_matching_seconds:.0f}s)"
                ),
                "meeting-asr voiceprint review",
            )
        )
    if not checks:
        checks.append(
            VoiceprintHealthCheck(
                "coverage",
                HEALTH_LEVEL_OK,
                f"{len(matching)} matching samples, {matching_seconds:.1f}s speech",
            )
        )
    return checks


def _diversity_checks(
    matching: list[VoiceprintSampleRow],
    project_ids: set[str],
    params: VoiceprintHealthParams,
) -> list[VoiceprintHealthCheck]:
    """Check that samples cover more than one recording."""
    if not matching:
        return []
    if len(project_ids) < params.min_projects:
        return [
            VoiceprintHealthCheck(
                "diversity",
                HEALTH_LEVEL_WARNING,
                (
                    f"all matching samples come from {len(project_ids)} project"
                    " (single recording condition)"
                ),
                "capture samples from another meeting",
            )
        ]
    return [
        VoiceprintHealthCheck(
            "diversity",
            HEALTH_LEVEL_OK,
            f"samples span {len(project_ids)} projects",
        )
    ]


def _embedding_checks(
    matching: list[VoiceprintSampleRow],
    missing_embedding: list[VoiceprintSampleRow],
    dead: list[VoiceprintSampleRow],
    model: str,
) -> list[VoiceprintHealthCheck]:
    """Check embedding completeness for the active model key."""
    if not matching:
        return []
    if len(missing_embedding) == len(matching):
        return [
            VoiceprintHealthCheck(
                "embedding",
                HEALTH_LEVEL_CRITICAL,
                f"no embeddings for model {model}; invisible to matching",
                "meeting-asr voiceprint embed",
            )
        ]
    if missing_embedding:
        repairable = len(missing_embedding) - len(dead)
        return [
            VoiceprintHealthCheck(
                "embedding",
                HEALTH_LEVEL_WARNING,
                (
                    f"{len(missing_embedding)} matching sample(s) missing embeddings"
                    f" for model {model}"
                    + (f" ({repairable} repairable)" if dead else "")
                ),
                "meeting-asr voiceprint embed",
            )
        ]
    return [
        VoiceprintHealthCheck(
            "embedding", HEALTH_LEVEL_OK, "embeddings complete for active model"
        )
    ]


def _clip_checks(
    missing_clip: list[VoiceprintSampleRow],
    dead: list[VoiceprintSampleRow],
) -> list[VoiceprintHealthCheck]:
    """Check that clip audio files still exist inside the store."""
    checks: list[VoiceprintHealthCheck] = []
    if dead:
        checks.append(
            VoiceprintHealthCheck(
                "clips",
                HEALTH_LEVEL_CRITICAL,
                (
                    f"{len(dead)} sample(s) have neither clip file nor embedding"
                    " and cannot be repaired"
                ),
                "meeting-asr voiceprint delete-sample",
            )
        )
    repairable_missing = len(missing_clip) - len(dead)
    if repairable_missing > 0:
        checks.append(
            VoiceprintHealthCheck(
                "clips",
                HEALTH_LEVEL_WARNING,
                (
                    f"{repairable_missing} clip file(s) missing on disk;"
                    " re-embedding after a model switch will fail"
                ),
            )
        )
    if not checks:
        checks.append(
            VoiceprintHealthCheck("clips", HEALTH_LEVEL_OK, "clip files present")
        )
    return checks


def _cohesion_checks(
    matching: list[VoiceprintSampleRow],
    quality_person: VoiceprintQualityPerson | None,
) -> list[VoiceprintHealthCheck]:
    """Fold sample-level quality diagnostics into one cohesion check."""
    if not matching:
        return []
    if quality_person is None or quality_person.mean_score is None:
        return [
            VoiceprintHealthCheck(
                "cohesion",
                HEALTH_LEVEL_OK,
                "not enough embedded samples to score cohesion",
            )
        ]
    if quality_person.critical_count:
        return [
            VoiceprintHealthCheck(
                "cohesion",
                HEALTH_LEVEL_CRITICAL,
                (
                    f"{quality_person.critical_count} critical outlier sample(s);"
                    " possible mislabeled identity"
                ),
                "meeting-asr voiceprint quality --review",
            )
        ]
    if quality_person.suspicious_count:
        return [
            VoiceprintHealthCheck(
                "cohesion",
                HEALTH_LEVEL_WARNING,
                f"{quality_person.suspicious_count} suspicious sample(s)",
                "meeting-asr voiceprint quality --review",
            )
        ]
    return [
        VoiceprintHealthCheck(
            "cohesion",
            HEALTH_LEVEL_OK,
            f"cluster consistent (mean score {quality_person.mean_score:.3f})",
        )
    ]


def _separation_checks(
    quality_person: VoiceprintQualityPerson | None,
    params: VoiceprintHealthParams,
) -> list[VoiceprintHealthCheck]:
    """Check centroid distance to the closest other person."""
    if quality_person is None or not quality_person.closest_people:
        return []
    nearest = quality_person.closest_people[0]
    if nearest.score >= params.separation_critical_score:
        return [
            VoiceprintHealthCheck(
                "separation",
                HEALTH_LEVEL_CRITICAL,
                (
                    f"centroid similarity {nearest.score:.3f} with"
                    f" {nearest.speaker_name} reaches the match threshold"
                    f" {params.separation_critical_score:.2f}; identities may be"
                    " confused during matching"
                ),
                (
                    "review both people's samples; merge with"
                    " `meeting-asr voiceprint people merge` if they are the"
                    " same person"
                ),
            )
        ]
    if nearest.score >= params.separation_warning_score:
        return [
            VoiceprintHealthCheck(
                "separation",
                HEALTH_LEVEL_WARNING,
                (
                    f"centroid similarity {nearest.score:.3f} with"
                    f" {nearest.speaker_name} is close to the match threshold"
                ),
                "meeting-asr voiceprint quality --review",
            )
        ]
    return [
        VoiceprintHealthCheck(
            "separation",
            HEALTH_LEVEL_OK,
            f"nearest person {nearest.speaker_name} at {nearest.score:.3f}",
        )
    ]


def _clip_missing(row: VoiceprintSampleRow, store_dir: Path) -> bool:
    """Return whether a sample's clip file is unavailable inside the store."""
    resolved = resolve_in_store_clip_path(row, store_dir)
    return resolved is None or not resolved.exists()


def _worst_level(checks: list[VoiceprintHealthCheck]) -> str:
    """Return the most severe level across checks."""
    if not checks:
        return HEALTH_LEVEL_OK
    return max(
        (check.level for check in checks), key=lambda level: _LEVEL_SEVERITY[level]
    )


def _person_sort_key(person: VoiceprintHealthPerson) -> tuple[int, str]:
    """Sort unhealthy people first."""
    return (-_LEVEL_SEVERITY[person.level], person.speaker_name.casefold())
