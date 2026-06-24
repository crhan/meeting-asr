"""CLI rendering for voiceprint quality diagnostics."""

from __future__ import annotations

from rich import box
from rich.table import Table

from app.voiceprint_quality import (
    VoiceprintQualityPerson,
    VoiceprintQualityReport,
    VoiceprintQualitySample,
)


def voiceprint_quality_payload(report: VoiceprintQualityReport) -> dict[str, object]:
    """
    Build a machine-readable voiceprint quality payload.

    Args:
        report: Quality report.

    Returns:
        JSON-ready quality payload.
    """
    return {
        "database": report.db_path,
        "model": report.model,
        "people_count": len(report.people),
        "sample_count": report.sample_count,
        "suspicious_count": report.suspicious_count,
        "critical_count": report.critical_count,
        "people": [_person_payload(person) for person in report.people],
    }


def voiceprint_quality_summary_lines(report: VoiceprintQualityReport) -> list[str]:
    """
    Return human-readable summary lines before the Rich table.

    Args:
        report: Quality report.

    Returns:
        Summary lines.
    """
    return [
        f"Database: {report.db_path}",
        f"Model: {report.model}",
        (
            f"People: {len(report.people)} | Samples: {report.sample_count} | "
            f"Suspicious: {report.suspicious_count} | Critical: {report.critical_count}"
        ),
    ]


def voiceprint_quality_table(report: VoiceprintQualityReport) -> Table:
    """
    Build a compact quality table.

    Args:
        report: Quality report.

    Returns:
        Rich table.
    """
    table = Table(box=box.ROUNDED, show_edge=True, pad_edge=True, header_style="bold")
    table.add_column("Person ID", no_wrap=True, style="bold cyan")
    table.add_column("Speaker", no_wrap=True)
    table.add_column("Samples", justify="right", no_wrap=True)
    table.add_column("Mean", justify="right", no_wrap=True)
    table.add_column("Risk", no_wrap=True)
    table.add_column("Worst Samples")
    for person in report.people:
        table.add_row(
            person.speaker_public_id,
            person.speaker_name,
            f"{person.active_sample_count}/{person.sample_count}",
            _score_text(person.mean_score),
            _risk_text(person),
            _worst_samples_text(person),
        )
    return table


def _person_payload(person: VoiceprintQualityPerson) -> dict[str, object]:
    """Build one quality person payload."""
    return {
        "speaker_id": person.speaker_id,
        "speaker_public_id": person.speaker_public_id,
        "speaker_name": person.speaker_name,
        "sample_count": person.sample_count,
        "active_sample_count": person.active_sample_count,
        "mean_score": person.mean_score,
        "stdev_score": person.stdev_score,
        "suspicious_count": person.suspicious_count,
        "critical_count": person.critical_count,
        "projects": [
            {
                "project_id": project.project_id,
                "sample_count": project.sample_count,
                "matching_sample_count": project.matching_sample_count,
                "suspicious_count": project.suspicious_count,
                "critical_count": project.critical_count,
                "mean_score": project.mean_score,
                "min_score": project.min_score,
            }
            for project in person.projects
        ],
        "closest_people": [
            {
                "speaker_id": neighbor.speaker_id,
                "speaker_public_id": neighbor.speaker_public_id,
                "speaker_name": neighbor.speaker_name,
                "score": neighbor.score,
            }
            for neighbor in person.closest_people
        ],
        "samples": [_sample_payload(sample) for sample in person.samples],
    }


def _sample_payload(sample: VoiceprintQualitySample) -> dict[str, object]:
    """Build one quality sample payload."""
    return {
        "sample_id": sample.sample_id,
        "sample_public_id": sample.sample_public_id,
        "speaker_id": sample.speaker_id,
        "speaker_public_id": sample.speaker_public_id,
        "speaker_name": sample.speaker_name,
        "project_id": sample.project_id,
        "clip_path": sample.clip_path,
        "status": sample.status,
        "score": sample.score,
        "label": sample.label,
        "reason": sample.reason,
        "transcript_text": sample.transcript_text,
    }


def _risk_text(person: VoiceprintQualityPerson) -> str:
    """Render person quality risk."""
    text = f"{person.suspicious_count} suspicious, {person.critical_count} critical"
    if person.critical_count:
        return f"[red]{text}[/]"
    if person.suspicious_count:
        return f"[yellow]{text}[/]"
    return f"[green]{text}[/]"


def _worst_samples_text(person: VoiceprintQualityPerson) -> str:
    """Render worst samples for one person."""
    rows = [
        sample
        for sample in person.samples
        if sample.status == "active" and sample.label in {"critical", "warning"}
    ][:3]
    if not rows:
        return "-"
    return "\n".join(
        f"{sample.sample_public_id} {_score_text(sample.score)} {sample.label}"
        for sample in rows
    )


def _score_text(score: float | None) -> str:
    """Format optional score."""
    return "-" if score is None else f"{score:.3f}"
