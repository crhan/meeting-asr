"""CLI rendering for voiceprint library health diagnostics."""

from __future__ import annotations

from rich import box
from rich.table import Table

from app.voiceprint_health import (
    HEALTH_LEVEL_CRITICAL,
    HEALTH_LEVEL_OK,
    HEALTH_LEVEL_WARNING,
    VoiceprintHealthCheck,
    VoiceprintHealthPerson,
    VoiceprintHealthReport,
)

_LEVEL_STYLES = {
    HEALTH_LEVEL_OK: "green",
    HEALTH_LEVEL_WARNING: "yellow",
    HEALTH_LEVEL_CRITICAL: "red",
}


def voiceprint_health_payload(report: VoiceprintHealthReport) -> dict[str, object]:
    """
    Build a machine-readable voiceprint health payload.

    Args:
        report: Health report.

    Returns:
        JSON-ready health payload.
    """
    return {
        "database": report.db_path,
        "store_dir": report.store_dir,
        "model": report.model,
        "people_count": len(report.people),
        "sample_count": report.sample_count,
        "matching_sample_count": report.matching_sample_count,
        "ok_count": report.ok_count,
        "warning_count": report.warning_count,
        "critical_count": report.critical_count,
        "people": [_person_payload(person) for person in report.people],
    }


def voiceprint_health_summary_lines(report: VoiceprintHealthReport) -> list[str]:
    """
    Return human-readable summary lines before the Rich table.

    Args:
        report: Health report.

    Returns:
        Summary lines.
    """
    return [
        f"Database: {report.db_path}",
        f"Model: {report.model}",
        (
            f"People: {len(report.people)}"
            f" (ok {report.ok_count} / warning {report.warning_count}"
            f" / critical {report.critical_count})"
            f" | Samples: {report.matching_sample_count} matching"
            f" / {report.sample_count} total"
        ),
    ]


def voiceprint_health_table(report: VoiceprintHealthReport) -> Table:
    """
    Build a compact per-person health table.

    Args:
        report: Health report.

    Returns:
        Rich table.
    """
    table = Table(box=box.ROUNDED, show_edge=True, pad_edge=True, header_style="bold")
    table.add_column("Person ID", no_wrap=True, style="bold cyan")
    table.add_column("Speaker", no_wrap=True)
    table.add_column("Health", no_wrap=True)
    table.add_column("Samples", justify="right", no_wrap=True)
    table.add_column("Seconds", justify="right", no_wrap=True)
    table.add_column("Projects", justify="right", no_wrap=True)
    table.add_column("Mean", justify="right", no_wrap=True)
    table.add_column("Nearest", no_wrap=True)
    table.add_column("Issues")
    for person in report.people:
        table.add_row(
            person.speaker_public_id,
            person.speaker_name,
            _level_text(person.level),
            f"{person.matching_sample_count}/{person.sample_count}",
            f"{person.matching_seconds:.1f}",
            str(person.project_count),
            _score_text(person.mean_score),
            _nearest_text(person),
            _issues_text(person),
        )
    return table


def voiceprint_health_action_lines(report: VoiceprintHealthReport) -> list[str]:
    """
    Return suggested follow-up actions for unhealthy people.

    Args:
        report: Health report.

    Returns:
        Action lines, empty when everyone is healthy.
    """
    lines: list[str] = []
    for person in report.people:
        actions = _unique_actions(person.issues)
        if not actions:
            continue
        lines.append(f"{person.speaker_name} ({person.speaker_public_id}):")
        lines.extend(f"  {action}" for action in actions)
    return lines


def _person_payload(person: VoiceprintHealthPerson) -> dict[str, object]:
    """Build one health person payload."""
    return {
        "speaker_id": person.speaker_id,
        "speaker_public_id": person.speaker_public_id,
        "speaker_name": person.speaker_name,
        "level": person.level,
        "sample_count": person.sample_count,
        "matching_sample_count": person.matching_sample_count,
        "matching_seconds": person.matching_seconds,
        "project_count": person.project_count,
        "missing_embedding_count": person.missing_embedding_count,
        "missing_clip_count": person.missing_clip_count,
        "mean_score": person.mean_score,
        "suspicious_count": person.suspicious_count,
        "critical_count": person.critical_count,
        "nearest_name": person.nearest_name,
        "nearest_score": person.nearest_score,
        "checks": [
            {
                "key": check.key,
                "level": check.level,
                "detail": check.detail,
                "action": check.action,
            }
            for check in person.checks
        ],
    }


def _unique_actions(issues: tuple[VoiceprintHealthCheck, ...]) -> list[str]:
    """Return de-duplicated actions preserving severity order."""
    seen: set[str] = set()
    actions: list[str] = []
    for check in sorted(issues, key=lambda item: item.level != HEALTH_LEVEL_CRITICAL):
        if not check.action or check.action in seen:
            continue
        seen.add(check.action)
        actions.append(check.action)
    return actions


def _issues_text(person: VoiceprintHealthPerson) -> str:
    """Render non-ok check details for one person."""
    issues = person.issues
    if not issues:
        return "-"
    return "\n".join(
        f"[{_LEVEL_STYLES[check.level]}]{check.detail}[/]" for check in issues
    )


def _nearest_text(person: VoiceprintHealthPerson) -> str:
    """Render the closest other person."""
    if person.nearest_name is None or person.nearest_score is None:
        return "-"
    return f"{person.nearest_name} {person.nearest_score:.3f}"


def _level_text(level: str) -> str:
    """Render a colored health level."""
    return f"[{_LEVEL_STYLES.get(level, 'white')}]{level}[/]"


def _score_text(score: float | None) -> str:
    """Format optional score."""
    return "-" if score is None else f"{score:.3f}"
