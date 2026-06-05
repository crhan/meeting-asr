"""Rich rendering for project overview output."""

from __future__ import annotations

import json
import shlex
from dataclasses import dataclass
from pathlib import Path

from rich import box
from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel
from rich.table import Table

from app.asr_pricing import asr_cost_from_dict, format_asr_cost
from app.core.project_models import (
    TITLE_SOURCE_LLM,
    TITLE_SOURCE_MANUAL,
    TITLE_SOURCE_SOURCE,
    TITLE_SOURCE_UNKNOWN,
    ProjectManifest,
)
from app.core.project_workflow import ProjectWorkflowSummary
from app.presentation.cli.output import cli_console
from app.presentation.cli.speaker_match_table import (
    SpeakerMatchRow,
    render_speaker_match_table,
    speaker_match_rows,
)
from app.postprocess import speaker_id_to_label
from app.speaker_labeling import load_project_ignored_speakers
from app.speaker_match_status import (
    MATCH_STATUS_CROSSTALK,
    MATCH_STATUS_IGNORED,
    MATCH_STATUS_MATCHED,
)


@dataclass(frozen=True, slots=True)
class ProjectShowView:
    """Presentation data for one project overview."""

    project_dir: Path
    project_ref: str
    manifest: ProjectManifest
    workflow: ProjectWorkflowSummary


@dataclass(frozen=True, slots=True)
class _OutputRow:
    """One project output row."""

    label: str
    kind: str
    path: Path | None


def render_project_show(view: ProjectShowView) -> None:
    """
    Render a project overview for humans.

    Args:
        view: Fully prepared project overview data.

    Returns:
        None.
    """
    console = _show_console()
    console.print(_status_panel(view))
    console.print(_details_table(view))
    summary_panel = _summary_panel(view)
    if summary_panel is not None:
        console.print(summary_panel)
    match_table = _speaker_match_table(view)
    if match_table is not None:
        console.print(match_table)
    console.print(_outputs_table(view))
    console.print(_commands_table(view))


def _status_panel(view: ProjectShowView) -> Panel:
    """Build the high-level project status panel."""
    style = _state_style(view.workflow.state_key)
    grid = Table.grid(expand=True)
    grid.add_column(ratio=3)
    grid.add_column(justify="right", no_wrap=True)
    grid.add_row(f"[bold]{view.manifest.title}[/]", f"[bold cyan]{view.project_ref}[/]")
    grid.add_row(f"[dim]{view.project_dir}[/]", f"[{style}]{view.workflow.state}[/]")
    return Panel(
        grid, title=f"[bold {style}]Project[/]", border_style=style, expand=False
    )


def _details_table(view: ProjectShowView) -> Table:
    """Build key project metadata rows."""
    table = Table(
        title="Details", box=box.SIMPLE_HEAVY, show_edge=False, pad_edge=False
    )
    table.add_column("Item", style="bold", no_wrap=True)
    table.add_column("Value")
    for item, value in _detail_rows(view):
        table.add_row(item, value)
    return table


def _detail_rows(view: ProjectShowView) -> list[tuple[str, str]]:
    """Return scan-friendly project metadata rows."""
    manifest = view.manifest
    rows = [
        ("Project ID", manifest.project_id),
        ("Title source", _title_source_label(manifest)),
        ("State", view.workflow.state),
        ("Updated", _compact_timestamp(manifest.updated_at)),
        ("Meeting time", manifest.source.meeting_time or "-"),
        ("Source file", _source_file_label(manifest)),
    ]
    segments_label = _segments_label(manifest)
    if segments_label:
        rows.append(("Segments", segments_label))
    rows.extend(
        [
            ("Duration", _duration_label(manifest.audio.get("duration_seconds"))),
            ("Speakers", _speakers_label(view)),
            ("ASR cost", _cost_label(manifest.asr.get("cost"))),
        ]
    )
    local_correction_label = _local_correction_label(view)
    if local_correction_label:
        rows.append(("Local correction", local_correction_label))
    polish_label = _polish_label(view)
    if polish_label:
        rows.append(("Transcript polish", polish_label))
    rows.extend(_runtime_rows(manifest))
    if view.workflow.missing:
        rows.append(("Missing", ", ".join(view.workflow.missing)))
    return rows


def _source_file_label(manifest: ProjectManifest) -> str:
    """Return the source label, listing segment origins for multi-input projects."""
    segments = manifest.audio.get("segments")
    if isinstance(segments, list) and len(segments) > 1:
        names = [str(seg.get("filename") or "?") for seg in segments]
        return " + ".join(names)
    return manifest.source.original_path or manifest.source.path


def _segments_label(manifest: ProjectManifest) -> str | None:
    """Return per-segment provenance for concatenated multi-input projects."""
    segments = manifest.audio.get("segments")
    if not isinstance(segments, list) or len(segments) <= 1:
        return None
    parts = []
    for seg in segments:
        name = str(seg.get("filename") or "?")
        duration = seg.get("duration_seconds")
        offset = seg.get("offset_seconds")
        if isinstance(offset, int | float) and isinstance(duration, int | float):
            parts.append(f"{name} (@{_duration_label(offset)} +{_duration_label(duration)})")
        else:
            parts.append(name)
    return f"{len(segments)} segments: " + "; ".join(parts)


def _title_source_label(manifest: ProjectManifest) -> str:
    """Return how the current project title was produced."""
    if manifest.title_source == TITLE_SOURCE_LLM:
        return f"LLM ({manifest.title_model or 'configured model'})"
    if manifest.title_source == TITLE_SOURCE_MANUAL:
        return "manual"
    if manifest.title_source == TITLE_SOURCE_SOURCE:
        return "source filename"
    if manifest.title_source == TITLE_SOURCE_UNKNOWN and _has_legacy_custom_title(
        manifest
    ):
        return "manual (legacy)"
    return "unknown"


def _has_legacy_custom_title(manifest: ProjectManifest) -> bool:
    """Return whether an unknown title looks like a preserved custom title."""
    current_title = manifest.title.strip()
    source_stem = Path(manifest.source.filename).stem.strip()
    return bool(current_title) and current_title != source_stem


def _runtime_rows(manifest: ProjectManifest) -> list[tuple[str, str]]:
    """Return current long-task stage rows from project runtime metadata."""
    runtime = manifest.runtime
    if not runtime:
        return []
    rows = [
        ("Current stage", str(runtime.get("current_stage") or "-")),
        (
            "Stage updated",
            str(
                runtime.get("last_heartbeat_at")
                or runtime.get("stage_started_at")
                or "-"
            ),
        ),
    ]
    external = runtime.get("external_ids")
    if isinstance(external, dict) and external:
        rows.append(("External IDs", _external_ids_label(external)))
    last_error = runtime.get("last_error")
    if isinstance(last_error, dict) and last_error.get("message"):
        rows.append(("Last error", str(last_error["message"])))
    elif last_error:
        rows.append(("Last error", str(last_error)))
    return rows


def _external_ids_label(external: dict) -> str:
    """Render compact non-secret external identifiers."""
    parts = []
    for key in sorted(external):
        parts.append(f"{key}={external[key]}")
    return ", ".join(parts)


def _outputs_table(view: ProjectShowView) -> Table:
    """Build output artifact rows."""
    table = Table(
        title="Outputs", box=box.SIMPLE_HEAVY, show_edge=False, pad_edge=False
    )
    table.add_column("Artifact", style="bold", no_wrap=True)
    table.add_column("Status", no_wrap=True)
    table.add_column("Command", no_wrap=True)
    for row in _output_rows(view):
        table.add_row(row.label, _output_status(row), _view_command(view, row))
    return table


def _summary_panel(view: ProjectShowView) -> Panel | None:
    """Build a direct meeting-summary preview when the artifact exists."""
    row = _manifest_output(
        view.project_dir, view.manifest, "Memory index", "summary", ("meeting_summary",)
    )
    if row.path is None:
        return None
    try:
        content = _display_summary_content(row.path.read_text(encoding="utf-8"))
    except OSError:
        return None
    if not content:
        return None
    return Panel(
        Markdown(content),
        title="[bold]Memory Index[/]",
        border_style="cyan",
        expand=False,
    )


def _speaker_match_table(view: ProjectShowView) -> Table | None:
    """Build the voiceprint match table when match artifacts exist."""
    return render_speaker_match_table(_speaker_match_rows(view))


def _speaker_match_rows(view: ProjectShowView) -> tuple[SpeakerMatchRow, ...]:
    """Load voiceprint match rows when match artifacts exist."""
    path = view.project_dir / "speakers" / "speaker_matches.json"
    if not path.exists():
        return ()
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except OSError, json.JSONDecodeError:
        return ()
    if not isinstance(payload, dict):
        return ()
    ignored = load_project_ignored_speakers(view.project_dir)
    return speaker_match_rows(
        payload.get("matches", []),
        default_threshold=_safe_float(payload.get("threshold")),
        ignored_speaker_ids=ignored,
    )


def _output_rows(view: ProjectShowView) -> list[_OutputRow]:
    """Return project output rows in user-facing order."""
    root = view.project_dir
    return [
        _manifest_output(
            root,
            view.manifest,
            "Final transcript",
            "auto",
            (
                "corrected_named_transcript",
                "named_transcript",
                "anonymous_transcript",
                "plain_transcript",
            ),
        ),
        _manifest_output(
            root,
            view.manifest,
            "Final subtitles",
            "srt",
            ("corrected_named_subtitle", "named_subtitle", "subtitle"),
        ),
        _manifest_output(
            root,
            view.manifest,
            "Speaker transcript",
            "speakers",
            ("anonymous_transcript",),
        ),
        _manifest_output(
            root, view.manifest, "Plain transcript", "plain", ("plain_transcript",)
        ),
    ]


def _commands_table(view: ProjectShowView) -> Table:
    """Build copyable next commands."""
    table = Table(
        title="Commands", box=box.SIMPLE_HEAVY, show_edge=False, pad_edge=False
    )
    table.add_column("#", justify="right", style="bold cyan", no_wrap=True)
    table.add_column("Action", style="bold", no_wrap=True)
    table.add_column("Command")
    for index, (action, command) in enumerate(_command_rows(view), start=1):
        table.add_row(str(index), action, command)
    return table


def _command_rows(view: ProjectShowView) -> list[tuple[str, str]]:
    """Return common project follow-up commands."""
    quoted_ref = shlex.quote(view.project_ref)
    next_command = (
        f"meeting-asr project review {quoted_ref}"
        if _has_unresolved_matches(view)
        else view.workflow.next_command
    )
    rows = [
        ("Next", next_command),
        *_polish_command_rows(view, quoted_ref),
        ("Show transcript", f"meeting-asr project transcript show {quoted_ref}"),
        ("List outputs", f"meeting-asr project transcript list {quoted_ref}"),
        ("Review speakers", f"meeting-asr project review {quoted_ref}"),
        ("Preview subtitles", f"meeting-asr project speakers preview {quoted_ref}"),
    ]
    return _unique_command_rows(rows)


def _has_unresolved_matches(view: ProjectShowView) -> bool:
    """Return whether voiceprint results still need human review."""
    return any(
        row.status
        not in (MATCH_STATUS_MATCHED, MATCH_STATUS_IGNORED, MATCH_STATUS_CROSSTALK)
        for row in _speaker_match_rows(view)
    )


def _polish_label(view: ProjectShowView) -> str | None:
    """Return the user-facing transcript polish state."""
    state = _polish_state(view)
    if state is None:
        return None
    status = str(state.get("status") or "unknown")
    count = _safe_int(state.get("proposed_changes"))
    if status == "accepted":
        accepted = _safe_int(state.get("accepted_changes"))
        return f"accepted ({accepted}/{count} change(s)); corrected transcript ready"
    if status == "proposal_ready":
        return f"proposal ready ({count} change(s)); accept or inspect diff if needed"
    if status == "no_changes":
        return "done; no changes proposed"
    if status == "failed":
        error = str(state.get("error") or "unknown error")
        return f"failed; {error}"
    return status.replace("_", " ")


def _local_correction_label(view: ProjectShowView) -> str | None:
    """Return the user-facing local correction state."""
    state = view.manifest.runtime.get("local_correction")
    if not isinstance(state, dict):
        return None
    status = str(state.get("status") or "unknown")
    changed = _safe_int(state.get("changed_sentences"))
    rules = _safe_int(state.get("rules_applied"))
    if status == "applied":
        return f"applied ({changed} sentence(s), {rules} rule(s))"
    if status == "no_changes":
        return "done; no local lexicon changes"
    return status.replace("_", " ")


def _polish_command_rows(
    view: ProjectShowView, quoted_ref: str
) -> list[tuple[str, str]]:
    """Return transcript-polish follow-up commands for project show."""
    state = _polish_state(view)
    if state is None:
        return []
    status = str(state.get("status") or "")
    if status == "proposal_ready":
        proposal = _proposal_option(view, state)
        return [
            (
                "Accept transcript polish",
                f"meeting-asr project correct accept {quoted_ref}{proposal}",
            ),
            (
                "Inspect transcript polish diff",
                f"meeting-asr project correct diff {quoted_ref}{proposal}",
            ),
        ]
    if status == "failed":
        model = str(state.get("model") or "").strip()
        suffix = f" --model {shlex.quote(model)}" if model else ""
        return [
            (
                "Retry transcript polish",
                f"meeting-asr project correct polish {quoted_ref}{suffix}",
            )
        ]
    return []


def _proposal_option(view: ProjectShowView, state: dict) -> str:
    """Return a specific proposal option so polish commands do not pick a newer correction."""
    value = state.get("proposal_json")
    if not value:
        return ""
    path = Path(str(value)).expanduser()
    if not path.is_absolute():
        path = view.project_dir / path
    return f" --proposal {shlex.quote(str(path))}"


def _polish_state(view: ProjectShowView) -> dict | None:
    """Return persisted or inferred transcript polish state."""
    runtime_state = view.manifest.runtime.get("polish")
    if isinstance(runtime_state, dict):
        return runtime_state
    return _latest_polish_proposal_state(view.project_dir)


def _latest_polish_proposal_state(project_dir: Path) -> dict | None:
    """Infer pending polish state from old proposal artifacts."""
    proposal_dir = project_dir / "tmp" / "corrections"
    proposals = sorted(proposal_dir.glob("proposal_*.json"))
    for path in reversed(proposals):
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except OSError, json.JSONDecodeError:
            continue
        if not isinstance(payload, dict) or payload.get("category") != "polish":
            continue
        return {
            "status": "proposal_ready",
            "model": payload.get("model"),
            "error": payload.get("model_error"),
            "proposed_changes": len(payload.get("proposed_changes") or []),
            "proposal_json": str(path.relative_to(project_dir)),
            "proposal_diff": payload.get("diff_path"),
        }
    return None


def _manifest_output(
    root: Path, manifest: ProjectManifest, label: str, kind: str, keys: tuple[str, ...]
) -> _OutputRow:
    """Resolve the first existing manifest output for a row."""
    for key in keys:
        value = manifest.outputs.get(key)
        if isinstance(value, str) and value:
            path = _stored_path(root, value)
            if path.exists():
                return _OutputRow(label, kind, path)
    return _OutputRow(label, kind, None)


def _display_summary_content(content: str) -> str:
    """Return meeting memory-index markdown without generation metadata."""
    lines = []
    for line in content.splitlines():
        stripped = line.strip()
        if stripped.startswith(("模型：", "Model:")):
            continue
        lines.append(line)
    return "\n".join(lines).strip()


def _output_status(row: _OutputRow) -> str:
    """Return styled artifact availability."""
    return "[green]ready[/]" if row.path else "[red]missing[/]"


def _view_command(view: ProjectShowView, row: _OutputRow) -> str:
    """Return a complete command for viewing one artifact."""
    if row.path is None:
        return "-"
    return f"meeting-asr project transcript show {shlex.quote(view.project_ref)} --kind {row.kind}"


def _unique_command_rows(rows: list[tuple[str, str]]) -> list[tuple[str, str]]:
    """Remove duplicate commands while preserving order."""
    seen: set[str] = set()
    unique_rows = []
    for action, command in rows:
        if command in seen:
            continue
        seen.add(command)
        unique_rows.append((action, command))
    return unique_rows


def _state_style(state_key: str) -> str:
    """Return a Rich style for a workflow state."""
    styles = {
        "created": "yellow",
        "prepared": "yellow",
        "transcribed": "cyan",
        "completed": "green",
        "corrected": "green",
        "broken": "red",
    }
    return styles.get(state_key, "white")


def _speakers_label(view: ProjectShowView) -> str:
    """Return detected and mapped speaker summary."""
    manifest = view.manifest
    detected = manifest.speakers.get("detected_ids") or []
    mapped = manifest.speakers.get("mapped") or {}
    active_ids = _active_speaker_ids(view.project_dir)
    visible_names = _mapped_speaker_names(mapped, active_ids)
    names = ", ".join(visible_names)
    detected_count = len(active_ids) if active_ids is not None else len(detected)
    ignored_count = _ignored_speaker_count(detected, active_ids)
    base = f"{detected_count} detected"
    if ignored_count:
        base = f"{base}; ignored {ignored_count} low-info"
    return f"{base}; {names}" if names else base


def _cost_label(value: object) -> str:
    """Return formatted ASR cost from manifest data."""
    if not isinstance(value, dict):
        return "-"
    try:
        return format_asr_cost(asr_cost_from_dict(value))
    except TypeError, ValueError, KeyError:
        return "-"


def _active_speaker_ids(project_dir: Path) -> set[int] | None:
    """Return speaker ids that survive low-information filtering."""
    path = project_dir / "asr" / "sentences.json"
    if not path.exists():
        return None
    try:
        from app.speaker_labeling import load_transcript_result

        return set(
            load_transcript_result(path, include_low_information=True).detected_speakers
        )
    except OSError, ValueError, TypeError, KeyError:
        return None


def _mapped_speaker_names(mapped: object, active_ids: set[int] | None) -> list[str]:
    """Return mapped names for active speakers only."""
    if not isinstance(mapped, dict):
        return []
    names = []
    for speaker_id, name in sorted(mapped.items(), key=lambda item: str(item[0])):
        try:
            numeric_id = int(speaker_id)
        except TypeError, ValueError:
            continue
        if active_ids is not None and numeric_id not in active_ids:
            continue
        name_text = str(name).strip()
        if not name_text or name_text == speaker_id_to_label(numeric_id):
            continue
        names.append(name_text)
    return names


def _safe_float(value: object) -> float | None:
    """Return a float value when possible."""
    try:
        return float(value)
    except TypeError, ValueError:
        return None


def _safe_int(value: object) -> int:
    """Return a non-negative integer display value."""
    try:
        return max(0, int(str(value)))
    except TypeError, ValueError:
        return 0


def _ignored_speaker_count(detected: object, active_ids: set[int] | None) -> int:
    """Return how many manifest speakers are now ignored as low-information tracks."""
    if active_ids is None or not isinstance(detected, list):
        return 0
    detected_ids = set()
    for speaker_id in detected:
        try:
            detected_ids.add(int(speaker_id))
        except TypeError, ValueError:
            continue
    return len(detected_ids - active_ids)


def _duration_label(value: object) -> str:
    """Return a compact duration label."""
    if not isinstance(value, int | float):
        return "-"
    seconds = int(value)
    minutes, remaining_seconds = divmod(seconds, 60)
    hours, remaining_minutes = divmod(minutes, 60)
    if hours:
        return f"{hours}h{remaining_minutes:02d}m{remaining_seconds:02d}s"
    return f"{remaining_minutes}m{remaining_seconds:02d}s"


def _compact_timestamp(value: str) -> str:
    """Return a compact timestamp label."""
    return value[:16].replace("T", " ")


def _stored_path(root: Path, value: str) -> Path:
    """Resolve a project-stored path."""
    path = Path(value).expanduser()
    if path.is_absolute():
        return path
    return root / path


def _show_console() -> Console:
    """Build the stdout console used for project overview tables."""
    return cli_console(width=120)
