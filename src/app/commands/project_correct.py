"""Project transcript correction commands."""

from __future__ import annotations

import json
import shlex
from dataclasses import replace
from datetime import datetime
from pathlib import Path
from typing import Optional

import typer

from app.core.project_models import ProjectManifest
from app.core.project_refs import resolve_project_ref
from app.correction_proposals import load_correction_proposal
from app.presentation.cli.errors import run_with_cli_errors
from app.presentation.cli.progress import run_with_progress
from app.presentation.cli.typer_context import HELP_CONTEXT, MeetingAsrTyper
from app.project_manager import load_manifest, project_paths, save_manifest
from app.speaker_labeling import build_default_mapping, load_transcript_result
from app.transcript_corrections import (
    CorrectionEditOptions,
    CorrectionEditSummary,
    accept_correction_proposal,
    prepare_editor_correction,
    prepare_inline_corrections,
    prepare_transcript_polish,
)

app = MeetingAsrTyper(
    add_completion=False,
    context_settings=HELP_CONTEXT,
    no_args_is_help=True,
    pretty_exceptions_enable=False,
)


@app.command("edit")
def edit_command(
    project_dir: Path = typer.Argument(Path("."), metavar="PROJECT", file_okay=False, dir_okay=True),
    projects_dir: Optional[Path] = typer.Option(None, "--projects-dir", file_okay=False, dir_okay=True, hidden=True),
    editor: Optional[str] = typer.Option(None, "--editor", help="Editor command. Use {file} as optional placeholder."),
    review_file: Optional[Path] = typer.Option(None, "--review-file", exists=True, dir_okay=False, file_okay=True),
    no_open: bool = typer.Option(False, "--no-open", help="Only write the review file; do not launch an editor."),
    no_ai: bool = typer.Option(False, "--no-ai", help="Disable DashScope proposal generation and use local rules."),
    no_proposal_open: bool = typer.Option(False, "--no-proposal-open", help="Do not open the generated proposal file."),
    yes: bool = typer.Option(False, "--yes", "-y", help="Accept the generated full-document proposal without prompting."),
    model: Optional[str] = typer.Option(None, "--model", help="DashScope correction model id."),
    category: str = typer.Option("unknown", "--category", help="Category for learned terms."),
    lexicon_db: Optional[Path] = typer.Option(None, "--lexicon-db", help="Override lexicon SQLite path."),
    from_original: bool = typer.Option(False, "--from-original", help="Ignore an existing corrected transcript."),
) -> None:
    """Open an editable transcript, generate a full-document proposal, and accept it on confirmation."""
    paths, manifest, speaker_mapping = _load_command_context(project_dir, projects_dir)
    options = CorrectionEditOptions(
        editor=editor,
        review_file=review_file,
        open_editor=not no_open,
        open_proposal=not no_open and not no_proposal_open,
        category=category,
        lexicon_db=lexicon_db,
        from_original=from_original,
        use_ai=not no_ai,
        model=model,
    )
    summary = run_with_cli_errors(
        lambda: prepare_editor_correction(paths=paths, manifest=manifest, speaker_mapping=speaker_mapping, options=options)
    )
    _finish_correction_edit(paths, manifest, speaker_mapping, summary, lexicon_db, yes, auto_accept=not no_open)


@app.command("polish")
def polish_command(
    project_dir: Path = typer.Argument(Path("."), metavar="PROJECT", file_okay=False, dir_okay=True),
    projects_dir: Optional[Path] = typer.Option(None, "--projects-dir", file_okay=False, dir_okay=True, hidden=True),
    yes: bool = typer.Option(False, "--yes", "-y", help="Accept the generated polish proposal without prompting."),
    model: Optional[str] = typer.Option(None, "--model", help="DashScope correction model id."),
    concurrency: Optional[int] = typer.Option(
        None,
        "--concurrency",
        min=1,
        max=64,
        help="Parallel DashScope batch requests for transcript polish.",
    ),
    lexicon_db: Optional[Path] = typer.Option(None, "--lexicon-db", help="Override lexicon SQLite path."),
    from_original: bool = typer.Option(False, "--from-original", help="Polish from the original ASR transcript."),
    legacy: bool = typer.Option(
        False,
        "--legacy-polish",
        help="Use the legacy aggressive-rewrite polish prompt (pre-2026 behavior). "
        "Default is the strict downstream-summary-friendly polish.",
    ),
    progress: bool = typer.Option(True, "--progress/--no-progress", help="Show interactive progress on a terminal."),
    agent_log: bool = typer.Option(
        False,
        "--agent-log",
        help="Print structured polish stage and heartbeat logs; combine with --no-progress for clean logs.",
    ),
) -> None:
    """Generate an AI transcript polish proposal.

    Strict polish (default) cleans ASR noise (repetitions, fillers, restarts, emphasis)
    and fixes obvious typos/terms while preserving fact-bearing modifiers (我觉得/可能/对吧/...)
    that downstream summary agents rely on. Each kept change carries a change_type tag
    (typo/term/case/punct/dup/filler/restart/emphasis) so `accept --types ...` can pick.
    Use --legacy-polish for the older aggressive-rewrite behavior.
    """
    paths, manifest, speaker_mapping = _load_command_context(project_dir, projects_dir)
    options = CorrectionEditOptions(
        open_editor=False,
        open_proposal=False,
        category="polish",
        lexicon_db=lexicon_db,
        from_original=from_original,
        use_ai=True,
        model=model,
        polish_concurrency=concurrency,
        polish_legacy=legacy,
    )
    summary = run_with_progress(
        lambda reporter: prepare_transcript_polish(
            paths=paths,
            manifest=manifest,
            speaker_mapping=speaker_mapping,
            options=options,
            progress=reporter,
        ),
        description="Generating transcript polish proposal",
        enabled=progress,
        structured_log=agent_log,
    )
    _finish_correction_edit(paths, manifest, speaker_mapping, summary, lexicon_db, yes, auto_accept=True)


def finish_editor_correction(
    *,
    paths,
    manifest: ProjectManifest,
    speaker_mapping: dict[int, str],
    options: CorrectionEditOptions,
    yes: bool,
) -> CorrectionEditSummary:
    """
    Run the same editor-driven correction flow used by ``project correct edit``.

    Args:
        paths: Project paths.
        manifest: Loaded project manifest.
        speaker_mapping: Speaker id to display name mapping.
        options: Correction options.
        yes: Whether to accept the generated proposal without prompting.

    Returns:
        Final correction summary.
    """
    summary = prepare_editor_correction(paths=paths, manifest=manifest, speaker_mapping=speaker_mapping, options=options)
    return _finish_correction_edit(
        paths,
        manifest,
        speaker_mapping,
        summary,
        options.lexicon_db,
        yes,
        auto_accept=options.open_editor,
    )


def prepare_transcript_polish_for_review(
    *,
    paths,
    manifest: ProjectManifest,
    speaker_mapping: dict[int, str],
    options: CorrectionEditOptions,
    progress=None,
) -> CorrectionEditSummary:
    """
    Prepare a pending transcript polish proposal without printing or prompting.

    Args:
        paths: Project paths.
        manifest: Loaded project manifest.
        speaker_mapping: Speaker id to display name mapping.
        options: Correction options.

    Returns:
        Pending polish summary, or a no-change summary.
    """
    polish_options = replace(
        options,
        open_editor=False,
        open_proposal=False,
        category=options.category or "polish",
        use_ai=True,
    )
    return prepare_transcript_polish(
        paths=paths,
        manifest=manifest,
        speaker_mapping=speaker_mapping,
        options=polish_options,
        progress=progress,
    )


def finish_inline_correction(
    *,
    paths,
    manifest: ProjectManifest,
    speaker_mapping: dict[int, str],
    correction_edit: object,
    options: CorrectionEditOptions,
    yes: bool,
) -> CorrectionEditSummary:
    """
    Run the TUI inline correction flow without launching an external editor.

    Args:
        paths: Project paths.
        manifest: Loaded project manifest.
        speaker_mapping: Speaker id to display name mapping.
        correction_edit: TUI sentence edit.
        options: Correction options.
        yes: Whether to accept the generated proposal without prompting.

    Returns:
        Final correction summary.
    """
    return finish_inline_corrections(
        paths=paths,
        manifest=manifest,
        speaker_mapping=speaker_mapping,
        correction_edits=[correction_edit],
        options=options,
        yes=yes,
    )


def finish_inline_corrections(
    *,
    paths,
    manifest: ProjectManifest,
    speaker_mapping: dict[int, str],
    correction_edits: list[object],
    options: CorrectionEditOptions,
    yes: bool,
) -> CorrectionEditSummary:
    """
    Run the TUI inline correction flow for multiple edited sentences.

    Args:
        paths: Project paths.
        manifest: Loaded project manifest.
        speaker_mapping: Speaker id to display name mapping.
        correction_edits: TUI sentence edits.
        options: Correction options.
        yes: Whether to accept the generated proposal without prompting.

    Returns:
        Final correction summary.
    """
    summary = prepare_inline_corrections_for_review(
        paths=paths,
        manifest=manifest,
        speaker_mapping=speaker_mapping,
        correction_edits=correction_edits,
        options=options,
    )
    return _finish_correction_edit(
        paths,
        manifest,
        speaker_mapping,
        summary,
        options.lexicon_db,
        yes,
        auto_accept=True,
    )


def prepare_inline_corrections_for_review(
    *,
    paths,
    manifest: ProjectManifest,
    speaker_mapping: dict[int, str],
    correction_edits: list[object],
    options: CorrectionEditOptions,
) -> CorrectionEditSummary:
    """
    Prepare pending correction proposal from TUI edits without printing or prompting.

    Args:
        paths: Project paths.
        manifest: Loaded project manifest.
        speaker_mapping: Speaker id to display name mapping.
        correction_edits: TUI sentence edits.
        options: Correction options.

    Returns:
        Pending proposal summary.
    """
    inline_options = replace(options, open_editor=False, open_proposal=False)
    return prepare_inline_corrections(
        paths=paths,
        manifest=manifest,
        speaker_mapping=speaker_mapping,
        correction_edits=correction_edits,
        options=inline_options,
    )


def accept_correction_for_review(
    *,
    paths,
    manifest: ProjectManifest,
    speaker_mapping: dict[int, str],
    proposal_path: Path | None,
    lexicon_db: Path | None,
    selected_change_indices: tuple[int, ...] | None = None,
) -> CorrectionEditSummary:
    """
    Accept a pending correction proposal without CLI prompting or printing.

    Args:
        paths: Project paths.
        manifest: Loaded project manifest.
        speaker_mapping: Speaker id to display name mapping.
        proposal_path: Pending proposal JSON path.
        lexicon_db: Optional lexicon database override.
        selected_change_indices: Optional zero-based proposed change indices to accept.

    Returns:
        Accepted correction summary.
    """
    accepted = accept_correction_proposal(
        paths=paths,
        manifest=manifest,
        speaker_mapping=speaker_mapping,
        proposal_path=proposal_path,
        lexicon_db=lexicon_db,
        selected_change_indices=selected_change_indices,
    )
    record_correction_outputs(paths.root, manifest, accepted)
    save_manifest(paths.root, manifest)
    return accepted


@app.command("accept")
def accept_command(
    project_dir: Path = typer.Argument(Path("."), metavar="PROJECT", file_okay=False, dir_okay=True),
    projects_dir: Optional[Path] = typer.Option(None, "--projects-dir", file_okay=False, dir_okay=True, hidden=True),
    proposal: Optional[Path] = typer.Option(None, "--proposal", exists=True, dir_okay=False, file_okay=True),
    lexicon_db: Optional[Path] = typer.Option(None, "--lexicon-db", help="Override lexicon SQLite path."),
    select: Optional[str] = typer.Option(
        None,
        "--select",
        help="Accept only the listed change indices (0-based). Comma/space separated, "
        "supports ranges, e.g. '0,3,7-12'. Indices come from the proposal markdown.",
    ),
    types: Optional[str] = typer.Option(
        None,
        "--types",
        help="Accept only changes whose primary change_type is in this list. "
        "Comma separated, e.g. 'typo,term,case'. "
        "Allowed: typo,term,case,punct,dup,filler,restart,emphasis.",
    ),
) -> None:
    """Accept the latest or specified transcript correction proposal.

    Without --select/--types, accepts all proposed changes. With either filter,
    only the matching subset is applied — useful for accepting term corrections
    while declining filler/restart cleanups, etc.
    """
    paths, manifest, speaker_mapping = _load_command_context(project_dir, projects_dir)
    correction_proposal = run_with_cli_errors(lambda: load_correction_proposal(paths, proposal))
    selected_indices = run_with_cli_errors(
        lambda: _resolve_selected_indices(correction_proposal.proposed_changes, select, types)
    )
    summary = run_with_cli_errors(
        lambda: accept_correction_proposal(
            paths=paths,
            manifest=manifest,
            speaker_mapping=speaker_mapping,
            proposal_path=proposal,
            lexicon_db=lexicon_db,
            selected_change_indices=selected_indices,
        )
    )
    record_correction_outputs(paths.root, manifest, summary)
    run_with_cli_errors(lambda: save_manifest(paths.root, manifest))
    _echo_correction_summary(summary)


@app.command("diff")
def diff_command(
    project_dir: Path = typer.Argument(Path("."), metavar="PROJECT", file_okay=False, dir_okay=True),
    projects_dir: Optional[Path] = typer.Option(None, "--projects-dir", file_okay=False, dir_okay=True, hidden=True),
    proposal: Optional[Path] = typer.Option(None, "--proposal", exists=True, dir_okay=False, file_okay=True),
) -> None:
    """Print the latest or specified correction proposal diff for human review."""
    paths, _, _ = _load_command_context(project_dir, projects_dir)
    correction_proposal = run_with_cli_errors(lambda: load_correction_proposal(paths, proposal))
    typer.echo(correction_proposal.diff_path.read_text(encoding="utf-8"), nl=False)


_POLISH_PRIMARY_TYPES = {"typo", "term", "case", "punct", "dup", "filler", "restart", "emphasis"}


def _resolve_selected_indices(
    changes,
    select: str | None,
    types: str | None,
) -> tuple[int, ...] | None:
    """
    Resolve --select / --types into a tuple of accepted indices.

    --select takes precedence as the explicit override; --types filters by the
    primary change_type tag of each change. Returns None when both are absent
    so the underlying accept path keeps its 'accept all' semantics.
    """
    if select is None and types is None:
        return None
    indices: set[int]
    if select is not None:
        indices = _parse_select_spec(select, len(changes))
    else:
        indices = set(range(len(changes)))
    if types is not None:
        wanted = _parse_types_spec(types)
        indices = {i for i in indices if _primary_type_of(changes[i]) in wanted}
    return tuple(sorted(indices))


def _parse_select_spec(spec: str, total: int) -> set[int]:
    """Parse a comma/space-separated list with ranges into a set of indices."""
    out: set[int] = set()
    for token in spec.replace(" ", ",").split(","):
        token = token.strip()
        if not token:
            continue
        if "-" in token:
            lo_text, hi_text = token.split("-", 1)
            lo, hi = int(lo_text), int(hi_text)
            if lo > hi:
                lo, hi = hi, lo
            out.update(range(lo, hi + 1))
        else:
            out.add(int(token))
    invalid = [i for i in out if i < 0 or i >= total]
    if invalid:
        raise ValueError(f"--select contains out-of-range indices for proposal of size {total}: {sorted(invalid)}")
    return out


def _parse_types_spec(spec: str) -> set[str]:
    """Parse a comma-separated --types list and validate against allowed tags."""
    wanted = {token.strip().lower() for token in spec.split(",") if token.strip()}
    invalid = wanted - _POLISH_PRIMARY_TYPES
    if invalid:
        raise ValueError(f"--types contains unknown tags {sorted(invalid)}; allowed: {sorted(_POLISH_PRIMARY_TYPES)}")
    return wanted


def _primary_type_of(change) -> str:
    """Return the leading change_type tag from a multi-tag string like 'dup|filler'."""
    raw = (change.change_type or "").strip().lower()
    if not raw:
        return ""
    for sep in ("|", ",", "+", "/", "&"):
        if sep in raw:
            return raw.split(sep, 1)[0].strip()
    return raw


def _load_command_context(
    project_dir: Path,
    projects_dir: Path | None,
) -> tuple:
    """
    Load paths, manifest, and speaker mapping for correction commands.

    Args:
        project_dir: Project reference or path.
        projects_dir: Optional projects parent.

    Returns:
        Tuple of project paths, manifest, and speaker mapping.
    """
    resolved_project_dir = run_with_cli_errors(lambda: resolve_project_ref(project_dir, projects_dir))
    paths = project_paths(resolved_project_dir)
    manifest = run_with_cli_errors(lambda: load_manifest(paths.root))
    speaker_mapping = run_with_cli_errors(lambda: _load_speaker_mapping(paths.root))
    return paths, manifest, speaker_mapping


def _accept_or_leave_pending(
    paths,
    manifest: ProjectManifest,
    speaker_mapping: dict[int, str],
    summary: CorrectionEditSummary,
    lexicon_db: Path | None,
    yes: bool,
) -> CorrectionEditSummary:
    """
    Accept a proposal immediately, or leave it pending.

    Args:
        paths: Project paths.
        manifest: Loaded project manifest.
        speaker_mapping: Speaker id to name mapping.
        summary: Pending proposal summary.
        lexicon_db: Optional lexicon database override.
        yes: Whether to skip confirmation.

    Returns:
        Accepted summary when confirmed, otherwise the pending proposal summary.
    """
    if yes or _confirm_accept():
        return _accept_summary(paths, manifest, speaker_mapping, summary, lexicon_db)
    _echo_pending_accept_command(paths.root, summary.proposal_json_path)
    return summary


def _finish_correction_edit(
    paths,
    manifest: ProjectManifest,
    speaker_mapping: dict[int, str],
    summary: CorrectionEditSummary,
    lexicon_db: Path | None,
    yes: bool,
    *,
    auto_accept: bool,
) -> CorrectionEditSummary:
    """Print a correction summary and optionally accept a generated proposal."""
    _echo_correction_summary(summary)
    if summary.proposal_json_path is None or not auto_accept:
        return summary
    return _accept_or_leave_pending(paths, manifest, speaker_mapping, summary, lexicon_db, yes)


def _accept_summary(
    paths,
    manifest: ProjectManifest,
    speaker_mapping: dict[int, str],
    summary: CorrectionEditSummary,
    lexicon_db: Path | None,
) -> CorrectionEditSummary:
    """Apply and print one accepted pending proposal."""
    accepted = run_with_cli_errors(
        lambda: accept_correction_proposal(
            paths=paths,
            manifest=manifest,
            speaker_mapping=speaker_mapping,
            proposal_path=summary.proposal_json_path,
            lexicon_db=lexicon_db,
        )
    )
    record_correction_outputs(paths.root, manifest, accepted)
    run_with_cli_errors(lambda: save_manifest(paths.root, manifest))
    _echo_correction_summary(accepted)
    return accepted


def _echo_pending_accept_command(project_dir: Path, proposal_path: Path | None) -> None:
    """Print the follow-up command for a pending proposal."""
    typer.echo("Correction proposal left pending.")
    project_arg = shlex.quote(str(project_dir))
    proposal_arg = shlex.quote(str(proposal_path))
    typer.echo(f"Accept later: meeting-asr project correct accept {project_arg} --proposal {proposal_arg}")


def _load_speaker_mapping(project_dir: Path) -> dict[int, str]:
    """
    Load project speaker names, falling back to anonymous labels.

    Args:
        project_dir: Project root.

    Returns:
        Speaker mapping.
    """
    paths = project_paths(project_dir)
    result = load_transcript_result(paths.asr_dir / "sentences.json")
    mapping = build_default_mapping(result)
    mapping_path = paths.speakers_dir / "speaker_map.json"
    if not mapping_path.exists():
        return mapping
    payload = json.loads(mapping_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        return mapping
    for key, value in payload.items():
        try:
            speaker_id = int(key)
        except ValueError:
            continue
        mapping[speaker_id] = str(value)
    return mapping


def load_speaker_mapping_for_correction(project_dir: Path) -> dict[int, str]:
    """
    Load the speaker mapping used by correction review.

    Args:
        project_dir: Project root.

    Returns:
        Speaker id to display name mapping.
    """
    return _load_speaker_mapping(project_dir)


def _confirm_accept() -> bool:
    """
    Ask whether to accept a generated correction proposal.

    Returns:
        True when the user confirms acceptance.
    """
    try:
        return typer.confirm("Accept proposed full-document corrections?")
    except typer.Abort:
        return False


def _echo_correction_summary(summary: CorrectionEditSummary) -> None:
    """
    Print correction edit results.

    Args:
        summary: Correction edit summary.

    Returns:
        None.
    """
    label = _correction_label(summary)
    header = f"{label} accepted." if summary.accepted else f"{label} proposal ready."
    if summary.proposal_json_path is None and summary.change_count == 0:
        header = f"{label} review complete."
    typer.echo(header)
    typer.echo(f"Review file: {summary.review_path}")
    _echo_proposal_fields(summary)
    typer.echo(f"Changed sentences: {summary.change_count}")
    typer.echo(f"Sample changes: {summary.sample_change_count}")
    typer.echo(f"Proposed changes: {summary.proposed_change_count}")
    typer.echo(f"Learned contexts: {summary.learned_count}")
    typer.echo(f"Lexicon DB: {summary.lexicon_db}")
    _echo_understanding(summary)
    if summary.change_count == 0 and not summary.accepted:
        typer.echo("No corrected outputs were written.")
        return
    if summary.accepted:
        _echo_output_paths(summary)


def _correction_label(summary: CorrectionEditSummary) -> str:
    """Return the user-facing correction workflow label."""
    if summary.review_path.name.startswith("review_polish_"):
        return "Transcript polish"
    return "Vocabulary correction"


def _echo_proposal_fields(summary: CorrectionEditSummary) -> None:
    """
    Print pending proposal paths.

    Args:
        summary: Correction edit summary.

    Returns:
        None.
    """
    if summary.model:
        typer.echo(f"Correction model: {summary.model}")
    if summary.model_error:
        typer.echo(f"Model fallback: {summary.model_error}")
    if summary.proposal_path:
        typer.echo(f"Proposal file: {summary.proposal_path}")
    if summary.proposal_diff_path:
        typer.echo(f"Proposal diff: {summary.proposal_diff_path}")
    if summary.proposal_json_path:
        typer.echo(f"Proposal JSON: {summary.proposal_json_path}")


def _echo_understanding(summary: CorrectionEditSummary) -> None:
    """
    Print inferred correction understanding.

    Args:
        summary: Correction edit summary.

    Returns:
        None.
    """
    if not summary.understanding:
        return
    typer.echo("")
    typer.echo("Understanding:")
    for item in summary.understanding:
        typer.echo(f"  - {item.wrong_text} -> {item.corrected_text} ({item.proposed_count} proposed)")


def _echo_output_paths(summary: CorrectionEditSummary) -> None:
    """
    Print accepted correction output artifacts.

    Args:
        summary: Correction edit summary.

    Returns:
        None.
    """
    typer.echo("")
    typer.echo("Outputs:")
    typer.echo(f"  {summary.corrected_sentences_path}")
    typer.echo(f"  {summary.corrected_named_transcript_path}")
    typer.echo(f"  {summary.corrected_srt_path}")
    typer.echo(f"  {summary.hotwords_path}")
    typer.echo(f"  {summary.applied_path}")


def record_correction_outputs(
    project_dir: Path,
    manifest: ProjectManifest,
    summary: CorrectionEditSummary,
) -> None:
    """
    Record corrected artifacts in the project manifest.

    Args:
        project_dir: Project root.
        manifest: Loaded project manifest.
        summary: Correction edit summary.

    Returns:
        None.
    """
    manifest.status = "corrected"
    if summary.review_path.name.startswith("review_polish_"):
        _record_accepted_polish_runtime(project_dir, manifest, summary)
    for key, path in {
        "corrected_sentences": summary.corrected_sentences_path,
        "corrected_transcript": summary.corrected_transcript_path,
        "corrected_named_transcript": summary.corrected_named_transcript_path,
        "corrected_named_subtitle": summary.corrected_srt_path,
        "asr_hotwords": summary.hotwords_path,
        "vocabulary_corrections": summary.applied_path,
    }.items():
        if path is not None:
            manifest.outputs[key] = str(path.relative_to(project_dir))


def _record_accepted_polish_runtime(
    project_dir: Path,
    manifest: ProjectManifest,
    summary: CorrectionEditSummary,
) -> None:
    """Record accepted transcript polish state in project runtime metadata."""
    runtime = dict(manifest.runtime)
    runtime["polish"] = {
        "status": "accepted",
        "updated_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "model": summary.model,
        "error": summary.model_error,
        "proposed_changes": summary.proposed_change_count,
        "accepted_changes": summary.change_count,
        "proposal_json": _relative_optional_path(project_dir, summary.proposal_json_path),
        "proposal_diff": _relative_optional_path(project_dir, summary.proposal_diff_path),
    }
    manifest.runtime = runtime


def _relative_optional_path(project_dir: Path, path: Path | None) -> str | None:
    """Return a project-relative path when possible."""
    if path is None:
        return None
    try:
        return str(path.relative_to(project_dir))
    except ValueError:
        return str(path)
