"""Project vocabulary correction commands."""

from __future__ import annotations

import json
import shlex
from pathlib import Path
from typing import Optional

import typer

from app.core.project_models import ProjectManifest
from app.presentation.cli.errors import run_with_cli_errors
from app.project_manager import load_manifest, project_paths, resolve_project_ref, save_manifest
from app.speaker_labeling import build_default_mapping, load_transcript_result
from app.transcript_corrections import (
    CorrectionEditOptions,
    CorrectionEditSummary,
    accept_correction_proposal,
    prepare_editor_correction,
)

app = typer.Typer(add_completion=False, no_args_is_help=True, pretty_exceptions_enable=False)


@app.command("edit")
def edit_command(
    project_dir: Path = typer.Argument(Path("."), metavar="PROJECT", file_okay=False, dir_okay=True),
    projects_dir: Optional[Path] = typer.Option(None, "--projects-dir", file_okay=False, dir_okay=True),
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
    _echo_correction_summary(summary)
    if summary.proposal_json_path is None or no_open:
        return
    _accept_or_leave_pending(paths, manifest, speaker_mapping, summary, lexicon_db, yes)


@app.command("accept")
def accept_command(
    project_dir: Path = typer.Argument(Path("."), metavar="PROJECT", file_okay=False, dir_okay=True),
    projects_dir: Optional[Path] = typer.Option(None, "--projects-dir", file_okay=False, dir_okay=True),
    proposal: Optional[Path] = typer.Option(None, "--proposal", exists=True, dir_okay=False, file_okay=True),
    lexicon_db: Optional[Path] = typer.Option(None, "--lexicon-db", help="Override lexicon SQLite path."),
) -> None:
    """Accept the latest or specified vocabulary correction proposal."""
    paths, manifest, speaker_mapping = _load_command_context(project_dir, projects_dir)
    summary = run_with_cli_errors(
        lambda: accept_correction_proposal(
            paths=paths,
            manifest=manifest,
            speaker_mapping=speaker_mapping,
            proposal_path=proposal,
            lexicon_db=lexicon_db,
        )
    )
    _record_correction_outputs(paths.root, manifest, summary)
    run_with_cli_errors(lambda: save_manifest(paths.root, manifest))
    _echo_correction_summary(summary)


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
) -> None:
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
        None.
    """
    if yes or _confirm_accept():
        _accept_summary(paths, manifest, speaker_mapping, summary, lexicon_db)
        return
    _echo_pending_accept_command(paths.root, summary.proposal_json_path)


def _accept_summary(
    paths,
    manifest: ProjectManifest,
    speaker_mapping: dict[int, str],
    summary: CorrectionEditSummary,
    lexicon_db: Path | None,
) -> None:
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
    _record_correction_outputs(paths.root, manifest, accepted)
    run_with_cli_errors(lambda: save_manifest(paths.root, manifest))
    _echo_correction_summary(accepted)


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
    header = "Vocabulary correction accepted." if summary.accepted else "Vocabulary correction proposal ready."
    if summary.proposal_json_path is None and summary.change_count == 0:
        header = "Vocabulary correction review complete."
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


def _record_correction_outputs(
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
