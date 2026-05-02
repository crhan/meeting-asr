"""Project vocabulary correction commands."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

import typer

from app.core.project_models import ProjectManifest
from app.presentation.cli.errors import run_with_cli_errors
from app.project_manager import load_manifest, project_paths, resolve_project_ref, save_manifest
from app.speaker_labeling import build_default_mapping, load_transcript_result
from app.transcript_corrections import CorrectionEditOptions, CorrectionEditSummary, run_editor_correction

app = typer.Typer(add_completion=False, no_args_is_help=True, pretty_exceptions_enable=False)


@app.command("edit")
def edit_command(
    project_dir: Path = typer.Argument(Path("."), metavar="PROJECT", file_okay=False, dir_okay=True),
    projects_dir: Optional[Path] = typer.Option(None, "--projects-dir", file_okay=False, dir_okay=True),
    editor: Optional[str] = typer.Option(None, "--editor", help="Editor command. Use {file} as optional placeholder."),
    no_open: bool = typer.Option(False, "--no-open", help="Only write the review file; do not launch an editor."),
    category: str = typer.Option("unknown", "--category", help="Category for learned terms."),
    lexicon_db: Optional[Path] = typer.Option(None, "--lexicon-db", help="Override lexicon SQLite path."),
    from_original: bool = typer.Option(False, "--from-original", help="Ignore an existing corrected transcript."),
) -> None:
    """Open an editable transcript review file and learn vocabulary corrections from the diff."""
    resolved_project_dir = run_with_cli_errors(lambda: resolve_project_ref(project_dir, projects_dir))
    paths = project_paths(resolved_project_dir)
    manifest = run_with_cli_errors(lambda: load_manifest(paths.root))
    speaker_mapping = run_with_cli_errors(lambda: _load_speaker_mapping(paths.root))
    options = CorrectionEditOptions(
        editor=editor,
        open_editor=not no_open,
        category=category,
        lexicon_db=lexicon_db,
        from_original=from_original,
    )
    summary = run_with_cli_errors(
        lambda: run_editor_correction(
            paths=paths,
            manifest=manifest,
            speaker_mapping=speaker_mapping,
            options=options,
        )
    )
    if summary.change_count:
        _record_correction_outputs(paths.root, manifest, summary)
        run_with_cli_errors(lambda: save_manifest(paths.root, manifest))
    _echo_correction_summary(summary)


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


def _echo_correction_summary(summary: CorrectionEditSummary) -> None:
    """
    Print correction edit results.

    Args:
        summary: Correction edit summary.

    Returns:
        None.
    """
    typer.echo("Vocabulary correction review complete.")
    typer.echo(f"Review file: {summary.review_path}")
    typer.echo(f"Changed sentences: {summary.change_count}")
    typer.echo(f"Learned contexts: {summary.learned_count}")
    typer.echo(f"Lexicon DB: {summary.lexicon_db}")
    if summary.change_count == 0:
        typer.echo("No corrected outputs were written.")
        return
    typer.echo("")
    typer.echo("Outputs:")
    typer.echo(f"  {summary.corrected_sentences_path}")
    typer.echo(f"  {summary.corrected_named_transcript_path}")
    typer.echo(f"  {summary.corrected_srt_path}")
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
        "vocabulary_corrections": summary.applied_path,
    }.items():
        if path is not None:
            manifest.outputs[key] = str(path.relative_to(project_dir))
