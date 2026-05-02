"""Root CLI for Meeting-ASR workflows."""

from __future__ import annotations

from importlib.metadata import PackageNotFoundError
from importlib.metadata import version as package_version

import typer
from typer.completion import completion_init

from app.commands import completion, config, doctor, lexicon, oss, paths, project, voiceprint
from app.presentation.cli.output import configure_cli_output

completion_init()

ROOT_HELP = """Project-based CLI for DashScope meeting transcription workflows.

Quick start:
  meeting-asr project run <video>
  meeting-asr project review <project-id-or-path>
  meeting-asr project transcript show <project-id-or-path> --kind corrected

Inspect state:
  meeting-asr project list
  meeting-asr paths
  meeting-asr doctor
"""

app = typer.Typer(
    add_completion=False,
    no_args_is_help=True,
    pretty_exceptions_enable=False,
    help=ROOT_HELP,
)


def _version_callback(value: bool) -> None:
    """
    Print package version and exit when requested.

    Args:
        value: Whether the eager ``--version`` flag was passed.

    Returns:
        None.
    """
    if not value:
        return
    typer.echo(f"meeting-asr {_installed_version()}")
    raise typer.Exit()


def _installed_version() -> str:
    """
    Return the installed package version.

    Returns:
        Package version, or a local-development fallback.
    """
    try:
        return package_version("meeting-asr")
    except PackageNotFoundError:
        return "0.0.0+local"


@app.callback()
def root(
    version: bool = typer.Option(
        False,
        "--version",
        callback=_version_callback,
        is_eager=True,
        help="Show version and exit.",
    ),
    no_color: bool = typer.Option(False, "--no-color", help="Disable colored Rich output."),
) -> None:
    """Configure root command options."""
    configure_cli_output(no_color=no_color)


app.command("doctor")(doctor.command)
app.add_typer(config.app, name="config", help="Manage global XDG configuration.")
app.add_typer(project.app, name="project", help="Manage project-based transcription workflows.")
app.add_typer(voiceprint.app, name="voiceprint", help="Manage the cross-project voiceprint registry.")
app.add_typer(lexicon.app, name="lexicon", help="Manage the cross-project correction lexicon.")
app.add_typer(oss.app, name="oss", help="Upload, sign, and configure OSS objects.")
app.add_typer(completion.app, name="completion", help="Generate or install shell completion scripts.")
app.command("paths")(paths.command)


def main() -> None:
    """Run the root Typer app."""
    app()


if __name__ == "__main__":
    main()
