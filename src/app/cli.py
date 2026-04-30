"""Root CLI for Meeting-ASR workflows."""

from __future__ import annotations

import typer
from typer.completion import completion_init

from app.commands import audio, completion, config, doctor, oss, project, transcript

completion_init()

app = typer.Typer(
    add_completion=False,
    no_args_is_help=True,
    pretty_exceptions_enable=False,
    help="CLI for DashScope Fun-ASR meeting transcription workflows.",
)

app.command("doctor")(doctor.command)
app.add_typer(config.app, name="config", help="Manage global XDG configuration.")
app.add_typer(project.app, name="project", help="Manage project-based transcription workflows.")
app.add_typer(transcript.app, name="transcript", help="View project transcript artifacts.")
app.add_typer(audio.app, name="audio", help="Prepare local audio for ASR.")
app.add_typer(oss.app, name="oss", help="Upload, sign, and configure OSS objects.")
app.add_typer(completion.app, name="completion", help="Generate or install shell completion scripts.")


def main() -> None:
    """Run the root Typer app."""
    app()


if __name__ == "__main__":
    main()
