"""Global configuration subcommands."""

from __future__ import annotations

from pathlib import Path

import typer

from app.presentation.cli.errors import run_with_cli_errors
from app.presentation.cli.json_output import emit_json
from app.presentation.cli.typer_context import HELP_CONTEXT
from app.completion_helpers import complete_config_key
from app.config import (
    CONFIG_KEYS,
    get_config_path,
    import_env_file,
    set_config_value,
    unset_config_value,
    visible_config_items,
)

app = typer.Typer(
    add_completion=False,
    context_settings=HELP_CONTEXT,
    no_args_is_help=True,
    pretty_exceptions_enable=False,
)


@app.command("path")
def path_command() -> None:
    """Print the XDG global config file path."""
    typer.echo(get_config_path())


@app.command("show")
def show(
    reveal: bool = typer.Option(False, "--reveal", help="Show secret values."),
    as_json: bool = typer.Option(False, "--json", help="Print machine-readable JSON."),
) -> None:
    """Show configured values with secrets masked by default."""
    items = run_with_cli_errors(lambda: visible_config_items(reveal=reveal))
    if as_json:
        emit_json(
            {
                "config_file": get_config_path(),
                "revealed": reveal,
                "values": {key: value for key, value in items},
            }
        )
        return
    typer.echo(f"Config file: {get_config_path()}")
    for key, value in items:
        typer.echo(f"{key}={value}")


@app.command("keys")
def keys() -> None:
    """List supported config keys."""
    for config_key in CONFIG_KEYS:
        typer.echo(config_key.name)


@app.command("set")
def set_command(
    key: str = typer.Argument(
        ...,
        help="Config key, for example dashscope.api_key.",
        autocompletion=complete_config_key,
    ),
    value: str = typer.Argument(..., help="Config value."),
) -> None:
    """Set one global config value."""
    normalized_key, written_path = run_with_cli_errors(lambda: set_config_value(key, value))
    typer.echo(f"Set {normalized_key} in {written_path}")


@app.command("unset")
def unset_command(
    key: str = typer.Argument(..., help="Config key to remove.", autocompletion=complete_config_key),
) -> None:
    """Unset one global config value."""
    normalized_key, written_path = run_with_cli_errors(lambda: unset_config_value(key))
    typer.echo(f"Unset {normalized_key} in {written_path}")


@app.command("import-env")
def import_env(
    env_file: Path = typer.Argument(Path(".env"), help="Legacy dotenv file to import."),
    overwrite: bool = typer.Option(False, "--overwrite", help="Replace existing config values."),
) -> None:
    """Import a legacy .env file into the XDG global config."""
    imported_count, written_path = run_with_cli_errors(lambda: import_env_file(env_file, overwrite=overwrite))
    typer.echo(f"Imported {imported_count} value(s) into {written_path}")
