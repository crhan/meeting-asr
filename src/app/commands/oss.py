"""OSS utility commands."""

from __future__ import annotations

from pathlib import Path

import typer

from app.presentation.cli.errors import run_with_cli_errors
from app.config import load_settings
from app.oss_lifecycle import set_lifecycle_rule
from app.uploader import (
    SIGNED_URL_EXPIRES_SECONDS,
    build_oss_bucket,
    upload_file_to_oss,
)

app = typer.Typer(add_completion=False, no_args_is_help=True, pretty_exceptions_enable=False)
lifecycle_app = typer.Typer(add_completion=False, no_args_is_help=True, pretty_exceptions_enable=False)
app.add_typer(lifecycle_app, name="lifecycle", help="Configure OSS lifecycle rules.")


@app.command("upload")
def upload(
    local_path: Path = typer.Argument(..., exists=True, file_okay=True, dir_okay=False),
    object_name: str | None = typer.Option(None, "--object-name"),
    expires_seconds: int = typer.Option(SIGNED_URL_EXPIRES_SECONDS, "--expires-seconds", min=60),
) -> None:
    """Upload a local file to OSS and print a signed URL."""
    settings = run_with_cli_errors(lambda: load_settings(require_oss=True))
    url = run_with_cli_errors(
        lambda: upload_file_to_oss(
            local_path,
            object_name=object_name,
            settings=settings,
            expires_seconds=expires_seconds,
        )
    )
    typer.echo(url)


@app.command("presign")
def presign(
    oss_uri: str = typer.Argument(..., help="OSS URI like oss://bucket/path/to/object"),
    expires_seconds: int = typer.Option(SIGNED_URL_EXPIRES_SECONDS, "--expires-seconds", min=60),
) -> None:
    """Create a signed GET URL for an existing OSS object."""
    bucket_name, object_key = _parse_oss_uri(oss_uri)
    settings = run_with_cli_errors(lambda: load_settings(require_oss=True))
    if settings.oss_bucket_name and bucket_name != settings.oss_bucket_name:
        raise typer.BadParameter("OSS URI bucket must match configured oss.bucket_name.")
    bucket = run_with_cli_errors(lambda: build_oss_bucket(settings))
    typer.echo(bucket.sign_url("GET", object_key, expires_seconds, slash_safe=True))


@lifecycle_app.command("set")
def lifecycle_set(
    prefix: str = typer.Option("meeting-asr/", "--prefix"),
    days: int = typer.Option(7, "--days", min=1),
    rule_id: str = typer.Option("meeting-asr-auto-delete", "--rule-id"),
) -> None:
    """Set an OSS lifecycle rule that deletes matching objects after N days."""
    settings = run_with_cli_errors(lambda: load_settings(require_oss=True))
    run_with_cli_errors(lambda: set_lifecycle_rule(settings, prefix=prefix, days=days, rule_id=rule_id))
    typer.echo(f"Lifecycle rule set: prefix={prefix}, days={days}, rule_id={rule_id}")
    typer.echo("Deletion is based on object age, not last access time.")


def _parse_oss_uri(value: str) -> tuple[str, str]:
    """Parse oss://bucket/key URI."""
    if not value.startswith("oss://"):
        raise typer.BadParameter("Expected OSS URI: oss://bucket/path/to/object")
    rest = value[len("oss://") :]
    if "/" not in rest:
        raise typer.BadParameter("OSS URI must include an object key.")
    bucket, key = rest.split("/", 1)
    if not bucket or not key:
        raise typer.BadParameter("OSS URI must include bucket and object key.")
    return bucket, key
