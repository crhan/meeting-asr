"""Environment diagnostics command."""

from __future__ import annotations

import importlib.util
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from uuid import uuid4

import typer

from app.config import load_settings
from app.uploader import build_oss_bucket, import_oss2


@dataclass(slots=True)
class CheckResult:
    """One diagnostic check result."""

    name: str
    status: str
    detail: str

    @property
    def failed(self) -> bool:
        """Return whether this check should fail the command."""
        return self.status == "fail"


def command(
    require_oss: bool = typer.Option(False, "--require-oss"),
    check_oss_access: bool = typer.Option(False, "--check-oss-access"),
    oss_upload_probe: bool = typer.Option(False, "--oss-upload-probe"),
) -> None:
    """Check runtime dependencies and global config."""
    effective_require_oss = require_oss or check_oss_access or oss_upload_probe
    checks = [
        _check_python(),
        _check_python_packages(require_oss=effective_require_oss),
        _check_ffmpeg(),
        _check_preview_player(),
        _check_settings(require_oss=effective_require_oss),
    ]
    if check_oss_access or oss_upload_probe:
        checks.append(_check_oss_access(upload_probe=oss_upload_probe))
    for check in checks:
        typer.echo(f"{check.status:4} {check.name}: {check.detail}")
    if any(check.failed for check in checks):
        raise typer.Exit(code=1)


def _check_python() -> CheckResult:
    """Check current Python version."""
    version = ".".join(str(part) for part in sys.version_info[:3])
    detail = f"Python {version} at {sys.executable}"
    return CheckResult("python", "ok" if sys.version_info >= (3, 14) else "fail", detail)


def _check_python_packages(*, require_oss: bool) -> CheckResult:
    """Check Python packages imported by the CLI."""
    packages = ["dashscope", "requests", "typer", "dotenv"]
    if require_oss:
        packages.append("oss2")
    missing = [package for package in packages if importlib.util.find_spec(package) is None]
    if not missing:
        return CheckResult("python-packages", "ok", f"installed: {', '.join(packages)}")
    return CheckResult("python-packages", "fail", f"missing: {', '.join(missing)}; run `uv sync`")


def _check_ffmpeg() -> CheckResult:
    """Check whether ffmpeg is available."""
    ffmpeg_path = shutil.which("ffmpeg")
    if not ffmpeg_path:
        return CheckResult("ffmpeg", "fail", "not found in PATH; install ffmpeg")
    version = _command_first_line([ffmpeg_path, "-version"])
    return CheckResult("ffmpeg", "ok", f"{ffmpeg_path}; {version}")


def _check_preview_player() -> CheckResult:
    """Check whether a subtitle-capable preview player is available."""
    mpv = shutil.which("mpv")
    if mpv:
        return CheckResult("preview-player", "ok", f"mpv at {mpv}")
    iina = _find_iina_cli()
    if iina:
        return CheckResult("preview-player", "ok", f"IINA at {iina}")
    return CheckResult("preview-player", "warn", "not found; install mpv or IINA for speaker review")


def _check_settings(*, require_oss: bool) -> CheckResult:
    """Check required global config without printing secrets."""
    try:
        settings = load_settings(require_oss=require_oss)
    except ValueError as exc:
        return CheckResult("config", "fail", str(exc))
    detail = f"config={settings.config_path}; dashscope.base_url={settings.dashscope_base_url}"
    if require_oss:
        detail = f"{detail}; oss.bucket_name={settings.oss_bucket_name}; oss.endpoint={settings.oss_endpoint}"
    return CheckResult("config", "ok", detail)


def _check_oss_access(*, upload_probe: bool) -> CheckResult:
    """Check OSS credentials with a real request."""
    oss2 = import_oss2()

    try:
        bucket = build_oss_bucket(load_settings(require_oss=True))
        if upload_probe:
            return _check_oss_upload_probe(bucket)
        bucket.get_bucket_info()
    except oss2.exceptions.OssError as exc:
        return CheckResult("oss-access", "fail", _format_oss_error(exc))
    except Exception as exc:  # noqa: BLE001
        return CheckResult("oss-access", "fail", str(exc))
    return CheckResult("oss-access", "ok", "bucket metadata request succeeded; no object uploaded")


def _check_oss_upload_probe(bucket: Any) -> CheckResult:
    """Upload a tiny probe object, verify signed GET, and delete it."""
    import requests

    object_name = f"meeting-asr/doctor-probe/{uuid4().hex}.txt"
    payload = b"meeting-asr doctor oss probe\n"
    try:
        bucket.put_object(object_name, payload)
        response = requests.get(bucket.sign_url("GET", object_name, 300, slash_safe=True), timeout=10)
        response.raise_for_status()
        ok = response.content == payload
    finally:
        _delete_probe_object(bucket, object_name)
    return CheckResult("oss-upload-probe", "ok" if ok else "fail", "put_object + signed GET succeeded")


def _delete_probe_object(bucket: Any, object_name: str) -> None:
    """Delete a probe object."""
    try:
        bucket.delete_object(object_name)
    except Exception:
        pass


def _format_oss_error(exc: Any) -> str:
    """Format OSS errors without secrets."""
    return f"OSS request failed: status={getattr(exc, 'status', None)}, code={getattr(exc, 'code', None)}"


def _find_iina_cli() -> str | None:
    """Find IINA CLI."""
    cli = shutil.which("iina")
    if cli:
        return cli
    app_cli = Path("/Applications/IINA.app/Contents/MacOS/iina-cli")
    return str(app_cli) if app_cli.exists() else None


def _command_first_line(command: list[str]) -> str:
    """Run a command and return first output line."""
    completed = subprocess.run(command, capture_output=True, text=True, check=False)
    output = (completed.stdout or completed.stderr).strip()
    return output.splitlines()[0] if output else ""
