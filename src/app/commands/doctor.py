"""Environment diagnostics command."""

from __future__ import annotations

import importlib.util
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from urllib.parse import urlparse
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
    fix_prompt: str | None = None

    @property
    def failed(self) -> bool:
        """Return whether this check should fail the command."""
        return self.status == "fail"

    @property
    def needs_attention(self) -> bool:
        """Return whether this check should print a repair prompt."""
        return self.status in {"fail", "warn"} and self.fix_prompt is not None


def command(
    require_oss: bool = typer.Option(False, "--require-oss"),
    check_oss_access: bool = typer.Option(False, "--check-oss-access"),
    oss_upload_probe: bool = typer.Option(False, "--oss-upload-probe"),
    require_voiceprint_embedding: bool = typer.Option(False, "--require-voiceprint-embedding"),
) -> None:
    """
    Check runtime dependencies and global config.

    Args:
        require_oss: Require OSS configuration values.
        check_oss_access: Verify OSS bucket metadata with a real request.
        oss_upload_probe: Upload, read, and delete a tiny OSS probe object.
        require_voiceprint_embedding: Require voiceprint embedding configuration.

    Returns:
        None.
    """
    effective_require_oss = require_oss or check_oss_access or oss_upload_probe or require_voiceprint_embedding
    checks = [
        _check_python(),
        _check_python_packages(require_oss=effective_require_oss),
        _check_ffmpeg(),
        _check_preview_player(),
        _check_settings(require_oss=effective_require_oss),
        _check_voiceprint_embedding_settings(required=require_voiceprint_embedding),
    ]
    if check_oss_access or oss_upload_probe:
        checks.append(_check_oss_access(upload_probe=oss_upload_probe))
    for check in checks:
        typer.echo(f"{check.status:4} {check.name}: {check.detail}")
    _echo_fix_prompts(checks)
    if any(check.failed for check in checks):
        raise typer.Exit(code=1)


def _check_python() -> CheckResult:
    """Check current Python version."""
    version = ".".join(str(part) for part in sys.version_info[:3])
    detail = f"Python {version} at {sys.executable}"
    prompt = _fix_prompt(
        "python",
        detail,
        "Use Python 3.14 or newer, then recreate the virtualenv and rerun `uv sync --all-groups`.",
        "meeting-asr doctor",
    )
    return CheckResult("python", "ok" if sys.version_info >= (3, 14) else "fail", detail, prompt)


def _check_python_packages(*, require_oss: bool) -> CheckResult:
    """Check Python packages imported by the CLI."""
    packages = ["dashscope", "requests", "typer", "dotenv"]
    if require_oss:
        packages.append("oss2")
    missing = [package for package in packages if importlib.util.find_spec(package) is None]
    if not missing:
        return CheckResult("python-packages", "ok", f"installed: {', '.join(packages)}")
    detail = f"missing: {', '.join(missing)}; run `uv sync`"
    prompt = _fix_prompt(
        "python-packages",
        detail,
        "Run `uv sync --all-groups` in the meeting-asr repository. Do not install packages into an unrelated Python.",
        "uv run meeting-asr doctor",
    )
    return CheckResult("python-packages", "fail", detail, prompt)


def _check_ffmpeg() -> CheckResult:
    """Check whether ffmpeg is available."""
    ffmpeg_path = shutil.which("ffmpeg")
    if not ffmpeg_path:
        detail = "not found in PATH; install ffmpeg"
        prompt = _fix_prompt(
            "ffmpeg",
            detail,
            "Install ffmpeg, for example `brew install ffmpeg` on macOS, and ensure it is on PATH.",
            "meeting-asr doctor",
        )
        return CheckResult("ffmpeg", "fail", detail, prompt)
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
    detail = "not found; install mpv or IINA for speaker review"
    prompt = _fix_prompt(
        "preview-player",
        detail,
        "Install a subtitle-capable preview player. Prefer `brew install mpv`; IINA is also supported.",
        "meeting-asr doctor",
    )
    return CheckResult("preview-player", "warn", detail, prompt)


def _check_settings(*, require_oss: bool) -> CheckResult:
    """Check required global config without printing secrets."""
    try:
        settings = load_settings(require_oss=require_oss)
    except ValueError as exc:
        detail = str(exc)
        prompt = _fix_prompt(
            "config",
            detail,
            (
                "Set the missing meeting-asr config key with `meeting-asr config set <key> <value>` "
                "or export the matching environment variable. Never print secrets."
            ),
            "meeting-asr doctor --require-oss" if require_oss else "meeting-asr doctor",
        )
        return CheckResult("config", "fail", detail, prompt)
    detail = f"config={settings.config_path}; dashscope.base_url={settings.dashscope_base_url}"
    if require_oss:
        detail = f"{detail}; oss.bucket_name={settings.oss_bucket_name}; oss.endpoint={settings.oss_endpoint}"
    return CheckResult("config", "ok", detail)


def _check_voiceprint_embedding_settings(*, required: bool) -> CheckResult:
    """
    Check whether voiceprint embedding has enough config to run.

    Args:
        required: Whether missing voiceprint embedding config should fail.

    Returns:
        Diagnostic check result for voiceprint embedding config.
    """
    try:
        settings = load_settings(require_oss=False)
    except ValueError as exc:
        detail = f"skipped because base config is invalid: {exc}"
        return CheckResult("voiceprint-embedding", "warn", detail)
    provider = settings.voiceprint_embedding_provider.strip().lower()
    if provider != "bailian":
        return _voiceprint_problem(
            required=required,
            detail=f"unsupported provider={settings.voiceprint_embedding_provider}; supported: bailian",
        )
    endpoint = settings.voiceprint_embedding_endpoint
    if not endpoint:
        return _voiceprint_problem(required=required, detail="voiceprint.embedding_endpoint is not configured")
    endpoint_problem = _validate_voiceprint_endpoint(endpoint)
    if endpoint_problem:
        return _voiceprint_problem(required=required, detail=endpoint_problem)
    return CheckResult("voiceprint-embedding", "ok", f"provider=bailian; endpoint={endpoint}")


def _validate_voiceprint_endpoint(endpoint: str) -> str | None:
    """
    Validate the configured voiceprint embedding endpoint shape.

    Args:
        endpoint: Configured endpoint URL.

    Returns:
        Problem text, or ``None`` when the endpoint shape is valid.
    """
    parsed = urlparse(endpoint)
    if parsed.scheme not in {"http", "https"} or not parsed.netloc:
        return f"voiceprint.embedding_endpoint must be an HTTP URL; got {endpoint}"
    if parsed.path.rstrip("/") != "/audio/embedding":
        return f"voiceprint.embedding_endpoint should end with /audio/embedding; got {endpoint}"
    return None


def _voiceprint_problem(*, required: bool, detail: str) -> CheckResult:
    """
    Build a voiceprint embedding problem result.

    Args:
        required: Whether the problem should fail the command.
        detail: Human-readable problem detail.

    Returns:
        Diagnostic result with an agent repair prompt.
    """
    status = "fail" if required else "warn"
    prompt = _fix_prompt(
        "voiceprint-embedding",
        detail,
        (
            "Configure the AnalyticDB voiceprint embedding service endpoint. Use the AI application host from "
            "AnalyticDB voiceprint retrieval, not a Tongyi vision embedding model name: "
            "`meeting-asr config set voiceprint.embedding_endpoint "
            "\"http://<adb-ai-app-host>:8100/audio/embedding\"`. Then verify OSS and voiceprint config."
        ),
        "meeting-asr doctor --require-oss --require-voiceprint-embedding",
    )
    return CheckResult("voiceprint-embedding", status, detail, prompt)


def _check_oss_access(*, upload_probe: bool) -> CheckResult:
    """Check OSS credentials with a real request."""
    oss2 = import_oss2()

    try:
        bucket = build_oss_bucket(load_settings(require_oss=True))
        if upload_probe:
            return _check_oss_upload_probe(bucket)
        bucket.get_bucket_info()
    except oss2.exceptions.OssError as exc:
        detail = _format_oss_error(exc)
        prompt = _fix_prompt(
            "oss-access",
            detail,
            (
                "Fix OSS config or bucket permissions. Check oss.access_key_id, "
                "oss.bucket_name, oss.region, and oss.endpoint."
            ),
            "meeting-asr doctor --oss-upload-probe",
        )
        return CheckResult("oss-access", "fail", detail, prompt)
    except Exception as exc:  # noqa: BLE001
        detail = str(exc)
        prompt = _fix_prompt(
            "oss-access",
            detail,
            "Fix the OSS access failure without logging secrets, then rerun the upload probe.",
            "meeting-asr doctor --oss-upload-probe",
        )
        return CheckResult("oss-access", "fail", detail, prompt)
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


def _echo_fix_prompts(checks: list[CheckResult]) -> None:
    """
    Print repair prompts for failed or warning checks.

    Args:
        checks: Diagnostic results.

    Returns:
        None.
    """
    problem_checks = [check for check in checks if check.needs_attention]
    if not problem_checks:
        return
    typer.echo("")
    typer.echo("Repair prompts:")
    for check in problem_checks:
        typer.echo(f"[{check.name}]")
        typer.echo(check.fix_prompt)


def _fix_prompt(check_name: str, detail: str, fix: str, verify: str) -> str:
    """
    Build a prompt that another model can use to repair a doctor issue.

    Args:
        check_name: Name of the failing or warning check.
        detail: Observed problem detail.
        fix: Concrete repair guidance.
        verify: Command to verify the repair.

    Returns:
        Prompt text safe to print in terminal output.
    """
    return (
        "Prompt: You are fixing `meeting-asr doctor` output. "
        f"Check `{check_name}` reported: {detail}. "
        f"Repair: {fix} "
        f"Verify with `{verify}`. "
        "Do not print or commit secrets."
    )


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
