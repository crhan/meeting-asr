"""Small helpers for readable CLI failures."""

from __future__ import annotations

import re
from collections.abc import Callable
from dataclasses import dataclass
from typing import TypeVar

import click
import typer
from rich.panel import Panel
from rich.text import Text

from app.presentation.cli.i18n import current_cli_language
from app.presentation.cli.output import cli_console

T = TypeVar("T")
AdviceRule = tuple[Callable[[str], bool], str, str, str]

ERROR_LABELS = {
    "en": {
        "title": "Error",
        "usage": "Usage",
        "problem": "Problem",
        "next": "Next step",
    },
    "zh": {"title": "错误", "usage": "用法", "problem": "问题", "next": "下一步"},
}

_NO_SUCH_COMMAND_RE = re.compile(r"No such command '([^']+)'\.")


@dataclass(frozen=True, slots=True)
class CliErrorAdvice:
    """Actionable advice for a failed CLI command."""

    problem: str
    doctor_command: str
    detail: str
    retry_exhausted: bool = False


def run_with_cli_errors(operation: Callable[[], T]) -> T:
    """
    Run an operation and convert unexpected exceptions into readable CLI errors.

    Args:
        operation: Callable to execute.

    Returns:
        The callable result.
    """
    try:
        return operation()
    except click.ClickException, typer.Exit:
        raise
    except Exception as exc:  # noqa: BLE001
        _echo_cli_error(exc)
        raise typer.Exit(code=1) from exc


def show_click_usage_error(exc: click.ClickException) -> None:
    """
    Print a localized, actionable parse error for human-facing CLI use.

    Args:
        exc: Click or Typer parse exception.

    Returns:
        None.
    """
    lang = current_cli_language()
    ctx = getattr(exc, "ctx", None)
    console = cli_console(stderr=True)
    usage = _localized_usage(ctx, lang)
    if usage:
        console.print(Text(usage, style="bold"))
    plain_message = _plain_compat_message(exc)
    if plain_message:
        click.echo(f"Error: {plain_message}", err=True)
    console.print(_usage_error_panel(exc, ctx, lang))


def _usage_error_panel(
    exc: click.ClickException, ctx: click.Context | None, lang: str
) -> Panel:
    """
    Build the localized usage error panel.

    Args:
        exc: Click parse exception.
        ctx: Click context, when available.
        lang: Resolved display language.

    Returns:
        Rich panel describing the problem and the next command to run.
    """
    labels = ERROR_LABELS[lang]
    lines = [
        f"{labels['problem']}: {_localized_click_message(exc, lang)}",
        f"{labels['next']}: {_localized_next_step(ctx, lang)}",
    ]
    return Panel(Text("\n".join(lines)), title=labels["title"], border_style="red")


def _localized_usage(ctx: click.Context | None, lang: str) -> str | None:
    """
    Return a localized usage line for a Click context.

    Args:
        ctx: Click context.
        lang: Resolved display language.

    Returns:
        Usage text, or ``None`` when context is unavailable.
    """
    if ctx is None:
        return None
    pieces = " ".join(ctx.command.collect_usage_pieces(ctx))
    suffix = f" {pieces}" if pieces else ""
    return f"{ERROR_LABELS[lang]['usage']}: {_context_command_path(ctx)}{suffix}"


def _localized_next_step(ctx: click.Context | None, lang: str) -> str:
    """
    Return the next command a user should run after a parse error.

    Args:
        ctx: Click context.
        lang: Resolved display language.

    Returns:
        Actionable next-step text.
    """
    command = (
        f"{_context_command_path(ctx)} -h" if ctx is not None else "meeting-asr -h"
    )
    if lang == "zh":
        return f"运行 `{command}` 查看可用参数和命令。"
    return f"Run `{command}` to see available options and commands."


def _localized_click_message(exc: click.ClickException, lang: str) -> str:
    """
    Return a localized message for common Click parse exceptions.

    Args:
        exc: Click parse exception.
        lang: Resolved display language.

    Returns:
        Human-readable problem text.
    """
    if lang != "zh":
        return _english_click_message(exc)
    if isinstance(exc, click.NoSuchOption):
        return f"没有这个选项：{exc.option_name}"
    if isinstance(exc, click.MissingParameter):
        return _zh_missing_parameter(exc)
    if isinstance(exc, click.BadParameter):
        return _zh_bad_parameter(exc)
    match = _NO_SUCH_COMMAND_RE.fullmatch(str(exc).strip())
    if match:
        return f"没有这个命令：{match.group(1)}"
    return str(exc)


def _plain_compat_message(exc: click.ClickException) -> str | None:
    """
    Return a plain compatibility line for application-raised BadParameter errors.

    Args:
        exc: Click parse exception.

    Returns:
        Plain message when existing callers may rely on contiguous text.
    """
    if not isinstance(exc, click.BadParameter):
        return None
    if exc.param is not None or getattr(exc, "param_hint", None):
        return None
    return _bad_parameter_detail(exc)


def _english_click_message(exc: click.ClickException) -> str:
    """
    Return a stable English parse error message.

    Args:
        exc: Click parse exception.

    Returns:
        Human-readable English problem text.
    """
    if isinstance(exc, click.NoSuchOption):
        return f"No such option: {exc.option_name}"
    if isinstance(exc, click.MissingParameter):
        return f"Missing required {_parameter_kind(exc)}: {_parameter_display_name(exc.param)}"
    if isinstance(exc, click.BadParameter):
        return f"Invalid value for {_bad_parameter_name(exc)}: {_bad_parameter_detail(exc)}"
    match = _NO_SUCH_COMMAND_RE.fullmatch(str(exc).strip())
    if match:
        return f"No such command: {match.group(1)}"
    return str(exc)


def _zh_missing_parameter(exc: click.MissingParameter) -> str:
    """
    Return a Chinese message for a missing parameter.

    Args:
        exc: Missing parameter exception.

    Returns:
        Localized problem text.
    """
    kind = "选项" if _parameter_kind(exc) == "option" else "参数"
    return f"缺少必填{kind}：{_parameter_display_name(exc.param)}"


def _zh_bad_parameter(exc: click.BadParameter) -> str:
    """
    Return a Chinese message for an invalid parameter value.

    Args:
        exc: Bad parameter exception.

    Returns:
        Localized problem text.
    """
    detail = _bad_parameter_detail(exc)
    if "CLI language must be one of" in detail:
        detail = "语言必须是 auto、en 或 zh。"
    return f"参数值无效：{_bad_parameter_name(exc)}。{detail}"


def _parameter_kind(exc: click.MissingParameter) -> str:
    """
    Return whether a missing parameter is an option or argument.

    Args:
        exc: Missing parameter exception.

    Returns:
        ``option`` or ``argument``.
    """
    param = exc.param
    return "option" if isinstance(param, click.Option) else "argument"


def _parameter_display_name(param: click.Parameter | None) -> str:
    """
    Return the user-facing name for a Click parameter.

    Args:
        param: Click parameter.

    Returns:
        Display name.
    """
    if param is None:
        return "value"
    if isinstance(param, click.Option) and param.opts:
        return param.opts[0]
    return param.human_readable_name


def _bad_parameter_detail(exc: click.BadParameter) -> str:
    """
    Return the BadParameter detail without repeating Click's prefix.

    Args:
        exc: Bad parameter exception.

    Returns:
        Detail text.
    """
    message = str(exc).strip()
    prefix = f"Invalid value for {_bad_parameter_name(exc)}: "
    return message.removeprefix(prefix)


def _bad_parameter_name(exc: click.BadParameter) -> str:
    """
    Return the user-facing name for a bad parameter.

    Args:
        exc: Bad parameter exception.

    Returns:
        Display name.
    """
    hint = getattr(exc, "param_hint", None)
    if isinstance(hint, str) and hint:
        return hint
    return _parameter_display_name(exc.param)


def _context_command_path(ctx: click.Context) -> str:
    """
    Return a stable command path for real CLI and CliRunner tests.

    Args:
        ctx: Click context.

    Returns:
        Command path starting with ``meeting-asr``.
    """
    command_path = ctx.command_path
    if command_path == "root":
        return "meeting-asr"
    if command_path.startswith("root "):
        return f"meeting-asr{command_path[4:]}"
    return command_path


def build_cli_error_advice(exc: Exception) -> CliErrorAdvice | None:
    """
    Build doctor guidance for configuration, environment, and exhausted transient errors.

    Args:
        exc: Exception raised by the failed command.

    Returns:
        Advice when the CLI can suggest a useful next step.
    """
    messages = _exception_messages(exc)
    lowered = "\n".join(messages).lower()
    retry_exhausted = _looks_like_retry_exhausted(lowered)
    for matcher, problem, doctor_command, detail in _ADVICE_RULES:
        if not matcher(lowered):
            continue
        return CliErrorAdvice(
            problem,
            doctor_command,
            detail,
            retry_exhausted=retry_exhausted,
        )
    if retry_exhausted:
        return CliErrorAdvice(
            "transient network or service failure",
            "meeting-asr doctor",
            "The operation was retried because it looked transient, but it still failed.",
            retry_exhausted=True,
        )
    return None


def _echo_cli_error(exc: Exception) -> None:
    """
    Print a readable CLI error and optional repair advice.

    Args:
        exc: Exception raised by the failed command.

    Returns:
        None.
    """
    typer.echo(f"Error: {exc}", err=True)
    advice = build_cli_error_advice(exc)
    if advice is None:
        return
    if advice.retry_exhausted:
        typer.echo(
            "Retry: this looked transient and was already retried before failing.",
            err=True,
        )
    typer.echo(f"Problem: {advice.problem}", err=True)
    typer.echo(f"Diagnosis: {advice.detail}", err=True)
    typer.echo(f"Next step: run `{advice.doctor_command}`.", err=True)
    typer.echo(
        "Agent prompt: Fix the meeting-asr failure above. "
        f"First run `{advice.doctor_command}`, apply the emitted Repair Prompt, "
        "then rerun the failed command. Do not print or commit secrets.",
        err=True,
    )


def _exception_messages(exc: Exception) -> list[str]:
    """
    Collect messages from an exception chain.

    Args:
        exc: Root exception.

    Returns:
        Non-empty message list.
    """
    messages: list[str] = []
    current: BaseException | None = exc
    while current is not None:
        text = str(current).strip()
        if text:
            messages.append(text)
        current = current.__cause__ or current.__context__
    return messages or [exc.__class__.__name__]


def _looks_like_retry_exhausted(lowered_message: str) -> bool:
    """
    Return whether retry already happened and failed.

    Args:
        lowered_message: Lowercase combined exception message.

    Returns:
        True when retry wrapper exhausted all attempts.
    """
    return "retryable attempts" in lowered_message


def _looks_like_voiceprint_dependency(lowered_message: str) -> bool:
    """
    Return whether the error is a local voiceprint dependency problem.

    Args:
        lowered_message: Lowercase combined exception message.

    Returns:
        True for missing SpeechBrain/Torch dependencies.
    """
    markers = (
        "local-speechbrain voiceprint embedding requires standard dependencies",
        "local-speechbrain voiceprint embedding requires optional dependencies",
    )
    return any(marker in lowered_message for marker in markers)


def _looks_like_voiceprint_config(lowered_message: str) -> bool:
    """
    Return whether the error is a voiceprint configuration problem.

    Args:
        lowered_message: Lowercase combined exception message.

    Returns:
        True for unsupported local voiceprint configuration failures.
    """
    markers = ("unsupported voiceprint embedding provider",)
    return any(marker in lowered_message for marker in markers)


def _looks_like_oss_config_or_access(lowered_message: str) -> bool:
    """
    Return whether the error is likely caused by OSS configuration or access.

    Args:
        lowered_message: Lowercase combined exception message.

    Returns:
        True for OSS config, auth, bucket, or signed URL failures.
    """
    markers = (
        "missing required oss config",
        "oss.access_key_id",
        "oss.access_key_secret",
        "oss.bucket_name",
        "oss.region",
        "oss.endpoint",
        "oss request failed",
        "signaturedoesnotmatch",
        "accessdenied",
        "invalidaccesskeyid",
        "nosuchbucket",
        "file_403_forbidden",
    )
    return any(marker in lowered_message for marker in markers)


def _looks_like_dashscope_config_or_access(lowered_message: str) -> bool:
    """
    Return whether the error is likely caused by DashScope configuration or access.

    Args:
        lowered_message: Lowercase combined exception message.

    Returns:
        True for missing API key or authorization failures.
    """
    if "dashscope.api_key" in lowered_message:
        return True
    if "dashscope" not in lowered_message:
        return False
    auth_markers = (
        "http 401",
        "http 403",
        "unauthorized",
        "forbidden",
        "invalid api key",
    )
    return any(marker in lowered_message for marker in auth_markers)


def _looks_like_local_environment(lowered_message: str) -> bool:
    """
    Return whether the error is likely caused by missing local tools.

    Args:
        lowered_message: Lowercase combined exception message.

    Returns:
        True for ffmpeg or preview player problems.
    """
    markers = (
        "ffmpeg was not found",
        "no supported subtitle preview player found",
        "no supported audio preview player found",
    )
    return any(marker in lowered_message for marker in markers)


_ADVICE_RULES: tuple[AdviceRule, ...] = (
    (
        _looks_like_voiceprint_dependency,
        "voiceprint embedding dependencies",
        "meeting-asr doctor --require-voiceprint-embedding",
        "The local voiceprint provider is missing standard dependencies.",
    ),
    (
        _looks_like_voiceprint_config,
        "voiceprint embedding configuration",
        "meeting-asr doctor --require-voiceprint-embedding",
        "The local voiceprint embedding configuration is invalid.",
    ),
    (
        _looks_like_oss_config_or_access,
        "OSS configuration or access",
        "meeting-asr doctor --oss-upload-probe",
        "The command needs OSS config, bucket access, or signed URL verification.",
    ),
    (
        _looks_like_dashscope_config_or_access,
        "DashScope configuration or access",
        "meeting-asr doctor",
        "The command needs a valid DashScope API key and service access.",
    ),
    (
        _looks_like_local_environment,
        "local environment",
        "meeting-asr doctor",
        "A required local dependency or preview tool is missing.",
    ),
)
