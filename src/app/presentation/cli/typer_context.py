"""Shared Typer settings and help classes for CLI compatibility.

Typer 0.26 vendored Click into its private ``typer._click`` package and dropped
the external ``click`` dependency. This module therefore customizes the CLI
through Typer's PUBLIC surface only: it subclasses the public
``typer.core.TyperGroup``/``TyperCommand`` and overrides their public
``main``/``parse_args``/``format_help`` methods. The objects Typer hands those
methods (the context and parse exceptions) are consumed by duck-typing — never
by importing private classes.
"""

from __future__ import annotations

import sys
from collections.abc import Sequence
from typing import Any

import typer
from typer.core import TyperCommand
from typer.core import TyperGroup

HELP_CONTEXT = {"help_option_names": ["-h", "--help"]}


class LocalizedTyperGroup(TyperGroup):
    """Typer group that renders Meeting-ASR localized Rich help."""

    def main(
        self,
        args: Sequence[str] | None = None,
        prog_name: str | None = None,
        complete_var: str | None = None,
        standalone_mode: bool = True,
        windows_expand_args: bool = True,
        **extra: Any,
    ) -> Any:
        """
        Run the command and render Meeting-ASR parse errors.

        Args:
            args: Command-line arguments.
            prog_name: Program name for usage output.
            complete_var: Completion environment variable.
            standalone_mode: Whether to exit the process on completion.
            windows_expand_args: Whether arguments should be expanded on Windows.
            **extra: Extra context values forwarded to Typer.

        Returns:
            Command result when ``standalone_mode`` is false.
        """
        if not standalone_mode:
            return super().main(
                args=args,
                prog_name=prog_name,
                complete_var=complete_var,
                standalone_mode=False,
                windows_expand_args=windows_expand_args,
                **extra,
            )
        _run_standalone_with_localized_errors(
            self,
            args=args,
            prog_name=prog_name,
            complete_var=complete_var,
            windows_expand_args=windows_expand_args,
            extra=extra,
        )

    def parse_args(self, ctx: Any, args: list[str]) -> list[str]:
        """
        Show help with exit code 0 when a group is called without a command.

        Args:
            ctx: Command context (duck-typed).
            args: Remaining command-line arguments.

        Returns:
            Remaining parsed arguments.
        """
        if not args and self.no_args_is_help and not ctx.resilient_parsing:
            self.format_help(ctx, ctx.make_formatter())
            raise typer.Exit(0)
        try:
            return super().parse_args(ctx, args)
        except Exception as exc:  # noqa: BLE001
            if not _looks_like_usage_error(exc):
                raise
            _show_help_if_requested(self, ctx, args, exc)
            raise

    def format_help(self, ctx: Any, formatter: Any) -> None:
        """
        Render localized help through the Meeting-ASR renderer.

        Args:
            ctx: Command context handed in by Typer (duck-typed).
            formatter: Help formatter kept for API compatibility (unused).

        Returns:
            None.
        """
        from app.presentation.cli.help import render_help

        render_help(self, _command_path(ctx))


class LocalizedTyperCommand(TyperCommand):
    """Typer command that renders Meeting-ASR localized Rich help."""

    def main(
        self,
        args: Sequence[str] | None = None,
        prog_name: str | None = None,
        complete_var: str | None = None,
        standalone_mode: bool = True,
        windows_expand_args: bool = True,
        **extra: Any,
    ) -> Any:
        """
        Run the command and render Meeting-ASR parse errors.

        Args:
            args: Command-line arguments.
            prog_name: Program name for usage output.
            complete_var: Completion environment variable.
            standalone_mode: Whether to exit the process on completion.
            windows_expand_args: Whether arguments should be expanded on Windows.
            **extra: Extra context values forwarded to Typer.

        Returns:
            Command result when ``standalone_mode`` is false.
        """
        if not standalone_mode:
            return super().main(
                args=args,
                prog_name=prog_name,
                complete_var=complete_var,
                standalone_mode=False,
                windows_expand_args=windows_expand_args,
                **extra,
            )
        _run_standalone_with_localized_errors(
            self,
            args=args,
            prog_name=prog_name,
            complete_var=complete_var,
            windows_expand_args=windows_expand_args,
            extra=extra,
        )

    def parse_args(self, ctx: Any, args: list[str]) -> list[str]:
        """
        Show help with exit code 0 when a command is configured that way.

        Args:
            ctx: Command context (duck-typed).
            args: Remaining command-line arguments.

        Returns:
            Remaining parsed arguments.
        """
        if not args and self.no_args_is_help and not ctx.resilient_parsing:
            self.format_help(ctx, ctx.make_formatter())
            raise typer.Exit(0)
        try:
            return super().parse_args(ctx, args)
        except Exception as exc:  # noqa: BLE001
            if not _looks_like_usage_error(exc):
                raise
            _show_help_if_requested(self, ctx, args, exc)
            raise

    def format_help(self, ctx: Any, formatter: Any) -> None:
        """
        Render localized help through the Meeting-ASR renderer.

        Args:
            ctx: Command context handed in by Typer (duck-typed).
            formatter: Help formatter kept for API compatibility (unused).

        Returns:
            None.
        """
        from app.presentation.cli.help import render_help

        render_help(self, _command_path(ctx))


class MeetingAsrTyper(typer.Typer):
    """Typer app with Meeting-ASR default CLI conventions."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        """
        Create a Typer app using localized help and ``-h`` by default.

        Args:
            *args: Positional Typer arguments.
            **kwargs: Keyword Typer arguments.

        Returns:
            None.
        """
        kwargs.setdefault("cls", LocalizedTyperGroup)
        kwargs.setdefault("context_settings", HELP_CONTEXT)
        super().__init__(*args, **kwargs)

    def command(self, *args: Any, **kwargs: Any) -> Any:
        """
        Register a command with localized help by default.

        Args:
            *args: Positional command registration arguments.
            **kwargs: Keyword command registration arguments.

        Returns:
            Typer command decorator.
        """
        kwargs.setdefault("cls", LocalizedTyperCommand)
        kwargs.setdefault("context_settings", HELP_CONTEXT)
        return super().command(*args, **kwargs)

    def add_typer(self, *args: Any, **kwargs: Any) -> None:
        """
        Register a subgroup with shared help options by default.

        Args:
            *args: Positional subgroup registration arguments.
            **kwargs: Keyword subgroup registration arguments.

        Returns:
            None.
        """
        kwargs.setdefault("context_settings", HELP_CONTEXT)
        super().add_typer(*args, **kwargs)


def _command_path(ctx: Any) -> tuple[str, ...]:
    """
    Return the command path after the executable name.

    Args:
        ctx: Command context (duck-typed).

    Returns:
        Command path tuple such as ``("project", "list")``.
    """
    parts = tuple(part for part in ctx.command_path.split() if part)
    return parts[1:] if parts else ()


def _has_help_flag(args: list[str]) -> bool:
    """
    Return whether help was requested anywhere in the remaining args.

    Args:
        args: Remaining command-line arguments.

    Returns:
        True when ``-h`` or ``--help`` is present.
    """
    return any(arg in HELP_CONTEXT["help_option_names"] for arg in args)


def _looks_like_usage_error(exc: BaseException) -> bool:
    """
    Return whether an exception is a Click/Typer usage error.

    Typer 0.26 no longer exposes the vendored ``ClickException``/``UsageError``
    classes publicly, so we recognize them by shape: a usage error carries the
    command context, renders its own message, and declares a process exit code.

    Args:
        exc: Exception raised while parsing or invoking a command.

    Returns:
        True when the exception looks like a parse/usage error to localize.
    """
    return (
        hasattr(exc, "ctx")
        and hasattr(exc, "format_message")
        and hasattr(exc, "exit_code")
    )


def _show_help_if_requested(
    command: Any,
    ctx: Any,
    args: list[str],
    exc: Any,
) -> None:
    """
    Prefer help over parse errors when ``-h`` or ``--help`` is present.

    Args:
        command: Command whose parsing failed.
        ctx: Command context (duck-typed).
        args: Remaining command-line arguments.
        exc: Original parse exception (duck-typed).

    Returns:
        None.
    """
    if not _has_help_flag(args) or ctx.resilient_parsing:
        return
    _configure_presentation_from_context(ctx)
    command.format_help(ctx, ctx.make_formatter())
    raise typer.Exit(0) from exc


def _run_standalone_with_localized_errors(
    command: Any,
    *,
    args: Sequence[str] | None,
    prog_name: str | None,
    complete_var: str | None,
    windows_expand_args: bool,
    extra: dict[str, Any],
) -> None:
    """
    Run a Typer command in standalone mode with localized parse errors.

    Runs the command with ``standalone_mode=False`` so Typer re-raises usage
    errors instead of printing its own English panel, then renders the
    Meeting-ASR localized error. ``typer.Exit`` is converted by Typer into an
    integer return value in this mode, so the success path exits on it.

    Args:
        command: Command generated by Typer.
        args: Command-line arguments.
        prog_name: Program name for usage output.
        complete_var: Completion environment variable.
        windows_expand_args: Whether arguments should be expanded on Windows.
        extra: Extra context values forwarded to Typer.

    Returns:
        None.
    """
    try:
        result = command.main(
            args=args,
            prog_name=prog_name,
            complete_var=complete_var,
            standalone_mode=False,
            windows_expand_args=windows_expand_args,
            **extra,
        )
    except typer.Abort:
        typer.echo("Aborted!", err=True)
        sys.exit(1)
    except Exception as exc:  # noqa: BLE001
        if not _looks_like_usage_error(exc):
            raise
        _configure_presentation_from_error(exc)
        from app.presentation.cli.errors import show_cli_usage_error

        show_cli_usage_error(exc)
        sys.exit(getattr(exc, "exit_code", 2))
    sys.exit(result if type(result) is int else 0)


def _configure_presentation_from_context(ctx: Any) -> None:
    """
    Apply parsed root presentation options before rendering rescued help.

    Args:
        ctx: Command context where parsing failed (duck-typed).

    Returns:
        None.
    """
    from app.presentation.cli.i18n import configure_cli_language
    from app.presentation.cli.output import configure_cli_output

    params = _merged_context_params(ctx)
    try:
        configure_cli_language(params.get("lang"))
    except ValueError:
        configure_cli_language(None)
    configure_cli_output(
        no_color=bool(params.get("no_color")), verbose=bool(params.get("verbose"))
    )


def _configure_presentation_from_error(exc: Any) -> None:
    """
    Apply root presentation options before rendering a parse error.

    Args:
        exc: Usage exception (duck-typed).

    Returns:
        None.
    """
    ctx = getattr(exc, "ctx", None)
    if ctx is None:
        return
    _configure_presentation_from_context(ctx)


def _merged_context_params(ctx: Any) -> dict[str, Any]:
    """
    Merge parameters from the current context and its parents.

    Args:
        ctx: Command context where parsing failed (duck-typed).

    Returns:
        Parameter mapping with child values overriding parent values.
    """
    chain: list[Any] = []
    current: Any = ctx
    while current is not None:
        chain.append(current)
        current = current.parent
    params: dict[str, Any] = {}
    for item in reversed(chain):
        params.update(item.params)
    return params
