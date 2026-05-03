"""Shared Typer settings and help classes for CLI compatibility."""

from __future__ import annotations

import sys
from collections.abc import Sequence
from typing import Any

import click
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
            windows_expand_args: Whether Click should expand Windows args.
            **extra: Extra Click context values.

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

    def parse_args(self, ctx: click.Context, args: list[str]) -> list[str]:
        """
        Show help with exit code 0 when a group is called without a command.

        Args:
            ctx: Click context.
            args: Remaining command-line arguments.

        Returns:
            Remaining parsed arguments.
        """
        if not args and self.no_args_is_help and not ctx.resilient_parsing:
            self.format_help(ctx, ctx.make_formatter())
            raise click.exceptions.Exit(0)
        try:
            return super().parse_args(ctx, args)
        except click.ClickException as exc:
            _show_help_if_requested(self, ctx, args, exc)
            raise

    def format_help(self, ctx: click.Context, formatter: click.HelpFormatter) -> None:
        """
        Render localized help through the Meeting-ASR renderer.

        Args:
            ctx: Click context.
            formatter: Click formatter kept for Click API compatibility.

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
            windows_expand_args: Whether Click should expand Windows args.
            **extra: Extra Click context values.

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

    def parse_args(self, ctx: click.Context, args: list[str]) -> list[str]:
        """
        Show help with exit code 0 when a command is configured that way.

        Args:
            ctx: Click context.
            args: Remaining command-line arguments.

        Returns:
            Remaining parsed arguments.
        """
        if not args and self.no_args_is_help and not ctx.resilient_parsing:
            self.format_help(ctx, ctx.make_formatter())
            raise click.exceptions.Exit(0)
        try:
            return super().parse_args(ctx, args)
        except click.ClickException as exc:
            _show_help_if_requested(self, ctx, args, exc)
            raise

    def format_help(self, ctx: click.Context, formatter: click.HelpFormatter) -> None:
        """
        Render localized help through the Meeting-ASR renderer.

        Args:
            ctx: Click context.
            formatter: Click formatter kept for Click API compatibility.

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


def _command_path(ctx: click.Context) -> tuple[str, ...]:
    """
    Return the command path after the executable name.

    Args:
        ctx: Click context.

    Returns:
        Command path tuple such as ``("project", "list")``.
    """
    parts = tuple(part for part in ctx.command_path.split() if part)
    return parts[1:] if parts else ()


def _show_help_if_requested(
    command: click.Command,
    ctx: click.Context,
    args: list[str],
    exc: click.ClickException,
) -> None:
    """
    Prefer help over parse errors when ``-h`` or ``--help`` is present.

    Args:
        command: Command whose parsing failed.
        ctx: Click context.
        args: Remaining command-line arguments.
        exc: Original Click parse exception.

    Returns:
        None.
    """
    if not _has_help_flag(args) or ctx.resilient_parsing:
        return
    _configure_presentation_from_context(ctx)
    command.format_help(ctx, ctx.make_formatter())
    raise click.exceptions.Exit(0) from exc


def _run_standalone_with_localized_errors(
    command: click.Command,
    *,
    args: Sequence[str] | None,
    prog_name: str | None,
    complete_var: str | None,
    windows_expand_args: bool,
    extra: dict[str, Any],
) -> None:
    """
    Run a Typer command in standalone mode with localized parse errors.

    Args:
        command: Click command generated by Typer.
        args: Command-line arguments.
        prog_name: Program name for usage output.
        complete_var: Completion environment variable.
        windows_expand_args: Whether Click should expand Windows args.
        extra: Extra Click context values.

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
    except click.ClickException as exc:
        _configure_presentation_from_error(exc)
        from app.presentation.cli.errors import show_click_usage_error

        show_click_usage_error(exc)
        sys.exit(exc.exit_code)
    except click.Abort:
        click.echo("Aborted!", err=True)
        sys.exit(1)
    sys.exit(result if type(result) is int else 0)


def _has_help_flag(args: list[str]) -> bool:
    """
    Return whether help was requested anywhere in the remaining args.

    Args:
        args: Remaining command-line arguments.

    Returns:
        True when ``-h`` or ``--help`` is present.
    """
    return any(arg in HELP_CONTEXT["help_option_names"] for arg in args)


def _configure_presentation_from_context(ctx: click.Context) -> None:
    """
    Apply parsed root presentation options before rendering rescued help.

    Args:
        ctx: Click context where parsing failed.

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
    configure_cli_output(no_color=bool(params.get("no_color")), verbose=bool(params.get("verbose")))


def _configure_presentation_from_error(exc: click.ClickException) -> None:
    """
    Apply root presentation options before rendering a parse error.

    Args:
        exc: Click parse exception.

    Returns:
        None.
    """
    ctx = getattr(exc, "ctx", None)
    if ctx is None:
        return
    _configure_presentation_from_context(ctx)


def _merged_context_params(ctx: click.Context) -> dict[str, Any]:
    """
    Merge parameters from the current context and its parents.

    Args:
        ctx: Click context where parsing failed.

    Returns:
        Parameter mapping with child values overriding parent values.
    """
    chain: list[click.Context] = []
    current: click.Context | None = ctx
    while current is not None:
        chain.append(current)
        current = current.parent
    params: dict[str, Any] = {}
    for item in reversed(chain):
        params.update(item.params)
    return params
