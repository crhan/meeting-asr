"""Shared Typer settings and help classes for CLI compatibility."""

from __future__ import annotations

from typing import Any

import click
import typer
from typer.core import TyperCommand
from typer.core import TyperGroup

HELP_CONTEXT = {"help_option_names": ["-h", "--help"]}


class LocalizedTyperGroup(TyperGroup):
    """Typer group that renders Meeting-ASR localized Rich help."""

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
        return super().parse_args(ctx, args)

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
        return super().parse_args(ctx, args)

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
