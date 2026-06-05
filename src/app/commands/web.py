"""`meeting-asr web` -- launch the local web UI server.

Thin Typer adapter registered as a single root command (like ``doctor``/``paths``), so
it picks up the standard bilingual help handling. The web package and its
FastAPI/uvicorn dependencies are imported lazily so the rest of the CLI keeps working
when the optional ``web`` extra is absent; a missing import surfaces an actionable
install hint instead of a stack trace.
"""

from __future__ import annotations

from pathlib import Path

import typer

_INSTALL_HINT = (
    "Web UI dependencies are not installed.\n"
    "Install them with:\n"
    "  uv tool install --force 'meeting-asr[web]'\n"
    "or, from a checkout:\n"
    "  uv sync --extra web"
)


def _port_available(host: str, port: int) -> bool:
    """Return whether ``(host, port)`` can be bound right now.

    A pre-flight check gives a clean, actionable error instead of uvicorn's confusing
    "started ... address already in use ... shutdown" sequence when the port is taken.
    """
    import socket

    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as probe:
        probe.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        try:
            probe.bind((host, port))
        except OSError:
            return False
    return True


def command(
    host: str = typer.Option(
        "127.0.0.1", "--host", help="Bind host. Non-loopback binds require a token."
    ),
    port: int = typer.Option(8765, "--port", help="Bind port."),
    projects_dir: Path | None = typer.Option(
        None,
        "--projects-dir",
        help="Projects parent directory (default: XDG data dir).",
    ),
    store_dir: Path | None = typer.Option(
        None,
        "--store-dir",
        help="Voiceprint/lexicon store directory (default: XDG data dir). "
        "Point this at a copy when experimenting to protect the real library.",
    ),
    token: str | None = typer.Option(
        None,
        "--token",
        help="Bearer token for non-loopback binds (auto-generated if omitted).",
    ),
    open_browser: bool = typer.Option(
        True, "--open/--no-open", help="Open the default browser on startup."
    ),
) -> None:
    """Start the web UI server and listen for connections."""
    try:
        from app.web.server import resolve_token, run_server
        from app.web.settings import WebSettings
    except ImportError as exc:  # web extra missing
        typer.echo(_INSTALL_HINT, err=True)
        raise typer.Exit(code=1) from exc

    if not _port_available(host, port):
        typer.echo(
            f"Port {port} on {host} is already in use by another process.\n"
            f"Pick a free port, e.g.:  meeting-asr web --port {port + 1}",
            err=True,
        )
        raise typer.Exit(code=1)

    resolved_token = resolve_token(host, token)
    settings = WebSettings(
        host=host,
        port=port,
        projects_dir=projects_dir.expanduser().resolve() if projects_dir else None,
        store_dir=store_dir.expanduser().resolve() if store_dir else None,
        open_browser=open_browser,
        token=resolved_token,
    )
    typer.echo(f"meeting-asr web serving at http://{host}:{port}/")
    if resolved_token is not None:
        typer.echo(f"Bearer token (required for non-loopback access): {resolved_token}")
    run_server(settings)
