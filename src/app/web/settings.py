"""Runtime configuration for the web server.

The web server is a single-user, single-worker local tool. Store directories are
resolved once at startup and injected into every request via ``Depends`` so a request
can never redirect writes to an unexpected voiceprint/lexicon store -- that is the
guard against the CLAUDE.md blood-pitfall where sentence reassignment deletes samples
from whichever global voiceprint store it is pointed at.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

_LOCAL_HOSTS = frozenset({"127.0.0.1", "localhost", "::1", "0:0:0:0:0:0:0:1"})


@dataclass(frozen=True, slots=True)
class WebSettings:
    """Resolved settings for one ``meeting-asr web`` invocation."""

    host: str
    port: int
    projects_dir: Path | None
    store_dir: Path | None
    open_browser: bool
    token: str | None

    @property
    def is_local(self) -> bool:
        """Return whether the bind host is loopback-only.

        Non-local binds expose the API to the network, so a bearer token is required
        there; loopback binds stay token-free for zero-friction local use.
        """
        return self.host in _LOCAL_HOSTS
