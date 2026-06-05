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

from app.voiceprint_store import VOICEPRINT_STORE_DIR

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

    @property
    def voiceprint_store_dir(self) -> Path | None:
        """Voiceprint store dir derived from the data-root ``store_dir``.

        ``store_dir`` is the data root (the XDG data dir, or a copy of it -- as the
        ``--store-dir`` help says), the same root the lexicon path rebases onto
        (``<store_dir>/lexicon/lexicon.sqlite``). The voiceprint store lives under
        ``<store_dir>/voiceprints``; pass *this* to the voiceprint APIs, never the bare
        ``store_dir`` -- ``get_voiceprint_db_path`` would otherwise resolve the flat
        ``<store_dir>/voiceprints.sqlite`` and miss the real database, so an isolated
        ``--store-dir`` copy would read an empty library and write captures to the wrong
        place. ``None`` keeps the XDG default.
        """
        if self.store_dir is None:
            return None
        return self.store_dir / VOICEPRINT_STORE_DIR
