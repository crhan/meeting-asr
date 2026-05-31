"""Shared pytest fixtures for deterministic CLI/TUI tests."""

from __future__ import annotations

import pytest


@pytest.fixture(autouse=True)
def _hermetic_cli_state(monkeypatch: pytest.MonkeyPatch) -> None:
    """
    Pin a neutral CLI presentation baseline before every test.

    The CLI display language and output flags live in module-level globals
    (``app.presentation.cli.i18n`` / ``app.presentation.cli.output``), and the
    default language is resolved from the ambient locale. Both leak across
    tests: a CLI invocation with ``--lang zh`` (or a TUI test that switches
    language) mutates the process-wide language and the next test inherits it,
    while the machine locale decides the default. That makes assertions depend
    on test order and on the developer's locale.

    This fixture neutralizes the locale environment and resets the globals to a
    deterministic English baseline before each test. Tests that exercise Chinese
    still force it explicitly (via ``--lang``, ``MEETING_ASR_LANG``/locale env,
    or ``configure_cli_language``), so they keep working; everything else
    becomes order- and locale-independent.

    It also pins deterministic, CI-equivalent Rich rendering so output-string
    assertions stop depending on the developer's terminal. Two ambient factors
    leak in locally:

    1. Width. Rich reads ``COLUMNS`` for console width. A developer shell may
       export ``COLUMNS=0`` (non-interactive parents do this), which renders
       every panel/table at width 0 -> empty output; any narrow non-zero width
       also wraps the wide voiceprint speaker table so a name and its id no
       longer share one parseable row. We pin ``COLUMNS=200``.
    2. Color. ``CliRunner`` captures output into an in-memory buffer, but Rich
       still detects a real terminal here (and an inherited ``FORCE_COLOR``
       forces it), so it emits ANSI. ANSI codes split substrings such as
       ``6 OK`` and ``│`` table cells across style spans, so
       ``"6 OK" in result.output`` and speaker-table parsing fail even though
       the visible text is correct. CI pipes stdout (no TTY, no color) and
       passes. We force plain output via ``NO_COLOR=1`` (honored by
       ``should_disable_color`` -> ``cli_console``) and drop ``FORCE_COLOR``.

    Both are set via env, so they cost nothing and touch no production code;
    tests needing specific width/color can still override per ``invoke``.

    Args:
        monkeypatch: Pytest environment patcher (auto-reverted after the test).

    Returns:
        None.
    """
    for env_var in ("MEETING_ASR_LANG", "LC_ALL", "LC_MESSAGES", "LANG"):
        monkeypatch.delenv(env_var, raising=False)
    monkeypatch.delenv("LINES", raising=False)
    monkeypatch.delenv("FORCE_COLOR", raising=False)
    monkeypatch.setenv("COLUMNS", "200")
    monkeypatch.setenv("NO_COLOR", "1")
    from app.presentation.cli.i18n import configure_cli_language
    from app.presentation.cli.output import configure_cli_output

    configure_cli_language("en")
    configure_cli_output(no_color=False, verbose=False)
