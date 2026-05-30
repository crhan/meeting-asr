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

    Args:
        monkeypatch: Pytest environment patcher (auto-reverted after the test).

    Returns:
        None.
    """
    for env_var in ("MEETING_ASR_LANG", "LC_ALL", "LC_MESSAGES", "LANG"):
        monkeypatch.delenv(env_var, raising=False)
    from app.presentation.cli.i18n import configure_cli_language
    from app.presentation.cli.output import configure_cli_output

    configure_cli_language("en")
    configure_cli_output(no_color=False, verbose=False)
