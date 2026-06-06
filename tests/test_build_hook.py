"""Unit tests for the hatch build hook's SPA build decision (``hatch_build.build_spa``).

The hook ships the React SPA into the wheel. The risk these tests guard is a *stale* bundle:
an explicit web build (``MEETING_ASR_BUILD_WEB=1``: CI / release / web install) must rebuild
fresh and never reuse a possibly-out-of-date ``src/app/web/static`` from an earlier build.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

# hatch_build.py lives at the repo root (it is the build backend hook, not part of the `app`
# package), so it is not importable by name from the test venv. Load it directly by path.
_HATCH_PATH = Path(__file__).resolve().parent.parent / "hatch_build.py"
_spec = importlib.util.spec_from_file_location("hatch_build", _HATCH_PATH)
assert _spec is not None and _spec.loader is not None
hatch_build = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(hatch_build)


def _web_sources(root: Path) -> None:
    (root / "web").mkdir()
    (root / "web" / "package.json").write_text("{}")


def test_build_spa_rebuilds_even_when_static_already_exists(tmp_path: Path) -> None:
    """BUILD_WEB=1 must rebuild even if a (possibly stale) static/index.html exists -- otherwise
    a release could ship an old UI after web/src changed."""
    _web_sources(tmp_path)
    static = tmp_path / "src" / "app" / "web" / "static"
    static.mkdir(parents=True)
    (static / "index.html").write_text("STALE")

    calls: list[list[str]] = []
    hatch_build.build_spa(
        tmp_path,
        build_web=True,
        run=lambda cmd, **_kw: calls.append(cmd),
        which=lambda _name: "/usr/bin/npm",
    )

    assert any(cmd[-1] == "ci" for cmd in calls), calls
    assert any(cmd[-1] == "build" for cmd in calls), calls


def test_build_spa_is_noop_without_the_flag(tmp_path: Path) -> None:
    """Without BUILD_WEB the hook never shells out to npm (the base-CLI build path)."""
    _web_sources(tmp_path)
    calls: list[object] = []
    hatch_build.build_spa(
        tmp_path,
        build_web=False,
        run=lambda *a, **_k: calls.append(a),
        which=lambda _name: "/usr/bin/npm",
    )
    assert calls == []


def test_build_spa_fails_loudly_when_npm_missing(tmp_path: Path) -> None:
    """An explicit web build with npm unavailable must raise, not silently ship stale/no UI."""
    _web_sources(tmp_path)
    with pytest.raises(RuntimeError, match="npm is unavailable"):
        hatch_build.build_spa(
            tmp_path,
            build_web=True,
            run=lambda *a, **_k: None,
            which=lambda _name: None,
        )


def test_build_spa_fails_loudly_when_web_sources_missing(tmp_path: Path) -> None:
    """BUILD_WEB=1 in a tree without web/ sources must raise rather than ship no UI."""
    with pytest.raises(RuntimeError, match="web/ frontend sources are missing"):
        hatch_build.build_spa(
            tmp_path,
            build_web=True,
            run=lambda *a, **_k: None,
            which=lambda _name: "/usr/bin/npm",
        )
