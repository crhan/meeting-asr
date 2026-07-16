"""Unit tests for the hatch build hook's SPA build decision (``hatch_build.build_spa``).

The hook ships the React SPA into the wheel. The risk these tests guard is a *stale* bundle:
an explicit web build (``MEETING_ASR_BUILD_WEB=1``: CI / release / web install) must rebuild
fresh and never reuse a possibly-out-of-date ``src/app/web/static`` from an earlier build.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path
from typing import Any

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


def _make_hook(root: Path, target_name: str = "wheel") -> Any:
    """Build a CustomBuildHook without touching the base-class constructor.

    With hatchling installed, ``BuildHookInterface.__init__`` takes required build-state
    arguments and ``root``/``target_name`` are read-only properties; without it the module
    falls back to ``object``. Shadowing both names with subclass attributes and skipping
    ``__init__`` via ``__new__`` keeps these tests independent of which base is active.
    """
    cls = type(
        "TestableBuildHook",
        (hatch_build.CustomBuildHook,),
        {"root": str(root), "target_name": target_name},
    )
    return cls.__new__(cls)


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


def test_build_hook_refuses_web_wheel_when_spa_missing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """If BUILD_WEB=1 did not leave index.html in static, the wheel must fail."""
    _web_sources(tmp_path)
    hook = _make_hook(tmp_path)
    monkeypatch.setenv("MEETING_ASR_BUILD_WEB", "1")
    monkeypatch.setattr(hatch_build, "build_spa", lambda *_a, **_k: None)

    with pytest.raises(RuntimeError, match="static/index.html"):
        hook.initialize("0.0.0", {})


def test_build_hook_includes_existing_spa_as_natural_path_artifact(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Built static assets are included through their natural ``src/app`` package path.

    Artifacts override the VCS ignore without adding a second archive entry when ``uv build``
    builds a wheel from its sdist. The ``src/app`` package mapping places this pattern at the
    served ``app/web/static`` path in the wheel.
    """
    static = tmp_path / "src" / "app" / "web" / "static"
    static.mkdir(parents=True)
    (static / "index.html").write_text("BUILT")
    monkeypatch.delenv("MEETING_ASR_BUILD_WEB", raising=False)
    hook = _make_hook(tmp_path)

    build_data: dict = {}
    hook.initialize("0.0.0", build_data)

    assert build_data["artifacts"] == ["/src/app/web/static/**"]
    assert "force_include" not in build_data


def test_build_hook_skips_non_wheel_targets(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The sdist carries web/ sources, never the built SPA; the hook must not touch it."""
    static = tmp_path / "src" / "app" / "web" / "static"
    static.mkdir(parents=True)
    (static / "index.html").write_text("BUILT")
    monkeypatch.setenv("MEETING_ASR_BUILD_WEB", "1")
    hook = _make_hook(tmp_path, target_name="sdist")

    build_data: dict = {}
    hook.initialize("0.0.0", build_data)

    assert build_data == {}
