"""Hatch build hook: ship the React SPA in the wheel without making it mandatory.

The web UI assets live in ``src/app/web/static`` (a gitignored Vite build output). The
wheel ships them via a *conditional* force-include set here in ``build_data`` (not an
unconditional ``pyproject`` force-include), so a base CLI build that has no SPA does not
fail on a missing path. The CLI imports its web dependencies lazily, so a wheel without
the SPA is a valid artifact for someone installing just the CLI.

Build behaviour:

* If the assets are already built (the usual case: ``scripts/install-tool.sh`` /
  ``npm run build`` produced them, or a previous build step), they are force-included.
* Otherwise, the SPA is built from ``web/`` **only when the build explicitly asks for it**
  via ``MEETING_ASR_BUILD_WEB=1`` (set by CI / release / an explicit web install). That is
  the one path that requires ``npm``; a plain ``uv build`` / ``uv tool install .`` of the
  base CLI never drags in node/npm.
* A web-requested build with no built assets and no ``npm`` fails with a clear message, so
  a release wheel can never silently ship without its UI.
"""

from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path

try:
    from hatchling.builders.hooks.plugin.interface import BuildHookInterface
except ModuleNotFoundError:  # pragma: no cover - hatchling is only present at build time
    # hatchling is a build-backend dependency, absent from the runtime/test venv. Falling back
    # to ``object`` lets the module import so ``build_spa`` can be unit-tested; the real base
    # class is always present during an actual ``uv build`` (where this hook runs).
    BuildHookInterface = object  # type: ignore[assignment,misc]


def build_spa(
    root: Path,
    *,
    build_web: bool,
    run=subprocess.run,
    which=shutil.which,
) -> None:
    """(Re)build the React SPA from ``web/`` when an explicit web build is requested.

    ``MEETING_ASR_BUILD_WEB=1`` means "produce fresh web assets" (CI / release / explicit web
    install), so it rebuilds *unconditionally* -- it must never trust an existing (gitignored,
    possibly stale) ``src/app/web/static`` from an earlier build, or a release could ship an old
    UI after ``web/src`` changed. Without the flag this is a no-op and the caller ships whatever
    static already exists (the base-CLI build path). Missing ``web/`` sources or ``npm`` under
    the flag fail loudly rather than silently shipping (or omitting) the UI.
    """
    if not build_web:
        return
    web_dir = root / "web"
    if not (web_dir / "package.json").is_file():
        raise RuntimeError(
            "MEETING_ASR_BUILD_WEB=1 but web/ frontend sources are missing; "
            "cannot build the SPA."
        )
    npm = which("npm")
    if npm is None:
        raise RuntimeError(
            "MEETING_ASR_BUILD_WEB=1 but npm is unavailable; cannot build the SPA. "
            "Run `npm --prefix web ci && npm --prefix web run build` before building."
        )
    run([npm, "--prefix", str(web_dir), "ci"], check=True)
    run([npm, "--prefix", str(web_dir), "run", "build"], check=True)


class CustomBuildHook(BuildHookInterface):
    """Conditionally build + include the SPA into the wheel."""

    def initialize(self, version: str, build_data: dict) -> None:
        # Only the wheel ships the built SPA; the sdist carries the web/ sources instead and
        # a wheel built from that sdist re-runs this hook.
        if self.target_name != "wheel":
            return

        root = Path(self.root)
        static_dir = root / "src" / "app" / "web" / "static"

        build_spa(root, build_web=os.environ.get("MEETING_ASR_BUILD_WEB") == "1")

        # Ship the SPA only if it exists now. Doing this through build_data (instead of an
        # unconditional pyproject force-include) is what lets a base build with no static
        # succeed rather than erroring on a missing force-include path.
        if (static_dir / "index.html").is_file():
            build_data.setdefault("force_include", {})[str(static_dir)] = "app/web/static"
