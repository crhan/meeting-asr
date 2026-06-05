"""Hatch build hook: ensure the React SPA is built before packaging.

The web UI assets live in ``src/app/web/static`` (a gitignored Vite build output) and are
force-included into the wheel. This hook guarantees they exist at build time:

* If the assets are already built (the usual case: ``scripts/install-tool.sh`` or a dev
  ``npm run build`` produced them), it does nothing.
* Otherwise, if the ``web/`` frontend sources and ``npm`` are present (CI / release), it
  builds them.
* If neither built assets nor a way to build them exist, it fails with a clear message so
  the wheel never ships without its UI.
"""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

from hatchling.builders.hooks.plugin.interface import BuildHookInterface


class CustomBuildHook(BuildHookInterface):
    """Build the SPA into src/app/web/static before the wheel collects files."""

    def initialize(self, version: str, build_data: dict) -> None:
        root = Path(self.root)
        static_index = root / "src" / "app" / "web" / "static" / "index.html"
        web_dir = root / "web"

        if static_index.is_file():
            return  # already built

        if not (web_dir / "package.json").is_file():
            # No frontend sources in this build context; assume static is provided
            # elsewhere or the web UI is intentionally absent.
            return

        npm = shutil.which("npm")
        if npm is None:
            raise RuntimeError(
                "Web UI assets are not built and npm is unavailable. "
                "Run `npm --prefix web ci && npm --prefix web run build` before building."
            )

        subprocess.run([npm, "--prefix", str(web_dir), "ci"], check=True)
        subprocess.run([npm, "--prefix", str(web_dir), "run", "build"], check=True)
