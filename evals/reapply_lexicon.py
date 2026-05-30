"""Re-apply the current local lexicon correction rules to existing projects.

Lexicon corrections (e.g. P叉一 -> PXE) only run during a project's original
transcription post-process, so terms added to the lexicon AFTER a project was
transcribed never reach its saved transcript. This replays JUST the lexicon
correction step — the exact call ``project run`` uses, no ASR and no polish model
— so newly added vocabulary backfills into old projects. The raw
``asr/sentences.json`` is the source and is left untouched; only the derived
corrected outputs are rewritten, so a bad run is recoverable.

Run one project first to inspect, then all:
    python -m evals.reapply_lexicon p-8a50d4d04d6c59a6   # one
    python -m evals.reapply_lexicon                        # all
"""

from __future__ import annotations

import sys
from pathlib import Path

from app.commands.project import _apply_run_lexicon_corrections

from evals._log import log

PROJ = Path.home() / ".local" / "share" / "meeting-asr" / "projects"


def main() -> None:
    """Re-apply lexicon corrections to one project (arg) or every project."""
    only = sys.argv[1] if len(sys.argv) > 1 else None
    dirs = [PROJ / only] if only else sorted(PROJ.glob("p-*"))
    log.info("start", projects=len(dirs), only=only)

    total_changed = 0
    touched = 0
    for d in dirs:
        if not (d / "project.json").exists():
            continue
        try:
            summary = _apply_run_lexicon_corrections(d, progress=None)
        except Exception as exc:  # noqa: BLE001 — one bad project must not abort the sweep
            log.warning("project_failed", proj=d.name, err=type(exc).__name__,
                        msg=str(exc)[:100])
            continue
        n = summary.change_count or 0
        if n:
            total_changed += n
            touched += 1
            log.info("corrected", proj=d.name, changes=n)
    log.info("done", projects=len(dirs), touched=touched, total_changes=total_changed)


if __name__ == "__main__":
    main()
