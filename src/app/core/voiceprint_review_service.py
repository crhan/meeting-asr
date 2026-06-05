"""Presentation-neutral voiceprint capture workflow + cross-request transaction registry.

The capture flow has three steps the TUI runs inline behind one modal:

1. **plan** (read-only): ``plan_voiceprint_capture`` lists candidate clips per named
   speaker with selection/audio scores -- no store writes.
2. **run** (destructive): ``run_voiceprint_review_workflow`` persists the selected clips,
   embeds them, evaluates score impact, and returns a filesystem-snapshot transaction.
3. **accept / rollback**: keep or undo the run.

Over HTTP these become separate requests, so the transaction (which owns on-disk backup
snapshots) must outlive the run response. This module keeps a process-local registry keyed
by a transaction id; ``accept``/``rollback`` look it up. A startup sweep removes orphaned
backup directories left by a crash mid-run. Store-mutating runs are serialised by a
threading lock so two captures cannot corrupt the shared global store.
"""

from __future__ import annotations

import shutil
import tempfile
import threading
import time
import uuid
from pathlib import Path

from app.presentation.tui.voiceprint_review_workflow import (
    VoiceprintReviewTransaction,
    VoiceprintReviewWorkflowSummary,
    run_voiceprint_review_workflow,
)
from app.voiceprints import VoiceprintCaptureSummary, plan_voiceprint_capture

DEFAULT_SAMPLE_COUNT = 3
DEFAULT_MAX_SECONDS = 12.0
DEFAULT_PADDING_SECONDS = 0.5

_BACKUP_PREFIX = "meeting-asr-voiceprint-review-"
_ORPHAN_MAX_AGE_SECONDS = 6 * 3600


class CaptureConflictError(RuntimeError):
    """A store mutation was attempted while a capture transaction is pending.

    A pending capture holds a pre-run snapshot of the GLOBAL voiceprint store; rolling it
    back restores that snapshot. If unrelated store edits (rename/delete/merge a person,
    sample status, another capture) landed in between, the rollback would silently discard
    them. The web layer therefore makes a pending capture exclusive over the store and
    maps this error to HTTP 409. It is ``RuntimeError`` (not ``ValueError``) so the generic
    400 handler does not swallow it.
    """


def plan_capture(
    project_dir: Path,
    *,
    store_dir: Path | None = None,
    sample_count: int = DEFAULT_SAMPLE_COUNT,
    max_seconds: float = DEFAULT_MAX_SECONDS,
    padding_seconds: float = DEFAULT_PADDING_SECONDS,
) -> VoiceprintCaptureSummary:
    """Plan voiceprint capture for one project (read-only dry run)."""
    return plan_voiceprint_capture(
        project_dir,
        sample_count=sample_count,
        max_seconds=max_seconds,
        padding_seconds=padding_seconds,
        store_dir=store_dir,
    )


class CaptureTransactionRegistry:
    """Holds pending capture transactions between the run and the accept/rollback calls."""

    def __init__(self) -> None:
        self._txns: dict[str, tuple[float, VoiceprintReviewTransaction]] = {}
        self._registry_lock = threading.Lock()
        self._store_write_lock = threading.Lock()

    def has_pending(self) -> bool:
        """Return whether any capture transaction is awaiting accept/rollback."""
        with self._registry_lock:
            return bool(self._txns)

    def run(
        self,
        *,
        project_dir: Path,
        planned: VoiceprintCaptureSummary,
        selected_clip_rel_paths: frozenset[str],
        store_dir: Path | None,
    ) -> tuple[str, VoiceprintReviewWorkflowSummary]:
        """Run the capture+embed+evaluate workflow and register its transaction.

        Only one capture may be pending at a time: a second run while an earlier one still
        awaits accept/rollback would snapshot a store that already includes the first run's
        writes, so rolling back either one could corrupt the other. The check and the
        registration happen under ``_store_write_lock`` so they cannot interleave.
        """
        with self._store_write_lock:
            with self._registry_lock:
                if self._txns:
                    raise CaptureConflictError(
                        "A previous voiceprint capture is still awaiting accept/rollback; "
                        "resolve it before starting another."
                    )
            summary = run_voiceprint_review_workflow(
                project_dir=project_dir,
                planned=planned,
                selected_clip_rel_paths=selected_clip_rel_paths,
                store_dir=store_dir,
            )
            txn_id = uuid.uuid4().hex
            with self._registry_lock:
                self._txns[txn_id] = (time.time(), summary.transaction)
        return txn_id, summary

    def accept(self, txn_id: str) -> None:
        """Accept a pending transaction (drop the rollback snapshot)."""
        self._pop(txn_id).accept()

    def rollback(self, txn_id: str) -> None:
        """Roll back a pending transaction (restore the snapshot)."""
        self._pop(txn_id).rollback()

    def _pop(self, txn_id: str) -> VoiceprintReviewTransaction:
        with self._registry_lock:
            entry = self._txns.pop(txn_id, None)
        if entry is None:
            raise FileNotFoundError(f"Unknown or expired capture transaction: {txn_id}")
        return entry[1]

    def sweep_stale(self, *, max_age_seconds: float = _ORPHAN_MAX_AGE_SECONDS) -> None:
        """Accept (commit) transactions older than the cutoff to reclaim disk.

        A stale transaction means the user never decided; its changes are already
        committed to the store, so we keep them and just drop the rollback snapshot.
        """
        now = time.time()
        with self._registry_lock:
            stale = [
                (txn_id, txn)
                for txn_id, (created, txn) in self._txns.items()
                if now - created > max_age_seconds
            ]
            for txn_id, _ in stale:
                self._txns.pop(txn_id, None)
        for _, txn in stale:
            txn.accept()


def cleanup_orphan_backups() -> None:
    """Remove leftover capture backup directories from crashed runs (startup sweep)."""
    tmp_root = Path(tempfile.gettempdir())
    if not tmp_root.is_dir():
        return
    cutoff = time.time() - _ORPHAN_MAX_AGE_SECONDS
    for child in tmp_root.glob(f"{_BACKUP_PREFIX}*"):
        try:
            if child.is_dir() and child.stat().st_mtime < cutoff:
                shutil.rmtree(child, ignore_errors=True)
        except OSError:
            continue


# Process-local singleton: the web server runs a single uvicorn worker.
REGISTRY = CaptureTransactionRegistry()
