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
        # value: (created_at, transaction, project_dir) -- project_dir lets accept/rollback
        # take that project's lock so a restore can't race a concurrent project-local write.
        self._txns: dict[
            str, tuple[float, VoiceprintReviewTransaction, Path]
        ] = {}
        self._registry_lock = threading.Lock()
        self._store_write_lock = threading.Lock()

    def has_pending(self) -> bool:
        """Return whether any capture transaction is awaiting accept/rollback.

        Reaps abandoned transactions first (see ``sweep_stale``): nothing else calls the
        sweep, so without this a capture whose browser cleanup never arrived (tab crash,
        dropped sendBeacon) would block every later store write with 409 until the server
        restarts. Sweeping on the authoritative pending gate bounds that to the sweep age.
        """
        self.sweep_stale()
        with self._registry_lock:
            return bool(self._txns)

    def pending_transaction(self) -> tuple[str, Path] | None:
        """Return the (id, project_dir) of the pending capture, or None.

        Only one capture may be pending at a time, so this is unambiguous. The web exposes it
        so a recovery banner can offer accept/rollback for a transaction whose originating
        page is gone (e.g. the user left while the capture job was still running, so no page
        ever learned the transaction id) -- otherwise it would wedge the store until swept.
        """
        self.sweep_stale()
        with self._registry_lock:
            for txn_id, (_created, _txn, project) in self._txns.items():
                return txn_id, project
        return None

    def project_dir_for(self, txn_id: str) -> Path | None:
        """Return the project dir a pending transaction belongs to, or None.

        Callers take that project's lock around accept/rollback so the snapshot restore
        serialises with project-local writes (speaker save, correction accept).
        """
        with self._registry_lock:
            entry = self._txns.get(txn_id)
        return entry[2] if entry else None

    def _raise_if_pending_locked(self) -> None:
        """Raise if a capture is pending. Caller must hold ``_store_write_lock``.

        Reaps abandoned transactions first so a store write is not blocked indefinitely by a
        capture whose browser cleanup was lost. ``sweep_stale`` takes only ``_registry_lock``
        (never ``_store_write_lock``), so calling it while the caller holds the store-write
        lock is safe.
        """
        self.sweep_stale()
        with self._registry_lock:
            if self._txns:
                raise CaptureConflictError(
                    "A voiceprint capture is awaiting accept/rollback; resolve it before "
                    "editing the store."
                )

    def run_store_write(self, fn):
        """Run a global-store mutation in the same critical section as capture runs.

        Every write to the global voiceprint store (people/sample CRUD, and the sample
        invalidation a speaker reassignment performs) must go through here. It holds the
        same ``_store_write_lock`` a capture run holds across its snapshot+write+register
        window, and re-checks for a pending capture *under that lock*. That ordering is
        what closes the race a bare ``has_pending()`` pre-check cannot: a mutation can only
        run either fully before a capture's snapshot (so the snapshot includes it) or after
        the capture has registered its transaction (so the pending check now refuses it) --
        never interleaved with the snapshot where a later rollback would silently drop it.
        """
        with self._store_write_lock:
            self._raise_if_pending_locked()
            return fn()

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
            self._raise_if_pending_locked()
            summary = run_voiceprint_review_workflow(
                project_dir=project_dir,
                planned=planned,
                selected_clip_rel_paths=selected_clip_rel_paths,
                store_dir=store_dir,
            )
            txn_id = uuid.uuid4().hex
            with self._registry_lock:
                self._txns[txn_id] = (
                    time.time(),
                    summary.transaction,
                    project_dir.resolve(),
                )
        return txn_id, summary

    def accept(self, txn_id: str) -> None:
        """Accept a pending transaction (drop the rollback snapshot).

        Accept only removes the backup directory; it never touches the live store, so it
        needs no store-write lock. Popping first makes the capture no longer pending.
        """
        self._pop(txn_id).accept()

    def rollback(self, txn_id: str) -> None:
        """Roll back a pending transaction (restore the snapshot).

        The whole pop+restore runs under ``_store_write_lock`` -- the same critical section
        every global-store write holds. Otherwise a CRUD write or speaker reassignment could
        slip in after the transaction is popped (so ``run_store_write`` no longer sees it as
        pending) and race the snapshot restore, losing or corrupting that write -- exactly
        the data-loss class the guard exists to prevent.
        """
        with self._store_write_lock:
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
                for txn_id, (created, txn, _project) in self._txns.items()
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
