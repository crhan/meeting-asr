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

import json
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
_METADATA_FILE = "transaction.json"
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

    def __init__(self, *, load_persisted: bool = False) -> None:
        # value: (created_at, transaction, project_dir) -- project_dir lets accept/rollback
        # take that project's lock so a restore can't race a concurrent project-local write.
        self._txns: dict[str, tuple[float, VoiceprintReviewTransaction, Path]] = {}
        self._registry_lock = threading.Lock()
        self._store_write_lock = threading.Lock()
        if load_persisted:
            self._load_persisted()

    def has_pending(self) -> bool:
        """Return whether any capture transaction is awaiting accept/rollback.

        Keeps persisted transactions pending until the user accepts or rolls them back.
        That is deliberate: a restart must not silently commit or discard the rollback
        snapshot. ``sweep_stale`` only drops entries whose backup directory already vanished.
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

        ``sweep_stale`` takes only ``_registry_lock`` (never ``_store_write_lock``), so
        calling it while the caller holds the store-write lock is safe.
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
            created_at = time.time()
            try:
                _persist_transaction_metadata(
                    txn_id=txn_id,
                    created_at=created_at,
                    transaction=summary.transaction,
                    project_dir=project_dir.resolve(),
                )
            except Exception:
                summary.transaction.rollback()
                raise
            with self._registry_lock:
                self._txns[txn_id] = (
                    created_at,
                    summary.transaction,
                    project_dir.resolve(),
                )
        return txn_id, summary

    def accept(self, txn_id: str) -> None:
        """Accept a pending transaction (drop the rollback snapshot).

        Accept only removes the backup directory; it never touches the live store, so it
        needs no store-write lock. The registry entry is removed only after accept succeeds,
        otherwise the user has no handle to retry a failed cleanup.
        """
        self._get(txn_id).accept()
        self._discard(txn_id)

    def rollback(self, txn_id: str) -> None:
        """Roll back a pending transaction (restore the snapshot).

        The whole pop+restore runs under ``_store_write_lock`` -- the same critical section
        every global-store write holds. Otherwise a CRUD write or speaker reassignment could
        slip in after the transaction is popped (so ``run_store_write`` no longer sees it as
        pending) and race the snapshot restore, losing or corrupting that write -- exactly
        the data-loss class the guard exists to prevent.
        """
        with self._store_write_lock:
            self._get(txn_id).rollback()
            self._discard(txn_id)

    def _get(self, txn_id: str) -> VoiceprintReviewTransaction:
        with self._registry_lock:
            entry = self._txns.get(txn_id)
        if entry is None:
            raise FileNotFoundError(f"Unknown or expired capture transaction: {txn_id}")
        return entry[1]

    def _discard(self, txn_id: str) -> None:
        with self._registry_lock:
            self._txns.pop(txn_id, None)

    def sweep_stale(self, *, max_age_seconds: float = _ORPHAN_MAX_AGE_SECONDS) -> None:
        """Drop registry entries whose rollback snapshot has already disappeared.

        Age alone is not a reason to accept: if a transaction has persisted metadata, the
        recovery banner can still surface it after a restart and the user should decide.
        """
        del max_age_seconds
        with self._registry_lock:
            vanished = [
                txn_id
                for txn_id, (_created, txn, _project) in self._txns.items()
                if isinstance(getattr(txn, "backup_dir", None), Path)
                and not txn.backup_dir.exists()
            ]
            for txn_id in vanished:
                self._txns.pop(txn_id, None)

    def _load_persisted(self) -> None:
        """Restore pending transaction handles from backup metadata on process start."""
        for backup_dir in _backup_root().glob(f"{_BACKUP_PREFIX}*"):
            metadata_path = backup_dir / _METADATA_FILE
            if not metadata_path.is_file():
                continue
            try:
                txn_id, created_at, transaction, project_dir = (
                    _load_transaction_metadata(metadata_path)
                )
            except Exception:
                continue
            self._txns[txn_id] = (created_at, transaction, project_dir)


def cleanup_orphan_backups() -> None:
    """Remove old backup dirs that never reached transaction registration."""
    tmp_root = _backup_root()
    if not tmp_root.is_dir():
        return
    cutoff = time.time() - _ORPHAN_MAX_AGE_SECONDS
    for child in tmp_root.glob(f"{_BACKUP_PREFIX}*"):
        try:
            # A metadata-bearing directory is a recoverable pending transaction. Keep it so
            # the registry can restore the accept/rollback handle after restart.
            if (
                child.is_dir()
                and not (child / _METADATA_FILE).exists()
                and child.stat().st_mtime < cutoff
            ):
                shutil.rmtree(child, ignore_errors=True)
        except OSError:
            continue


def _backup_root() -> Path:
    """Return the temp root used for voiceprint rollback snapshots."""
    return Path(tempfile.gettempdir())


def _persist_transaction_metadata(
    *,
    txn_id: str,
    created_at: float,
    transaction: VoiceprintReviewTransaction,
    project_dir: Path,
) -> None:
    """Persist enough transaction state to recover accept/rollback after restart."""
    if not isinstance(transaction, VoiceprintReviewTransaction):
        return
    metadata = {
        "version": 1,
        "transaction_id": txn_id,
        "created_at": created_at,
        "project_dir": str(project_dir),
        "backup_dir": str(transaction.backup_dir),
        "db_path": str(transaction.db_path),
        "db_backup_path": str(transaction.db_backup_path),
        "db_existed": transaction.db_existed,
        "project_manifest_path": str(transaction.project_manifest_path),
        "project_manifest_backup_path": str(transaction.project_manifest_backup_path),
        "project_manifest_existed": transaction.project_manifest_existed,
        "match_path": str(transaction.match_path),
        "match_backup_path": str(transaction.match_backup_path),
        "match_existed": transaction.match_existed,
        "clip_backups": [
            {
                "clip_path": str(clip_path),
                "backup_path": str(backup_path),
                "existed": existed,
            }
            for clip_path, backup_path, existed in transaction.clip_backups
        ],
    }
    metadata_path = transaction.backup_dir / _METADATA_FILE
    metadata_path.write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )


def _load_transaction_metadata(
    metadata_path: Path,
) -> tuple[str, float, VoiceprintReviewTransaction, Path]:
    """Load one persisted capture transaction metadata file."""
    payload = json.loads(metadata_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or payload.get("version") != 1:
        raise ValueError(f"Unsupported capture transaction metadata: {metadata_path}")
    backup_dir = Path(str(payload["backup_dir"]))
    transaction = VoiceprintReviewTransaction(
        backup_dir=backup_dir,
        db_path=Path(str(payload["db_path"])),
        db_backup_path=Path(str(payload["db_backup_path"])),
        db_existed=bool(payload["db_existed"]),
        project_manifest_path=Path(str(payload["project_manifest_path"])),
        project_manifest_backup_path=Path(str(payload["project_manifest_backup_path"])),
        project_manifest_existed=bool(payload["project_manifest_existed"]),
        match_path=Path(str(payload["match_path"])),
        match_backup_path=Path(str(payload["match_backup_path"])),
        match_existed=bool(payload["match_existed"]),
        clip_backups=tuple(
            (
                Path(str(item["clip_path"])),
                Path(str(item["backup_path"])),
                bool(item["existed"]),
            )
            for item in payload.get("clip_backups", [])
            if isinstance(item, dict)
        ),
    )
    txn_id = str(payload["transaction_id"])
    created_at = float(payload.get("created_at", metadata_path.stat().st_mtime))
    project_dir = Path(str(payload["project_dir"])).resolve()
    return txn_id, created_at, transaction, project_dir


# Process-local singleton: the web server runs a single uvicorn worker.
REGISTRY = CaptureTransactionRegistry(load_persisted=True)
