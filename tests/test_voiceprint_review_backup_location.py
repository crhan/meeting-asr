"""Pending voiceprint-review snapshots must outlive a reboot.

A pending capture transaction owns the ONLY copy of the pre-run voiceprint store.
Until 0.21.0 those snapshots were written under ``tempfile.gettempdir()``, so a reboot
(or any /tmp reaper) silently destroyed a transaction the recovery banner would
otherwise still offer to roll back -- the store keeps the half-applied capture and the
undo is simply gone, with nothing logged. These tests pin the durable location and the
upgrade path off the old one.
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

from app.core import voiceprint_review_service as svc
from app.presentation.tui import voiceprint_review_workflow
from app.voiceprint_store import (
    VOICEPRINT_PENDING_REVIEW_DIR,
    VOICEPRINT_REVIEW_BACKUP_PREFIX,
    get_voiceprint_review_backup_root,
)


def test_backup_root_follows_the_store_it_snapshots(tmp_path: Path) -> None:
    """The snapshot root lives beside the store, so a custom store stays self-contained."""
    store_dir = tmp_path / "voiceprints"

    assert (
        get_voiceprint_review_backup_root(store_dir)
        == store_dir.resolve() / VOICEPRINT_PENDING_REVIEW_DIR
    )


def test_begin_transaction_writes_nothing_to_the_system_temp_dir(
    monkeypatch, tmp_path: Path
) -> None:
    """A real transaction snapshot must land under the store, never in the temp root."""
    fake_tmp = tmp_path / "system-tmp"
    fake_tmp.mkdir()
    monkeypatch.setattr(tempfile, "gettempdir", lambda: str(fake_tmp))
    store_dir = tmp_path / "voiceprints"
    db_path = store_dir / "voiceprints.sqlite"
    db_path.parent.mkdir(parents=True, exist_ok=True)
    db_path.write_text("db", encoding="utf-8")
    planned = _planned_capture(store_dir, db_path)

    txn = voiceprint_review_workflow._begin_transaction(
        tmp_path / "project", planned, frozenset()
    )

    assert txn.backup_dir.parent == get_voiceprint_review_backup_root(store_dir)
    assert list(fake_tmp.iterdir()) == []
    assert txn.db_backup_path.read_text(encoding="utf-8") == "db"


def test_pending_transaction_survives_a_wiped_system_temp_dir(
    monkeypatch, tmp_path: Path
) -> None:
    """Simulate the reboot: clear /tmp, restart the process, keep the rollback handle."""
    store_dir = tmp_path / "voiceprints"
    fake_tmp = tmp_path / "system-tmp"
    fake_tmp.mkdir()
    monkeypatch.setattr(tempfile, "gettempdir", lambda: str(fake_tmp))
    monkeypatch.setenv("XDG_DATA_HOME", str(tmp_path / "data"))
    del (
        store_dir
    )  # startup recovery only knows the default store; the web server uses it
    backup_dir = _write_snapshot(get_voiceprint_review_backup_root(), tmp_path)

    # The reboot: everything the old implementation relied on disappears.
    for stale in fake_tmp.iterdir():
        stale.unlink()

    registry = svc.CaptureTransactionRegistry(load_persisted=True)

    assert backup_dir.exists()
    assert "txn-durable" in registry._txns


def test_transactions_left_in_the_legacy_temp_root_are_still_recovered(
    monkeypatch, tmp_path: Path
) -> None:
    """Upgrading must not strand a transaction an older release wrote to /tmp."""
    fake_tmp = tmp_path / "system-tmp"
    fake_tmp.mkdir()
    monkeypatch.setattr(tempfile, "gettempdir", lambda: str(fake_tmp))
    monkeypatch.setenv("XDG_DATA_HOME", str(tmp_path / "data"))
    _write_snapshot(fake_tmp, tmp_path, txn_id="txn-legacy")

    registry = svc.CaptureTransactionRegistry(load_persisted=True)

    assert "txn-legacy" in registry._txns


def _write_snapshot(root: Path, tmp_path: Path, *, txn_id: str = "txn-durable") -> Path:
    """Write a snapshot directory complete enough for registry recovery."""
    backup_dir = root / f"{VOICEPRINT_REVIEW_BACKUP_PREFIX}{txn_id}"
    backup_dir.mkdir(parents=True)
    project_dir = tmp_path / "project"
    metadata = {
        "version": 1,
        "transaction_id": txn_id,
        "created_at": 1.0,
        "project_dir": str(project_dir),
        "backup_dir": str(backup_dir),
        "db_path": str(tmp_path / "voiceprints" / "voiceprints.sqlite"),
        "db_backup_path": str(backup_dir / "voiceprints.sqlite"),
        "db_existed": True,
        "project_manifest_path": str(project_dir / "project.json"),
        "project_manifest_backup_path": str(backup_dir / "project.json"),
        "project_manifest_existed": False,
        "match_path": str(project_dir / "speakers" / "speaker_matches.json"),
        "match_backup_path": str(backup_dir / "speaker_matches.json"),
        "match_existed": False,
        "clip_backups": [],
    }
    (backup_dir / svc._METADATA_FILE).write_text(json.dumps(metadata), encoding="utf-8")
    return backup_dir


def _planned_capture(store_dir: Path, db_path: Path):
    """Build the minimal capture summary ``_begin_transaction`` reads."""
    from app.voiceprints import VoiceprintCaptureSummary

    return VoiceprintCaptureSummary(store_dir, db_path, store_dir / "clips", [], False)
