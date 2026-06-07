"""Unit tests for the cross-request capture transaction registry exclusivity.

A pending capture holds a rollback snapshot of the global voiceprint store; a second
overlapping capture (or any store edit) would make rollback restore a stale snapshot and
silently discard the intervening writes. The registry enforces one-pending-at-a-time.
"""

from __future__ import annotations

from pathlib import Path

import pytest

import app.core.voiceprint_review_service as svc


class _FakeTxn:
    def __init__(self) -> None:
        self.accepted = False
        self.rolled_back = False

    def accept(self) -> None:
        self.accepted = True

    def rollback(self) -> None:
        self.rolled_back = True


class _FailingTxn(_FakeTxn):
    def __init__(
        self, *, fail_accept: bool = False, fail_rollback: bool = False
    ) -> None:
        super().__init__()
        self.fail_accept = fail_accept
        self.fail_rollback = fail_rollback

    def accept(self) -> None:
        if self.fail_accept:
            raise RuntimeError("accept failed")
        super().accept()

    def rollback(self) -> None:
        if self.fail_rollback:
            raise RuntimeError("rollback failed")
        super().rollback()


class _FakeSummary:
    def __init__(self, txn: object) -> None:
        self.transaction = txn


def _run(reg: svc.CaptureTransactionRegistry) -> tuple[str, _FakeSummary]:
    return reg.run(
        project_dir=Path("/does/not/matter"),
        planned=None,  # the fake workflow ignores it
        selected_clip_rel_paths=frozenset(),
        store_dir=None,
    )


def test_registry_rejects_overlapping_pending(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        svc, "run_voiceprint_review_workflow", lambda **_: _FakeSummary(_FakeTxn())
    )
    reg = svc.CaptureTransactionRegistry()

    assert not reg.has_pending()
    txn_id, _ = _run(reg)
    assert reg.has_pending()

    # A second run while one is unresolved must be refused, not snapshot a tainted store.
    with pytest.raises(svc.CaptureConflictError):
        _run(reg)

    # Resolving the first frees the registry for the next capture.
    reg.accept(txn_id)
    assert not reg.has_pending()
    txn_id2, _ = _run(reg)
    assert reg.has_pending()
    reg.rollback(txn_id2)
    assert not reg.has_pending()


def test_run_store_write_refused_while_pending(monkeypatch: pytest.MonkeyPatch) -> None:
    """Store writes must join the capture critical section and be refused while pending."""
    monkeypatch.setattr(
        svc, "run_voiceprint_review_workflow", lambda **_: _FakeSummary(_FakeTxn())
    )
    reg = svc.CaptureTransactionRegistry()

    # No pending capture: the write runs.
    assert reg.run_store_write(lambda: "ok") == "ok"

    txn_id, _ = _run(reg)
    # Pending capture: the same write path is refused, not silently allowed through.
    with pytest.raises(svc.CaptureConflictError):
        reg.run_store_write(lambda: "should-not-run")

    reg.accept(txn_id)
    assert reg.run_store_write(lambda: "ok-again") == "ok-again"


def test_rollback_restores_under_store_write_lock(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The snapshot restore must hold the store-write lock so no write races it."""
    reg = svc.CaptureTransactionRegistry()
    observed: dict[str, bool] = {}

    class _LockAssertTxn:
        def accept(self) -> None:  # pragma: no cover - not exercised here
            pass

        def rollback(self) -> None:
            # The same lock every global-store write takes must be held during restore.
            observed["locked_during_restore"] = reg._store_write_lock.locked()

    monkeypatch.setattr(
        svc,
        "run_voiceprint_review_workflow",
        lambda **_: _FakeSummary(_LockAssertTxn()),
    )

    txn_id, _ = _run(reg)
    reg.rollback(txn_id)

    assert observed["locked_during_restore"] is True
    assert not reg.has_pending()


def test_has_pending_does_not_auto_accept_stale_transactions(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Age alone must not silently accept or clear a pending capture transaction."""
    fake = _FakeTxn()
    monkeypatch.setattr(
        svc, "run_voiceprint_review_workflow", lambda **_: _FakeSummary(fake)
    )
    reg = svc.CaptureTransactionRegistry()
    txn_id, _ = _run(reg)
    assert reg.has_pending() is True  # fresh -> still blocking

    # Age past the historical sweep cutoff; it must still wait for an explicit decision.
    created, txn, project = reg._txns[txn_id]
    reg._txns[txn_id] = (created - svc._ORPHAN_MAX_AGE_SECONDS - 1, txn, project)
    assert reg.has_pending() is True
    assert fake.accepted is False
    assert txn_id in reg._txns


def test_accept_failure_keeps_transaction_handle(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A failed accept must remain retryable instead of popping the pending handle."""
    fake = _FailingTxn(fail_accept=True)
    monkeypatch.setattr(
        svc, "run_voiceprint_review_workflow", lambda **_: _FakeSummary(fake)
    )
    reg = svc.CaptureTransactionRegistry()
    txn_id, _ = _run(reg)

    with pytest.raises(RuntimeError, match="accept failed"):
        reg.accept(txn_id)

    assert reg.has_pending() is True
    assert reg.project_dir_for(txn_id) == Path("/does/not/matter").resolve()


def test_rollback_failure_keeps_transaction_handle(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A failed rollback must remain retryable instead of popping the pending handle."""
    fake = _FailingTxn(fail_rollback=True)
    monkeypatch.setattr(
        svc, "run_voiceprint_review_workflow", lambda **_: _FakeSummary(fake)
    )
    reg = svc.CaptureTransactionRegistry()
    txn_id, _ = _run(reg)

    with pytest.raises(RuntimeError, match="rollback failed"):
        reg.rollback(txn_id)

    assert reg.has_pending() is True
    assert reg.project_dir_for(txn_id) == Path("/does/not/matter").resolve()


def test_persisted_transaction_restores_after_registry_recreation(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """A service restart must recover the pending accept/rollback transaction handle."""
    monkeypatch.setattr(svc.tempfile, "gettempdir", lambda: str(tmp_path))
    backup_dir = tmp_path / "meeting-asr-voiceprint-review-abc"
    backup_dir.mkdir()
    project_dir = tmp_path / "project"
    project_dir.mkdir()
    db_path = tmp_path / "store" / "voiceprints.sqlite"
    db_path.parent.mkdir()
    db_path.write_text("after", encoding="utf-8")
    db_backup_path = backup_dir / "voiceprints.sqlite"
    db_backup_path.write_text("before", encoding="utf-8")
    manifest_path = project_dir / "project.json"
    manifest_path.write_text('{"status":"after"}', encoding="utf-8")
    manifest_backup_path = backup_dir / "project.json"
    manifest_backup_path.write_text('{"status":"before"}', encoding="utf-8")
    match_path = project_dir / "speakers" / "speaker_matches.json"
    match_path.parent.mkdir()
    match_path.write_text('{"after":true}', encoding="utf-8")
    match_backup_path = backup_dir / "speaker_matches.json"
    match_backup_path.write_text('{"before":true}', encoding="utf-8")
    txn = svc.VoiceprintReviewTransaction(
        backup_dir=backup_dir,
        db_path=db_path,
        db_backup_path=db_backup_path,
        db_existed=True,
        project_manifest_path=manifest_path,
        project_manifest_backup_path=manifest_backup_path,
        project_manifest_existed=True,
        match_path=match_path,
        match_backup_path=match_backup_path,
        match_existed=True,
        clip_backups=(),
    )
    monkeypatch.setattr(
        svc, "run_voiceprint_review_workflow", lambda **_: _FakeSummary(txn)
    )

    reg = svc.CaptureTransactionRegistry()
    txn_id, _ = reg.run(
        project_dir=project_dir,
        planned=None,
        selected_clip_rel_paths=frozenset(),
        store_dir=None,
    )
    restored = svc.CaptureTransactionRegistry(load_persisted=True)

    assert restored.pending_transaction() == (txn_id, project_dir.resolve())
    restored.rollback(txn_id)
    assert db_path.read_text(encoding="utf-8") == "before"
    assert manifest_path.read_text(encoding="utf-8") == '{"status":"before"}'
    assert match_path.read_text(encoding="utf-8") == '{"before":true}'
    assert not backup_dir.exists()


def test_pending_transaction_exposes_id_and_project(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The recovery banner needs the pending txn id + its project even when the originating
    page is gone."""
    monkeypatch.setattr(
        svc, "run_voiceprint_review_workflow", lambda **_: _FakeSummary(_FakeTxn())
    )
    reg = svc.CaptureTransactionRegistry()
    assert reg.pending_transaction() is None
    txn_id, _ = _run(reg)
    pending = reg.pending_transaction()
    assert pending is not None
    assert pending[0] == txn_id
    assert pending[1] == Path("/does/not/matter").resolve()
    reg.accept(txn_id)
    assert reg.pending_transaction() is None


def test_capture_conflict_is_not_a_value_error() -> None:
    """It must be RuntimeError so the web 400 ValueError handler doesn't swallow it."""
    assert issubclass(svc.CaptureConflictError, RuntimeError)
    assert not issubclass(svc.CaptureConflictError, ValueError)
