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


class _FakeSummary:
    def __init__(self, txn: _FakeTxn) -> None:
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


def test_capture_conflict_is_not_a_value_error() -> None:
    """It must be RuntimeError so the web 400 ValueError handler doesn't swallow it."""
    assert issubclass(svc.CaptureConflictError, RuntimeError)
    assert not issubclass(svc.CaptureConflictError, ValueError)
