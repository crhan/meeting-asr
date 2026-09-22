"""Tests for the process-wide torch thread budget."""

from __future__ import annotations

import sys
from types import SimpleNamespace

import pytest

import app.infra.torch_runtime as tr


@pytest.fixture(autouse=True)
def _reset_applied(monkeypatch: pytest.MonkeyPatch) -> None:
    """Each test starts from an un-applied budget and a fake torch module."""
    monkeypatch.setattr(tr, "_APPLIED", None)


def _fake_torch(monkeypatch: pytest.MonkeyPatch) -> list[int]:
    calls: list[int] = []
    monkeypatch.setitem(
        sys.modules, "torch", SimpleNamespace(set_num_threads=calls.append)
    )
    return calls


def test_default_budget_is_one_thread(monkeypatch: pytest.MonkeyPatch) -> None:
    """Without config the embedding models run single-threaded per call; the bulk
    callers parallelize over a thread pool instead of oversubscribing OpenMP."""
    monkeypatch.setattr(tr, "get_configured_torch_threads", lambda: None)
    calls = _fake_torch(monkeypatch)

    assert tr.configure_torch_threads() == tr.DEFAULT_TORCH_THREADS == 1
    assert calls == [1]


def test_configured_budget_is_applied_once(monkeypatch: pytest.MonkeyPatch) -> None:
    """The configured value wins and torch is only touched on the first call."""
    monkeypatch.setattr(tr, "get_configured_torch_threads", lambda: 3)
    calls = _fake_torch(monkeypatch)

    assert tr.configure_torch_threads() == 3
    assert tr.configure_torch_threads() == 3
    assert calls == [3]
