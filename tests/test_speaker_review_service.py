"""Unit tests for the presentation-neutral speaker-review save service."""

from __future__ import annotations

from pathlib import Path

import pytest

import app.core.speaker_review_service as svc


def _patch_apply(monkeypatch: pytest.MonkeyPatch) -> dict:
    """Stub apply_project_speakers and capture the kwargs it was called with."""
    captured: dict = {}

    def fake_apply(project_dir, mapping, **kwargs):
        captured["project_dir"] = project_dir
        captured["mapping"] = mapping
        captured.update(kwargs)
        return (Path("map.json"), Path("t.txt"), Path("s.srt"))

    monkeypatch.setattr(svc, "apply_project_speakers", fake_apply)
    return captured


def test_empty_ignored_set_passes_through_to_clear_stale_state(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Un-ignoring the last speaker sends an empty set, which must reach
    apply_project_speakers as an empty collection (not None) so the stale
    speaker_ignore.json gets cleared. `or None` here was a regression."""
    captured = _patch_apply(monkeypatch)

    svc.save_speaker_review(tmp_path, mapping={1: "Alice"}, ignored_speaker_ids=())

    assert captured["ignored_speaker_ids"] is not None
    assert list(captured["ignored_speaker_ids"]) == []


def test_nonempty_ignored_set_passes_through_unchanged(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """A populated ignore set must reach apply_project_speakers verbatim."""
    captured = _patch_apply(monkeypatch)

    svc.save_speaker_review(tmp_path, mapping={}, ignored_speaker_ids={3, 5})

    assert set(captured["ignored_speaker_ids"]) == {3, 5}
