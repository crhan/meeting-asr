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


def test_deleted_speakers_are_stripped_before_mapping_apply(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Deleted speakers must not leave stale names, person bindings, or ignore flags."""
    captured = _patch_apply(monkeypatch)
    deleted_calls: list[tuple[Path, list[int]]] = []

    def fake_delete(project_dir, speaker_ids):
        deleted_calls.append((project_dir, list(speaker_ids)))
        return svc.EmptySpeakerDeletionApplyResult(
            sentence_files=(),
            anonymous_transcript_path=None,
            deleted_sentence_count=0,
        )

    monkeypatch.setattr(svc, "apply_project_empty_speaker_deletions", fake_delete)

    result = svc.save_speaker_review(
        tmp_path,
        mapping={1: "Alice", 2: "Delete Me"},
        person_mapping={1: 11, 2: 22},
        person_public_mapping={1: "vpp-0000000000000001", 2: "vpp-0000000000000002"},
        ignored_speaker_ids=[2],
        deleted_speaker_ids=[2],
    )

    assert deleted_calls == [(tmp_path, [2])]
    assert captured["mapping"] == {1: "Alice"}
    assert captured["person_mapping"] == {1: 11}
    assert captured["person_public_mapping"] == {1: "vpp-0000000000000001"}
    assert list(captured["ignored_speaker_ids"]) == []
    assert result.deletion is not None


def test_new_person_names_create_and_bind_voiceprint_person(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """A free-name identity save should create a stable person object and bind it."""
    captured = _patch_apply(monkeypatch)
    store_dir = tmp_path / "voiceprints"

    result = svc.save_speaker_review(
        tmp_path,
        mapping={1: "Charlie"},
        new_person_names={1: "Charlie"},
        store_dir=store_dir,
    )

    public_id = captured["person_public_mapping"][1]
    assert public_id.startswith("vpp-")
    assert captured["mapping"] == {1: "Charlie"}
    assert captured["person_mapping"] == {}
    assert result.created_person_count == 1

    second = svc.save_speaker_review(
        tmp_path,
        mapping={2: "Charlie"},
        new_person_names={2: "Charlie"},
        store_dir=store_dir,
    )

    assert captured["person_public_mapping"][2] == public_id
    assert second.created_person_count == 0
