"""Tests for the sentence-reassignment downstream pipeline."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from app import sentence_reassignment as reassignment_module
from app.project_manager import create_project
from app.sentence_reassignment import (
    SentenceReassignmentApplyResult,
    apply_project_sentence_reassignments,
)
from app.speaker_labeling import SentenceReassignmentSpec
from app.voiceprint_models import StoredVoiceprintSample
from app.voiceprint_store import (
    delete_voiceprint_samples_by_ids,
    get_voiceprint_db_path,
    list_voiceprint_samples_for_project,
    store_voiceprint_samples,
)


def test_apply_returns_empty_result_when_no_reassignments(tmp_path: Path) -> None:
    """The orchestrator must short-circuit cleanly with no work."""
    project_dir = _make_project(tmp_path)
    _write_sentences(project_dir, _sentences_payload())

    result = apply_project_sentence_reassignments(project_dir, [], rematch=False)

    assert result == SentenceReassignmentApplyResult(
        sentence_files=(),
        anonymous_transcript_path=None,
        deleted_samples=(),
        match_summary=None,
        rematch_skipped_reason=None,
    )


def test_apply_writes_anonymous_transcript_with_new_speakers(tmp_path: Path) -> None:
    """Reassigning a sentence should refresh transcript_speakers.txt."""
    project_dir = _make_project(tmp_path)
    _write_sentences(project_dir, _sentences_payload())

    result = apply_project_sentence_reassignments(
        project_dir,
        [
            SentenceReassignmentSpec(
                sentence_id=2,
                begin_time_ms=2000,
                end_time_ms=2500,
                new_speaker_id=0,
                original_speaker_id=1,
            )
        ],
        rematch=False,
    )

    transcript_path = result.anonymous_transcript_path
    assert transcript_path is not None
    text = transcript_path.read_text(encoding="utf-8")
    # Sentence 2 should now appear under Speaker A (id 0), not Speaker B.
    assert "Speaker A:" in text
    assert "Speaker B:" not in text  # nothing left on speaker 1 in this fixture
    raw = json.loads((project_dir / "asr" / "sentences.json").read_text(encoding="utf-8"))
    assert raw["sentences"][1]["speaker_id"] == 0


def test_apply_invalidates_only_overlapping_voiceprint_samples(tmp_path: Path) -> None:
    """Voiceprint samples for the original speaker that overlap a reassigned sentence are dropped."""
    project_dir = _make_project(tmp_path)
    _write_sentences(project_dir, _sentences_payload())
    store_dir = tmp_path / "voiceprints"
    project_id = _project_id(project_dir)
    overlapping = _stored_sample(
        store_dir,
        project_dir / "source" / "meeting.mp4",
        speaker_name="speaker-1-overlap",
        project_id=project_id,
        project_speaker_id=1,
        source_begin_time_ms=2000,
        source_end_time_ms=2400,
        clip_filename="clip_overlap.wav",
    )
    other_speaker = _stored_sample(
        store_dir,
        project_dir / "source" / "meeting.mp4",
        speaker_name="speaker-0-untouched",
        project_id=project_id,
        project_speaker_id=0,
        source_begin_time_ms=0,
        source_end_time_ms=900,
        clip_filename="clip_other.wav",
    )
    other_time = _stored_sample(
        store_dir,
        project_dir / "source" / "meeting.mp4",
        speaker_name="speaker-1-other-time",
        project_id=project_id,
        project_speaker_id=1,
        source_begin_time_ms=10_000,
        source_end_time_ms=11_000,
        clip_filename="clip_other_time.wav",
    )
    store_voiceprint_samples(
        [overlapping, other_speaker, other_time],
        get_voiceprint_db_path(store_dir),
    )

    result = apply_project_sentence_reassignments(
        project_dir,
        [
            SentenceReassignmentSpec(
                sentence_id=2,
                begin_time_ms=2000,
                end_time_ms=2500,
                new_speaker_id=0,
                original_speaker_id=1,
            )
        ],
        store_dir=store_dir,
        rematch=False,
    )

    assert len(result.deleted_samples) == 1
    deleted_clip_names = {Path(item.clip_path).name for item in result.deleted_samples}
    assert deleted_clip_names == {"clip_overlap.wav"}
    # Untouched samples should still exist.
    remaining = list_voiceprint_samples_for_project(project_id, get_voiceprint_db_path(store_dir))
    remaining_clips = {row.clip_path.name for row in remaining}
    assert remaining_clips == {"clip_other.wav", "clip_other_time.wav"}


def test_apply_runs_rematch_when_requested(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """``rematch=True`` should invoke ``match_project_speakers`` once."""
    project_dir = _make_project(tmp_path)
    _write_sentences(project_dir, _sentences_payload())
    captured: list[Path] = []

    class _StubSummary:
        match_path = project_dir / "speakers" / "speaker_matches.json"
        matches: list[object] = []
        threshold = 0.75

    def _stub_match(project_root: Path, **_kwargs: object) -> _StubSummary:
        captured.append(project_root)
        return _StubSummary()

    monkeypatch.setattr(reassignment_module, "match_project_speakers", _stub_match)

    result = apply_project_sentence_reassignments(
        project_dir,
        [
            SentenceReassignmentSpec(
                sentence_id=2,
                begin_time_ms=2000,
                end_time_ms=2500,
                new_speaker_id=0,
                original_speaker_id=1,
            )
        ],
        rematch=True,
    )

    assert captured == [project_dir.resolve()]
    assert result.match_summary is not None
    assert result.rematch_skipped_reason is None


def test_apply_rematch_failure_is_reported_not_raised(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Rematch failure must surface as a skip reason rather than killing the save."""
    project_dir = _make_project(tmp_path)
    _write_sentences(project_dir, _sentences_payload())

    def _failing_match(*_args: object, **_kwargs: object) -> None:
        raise RuntimeError("voiceprint store unavailable")

    monkeypatch.setattr(reassignment_module, "match_project_speakers", _failing_match)

    result = apply_project_sentence_reassignments(
        project_dir,
        [
            SentenceReassignmentSpec(
                sentence_id=2,
                begin_time_ms=2000,
                end_time_ms=2500,
                new_speaker_id=0,
                original_speaker_id=1,
            )
        ],
        rematch=True,
    )

    assert result.match_summary is None
    assert result.rematch_skipped_reason == "voiceprint store unavailable"


def test_delete_voiceprint_samples_by_ids_removes_rows_and_clips(tmp_path: Path) -> None:
    """The bulk delete helper must remove SQLite rows and the underlying clip files."""
    store_dir = tmp_path / "voiceprints"
    project_dir = tmp_path / "project"
    project_dir.mkdir()
    source = project_dir / "source.mp4"
    source.write_bytes(b"fake")
    sample_one = _stored_sample(
        store_dir,
        source,
        speaker_name="dup-speaker",
        project_id="proj-1",
        project_speaker_id=0,
        source_begin_time_ms=0,
        source_end_time_ms=1000,
        clip_filename="one.wav",
    )
    sample_two = _stored_sample(
        store_dir,
        source,
        speaker_name="dup-speaker",
        project_id="proj-1",
        project_speaker_id=0,
        source_begin_time_ms=2000,
        source_end_time_ms=3000,
        clip_filename="two.wav",
    )
    db_path = store_voiceprint_samples([sample_one, sample_two], get_voiceprint_db_path(store_dir))
    rows = list_voiceprint_samples_for_project("proj-1", db_path)
    assert len(rows) == 2
    target_id = next(row.sample_id for row in rows if row.clip_path.name == "one.wav")
    target_clip = next(row.clip_path for row in rows if row.clip_path.name == "one.wav")

    deleted = delete_voiceprint_samples_by_ids([target_id], db_path=db_path)

    assert len(deleted) == 1
    assert deleted[0].sample_id == target_id
    assert deleted[0].clip_deleted is True
    assert not target_clip.exists()
    remaining = list_voiceprint_samples_for_project("proj-1", db_path)
    assert {row.clip_path.name for row in remaining} == {"two.wav"}


def _make_project(tmp_path: Path) -> Path:
    source = tmp_path / "meeting.mp4"
    source.write_bytes(b"fake video")
    project_dir = tmp_path / "project"
    create_project(
        source,
        title="Demo",
        projects_dir=tmp_path,
        project_dir=project_dir,
        meeting_time=None,
        hash_source=False,
    )
    return project_dir


def _project_id(project_dir: Path) -> str:
    manifest = json.loads((project_dir / "project.json").read_text(encoding="utf-8"))
    return str(manifest["project_id"])


def _sentences_payload() -> dict:
    return {
        "full_text": "第一句。第二句。",
        "detected_speakers": [0, 1],
        "sentences": [
            {
                "begin_time_ms": 0,
                "end_time_ms": 1000,
                "text": "第一句",
                "speaker_id": 0,
                "sentence_id": 1,
            },
            {
                "begin_time_ms": 2000,
                "end_time_ms": 2500,
                "text": "第二句",
                "speaker_id": 1,
                "sentence_id": 2,
            },
        ],
    }


def _write_sentences(project_dir: Path, payload: dict) -> None:
    asr_dir = project_dir / "asr"
    asr_dir.mkdir(parents=True, exist_ok=True)
    (asr_dir / "sentences.json").write_text(
        json.dumps(payload, ensure_ascii=False),
        encoding="utf-8",
    )


def _stored_sample(
    store_dir: Path,
    source_path: Path,
    *,
    speaker_name: str,
    project_id: str,
    project_speaker_id: int,
    source_begin_time_ms: int,
    source_end_time_ms: int,
    clip_filename: str,
) -> StoredVoiceprintSample:
    clip_path = store_dir / "clips" / project_id / f"speaker_{project_speaker_id}" / clip_filename
    clip_path.parent.mkdir(parents=True, exist_ok=True)
    clip_path.write_bytes(f"wav-{clip_filename}-{source_begin_time_ms}".encode())
    return StoredVoiceprintSample(
        speaker_name=speaker_name,
        project_id=project_id,
        project_path=source_path.parent,
        project_speaker_id=project_speaker_id,
        source_path=source_path,
        clip_path=clip_path,
        clip_rel_path=str(clip_path.relative_to(store_dir)),
        source_begin_time_ms=source_begin_time_ms,
        source_end_time_ms=source_end_time_ms,
        clip_begin_time_ms=0,
        clip_end_time_ms=source_end_time_ms - source_begin_time_ms,
        transcript_text="sample",
    )
