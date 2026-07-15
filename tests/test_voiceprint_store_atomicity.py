"""Regression tests for voiceprint store consistency boundaries."""

from __future__ import annotations

import hashlib
from pathlib import Path

from app.voiceprint_embedding import LOCAL_SPEECHBRAIN_MODEL
from app.voiceprint_store import (
    StoredVoiceprintSample,
    get_voiceprint_db_path,
    list_all_voiceprint_samples,
    list_voiceprint_embeddings,
    store_voiceprint_samples,
    store_voiceprint_samples_with_rows,
    upsert_voiceprint_embedding,
)


def test_existing_path_refreshes_when_new_hash_already_exists_elsewhere(
    tmp_path: Path,
) -> None:
    """A reused path must keep its file, row metadata, and embedding consistent."""
    store_dir = tmp_path / "voiceprints"
    db_path = get_voiceprint_db_path(store_dir)
    first_clip = store_dir / "clips" / "project-1" / "speaker_0" / "clip_001.wav"
    second_clip = store_dir / "clips" / "project-2" / "speaker_0" / "clip_001.wav"
    first_clip.parent.mkdir(parents=True, exist_ok=True)
    second_clip.parent.mkdir(parents=True, exist_ok=True)
    first_clip.write_bytes(b"first audio")
    second_clip.write_bytes(b"second audio")
    store_voiceprint_samples(
        [
            _sample(first_clip, store_dir, project_id="project-1", text="old"),
            _sample(second_clip, store_dir, project_id="project-2", text="other"),
        ],
        db_path,
    )
    original = next(
        row
        for row in list_all_voiceprint_samples(db_path)
        if row.clip_path == first_clip
    )
    upsert_voiceprint_embedding(
        original.sample_id, LOCAL_SPEECHBRAIN_MODEL, [1.0, 0.0], db_path
    )

    first_clip.write_bytes(second_clip.read_bytes())
    _database_path, stored = store_voiceprint_samples_with_rows(
        [
            _sample(
                first_clip,
                store_dir,
                project_id="project-1",
                text="refreshed",
                person_id=original.speaker_id,
                begin_ms=9000,
            )
        ],
        db_path,
    )

    rows = list_all_voiceprint_samples(db_path)
    assert len(rows) == 1
    assert len({(row.speaker_id, row.clip_sha256) for row in rows}) == 1
    refreshed = rows[0]
    assert refreshed.clip_path == first_clip
    assert [row.sample_id for row in stored] == [original.sample_id]
    assert refreshed.public_id == original.public_id
    assert refreshed.clip_sha256 == hashlib.sha256(b"second audio").hexdigest()
    assert refreshed.source_begin_time_ms == 9000
    assert refreshed.transcript_text == "refreshed"
    assert list_voiceprint_embeddings(LOCAL_SPEECHBRAIN_MODEL, db_path) == []


def test_batch_hash_swap_keeps_both_stable_sample_rows(tmp_path: Path) -> None:
    """Final-state deduplication must allow two deterministic paths to swap bytes."""
    store_dir = tmp_path / "voiceprints"
    db_path = get_voiceprint_db_path(store_dir)
    first_clip = store_dir / "clips" / "project-1" / "speaker_0" / "clip_001.wav"
    second_clip = store_dir / "clips" / "project-2" / "speaker_0" / "clip_001.wav"
    first_clip.parent.mkdir(parents=True, exist_ok=True)
    second_clip.parent.mkdir(parents=True, exist_ok=True)
    first_clip.write_bytes(b"first audio")
    second_clip.write_bytes(b"second audio")
    store_voiceprint_samples(
        [
            _sample(first_clip, store_dir, project_id="project-1", text="first"),
            _sample(second_clip, store_dir, project_id="project-2", text="second"),
        ],
        db_path,
    )
    originals = {row.clip_path: row for row in list_all_voiceprint_samples(db_path)}
    person_id = originals[first_clip].speaker_id

    first_clip.write_bytes(b"second audio")
    second_clip.write_bytes(b"first audio")
    _database_path, stored = store_voiceprint_samples_with_rows(
        [
            _sample(
                first_clip,
                store_dir,
                project_id="project-1",
                text="first swapped",
                person_id=person_id,
            ),
            _sample(
                second_clip,
                store_dir,
                project_id="project-2",
                text="second swapped",
                person_id=person_id,
            ),
        ],
        db_path,
        require_all=True,
    )

    rows = {row.clip_path: row for row in list_all_voiceprint_samples(db_path)}
    assert len(stored) == 2
    assert len(rows) == 2
    assert rows[first_clip].sample_id == originals[first_clip].sample_id
    assert rows[second_clip].sample_id == originals[second_clip].sample_id
    assert rows[first_clip].clip_sha256 == hashlib.sha256(b"second audio").hexdigest()
    assert rows[second_clip].clip_sha256 == hashlib.sha256(b"first audio").hexdigest()


def _sample(
    clip_path: Path,
    store_dir: Path,
    *,
    project_id: str,
    text: str,
    person_id: int | None = None,
    begin_ms: int = 1000,
) -> StoredVoiceprintSample:
    """Build one stored sample fixture."""
    return StoredVoiceprintSample(
        speaker_name="Alice",
        person_id=person_id,
        project_id=project_id,
        project_path=store_dir / project_id,
        project_speaker_id=0,
        source_path=store_dir / "meeting.wav",
        clip_path=clip_path,
        clip_rel_path=str(clip_path.relative_to(store_dir)),
        source_begin_time_ms=begin_ms,
        source_end_time_ms=begin_ms + 1000,
        clip_begin_time_ms=begin_ms,
        clip_end_time_ms=begin_ms + 1000,
        transcript_text=text,
    )
