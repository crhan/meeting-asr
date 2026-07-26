"""Tests for voiceprint library health diagnostics."""

from __future__ import annotations

import json
from pathlib import Path

from typer.testing import CliRunner

from app.cli import app
from app.voiceprint_embedding import LOCAL_CAMPP_MODEL
from app.voiceprint_health import analyze_voiceprint_health
from app.voiceprint_store import (
    StoredVoiceprintSample,
    get_voiceprint_db_path,
    list_voiceprint_samples,
    store_voiceprint_samples,
    upsert_voiceprint_embedding,
)

runner = CliRunner()


def _sample(
    store_dir: Path,
    source_path: Path,
    speaker_name: str,
    *,
    project_id: str,
    index: int,
    duration_ms: int = 10_000,
    status: str = "active",
) -> StoredVoiceprintSample:
    """Build one stored sample fixture with a unique clip payload."""
    clip_path = (
        store_dir / "clips" / project_id / speaker_name / f"clip_{index:03d}.wav"
    )
    clip_path.parent.mkdir(parents=True, exist_ok=True)
    clip_path.write_bytes(f"{speaker_name}-{project_id}-{index}".encode())
    begin_ms = index * 60_000
    return StoredVoiceprintSample(
        speaker_name=speaker_name,
        project_id=project_id,
        project_path=store_dir / project_id,
        project_speaker_id=0,
        source_path=source_path,
        clip_path=clip_path,
        clip_rel_path=str(clip_path.relative_to(store_dir)),
        source_begin_time_ms=begin_ms,
        source_end_time_ms=begin_ms + duration_ms,
        clip_begin_time_ms=0,
        clip_end_time_ms=duration_ms,
        transcript_text=f"{speaker_name} sample {index}",
        sample_status=status,
    )


def _store_person(
    store_dir: Path,
    source_path: Path,
    name: str,
    *,
    vectors: list[list[float] | None],
    projects: list[str],
    duration_ms: int = 10_000,
    status: str = "active",
) -> None:
    """Store one person's samples and optional embeddings."""
    samples = [
        _sample(
            store_dir,
            source_path,
            name,
            project_id=projects[index % len(projects)],
            index=index,
            duration_ms=duration_ms,
            status=status,
        )
        for index in range(len(vectors))
    ]
    db_path = store_voiceprint_samples(samples, get_voiceprint_db_path(store_dir))
    rows = list_voiceprint_samples(name, db_path)
    for row, vector in zip(rows, vectors, strict=True):
        if vector is not None:
            upsert_voiceprint_embedding(
                row.sample_id, LOCAL_CAMPP_MODEL, vector, db_path
            )


def _person(report: object, name: str) -> object:
    """Return one person from a health report by name."""
    for person in report.people:
        if person.speaker_name == name:
            return person
    raise AssertionError(f"person not found: {name}")


def _check(person: object, key: str) -> object:
    """Return the first check with a key."""
    for check in person.checks:
        if check.key == key:
            return check
    raise AssertionError(f"check not found: {key}")


def test_healthy_people_report_ok(tmp_path: Path) -> None:
    source = tmp_path / "meeting.mp4"
    source.write_bytes(b"source")
    store_dir = tmp_path / "voiceprints"
    _store_person(
        store_dir,
        source,
        "Alice",
        vectors=[[1.0, 0.0], [0.98, 0.02], [0.99, 0.01]],
        projects=["p-one", "p-two"],
    )
    _store_person(
        store_dir,
        source,
        "Bob",
        vectors=[[0.0, 1.0], [0.02, 0.98], [0.01, 0.99]],
        projects=["p-one", "p-two"],
    )
    report = analyze_voiceprint_health(store_dir=store_dir)
    assert report.model == LOCAL_CAMPP_MODEL
    assert report.ok_count == 2
    assert report.critical_count == 0
    alice = _person(report, "Alice")
    assert alice.level == "ok"
    assert alice.matching_sample_count == 3
    assert alice.matching_seconds == 30.0
    assert alice.project_count == 2
    assert alice.nearest_name == "Bob"
    assert alice.nearest_score is not None and alice.nearest_score < 0.65
    assert {check.key for check in alice.checks} >= {
        "coverage",
        "diversity",
        "embedding",
        "clips",
        "cohesion",
        "separation",
    }
    assert not alice.issues


def test_missing_embeddings_flagged(tmp_path: Path) -> None:
    source = tmp_path / "meeting.mp4"
    source.write_bytes(b"source")
    store_dir = tmp_path / "voiceprints"
    _store_person(
        store_dir,
        source,
        "NoEmbed",
        vectors=[None, None, None],
        projects=["p-one", "p-two"],
    )
    _store_person(
        store_dir,
        source,
        "PartialEmbed",
        vectors=[[1.0, 0.0], [0.99, 0.01], [0.98, 0.02], None],
        projects=["p-one", "p-two"],
    )
    report = analyze_voiceprint_health(store_dir=store_dir)
    no_embed = _person(report, "NoEmbed")
    assert no_embed.level == "critical"
    assert no_embed.missing_embedding_count == 3
    embedding_check = _check(no_embed, "embedding")
    assert embedding_check.level == "critical"
    assert "invisible" in embedding_check.detail
    assert embedding_check.action == "meeting-asr voiceprint embed"
    partial = _person(report, "PartialEmbed")
    assert partial.missing_embedding_count == 1
    assert _check(partial, "embedding").level == "warning"


def test_quarantined_only_person_is_critical(tmp_path: Path) -> None:
    source = tmp_path / "meeting.mp4"
    source.write_bytes(b"source")
    store_dir = tmp_path / "voiceprints"
    _store_person(
        store_dir,
        source,
        "Quarantined",
        vectors=[[1.0, 0.0], [0.99, 0.01]],
        projects=["p-one"],
        status="quarantined",
    )
    report = analyze_voiceprint_health(store_dir=store_dir)
    person = _person(report, "Quarantined")
    assert person.level == "critical"
    assert person.matching_sample_count == 0
    assert person.sample_count == 2
    coverage = _check(person, "coverage")
    assert coverage.level == "critical"
    assert "none active" in coverage.detail


def test_low_coverage_warnings(tmp_path: Path) -> None:
    source = tmp_path / "meeting.mp4"
    source.write_bytes(b"source")
    store_dir = tmp_path / "voiceprints"
    _store_person(
        store_dir,
        source,
        "Sparse",
        vectors=[[1.0, 0.0], [0.99, 0.01]],
        projects=["p-one"],
        duration_ms=2_000,
    )
    report = analyze_voiceprint_health(store_dir=store_dir)
    person = _person(report, "Sparse")
    assert person.level == "warning"
    assert _check(person, "coverage").level == "warning"
    assert _check(person, "duration").level == "warning"
    assert _check(person, "diversity").level == "warning"


def test_confusable_people_are_critical(tmp_path: Path) -> None:
    source = tmp_path / "meeting.mp4"
    source.write_bytes(b"source")
    store_dir = tmp_path / "voiceprints"
    _store_person(
        store_dir,
        source,
        "Alice",
        vectors=[[1.0, 0.0], [0.99, 0.01], [0.98, 0.02]],
        projects=["p-one", "p-two"],
    )
    _store_person(
        store_dir,
        source,
        "AliceTwin",
        vectors=[[0.99, 0.02], [1.0, 0.01], [0.98, 0.01]],
        projects=["p-one", "p-two"],
    )
    report = analyze_voiceprint_health(store_dir=store_dir)
    alice = _person(report, "Alice")
    assert alice.level == "critical"
    separation = _check(alice, "separation")
    assert separation.level == "critical"
    assert "AliceTwin" in separation.detail
    assert "people merge" in separation.action


def test_missing_clip_files_flagged(tmp_path: Path) -> None:
    source = tmp_path / "meeting.mp4"
    source.write_bytes(b"source")
    store_dir = tmp_path / "voiceprints"
    _store_person(
        store_dir,
        source,
        "ClipLoss",
        vectors=[[1.0, 0.0], [0.99, 0.01], [0.98, 0.02], None],
        projects=["p-one", "p-two"],
    )
    db_path = get_voiceprint_db_path(store_dir)
    rows = list_voiceprint_samples("ClipLoss", db_path)
    # Embedded sample loses its clip: repairable-later warning.
    rows[0].clip_path.unlink()
    # Un-embedded sample loses its clip too: dead sample, critical.
    rows[-1].clip_path.unlink()
    report = analyze_voiceprint_health(store_dir=store_dir)
    person = _person(report, "ClipLoss")
    assert person.missing_clip_count == 2
    clips = _check(person, "clips")
    assert clips.level == "critical"
    assert "cannot be repaired" in clips.detail
    assert person.level == "critical"


def test_cli_health_json_and_table(tmp_path: Path) -> None:
    source = tmp_path / "meeting.mp4"
    source.write_bytes(b"source")
    store_dir = tmp_path / "voiceprints"
    _store_person(
        store_dir,
        source,
        "Alice",
        vectors=[[1.0, 0.0], [0.98, 0.02], [0.99, 0.01]],
        projects=["p-one", "p-two"],
    )
    result = runner.invoke(
        app, ["voiceprint", "health", "--store-dir", str(store_dir), "--json"]
    )
    assert result.exit_code == 0, result.output
    payload = json.loads(result.output)
    assert payload["model"] == LOCAL_CAMPP_MODEL
    assert payload["people_count"] == 1
    assert payload["people"][0]["speaker_name"] == "Alice"
    assert payload["people"][0]["level"] == "ok"
    assert {check["key"] for check in payload["people"][0]["checks"]} >= {
        "coverage",
        "embedding",
        "cohesion",
    }
    table_result = runner.invoke(
        app, ["voiceprint", "health", "--store-dir", str(store_dir)]
    )
    assert table_result.exit_code == 0, table_result.output
    assert f"Model: {LOCAL_CAMPP_MODEL}" in table_result.output
    assert "Alice" in table_result.output


def test_cli_health_speaker_filter_and_empty_store(tmp_path: Path) -> None:
    source = tmp_path / "meeting.mp4"
    source.write_bytes(b"source")
    store_dir = tmp_path / "voiceprints"
    _store_person(
        store_dir,
        source,
        "Alice",
        vectors=[[1.0, 0.0]],
        projects=["p-one"],
    )
    _store_person(
        store_dir,
        source,
        "Bob",
        vectors=[[0.0, 1.0]],
        projects=["p-one"],
    )
    report = analyze_voiceprint_health(store_dir=store_dir, speaker="alice")
    assert [person.speaker_name for person in report.people] == ["Alice"]
    empty = runner.invoke(
        app, ["voiceprint", "health", "--store-dir", str(tmp_path / "missing")]
    )
    assert empty.exit_code == 0, empty.output
    assert "No voiceprint people found." in empty.output
