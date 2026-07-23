"""Tests for cross-project voiceprint commands."""

from __future__ import annotations

import json
import re
import wave
from pathlib import Path

from typer.testing import CliRunner

from app.cli import app
from app.project_manager import create_project, load_manifest
from app.voiceprint_embedding import LOCAL_CAMPP_MODEL
from app.voiceprint_people import (
    create_voiceprint_person,
    get_voiceprint_person,
    merge_voiceprint_people,
)
from app.voiceprint_store import (
    StoredVoiceprintSample,
    get_voiceprint_db_path,
    list_voiceprint_samples,
    list_voiceprint_embeddings,
    list_voiceprint_samples_for_project,
    store_voiceprint_samples,
    update_voiceprint_sample_status,
    upsert_voiceprint_embedding,
)

runner = CliRunner()


def test_voiceprint_capture_writes_xdg_store_and_sqlite(
    monkeypatch,
    tmp_path: Path,
) -> None:
    """Capture should store WAV clips outside the project and index them in SQLite."""
    project_dir = _sample_project(tmp_path)
    store_dir = tmp_path / "data" / "meeting-asr" / "voiceprints"
    _write_named_speaker_inputs(project_dir)
    monkeypatch.setattr("app.voiceprints.extract_audio_clip", _fake_extract_audio_clip)

    result = runner.invoke(
        app,
        [
            "voiceprint",
            "capture",
            str(project_dir),
            "--sample-count",
            "1",
            "--store-dir",
            str(store_dir),
        ],
    )

    manifest = load_manifest(project_dir)
    assert result.exit_code == 0
    assert "Captured voiceprint samples: 2" in result.output
    assert "Next steps:" in result.output
    assert (
        f"meeting-asr voiceprint embed --store-dir {store_dir.resolve()}"
        in result.output
    )
    assert "meeting-asr voiceprint list" in result.output
    assert (store_dir / "voiceprints.sqlite").exists()
    assert (
        store_dir / "clips" / manifest.project_id / "speaker_0" / "clip_001.wav"
    ).exists()
    assert not (project_dir / "speakers" / "voiceprints").exists()
    assert manifest.speakers["voiceprints"]["sample_count"] == 2
    project_samples = list_voiceprint_samples_for_project(
        manifest.project_id, get_voiceprint_db_path(store_dir)
    )
    assert len(project_samples) == 2
    assert {sample.project_id for sample in project_samples} == {manifest.project_id}

    list_result = runner.invoke(
        app, ["voiceprint", "list", "--store-dir", str(store_dir)]
    )
    list_plain_result = runner.invoke(
        app, ["voiceprint", "list", "--store-dir", str(store_dir), "--plain"]
    )
    speaker_id = _speaker_id_from_list(list_result.output, "欧丁")
    show_result = runner.invoke(
        app, ["voiceprint", "show", speaker_id, "--store-dir", str(store_dir)]
    )
    show_by_name_result = runner.invoke(
        app, ["voiceprint", "show", "欧丁", "--store-dir", str(store_dir)]
    )
    list_json_result = runner.invoke(
        app, ["voiceprint", "list", "--store-dir", str(store_dir), "--json"]
    )
    show_json_result = runner.invoke(
        app, ["voiceprint", "show", speaker_id, "--store-dir", str(store_dir), "--json"]
    )
    list_payload = json.loads(list_json_result.output)
    show_payload = json.loads(show_json_result.output)

    assert list_result.exit_code == 0
    assert "Speakers: 2 | Samples: 2 | Embedded samples: 0/2" in list_result.output
    assert re.fullmatch(r"vpp-[0-9a-f]{16}", speaker_id)
    assert "ID" in list_result.output
    assert "Speaker" in list_result.output
    assert "Embedded" in list_result.output
    assert "欧丁" in list_result.output
    assert list_plain_result.exit_code == 0
    assert list_plain_result.output.splitlines()[0] == (
        "id\tinternal_id\tspeaker\tsamples\tprojects\tembedded\tmodels\tupdated"
    )
    assert "\t欧丁\t" in list_plain_result.output
    assert "╭" not in list_plain_result.output
    assert show_result.exit_code == 0
    assert show_by_name_result.exit_code == 0
    assert "[1] 欧丁" in show_result.output
    assert f"person_id: {speaker_id}" in show_result.output
    assert re.search(r"sample_id: vps-[0-9a-f]{16}", show_result.output)
    assert "sample_id:" in show_result.output
    assert manifest.project_id in show_result.output
    assert "clip_002.wav" in show_result.output
    assert list_json_result.exit_code == 0
    assert list_payload["database"] == str(store_dir.resolve() / "voiceprints.sqlite")
    assert list_payload["count"] == 2
    assert list_payload["sample_count"] == 2
    assert any(speaker["name"] == "欧丁" for speaker in list_payload["speakers"])
    assert re.fullmatch(r"vpp-[0-9a-f]{16}", list_payload["speakers"][0]["public_id"])
    assert show_json_result.exit_code == 0
    assert show_payload["speaker"] == speaker_id
    assert show_payload["count"] == 1
    assert re.fullmatch(r"vps-[0-9a-f]{16}", show_payload["samples"][0]["public_id"])
    assert show_payload["samples"][0]["speaker_public_id"] == speaker_id
    assert show_payload["samples"][0]["speaker_name"] == "欧丁"
    assert show_payload["samples"][0]["project_id"] == manifest.project_id
    assert show_payload["samples"][0]["clip_path"].endswith("clip_002.wav")


def test_voiceprint_people_lifecycle_uses_stable_ids(tmp_path: Path) -> None:
    """People commands should create and rename by stable id, not by display name."""
    store_dir = tmp_path / "voiceprints"

    add_result = runner.invoke(
        app, ["voiceprint", "people", "add", "欧丁", "--store-dir", str(store_dir)]
    )
    duplicate_result = runner.invoke(
        app, ["voiceprint", "people", "add", "欧丁", "--store-dir", str(store_dir)]
    )
    list_result = runner.invoke(
        app, ["voiceprint", "people", "list", "--store-dir", str(store_dir)]
    )
    person_id = _speaker_id_from_list(list_result.output, "欧丁")
    rename_result = runner.invoke(
        app,
        [
            "voiceprint",
            "people",
            "rename",
            person_id,
            "欧丁-新版",
            "--store-dir",
            str(store_dir),
        ],
    )
    show_result = runner.invoke(
        app, ["voiceprint", "people", "show", person_id, "--store-dir", str(store_dir)]
    )

    assert add_result.exit_code == 0
    assert "Person ID:" in add_result.output
    assert duplicate_result.exit_code != 0
    assert "already exists" in duplicate_result.output
    assert list_result.exit_code == 0
    assert "People: 1 | Samples: 0 | Embedded samples: 0/0" in list_result.output
    assert rename_result.exit_code == 0
    assert f"Renamed person {person_id}: 欧丁-新版" in rename_result.output
    assert show_result.exit_code == 0
    assert "Name: 欧丁-新版" in show_result.output


def test_voiceprint_browse_summary_uses_global_store(
    monkeypatch,
    tmp_path: Path,
) -> None:
    """Browse summary should expose the same global library data as the TUI."""
    project_dir = _sample_project(tmp_path)
    store_dir = tmp_path / "data" / "meeting-asr" / "voiceprints"
    _write_named_speaker_inputs(project_dir)
    monkeypatch.setattr("app.voiceprints.extract_audio_clip", _fake_extract_audio_clip)
    runner.invoke(
        app,
        [
            "voiceprint",
            "capture",
            str(project_dir),
            "--sample-count",
            "1",
            "--store-dir",
            str(store_dir),
        ],
    )

    result = runner.invoke(
        app, ["voiceprint", "browse", "--summary", "--store-dir", str(store_dir)]
    )
    tui_result = runner.invoke(
        app, ["voiceprint", "browse", "--store-dir", str(store_dir)]
    )

    assert result.exit_code == 0
    assert (
        f"Voiceprint library: {store_dir.resolve() / 'voiceprints.sqlite'}"
        in result.output
    )
    assert "Speakers: 2 | Samples: 2 | Embedded: 0/2" in result.output
    assert "欧丁 id=" in result.output
    assert tui_result.exit_code != 0
    assert "requires an interactive terminal" in tui_result.output


def test_voiceprint_capture_skips_anonymous_speaker_labels(
    monkeypatch,
    tmp_path: Path,
) -> None:
    """Anonymous fallback labels should not become voiceprint identities."""
    project_dir = _sample_project(tmp_path)
    store_dir = tmp_path / "data" / "meeting-asr" / "voiceprints"
    _write_partially_named_speaker_inputs(project_dir)
    monkeypatch.setattr("app.voiceprints.extract_audio_clip", _fake_extract_audio_clip)

    result = runner.invoke(
        app,
        [
            "voiceprint",
            "capture",
            str(project_dir),
            "--sample-count",
            "1",
            "--store-dir",
            str(store_dir),
        ],
    )

    manifest = load_manifest(project_dir)
    list_result = runner.invoke(
        app, ["voiceprint", "list", "--store-dir", str(store_dir)]
    )

    assert result.exit_code == 0
    assert "Captured voiceprint samples: 1" in result.output
    assert (
        store_dir / "clips" / manifest.project_id / "speaker_0" / "clip_001.wav"
    ).exists()
    assert not (store_dir / "clips" / manifest.project_id / "speaker_2").exists()
    assert "Speakers: 1 | Samples: 1 | Embedded samples: 0/1" in list_result.output
    assert "欧丁" in list_result.output
    assert "Speaker C" not in list_result.output


def test_voiceprint_capture_uses_project_person_map(
    monkeypatch,
    tmp_path: Path,
) -> None:
    """Capture should attach samples to existing person ids when the project saved a person map."""
    project_dir = _sample_project(tmp_path)
    store_dir = tmp_path / "data" / "meeting-asr" / "voiceprints"
    _write_partially_named_speaker_inputs(project_dir)
    add_result = runner.invoke(
        app, ["voiceprint", "people", "add", "欧丁", "--store-dir", str(store_dir)]
    )
    public_id = add_result.output.split("Person ID:", 1)[1].strip().splitlines()[0]
    person_payload = json.loads(
        runner.invoke(
            app,
            [
                "voiceprint",
                "people",
                "show",
                public_id,
                "--store-dir",
                str(store_dir),
                "--json",
            ],
        ).output
    )
    person_id = int(person_payload["speaker_id"])
    (project_dir / "speakers" / "speaker_person_map.json").write_text(
        json.dumps({"0": public_id}, ensure_ascii=False),
        encoding="utf-8",
    )
    monkeypatch.setattr("app.voiceprints.extract_audio_clip", _fake_extract_audio_clip)

    result = runner.invoke(
        app,
        [
            "voiceprint",
            "capture",
            str(project_dir),
            "--sample-count",
            "1",
            "--store-dir",
            str(store_dir),
        ],
    )

    manifest = load_manifest(project_dir)
    samples = list_voiceprint_samples_for_project(
        manifest.project_id, get_voiceprint_db_path(store_dir)
    )

    assert result.exit_code == 0
    assert f"person {public_id}" in result.output
    assert len(samples) == 1
    assert samples[0].speaker_id == person_id
    assert samples[0].speaker_name == "欧丁"


def test_voiceprint_play_dry_run_prints_clip_command(
    monkeypatch,
    tmp_path: Path,
) -> None:
    """Play should target one numbered clip without modifying the store."""
    project_dir = _sample_project(tmp_path)
    store_dir = tmp_path / "data" / "meeting-asr" / "voiceprints"
    _write_named_speaker_inputs(project_dir)
    monkeypatch.setattr("app.voiceprints.extract_audio_clip", _fake_extract_audio_clip)
    runner.invoke(
        app,
        [
            "voiceprint",
            "capture",
            str(project_dir),
            "--sample-count",
            "1",
            "--store-dir",
            str(store_dir),
        ],
    )

    result = runner.invoke(
        app,
        [
            "voiceprint",
            "play",
            "欧丁",
            "--sample",
            "1",
            "--store-dir",
            str(store_dir),
            "--dry-run",
        ],
    )

    assert result.exit_code == 0
    assert "clip_002.wav" in result.output


def test_voiceprint_embed_stores_sample_embeddings(
    monkeypatch,
    tmp_path: Path,
) -> None:
    """Embedding should store one vector per captured sample."""
    project_dir = _sample_project(tmp_path)
    store_dir = tmp_path / "data" / "meeting-asr" / "voiceprints"
    _write_named_speaker_inputs(project_dir)
    monkeypatch.setattr("app.voiceprints.extract_audio_clip", _fake_extract_audio_clip)
    monkeypatch.setattr(
        "app.voiceprint_embedding.embed_audio_file", _fake_embed_audio_file
    )
    runner.invoke(
        app,
        [
            "voiceprint",
            "capture",
            str(project_dir),
            "--sample-count",
            "1",
            "--store-dir",
            str(store_dir),
        ],
    )

    result = runner.invoke(app, ["voiceprint", "embed", "--store-dir", str(store_dir)])
    list_result = runner.invoke(
        app, ["voiceprint", "list", "--store-dir", str(store_dir)]
    )
    embeddings = list_voiceprint_embeddings(
        LOCAL_CAMPP_MODEL, get_voiceprint_db_path(store_dir)
    )

    assert result.exit_code == 0
    assert "Provider: local-campp" in result.output
    assert f"Model: {LOCAL_CAMPP_MODEL}" in result.output
    assert "Embedded: 2" in result.output
    assert len(embeddings) == 2
    assert "Embedded samples: 2/2" in list_result.output


def test_voiceprint_store_skips_duplicate_clip_hash(tmp_path: Path) -> None:
    """The global library should not store the same audio bytes twice."""
    store_dir = tmp_path / "voiceprints"
    source_path = tmp_path / "meeting.mp4"
    source_path.write_bytes(b"source")
    first_clip = store_dir / "clips" / "project-a" / "speaker_0" / "clip_001.wav"
    second_clip = store_dir / "clips" / "project-b" / "speaker_0" / "clip_001.wav"
    first_clip.parent.mkdir(parents=True, exist_ok=True)
    second_clip.parent.mkdir(parents=True, exist_ok=True)
    first_clip.write_bytes(b"same-audio")
    second_clip.write_bytes(b"same-audio")
    samples = [
        _stored_voiceprint_sample(store_dir, source_path, first_clip, "project-a"),
        _stored_voiceprint_sample(store_dir, source_path, second_clip, "project-b"),
    ]

    db_path = store_voiceprint_samples(samples, get_voiceprint_db_path(store_dir))

    rows = list_voiceprint_samples("Alice", db_path)
    assert len(rows) == 1
    assert rows[0].project_id == "project-a"


def test_voiceprint_quality_flags_outliers_and_quarantine_excludes_embedding(
    tmp_path: Path,
) -> None:
    """Quality review should expose outliers and statuses should affect matching inputs."""
    store_dir = _quality_store(tmp_path)
    db_path = get_voiceprint_db_path(store_dir)

    result = runner.invoke(
        app, ["voiceprint", "quality", "Alice", "--store-dir", str(store_dir)]
    )
    json_result = runner.invoke(
        app, ["voiceprint", "quality", "Alice", "--store-dir", str(store_dir), "--json"]
    )
    payload = json.loads(json_result.output)
    critical_sample = next(
        sample
        for sample in payload["people"][0]["samples"]
        if sample["label"] == "critical"
    )
    update_voiceprint_sample_status(
        critical_sample["sample_public_id"], "quarantined", db_path
    )

    active_embeddings = list_voiceprint_embeddings(LOCAL_CAMPP_MODEL, db_path)
    all_embeddings = list_voiceprint_embeddings(
        LOCAL_CAMPP_MODEL, db_path, include_inactive=True
    )
    show_result = runner.invoke(
        app, ["voiceprint", "show", "Alice", "--store-dir", str(store_dir)]
    )

    assert result.exit_code == 0
    assert "Suspicious: 1 | Critical: 1" in result.output
    assert "Review suspicious samples:" in result.output
    assert json_result.exit_code == 0
    assert critical_sample["score"] < 0.6
    assert len(active_embeddings) == 3
    assert len(all_embeddings) == 4
    assert "status: quarantined" in show_result.output


def test_voiceprint_quality_verified_active_keeps_matching_and_keeps_quality_risk(
    tmp_path: Path,
) -> None:
    """Identity confirmation must not hide a low-quality matching sample."""
    store_dir = _quality_store(tmp_path)
    db_path = get_voiceprint_db_path(store_dir)
    payload = json.loads(
        runner.invoke(
            app,
            ["voiceprint", "quality", "Alice", "--store-dir", str(store_dir), "--json"],
        ).output
    )
    critical_sample = next(
        sample
        for sample in payload["people"][0]["samples"]
        if sample["label"] == "critical"
    )

    update_voiceprint_sample_status(
        critical_sample["sample_public_id"], "verified-active", db_path
    )

    verified_result = runner.invoke(
        app, ["voiceprint", "quality", "Alice", "--store-dir", str(store_dir), "--json"]
    )
    verified_payload = json.loads(verified_result.output)
    verified_sample = next(
        sample
        for sample in verified_payload["people"][0]["samples"]
        if sample["sample_public_id"] == critical_sample["sample_public_id"]
    )
    active_embeddings = list_voiceprint_embeddings(LOCAL_CAMPP_MODEL, db_path)

    assert verified_result.exit_code == 0
    assert verified_payload["suspicious_count"] == 1
    assert verified_payload["critical_count"] == 1
    assert verified_sample["status"] == "verified-active"
    assert verified_sample["label"] == "critical"
    assert verified_sample["reason"] == "identity confirmed; score<0.60"
    assert len(active_embeddings) == 4


def test_voiceprint_delete_sample_removes_row_and_clip(
    monkeypatch,
    tmp_path: Path,
) -> None:
    """Deleting one sample should remove its row and exact WAV file."""
    project_dir = _sample_project(tmp_path)
    store_dir = tmp_path / "data" / "meeting-asr" / "voiceprints"
    _write_named_speaker_inputs(project_dir)
    monkeypatch.setattr("app.voiceprints.extract_audio_clip", _fake_extract_audio_clip)
    runner.invoke(
        app,
        [
            "voiceprint",
            "capture",
            str(project_dir),
            "--sample-count",
            "1",
            "--store-dir",
            str(store_dir),
        ],
    )
    clip_path = (
        store_dir
        / "clips"
        / load_manifest(project_dir).project_id
        / "speaker_0"
        / "clip_002.wav"
    )

    result = runner.invoke(
        app,
        [
            "voiceprint",
            "delete-sample",
            "欧丁",
            "--sample",
            "1",
            "--store-dir",
            str(store_dir),
        ],
    )
    show_result = runner.invoke(
        app, ["voiceprint", "show", "欧丁", "--store-dir", str(store_dir)]
    )

    assert result.exit_code == 0
    assert "Deleted sample:" in result.output
    assert "clip file: deleted" in result.output
    assert not clip_path.exists()
    assert show_result.exit_code == 1


def test_voiceprint_delete_speaker_removes_all_samples(
    monkeypatch,
    tmp_path: Path,
) -> None:
    """Deleting one speaker should remove its rows and exact WAV files."""
    project_dir = _sample_project(tmp_path)
    store_dir = tmp_path / "data" / "meeting-asr" / "voiceprints"
    _write_named_speaker_inputs(project_dir)
    monkeypatch.setattr("app.voiceprints.extract_audio_clip", _fake_extract_audio_clip)
    runner.invoke(
        app,
        [
            "voiceprint",
            "capture",
            str(project_dir),
            "--sample-count",
            "1",
            "--store-dir",
            str(store_dir),
        ],
    )
    clip_path = (
        store_dir
        / "clips"
        / load_manifest(project_dir).project_id
        / "speaker_1"
        / "clip_001.wav"
    )

    speaker_id = _speaker_id_from_list(
        runner.invoke(
            app, ["voiceprint", "list", "--store-dir", str(store_dir)]
        ).output,
        "敬悦",
    )

    result = runner.invoke(
        app,
        [
            "voiceprint",
            "delete-speaker",
            speaker_id,
            "--store-dir",
            str(store_dir),
            "--yes",
        ],
    )
    list_result = runner.invoke(
        app, ["voiceprint", "list", "--store-dir", str(store_dir)]
    )

    assert result.exit_code == 0
    assert f"Deleted speaker: 敬悦 (id {speaker_id})" in result.output
    assert not clip_path.exists()
    assert "敬悦" not in list_result.output
    assert "Speakers: 1 | Samples: 1 | Embedded samples: 0/1" in list_result.output
    assert "欧丁" in list_result.output


def test_voiceprint_capture_dry_run_does_not_write_store(
    monkeypatch,
    tmp_path: Path,
) -> None:
    """Dry-run should plan global paths without writing clips or SQLite."""
    project_dir = _sample_project(tmp_path)
    store_dir = tmp_path / "data" / "meeting-asr" / "voiceprints"
    _write_named_speaker_inputs(project_dir)
    monkeypatch.setattr("app.voiceprints.extract_audio_clip", _fake_extract_audio_clip)

    result = runner.invoke(
        app,
        [
            "voiceprint",
            "capture",
            str(project_dir),
            "--sample-count",
            "1",
            "--store-dir",
            str(store_dir),
            "--dry-run",
        ],
    )

    assert result.exit_code == 0
    assert "Planned voiceprint samples: 3" in result.output
    assert "Next steps:" not in result.output
    assert "meeting-asr voiceprint embed" not in result.output
    assert not (store_dir / "voiceprints.sqlite").exists()
    assert not (store_dir / "clips").exists()


def test_voiceprint_capture_filters_repeatable_and_csv_speaker_ids(
    tmp_path: Path,
) -> None:
    """Explicit speaker ids should be merged, deduplicated, and machine-readable."""
    project_dir = _sample_project(tmp_path)
    store_dir = tmp_path / "voiceprints"
    _write_named_speaker_inputs(project_dir)

    result = runner.invoke(
        app,
        [
            "voiceprint",
            "capture",
            str(project_dir),
            "--speaker-id",
            "0",
            "--speaker-ids",
            "0,1",
            "--store-dir",
            str(store_dir),
            "--dry-run",
            "--json",
            "--no-progress",
        ],
    )

    assert result.exit_code == 0, result.output
    payload = json.loads(result.output)
    assert payload["status"] == "planned"
    assert [item["speaker_id"] for item in payload["selected_speakers"]] == [0, 1]
    assert {item["decision"] for item in payload["selected_speakers"]} == {"capture"}
    assert not (store_dir / "voiceprints.sqlite").exists()


def test_voiceprint_capture_rejects_saved_placeholder_name(
    tmp_path: Path,
) -> None:
    """A UI placeholder is still unnamed and must never enter the global store."""
    project_dir = _sample_project(tmp_path)
    store_dir = tmp_path / "voiceprints"
    _write_named_speaker_inputs(project_dir)
    (project_dir / "speakers" / "speaker_map.json").write_text(
        json.dumps({"0": "待确认发言人2", "1": "敬悦"}, ensure_ascii=False),
        encoding="utf-8",
    )

    result = runner.invoke(
        app,
        [
            "voiceprint",
            "capture",
            str(project_dir),
            "--speaker-id",
            "0",
            "--store-dir",
            str(store_dir),
            "--no-progress",
        ],
    )

    assert result.exit_code == 1
    assert "not confirmed and named" in result.output
    assert not (store_dir / "voiceprints.sqlite").exists()
    assert not (store_dir / "clips").exists()


def test_voiceprint_capture_only_needed_skips_well_sampled_person(
    monkeypatch,
    tmp_path: Path,
) -> None:
    """Only-needed must not create any new sample for a well-sampled attendee."""
    project_dir = _sample_project(tmp_path)
    store_dir = tmp_path / "voiceprints"
    db_path = get_voiceprint_db_path(store_dir)
    _write_named_speaker_inputs(project_dir)
    existing = create_voiceprint_person("敬悦", db_path)
    seed_source = tmp_path / "seed.wav"
    seed_source.write_bytes(b"seed")
    samples: list[StoredVoiceprintSample] = []
    for index in range(66):
        clip = store_dir / "clips" / "seed" / f"sample_{index:03d}.wav"
        clip.parent.mkdir(parents=True, exist_ok=True)
        clip.write_bytes(f"sample-{index}".encode())
        samples.append(
            StoredVoiceprintSample(
                speaker_name=existing.name,
                person_id=existing.speaker_id,
                project_id=f"seed-{index}",
                project_path=tmp_path / f"seed-{index}",
                project_speaker_id=1,
                source_path=seed_source,
                clip_path=clip,
                clip_rel_path=str(clip.relative_to(store_dir)),
                source_begin_time_ms=index * 1000,
                source_end_time_ms=index * 1000 + 500,
                clip_begin_time_ms=0,
                clip_end_time_ms=500,
                transcript_text=f"seed {index}",
            )
        )
    store_voiceprint_samples(samples, db_path)
    monkeypatch.setattr("app.voiceprints.extract_audio_clip", _fake_extract_audio_clip)

    result = runner.invoke(
        app,
        [
            "voiceprint",
            "capture",
            str(project_dir),
            "--only-needed",
            "--min-samples",
            "10",
            "--sample-count",
            "1",
            "--store-dir",
            str(store_dir),
            "--json",
            "--no-progress",
        ],
    )

    assert result.exit_code == 0, result.output
    payload = json.loads(result.output)
    decisions = {item["speaker_id"]: item for item in payload["selected_speakers"]}
    assert decisions[0]["decision"] == "captured"
    assert decisions[0]["reason"] == "no_samples"
    assert decisions[0]["existing_sample_count"] == 0
    assert len(decisions[0]["samples"]) == 1
    assert re.fullmatch(r"vps-[0-9a-f]{16}", decisions[0]["samples"][0]["public_id"])
    assert decisions[0]["samples"][0]["embedding_generated"] is False
    assert decisions[1]["decision"] == "skip"
    assert decisions[1]["reason"] == "enough_samples"
    assert decisions[1]["existing_sample_count"] == 66
    refreshed = get_voiceprint_person(existing.public_id, db_path)
    assert refreshed is not None
    assert refreshed.sample_count == 66
    project_id = load_manifest(project_dir).project_id
    assert not (store_dir / "clips" / project_id / "speaker_1").exists()


def test_voiceprint_capture_failure_rolls_back_one_speaker_files_and_rows(
    monkeypatch,
    tmp_path: Path,
) -> None:
    """A slicing failure must leave neither sample rows nor partial clip files."""
    project_dir = _sample_project(tmp_path)
    store_dir = tmp_path / "voiceprints"
    _write_named_speaker_inputs(project_dir)
    calls = 0

    def flaky_extract(input_path, output_path, *, start_seconds, duration_seconds):
        nonlocal calls
        calls += 1
        _fake_extract_audio_clip(
            input_path,
            output_path,
            start_seconds=start_seconds,
            duration_seconds=duration_seconds,
        )
        if calls == 2:
            raise RuntimeError("simulated slicing failure")
        return output_path

    monkeypatch.setattr("app.voiceprints.extract_audio_clip", flaky_extract)
    result = runner.invoke(
        app,
        [
            "voiceprint",
            "capture",
            str(project_dir),
            "--speaker-id",
            "0",
            "--sample-count",
            "2",
            "--store-dir",
            str(store_dir),
            "--json",
            "--no-progress",
        ],
    )

    assert result.exit_code == 1, result.output
    payload = json.loads(result.output)
    assert payload["status"] == "failed"
    assert payload["selected_speakers"][0]["decision"] == "failed"
    assert "simulated slicing failure" in payload["selected_speakers"][0]["error"]
    assert not (store_dir / "voiceprints.sqlite").exists()
    assert list(store_dir.rglob("*.wav")) == []


def test_voiceprint_capture_database_failure_restores_clip_targets(
    monkeypatch,
    tmp_path: Path,
) -> None:
    """A failed SQLite batch must roll back every WAV written for that speaker."""
    project_dir = _sample_project(tmp_path)
    store_dir = tmp_path / "voiceprints"
    _write_named_speaker_inputs(project_dir)
    project_id = load_manifest(project_dir).project_id
    existing_clip = store_dir / "clips" / project_id / "speaker_0" / "clip_001.wav"
    existing_clip.parent.mkdir(parents=True, exist_ok=True)
    existing_clip.write_bytes(b"previous valid sample")
    monkeypatch.setattr("app.voiceprints.extract_audio_clip", _fake_extract_audio_clip)
    monkeypatch.setattr(
        "app.voiceprints.store_voiceprint_samples_with_rows",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            RuntimeError("simulated database failure")
        ),
    )

    result = runner.invoke(
        app,
        [
            "voiceprint",
            "capture",
            str(project_dir),
            "--speaker-id",
            "0",
            "--sample-count",
            "1",
            "--store-dir",
            str(store_dir),
            "--json",
            "--no-progress",
        ],
    )

    assert result.exit_code == 1, result.output
    payload = json.loads(result.output)
    assert payload["selected_speakers"][0]["decision"] == "failed"
    assert "simulated database failure" in payload["selected_speakers"][0]["error"]
    assert existing_clip.read_bytes() == b"previous valid sample"
    assert list(store_dir.rglob("*.wav")) == [existing_clip]


def test_voiceprint_capture_rejects_canonical_person_name_conflict_before_writes(
    monkeypatch,
    tmp_path: Path,
) -> None:
    """A stale person link must fail before slicing or inserting any sample."""
    project_dir = _sample_project(tmp_path)
    store_dir = tmp_path / "voiceprints"
    db_path = get_voiceprint_db_path(store_dir)
    _write_named_speaker_inputs(project_dir)
    person = create_voiceprint_person("Canonical Alice", db_path)
    (project_dir / "speakers" / "speaker_person_map.json").write_text(
        json.dumps({"0": person.public_id}), encoding="utf-8"
    )
    extracted = False

    def unexpected_extract(*args, **kwargs):
        nonlocal extracted
        extracted = True
        raise AssertionError("canonical conflict must fail before slicing")

    monkeypatch.setattr("app.voiceprints.extract_audio_clip", unexpected_extract)
    result = runner.invoke(
        app,
        [
            "voiceprint",
            "capture",
            str(project_dir),
            "--speaker-id",
            "0",
            "--store-dir",
            str(store_dir),
            "--no-progress",
        ],
    )

    assert result.exit_code == 1
    assert "conflicts with canonical voiceprint person name" in result.output
    assert extracted is False
    refreshed = get_voiceprint_person(person.public_id, db_path)
    assert refreshed is not None
    assert refreshed.sample_count == 0
    assert list(store_dir.rglob("*.wav")) == []


def test_voiceprint_delete_sample_accepts_stable_public_id(
    monkeypatch,
    tmp_path: Path,
) -> None:
    """Stable vps ids should delete exact samples without a mutable list index."""
    project_dir = _sample_project(tmp_path)
    store_dir = tmp_path / "voiceprints"
    _write_named_speaker_inputs(project_dir)
    monkeypatch.setattr("app.voiceprints.extract_audio_clip", _fake_extract_audio_clip)
    capture = runner.invoke(
        app,
        [
            "voiceprint",
            "capture",
            str(project_dir),
            "--speaker-id",
            "0",
            "--sample-count",
            "1",
            "--store-dir",
            str(store_dir),
            "--json",
            "--no-progress",
        ],
    )
    payload = json.loads(capture.output)
    sample = payload["selected_speakers"][0]["samples"][0]
    clip_path = Path(sample["clip_path"])

    deleted = runner.invoke(
        app,
        [
            "voiceprint",
            "delete-sample",
            "--sample-id",
            sample["public_id"],
            "--store-dir",
            str(store_dir),
            "--keep-clip",
        ],
    )

    assert deleted.exit_code == 0, deleted.output
    assert "clip file: kept" in deleted.output
    assert clip_path.exists()
    assert list_voiceprint_samples("欧丁", get_voiceprint_db_path(store_dir)) == []


def test_voiceprint_path_prints_xdg_paths(tmp_path: Path) -> None:
    """Path command should expose store, database, and clip roots."""
    store_dir = tmp_path / "voiceprints"

    result = runner.invoke(app, ["voiceprint", "path", "--store-dir", str(store_dir)])

    assert result.exit_code == 0
    assert f"Store: {store_dir.resolve()}" in result.output
    assert f"Database: {store_dir.resolve() / 'voiceprints.sqlite'}" in result.output
    assert f"Clips: {store_dir.resolve() / 'clips'}" in result.output


def _sample_project(tmp_path: Path) -> Path:
    """Create a minimal project for voiceprint tests."""
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


def _write_named_speaker_inputs(project_dir: Path) -> None:
    """Write normalized transcript and speaker mapping fixtures."""
    sentences = {
        "full_text": "大家好。收到。",
        "detected_speakers": [0, 1],
        "sentences": [
            {
                "begin_time_ms": 0,
                "end_time_ms": 1000,
                "text": "短句。",
                "speaker_id": 0,
                "sentence_id": 1,
            },
            {
                "begin_time_ms": 2000,
                "end_time_ms": 8000,
                "text": "这是一段更适合作为样本的话。",
                "speaker_id": 0,
                "sentence_id": 2,
            },
            {
                "begin_time_ms": 9000,
                "end_time_ms": 12000,
                "text": "收到，我补充一下。",
                "speaker_id": 1,
                "sentence_id": 3,
            },
        ],
    }
    (project_dir / "asr" / "sentences.json").write_text(
        json.dumps(sentences, ensure_ascii=False),
        encoding="utf-8",
    )
    (project_dir / "speakers" / "speaker_map.json").write_text(
        json.dumps({"0": "欧丁", "1": "敬悦"}, ensure_ascii=False),
        encoding="utf-8",
    )


def _write_partially_named_speaker_inputs(project_dir: Path) -> None:
    """Write a transcript where one speaker still has the anonymous fallback name."""
    sentences = {
        "full_text": "大家好。还有一个人。",
        "detected_speakers": [0, 2],
        "sentences": [
            {
                "begin_time_ms": 0,
                "end_time_ms": 3000,
                "text": "我是欧丁。",
                "speaker_id": 0,
                "sentence_id": 1,
            },
            {
                "begin_time_ms": 4000,
                "end_time_ms": 7000,
                "text": "这个人还没有确认。",
                "speaker_id": 2,
                "sentence_id": 2,
            },
        ],
    }
    (project_dir / "asr" / "sentences.json").write_text(
        json.dumps(sentences, ensure_ascii=False),
        encoding="utf-8",
    )
    (project_dir / "speakers" / "speaker_map.json").write_text(
        json.dumps({"0": "欧丁", "2": "Speaker C"}, ensure_ascii=False),
        encoding="utf-8",
    )


def _fake_extract_audio_clip(
    input_path: Path,
    output_path: Path,
    *,
    start_seconds: float,
    duration_seconds: float,
) -> Path:
    """Write a fake WAV payload for tests."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(output_path), "wb") as writer:
        writer.setnchannels(1)
        writer.setsampwidth(2)
        writer.setframerate(16000)
        writer.writeframes((1000).to_bytes(2, "little", signed=True) * 160)
    return output_path


def _fake_embed_audio_file(path: Path, *, provider: str | None) -> list[float]:
    """Return deterministic vectors based on the speaker path."""
    return [0.0, 1.0] if "speaker_1" in str(path) else [1.0, 0.0]


def _quality_store(tmp_path: Path) -> Path:
    """Create a voiceprint store with one obvious outlier."""
    store_dir = tmp_path / "quality-voiceprints"
    source_path = tmp_path / "meeting.mp4"
    source_path.write_bytes(b"source")
    samples = [
        _stored_sample(store_dir, source_path, "Alice", index=index)
        for index in range(1, 5)
    ]
    db_path = store_voiceprint_samples(samples, get_voiceprint_db_path(store_dir))
    rows = list_voiceprint_samples_for_project("project-quality", db_path)
    vectors = ([1.0, 0.0], [0.98, 0.02], [0.99, 0.01], [0.0, 1.0])
    for row, vector in zip(rows, vectors, strict=True):
        upsert_voiceprint_embedding(
            row.sample_id, LOCAL_CAMPP_MODEL, vector, db_path
        )
    return store_dir


def _stored_sample(
    store_dir: Path,
    source_path: Path,
    speaker_name: str,
    *,
    index: int,
) -> StoredVoiceprintSample:
    """Build one stored sample fixture."""
    clip_path = (
        store_dir / "clips" / "project-quality" / "speaker_0" / f"clip_{index:03d}.wav"
    )
    clip_path.parent.mkdir(parents=True, exist_ok=True)
    clip_path.write_bytes(f"{speaker_name}-{index}".encode())
    return StoredVoiceprintSample(
        speaker_name=speaker_name,
        project_id="project-quality",
        project_path=store_dir / "project-quality",
        project_speaker_id=0,
        source_path=source_path,
        clip_path=clip_path,
        clip_rel_path=str(clip_path.relative_to(store_dir)),
        source_begin_time_ms=index * 1000,
        source_end_time_ms=index * 1000 + 500,
        clip_begin_time_ms=0,
        clip_end_time_ms=500,
        transcript_text=f"sample {index}",
    )


def _stored_voiceprint_sample(
    store_dir: Path,
    source_path: Path,
    clip_path: Path,
    project_id: str,
) -> StoredVoiceprintSample:
    """Build one stored sample fixture with a caller-provided clip."""
    return StoredVoiceprintSample(
        speaker_name="Alice",
        project_id=project_id,
        project_path=store_dir / project_id,
        project_speaker_id=0,
        source_path=source_path,
        clip_path=clip_path,
        clip_rel_path=str(clip_path.relative_to(store_dir)),
        source_begin_time_ms=1000,
        source_end_time_ms=2000,
        clip_begin_time_ms=0,
        clip_end_time_ms=1000,
        transcript_text="same sample",
    )


def _speaker_id_from_list(output: str, name: str) -> str:
    """Extract a speaker id from ``voiceprint list`` output."""
    for line in output.splitlines():
        if name not in line:
            continue
        columns = [column.strip() for column in line.split("|")]
        if (
            len(columns) >= 2
            and columns[1] == name
            and _is_voiceprint_public_id(columns[0])
        ):
            return columns[0]
        cells = [cell.strip() for cell in line.split("│") if cell.strip()]
        if len(cells) >= 2 and cells[1] == name and _is_voiceprint_public_id(cells[0]):
            return cells[0]
    raise AssertionError(f"speaker not found in list output: {name}")


def _is_voiceprint_public_id(value: str) -> bool:
    """Return whether a table cell is a voiceprint person public id."""
    return re.fullmatch(r"vpp-[0-9a-f]{16}", value) is not None


def test_speakers_apply_binds_person_by_public_id(monkeypatch, tmp_path: Path) -> None:
    """`apply --map N=@vpp-id` writes a person ref so capture binds, not duplicates."""
    project_dir = _sample_project(tmp_path)
    store_dir = tmp_path / "data" / "meeting-asr" / "voiceprints"
    _write_partially_named_speaker_inputs(project_dir)
    add_output = runner.invoke(
        app,
        ["voiceprint", "people", "add", "徐铤(彬川)", "--store-dir", str(store_dir)],
    ).output
    public_id = add_output.split("Person ID:", 1)[1].strip().splitlines()[0]

    apply_result = runner.invoke(
        app,
        [
            "project",
            "speakers",
            "apply",
            str(project_dir),
            "--map",
            f"2=@{public_id}",
            "--store-dir",
            str(store_dir),
        ],
    )
    assert apply_result.exit_code == 0, apply_result.output

    person_map = json.loads(
        (project_dir / "speakers" / "speaker_person_map.json").read_text(
            encoding="utf-8"
        )
    )
    assert person_map["2"] == public_id
    speaker_map = json.loads(
        (project_dir / "speakers" / "speaker_map.json").read_text(encoding="utf-8")
    )
    assert speaker_map["2"] == "徐铤(彬川)"

    monkeypatch.setattr("app.voiceprints.extract_audio_clip", _fake_extract_audio_clip)
    runner.invoke(
        app,
        [
            "voiceprint",
            "capture",
            str(project_dir),
            "--sample-count",
            "1",
            "--store-dir",
            str(store_dir),
        ],
    )
    # speaker 2 bound to the seeded person: its sample landed on that public id,
    # so no duplicate "徐铤(彬川)" person was created.
    captured = get_voiceprint_person(public_id, get_voiceprint_db_path(store_dir))
    assert captured is not None
    assert captured.sample_count >= 1


def test_merge_voiceprint_people_moves_samples_and_removes_source(
    tmp_path: Path,
) -> None:
    """Merge moves source samples onto the target and deletes the emptied source."""
    store_dir = tmp_path / "voiceprints"
    media = tmp_path / "meeting.mp4"
    media.write_bytes(b"media")
    db_path = get_voiceprint_db_path(store_dir)
    store_voiceprint_samples(
        [
            _stored_sample(store_dir, media, "源-彬川", index=1),
            _stored_sample(store_dir, media, "源-彬川", index=2),
            _stored_sample(store_dir, media, "徐铤(彬川)", index=3),
        ],
        db_path,
    )
    source = get_voiceprint_person("源-彬川", db_path)
    target = get_voiceprint_person("徐铤(彬川)", db_path)
    assert source is not None and target is not None

    result = merge_voiceprint_people(source.public_id, target.public_id, db_path)

    assert result.moved == 2
    assert result.duplicates == 0
    assert result.source_public_id == source.public_id
    assert get_voiceprint_person("源-彬川", db_path) is None
    kept = get_voiceprint_person(target.public_id, db_path)
    assert kept is not None
    assert kept.sample_count == 3


def test_merge_voiceprint_people_drops_duplicate_clips(tmp_path: Path) -> None:
    """A source sample whose audio already exists under the target is dropped, not moved."""
    store_dir = tmp_path / "voiceprints"
    media = tmp_path / "meeting.mp4"
    media.write_bytes(b"media")
    db_path = get_voiceprint_db_path(store_dir)
    target_sample = _stored_sample(store_dir, media, "into", index=1)
    source_sample = _stored_sample(store_dir, media, "from", index=2)
    source_sample.clip_path.write_bytes(target_sample.clip_path.read_bytes())
    store_voiceprint_samples([target_sample, source_sample], db_path)
    source = get_voiceprint_person("from", db_path)
    target = get_voiceprint_person("into", db_path)
    assert source is not None and target is not None

    result = merge_voiceprint_people(source.public_id, target.public_id, db_path)

    assert result.moved == 0
    assert result.duplicates == 1
    assert get_voiceprint_person("from", db_path) is None
    kept = get_voiceprint_person(target.public_id, db_path)
    assert kept is not None
    assert kept.sample_count == 1


def test_voiceprint_people_merge_cli_reports_summary(tmp_path: Path) -> None:
    """`voiceprint people merge --yes` merges samples and prints a summary."""
    store_dir = tmp_path / "voiceprints"
    media = tmp_path / "meeting.mp4"
    media.write_bytes(b"media")
    db_path = get_voiceprint_db_path(store_dir)
    store_voiceprint_samples(
        [
            _stored_sample(store_dir, media, "源-沛行", index=1),
            _stored_sample(store_dir, media, "黄睿(沛行)", index=2),
        ],
        db_path,
    )
    source = get_voiceprint_person("源-沛行", db_path)
    target = get_voiceprint_person("黄睿(沛行)", db_path)
    assert source is not None and target is not None

    result = runner.invoke(
        app,
        [
            "voiceprint",
            "people",
            "merge",
            source.public_id,
            target.public_id,
            "--store-dir",
            str(store_dir),
            "--yes",
        ],
    )

    assert result.exit_code == 0, result.output
    assert "Merged" in result.output
    assert get_voiceprint_person("源-沛行", db_path) is None
