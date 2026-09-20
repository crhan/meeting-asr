"""Tests for the durable/recomputable split of a project directory."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from typer.testing import CliRunner

from app.cli import app
from app.project_layout import (
    CLIP_EMBEDDING_CACHE_RELATIVE_PATH,
    CORRECTION_REVIEW_RELATIVE_DIR,
    LEGACY_CLIP_EMBEDDING_CACHE_RELATIVE_PATH,
    LEGACY_CORRECTION_REVIEW_RELATIVE_DIR,
    LEGACY_PROBE_EMBEDDING_CACHE_RELATIVE_PATH,
    PROBE_EMBEDDING_CACHE_RELATIVE_PATH,
    TMP_DIR_NAME,
    clean_project_tmp,
    migrate_project_layout,
    resolve_recorded_project_path,
)
from app.speaker_clip_embeddings import read_clip_embedding_cache

runner = CliRunner()


def _write_json(path: Path, payload: object) -> Path:
    """Write a JSON file, creating parents."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
    return path


def _legacy_project(tmp_path: Path) -> Path:
    """Build a project directory in the pre-0.21 layout."""
    root = tmp_path / "project"
    _write_json(root / "project.json", {"project_id": "p-test", "title": "t"})
    _write_json(
        root / LEGACY_CLIP_EMBEDDING_CACHE_RELATIVE_PATH, {"clip-key": [1.0, 2.0]}
    )
    _write_json(
        root / LEGACY_PROBE_EMBEDDING_CACHE_RELATIVE_PATH, {"probe-key": [3.0, 4.0]}
    )
    legacy_reviews = root / LEGACY_CORRECTION_REVIEW_RELATIVE_DIR
    legacy_reviews.mkdir(parents=True, exist_ok=True)
    (legacy_reviews / "review_20260101_101010.md").write_text(
        "hand edited\n", encoding="utf-8"
    )
    (legacy_reviews / "proposal_20260101_101010.json").write_text(
        '{"category": "polish"}\n', encoding="utf-8"
    )
    clip_dir = root / TMP_DIR_NAME / "speaker_cluster" / "speaker_0"
    clip_dir.mkdir(parents=True, exist_ok=True)
    (clip_dir / "clip_001.wav").write_bytes(b"0" * 2048)
    return root


def test_durable_artifacts_are_declared_outside_tmp() -> None:
    """The point of the layout module: nothing durable may live under tmp/."""
    durable = (
        CLIP_EMBEDDING_CACHE_RELATIVE_PATH,
        PROBE_EMBEDDING_CACHE_RELATIVE_PATH,
        CORRECTION_REVIEW_RELATIVE_DIR,
    )
    for relative in durable:
        assert relative.parts[0] != TMP_DIR_NAME


def test_migration_relocates_paid_artifacts_without_losing_vectors(
    tmp_path: Path,
) -> None:
    """Pre-0.21 caches move out of tmp/ with every vector intact."""
    root = _legacy_project(tmp_path)

    migration = migrate_project_layout(root)

    assert migration.changed
    assert not (root / LEGACY_CLIP_EMBEDDING_CACHE_RELATIVE_PATH).exists()
    assert not (root / LEGACY_PROBE_EMBEDDING_CACHE_RELATIVE_PATH).exists()
    assert json.loads(
        (root / CLIP_EMBEDDING_CACHE_RELATIVE_PATH).read_text(encoding="utf-8")
    ) == {"clip-key": [1.0, 2.0]}
    assert json.loads(
        (root / PROBE_EMBEDDING_CACHE_RELATIVE_PATH).read_text(encoding="utf-8")
    ) == {"probe-key": [3.0, 4.0]}
    assert (
        root / CORRECTION_REVIEW_RELATIVE_DIR / "review_20260101_101010.md"
    ).read_text(encoding="utf-8") == "hand edited\n"
    assert not (root / LEGACY_CORRECTION_REVIEW_RELATIVE_DIR).exists()
    # Recomputable clips are none of the migration's business.
    assert (
        root / TMP_DIR_NAME / "speaker_cluster" / "speaker_0" / "clip_001.wav"
    ).exists()


def test_migration_is_idempotent_and_cheap_on_a_fresh_project(tmp_path: Path) -> None:
    """Running twice changes nothing; a project with no tmp/ is a no-op."""
    root = _legacy_project(tmp_path)
    migrate_project_layout(root)
    payload = (root / CLIP_EMBEDDING_CACHE_RELATIVE_PATH).read_text(encoding="utf-8")

    second = migrate_project_layout(root)

    assert not second.changed
    assert (root / CLIP_EMBEDDING_CACHE_RELATIVE_PATH).read_text(
        encoding="utf-8"
    ) == payload
    assert not migrate_project_layout(tmp_path / "missing").changed


def test_migration_unions_caches_when_both_copies_exist(tmp_path: Path) -> None:
    """A downgrade-then-upgrade leaves two caches; neither may lose vectors."""
    root = tmp_path / "project"
    _write_json(root / CLIP_EMBEDDING_CACHE_RELATIVE_PATH, {"new-key": [1.0]})
    _write_json(root / LEGACY_CLIP_EMBEDDING_CACHE_RELATIVE_PATH, {"old-key": [2.0]})

    migration = migrate_project_layout(root)

    assert migration.merged == (root / CLIP_EMBEDDING_CACHE_RELATIVE_PATH,)
    assert read_clip_embedding_cache(root) == {"new-key": [1.0], "old-key": [2.0]}
    assert not (root / LEGACY_CLIP_EMBEDDING_CACHE_RELATIVE_PATH).exists()


def test_migration_never_overwrites_a_colliding_review_file(tmp_path: Path) -> None:
    """A name collision keeps both files rather than clobbering either."""
    root = tmp_path / "project"
    (root / CORRECTION_REVIEW_RELATIVE_DIR).mkdir(parents=True)
    (root / CORRECTION_REVIEW_RELATIVE_DIR / "review_a.md").write_text(
        "keep me\n", encoding="utf-8"
    )
    legacy = root / LEGACY_CORRECTION_REVIEW_RELATIVE_DIR
    legacy.mkdir(parents=True)
    (legacy / "review_a.md").write_text("older\n", encoding="utf-8")

    migration = migrate_project_layout(root)

    assert migration.blocked == (legacy / "review_a.md",)
    assert (root / CORRECTION_REVIEW_RELATIVE_DIR / "review_a.md").read_text(
        encoding="utf-8"
    ) == "keep me\n"
    assert (legacy / "review_a.md").read_text(encoding="utf-8") == "older\n"


def test_recorded_legacy_paths_still_resolve_after_migration(tmp_path: Path) -> None:
    """Proposal JSON written before the move keeps pointing at a real file."""
    root = _legacy_project(tmp_path)
    migrate_project_layout(root)

    resolved = resolve_recorded_project_path(
        root, "tmp/corrections/review_20260101_101010.md"
    )

    assert (
        resolved == root / CORRECTION_REVIEW_RELATIVE_DIR / "review_20260101_101010.md"
    )
    assert resolved.read_text(encoding="utf-8") == "hand edited\n"


def test_recorded_path_prefers_the_file_that_is_actually_there(tmp_path: Path) -> None:
    """A half-migrated project must not be redirected away from its own file."""
    root = tmp_path / "project"
    legacy = root / LEGACY_CORRECTION_REVIEW_RELATIVE_DIR / "review_a.md"
    legacy.parent.mkdir(parents=True)
    legacy.write_text("legacy\n", encoding="utf-8")

    assert resolve_recorded_project_path(root, "tmp/corrections/review_a.md") == legacy


def test_clean_dry_run_measures_without_deleting(tmp_path: Path) -> None:
    """The default is a measurement, not a deletion."""
    root = _legacy_project(tmp_path)

    summary = clean_project_tmp(root, apply=False)

    assert summary.applied is False
    assert summary.freed_bytes == 2048
    assert (root / TMP_DIR_NAME / "speaker_cluster").exists()
    assert [path.name for path in summary.removed] == ["speaker_cluster"]


def test_clean_removes_only_recomputable_data(tmp_path: Path) -> None:
    """Applying clean wipes tmp/ and leaves every paid artifact readable."""
    root = _legacy_project(tmp_path)

    summary = clean_project_tmp(root, apply=True)

    assert summary.applied is True
    assert not (root / TMP_DIR_NAME).exists()
    assert read_clip_embedding_cache(root) == {"clip-key": [1.0, 2.0]}
    assert json.loads(
        (root / PROBE_EMBEDDING_CACHE_RELATIVE_PATH).read_text(encoding="utf-8")
    ) == {"probe-key": [3.0, 4.0]}
    assert (
        root / CORRECTION_REVIEW_RELATIVE_DIR / "review_20260101_101010.md"
    ).exists()


def test_clean_refuses_to_delete_unmigratable_durable_data(tmp_path: Path) -> None:
    """Whatever migration could not relocate is kept, never cleaned away."""
    root = tmp_path / "project"
    (root / CORRECTION_REVIEW_RELATIVE_DIR).mkdir(parents=True)
    (root / CORRECTION_REVIEW_RELATIVE_DIR / "review_a.md").write_text(
        "keep me\n", encoding="utf-8"
    )
    legacy = root / LEGACY_CORRECTION_REVIEW_RELATIVE_DIR
    legacy.mkdir(parents=True)
    (legacy / "review_a.md").write_text("older\n", encoding="utf-8")
    (root / TMP_DIR_NAME / "web_clips").mkdir(parents=True)
    (root / TMP_DIR_NAME / "web_clips" / "clip.wav").write_bytes(b"0" * 16)

    summary = clean_project_tmp(root, apply=True)

    assert summary.kept == (legacy,)
    assert (legacy / "review_a.md").read_text(encoding="utf-8") == "older\n"
    assert not (root / TMP_DIR_NAME / "web_clips").exists()


@pytest.mark.parametrize("flag", ["--json", None])
def test_project_clean_command_is_a_dry_run_by_default(
    tmp_path: Path, flag: str | None
) -> None:
    """``project clean`` must not delete anything until --apply is passed."""
    root = _legacy_project(tmp_path)
    argv = ["project", "clean", str(root)] + ([flag] if flag else [])

    result = runner.invoke(app, argv)

    assert result.exit_code == 0
    assert (root / TMP_DIR_NAME / "speaker_cluster").exists()
    if flag == "--json":
        payload = json.loads(result.output)
        assert payload["applied"] is False
        assert payload["freed_bytes"] == 2048
        assert payload["project_count"] == 1
    else:
        assert "Dry run" in result.output
        assert "would remove" in result.output


def test_project_clean_command_applies_with_confirmation(tmp_path: Path) -> None:
    """--apply deletes tmp/ once confirmed, and keeps the paid caches."""
    root = _legacy_project(tmp_path)

    result = runner.invoke(app, ["project", "clean", str(root), "--apply"], input="y\n")

    assert result.exit_code == 0
    assert not (root / TMP_DIR_NAME).exists()
    assert read_clip_embedding_cache(root) == {"clip-key": [1.0, 2.0]}


def test_project_clean_command_respects_a_declined_confirmation(tmp_path: Path) -> None:
    """Answering no leaves the project untouched."""
    root = _legacy_project(tmp_path)

    result = runner.invoke(app, ["project", "clean", str(root), "--apply"], input="n\n")

    assert result.exit_code == 0
    assert "cancelled" in result.output
    assert (root / TMP_DIR_NAME / "speaker_cluster").exists()
