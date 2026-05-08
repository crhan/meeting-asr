"""Tests for project overview rendering helpers."""

from __future__ import annotations

from app.core.project_models import ProjectManifest, ProjectSource
from app.presentation.cli.project_show import _title_source_label


def test_title_source_label_explains_legacy_custom_unknown_title() -> None:
    """A preserved legacy title should not be displayed as a bare unknown source."""
    manifest = ProjectManifest(
        schema_version=1,
        project_id="p-demo",
        title="旧自定义标题",
        title_source="unknown",
        title_model=None,
        created_at="2026-05-08T00:00:00+08:00",
        updated_at="2026-05-08T00:00:00+08:00",
        status="created",
        source=ProjectSource(
            path="source/meeting.mp4",
            filename="meeting.mp4",
            size_bytes=1,
            mtime="2026-05-08T00:00:00+08:00",
        ),
    )

    assert _title_source_label(manifest) == "manual (legacy)"


def test_title_source_label_keeps_real_unknown_when_title_is_source_name() -> None:
    """An unknown title matching the source stem should stay unknown until summary replaces it."""
    manifest = ProjectManifest(
        schema_version=1,
        project_id="p-demo",
        title="meeting",
        title_source="unknown",
        title_model=None,
        created_at="2026-05-08T00:00:00+08:00",
        updated_at="2026-05-08T00:00:00+08:00",
        status="created",
        source=ProjectSource(
            path="source/meeting.mp4",
            filename="meeting.mp4",
            size_bytes=1,
            mtime="2026-05-08T00:00:00+08:00",
        ),
    )

    assert _title_source_label(manifest) == "unknown"
