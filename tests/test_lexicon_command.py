"""Tests for lexicon hotword CLI commands."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from typer.testing import CliRunner

from app.cli import app
from app.commands import lexicon as lexicon_commands
from app.config import Settings
from app.correction_hotwords import hotword_hash
from app.lexicon_store import (
    AsrVocabularyState,
    LexiconContext,
    get_asr_vocabulary_state,
    list_asr_hotwords,
    record_lexicon_contexts,
    save_asr_vocabulary_state,
)

runner = CliRunner()


def test_lexicon_hotwords_export_writes_dashscope_table(tmp_path: Path) -> None:
    """Hotword export should use accepted correction terms."""
    db_path = tmp_path / "lexicon.sqlite"
    output = tmp_path / "hotwords.json"
    record_lexicon_contexts([_context("艾赛", "iSee")], db_path=db_path)

    result = runner.invoke(
        app,
        ["lexicon", "hotwords", "export", "--lexicon-db", str(db_path), "--output", str(output)],
    )
    payload = json.loads(output.read_text(encoding="utf-8"))

    assert result.exit_code == 0
    assert "ASR hotwords exported." in result.output
    assert payload["dashscope_vocabulary"] == [{"text": "iSee", "weight": 4}]


def test_lexicon_hotwords_list_prints_local_terms(tmp_path: Path) -> None:
    """List should show the local hotwords without writing an artifact."""
    db_path = tmp_path / "lexicon.sqlite"
    record_lexicon_contexts([_context("艾赛", "iSee")], db_path=db_path)

    result = runner.invoke(app, ["lexicon", "hotwords", "list", "--lexicon-db", str(db_path)])

    assert result.exit_code == 0
    assert "Hotwords: 1" in result.output
    assert "1. iSee weight=4 category=system" in result.output


def test_lexicon_hotwords_sync_dry_run_does_not_require_api_key(tmp_path: Path) -> None:
    """Dry-run sync should be usable for inspection without DashScope credentials."""
    db_path = tmp_path / "lexicon.sqlite"
    record_lexicon_contexts([_context("莫", "墨总")], db_path=db_path)

    result = runner.invoke(
        app,
        ["lexicon", "hotwords", "sync", "--lexicon-db", str(db_path), "--dry-run"],
    )

    assert result.exit_code == 0
    assert "ASR hotword sync dry run." in result.output
    assert "Hotwords: 1" in result.output


def test_lexicon_hotwords_status_reports_current_cache(tmp_path: Path) -> None:
    """Status should compare local hotword hash with the cached vocabulary id."""
    db_path = tmp_path / "lexicon.sqlite"
    record_lexicon_contexts([_context("艾赛", "iSee")], db_path=db_path)
    hotwords = list_asr_hotwords(db_path=db_path)
    save_asr_vocabulary_state(
        AsrVocabularyState(
            "fun-asr",
            "https://dashscope.aliyuncs.com/api/v1",
            hotword_hash(hotwords),
            "vocab-demo",
            len(hotwords),
        ),
        db_path=db_path,
    )

    result = runner.invoke(app, ["lexicon", "hotwords", "status", "--lexicon-db", str(db_path)])

    assert result.exit_code == 0
    assert "Cache: current" in result.output
    assert "Cached vocabulary ID: vocab-demo" in result.output


def test_lexicon_hotwords_clear_cache_removes_matching_state(tmp_path: Path) -> None:
    """Clear-cache should remove stale local vocabulary ids without DashScope access."""
    db_path = tmp_path / "lexicon.sqlite"
    save_asr_vocabulary_state(
        AsrVocabularyState("fun-asr", "https://dashscope.aliyuncs.com/api/v1", "hash", "vocab-demo", 1),
        db_path=db_path,
    )

    result = runner.invoke(
        app,
        [
            "lexicon",
            "hotwords",
            "clear-cache",
            "--lexicon-db",
            str(db_path),
            "--vocabulary-id",
            "vocab-demo",
        ],
    )

    assert result.exit_code == 0
    assert "ASR hotword cache cleared." in result.output
    assert get_asr_vocabulary_state(
        target_model="fun-asr",
        endpoint="https://dashscope.aliyuncs.com/api/v1",
        db_path=db_path,
    ) is None


def test_lexicon_hotwords_remote_list_uses_dashscope_wrapper(monkeypatch: pytest.MonkeyPatch) -> None:
    """Remote-list should expose DashScope vocabulary rows."""
    monkeypatch.setattr(lexicon_commands, "load_settings", lambda **kwargs: _settings())
    monkeypatch.setattr(
        lexicon_commands,
        "list_remote_asr_vocabularies",
        lambda **kwargs: [{"vocabulary_id": "vocab-demo", "status": "OK"}],
    )

    result = runner.invoke(app, ["lexicon", "hotwords", "remote-list", "--json"])
    payload = json.loads(result.output)

    assert result.exit_code == 0
    assert payload["vocabularies"] == [{"vocabulary_id": "vocab-demo", "status": "OK"}]


def test_lexicon_hotwords_remote_show_prints_vocabulary(monkeypatch: pytest.MonkeyPatch) -> None:
    """Remote-show should display one vocabulary payload."""
    monkeypatch.setattr(lexicon_commands, "load_settings", lambda **kwargs: _settings())
    monkeypatch.setattr(
        lexicon_commands,
        "query_remote_asr_vocabulary",
        lambda **kwargs: {
            "status": "OK",
            "target_model": "fun-asr",
            "vocabulary": [{"text": "iSee", "weight": 4}],
        },
    )

    result = runner.invoke(app, ["lexicon", "hotwords", "remote-show", "vocab-demo"])

    assert result.exit_code == 0
    assert "Vocabulary ID: vocab-demo" in result.output
    assert "1. iSee weight=4" in result.output


def test_lexicon_hotwords_remote_delete_can_clear_cache(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Remote-delete should require confirmation and optionally clear matching cache."""
    db_path = tmp_path / "lexicon.sqlite"
    deleted = []
    save_asr_vocabulary_state(
        AsrVocabularyState("fun-asr", "https://dashscope.aliyuncs.com/api/v1", "hash", "vocab-demo", 1),
        db_path=db_path,
    )
    monkeypatch.setattr(lexicon_commands, "load_settings", lambda **kwargs: _settings())
    monkeypatch.setattr(
        lexicon_commands,
        "delete_remote_asr_vocabulary",
        lambda **kwargs: deleted.append(kwargs["vocabulary_id"]),
    )

    result = runner.invoke(
        app,
        [
            "lexicon",
            "hotwords",
            "remote-delete",
            "vocab-demo",
            "--yes",
            "--clear-cache",
            "--lexicon-db",
            str(db_path),
        ],
    )

    assert result.exit_code == 0
    assert deleted == ["vocab-demo"]
    assert "Deleted remote ASR vocabulary: vocab-demo" in result.output
    assert get_asr_vocabulary_state(
        target_model="fun-asr",
        endpoint="https://dashscope.aliyuncs.com/api/v1",
        db_path=db_path,
    ) is None


def _context(wrong: str, corrected: str) -> LexiconContext:
    """Build one accepted lexicon context."""
    return LexiconContext(
        canonical=corrected,
        wrong_text=wrong,
        corrected_text=corrected,
        left_context="",
        right_context="",
        category="system",
        speaker_name=None,
        project_id="p-demo",
        sentence_id=1,
        source="test",
    )


def _settings() -> Settings:
    """Build minimal settings for remote command tests."""
    return Settings(dashscope_api_key="key", dashscope_base_url="https://dashscope.aliyuncs.com/api/v1")
