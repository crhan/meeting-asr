"""Tests for lexicon hotword CLI commands."""

from __future__ import annotations

import json
from pathlib import Path

from typer.testing import CliRunner

from app.cli import app
from app.lexicon_store import LexiconContext, record_lexicon_contexts

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
