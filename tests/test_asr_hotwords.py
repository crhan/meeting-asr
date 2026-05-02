"""Tests for ASR hotword generation and DashScope synchronization."""

from __future__ import annotations

import json
from pathlib import Path

from app.asr_hotwords import resolve_asr_hotwords, sync_asr_hotwords
from app.config import Settings
from app.lexicon_store import LexiconContext, record_lexicon_contexts


class FakeVocabularyClient:
    """Small fake for DashScope VocabularyService."""

    def __init__(self) -> None:
        """Initialize captured calls."""
        self.created = []
        self.updated = []

    def create_vocabulary(self, target_model: str, prefix: str, vocabulary: list[dict]) -> str:
        """Capture vocabulary creation."""
        self.created.append((target_model, prefix, vocabulary))
        return "vocab-demo"

    def update_vocabulary(self, vocabulary_id: str, vocabulary: list[dict]) -> None:
        """Capture vocabulary update."""
        self.updated.append((vocabulary_id, vocabulary))


def test_sync_asr_hotwords_creates_and_caches_vocabulary(tmp_path: Path) -> None:
    """Accepted correction terms should sync once and reuse cached vocabulary ids."""
    db_path = tmp_path / "lexicon.sqlite"
    record_lexicon_contexts([_context("艾赛", "iSee")], db_path=db_path)
    client = FakeVocabularyClient()

    first = sync_asr_hotwords(settings=_settings(), target_model="fun-asr", db_path=db_path, client=client)
    second = sync_asr_hotwords(settings=_settings(), target_model="fun-asr", db_path=db_path, client=client)

    assert first.changed is True
    assert first.vocabulary_id == "vocab-demo"
    assert first.hotword_count == 1
    assert client.created[0][2] == [{"text": "iSee", "weight": 4}]
    assert second.changed is False
    assert second.vocabulary_id == "vocab-demo"
    assert len(client.created) == 1


def test_sync_asr_hotwords_dry_run_writes_artifact(tmp_path: Path) -> None:
    """Dry run should export hotwords without touching DashScope."""
    db_path = tmp_path / "lexicon.sqlite"
    output = tmp_path / "hotwords.json"
    record_lexicon_contexts([_context("莫", "墨总")], db_path=db_path)

    summary = sync_asr_hotwords(
        settings=_settings(),
        target_model="fun-asr",
        db_path=db_path,
        output=output,
        dry_run=True,
    )
    payload = json.loads(output.read_text(encoding="utf-8"))

    assert summary.dry_run is True
    assert summary.vocabulary_id is None
    assert payload["dashscope_vocabulary"] == [{"text": "墨总", "weight": 4}]


def test_resolve_asr_hotwords_prefers_configured_vocabulary_id() -> None:
    """Auto mode should respect an explicitly configured vocabulary id."""
    settings = Settings(
        dashscope_api_key="key",
        dashscope_base_url="https://dashscope.example.com",
        dashscope_asr_vocabulary_id="vocab-config",
    )

    result = resolve_asr_hotwords(mode="auto", settings=settings, target_model="fun-asr")

    assert result.vocabulary_id == "vocab-config"
    assert result.source == "config"


def test_resolve_asr_hotwords_auto_degrades_on_sync_error(monkeypatch) -> None:
    """Auto mode must not break transcription when optional hotword sync fails."""
    def fail_sync(**kwargs):
        raise RuntimeError("remote hotword unavailable")

    monkeypatch.setattr("app.asr_hotwords.sync_asr_hotwords", fail_sync)

    result = resolve_asr_hotwords(
        mode="auto",
        settings=_settings(),
        target_model="fun-asr",
    )

    assert result.vocabulary_id is None
    assert result.source == "auto-error"
    assert result.error


def _context(wrong: str, corrected: str) -> LexiconContext:
    """Build one accepted lexicon context."""
    return LexiconContext(
        canonical=corrected,
        wrong_text=wrong,
        corrected_text=corrected,
        left_context="",
        right_context="系统",
        category="system",
        speaker_name="敬悦",
        project_id="p-demo",
        sentence_id=1,
        source="test",
    )


def _settings() -> Settings:
    """Build minimal settings for hotword sync tests."""
    return Settings(dashscope_api_key="key", dashscope_base_url="https://dashscope.example.com")
