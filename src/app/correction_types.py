"""Shared dataclasses for vocabulary correction workflows."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from app.models import TranscriptResult


@dataclass(frozen=True, slots=True)
class CorrectionEditOptions:
    """Options for editor-driven correction."""

    editor: str | None = None
    review_file: Path | None = None
    open_editor: bool = True
    open_proposal: bool = True
    category: str = "unknown"
    lexicon_db: Path | None = None
    from_original: bool = False
    use_ai: bool = True
    model: str | None = None


@dataclass(frozen=True, slots=True)
class CorrectionSource:
    """Transcript source used by one correction proposal."""

    result: TranscriptResult
    path: Path
    from_original: bool


@dataclass(frozen=True, slots=True)
class CorrectionReplacement:
    """One inferred lexical replacement."""

    wrong_text: str
    corrected_text: str
    left_context: str
    right_context: str


@dataclass(frozen=True, slots=True)
class CorrectionChange:
    """One edited transcript sentence."""

    sentence_id: int | None
    speaker_id: int | None
    speaker_name: str
    begin_time_ms: int
    end_time_ms: int
    original_text: str
    corrected_text: str
    replacements: list[CorrectionReplacement]


@dataclass(frozen=True, slots=True)
class CorrectionUnderstanding:
    """One correction rule inferred from user-edited samples."""

    wrong_text: str
    corrected_text: str
    sample_count: int
    proposed_count: int
    left_context: str
    right_context: str


@dataclass(frozen=True, slots=True)
class CorrectionProposal:
    """Full-document correction proposal awaiting user acceptance."""

    project_id: str
    category: str
    review_path: Path
    proposal_path: Path
    diff_path: Path
    json_path: Path
    source_path: Path
    sample_changes: list[CorrectionChange]
    proposed_changes: list[CorrectionChange]
    understanding: list[CorrectionUnderstanding]
    model: str
    model_error: str | None
    from_original: bool


@dataclass(frozen=True, slots=True)
class CorrectionEditSummary:
    """Result of an editor-driven correction run."""

    review_path: Path
    proposal_path: Path | None
    proposal_diff_path: Path | None
    proposal_json_path: Path | None
    change_count: int
    sample_change_count: int
    proposed_change_count: int
    learned_count: int
    accepted: bool
    model: str | None
    model_error: str | None
    understanding: list[CorrectionUnderstanding]
    corrected_sentences_path: Path | None
    corrected_transcript_path: Path | None
    corrected_named_transcript_path: Path | None
    corrected_srt_path: Path | None
    applied_path: Path | None
    lexicon_db: Path | None
