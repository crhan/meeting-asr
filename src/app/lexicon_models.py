"""Dataclasses for the cross-project correction lexicon."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class LexiconContext:
    """One accepted vocabulary correction context."""

    canonical: str
    wrong_text: str
    corrected_text: str
    left_context: str
    right_context: str
    category: str
    speaker_name: str | None
    project_id: str
    sentence_id: int | None
    source: str


@dataclass(frozen=True, slots=True)
class LexiconAlias:
    """One alias attached to a canonical lexicon term."""

    alias: str
    alias_type: str
    created_at: str
    updated_at: str
    # ``None`` = deterministic blanket replacement; non-empty = polish resolves
    # this alias by sentence context using the stored guidance (excluded from
    # blanket rules). Read-side only; the value is authored via ``disambiguate``.
    disambiguation: str | None = None


@dataclass(frozen=True, slots=True)
class LexiconContextRow:
    """One stored correction context for a canonical term."""

    wrong_text: str
    corrected_text: str
    left_context: str
    right_context: str
    speaker_name: str | None
    project_id: str
    sentence_id: int | None
    source: str
    created_at: str


@dataclass(frozen=True, slots=True)
class LexiconCorrectionRule:
    """One active local replacement rule derived from the lexicon."""

    wrong_text: str
    corrected_text: str
    left_context: str
    right_context: str
    canonical: str
    category: str
    source: str


@dataclass(frozen=True, slots=True)
class LexiconDisambiguation:
    """One ambiguous alias whose correct form depends on sentence context.

    ``guidance`` is user-authored business knowledge (e.g. "AC 指 Acme 平台时
    改成 Acme；指个人贡献者角色时保持原样") and is fed to the polish LLM so it
    decides per occurrence instead of blanket-replacing. Such an alias is
    excluded from deterministic local correction.
    """

    alias: str
    canonical: str
    category: str
    guidance: str


@dataclass(frozen=True, slots=True)
class LexiconTerm:
    """One local lexicon term row."""

    term_id: int
    public_id: str
    canonical: str
    category: str
    description: str
    status: str
    alias_count: int
    context_count: int
    created_at: str
    updated_at: str
    ambiguous_alias_count: int = 0


@dataclass(frozen=True, slots=True)
class LexiconTermDetail:
    """Full local lexicon term detail."""

    term: LexiconTerm
    aliases: tuple[LexiconAlias, ...]
    contexts: tuple[LexiconContextRow, ...]


@dataclass(frozen=True, slots=True)
class LexiconStats:
    """Aggregate local lexicon statistics."""

    active_terms: int
    inactive_terms: int
    aliases: int
    contexts: int
    hotwords: int
    cached_vocabularies: int


@dataclass(frozen=True, slots=True)
class AsrVocabularyState:
    """Cached DashScope vocabulary state for one ASR model."""

    target_model: str
    endpoint: str
    vocabulary_hash: str
    vocabulary_id: str
    hotword_count: int
    updated_at: str | None = None
