"""Editor-driven transcript vocabulary correction workflow."""

from __future__ import annotations

import difflib
import hashlib
import os
import re
import time
from collections.abc import Callable
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, TimeoutError as FutureTimeoutError, wait
from dataclasses import asdict, replace
from datetime import datetime
from pathlib import Path

from app.config import MAX_DASHSCOPE_CORRECTION_CONCURRENCY, Settings, load_settings
from app.correction_editor import open_editor
from app.correction_hotwords import hotwords_from_understanding, write_hotword_artifact
from app.correction_llm import (
    DASHSCOPE_TEXT_REQUEST_TIMEOUT_SECONDS,
    LlmCorrectionCandidate,
    LlmCorrectionSample,
    LlmPolishItem,
    propose_transcript_polish,
    propose_transcript_polish_strict,
    propose_vocabulary_corrections,
)
from app.core.progress import CliProgressReporter, emit_progress
from app.correction_proposals import load_correction_proposal, write_correction_proposal_files
from app.correction_types import (
    CorrectionChange,
    CorrectionEditOptions,
    CorrectionEditSummary,
    CorrectionProposal,
    CorrectionReplacement,
    CorrectionSource,
    CorrectionUnderstanding,
)
from app.correction_understanding import (
    join_model_errors,
    matching_correction_replacements,
    refine_sample_replacements,
)
from app.core.project_models import ProjectManifest, ProjectPaths
from app.lexicon_store import (
    LexiconContext,
    default_lexicon_db_path,
    list_lexicon_correction_rules,
    record_lexicon_contexts,
)
from app.models import SentenceSegment, TranscriptResult
from app.postprocess import detect_speaker_ids, render_plain_text, render_speaker_text, speaker_id_to_label
from app.speaker_labeling import load_transcript_result, render_named_speaker_text, render_named_srt
from app.srt_utils import build_srt
from app.utils import safe_write_json, safe_write_text

ANCHOR_RE = re.compile(r"^<!-- meeting-asr: (?P<fields>.+) -->$")
TIMESTAMP_LINE_RE = re.compile(r"^\[[^\]]+\]\s*(?P<label>.*?):\s*(?P<text>.*)$")
WORD_RE = re.compile(r"[\w\u4e00-\u9fff]+", re.UNICODE)
ASCII_TERM_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9_+.#-]*")
REVIEW_DIR = "corrections"
# Drives how many sentences ship to DashScope in one polish batch.
# See https://github.com/crhan/meeting-asr/issues/4 for the planned
# throughput retune; do not change this without coordinating with the
# request timeout in correction_llm.py.
POLISH_LLM_BATCH_SIZE = 30
# Vocabulary correction only sends sentences containing a learned wrong
# term, typically far fewer than the polish path. Kept as a separate
# constant so the two paths can move independently; the previous code
# reused POLISH_LLM_BATCH_SIZE for vocab correction, which was a naming
# accident, not a deliberate share.
VOCAB_CORRECTION_BATCH_SIZE = 30


def prepare_editor_correction(
    *,
    paths: ProjectPaths,
    manifest: ProjectManifest,
    speaker_mapping: dict[int, str],
    options: CorrectionEditOptions,
) -> CorrectionEditSummary:
    """
    Prepare a full-document correction proposal from user-edited samples.

    Args:
        paths: Project paths.
        manifest: Project manifest.
        speaker_mapping: Speaker id to display name mapping.
        options: Correction options.

    Returns:
        Correction edit summary.
    """
    source = _load_correction_source(paths, from_original=options.from_original)
    review_path = options.review_file or _write_review_file(paths, manifest, source.result, speaker_mapping)
    if options.open_editor and options.review_file is None:
        open_editor(review_path, options.editor)
    edited = review_path.read_text(encoding="utf-8")
    changes = _extract_changes(edited, source.result, speaker_mapping)
    lexicon_db = options.lexicon_db or default_lexicon_db_path()
    if not changes:
        return _empty_summary(review_path, lexicon_db)
    proposal = _build_proposal(paths, manifest, source, review_path, changes, speaker_mapping, options)
    if options.open_proposal:
        open_editor(proposal.proposal_path, options.editor)
    return _proposal_summary(proposal, lexicon_db)


def prepare_inline_correction(
    *,
    paths: ProjectPaths,
    manifest: ProjectManifest,
    speaker_mapping: dict[int, str],
    correction_edit: object,
    options: CorrectionEditOptions,
) -> CorrectionEditSummary:
    """
    Prepare a correction proposal from one TUI-edited sentence.

    Args:
        paths: Project paths.
        manifest: Project manifest.
        speaker_mapping: Speaker id to display name mapping.
        correction_edit: Object with sentence identity and corrected text fields.
        options: Correction options.

    Returns:
        Correction edit summary.
    """
    return prepare_inline_corrections(
        paths=paths,
        manifest=manifest,
        speaker_mapping=speaker_mapping,
        correction_edits=[correction_edit],
        options=options,
    )


def prepare_inline_corrections(
    *,
    paths: ProjectPaths,
    manifest: ProjectManifest,
    speaker_mapping: dict[int, str],
    correction_edits: list[object],
    options: CorrectionEditOptions,
) -> CorrectionEditSummary:
    """
    Prepare a correction proposal from one or more TUI-edited sentences.

    Args:
        paths: Project paths.
        manifest: Project manifest.
        speaker_mapping: Speaker id to display name mapping.
        correction_edits: TUI sentence edits.
        options: Correction options.

    Returns:
        Correction edit summary.
    """
    source = _load_correction_source(paths, from_original=options.from_original)
    lexicon_db = options.lexicon_db or default_lexicon_db_path()
    sample_changes = _inline_sample_changes(source.result, correction_edits, speaker_mapping)
    changed_samples = [
        change for change in sample_changes
        if change.corrected_text != change.original_text
    ]
    review_path = _write_inline_review_file(paths, manifest, sample_changes)
    if not changed_samples:
        return _empty_summary(review_path, lexicon_db)
    proposal = _build_proposal(paths, manifest, source, review_path, changed_samples, speaker_mapping, options)
    if options.open_proposal:
        open_editor(proposal.proposal_path, options.editor)
    return _proposal_summary(proposal, lexicon_db)


def prepare_transcript_polish(
    *,
    paths: ProjectPaths,
    manifest: ProjectManifest,
    speaker_mapping: dict[int, str],
    options: CorrectionEditOptions,
    progress: CliProgressReporter | None = None,
) -> CorrectionEditSummary:
    """
    Prepare a full-transcript readability polish proposal.

    Args:
        paths: Project paths.
        manifest: Project manifest.
        speaker_mapping: Speaker id to display name mapping.
        options: Correction options.

    Returns:
        Pending polish proposal summary, or a no-change summary.
    """
    source = _load_correction_source(paths, from_original=options.from_original)
    review_path = options.review_file or _write_polish_review_file(paths, manifest, source.result, speaker_mapping)
    lexicon_db = options.lexicon_db or default_lexicon_db_path()
    proposed_changes, model, model_error = _propose_polish_changes(
        paths,
        manifest,
        source.result,
        speaker_mapping,
        options,
        progress,
    )
    if not proposed_changes:
        return _empty_summary(review_path, lexicon_db, model=model, model_error=model_error)
    proposal = _build_polish_proposal(
        paths=paths,
        manifest=manifest,
        source=source,
        review_path=review_path,
        proposed_changes=proposed_changes,
        speaker_mapping=speaker_mapping,
        options=replace(options, category=options.category or "polish"),
        model=model,
        model_error=model_error,
    )
    return _proposal_summary(proposal, lexicon_db)


def apply_lexicon_corrections(
    *,
    paths: ProjectPaths,
    manifest: ProjectManifest,
    speaker_mapping: dict[int, str],
    options: CorrectionEditOptions,
    progress: CliProgressReporter | None = None,
) -> CorrectionEditSummary:
    """
    Apply active local lexicon correction rules without model inference.

    Args:
        paths: Project paths.
        manifest: Loaded project manifest.
        speaker_mapping: Speaker id to display name mapping.
        options: Correction options.
        progress: Optional progress reporter.

    Returns:
        Accepted correction summary when changes were written, otherwise a no-change summary.
    """
    lexicon_db = options.lexicon_db or default_lexicon_db_path()
    review_path = _lexicon_review_path(paths)
    rules = _lexicon_replacement_rules(lexicon_db)
    if not rules:
        return _empty_summary(review_path, lexicon_db, model="local-lexicon")
    source = _load_correction_source(paths, from_original=options.from_original)
    changes = _local_rule_changes(source.result, [], rules, speaker_mapping)
    emit_progress(progress, f"Applying local vocabulary corrections | changes {len(changes)}")
    if not changes:
        return _empty_summary(review_path, lexicon_db, model="local-lexicon")
    review_path = _write_lexicon_review_file(paths, manifest, lexicon_db)
    corrected = _apply_changes(source.result, changes)
    outputs = _write_corrected_outputs(paths, corrected, speaker_mapping, changes)
    understanding = _active_understanding(rules, changes)
    hotwords_path = _write_accept_hotwords(paths, options.category or "lexicon", understanding)
    return CorrectionEditSummary(
        review_path=review_path,
        proposal_path=None,
        proposal_diff_path=None,
        proposal_json_path=None,
        change_count=len(changes),
        sample_change_count=0,
        proposed_change_count=len(changes),
        learned_count=0,
        accepted=True,
        model="local-lexicon",
        model_error=None,
        understanding=understanding,
        corrected_sentences_path=outputs["sentences"],
        corrected_transcript_path=outputs["transcript"],
        corrected_named_transcript_path=outputs["named_transcript"],
        corrected_srt_path=outputs["srt"],
        hotwords_path=hotwords_path,
        applied_path=outputs["applied"],
        lexicon_db=lexicon_db,
    )


def accept_correction_proposal(
    *,
    paths: ProjectPaths,
    manifest: ProjectManifest,
    speaker_mapping: dict[int, str],
    proposal_path: Path | None,
    lexicon_db: Path | None = None,
    selected_change_indices: tuple[int, ...] | None = None,
) -> CorrectionEditSummary:
    """
    Accept a generated correction proposal and write final artifacts.

    Args:
        paths: Project paths.
        manifest: Project manifest.
        speaker_mapping: Speaker id to display name mapping.
        proposal_path: Proposal JSON path, or None for the latest proposal.
        lexicon_db: Optional lexicon database override.
        selected_change_indices: Optional zero-based proposed change indices to accept.

    Returns:
        Correction edit summary.
    """
    proposal = load_correction_proposal(paths, proposal_path)
    if proposal.project_id != manifest.project_id:
        raise RuntimeError(f"Correction proposal belongs to another project: {proposal.project_id}")
    source = _load_source_path(paths, proposal.source_path)
    accepted_changes = _selected_changes(proposal.proposed_changes, selected_change_indices)
    understanding = _selected_understanding(proposal.understanding, accepted_changes)
    corrected = _apply_changes(source.result, accepted_changes)
    outputs = _write_corrected_outputs(paths, corrected, speaker_mapping, accepted_changes)
    hotwords_path = _write_accept_hotwords(paths, proposal.category, understanding)
    database_path = lexicon_db or default_lexicon_db_path()
    contexts = _lexicon_contexts(accepted_changes, manifest.project_id, proposal.category, proposal.review_path)
    learned_count = record_lexicon_contexts(contexts, db_path=database_path)
    return CorrectionEditSummary(
        review_path=proposal.review_path,
        proposal_path=proposal.proposal_path,
        proposal_diff_path=proposal.diff_path,
        proposal_json_path=proposal.json_path,
        change_count=len(accepted_changes),
        sample_change_count=len(proposal.sample_changes),
        proposed_change_count=len(proposal.proposed_changes),
        learned_count=learned_count,
        accepted=True,
        model=proposal.model,
        model_error=proposal.model_error,
        understanding=understanding,
        corrected_sentences_path=outputs["sentences"],
        corrected_transcript_path=outputs["transcript"],
        corrected_named_transcript_path=outputs["named_transcript"],
        corrected_srt_path=outputs["srt"],
        hotwords_path=hotwords_path,
        applied_path=outputs["applied"],
        lexicon_db=database_path,
    )


def _load_correction_source(paths: ProjectPaths, *, from_original: bool) -> CorrectionSource:
    """Load the preferred correction source transcript."""
    corrected_path = paths.asr_dir / "sentences_corrected.json"
    if corrected_path.exists() and not from_original:
        return CorrectionSource(load_transcript_result(corrected_path), corrected_path, False)
    source_path = paths.asr_dir / "sentences.json"
    return CorrectionSource(load_transcript_result(source_path), source_path, from_original)


def _load_source_path(paths: ProjectPaths, source_path: Path) -> CorrectionSource:
    """Load a proposal source path, accepting project-relative paths."""
    resolved = source_path if source_path.is_absolute() else paths.root / source_path
    result = load_transcript_result(resolved)
    return CorrectionSource(result, resolved, resolved.name == "sentences.json")


def _selected_changes(
    changes: list[CorrectionChange],
    selected_indices: tuple[int, ...] | None,
) -> list[CorrectionChange]:
    """Return accepted proposal changes by index."""
    if selected_indices is None:
        return changes
    selected = set(selected_indices)
    return [change for index, change in enumerate(changes) if index in selected]


def _selected_understanding(
    understanding: list[CorrectionUnderstanding],
    changes: list[CorrectionChange],
) -> list[CorrectionUnderstanding]:
    """Keep only understanding rows represented by accepted changes."""
    counts = _replacement_counts(changes)
    selected = []
    for item in understanding:
        key = (item.wrong_text, item.corrected_text)
        count = counts.get(key)
        if count is None:
            continue
        selected.append(replace(item, proposed_count=count))
    return selected


def _replacement_counts(changes: list[CorrectionChange]) -> dict[tuple[str, str], int]:
    """Count accepted replacement pairs."""
    counts: dict[tuple[str, str], int] = {}
    for change in changes:
        for replacement in change.replacements:
            key = (replacement.wrong_text, replacement.corrected_text)
            counts[key] = counts.get(key, 0) + 1
    return counts


def _write_accept_hotwords(
    paths: ProjectPaths,
    category: str,
    understanding: list[CorrectionUnderstanding],
) -> Path:
    """Write ASR hotwords produced by the accepted correction proposal."""
    hotwords = hotwords_from_understanding(understanding, category=category)
    return write_hotword_artifact(paths.root / "corrections" / "asr_hotwords.json", hotwords)


def _empty_summary(
    review_path: Path,
    lexicon_db: Path,
    *,
    model: str | None = None,
    model_error: str | None = None,
) -> CorrectionEditSummary:
    """Build a no-change correction summary."""
    return CorrectionEditSummary(
        review_path=review_path,
        proposal_path=None,
        proposal_diff_path=None,
        proposal_json_path=None,
        change_count=0,
        sample_change_count=0,
        proposed_change_count=0,
        learned_count=0,
        accepted=False,
        model=model,
        model_error=model_error,
        understanding=[],
        corrected_sentences_path=None,
        corrected_transcript_path=None,
        corrected_named_transcript_path=None,
        corrected_srt_path=None,
        hotwords_path=None,
        applied_path=None,
        lexicon_db=lexicon_db,
    )


def _proposal_summary(proposal: CorrectionProposal, lexicon_db: Path) -> CorrectionEditSummary:
    """Build a pending-proposal correction summary."""
    return CorrectionEditSummary(
        review_path=proposal.review_path,
        proposal_path=proposal.proposal_path,
        proposal_diff_path=proposal.diff_path,
        proposal_json_path=proposal.json_path,
        change_count=0,
        sample_change_count=len(proposal.sample_changes),
        proposed_change_count=len(proposal.proposed_changes),
        learned_count=0,
        accepted=False,
        model=proposal.model,
        model_error=proposal.model_error,
        understanding=proposal.understanding,
        corrected_sentences_path=None,
        corrected_transcript_path=None,
        corrected_named_transcript_path=None,
        corrected_srt_path=None,
        hotwords_path=None,
        applied_path=None,
        lexicon_db=lexicon_db,
    )


def _write_review_file(
    paths: ProjectPaths,
    manifest: ProjectManifest,
    result: TranscriptResult,
    speaker_mapping: dict[int, str],
) -> Path:
    """Create the editor review file with stable sentence anchors."""
    review_dir = paths.root / "tmp" / REVIEW_DIR
    review_dir.mkdir(parents=True, exist_ok=True)
    review_path = review_dir / f"review_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md"
    safe_write_text(review_path, _render_review_file(manifest, result, speaker_mapping))
    return review_path


def _build_proposal(
    paths: ProjectPaths,
    manifest: ProjectManifest,
    source: CorrectionSource,
    review_path: Path,
    sample_changes: list[CorrectionChange],
    speaker_mapping: dict[int, str],
    options: CorrectionEditOptions,
) -> CorrectionProposal:
    """Build and persist a full-document correction proposal."""
    sample_changes, replacement_model_error = refine_sample_replacements(sample_changes, options)
    rules = _unique_replacements(sample_changes)
    proposed_changes, model, model_error = _propose_full_document_changes(
        source.result,
        sample_changes,
        rules,
        speaker_mapping,
        options,
    )
    model_error = join_model_errors(replacement_model_error, model_error)
    understanding = _build_understanding(rules, sample_changes, proposed_changes)
    proposed = _apply_changes(source.result, proposed_changes)
    return write_correction_proposal_files(
        paths=paths,
        manifest=manifest,
        source=source,
        proposed=proposed,
        review_path=review_path,
        sample_changes=sample_changes,
        proposed_changes=proposed_changes,
        understanding=understanding,
        speaker_mapping=speaker_mapping,
        options=options,
        model=model,
        model_error=model_error,
    )


def _build_polish_proposal(
    *,
    paths: ProjectPaths,
    manifest: ProjectManifest,
    source: CorrectionSource,
    review_path: Path,
    proposed_changes: list[CorrectionChange],
    speaker_mapping: dict[int, str],
    options: CorrectionEditOptions,
    model: str,
    model_error: str | None,
) -> CorrectionProposal:
    """Build and persist a transcript polish proposal."""
    proposed = _apply_changes(source.result, proposed_changes)
    return write_correction_proposal_files(
        paths=paths,
        manifest=manifest,
        source=source,
        proposed=proposed,
        review_path=review_path,
        sample_changes=[],
        proposed_changes=proposed_changes,
        understanding=[],
        speaker_mapping=speaker_mapping,
        options=options,
        model=model,
        model_error=model_error,
    )


def _inline_sample_change(
    result: TranscriptResult,
    correction_edit: object,
    speaker_mapping: dict[int, str],
) -> CorrectionChange:
    """Build one correction change from TUI edit metadata."""
    sentence = _find_inline_sentence(result, correction_edit)
    corrected_text = str(getattr(correction_edit, "corrected_text")).strip()
    if not corrected_text:
        raise ValueError("Corrected text must not be empty.")
    return _change_from_sentence(sentence, corrected_text, speaker_mapping)


def _inline_sample_changes(
    result: TranscriptResult,
    correction_edits: list[object],
    speaker_mapping: dict[int, str],
) -> list[CorrectionChange]:
    """Build correction changes from all TUI edit metadata."""
    if not correction_edits:
        raise ValueError("At least one inline correction edit is required.")
    return [
        _inline_sample_change(result, correction_edit, speaker_mapping)
        for correction_edit in correction_edits
    ]


def _find_inline_sentence(result: TranscriptResult, correction_edit: object) -> SentenceSegment:
    """Find the source sentence edited by the TUI."""
    sentence_id = _optional_int_from_any(getattr(correction_edit, "sentence_id"))
    speaker_id = _optional_int_from_any(getattr(correction_edit, "speaker_id"))
    begin = int(getattr(correction_edit, "begin_time_ms"))
    end = int(getattr(correction_edit, "end_time_ms"))
    for sentence in result.sentences:
        if (
            sentence.sentence_id == sentence_id
            and sentence.speaker_id == speaker_id
            and sentence.begin_time_ms == begin
            and sentence.end_time_ms == end
        ):
            return sentence
    raise RuntimeError(f"Inline correction sentence was not found: sentence_id={sentence_id} begin={begin} end={end}")


def _write_inline_review_file(
    paths: ProjectPaths,
    manifest: ProjectManifest,
    sample_changes: list[CorrectionChange],
) -> Path:
    """Write a compact review file for TUI correction samples."""
    review_dir = paths.root / "tmp" / REVIEW_DIR
    review_dir.mkdir(parents=True, exist_ok=True)
    review_path = review_dir / f"review_tui_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md"
    lines = [
        "# Meeting-ASR TUI Vocabulary Correction",
        "",
        f"Project ID: {manifest.project_id}",
        f"Title: {manifest.title}",
        "",
        "## Edited Samples",
    ]
    for index, sample_change in enumerate(sample_changes, start=1):
        lines.extend(
            [
                f"### Sample {index}",
                f"- Speaker: {sample_change.speaker_name}",
                f"- Before: {sample_change.original_text}",
                f"- After: {sample_change.corrected_text}",
                "",
            ]
        )
    return safe_write_text(review_path, "\n".join(lines))


def _write_polish_review_file(
    paths: ProjectPaths,
    manifest: ProjectManifest,
    result: TranscriptResult,
    speaker_mapping: dict[int, str],
) -> Path:
    """Write the source snapshot used by an automatic polish proposal."""
    review_dir = paths.root / "tmp" / REVIEW_DIR
    review_dir.mkdir(parents=True, exist_ok=True)
    review_path = review_dir / f"review_polish_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md"
    lines = [
        "# Meeting-ASR Transcript Polish Review",
        "",
        "This file records the source transcript used for an automatic polish proposal.",
        f"Project ID: {manifest.project_id}",
        f"Title: {manifest.title}",
        "",
    ]
    for sentence in result.sentences:
        if not sentence.text.strip():
            continue
        lines.append(_anchor(sentence))
        lines.append(_review_sentence_line(sentence, speaker_mapping))
        lines.append("")
    return safe_write_text(review_path, "\n".join(lines))


def _write_lexicon_review_file(paths: ProjectPaths, manifest: ProjectManifest, lexicon_db: Path) -> Path:
    """Write a small trace file for the automatic local lexicon pass."""
    review_path = _lexicon_review_path(paths)
    review_path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# Meeting-ASR Local Lexicon Correction",
        "",
        "This file records the local lexicon correction pass used by project run.",
        f"Project ID: {manifest.project_id}",
        f"Title: {manifest.title}",
        f"Lexicon DB: {lexicon_db}",
        "",
    ]
    return safe_write_text(review_path, "\n".join(lines))


def _lexicon_review_path(paths: ProjectPaths) -> Path:
    """Return the trace path for one automatic local lexicon pass."""
    return paths.root / "tmp" / REVIEW_DIR / f"review_lexicon_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md"


def _lexicon_replacement_rules(lexicon_db: Path) -> list[CorrectionReplacement]:
    """Load active lexicon rules as transcript correction replacements."""
    return [
        CorrectionReplacement(
            wrong_text=rule.wrong_text,
            corrected_text=rule.corrected_text,
            left_context=rule.left_context,
            right_context=rule.right_context,
        )
        for rule in list_lexicon_correction_rules(db_path=lexicon_db)
    ]


def _active_understanding(
    rules: list[CorrectionReplacement],
    changes: list[CorrectionChange],
) -> list[CorrectionUnderstanding]:
    """Return only local lexicon rules that changed this transcript."""
    return [item for item in _build_understanding(rules, [], changes) if item.proposed_count > 0]


def _propose_full_document_changes(
    result: TranscriptResult,
    sample_changes: list[CorrectionChange],
    rules: list[CorrectionReplacement],
    speaker_mapping: dict[int, str],
    options: CorrectionEditOptions,
) -> tuple[list[CorrectionChange], str, str | None]:
    """Propose full-document changes with DashScope, falling back to local rules."""
    if not rules:
        return sample_changes, "local-rules", None
    if not options.use_ai:
        return _local_rule_changes(result, sample_changes, rules, speaker_mapping), "local-rules", None
    try:
        settings = load_settings(require_oss=False, require_dashscope=True)
        model = options.model or settings.dashscope_correction_model
        changes = _ai_rule_changes(result, sample_changes, rules, speaker_mapping, settings, model)
        return changes, model, None
    except Exception as exc:
        changes = _local_rule_changes(result, sample_changes, rules, speaker_mapping)
        return changes, "local-rules", str(exc)


def _ai_rule_changes(
    result: TranscriptResult,
    sample_changes: list[CorrectionChange],
    rules: list[CorrectionReplacement],
    speaker_mapping: dict[int, str],
    settings: Settings,
    model: str,
) -> list[CorrectionChange]:
    """Use DashScope to propose correction text for candidate sentences."""
    candidates = _llm_candidates(result, rules, speaker_mapping)
    if not candidates:
        return sample_changes
    corrected_by_id: dict[str, str] = {}
    for batch in _batches(candidates, VOCAB_CORRECTION_BATCH_SIZE):
        llm_result = propose_vocabulary_corrections(
            samples=_llm_samples(sample_changes),
            candidates=batch,
            settings=settings,
            model=model,
        )
        corrected_by_id.update(llm_result.corrected_text_by_id)
    return _changes_from_llm_result(result, sample_changes, candidates, corrected_by_id, speaker_mapping, rules)


def _propose_polish_changes(
    paths: ProjectPaths,
    manifest: ProjectManifest,
    result: TranscriptResult,
    speaker_mapping: dict[int, str],
    options: CorrectionEditOptions,
    progress: CliProgressReporter | None,
) -> tuple[list[CorrectionChange], str, str | None]:
    """Propose sentence-level polish changes with DashScope."""
    if not options.use_ai:
        return [], "disabled", "transcript polish requires AI"
    try:
        settings = load_settings(require_oss=False, require_dashscope=True)
        model = options.model or settings.dashscope_correction_model
        concurrency = _polish_concurrency(options.polish_concurrency, settings.dashscope_correction_concurrency)
        if _polish_use_legacy(options):
            changes = _ai_polish_changes(
                paths,
                manifest,
                result,
                speaker_mapping,
                settings,
                model,
                progress,
                concurrency=concurrency,
            )
            return changes, model, None
        changes, partial_error = _strict_ai_polish_changes(
            paths,
            manifest,
            result,
            speaker_mapping,
            settings,
            model,
            progress,
            concurrency=concurrency,
        )
        return changes, model, partial_error
    except Exception as exc:
        return [], options.model or "dashscope-correction", str(exc)


def _ai_polish_changes(
    paths: ProjectPaths,
    manifest: ProjectManifest,
    result: TranscriptResult,
    speaker_mapping: dict[int, str],
    settings: Settings,
    model: str,
    progress: CliProgressReporter | None,
    *,
    concurrency: int,
) -> list[CorrectionChange]:
    """Use DashScope to propose transcript polish changes for all sentences."""
    candidates = _all_llm_candidates(result, speaker_mapping)
    corrected_by_id: dict[str, str] = {}
    batches = list(_batches(candidates, POLISH_LLM_BATCH_SIZE))
    if concurrency > 1 and len(batches) > 1:
        corrected_by_id.update(
            _run_polish_batches_parallel(
                batches,
                paths=paths,
                manifest=manifest,
                settings=settings,
                model=model,
                progress=progress,
                concurrency=concurrency,
            )
        )
        return _changes_from_polish_result(result, candidates, corrected_by_id, speaker_mapping)
    for index, batch in enumerate(batches, start=1):
        llm_result = _run_polish_batch(
            lambda batch=batch: propose_transcript_polish(candidates=batch, settings=settings, model=model),
            paths=paths,
            manifest=manifest,
            model=model,
            batch_index=index,
            batch_total=len(batches),
            progress=progress,
        )
        corrected_by_id.update(llm_result.corrected_text_by_id)
    return _changes_from_polish_result(result, candidates, corrected_by_id, speaker_mapping)


def _polish_use_legacy(options: CorrectionEditOptions) -> bool:
    """
    Strict polish (downstream-summary friendly) is now the default. Users can
    opt back into the legacy aggressive-rewrite polish via either:
      - env var MEETING_ASR_POLISH_LEGACY=1
      - options.polish_legacy=True (set by `--legacy-polish` CLI flag)
    """
    if getattr(options, "polish_legacy", False):
        return True
    return os.environ.get("MEETING_ASR_POLISH_LEGACY") == "1"


_POLISH_STRICT_BATCH_SIZE = 12
_POLISH_GUARD_MIN_LEN_RATIO = 0.3
_POLISH_GUARD_BORROW_MIN_LEN = 6
_POLISH_GUARD_MAX_LEN_DELTA = 30
_POLISH_GUARD_ASCII_TYPO_MAX_DISTANCE = 3
_POLISH_ALLOWED_TYPES = {"typo", "term", "case", "punct", "dup", "filler", "restart", "emphasis"}
_POLISH_CHANGE_TYPE_SEP_RE = re.compile(r"[|,+/&\s]+")


def _is_change_type_allowed(change_type: str) -> bool:
    """
    LLMs often emit multi-tag change_type like 'dup|restart' or 'filler,dup'.
    Treat any part that matches an allowed type as legal.
    """
    if not change_type:
        return False
    parts = [p.strip().lower() for p in _POLISH_CHANGE_TYPE_SEP_RE.split(change_type) if p.strip()]
    return any(part in _POLISH_ALLOWED_TYPES for part in parts)

# Words that must NOT be deleted by polish — they are the fact-bearing
# modifiers that downstream summary agents need in order to distinguish:
#   - decision vs proposal (attitude/confidence words)
#   - consensus vs unilateral statement (confirmation seekers)
#   - decision verbs themselves
# Order matters: longer phrases must be checked before shorter ones so that
# '我觉得' is matched as a unit rather than being approved by '觉得'.
_POLISH_PROTECTED_WORDS: tuple[str, ...] = (
    # attitude / confidence
    "我觉得", "我认为", "我感觉", "我个人觉得", "我个人认为",
    "可能", "也许", "或许", "大概", "应该", "似乎", "估计", "好像",
    # confirmation seekers
    "对吧", "对吗", "是不是", "是吧", "你说呢", "你看呢", "你觉得呢",
    "行不行", "好不好", "是吗",
    # decision verbs (positive + negative)
    "同意", "反对", "决定", "确认", "不同意", "不行", "可以", "不可以",
)


def _strict_ai_polish_changes(
    paths: ProjectPaths,
    manifest: ProjectManifest,
    result: TranscriptResult,
    speaker_mapping: dict[int, str],
    settings: Settings,
    model: str,
    progress: CliProgressReporter | None,
    *,
    concurrency: int,
) -> tuple[list[CorrectionChange], str | None]:
    """
    Run strict polish: hard-whitelist prompt + deterministic guard, write sidecar.

    Returns ``(changes, error_message)``. ``error_message`` is non-None when one
    or more batches failed but the run still produced usable output. When EVERY
    batch fails we raise so the caller surfaces the same recovery context as
    the legacy path — silently returning an empty proposal would let DashScope
    outages disguise themselves as 'no changes needed'.
    """
    candidates = _all_llm_candidates(result, speaker_mapping)
    if not candidates:
        return [], None
    batches = list(_batches(candidates, _POLISH_STRICT_BATCH_SIZE))
    request_timeout = _strict_polish_request_timeout(model)
    items_by_id: dict[str, LlmPolishItem] = {}
    failed_batches: list[tuple[int, str]] = []
    if concurrency > 1 and len(batches) > 1:
        merged, failures = _run_strict_polish_batches_parallel(
            batches,
            settings=settings,
            model=model,
            progress=progress,
            concurrency=concurrency,
            request_timeout=request_timeout,
        )
        items_by_id.update(merged)
        failed_batches.extend(failures)
    else:
        for index, batch in enumerate(batches, start=1):
            try:
                llm_result = propose_transcript_polish_strict(
                    candidates=batch,
                    settings=settings,
                    model=model,
                    request_timeout=request_timeout,
                )
            except Exception as exc:
                failed_batches.append((index, str(exc)))
                continue
            for entry in llm_result.items:
                items_by_id[entry.candidate_id] = entry
            _emit_polish_progress(
                progress,
                completed=index,
                total=len(batches),
                concurrency=1,
                active=0,
                model=model,
            )
    if failed_batches and len(failed_batches) == len(batches):
        first_index, first_error = failed_batches[0]
        wrapped = RuntimeError(first_error)
        raise RuntimeError(
            _polish_failure_message(manifest.project_id, model, first_index, len(batches), wrapped)
        )
    changes = _strict_polish_changes_from_items(
        paths=paths,
        result=result,
        candidates=candidates,
        items_by_id=items_by_id,
        speaker_mapping=speaker_mapping,
        model=model,
        failed_batches=failed_batches,
        total_batches=len(batches),
    )
    error_msg = (
        _strict_polish_partial_failure_message(model, failed_batches, len(batches))
        if failed_batches
        else None
    )
    return changes, error_msg


def _strict_polish_partial_failure_message(
    model: str,
    failed_batches: list[tuple[int, str]],
    total: int,
) -> str:
    """Compact summary for caller to surface when only some strict batches failed."""
    failed_count = len(failed_batches)
    sample_index, sample_error = failed_batches[0]
    return (
        f"Strict polish completed with partial failures: model={model} "
        f"failed_batches={failed_count}/{total} first_failure_batch={sample_index} "
        f"first_error={sample_error[:200]}"
    )


def _strict_polish_request_timeout(model: str) -> int:
    """Larger MoE models like qwen3-235b need a more generous per-call timeout."""
    if "235b" in model.lower() or "max" in model.lower():
        return 300
    return 180


def _run_strict_polish_batches_parallel(
    batches: list[list[LlmCorrectionCandidate]],
    *,
    settings: Settings,
    model: str,
    progress: CliProgressReporter | None,
    concurrency: int,
    request_timeout: int,
) -> tuple[dict[str, LlmPolishItem], list[tuple[int, str]]]:
    """Run strict polish batches concurrently. Failed batches are reported, not raised."""
    max_workers = min(concurrency, len(batches))
    items_by_id: dict[str, LlmPolishItem] = {}
    failures: list[tuple[int, str]] = []
    completed = 0
    _emit_polish_progress(
        progress,
        completed=0,
        total=len(batches),
        concurrency=max_workers,
        active=min(max_workers, len(batches)),
        model=model,
    )
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_by_index = {
            executor.submit(
                propose_transcript_polish_strict,
                candidates=batch,
                settings=settings,
                model=model,
                request_timeout=request_timeout,
            ): index
            for index, batch in enumerate(batches, start=1)
        }
        pending = set(future_by_index)
        while pending:
            done, pending = wait(pending, timeout=30.0, return_when=FIRST_COMPLETED)
            if not done:
                _emit_polish_progress(
                    progress,
                    completed=completed,
                    total=len(batches),
                    concurrency=max_workers,
                    active=min(max_workers, len(pending)),
                    model=model,
                )
                continue
            for future in done:
                batch_index = future_by_index[future]
                try:
                    llm_result = future.result()
                except Exception as exc:
                    failures.append((batch_index, str(exc)))
                    completed += 1
                    _emit_polish_progress(
                        progress,
                        completed=completed,
                        total=len(batches),
                        concurrency=max_workers,
                        active=min(max_workers, len(pending)),
                        model=model,
                    )
                    continue
                for entry in llm_result.items:
                    items_by_id[entry.candidate_id] = entry
                completed += 1
                _emit_polish_progress(
                    progress,
                    completed=completed,
                    total=len(batches),
                    concurrency=max_workers,
                    active=min(max_workers, len(pending)),
                    model=model,
                )
    return items_by_id, failures


def _strict_polish_changes_from_items(
    *,
    paths: ProjectPaths,
    result: TranscriptResult,
    candidates: list[LlmCorrectionCandidate],
    items_by_id: dict[str, LlmPolishItem],
    speaker_mapping: dict[int, str],
    model: str,
    failed_batches: list[tuple[int, str]] | None = None,
    total_batches: int = 0,
) -> list[CorrectionChange]:
    """Apply guard, build CorrectionChange list, write sidecar metadata."""
    sentences = result.sentences
    sentence_by_index = {f"c{idx}": (idx, sentence) for idx, sentence in enumerate(sentences)}
    changes: list[CorrectionChange] = []
    sidecar_rows: list[dict] = []
    for candidate in candidates:
        idx_pair = sentence_by_index.get(candidate.candidate_id)
        if idx_pair is None:
            continue
        idx, sentence = idx_pair
        original_text = sentence.text.strip()
        item = items_by_id.get(candidate.candidate_id)
        decision = "kept"
        change_type = item.change_type if item is not None else ""
        proposed_text = item.corrected_text if item is not None else original_text
        reason = item.reason if item is not None else ""
        if item is None:
            decision = "no_change"
        elif proposed_text == original_text:
            decision = "no_change"
        elif not _is_change_type_allowed(change_type):
            decision = f"reject_unknown_type:{change_type}"
        else:
            verdict = _polish_guard(idx, sentences, original_text, proposed_text)
            if verdict is not None:
                decision = f"reject:{verdict}"
        if decision == "kept" and proposed_text != original_text:
            changes.append(
                CorrectionChange(
                    sentence_id=sentence.sentence_id,
                    speaker_id=sentence.speaker_id,
                    speaker_name=_speaker_name(sentence.speaker_id, speaker_mapping),
                    begin_time_ms=sentence.begin_time_ms,
                    end_time_ms=sentence.end_time_ms,
                    original_text=original_text,
                    corrected_text=proposed_text,
                    replacements=[],
                    change_type=change_type,
                    reason=reason,
                )
            )
        sidecar_rows.append(
            {
                "candidate_id": candidate.candidate_id,
                "sentence_id": sentence.sentence_id,
                "begin_time_ms": sentence.begin_time_ms,
                "end_time_ms": sentence.end_time_ms,
                "speaker_name": _speaker_name(sentence.speaker_id, speaker_mapping),
                "original_text": original_text,
                "proposed_text": proposed_text,
                "change_type": change_type,
                "reason": reason,
                "decision": decision,
            }
        )
    _write_strict_polish_sidecar(
        paths, sidecar_rows, model,
        failed_batches=failed_batches or [],
        total_batches=total_batches,
    )
    return changes


def _polish_guard(
    sentence_index: int,
    sentences: list[SentenceSegment],
    original_text: str,
    proposed_text: str,
) -> str | None:
    """
    Deterministic post-LLM guard. Return None if accepted, else a short reason code.

    Aggressive-polish era: we now allow deleting ASR noise of any kind
    (long same-char runs, restart fragments, emphasis repeats, fillers).
    The remaining hard rules are:
      G1 length ratio: proposed must be at least 30% of original
         (catch LLM compressing whole-sentence content into a fragment)
      G2 length delta: |len(proposed) - len(original)| <= 30 chars
         (catch candidate-id swaps where one sentence is replaced by another's text)
      G3 ascii hallucination: any new ASCII token must be a typo-distance neighbor
         of an original ASCII token (CRI->CLI OK; 猫提卡->Mattika NOT OK)
      G4 protected-word deletion: must not strip any attitude/confirmation/
         decision word that downstream summary agents rely on
      G5 cross-sentence borrow: any inserted >=6-char Chinese chunk must not
         appear in either neighbor sentence (defends timestamp-content contract)
    """
    orig_len = max(1, len(original_text))
    new_len = len(proposed_text)
    if new_len < orig_len * _POLISH_GUARD_MIN_LEN_RATIO:
        return "len_ratio"
    if abs(new_len - orig_len) > _POLISH_GUARD_MAX_LEN_DELTA:
        return "len_delta"
    ascii_verdict = _ascii_hallucination_check(original_text, proposed_text)
    if ascii_verdict is not None:
        return ascii_verdict
    protected_verdict = _protected_word_deletion_check(original_text, proposed_text)
    if protected_verdict is not None:
        return protected_verdict
    neighbor_text = _neighbor_text(sentence_index, sentences)
    if neighbor_text and _has_cross_sentence_borrow(original_text, proposed_text, neighbor_text):
        return "cross_sentence_borrow"
    return None


def _protected_word_deletion_check(original: str, proposed: str) -> str | None:
    """
    Reject when polish strips a protected attitude/confirmation/decision word
    that the original sentence contained.

    Counts per-word occurrences: 删 1 个就拒。我们不容忍弱化『决议 vs 提议』
    『共识 vs 单方陈述』的分辨能力。
    """
    for word in _POLISH_PROTECTED_WORDS:
        before = original.count(word)
        if before == 0:
            continue
        if proposed.count(word) < before:
            return f"protected_word_deleted:{word}"
    return None


def _extract_ascii_tokens(text: str) -> list[str]:
    """Extract ASCII-only tokens (English words, identifiers, numbers) for retention check."""
    return ASCII_TERM_RE.findall(text)


def _ascii_hallucination_check(original: str, proposed: str) -> str | None:
    """
    Reject when the polish introduces an ASCII token that isn't plausibly a typo
    fix of something already in the original.

    Rationale: LLM 'Mattika' from Chinese '猫提卡' is a hallucination — there is no
    ASCII source word to typo-fix from. But CRI -> CLI is a 1-edit fix from an
    existing ASCII token, so it should pass.
    """
    orig_tokens = _extract_ascii_tokens(original)
    new_tokens = _extract_ascii_tokens(proposed)
    introduced = [token for token in new_tokens if token not in set(orig_tokens)]
    if not introduced:
        return None
    if not orig_tokens:
        # Polish invented ASCII out of nothing — almost always hallucination.
        return "ascii_hallucination"
    for token in introduced:
        threshold = max(_POLISH_GUARD_ASCII_TYPO_MAX_DISTANCE, len(token) // 2)
        if not any(_levenshtein(token.lower(), src.lower()) <= threshold for src in orig_tokens):
            return "ascii_hallucination"
    return None


def _levenshtein(a: str, b: str) -> int:
    """Bounded Levenshtein distance between two short strings."""
    if a == b:
        return 0
    if not a:
        return len(b)
    if not b:
        return len(a)
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, start=1):
        curr = [i] + [0] * len(b)
        for j, cb in enumerate(b, start=1):
            curr[j] = min(prev[j] + 1, curr[j - 1] + 1, prev[j - 1] + (0 if ca == cb else 1))
        prev = curr
    return prev[-1]


## NOTE: the previous _deletion_legality_check / _is_short_same_char_run_collapse
## were removed when polish was retargeted at downstream-summary friendliness.
## The conservative same-char-run heuristic is no longer wanted; deletion safety
## is now enforced by _protected_word_deletion_check + length guards instead.


def _neighbor_text(index: int, sentences: list[SentenceSegment]) -> str:
    """Return the concatenated text of immediate neighbors for borrow detection."""
    parts = []
    if index - 1 >= 0:
        parts.append(sentences[index - 1].text.strip())
    if index + 1 < len(sentences):
        parts.append(sentences[index + 1].text.strip())
    return "\n".join(parts)


_CJK_RUN_RE = re.compile(r"[一-鿿]+")


def _has_cross_sentence_borrow(original: str, proposed: str, neighbor_text: str) -> bool:
    """
    True if any contiguous Chinese chunk in proposed (>= _POLISH_GUARD_BORROW_MIN_LEN)
    is absent from original but present in either immediate neighbor.

    Why this matters: the LLM sees the whole batch and is tempted to pull content
    from the next sentence into the current one, which silently breaks the
    timestamp-content contract. We scan windows directly rather than relying on
    difflib's alignment, because difflib on mixed Chinese is easily fooled by
    incidental same-character matches into chopping a contiguous insertion into
    sub-threshold pieces (a previous attempt missed '第一次会议记录' that way).
    """
    if not neighbor_text:
        return False
    n = _POLISH_GUARD_BORROW_MIN_LEN
    for run in _CJK_RUN_RE.findall(proposed):
        if len(run) < n:
            continue
        for start in range(0, len(run) - n + 1):
            window = run[start : start + n]
            if window in original:
                continue
            if window in neighbor_text:
                return True
    return False


def _write_strict_polish_sidecar(
    paths: ProjectPaths,
    rows: list[dict],
    model: str,
    *,
    failed_batches: list[tuple[int, str]],
    total_batches: int,
) -> None:
    """Persist per-candidate decisions for offline 4-dimension analysis."""
    sidecar_dir = paths.root / "tmp" / REVIEW_DIR
    sidecar_dir.mkdir(parents=True, exist_ok=True)
    safe_model = model.replace("/", "_")
    payload = {
        "generated_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "model": model,
        "guard": {
            "len_ratio_min": _POLISH_GUARD_MIN_LEN_RATIO,
            "len_delta_max": _POLISH_GUARD_MAX_LEN_DELTA,
            "ascii_typo_max_distance": _POLISH_GUARD_ASCII_TYPO_MAX_DISTANCE,
            "borrow_min_len": _POLISH_GUARD_BORROW_MIN_LEN,
            "protected_words": list(_POLISH_PROTECTED_WORDS),
            "allowed_types": sorted(_POLISH_ALLOWED_TYPES),
        },
        "batches": {
            "total": total_batches,
            "failed_count": len(failed_batches),
            "failed": [{"index": idx, "error": err} for idx, err in failed_batches],
        },
        "items": rows,
    }
    path = sidecar_dir / f"polish_strict_meta_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{safe_model}.json"
    safe_write_json(path, payload)


def _polish_concurrency(option_value: int | None, configured_value: int) -> int:
    """Return the validated transcript polish concurrency."""
    value = option_value if option_value is not None else configured_value
    if value < 1 or value > MAX_DASHSCOPE_CORRECTION_CONCURRENCY:
        raise ValueError(
            "Transcript polish concurrency must be between "
            f"1 and {MAX_DASHSCOPE_CORRECTION_CONCURRENCY}, got {value}"
        )
    return value


def _run_polish_batches_parallel(
    batches: list[list[LlmCorrectionCandidate]],
    *,
    paths: ProjectPaths,
    manifest: ProjectManifest,
    settings: Settings,
    model: str,
    progress: CliProgressReporter | None,
    concurrency: int,
) -> dict[str, str]:
    """Run independent polish batches concurrently and merge model results."""
    max_workers = min(concurrency, len(batches))
    input_file = manifest.source.original_path or manifest.source.path
    fields = {"model": model, "batch": f"0/{len(batches)}", "concurrency": max_workers}
    _record_polish_runtime(
        paths,
        manifest,
        "polish",
        input_file,
        fields,
        f"starting {len(batches)} polish batches with concurrency {max_workers}",
    )
    _emit_polish_heartbeat(
        progress,
        manifest,
        paths,
        input_file,
        elapsed_seconds=0.0,
        last_success=f"submitted 0/{len(batches)} polish batches",
        next_action="waiting for parallel polish batches",
        fields=fields,
    )
    _emit_polish_progress(
        progress,
        completed=0,
        total=len(batches),
        concurrency=max_workers,
        active=min(max_workers, len(batches)),
        model=model,
    )
    started_at = time.monotonic()
    corrected_by_id: dict[str, str] = {}
    completed = 0
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_by_index = {
            executor.submit(propose_transcript_polish, candidates=batch, settings=settings, model=model): index
            for index, batch in enumerate(batches, start=1)
        }
        pending = set(future_by_index)
        while pending:
            done, pending = wait(pending, timeout=30.0, return_when=FIRST_COMPLETED)
            if not done:
                elapsed = time.monotonic() - started_at
                last_success = f"{completed}/{len(batches)} polish batches complete"
                heartbeat_fields = {"model": model, "batch": f"{completed}/{len(batches)}", "concurrency": max_workers}
                _record_polish_runtime(paths, manifest, "polish", input_file, heartbeat_fields, last_success)
                _emit_polish_heartbeat(
                    progress,
                    manifest,
                    paths,
                    input_file,
                    elapsed_seconds=elapsed,
                    last_success=last_success,
                    next_action="waiting for parallel polish batches",
                    fields=heartbeat_fields,
                )
                _emit_polish_progress(
                    progress,
                    completed=completed,
                    total=len(batches),
                    concurrency=max_workers,
                    active=min(max_workers, len(pending)),
                    model=model,
                )
                continue
            for future in done:
                batch_index = future_by_index[future]
                try:
                    result = future.result()
                except Exception as exc:
                    message = _polish_failure_message(manifest.project_id, model, batch_index, len(batches), exc)
                    error_fields = {"model": model, "batch": f"{batch_index}/{len(batches)}", "concurrency": max_workers}
                    _record_polish_runtime(paths, manifest, "polish", input_file, error_fields, None, last_error=message)
                    raise RuntimeError(message) from exc
                completed += 1
                corrected_by_id.update(result.corrected_text_by_id)
                done_fields = {"model": model, "batch": f"{completed}/{len(batches)}", "concurrency": max_workers}
                _record_polish_runtime(
                    paths,
                    manifest,
                    "polish",
                    input_file,
                    done_fields,
                    f"completed polish batch {batch_index}/{len(batches)}",
                )
                _emit_polish_progress(
                    progress,
                    completed=completed,
                    total=len(batches),
                    concurrency=max_workers,
                    active=min(max_workers, len(pending)),
                    model=model,
                )
    return corrected_by_id


def _run_polish_batch(
    operation: Callable[[], object],
    *,
    paths: ProjectPaths,
    manifest: ProjectManifest,
    model: str,
    batch_index: int,
    batch_total: int,
    progress: CliProgressReporter | None,
) -> object:
    """Run one polish LLM batch with heartbeat and recovery context."""
    stage = "polish"
    input_file = manifest.source.original_path or manifest.source.path
    fields = {"model": model, "batch": f"{batch_index}/{batch_total}"}
    _record_polish_runtime(paths, manifest, stage, input_file, fields, f"starting batch {batch_index}/{batch_total}")
    _emit_polish_progress(
        progress,
        completed=batch_index - 1,
        total=batch_total,
        concurrency=1,
        active=1,
        model=model,
    )
    _emit_polish_heartbeat(
        progress,
        manifest,
        paths,
        input_file,
        elapsed_seconds=0.0,
        last_success=f"starting batch {batch_index}/{batch_total}",
        next_action="waiting for DashScope correction model",
        fields=fields,
    )
    try:
        result = _run_blocking_polish_batch(
            operation,
            paths=paths,
            manifest=manifest,
            input_file=input_file,
            fields=fields,
            progress=progress,
        )
        _emit_polish_progress(
            progress,
            completed=batch_index,
            total=batch_total,
            concurrency=1,
            active=0,
            model=model,
        )
        return result
    except Exception as exc:
        message = _polish_failure_message(manifest.project_id, model, batch_index, batch_total, exc)
        _record_polish_runtime(paths, manifest, stage, input_file, fields, None, last_error=message)
        raise RuntimeError(message) from exc


def _run_blocking_polish_batch(
    operation: Callable[[], object],
    *,
    paths: ProjectPaths,
    manifest: ProjectManifest,
    input_file: str,
    fields: dict[str, object],
    progress: CliProgressReporter | None,
) -> object:
    """Emit 30-second heartbeats while one blocking LLM request is in flight."""
    if progress is None:
        return operation()
    started_at = time.monotonic()
    heartbeat_index = 0
    with ThreadPoolExecutor(max_workers=1) as executor:
        future = executor.submit(operation)
        while True:
            try:
                return future.result(timeout=30.0)
            except FutureTimeoutError:
                heartbeat_index += 1
                elapsed = time.monotonic() - started_at
                last_success = f"batch request submitted; heartbeat {heartbeat_index}"
                _record_polish_runtime(paths, manifest, "polish", input_file, fields, last_success)
                _emit_polish_heartbeat(
                    progress,
                    manifest,
                    paths,
                    input_file,
                    elapsed_seconds=elapsed,
                    last_success=last_success,
                    next_action="waiting for current polish batch",
                    fields=fields,
                )
                _emit_polish_progress_from_fields(progress, fields)


def _record_polish_runtime(
    paths: ProjectPaths,
    manifest: ProjectManifest,
    stage: str,
    input_file: str,
    fields: dict[str, object],
    last_success: str | None,
    *,
    last_error: str | None = None,
) -> None:
    """Persist polish runtime state without importing project_manager."""
    now = datetime.now().astimezone().isoformat(timespec="seconds")
    runtime = dict(manifest.runtime)
    if runtime.get("current_stage") != stage:
        runtime["stage_started_at"] = now
    runtime["current_stage"] = stage
    runtime["last_heartbeat_at"] = now
    runtime["input_file"] = input_file
    runtime["external_ids"] = {**dict(runtime.get("external_ids") or {}), **fields}
    if last_success:
        runtime["last_success"] = last_success
    if last_error:
        runtime["last_error"] = {"at": now, "stage": stage, "message": last_error}
    manifest.runtime = runtime
    safe_write_json(paths.manifest, manifest.to_dict())


def _emit_polish_heartbeat(
    progress: CliProgressReporter | None,
    manifest: ProjectManifest,
    paths: ProjectPaths,
    input_file: str,
    *,
    elapsed_seconds: float,
    last_success: str,
    next_action: str,
    fields: dict[str, object],
) -> None:
    """Emit a structured polish heartbeat."""
    emit_progress(
        progress,
        None,
        log_kind="heartbeat",
        stage="polish",
        project_id=manifest.project_id,
        project_path=str(paths.root),
        input_file=input_file,
        timestamp=datetime.now().astimezone().isoformat(timespec="seconds"),
        elapsed_seconds=elapsed_seconds,
        last_success=last_success,
        next_action=next_action,
        log_fields=tuple(fields.items()),
    )


def _emit_polish_progress_from_fields(progress: CliProgressReporter | None, fields: dict[str, object]) -> None:
    """Emit human progress from persisted polish runtime fields."""
    batch = str(fields.get("batch") or "")
    current, total = _parse_batch_progress(batch)
    if total <= 0:
        return
    concurrency = _positive_int(fields.get("concurrency"), default=1)
    model = str(fields.get("model") or "configured-model")
    _emit_polish_progress(
        progress,
        completed=max(0, current - 1),
        total=total,
        concurrency=concurrency,
        active=1,
        model=model,
    )


def _emit_polish_progress(
    progress: CliProgressReporter | None,
    *,
    completed: int,
    total: int,
    concurrency: int,
    active: int,
    model: str,
) -> None:
    """Emit human-readable transcript polish batch progress."""
    if total <= 0:
        return
    safe_completed = max(0, min(completed, total))
    safe_concurrency = max(1, concurrency)
    safe_active = max(0, min(active, safe_concurrency, total - safe_completed))
    emit_progress(
        progress,
        _polish_progress_description(
            completed=safe_completed,
            total=total,
            concurrency=safe_concurrency,
            active=safe_active,
            model=model,
        ),
        total=total,
        completed=safe_completed,
    )


def _polish_progress_description(
    *,
    completed: int,
    total: int,
    concurrency: int,
    active: int,
    model: str,
) -> str:
    """Return the progress row description for transcript polish batches."""
    return (
        "Generating transcript polish proposal"
        f" | batches {completed}/{total}"
        f" | parallel {concurrency}"
        f" | active {active}"
        f" | model {model}"
    )


def _parse_batch_progress(value: str) -> tuple[int, int]:
    """Parse a ``current/total`` batch marker."""
    if "/" not in value:
        return 0, 0
    current_text, total_text = value.split("/", 1)
    try:
        return int(current_text), int(total_text)
    except ValueError:
        return 0, 0


def _positive_int(value: object, *, default: int) -> int:
    """Return a positive integer value or a default."""
    try:
        parsed = int(str(value))
    except (TypeError, ValueError):
        return default
    return parsed if parsed > 0 else default


def _polish_failure_message(
    project_id: str,
    model: str,
    batch_index: int,
    batch_total: int,
    error: Exception,
) -> str:
    """Return a transcript polish failure message with recovery commands."""
    return (
        f"Transcript polish failed: project_id={project_id} stage=polish "
        f"batch={batch_index}/{batch_total} model={model} "
        f"timeout={DASHSCOPE_TEXT_REQUEST_TIMEOUT_SECONDS}s error={error}. "
        f"Show: meeting-asr project show {project_id}. "
        f"Review: meeting-asr project review {project_id}. "
        f"Retry polish: meeting-asr project correct polish {project_id} --model {model}. "
        "Skip polish: rerun project run with --no-polish, or continue with project review."
    )


def _local_rule_changes(
    result: TranscriptResult,
    sample_changes: list[CorrectionChange],
    rules: list[CorrectionReplacement],
    speaker_mapping: dict[int, str],
) -> list[CorrectionChange]:
    """Apply inferred replacement rules with deterministic string replacement."""
    sample_by_key = {_change_key(change): change for change in sample_changes}
    changes = []
    for sentence in result.sentences:
        text = sample_by_key.get(_sentence_change_key(sentence), None)
        proposed_text = text.corrected_text if text is not None else sentence.text
        proposed_text = _apply_rules_to_text(proposed_text, rules)
        if proposed_text != sentence.text.strip():
            changes.append(_change_from_sentence_with_rules(sentence, proposed_text, speaker_mapping, rules))
    return changes


def _build_understanding(
    rules: list[CorrectionReplacement],
    sample_changes: list[CorrectionChange],
    proposed_changes: list[CorrectionChange],
) -> list[CorrectionUnderstanding]:
    """Summarize inferred correction rules for human review."""
    rows = []
    for rule in rules:
        rows.append(
            CorrectionUnderstanding(
                wrong_text=rule.wrong_text,
                corrected_text=rule.corrected_text,
                sample_count=_replacement_count(sample_changes, rule),
                proposed_count=_replacement_count(proposed_changes, rule),
                left_context=rule.left_context,
                right_context=rule.right_context,
            )
        )
    return rows


def _replacement_count(changes: list[CorrectionChange], rule: CorrectionReplacement) -> int:
    """Count changes that contain one wrong-to-corrected replacement."""
    return sum(
        1
        for change in changes
        if rule.wrong_text in change.original_text and rule.corrected_text in change.corrected_text
    )


def _render_review_file(
    manifest: ProjectManifest,
    result: TranscriptResult,
    speaker_mapping: dict[int, str],
) -> str:
    """Render the editable review file."""
    lines = [
        "# Meeting-ASR Vocabulary Correction Review",
        "",
        "Edit only transcript text after the speaker label. Keep HTML anchor comments intact.",
        f"Project ID: {manifest.project_id}",
        f"Title: {manifest.title}",
        "",
    ]
    for sentence in result.sentences:
        if not sentence.text.strip():
            continue
        lines.append(_anchor(sentence))
        lines.append(_review_sentence_line(sentence, speaker_mapping))
        lines.append("")
    return "\n".join(lines)


def _unique_replacements(changes: list[CorrectionChange]) -> list[CorrectionReplacement]:
    """Return first-seen replacement rules inferred from sample changes."""
    seen: set[tuple[str, str]] = set()
    replacements = []
    for change in changes:
        for replacement in change.replacements:
            key = (replacement.wrong_text, replacement.corrected_text)
            if key in seen:
                continue
            seen.add(key)
            replacements.append(replacement)
    return replacements


def _llm_candidates(
    result: TranscriptResult,
    rules: list[CorrectionReplacement],
    speaker_mapping: dict[int, str],
) -> list[LlmCorrectionCandidate]:
    """Build model candidates from sentences containing observed wrong terms."""
    candidates = []
    for index, sentence in enumerate(result.sentences):
        if not any(rule.wrong_text in sentence.text for rule in rules):
            continue
        candidates.append(
            LlmCorrectionCandidate(
                candidate_id=f"c{index}",
                sentence_id=sentence.sentence_id,
                speaker_name=_speaker_name(sentence.speaker_id, speaker_mapping),
                text=sentence.text,
            )
        )
    return candidates


def _all_llm_candidates(
    result: TranscriptResult,
    speaker_mapping: dict[int, str],
) -> list[LlmCorrectionCandidate]:
    """Build model candidates from all non-empty transcript sentences."""
    candidates = []
    for index, sentence in enumerate(result.sentences):
        if not sentence.text.strip():
            continue
        candidates.append(
            LlmCorrectionCandidate(
                candidate_id=f"c{index}",
                sentence_id=sentence.sentence_id,
                speaker_name=_speaker_name(sentence.speaker_id, speaker_mapping),
                text=sentence.text,
            )
        )
    return candidates


def _llm_samples(changes: list[CorrectionChange]) -> list[LlmCorrectionSample]:
    """Build model samples from user-edited sentence changes."""
    return [
        LlmCorrectionSample(
            original_text=change.original_text,
            corrected_text=change.corrected_text,
            replacements=[asdict(replacement) for replacement in change.replacements],
        )
        for change in changes
    ]


def _changes_from_llm_result(
    result: TranscriptResult,
    sample_changes: list[CorrectionChange],
    candidates: list[LlmCorrectionCandidate],
    corrected_by_id: dict[str, str],
    speaker_mapping: dict[int, str],
    rules: list[CorrectionReplacement],
) -> list[CorrectionChange]:
    """Merge sample edits with validated model-suggested corrections."""
    sample_by_key = {_change_key(change): change for change in sample_changes}
    candidate_ids = {candidate.candidate_id for candidate in candidates}
    changes = []
    for index, sentence in enumerate(result.sentences):
        sample_change = sample_by_key.get(_sentence_change_key(sentence))
        proposed_text = sample_change.corrected_text if sample_change else corrected_by_id.get(f"c{index}")
        if f"c{index}" not in candidate_ids and sample_change is None:
            continue
        if not _valid_model_text(sentence.text, proposed_text):
            proposed_text = sample_change.corrected_text if sample_change else None
        if proposed_text and proposed_text != sentence.text.strip():
            changes.append(_change_from_sentence_with_rules(sentence, proposed_text, speaker_mapping, rules))
    return changes


def _changes_from_polish_result(
    result: TranscriptResult,
    candidates: list[LlmCorrectionCandidate],
    corrected_by_id: dict[str, str],
    speaker_mapping: dict[int, str],
) -> list[CorrectionChange]:
    """Convert model polish text into safe non-lexicon correction changes."""
    candidate_ids = {candidate.candidate_id for candidate in candidates}
    changes = []
    for index, sentence in enumerate(result.sentences):
        if f"c{index}" not in candidate_ids:
            continue
        proposed_text = corrected_by_id.get(f"c{index}")
        if not _valid_model_text(sentence.text, proposed_text):
            continue
        if proposed_text == sentence.text.strip():
            continue
        change = _change_from_sentence(sentence, proposed_text, speaker_mapping)
        changes.append(replace(change, replacements=[]))
    return changes


def _valid_model_text(original_text: str, proposed_text: str | None) -> bool:
    """Return whether model text is safe enough to enter a proposal."""
    if proposed_text is None:
        return False
    cleaned = proposed_text.strip()
    if not cleaned:
        return False
    return len(cleaned) <= max(len(original_text) * 3, len(original_text) + 80)


def _apply_rules_to_text(text: str, rules: list[CorrectionReplacement]) -> str:
    """Apply deterministic replacement rules to text."""
    corrected = text.strip()
    for rule in rules:
        corrected = _apply_one_rule_to_text(corrected, rule)
    return corrected


def _apply_one_rule_to_text(text: str, rule: CorrectionReplacement) -> str:
    """Apply one rule, using token boundaries for ASCII terms."""
    if not rule.wrong_text:
        return text
    if ASCII_TERM_RE.fullmatch(rule.wrong_text):
        pattern = re.compile(rf"(?<![A-Za-z0-9_+.#-]){re.escape(rule.wrong_text)}(?![A-Za-z0-9_+.#-])")
        return pattern.sub(rule.corrected_text, text)
    return text.replace(rule.wrong_text, rule.corrected_text)


def _anchor(sentence: SentenceSegment) -> str:
    """Return a stable HTML anchor for one sentence."""
    fields = {
        "sentence_id": "" if sentence.sentence_id is None else str(sentence.sentence_id),
        "speaker_id": "" if sentence.speaker_id is None else str(sentence.speaker_id),
        "begin": str(sentence.begin_time_ms),
        "end": str(sentence.end_time_ms),
        "hash": _text_hash(sentence.text),
    }
    payload = " ".join(f"{key}={value}" for key, value in fields.items())
    return f"<!-- meeting-asr: {payload} -->"

def _review_sentence_line(sentence: SentenceSegment, speaker_mapping: dict[int, str]) -> str:
    """Render one editable transcript line."""
    label = _speaker_name(sentence.speaker_id, speaker_mapping)
    return f"[{_plain_timestamp(sentence.begin_time_ms)} - {_plain_timestamp(sentence.end_time_ms)}] {label}: {sentence.text}"


def _extract_changes(
    edited: str,
    original: TranscriptResult,
    speaker_mapping: dict[int, str],
) -> list[CorrectionChange]:
    """Extract sentence text changes from an edited review file."""
    edited_by_key = _parse_review_text(edited)
    original_by_key = {_sentence_key(sentence): sentence for sentence in original.sentences}
    changes: list[CorrectionChange] = []
    for key, sentence in original_by_key.items():
        corrected_text = edited_by_key.get(key)
        if corrected_text is None:
            continue
        original_text = sentence.text.strip()
        corrected_text = corrected_text.strip()
        if corrected_text == original_text:
            continue
        changes.append(
            CorrectionChange(
                sentence_id=sentence.sentence_id,
                speaker_id=sentence.speaker_id,
                speaker_name=_speaker_name(sentence.speaker_id, speaker_mapping),
                begin_time_ms=sentence.begin_time_ms,
                end_time_ms=sentence.end_time_ms,
                original_text=original_text,
                corrected_text=corrected_text,
                replacements=_infer_replacements(original_text, corrected_text),
            )
        )
    return changes


def _parse_review_text(edited: str) -> dict[tuple[int | None, int, int, str], str]:
    """Parse edited review text into sentence-keyed transcript text."""
    lines = edited.splitlines()
    parsed: dict[tuple[int | None, int, int, str], str] = {}
    pending_key: tuple[int | None, int, int, str] | None = None
    for line in lines:
        anchor_match = ANCHOR_RE.match(line.strip())
        if anchor_match:
            pending_key = _anchor_key(anchor_match.group("fields"))
            continue
        if pending_key is not None:
            text = _line_transcript_text(line)
            if text is not None:
                parsed[pending_key] = text
                pending_key = None
    return parsed

def _anchor_key(fields_text: str) -> tuple[int | None, int, int, str]:
    """Parse an anchor into a stable sentence key."""
    fields = {}
    for item in fields_text.split():
        if "=" not in item:
            continue
        key, value = item.split("=", 1)
        fields[key] = value
    sentence_id = _optional_int(fields.get("sentence_id"))
    begin = int(fields.get("begin") or 0)
    end = int(fields.get("end") or 0)
    text_hash = fields.get("hash") or ""
    return sentence_id, begin, end, text_hash

def _sentence_key(sentence: SentenceSegment) -> tuple[int | None, int, int, str]:
    """Return the stable key for one sentence."""
    return sentence.sentence_id, sentence.begin_time_ms, sentence.end_time_ms, _text_hash(sentence.text)

def _line_transcript_text(line: str) -> str | None:
    """Extract editable transcript text from one review line."""
    match = TIMESTAMP_LINE_RE.match(line)
    if match:
        return match.group("text")
    stripped = line.strip()
    return stripped or None


def _apply_changes(original: TranscriptResult, changes: list[CorrectionChange]) -> TranscriptResult:
    """Apply sentence-level changes to a transcript result."""
    changes_by_key = {(change.sentence_id, change.begin_time_ms, change.end_time_ms): change for change in changes}
    sentences = []
    for sentence in original.sentences:
        change = changes_by_key.get((sentence.sentence_id, sentence.begin_time_ms, sentence.end_time_ms))
        text = change.corrected_text if change is not None else sentence.text
        sentences.append(
            SentenceSegment(
                begin_time_ms=sentence.begin_time_ms,
                end_time_ms=sentence.end_time_ms,
                text=text,
                speaker_id=sentence.speaker_id,
                sentence_id=sentence.sentence_id,
            )
        )
    result = TranscriptResult("".join(sentence.text for sentence in sentences), sentences, [])
    result.detected_speakers = detect_speaker_ids(result)
    return result


def _change_from_sentence(
    sentence: SentenceSegment,
    corrected_text: str,
    speaker_mapping: dict[int, str],
) -> CorrectionChange:
    """Build a correction change from a transcript sentence."""
    original_text = sentence.text.strip()
    corrected = corrected_text.strip()
    return CorrectionChange(
        sentence_id=sentence.sentence_id,
        speaker_id=sentence.speaker_id,
        speaker_name=_speaker_name(sentence.speaker_id, speaker_mapping),
        begin_time_ms=sentence.begin_time_ms,
        end_time_ms=sentence.end_time_ms,
        original_text=original_text,
        corrected_text=corrected,
        replacements=_infer_replacements(original_text, corrected),
    )


def _change_from_sentence_with_rules(
    sentence: SentenceSegment,
    corrected_text: str,
    speaker_mapping: dict[int, str],
    rules: list[CorrectionReplacement],
) -> CorrectionChange:
    """Build a correction change while preserving term-level replacement rules."""
    change = _change_from_sentence(sentence, corrected_text, speaker_mapping)
    replacements = matching_correction_replacements(change, rules)
    return replace(change, replacements=replacements or change.replacements)


def _change_key(change: CorrectionChange) -> tuple[int | None, int, int]:
    """Return the sentence identity for a correction change."""
    return change.sentence_id, change.begin_time_ms, change.end_time_ms

def _sentence_change_key(sentence: SentenceSegment) -> tuple[int | None, int, int]:
    """Return the correction identity for a transcript sentence."""
    return sentence.sentence_id, sentence.begin_time_ms, sentence.end_time_ms

def _batches(items: list[LlmCorrectionCandidate], size: int) -> list[list[LlmCorrectionCandidate]]:
    """Split items into fixed-size batches."""
    return [items[index : index + size] for index in range(0, len(items), size)]


def _write_corrected_outputs(
    paths: ProjectPaths,
    result: TranscriptResult,
    speaker_mapping: dict[int, str],
    changes: list[CorrectionChange],
) -> dict[str, Path]:
    """Write corrected transcript artifacts and applied change metadata."""
    corrections_dir = paths.root / "corrections"
    corrections_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "full_text": result.full_text,
        "detected_speakers": result.detected_speakers,
        "sentences": [sentence.to_dict() for sentence in result.sentences],
    }
    applied_payload = {
        "changes": [_change_payload(change) for change in changes],
    }
    outputs = {
        "sentences": safe_write_json(paths.asr_dir / "sentences_corrected.json", payload),
        "transcript": safe_write_text(paths.exports_dir / "transcript_corrected.txt", render_plain_text(result)),
        "speaker_transcript": safe_write_text(
            paths.exports_dir / "transcript_speakers_corrected.txt",
            render_speaker_text(result),
        ),
        "named_transcript": safe_write_text(
            paths.exports_dir / "transcript_named_corrected.txt",
            render_named_speaker_text(result, speaker_mapping),
        ),
        "srt": safe_write_text(
            paths.exports_dir / "subtitle_named_corrected.srt",
            render_named_srt(result, speaker_mapping),
        ),
        "anonymous_srt": safe_write_text(paths.exports_dir / "subtitle_corrected.srt", build_srt(result.sentences)),
        "applied": safe_write_json(corrections_dir / "applied.json", applied_payload),
    }
    return outputs


def _change_payload(change: CorrectionChange) -> dict:
    """Convert one change to a JSON-ready payload."""
    return {
        "sentence_id": change.sentence_id,
        "speaker_id": change.speaker_id,
        "speaker_name": change.speaker_name,
        "begin_time_ms": change.begin_time_ms,
        "end_time_ms": change.end_time_ms,
        "original_text": change.original_text,
        "corrected_text": change.corrected_text,
        "replacements": [asdict(replacement) for replacement in change.replacements],
    }


def _lexicon_contexts(
    changes: list[CorrectionChange],
    project_id: str,
    category: str,
    review_path: Path,
) -> list[LexiconContext]:
    """Build lexicon contexts from accepted editor changes."""
    contexts = []
    for change in changes:
        for replacement in change.replacements:
            contexts.append(
                LexiconContext(
                    canonical=replacement.corrected_text,
                    wrong_text=replacement.wrong_text,
                    corrected_text=replacement.corrected_text,
                    left_context=replacement.left_context,
                    right_context=replacement.right_context,
                    category=category,
                    speaker_name=change.speaker_name,
                    project_id=project_id,
                    sentence_id=change.sentence_id,
                    source=f"project_correct_edit:{review_path.name}",
                )
            )
    return contexts


def _infer_replacements(original_text: str, corrected_text: str) -> list[CorrectionReplacement]:
    """Infer lexical replacement spans from before/after sentence text."""
    replacements = []
    matcher = difflib.SequenceMatcher(a=original_text, b=corrected_text, autojunk=False)
    for tag, first_start, first_end, second_start, second_end in matcher.get_opcodes():
        if tag != "replace":
            continue
        first_start, first_end, second_start, second_end = _expand_replacement_span(
            original_text,
            corrected_text,
            first_start,
            first_end,
            second_start,
            second_end,
        )
        wrong = original_text[first_start:first_end].strip()
        right = corrected_text[second_start:second_end].strip()
        if not _learnable_replacement(wrong, right):
            continue
        replacements.append(
            CorrectionReplacement(
                wrong_text=wrong,
                corrected_text=right,
                left_context=original_text[max(0, first_start - 24) : first_start].strip(),
                right_context=original_text[first_end : first_end + 24].strip(),
            )
        )
    return replacements


def _expand_replacement_span(
    original_text: str,
    corrected_text: str,
    original_start: int,
    original_end: int,
    corrected_start: int,
    corrected_end: int,
) -> tuple[int, int, int, int]:
    """
    Expand partial ASCII edits to whole-term boundaries.

    Args:
        original_text: Text before the user edit.
        corrected_text: Text after the user edit.
        original_start: Start offset of the original replacement span.
        original_end: End offset of the original replacement span.
        corrected_start: Start offset of the corrected replacement span.
        corrected_end: End offset of the corrected replacement span.

    Returns:
        Replacement span offsets after safe term expansion.
    """
    original_start, original_end = _expand_ascii_term(original_text, original_start, original_end)
    corrected_start, corrected_end = _expand_ascii_term(corrected_text, corrected_start, corrected_end)
    return original_start, original_end, corrected_start, corrected_end


def _expand_ascii_term(text: str, start: int, end: int) -> tuple[int, int]:
    """
    Expand a character-level edit span to the enclosing ASCII term.

    Args:
        text: Source text.
        start: Start offset from SequenceMatcher.
        end: End offset from SequenceMatcher.

    Returns:
        Expanded span when the edit intersects an ASCII term, otherwise the input span.
    """
    for match in ASCII_TERM_RE.finditer(text):
        term_start, term_end = match.span()
        if term_start < end and start < term_end:
            return term_start, term_end
    return start, end


def _learnable_replacement(wrong_text: str, corrected_text: str) -> bool:
    """Return whether a replacement looks like vocabulary, not punctuation."""
    if not wrong_text or not corrected_text or wrong_text == corrected_text:
        return False
    return bool(WORD_RE.search(wrong_text) and WORD_RE.search(corrected_text))


def _speaker_name(speaker_id: int | None, speaker_mapping: dict[int, str]) -> str:
    """Return mapped speaker name or anonymous fallback."""
    if speaker_id is None:
        return "Speaker Unknown"
    return speaker_mapping.get(speaker_id, speaker_id_to_label(speaker_id))

def _plain_timestamp(ms: int) -> str:
    """Format milliseconds as HH:MM:SS.mmm."""
    value = max(0, int(ms))
    hours, rem = divmod(value, 3_600_000)
    minutes, rem = divmod(rem, 60_000)
    seconds, millis = divmod(rem, 1000)
    return f"{hours:02d}:{minutes:02d}:{seconds:02d}.{millis:03d}"

def _text_hash(text: str) -> str:
    """Return a compact text hash for sentence anchors."""
    return hashlib.sha1(text.encode("utf-8")).hexdigest()[:12]

def _optional_int(value: str | None) -> int | None:
    """Parse optional integer text."""
    if value is None or value == "":
        return None
    return int(value)

def _optional_int_from_any(value: object) -> int | None:
    """Parse optional integer values from JSON."""
    if value is None or value == "":
        return None
    return int(value)
