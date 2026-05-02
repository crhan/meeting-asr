"""Editor-driven transcript vocabulary correction workflow."""

from __future__ import annotations

import difflib
import hashlib
import re
from dataclasses import asdict
from datetime import datetime
from pathlib import Path

from app.config import Settings, load_settings
from app.correction_editor import open_editor
from app.correction_hotwords import hotwords_from_understanding, write_hotword_artifact
from app.correction_llm import LlmCorrectionCandidate, LlmCorrectionSample, propose_vocabulary_corrections
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
from app.core.project_models import ProjectManifest, ProjectPaths
from app.lexicon_store import LexiconContext, default_lexicon_db_path, record_lexicon_contexts
from app.models import SentenceSegment, TranscriptResult
from app.postprocess import detect_speaker_ids, render_plain_text, render_speaker_text, speaker_id_to_label
from app.speaker_labeling import load_transcript_result, render_named_speaker_text, render_named_srt
from app.srt_utils import build_srt
from app.utils import safe_write_json, safe_write_text

ANCHOR_RE = re.compile(r"^<!-- meeting-asr: (?P<fields>.+) -->$")
TIMESTAMP_LINE_RE = re.compile(r"^\[[^\]]+\]\s*(?P<label>.*?):\s*(?P<text>.*)$")
WORD_RE = re.compile(r"[\w\u4e00-\u9fff]+", re.UNICODE)
REVIEW_DIR = "corrections"
MAX_LLM_BATCH_SIZE = 80


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


def accept_correction_proposal(
    *,
    paths: ProjectPaths,
    manifest: ProjectManifest,
    speaker_mapping: dict[int, str],
    proposal_path: Path | None,
    lexicon_db: Path | None = None,
) -> CorrectionEditSummary:
    """
    Accept a generated correction proposal and write final artifacts.

    Args:
        paths: Project paths.
        manifest: Project manifest.
        speaker_mapping: Speaker id to display name mapping.
        proposal_path: Proposal JSON path, or None for the latest proposal.
        lexicon_db: Optional lexicon database override.

    Returns:
        Correction edit summary.
    """
    proposal = load_correction_proposal(paths, proposal_path)
    if proposal.project_id != manifest.project_id:
        raise RuntimeError(f"Correction proposal belongs to another project: {proposal.project_id}")
    source = _load_source_path(paths, proposal.source_path)
    corrected = _apply_changes(source.result, proposal.proposed_changes)
    outputs = _write_corrected_outputs(paths, corrected, speaker_mapping, proposal.proposed_changes)
    hotwords_path = _write_accept_hotwords(paths, proposal)
    database_path = lexicon_db or default_lexicon_db_path()
    contexts = _lexicon_contexts(proposal.proposed_changes, manifest.project_id, proposal.category, proposal.review_path)
    learned_count = record_lexicon_contexts(contexts, db_path=database_path)
    return CorrectionEditSummary(
        review_path=proposal.review_path,
        proposal_path=proposal.proposal_path,
        proposal_diff_path=proposal.diff_path,
        proposal_json_path=proposal.json_path,
        change_count=len(proposal.proposed_changes),
        sample_change_count=len(proposal.sample_changes),
        proposed_change_count=len(proposal.proposed_changes),
        learned_count=learned_count,
        accepted=True,
        model=proposal.model,
        model_error=proposal.model_error,
        understanding=proposal.understanding,
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


def _write_accept_hotwords(paths: ProjectPaths, proposal: CorrectionProposal) -> Path:
    """Write ASR hotwords produced by the accepted correction proposal."""
    hotwords = hotwords_from_understanding(proposal.understanding, category=proposal.category)
    return write_hotword_artifact(paths.root / "corrections" / "asr_hotwords.json", hotwords)


def _empty_summary(review_path: Path, lexicon_db: Path) -> CorrectionEditSummary:
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
        model=None,
        model_error=None,
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
    rules = _unique_replacements(sample_changes)
    proposed_changes, model, model_error = _propose_full_document_changes(
        source.result,
        sample_changes,
        rules,
        speaker_mapping,
        options,
    )
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
    for batch in _batches(candidates, MAX_LLM_BATCH_SIZE):
        llm_result = propose_vocabulary_corrections(
            samples=_llm_samples(sample_changes),
            candidates=batch,
            settings=settings,
            model=model,
        )
        corrected_by_id.update(llm_result.corrected_text_by_id)
    return _changes_from_llm_result(result, sample_changes, candidates, corrected_by_id, speaker_mapping)


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
            changes.append(_change_from_sentence(sentence, proposed_text, speaker_mapping))
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
            changes.append(_change_from_sentence(sentence, proposed_text, speaker_mapping))
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
        corrected = corrected.replace(rule.wrong_text, rule.corrected_text)
    return corrected

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
