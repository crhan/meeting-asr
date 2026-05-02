"""Editor-driven transcript vocabulary correction workflow."""

from __future__ import annotations

import difflib
import hashlib
import re
import shlex
import shutil
import subprocess
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path

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


@dataclass(frozen=True, slots=True)
class CorrectionEditOptions:
    """Options for editor-driven correction."""

    editor: str | None = None
    open_editor: bool = True
    category: str = "unknown"
    lexicon_db: Path | None = None
    from_original: bool = False


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
class CorrectionEditSummary:
    """Result of an editor-driven correction run."""

    review_path: Path
    change_count: int
    learned_count: int
    corrected_sentences_path: Path | None
    corrected_transcript_path: Path | None
    corrected_named_transcript_path: Path | None
    corrected_srt_path: Path | None
    applied_path: Path | None
    lexicon_db: Path | None


def run_editor_correction(
    *,
    paths: ProjectPaths,
    manifest: ProjectManifest,
    speaker_mapping: dict[int, str],
    options: CorrectionEditOptions,
) -> CorrectionEditSummary:
    """
    Run the editor-based correction workflow for one project.

    Args:
        paths: Project paths.
        manifest: Project manifest.
        speaker_mapping: Speaker id to display name mapping.
        options: Correction options.

    Returns:
        Correction edit summary.
    """
    result = _load_correction_source(paths, from_original=options.from_original)
    review_path = _write_review_file(paths, manifest, result, speaker_mapping)
    if options.open_editor:
        _open_editor(review_path, options.editor)
    edited = review_path.read_text(encoding="utf-8")
    changes = _extract_changes(edited, result, speaker_mapping)
    lexicon_db = options.lexicon_db or default_lexicon_db_path()
    if not changes:
        return CorrectionEditSummary(review_path, 0, 0, None, None, None, None, None, lexicon_db)
    corrected = _apply_changes(result, changes)
    outputs = _write_corrected_outputs(paths, corrected, speaker_mapping, changes)
    contexts = _lexicon_contexts(changes, manifest.project_id, options.category, review_path)
    learned_count = record_lexicon_contexts(contexts, db_path=lexicon_db)
    return CorrectionEditSummary(
        review_path=review_path,
        change_count=len(changes),
        learned_count=learned_count,
        corrected_sentences_path=outputs["sentences"],
        corrected_transcript_path=outputs["transcript"],
        corrected_named_transcript_path=outputs["named_transcript"],
        corrected_srt_path=outputs["srt"],
        applied_path=outputs["applied"],
        lexicon_db=lexicon_db,
    )


def _load_correction_source(paths: ProjectPaths, *, from_original: bool) -> TranscriptResult:
    """Load the preferred correction source transcript."""
    corrected_path = paths.asr_dir / "sentences_corrected.json"
    if corrected_path.exists() and not from_original:
        return load_transcript_result(corrected_path)
    return load_transcript_result(paths.asr_dir / "sentences.json")


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


def _open_editor(path: Path, editor: str | None) -> None:
    """Open the review file and wait until the editor exits."""
    command = _editor_command(editor, path)
    subprocess.run(command, check=True)


def _editor_command(editor: str | None, path: Path) -> list[str]:
    """Build an editor command for one file."""
    command_text = editor or _default_editor()
    parts = shlex.split(command_text)
    if not parts:
        raise ValueError("Editor command must not be empty.")
    file_text = str(path)
    if any("{file}" in part for part in parts):
        return [part.replace("{file}", file_text) for part in parts]
    return parts + [file_text]


def _default_editor() -> str:
    """Return a practical default editor command."""
    import os

    configured = os.environ.get("VISUAL") or os.environ.get("EDITOR")
    if configured:
        return configured
    if shutil.which("code"):
        return "code --wait"
    return "vim"


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
