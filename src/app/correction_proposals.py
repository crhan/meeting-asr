"""Correction proposal persistence and rendering."""

from __future__ import annotations

import difflib
import json
from dataclasses import asdict
from datetime import datetime
from pathlib import Path

from app.core.project_models import ProjectManifest, ProjectPaths
from app.correction_types import (
    CorrectionChange,
    CorrectionEditOptions,
    CorrectionProposal,
    CorrectionReplacement,
    CorrectionSource,
    CorrectionUnderstanding,
)
from app.models import SentenceSegment, TranscriptResult
from app.postprocess import speaker_id_to_label
from app.utils import safe_write_json, safe_write_text

REVIEW_DIR = "corrections"


def write_correction_proposal_files(
    *,
    paths: ProjectPaths,
    manifest: ProjectManifest,
    source: CorrectionSource,
    proposed: TranscriptResult,
    review_path: Path,
    sample_changes: list[CorrectionChange],
    proposed_changes: list[CorrectionChange],
    understanding: list[CorrectionUnderstanding],
    speaker_mapping: dict[int, str],
    options: CorrectionEditOptions,
    model: str,
    model_error: str | None,
) -> CorrectionProposal:
    """
    Write correction proposal markdown, diff, and JSON files.

    Args:
        paths: Project paths.
        manifest: Project manifest.
        source: Source transcript.
        proposed: Proposed transcript after all changes.
        review_path: User-edited review file.
        sample_changes: Direct user sample edits.
        proposed_changes: Full-document proposed changes.
        understanding: Inferred correction rules.
        speaker_mapping: Speaker id to display name mapping.
        options: Correction options.
        model: Proposal model name.
        model_error: Optional model failure detail.

    Returns:
        Written proposal record.
    """
    proposal_dir = paths.root / "tmp" / REVIEW_DIR
    proposal_dir.mkdir(parents=True, exist_ok=True)
    stem = f"proposal_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    diff_path = _write_diff(proposal_dir, stem, source.result, proposed, speaker_mapping)
    proposal_path = _write_markdown(
        proposal_dir, stem, manifest, understanding, sample_changes, proposed_changes, diff_path, model, model_error
    )
    json_path = _write_json(
        proposal_dir, stem, paths, manifest, source, review_path, sample_changes, proposed_changes, understanding,
        proposal_path, diff_path, model, model_error, options
    )
    return _proposal_record(
        manifest, options, review_path, proposal_path, diff_path, json_path, source, sample_changes, proposed_changes,
        understanding, model, model_error, paths.root
    )


def load_correction_proposal(paths: ProjectPaths, proposal_path: Path | None) -> CorrectionProposal:
    """
    Load a pending correction proposal JSON file.

    Args:
        paths: Project paths.
        proposal_path: Explicit proposal JSON path, or None for latest.

    Returns:
        Parsed proposal record.
    """
    json_path = _resolve_json(paths, proposal_path)
    payload = json.loads(json_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise RuntimeError(f"Correction proposal must be a JSON object: {json_path}")
    return CorrectionProposal(
        project_id=str(payload.get("project_id") or ""),
        category=str(payload.get("category") or "unknown"),
        review_path=_project_path(paths.root, payload.get("review_path")),
        proposal_path=_project_path(paths.root, payload.get("proposal_path")),
        diff_path=_project_path(paths.root, payload.get("diff_path")),
        json_path=json_path,
        source_path=_project_path(paths.root, payload.get("source_path")),
        sample_changes=_changes_from_payload(payload.get("sample_changes")),
        proposed_changes=_changes_from_payload(payload.get("proposed_changes")),
        understanding=_understanding_from_payload(payload.get("understanding")),
        model=str(payload.get("model") or "unknown"),
        model_error=_optional_str(payload.get("model_error")),
        from_original=bool(payload.get("from_original")),
    )


def _write_diff(
    proposal_dir: Path,
    stem: str,
    original: TranscriptResult,
    proposed: TranscriptResult,
    speaker_mapping: dict[int, str],
) -> Path:
    """Write unified diff for the proposal."""
    before = _diff_lines(original, speaker_mapping)
    after = _diff_lines(proposed, speaker_mapping)
    diff_text = "".join(difflib.unified_diff(before, after, fromfile="before", tofile="proposed", n=3))
    return safe_write_text(proposal_dir / f"{stem}.diff", diff_text)


def _write_markdown(
    proposal_dir: Path,
    stem: str,
    manifest: ProjectManifest,
    understanding: list[CorrectionUnderstanding],
    sample_changes: list[CorrectionChange],
    proposed_changes: list[CorrectionChange],
    diff_path: Path,
    model: str,
    model_error: str | None,
) -> Path:
    """Write the human-readable proposal markdown file."""
    markdown = _render_markdown(manifest, understanding, sample_changes, proposed_changes, diff_path, model, model_error)
    return safe_write_text(proposal_dir / f"{stem}.md", markdown)


def _write_json(
    proposal_dir: Path,
    stem: str,
    paths: ProjectPaths,
    manifest: ProjectManifest,
    source: CorrectionSource,
    review_path: Path,
    sample_changes: list[CorrectionChange],
    proposed_changes: list[CorrectionChange],
    understanding: list[CorrectionUnderstanding],
    proposal_path: Path,
    diff_path: Path,
    model: str,
    model_error: str | None,
    options: CorrectionEditOptions,
) -> Path:
    """Write the machine-readable proposal JSON file."""
    payload = _payload(
        paths.root, manifest, source, review_path, sample_changes, proposed_changes, understanding, proposal_path,
        diff_path, model, model_error, options
    )
    return safe_write_json(proposal_dir / f"{stem}.json", payload)


def _proposal_record(
    manifest: ProjectManifest,
    options: CorrectionEditOptions,
    review_path: Path,
    proposal_path: Path,
    diff_path: Path,
    json_path: Path,
    source: CorrectionSource,
    sample_changes: list[CorrectionChange],
    proposed_changes: list[CorrectionChange],
    understanding: list[CorrectionUnderstanding],
    model: str,
    model_error: str | None,
    project_root: Path,
) -> CorrectionProposal:
    """Build the in-memory proposal record."""
    return CorrectionProposal(
        project_id=manifest.project_id,
        category=options.category,
        review_path=review_path,
        proposal_path=proposal_path,
        diff_path=diff_path,
        json_path=json_path,
        source_path=_relative_path(project_root, source.path),
        sample_changes=sample_changes,
        proposed_changes=proposed_changes,
        understanding=understanding,
        model=model,
        model_error=model_error,
        from_original=source.from_original,
    )


def _render_markdown(
    manifest: ProjectManifest,
    understanding: list[CorrectionUnderstanding],
    sample_changes: list[CorrectionChange],
    proposed_changes: list[CorrectionChange],
    diff_path: Path,
    model: str,
    model_error: str | None,
) -> str:
    """Render a human-reviewable correction proposal."""
    lines = ["# Meeting-ASR Vocabulary Correction Proposal", "", f"Project ID: {manifest.project_id}"]
    lines.extend([f"Title: {manifest.title}", f"Model: {model}"])
    if model_error:
        lines.append(f"Model fallback: {model_error}")
    lines.extend(["", "## Understanding"])
    lines.extend(_understanding_lines(understanding))
    lines.extend(["", "## Counts", f"- User-edited samples: {len(sample_changes)}"])
    lines.append(f"- Proposed changed sentences: {len(proposed_changes)}")
    lines.extend(["", "## Diff", f"Full diff: `{diff_path}`", ""])
    lines.extend(_change_lines(proposed_changes))
    return "\n".join(lines) + "\n"


def _payload(
    project_root: Path,
    manifest: ProjectManifest,
    source: CorrectionSource,
    review_path: Path,
    sample_changes: list[CorrectionChange],
    proposed_changes: list[CorrectionChange],
    understanding: list[CorrectionUnderstanding],
    proposal_path: Path,
    diff_path: Path,
    model: str,
    model_error: str | None,
    options: CorrectionEditOptions,
) -> dict:
    """Build JSON payload for a pending correction proposal."""
    return {
        "project_id": manifest.project_id,
        "category": options.category,
        "review_path": str(_relative_path(project_root, review_path)),
        "proposal_path": str(_relative_path(project_root, proposal_path)),
        "diff_path": str(_relative_path(project_root, diff_path)),
        "source_path": str(_relative_path(project_root, source.path)),
        "from_original": source.from_original,
        "model": model,
        "model_error": model_error,
        "sample_changes": [_change_payload(change) for change in sample_changes],
        "proposed_changes": [_change_payload(change) for change in proposed_changes],
        "understanding": [asdict(item) for item in understanding],
    }


def _diff_lines(result: TranscriptResult, speaker_mapping: dict[int, str]) -> list[str]:
    """Render transcript lines suitable for unified diff."""
    return [_line(sentence, speaker_mapping) + "\n" for sentence in result.sentences]


def _line(sentence: SentenceSegment, speaker_mapping: dict[int, str]) -> str:
    """Render one transcript sentence line."""
    label = _speaker_name(sentence.speaker_id, speaker_mapping)
    return f"[{_timestamp(sentence.begin_time_ms)} - {_timestamp(sentence.end_time_ms)}] {label}: {sentence.text}"


def _understanding_lines(understanding: list[CorrectionUnderstanding]) -> list[str]:
    """Render inferred correction rules."""
    if not understanding:
        return ["- No learnable vocabulary replacement was inferred."]
    return [
        f"- `{item.wrong_text}` -> `{item.corrected_text}`; samples={item.sample_count}; "
        f"proposed={item.proposed_count}; context=`{item.left_context}__{item.right_context}`"
        for item in understanding
    ]


def _change_lines(changes: list[CorrectionChange]) -> list[str]:
    """Render proposed sentence-level changes."""
    if not changes:
        return ["No sentence changes proposed."]
    lines = ["## Proposed Changes"]
    for change in changes:
        lines.extend(["", f"### sentence_id={change.sentence_id} speaker={change.speaker_name}"])
        lines.extend([f"- Before: {change.original_text}", f"- After: {change.corrected_text}"])
    return lines


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


def _resolve_json(paths: ProjectPaths, proposal_path: Path | None) -> Path:
    """Resolve an explicit or latest proposal JSON path."""
    if proposal_path is not None:
        return proposal_path.expanduser().resolve()
    proposal_dir = paths.root / "tmp" / REVIEW_DIR
    proposals = sorted(proposal_dir.glob("proposal_*.json"))
    if not proposals:
        raise RuntimeError(f"No correction proposal found in {proposal_dir}")
    return proposals[-1]


def _changes_from_payload(value: object) -> list[CorrectionChange]:
    """Parse correction change rows from proposal JSON."""
    if not isinstance(value, list):
        return []
    return [_change_from_payload(item) for item in value if isinstance(item, dict)]


def _change_from_payload(payload: dict) -> CorrectionChange:
    """Parse one correction change from proposal JSON."""
    return CorrectionChange(
        sentence_id=_optional_int(payload.get("sentence_id")),
        speaker_id=_optional_int(payload.get("speaker_id")),
        speaker_name=str(payload.get("speaker_name") or ""),
        begin_time_ms=int(payload.get("begin_time_ms") or 0),
        end_time_ms=int(payload.get("end_time_ms") or 0),
        original_text=str(payload.get("original_text") or ""),
        corrected_text=str(payload.get("corrected_text") or ""),
        replacements=_replacements_from_payload(payload.get("replacements")),
    )


def _replacements_from_payload(value: object) -> list[CorrectionReplacement]:
    """Parse replacement rows from proposal JSON."""
    if not isinstance(value, list):
        return []
    return [_replacement_from_payload(item) for item in value if isinstance(item, dict)]


def _replacement_from_payload(payload: dict) -> CorrectionReplacement:
    """Parse one replacement row from proposal JSON."""
    return CorrectionReplacement(
        wrong_text=str(payload.get("wrong_text") or ""),
        corrected_text=str(payload.get("corrected_text") or ""),
        left_context=str(payload.get("left_context") or ""),
        right_context=str(payload.get("right_context") or ""),
    )


def _understanding_from_payload(value: object) -> list[CorrectionUnderstanding]:
    """Parse proposal understanding rows from JSON."""
    if not isinstance(value, list):
        return []
    return [_understanding_row(item) for item in value if isinstance(item, dict)]


def _understanding_row(payload: dict) -> CorrectionUnderstanding:
    """Parse one understanding row from JSON."""
    return CorrectionUnderstanding(
        wrong_text=str(payload.get("wrong_text") or ""),
        corrected_text=str(payload.get("corrected_text") or ""),
        sample_count=int(payload.get("sample_count") or 0),
        proposed_count=int(payload.get("proposed_count") or 0),
        left_context=str(payload.get("left_context") or ""),
        right_context=str(payload.get("right_context") or ""),
    )


def _project_path(project_root: Path, value: object) -> Path:
    """Resolve a project-relative path from JSON."""
    path = Path(str(value or ""))
    return path if path.is_absolute() else project_root / path


def _relative_path(project_root: Path, path: Path) -> Path:
    """Return a project-relative path when possible."""
    try:
        return path.resolve().relative_to(project_root.resolve())
    except ValueError:
        return path


def _speaker_name(speaker_id: int | None, speaker_mapping: dict[int, str]) -> str:
    """Return mapped speaker name or anonymous fallback."""
    if speaker_id is None:
        return "Speaker Unknown"
    return speaker_mapping.get(speaker_id, speaker_id_to_label(speaker_id))


def _timestamp(ms: int) -> str:
    """Format milliseconds as HH:MM:SS.mmm."""
    value = max(0, int(ms))
    hours, rem = divmod(value, 3_600_000)
    minutes, rem = divmod(rem, 60_000)
    seconds, millis = divmod(rem, 1000)
    return f"{hours:02d}:{minutes:02d}:{seconds:02d}.{millis:03d}"


def _optional_int(value: object) -> int | None:
    """Parse optional integer values from JSON."""
    if value is None or value == "":
        return None
    return int(value)


def _optional_str(value: object) -> str | None:
    """Return a stripped string or None."""
    if value is None:
        return None
    text = str(value).strip()
    return text or None
