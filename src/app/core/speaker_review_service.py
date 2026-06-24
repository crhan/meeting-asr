"""Presentation-neutral speaker-review save orchestration.

The Textual TUI and the web UI both let a user rename speakers, bind them to voiceprint
people, ignore them, and reassign individual sentences -- then save. The *save* step has
sharp edges that must not be reimplemented per front-end:

* ``apply_project_speakers`` merges (not replaces) the saved speaker map and preserves
  person-map entries only when a name is unchanged.
* ``apply_project_sentence_reassignments`` rewrites the sentence files, regenerates the
  anonymous transcript, **deletes overlapping samples from the global voiceprint store**,
  and reruns matching.

This module is the single shared entry point for that sequence. It takes primitives (not
the TUI ``SpeakerReviewDecision``) so ``app.core`` stays free of any presentation import;
each front-end adapts its own decision into these arguments.
"""

from __future__ import annotations

from collections.abc import Collection, Sequence
from dataclasses import dataclass
from pathlib import Path

from app.project_manager import apply_project_speakers
from app.sentence_reassignment import (
    EmptySpeakerDeletionApplyResult,
    SentenceReassignmentApplyResult,
    apply_project_empty_speaker_deletions,
    apply_project_sentence_reassignments,
)
from app.speaker_labeling import SentenceReassignmentSpec
from app.voiceprint_people import create_voiceprint_person, get_voiceprint_person
from app.voiceprint_store import get_voiceprint_db_path


@dataclass(frozen=True, slots=True)
class SpeakerReviewSaveResult:
    """Files written and side effects of one speaker-review save."""

    mapping_path: Path
    transcript_path: Path
    srt_path: Path
    reassignment: SentenceReassignmentApplyResult | None
    deletion: EmptySpeakerDeletionApplyResult | None
    created_person_count: int


def save_speaker_review(
    project_dir: Path,
    *,
    mapping: dict[int, str],
    person_mapping: dict[int, int] | None = None,
    person_public_mapping: dict[int, str] | None = None,
    new_person_names: dict[int, str] | None = None,
    ignored_speaker_ids: Collection[int] = (),
    reassignments: Sequence[SentenceReassignmentSpec] = (),
    deleted_speaker_ids: Collection[int] = (),
    store_dir: Path | None = None,
    rematch: bool = True,
) -> SpeakerReviewSaveResult:
    """Persist speaker names, person bindings, ignore flags, and sentence reassignments.

    Reassignments are applied first (they rewrite sentence files and rerun matching), then
    the speaker map is applied and named outputs are regenerated -- the same order the CLI
    review save has always used.

    Args:
        project_dir: Project root directory.
        mapping: ``{speaker_id: name}`` to merge into the saved speaker map.
        person_mapping: ``{speaker_id: voiceprint person id}`` bindings.
        person_public_mapping: ``{speaker_id: voiceprint person public id}`` bindings.
        new_person_names: ``{speaker_id: name}`` entries that should create or bind a
            stable voiceprint person at save time.
        ignored_speaker_ids: Speaker ids deliberately kept anonymous.
        reassignments: Sentence reassignment specs to apply before naming.
        deleted_speaker_ids: Speaker ids to remove after every visible sentence is empty
            or reassigned away.
        store_dir: Voiceprint store directory; ``None`` resolves to the default.
        rematch: Whether to rerun voiceprint matching after reassignment invalidation.

    Returns:
        The written mapping/transcript/SRT paths and the reassignment apply result.
    """
    reassignment_result: SentenceReassignmentApplyResult | None = None
    if reassignments:
        reassignment_result = apply_project_sentence_reassignments(
            project_dir,
            list(reassignments),
            store_dir=store_dir,
            rematch=rematch,
        )
    deletion_result: EmptySpeakerDeletionApplyResult | None = None
    deleted_ids = {int(speaker_id) for speaker_id in deleted_speaker_ids}
    if deleted_ids:
        deletion_result = apply_project_empty_speaker_deletions(
            project_dir, sorted(deleted_ids)
        )
    if deleted_ids:
        mapping = {
            speaker_id: name
            for speaker_id, name in mapping.items()
            if int(speaker_id) not in deleted_ids
        }
        person_mapping = {
            speaker_id: person_id
            for speaker_id, person_id in (person_mapping or {}).items()
            if int(speaker_id) not in deleted_ids
        }
        person_public_mapping = {
            speaker_id: person_public_id
            for speaker_id, person_public_id in (person_public_mapping or {}).items()
            if int(speaker_id) not in deleted_ids
        }
        ignored_speaker_ids = [
            speaker_id
            for speaker_id in ignored_speaker_ids
            if int(speaker_id) not in deleted_ids
        ]
        new_person_names = {
            speaker_id: name
            for speaker_id, name in (new_person_names or {}).items()
            if int(speaker_id) not in deleted_ids
        }
    new_person_result = _resolve_new_person_bindings(
        mapping,
        person_mapping or {},
        person_public_mapping or {},
        new_person_names or {},
        store_dir=store_dir,
    )
    mapping_path, transcript_path, srt_path = apply_project_speakers(
        project_dir,
        new_person_result.mapping,
        person_mapping=new_person_result.person_mapping,
        person_public_mapping=new_person_result.person_public_mapping,
        # Pass the ignore set straight through, even when empty: apply_project_speakers
        # treats None as "leave the existing ignore state untouched" but an empty
        # collection as "no speakers are ignored" -- which clears a stale
        # speaker_ignore.json. `or None` would collapse the empty case into None and strand
        # the last un-ignored speaker as anonymous. (matches origin/main CLI save behavior)
        ignored_speaker_ids=ignored_speaker_ids,
    )
    return SpeakerReviewSaveResult(
        mapping_path=mapping_path,
        transcript_path=transcript_path,
        srt_path=srt_path,
        reassignment=reassignment_result,
        deletion=deletion_result,
        created_person_count=new_person_result.created_count,
    )


@dataclass(frozen=True, slots=True)
class _NewPersonBindingResult:
    """Resolved speaker mappings after creating requested voiceprint people."""

    mapping: dict[int, str]
    person_mapping: dict[int, int]
    person_public_mapping: dict[int, str]
    created_count: int


def _resolve_new_person_bindings(
    mapping: dict[int, str],
    person_mapping: dict[int, int],
    person_public_mapping: dict[int, str],
    new_person_names: dict[int, str],
    *,
    store_dir: Path | None,
) -> _NewPersonBindingResult:
    """Create missing people requested by the UI and bind speakers to their public ids."""
    if not new_person_names:
        return _NewPersonBindingResult(
            mapping=dict(mapping),
            person_mapping=dict(person_mapping),
            person_public_mapping=dict(person_public_mapping),
            created_count=0,
        )
    resolved_mapping = dict(mapping)
    resolved_person_mapping = dict(person_mapping)
    resolved_public_mapping = dict(person_public_mapping)
    db_path = get_voiceprint_db_path(store_dir)
    created_count = 0
    for raw_speaker_id, raw_name in sorted(new_person_names.items()):
        speaker_id = int(raw_speaker_id)
        name = raw_name.strip()
        if not name:
            raise ValueError(f"New person name is empty for speaker {speaker_id}.")
        person = get_voiceprint_person(name, db_path)
        if person is None:
            person = create_voiceprint_person(name, db_path)
            created_count += 1
        resolved_mapping[speaker_id] = person.name
        resolved_person_mapping.pop(speaker_id, None)
        resolved_public_mapping[speaker_id] = person.public_id
    return _NewPersonBindingResult(
        mapping=resolved_mapping,
        person_mapping=resolved_person_mapping,
        person_public_mapping=resolved_public_mapping,
        created_count=created_count,
    )
