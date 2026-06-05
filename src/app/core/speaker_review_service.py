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
    SentenceReassignmentApplyResult,
    apply_project_sentence_reassignments,
)
from app.speaker_labeling import SentenceReassignmentSpec


@dataclass(frozen=True, slots=True)
class SpeakerReviewSaveResult:
    """Files written and side effects of one speaker-review save."""

    mapping_path: Path
    transcript_path: Path
    srt_path: Path
    reassignment: SentenceReassignmentApplyResult | None


def save_speaker_review(
    project_dir: Path,
    *,
    mapping: dict[int, str],
    person_mapping: dict[int, int] | None = None,
    person_public_mapping: dict[int, str] | None = None,
    ignored_speaker_ids: Collection[int] = (),
    reassignments: Sequence[SentenceReassignmentSpec] = (),
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
        ignored_speaker_ids: Speaker ids deliberately kept anonymous.
        reassignments: Sentence reassignment specs to apply before naming.
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
    mapping_path, transcript_path, srt_path = apply_project_speakers(
        project_dir,
        mapping,
        person_mapping=person_mapping,
        person_public_mapping=person_public_mapping,
        ignored_speaker_ids=ignored_speaker_ids or None,
    )
    return SpeakerReviewSaveResult(
        mapping_path=mapping_path,
        transcript_path=transcript_path,
        srt_path=srt_path,
        reassignment=reassignment_result,
    )
