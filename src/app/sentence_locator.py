"""Stable user-facing locators for transcript sentences."""

from __future__ import annotations

from dataclasses import dataclass
from urllib.parse import quote


@dataclass(frozen=True)
class ParsedSentenceRef:
    """A parsed sentence locator.

    ``project_id`` is optional so old bare numeric ids keep working inside an
    already selected project.
    """

    project_id: str | None
    sentence_id: int


def format_sentence_ref(project_id: str, sentence_id: int | None) -> str | None:
    """Return the copyable cross-project sentence locator."""
    if sentence_id is None:
        return None
    return f"{project_id}#{sentence_id}"


def parse_sentence_ref(value: str) -> ParsedSentenceRef:
    """Parse either ``123`` or ``project_id#123`` into locator parts."""
    raw = value.strip()
    if not raw:
        raise ValueError("Sentence locator cannot be empty.")
    if "#" in raw:
        project_id, sentence_raw = raw.split("#", 1)
        project_id = project_id.strip()
        if not project_id:
            raise ValueError("Sentence locator project id cannot be empty.")
    else:
        project_id = None
        sentence_raw = raw
    sentence_raw = sentence_raw.strip()
    if not sentence_raw:
        raise ValueError("Sentence id cannot be empty.")
    try:
        sentence_id = int(sentence_raw)
    except ValueError as exc:
        raise ValueError(f"Invalid sentence id: {sentence_raw!r}.") from exc
    if sentence_id < 0:
        raise ValueError("Sentence id must be non-negative.")
    return ParsedSentenceRef(project_id=project_id, sentence_id=sentence_id)


def sentence_review_web_path(project_id: str, sentence_id: int | None) -> str | None:
    """Return the speaker-review deep link for a sentence locator."""
    sentence_ref = format_sentence_ref(project_id, sentence_id)
    if sentence_ref is None:
        return None
    return (
        f"/projects/{quote(project_id, safe='')}/speakers"
        f"?sentence={quote(sentence_ref, safe='')}"
    )
