"""Project Review voiceprint rematch workflow."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from textual.app import ComposeResult
from textual.screen import ModalScreen
from textual.widgets import Static

from app.presentation.tui.i18n import tr
from app.presentation.tui.speaker_models import SpeakerReviewSession
from app.presentation.tui.speaker_session import load_speaker_review_session
from app.speaker_match_status import (
    MATCH_STATUS_BELOW_THRESHOLD,
    MATCH_STATUS_MATCHED,
    MATCH_STATUS_NO_CANDIDATE,
    voiceprint_match_status,
)
from app.speaker_matching import SpeakerMatchSummary, match_project_speakers

DEFAULT_REVIEW_MATCH_THRESHOLD = 0.75
DEFAULT_REVIEW_MATCH_SAMPLE_COUNT = 2
DEFAULT_REVIEW_MATCH_MAX_SECONDS = 12.0
DEFAULT_REVIEW_MATCH_PADDING_SECONDS = 0.5


@dataclass(frozen=True, slots=True)
class SpeakerRematchResult:
    """Result of rematching one project and reloading its review session."""

    summary: SpeakerMatchSummary
    session: SpeakerReviewSession


class SpeakerRematchProcessingScreen(ModalScreen[None]):
    """Modal shown while Project Review reruns voiceprint matching."""

    CSS = """
    SpeakerRematchProcessingScreen {
        align: center middle;
    }
    #speaker-rematch-processing {
        width: 82;
        height: auto;
        border: thick $accent;
        padding: 1 2;
        background: $surface;
    }
    """

    def compose(self) -> ComposeResult:
        """Build the processing popup."""
        yield Static(
            tr(
                "[b]Rematching speakers[/b]\n\n"
                "Comparing this project against the current global voiceprint library.\n"
                "The review screen will refresh after matching finishes.",
                "[b]正在重新匹配 speaker[/b]\n\n"
                "正在用当前全局声纹库重新匹配这个项目。\n"
                "匹配完成后会刷新 review 页面。",
            ),
            id="speaker-rematch-processing",
        )


def run_speaker_rematch(
    project_dir: Path,
    *,
    store_dir: Path | None,
    page_size: int | None,
    allow_correction: bool,
) -> SpeakerRematchResult:
    """
    Rematch project speakers and reload the Project Review session.

    Args:
        project_dir: Project root.
        store_dir: Optional voiceprint store directory.
        page_size: Optional review page size.
        allow_correction: Whether transcript correction remains enabled.

    Returns:
        Persisted match summary and freshly loaded review session.
    """
    summary = match_project_speakers(
        project_dir,
        store_dir=store_dir,
        provider=None,
        endpoint=None,
        model=None,
        threshold=DEFAULT_REVIEW_MATCH_THRESHOLD,
        sample_count=DEFAULT_REVIEW_MATCH_SAMPLE_COUNT,
        max_seconds=DEFAULT_REVIEW_MATCH_MAX_SECONDS,
        padding_seconds=DEFAULT_REVIEW_MATCH_PADDING_SECONDS,
        progress=None,
    )
    session = load_speaker_review_session(
        project_dir,
        page_size=page_size,
        store_dir=store_dir,
        allow_correction=allow_correction,
    )
    return SpeakerRematchResult(summary=summary, session=session)


def compact_rematch_line(result: SpeakerRematchResult) -> str:
    """
    Render a one-line rematch summary for the Project Review status bar.

    Args:
        result: Completed rematch result.

    Returns:
        Localized status text.
    """
    matched = _status_count(result.summary, MATCH_STATUS_MATCHED)
    below = _status_count(result.summary, MATCH_STATUS_BELOW_THRESHOLD)
    no_candidate = _status_count(result.summary, MATCH_STATUS_NO_CANDIDATE)
    total = len(result.summary.matches)
    threshold = result.summary.threshold
    return tr(
        f"Rematch complete: matched {matched}/{total}, below-threshold {below}, "
        f"no-candidate {no_candidate}, threshold {threshold:.3f}.",
        f"重新匹配完成：已接受 {matched}/{total}，低于阈值 {below}，无候选 {no_candidate}，阈值 {threshold:.3f}。",
    )


def _status_count(summary: SpeakerMatchSummary, status: str) -> int:
    """Count matches with one voiceprint status."""
    return sum(1 for match in summary.matches if voiceprint_match_status(match) == status)
