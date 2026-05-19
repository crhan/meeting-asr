"""Project Review voiceprint rematch workflow."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from rich.markup import escape
from textual.app import ComposeResult
from textual.screen import ModalScreen
from textual.widgets import Static

from app.core.progress import CliProgressEvent, CliProgressReporter, emit_progress
from app.presentation.tui.i18n import tr
from app.presentation.tui.speaker_models import SpeakerReviewSession
from app.presentation.tui.speaker_session import load_speaker_review_session
from app.speaker_match_status import (
    MATCH_STATUS_BELOW_THRESHOLD,
    MATCH_STATUS_MATCHED,
    MATCH_STATUS_NO_CANDIDATE,
    voiceprint_match_status,
)
from app.speaker_cluster_quality import analyze_project_speaker_clusters
from app.speaker_matching import SpeakerMatchSummary, match_project_speakers
from app.speaker_sample_matching import (
    DEFAULT_IDENTITY_AMBIGUOUS_MARGIN,
    DEFAULT_IDENTITY_CONFLICT_MARGIN,
    DEFAULT_SAMPLE_IDENTITY_THRESHOLD,
    match_project_speaker_samples,
)
from app.voiceprint_quality import DEFAULT_CRITICAL_SCORE, DEFAULT_WARNING_SCORE

DEFAULT_REVIEW_MATCH_THRESHOLD = 0.75
DEFAULT_REVIEW_MATCH_SAMPLE_COUNT = 2
DEFAULT_REVIEW_MATCH_MAX_SECONDS = 12.0
DEFAULT_REVIEW_MATCH_PADDING_SECONDS = 0.5
DEFAULT_REVIEW_CLUSTER_SAMPLE_COUNT = 40
DEFAULT_REVIEW_CLUSTER_SAME_SPEAKER_THRESHOLD = 0.60
DEFAULT_REVIEW_CLUSTER_MERGE_THRESHOLD = 0.62


@dataclass(frozen=True, slots=True)
class SpeakerRematchResult:
    """Result of refreshing one project's voiceprint diagnostics."""

    summary: SpeakerMatchSummary
    session: SpeakerReviewSession


class SpeakerRematchProcessingScreen(ModalScreen[None]):
    """Modal shown while Project Review refreshes voiceprint diagnostics."""

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

    def __init__(self) -> None:
        """Initialize mutable progress state for the refresh popup."""
        super().__init__()
        self._description = tr(
            "Preparing speaker diagnostics...",
            "正在准备 speaker 诊断...",
        )
        self._total: int | None = None
        self._completed = 0
        self._step_index: int | None = None
        self._step_total: int | None = None

    def compose(self) -> ComposeResult:
        """Build the processing popup."""
        yield Static(self._render_progress(), id="speaker-rematch-processing")

    def update_progress(self, event: CliProgressEvent) -> None:
        """
        Update the processing popup from a background refresh event.

        Args:
            event: Progress event emitted by the refresh workflow.
        """
        if event.step_index is not None:
            self._step_index = event.step_index
        if event.step_total is not None:
            self._step_total = event.step_total
        if event.reset_total:
            self._total = None
            self._completed = 0
        if event.total is not None:
            self._total = max(0, event.total)
        if event.completed is not None:
            self._completed = max(0, event.completed)
        if event.advance:
            self._completed = max(0, self._completed + event.advance)
        if event.description:
            self._description = event.description
        try:
            self.query_one("#speaker-rematch-processing", Static).update(self._render_progress())
        except Exception:  # noqa: BLE001
            return

    def _render_progress(self) -> str:
        """Render the current refresh progress."""
        step = ""
        if self._step_index is not None and self._step_total is not None:
            step = tr(
                f"Step {self._step_index}/{self._step_total}\n",
                f"步骤 {self._step_index}/{self._step_total}\n",
            )
        bar = _progress_bar(self._completed, self._total)
        return tr(
            "[b]Refreshing speaker diagnostics[/b]\n\n"
            "Updating voiceprint matches, cluster scores, and per-sample identity scores.\n\n"
            f"{step}{escape(self._description)}\n{bar}\n\n"
            "The review screen will refresh after diagnostics finish.",
            "[b]正在刷新 speaker 诊断[/b]\n\n"
            "正在更新声纹匹配、分桶分数和逐句身份分数。\n\n"
            f"{step}{escape(self._description)}\n{bar}\n\n"
            "诊断完成后会刷新 review 页面。",
        )


def run_speaker_rematch(
    project_dir: Path,
    *,
    store_dir: Path | None,
    page_size: int | None,
    allow_correction: bool,
    progress: CliProgressReporter | None = None,
) -> SpeakerRematchResult:
    """
    Refresh project speaker diagnostics and reload the Project Review session.

    Args:
        project_dir: Project root.
        store_dir: Optional voiceprint store directory.
        page_size: Optional review page size.
        allow_correction: Whether transcript correction remains enabled.
        progress: Optional progress reporter for the TUI popup.

    Returns:
        Persisted match summary and freshly loaded review session.
    """
    emit_progress(
        progress,
        "Matching speaker identities",
        step_index=1,
        step_total=3,
        reset_total=True,
    )
    summary = match_project_speakers(
        project_dir,
        store_dir=store_dir,
        provider=None,
        model=None,
        threshold=DEFAULT_REVIEW_MATCH_THRESHOLD,
        sample_count=DEFAULT_REVIEW_MATCH_SAMPLE_COUNT,
        max_seconds=DEFAULT_REVIEW_MATCH_MAX_SECONDS,
        padding_seconds=DEFAULT_REVIEW_MATCH_PADDING_SECONDS,
        progress=progress,
    )
    emit_progress(
        progress,
        "Scoring speaker clusters",
        step_index=2,
        step_total=3,
        reset_total=True,
    )
    analyze_project_speaker_clusters(
        project_dir,
        provider=None,
        model=None,
        sample_count=DEFAULT_REVIEW_CLUSTER_SAMPLE_COUNT,
        max_seconds=DEFAULT_REVIEW_MATCH_MAX_SECONDS,
        padding_seconds=DEFAULT_REVIEW_MATCH_PADDING_SECONDS,
        score_all_segments=True,
        same_speaker_threshold=DEFAULT_REVIEW_CLUSTER_SAME_SPEAKER_THRESHOLD,
        merge_speaker_threshold=DEFAULT_REVIEW_CLUSTER_MERGE_THRESHOLD,
        warning_score=DEFAULT_WARNING_SCORE,
        critical_score=DEFAULT_CRITICAL_SCORE,
        write_report=True,
        progress=progress,
    )
    emit_progress(
        progress,
        "Matching sample identities",
        step_index=3,
        step_total=3,
        reset_total=True,
    )
    match_project_speaker_samples(
        project_dir,
        store_dir=store_dir,
        provider=None,
        model=None,
        threshold=DEFAULT_SAMPLE_IDENTITY_THRESHOLD,
        conflict_margin=DEFAULT_IDENTITY_CONFLICT_MARGIN,
        ambiguous_margin=DEFAULT_IDENTITY_AMBIGUOUS_MARGIN,
        max_seconds=DEFAULT_REVIEW_MATCH_MAX_SECONDS,
        padding_seconds=DEFAULT_REVIEW_MATCH_PADDING_SECONDS,
        write_report=True,
        progress=progress,
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
    Render a one-line diagnostics summary for the Project Review status bar.

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
        f"no-candidate {no_candidate}, threshold {threshold:.3f}; diagnostics refreshed.",
        f"重新匹配完成：已接受 {matched}/{total}，低于阈值 {below}，无候选 {no_candidate}，阈值 {threshold:.3f}；诊断已刷新。",
    )


def _status_count(summary: SpeakerMatchSummary, status: str) -> int:
    """Count matches with one voiceprint status."""
    return sum(1 for match in summary.matches if voiceprint_match_status(match) == status)


def _progress_bar(completed: int, total: int | None, *, width: int = 30) -> str:
    """Render a compact ASCII progress bar."""
    if total is None or total <= 0:
        return tr("[dim]Working...[/]", "[dim]处理中...[/]")
    bounded_completed = min(max(completed, 0), total)
    ratio = bounded_completed / total
    filled = min(width, int(round(width * ratio)))
    empty = width - filled
    percent = int(round(ratio * 100))
    return f"[green]{'=' * filled}[/][dim]{'-' * empty}[/] {bounded_completed}/{total} {percent}%"
