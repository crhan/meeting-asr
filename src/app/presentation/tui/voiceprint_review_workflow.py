"""Workflow helpers for embedded Voiceprint Review."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from rich.markup import escape
from textual.app import ComposeResult
from textual.binding import Binding
from textual.screen import ModalScreen
from textual.widgets import Static

from app.presentation.tui.i18n import tr
from app.voiceprint_embedding import VoiceprintEmbedSummary, embed_voiceprint_samples
from app.voiceprint_evaluation import VoiceprintEvaluationSummary, evaluate_voiceprint_embedding
from app.voiceprints import VoiceprintCaptureSummary, persist_voiceprint_capture_selection


@dataclass(frozen=True, slots=True)
class VoiceprintReviewWorkflowSummary:
    """Result of saving project samples from embedded Voiceprint Review."""

    capture: VoiceprintCaptureSummary
    embedding: VoiceprintEmbedSummary
    evaluation: VoiceprintEvaluationSummary


class VoiceprintReviewResultScreen(ModalScreen[None]):
    """Modal result view for capture, embedding, and evaluation."""

    CSS = """
    VoiceprintReviewResultScreen {
        align: center middle;
    }
    #voiceprint-review-result {
        width: 118;
        max-height: 34;
        border: thick $accent;
        padding: 1 2;
        background: $surface;
    }
    """

    BINDINGS = [
        Binding("escape", "close_result", "Close", show=False),
        Binding("enter", "close_result", "Close"),
        Binding("q", "close_result", "Close", show=False),
    ]

    def __init__(self, summary: VoiceprintReviewWorkflowSummary) -> None:
        """
        Create a workflow result modal.

        Args:
            summary: Capture, embedding, and evaluation result.
        """
        super().__init__()
        self.summary = summary

    def compose(self) -> ComposeResult:
        """Build the result popup."""
        yield Static(_workflow_summary_text(self.summary), id="voiceprint-review-result")

    def action_close_result(self) -> None:
        """Close the result popup."""
        self.dismiss(None)


def run_voiceprint_review_workflow(
    *,
    project_dir: Path,
    planned: VoiceprintCaptureSummary,
    selected_clip_rel_paths: frozenset[str],
    store_dir: Path | None,
) -> VoiceprintReviewWorkflowSummary:
    """
    Capture selected clips, embed them, and evaluate matching scores.

    Args:
        project_dir: Project root.
        planned: Dry-run capture plan.
        selected_clip_rel_paths: Accepted clip relative paths.
        store_dir: Optional voiceprint store directory.

    Returns:
        Completed workflow summary.
    """
    capture = persist_voiceprint_capture_selection(
        project_dir,
        planned=planned,
        selected_clip_rel_paths=selected_clip_rel_paths,
    )
    embedding = embed_voiceprint_samples(
        store_dir=store_dir or capture.store_dir,
        provider=None,
        endpoint=None,
        model=None,
        rebuild=False,
    )
    evaluation = evaluate_voiceprint_embedding(
        project_dir,
        store_dir=store_dir or capture.store_dir,
        provider=None,
        endpoint=None,
        model=embedding.model,
    )
    return VoiceprintReviewWorkflowSummary(capture, embedding, evaluation)


def compact_workflow_line(summary: VoiceprintReviewWorkflowSummary) -> str:
    """Render one-line workflow result for overview/status panes."""
    return tr(
        (
            f"[b]Result[/b]   captured {summary.capture.sample_count}, embedded {summary.embedding.embedded_count}, "
            f"historical risks {summary.evaluation.historical_risk_count}"
        ),
        (
            f"[b]结果[/b]     采集 {summary.capture.sample_count}，embedding 新增 {summary.embedding.embedded_count}，"
            f"历史风险 {summary.evaluation.historical_risk_count}"
        ),
    )


def compact_evaluation_line(summary: VoiceprintEvaluationSummary) -> str:
    """Render one-line evaluation result."""
    return tr(
        (
            f"Evaluation: current improved {summary.current.improved_count}, declined {summary.current.declined_count}; "
            f"historical risks {summary.historical_risk_count}"
        ),
        (
            f"评测：当前项目提升 {summary.current.improved_count}，下降 {summary.current.declined_count}；"
            f"历史风险 {summary.historical_risk_count}"
        ),
    )


def _workflow_summary_text(summary: VoiceprintReviewWorkflowSummary) -> str:
    """Render capture, embedding, and evaluation details."""
    lines = [
        tr("[b green]Voiceprint embedding complete[/b green]", "[b green]声纹 embedding 已完成[/b green]"),
        "",
        tr(f"Captured samples: {summary.capture.sample_count}", f"已采集样本：{summary.capture.sample_count}"),
        tr(
            f"Embedded samples: {summary.embedding.embedded_count} new, {summary.embedding.skipped_count} skipped",
            f"Embedding：新增 {summary.embedding.embedded_count} 个，跳过 {summary.embedding.skipped_count} 个",
        ),
        "",
        _current_evaluation_text(summary.evaluation),
        "",
        _historical_evaluation_text(summary.evaluation),
        "",
        tr(
            "Press Enter/Esc/q to close this result; use Esc/q again to return to Project Review.",
            "按 Enter/Esc/q 关闭结果；再按 Esc/q 返回 Project Review。",
        ),
    ]
    return "\n".join(lines)


def _current_evaluation_text(summary: VoiceprintEvaluationSummary) -> str:
    """Render current-project score changes."""
    current = summary.current
    lines = [tr("[b]Current project score check[/b]", "[b]当前项目分数检查[/b]")]
    if not current.changes:
        lines.append(tr("No speaker matches were available.", "没有可用的 speaker 匹配结果。"))
        return "\n".join(lines)
    for change in current.changes[:6]:
        lines.append("  " + _score_change_line(change))
    return "\n".join(lines)


def _historical_evaluation_text(summary: VoiceprintEvaluationSummary) -> str:
    """Render historical regression summary."""
    lines = [tr("[b]Historical reverse check[/b]", "[b]历史项目反向评测[/b]")]
    lines.append(
        tr(
            f"Checked {summary.historical_project_count} project(s); risky changes {summary.historical_risk_count}.",
            f"检查 {summary.historical_project_count} 个历史项目；风险变化 {summary.historical_risk_count} 个。",
        )
    )
    lines.extend(_historical_risk_lines(summary))
    return "\n".join(lines)


def _historical_risk_lines(summary: VoiceprintEvaluationSummary) -> list[str]:
    """Render risky historical project lines."""
    lines: list[str] = []
    for project in summary.historical:
        if project.risk_count == 0:
            continue
        title = _trim_text(project.title or project.project_id, limit=40)
        lines.append(f"  [yellow]{escape(title)}[/] risk={project.risk_count}")
        lines.extend("    " + _score_change_line(change) for change in project.changes if change.status in {"declined", "changed-best"})
    return lines


def _score_change_line(change) -> str:
    """Render one before/after score line."""
    before = _candidate_score_text(change.before_name, change.before_score)
    after = _candidate_score_text(change.after_name, change.after_score)
    delta_text = "" if change.delta is None else f" ({change.delta:+.3f})"
    return f"{escape(change.label)}: {before} -> {after}{delta_text}"


def _candidate_score_text(name: str | None, score: float | None) -> str:
    """Render one candidate score."""
    if name is None:
        return "no-candidate"
    if score is None:
        return escape(name)
    return f"{escape(name)} {score:.3f}"


def _trim_text(text: str, *, limit: int) -> str:
    """Trim text for modal display."""
    preview = text.strip().replace("\n", " ")
    if len(preview) <= limit:
        return preview
    return preview[: limit - 3] + "..."
