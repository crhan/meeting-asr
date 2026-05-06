"""Workflow helpers for embedded Voiceprint Review."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import shutil
import tempfile

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
class VoiceprintReviewTransaction:
    """Filesystem snapshot used to accept or roll back a voiceprint workflow."""

    backup_dir: Path
    db_path: Path
    db_backup_path: Path
    db_existed: bool
    project_manifest_path: Path
    project_manifest_backup_path: Path
    project_manifest_existed: bool
    match_path: Path
    match_backup_path: Path
    match_existed: bool
    clip_backups: tuple[tuple[Path, Path, bool], ...]

    def accept(self) -> None:
        """Accept pending changes by removing the rollback snapshot."""
        shutil.rmtree(self.backup_dir, ignore_errors=True)

    def rollback(self) -> None:
        """Restore database, project metadata, match file, and captured clips."""
        _restore_file(self.db_path, self.db_backup_path, self.db_existed)
        _restore_file(self.project_manifest_path, self.project_manifest_backup_path, self.project_manifest_existed)
        _restore_file(self.match_path, self.match_backup_path, self.match_existed)
        for clip_path, backup_path, existed in self.clip_backups:
            _restore_file(clip_path, backup_path, existed)
            _remove_empty_parents(clip_path.parent, stop_at=self.db_path.parent)
        shutil.rmtree(self.backup_dir, ignore_errors=True)


@dataclass(frozen=True, slots=True)
class VoiceprintReviewWorkflowSummary:
    """Result of saving project samples from embedded Voiceprint Review."""

    capture: VoiceprintCaptureSummary
    embedding: VoiceprintEmbedSummary
    evaluation: VoiceprintEvaluationSummary
    transaction: VoiceprintReviewTransaction


class VoiceprintReviewProcessingScreen(ModalScreen[None]):
    """Modal screen shown while voiceprint capture and embedding are running."""

    CSS = """
    VoiceprintReviewProcessingScreen {
        align: center middle;
    }
    #voiceprint-review-processing {
        width: 84;
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
                "[b]Processing voiceprints[/b]\n\nCapturing samples, generating embeddings, and evaluating score impact.\nThis can take a while.",
                "[b]正在处理声纹[/b]\n\n正在采集样本、生成 embedding，并评测分数影响。\n这个过程可能需要一些时间。",
            ),
            id="voiceprint-review-processing",
        )


class VoiceprintReviewResultScreen(ModalScreen[bool]):
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
        Binding("a", "accept_result", "Accept"),
        Binding("r", "rollback_result", "Rollback"),
        Binding("escape", "rollback_result", "Rollback", show=False),
        Binding("q", "rollback_result", "Rollback", show=False),
    ]

    def __init__(self, summary: VoiceprintReviewWorkflowSummary) -> None:
        """
        Create a workflow result modal.

        Args:
            summary: Capture, embedding, and evaluation result.
        """
        super().__init__()
        self.summary = summary
        self.decided = False

    def compose(self) -> ComposeResult:
        """Build the result popup."""
        yield Static(_workflow_summary_text(self.summary), id="voiceprint-review-result")

    def on_unmount(self) -> None:
        """Roll back if the modal is closed without an explicit choice."""
        if not self.decided:
            self.summary.transaction.rollback()

    def action_accept_result(self) -> None:
        """Accept the pending voiceprint changes."""
        self.decided = True
        self.summary.transaction.accept()
        self.dismiss(True)

    def action_rollback_result(self) -> None:
        """Reject and roll back the pending voiceprint changes."""
        self.decided = True
        self.summary.transaction.rollback()
        self.dismiss(False)


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
    transaction = _begin_transaction(project_dir, planned, selected_clip_rel_paths)
    try:
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
    except Exception:
        transaction.rollback()
        raise
    return VoiceprintReviewWorkflowSummary(capture, embedding, evaluation, transaction)


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
        tr("Press a to accept these embeddings. Press r/Esc/q to roll back.", "按 a 接受这些 embedding。按 r/Esc/q 回滚。"),
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
    risk_style = "bold red" if summary.historical_risk_count else "green"
    lines.append(
        tr(
            f"Checked {summary.historical_project_count} project(s); [{risk_style}]risky changes {summary.historical_risk_count}[/].",
            f"检查 {summary.historical_project_count} 个历史项目；[{risk_style}]风险变化 {summary.historical_risk_count} 个[/]。",
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
        project_id = escape(project.project_id)
        lines.append(f"  [bold red]RISK[/] {project_id} | [red]{escape(title)}[/] | [bold red]risk={project.risk_count}[/]")
        lines.append(f"    [red]review: meeting-asr project review {project_id}[/]")
        lines.extend(
            "    " + _score_change_line(change, risk=True)
            for change in project.changes
            if change.status in {"declined", "changed-best"}
        )
    return lines


def _score_change_line(change, *, risk: bool = False) -> str:
    """Render one before/after score line."""
    before = _candidate_score_text(change.before_name, change.before_score)
    after = _candidate_score_text(change.after_name, change.after_score)
    delta_text = "" if change.delta is None else f" ({change.delta:+.3f})"
    status_text = "" if not risk else f" [bold red]{escape(change.status)}[/]"
    line = f"{escape(change.label)}: {before} -> {after}{delta_text}{status_text}"
    if risk:
        return f"[red]{line}[/]"
    return line


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


def _begin_transaction(
    project_dir: Path,
    planned: VoiceprintCaptureSummary,
    selected_clip_rel_paths: frozenset[str],
) -> VoiceprintReviewTransaction:
    """Create rollback snapshots before mutating the voiceprint store."""
    backup_dir = Path(tempfile.mkdtemp(prefix="meeting-asr-voiceprint-review-"))
    project_root = project_dir.expanduser().resolve()
    return VoiceprintReviewTransaction(
        backup_dir=backup_dir,
        db_path=planned.db_path,
        db_backup_path=backup_dir / "voiceprints.sqlite",
        db_existed=_backup_file(planned.db_path, backup_dir / "voiceprints.sqlite"),
        project_manifest_path=project_root / "project.json",
        project_manifest_backup_path=backup_dir / "project.json",
        project_manifest_existed=_backup_file(project_root / "project.json", backup_dir / "project.json"),
        match_path=project_root / "speakers" / "speaker_matches.json",
        match_backup_path=backup_dir / "speaker_matches.json",
        match_existed=_backup_file(project_root / "speakers" / "speaker_matches.json", backup_dir / "speaker_matches.json"),
        clip_backups=_backup_clips(backup_dir, planned, selected_clip_rel_paths),
    )


def _backup_clips(
    backup_dir: Path,
    planned: VoiceprintCaptureSummary,
    selected_clip_rel_paths: frozenset[str],
) -> tuple[tuple[Path, Path, bool], ...]:
    """Back up planned clip targets that may be overwritten."""
    backups: list[tuple[Path, Path, bool]] = []
    for speaker in planned.speakers:
        for clip in speaker.clips:
            if clip.rel_path in selected_clip_rel_paths:
                backup_path = backup_dir / "clips" / clip.rel_path
                backups.append((clip.path, backup_path, _backup_file(clip.path, backup_path)))
    return tuple(backups)


def _backup_file(path: Path, backup_path: Path) -> bool:
    """Copy an existing file to its backup path."""
    if not path.exists():
        return False
    backup_path.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(path, backup_path)
    return True


def _restore_file(path: Path, backup_path: Path, existed: bool) -> None:
    """Restore a file to its previous state."""
    if existed:
        path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(backup_path, path)
        return
    path.unlink(missing_ok=True)


def _remove_empty_parents(path: Path, *, stop_at: Path) -> None:
    """Remove empty clip directories created by a rolled-back workflow."""
    current = path
    stop = stop_at.expanduser().resolve()
    while current.exists() and current.resolve() != stop:
        try:
            current.rmdir()
        except OSError:
            return
        current = current.parent
