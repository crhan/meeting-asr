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
from app.voiceprint_quality import (
    VOICEPRINT_SAMPLE_STATUS_VERIFIED_ACTIVE,
    analyze_voiceprint_quality,
)
from app.voiceprint_store import (
    get_voiceprint_db_path,
    list_voiceprint_samples,
    update_voiceprint_sample_status,
)
from app.voiceprint_evaluation import (
    VoiceprintEvaluationSummary,
    evaluate_voiceprint_embedding,
)
from app.voiceprints import (
    VoiceprintCaptureSummary,
    persist_voiceprint_capture_selection,
)

CURRENT_CHANGE_DISPLAY_LIMIT = 6
HISTORICAL_PROJECT_DISPLAY_LIMIT = 3
HISTORICAL_CHANGE_DISPLAY_LIMIT = 3


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
        _restore_file(
            self.project_manifest_path,
            self.project_manifest_backup_path,
            self.project_manifest_existed,
        )
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
    quality_gate: VoiceprintQualityGateSummary


@dataclass(frozen=True, slots=True)
class VoiceprintQualityGateSummary:
    """Automatic post-capture sample quality gate result."""

    reviewed_sample_count: int = 0
    excluded_sample_count: int = 0
    warning_sample_count: int = 0
    critical_sample_count: int = 0


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
        yield Static(
            _workflow_summary_text(self.summary), id="voiceprint-review-result"
        )

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
            model=None,
            rebuild=False,
        )
        quality_gate = _apply_capture_quality_gate(
            capture, store_dir=store_dir or capture.store_dir, model=embedding.model
        )
        evaluation = evaluate_voiceprint_embedding(
            project_dir,
            store_dir=store_dir or capture.store_dir,
            provider=None,
            model=embedding.model,
        )
    except Exception:
        transaction.rollback()
        raise
    return VoiceprintReviewWorkflowSummary(
        capture, embedding, evaluation, transaction, quality_gate
    )


def compact_workflow_line(summary: VoiceprintReviewWorkflowSummary) -> str:
    """Render one-line workflow result for overview/status panes."""
    return tr(
        (
            f"[b]Result[/b]   captured {summary.capture.sample_count}, embedded {summary.embedding.embedded_count}, "
            f"excluded {summary.quality_gate.excluded_sample_count}, "
            f"historical risks {summary.evaluation.historical_risk_count}"
        ),
        (
            f"[b]结果[/b]     采集 {summary.capture.sample_count}，embedding 新增 {summary.embedding.embedded_count}，"
            f"自动排除 {summary.quality_gate.excluded_sample_count}，"
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


def _apply_capture_quality_gate(
    capture: VoiceprintCaptureSummary, *, store_dir: Path, model: str
) -> VoiceprintQualityGateSummary:
    """Mark reviewed capture samples and exclude low-quality ones from matching."""
    db_path = get_voiceprint_db_path(store_dir)
    captured_ids = _captured_sample_public_ids(capture, db_path)
    if not captured_ids:
        return VoiceprintQualityGateSummary()

    for sample_id in captured_ids:
        update_voiceprint_sample_status(
            sample_id, VOICEPRINT_SAMPLE_STATUS_VERIFIED_ACTIVE, db_path
        )

    report = analyze_voiceprint_quality(store_dir=store_dir, model=model)
    warning_count = 0
    critical_count = 0
    excluded: set[str] = set()
    for person in report.people:
        for sample in person.samples:
            if sample.sample_public_id not in captured_ids:
                continue
            if sample.label == "critical":
                critical_count += 1
            elif sample.label == "warning":
                warning_count += 1
            else:
                continue
            update_voiceprint_sample_status(
                sample.sample_public_id, "verified-quarantined", db_path
            )
            excluded.add(sample.sample_public_id)

    return VoiceprintQualityGateSummary(
        reviewed_sample_count=len(captured_ids),
        excluded_sample_count=len(excluded),
        warning_sample_count=warning_count,
        critical_sample_count=critical_count,
    )


def _captured_sample_public_ids(
    capture: VoiceprintCaptureSummary, db_path: Path
) -> set[str]:
    """Resolve stable sample ids for the clips written by this capture run."""
    captured: set[str] = set()
    for speaker in capture.speakers:
        rel_paths = {clip.rel_path for clip in speaker.clips}
        if not rel_paths:
            continue
        rows = list_voiceprint_samples(
            speaker.person_public_id or speaker.name, db_path
        )
        for row in rows:
            if row.clip_rel_path in rel_paths:
                captured.add(row.public_id)
    return captured


def _workflow_summary_text(summary: VoiceprintReviewWorkflowSummary) -> str:
    """Render capture, embedding, and evaluation details."""
    lines = [
        tr(
            "[b green]Voiceprint embedding complete[/b green]",
            "[b green]声纹 embedding 已完成[/b green]",
        ),
        "",
        tr(
            f"Captured samples: {summary.capture.sample_count}",
            f"已采集样本：{summary.capture.sample_count}",
        ),
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
            "Press a to accept these embeddings. Press r/Esc/q to roll back.",
            "按 a 接受这些 embedding。按 r/Esc/q 回滚。",
        ),
    ]
    return "\n".join(lines)


def _current_evaluation_text(summary: VoiceprintEvaluationSummary) -> str:
    """Render current-project score changes."""
    current = summary.current
    lines = [tr("[b]Current project score check[/b]", "[b]当前项目分数检查[/b]")]
    visible_changes = tuple(
        change for change in current.changes if change.status != "unchanged"
    )
    if not visible_changes:
        lines.append(tr("No material score changes.", "没有实质分数变化。"))
        return "\n".join(lines)
    for change in visible_changes[:CURRENT_CHANGE_DISPLAY_LIMIT]:
        lines.append("  " + _current_score_change_line(change))
    hidden_count = len(visible_changes) - CURRENT_CHANGE_DISPLAY_LIMIT
    if hidden_count > 0:
        lines.append(
            tr(
                f"  [dim]... {hidden_count} more current change(s) omitted.[/]",
                f"  [dim]... 省略 {hidden_count} 个当前项目变化。[/]",
            )
        )
    return "\n".join(lines)


def _historical_evaluation_text(summary: VoiceprintEvaluationSummary) -> str:
    """Render historical regression summary."""
    lines = [tr("[b]Historical reverse check[/b]", "[b]历史项目反向评测[/b]")]
    lines.append(_historical_severity_summary_line(summary))
    lines.extend(_historical_risk_lines(summary))
    return "\n".join(lines)


def _historical_severity_summary_line(summary: VoiceprintEvaluationSummary) -> str:
    """Render historical severity counts."""
    if summary.historical_risk_count == 0:
        return tr(
            f"Checked {summary.historical_project_count} project(s); [green]no risky changes[/].",
            f"检查 {summary.historical_project_count} 个历史项目；[green]无风险变化[/]。",
        )
    return tr(
        f"Checked {summary.historical_project_count} project(s); "
        f"[bold red]critical {summary.historical_critical_count}[/] | "
        f"[yellow]warnings {summary.historical_warning_count}[/].",
        f"检查 {summary.historical_project_count} 个历史项目；"
        f"[bold red]严重 {summary.historical_critical_count} 个[/] | "
        f"[yellow]警告 {summary.historical_warning_count} 个[/]。",
    )


def _historical_risk_lines(summary: VoiceprintEvaluationSummary) -> list[str]:
    """Render risky historical project lines."""
    lines: list[str] = []
    risky_projects = tuple(
        project for project in summary.historical if project.risk_count > 0
    )
    for project in risky_projects[:HISTORICAL_PROJECT_DISPLAY_LIMIT]:
        style = "bold red" if project.critical_count else "yellow"
        label = "CRITICAL" if project.critical_count else "WARNING"
        title = _trim_text(project.title or project.project_id, limit=40)
        project_id = escape(project.project_id)
        lines.append(
            f"  [{style}]{label}[/] {project_id} | [{style}]{escape(title)}[/] | "
            f"{_count_badge('critical', project.critical_count, 'bold red')} "
            f"{_count_badge('warning', project.warning_count, 'yellow')}"
        )
        lines.append(f"    [{style}]review: meeting-asr project review {project_id}[/]")
        risky_changes = tuple(
            change
            for change in project.changes
            if change.is_warning or change.is_critical
        )
        lines.extend(
            "    " + _score_change_line(change)
            for change in risky_changes[:HISTORICAL_CHANGE_DISPLAY_LIMIT]
        )
        hidden_change_count = len(risky_changes) - HISTORICAL_CHANGE_DISPLAY_LIMIT
        if hidden_change_count > 0:
            lines.append(
                tr(
                    f"    [dim]... {hidden_change_count} more risky change(s) omitted.[/]",
                    f"    [dim]... 省略 {hidden_change_count} 个风险变化。[/]",
                )
            )
    hidden_project_count = len(risky_projects) - HISTORICAL_PROJECT_DISPLAY_LIMIT
    if hidden_project_count > 0:
        lines.append(
            tr(
                f"  [dim]... {hidden_project_count} more risky project(s) omitted; open Project Review from project list.[/]",
                f"  [dim]... 省略 {hidden_project_count} 个风险项目；可从 project list 进入对应 Project Review。[/]",
            )
        )
    return lines


def _count_badge(label: str, count: int, style: str) -> str:
    """Render one severity count without highlighting zero values."""
    if count == 0:
        return f"[dim]{label}=0[/]"
    return f"[{style}]{label}={count}[/]"


def _score_change_line(change) -> str:
    """Render one before/after score line."""
    before = _candidate_score_text(change.before_name, change.before_score)
    after = _candidate_score_text(change.after_name, change.after_score)
    delta_text = "" if change.delta is None else f" ({change.delta:+.3f})"
    status_text = (
        f" {escape(change.status)}" if change.is_warning or change.is_critical else ""
    )
    threshold_text = (
        "" if change.threshold is None else f" threshold={change.threshold:.3f}"
    )
    line = f"{escape(change.label)}: {before} -> {after}{delta_text}{status_text}"
    if change.is_critical:
        return f"[red]{line}{threshold_text}[/]"
    if change.is_warning:
        return f"[yellow]{line}{threshold_text}[/]"
    return line


def _current_score_change_line(change) -> str:
    """Render one current-project score line as an expected workflow outcome."""
    before = _candidate_score_text(change.before_name, change.before_score)
    after = _candidate_score_text(change.after_name, change.after_score)
    delta_text = "" if change.delta is None else f" ({change.delta:+.3f})"
    status_text = "" if change.status == "improved" else f" {escape(change.status)}"
    threshold_text = (
        "" if change.threshold is None else f" threshold={change.threshold:.3f}"
    )
    line = f"{escape(change.label)}: {before} -> {after}{delta_text}{status_text}"
    if change.status in {"improved", "changed-best"}:
        return f"[green]{line}{threshold_text}[/]"
    if change.is_critical:
        return f"[red]{line}{threshold_text}[/]"
    if change.is_warning:
        return f"[yellow]{line}{threshold_text}[/]"
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
        project_manifest_existed=_backup_file(
            project_root / "project.json", backup_dir / "project.json"
        ),
        match_path=project_root / "speakers" / "speaker_matches.json",
        match_backup_path=backup_dir / "speaker_matches.json",
        match_existed=_backup_file(
            project_root / "speakers" / "speaker_matches.json",
            backup_dir / "speaker_matches.json",
        ),
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
                backups.append(
                    (clip.path, backup_path, _backup_file(clip.path, backup_path))
                )
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
