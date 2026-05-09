"""Textual UI for reviewing voiceprint sample quality."""

from __future__ import annotations

import subprocess
from dataclasses import dataclass
from pathlib import Path

from rich.markup import escape
from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.containers import Horizontal
from textual.screen import ModalScreen
from textual.widgets import Header, Static

from app.presentation.tui.i18n import tr
from app.utils import format_ms_timestamp
from app.voiceprint_audio import voiceprint_playback_clip_path
from app.voiceprint_playback import build_voiceprint_play_command
from app.voiceprint_quality import (
    VOICEPRINT_SAMPLE_STATUS_ACTIVE,
    VOICEPRINT_SAMPLE_STATUS_QUARANTINED,
    VoiceprintQualityPerson,
    VoiceprintQualityReport,
    VoiceprintQualitySample,
    analyze_voiceprint_quality,
)
from app.voiceprint_store import get_voiceprint_db_path, update_voiceprint_sample_status

FOCUSED_PANE_CLASS = "focused-pane"
UNFOCUSED_PANE_CLASS = "unfocused-pane"
COLUMNS = ("people", "samples")


@dataclass(frozen=True, slots=True)
class VoiceprintQualityDecision:
    """Changes requested by the voiceprint quality TUI."""

    saved: bool
    statuses: dict[str, str]


class VoiceprintQualityHelpScreen(ModalScreen[None]):
    """Modal shortcut help for voiceprint quality review."""

    CSS = """
    VoiceprintQualityHelpScreen {
        align: center middle;
    }
    #quality-help {
        width: 86;
        height: auto;
        border: thick $accent;
        padding: 1 2;
        background: $surface;
    }
    """

    BINDINGS = [
        Binding("escape", "close_help", "Close", show=False),
        Binding("q", "close_help", "Close"),
        Binding("?", "close_help", "Close", show=False),
    ]

    def compose(self) -> ComposeResult:
        """Build the help popup."""
        yield Static(_help_text(), id="quality-help")

    def action_close_help(self) -> None:
        """Close the shortcut help popup."""
        self.dismiss(None)


class VoiceprintQualityApp(App[VoiceprintQualityDecision]):
    """Keyboard-first TUI for reviewing suspicious voiceprint samples."""

    CSS = """
    Screen {
        layout: vertical;
        background: #0f1117;
        color: #d9e2ec;
    }
    #overview {
        border: heavy #ffb000;
        background: #151922;
        height: 8;
        padding: 0 1;
    }
    #main {
        height: 1fr;
    }
    .pane {
        border: round #3d4758;
        background: #10151e;
        height: 100%;
        padding: 0 1;
    }
    .pane.focused-pane {
        border: heavy #00d1ff;
        background: #18202b;
    }
    .pane.unfocused-pane {
        border: round #3d4758;
        color: #9aa7b8;
    }
    #people {
        width: 34%;
    }
    #samples {
        width: 66%;
    }
    #status {
        height: 2;
        background: #0b0f14;
        color: #8fb3ff;
    }
    """

    BINDINGS = [
        Binding("j", "down", "Down"),
        Binding("k", "up", "Up"),
        Binding("down", "down", "Down", show=False),
        Binding("up", "up", "Up", show=False),
        Binding("h", "left", "Left"),
        Binding("l", "right", "Right"),
        Binding("left", "left", "Left", show=False),
        Binding("right", "right", "Right", show=False),
        Binding("space", "play_sample", "Play/stop"),
        Binding("x", "toggle_quarantine", "Quarantine/active"),
        Binding("a", "mark_active", "Keep active"),
        Binding("r", "mark_quarantined", "Quarantine"),
        Binding("s", "save", "Save"),
        Binding("u", "refresh_quality", "Refresh"),
        Binding("?", "show_shortcuts", "Help"),
        Binding("escape", "quit", "Quit", show=False),
        Binding("q", "quit", "Quit"),
    ]

    def __init__(
        self,
        report: VoiceprintQualityReport,
        *,
        store_dir: Path | None = None,
        speaker: str | None = None,
        model: str | None = None,
    ) -> None:
        """
        Create the quality review app.

        Args:
            report: Quality report to review.
            store_dir: Optional voiceprint store directory for in-place save/refresh.
            speaker: Optional speaker filter to preserve on refresh.
            model: Optional embedding model key to preserve on refresh.
        """
        super().__init__()
        self.report = report
        self.store_dir = store_dir
        self.speaker = speaker
        self.model = model
        self.selected_person_index = 0
        self.selected_sample_indices = {person.speaker_public_id: 0 for person in report.people}
        self.focused_column = "people"
        self.statuses = _initial_statuses(report)
        self.playback_process: subprocess.Popen | None = None

    def compose(self) -> ComposeResult:
        """Build the TUI layout."""
        yield Header()
        yield Static(id="overview")
        with Horizontal(id="main"):
            yield Static(id="people", classes="pane")
            yield Static(id="samples", classes="pane")
        yield Static(_status_text(), id="status")

    def on_mount(self) -> None:
        """Render initial state."""
        self._refresh()

    def on_unmount(self) -> None:
        """Stop any child player when closing."""
        self._stop_playback()

    def action_down(self) -> None:
        """Move down in the focused pane."""
        if self.focused_column == "people":
            self._move_person(1)
            return
        self._move_sample(1)

    def action_up(self) -> None:
        """Move up in the focused pane."""
        if self.focused_column == "people":
            self._move_person(-1)
            return
        self._move_sample(-1)

    def action_left(self) -> None:
        """Focus the people pane."""
        self.focused_column = "people"
        self._refresh()

    def action_right(self) -> None:
        """Focus the samples pane."""
        self.focused_column = "samples"
        self._refresh()

    def action_play_sample(self) -> None:
        """Play or stop the selected sample."""
        if self._is_playing():
            self._stop_playback()
            self._set_status(tr("Stopped sample playback.", "已停止样本播放。"))
            return
        sample = self._sample()
        if sample is None:
            return
        self._stop_playback()
        clip_path = voiceprint_playback_clip_path(sample.clip_path, store_dir=self.store_dir)
        command = build_voiceprint_play_command(clip_path)
        self.playback_process = subprocess.Popen(
            command,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        self._set_status(tr(f"Playing {sample.sample_public_id}.", f"正在播放 {sample.sample_public_id}。"))

    def action_toggle_quarantine(self) -> None:
        """Toggle the selected sample between active and quarantined."""
        sample = self._sample()
        if sample is None:
            return
        current = self.statuses[sample.sample_public_id]
        target = VOICEPRINT_SAMPLE_STATUS_ACTIVE if current == VOICEPRINT_SAMPLE_STATUS_QUARANTINED else VOICEPRINT_SAMPLE_STATUS_QUARANTINED
        self.statuses[sample.sample_public_id] = target
        self._set_status(tr(f"Set {sample.sample_public_id} to {target}.", f"已把 {sample.sample_public_id} 标记为 {target}。"))
        self._refresh()

    def action_mark_active(self) -> None:
        """Mark the selected sample active."""
        self._mark_sample(VOICEPRINT_SAMPLE_STATUS_ACTIVE)

    def action_mark_quarantined(self) -> None:
        """Mark the selected sample quarantined."""
        self._mark_sample(VOICEPRINT_SAMPLE_STATUS_QUARANTINED)

    def action_save(self) -> None:
        """Persist changed sample statuses and refresh quality scores."""
        changes = _changed_statuses(self.report, self.statuses)
        if self.store_dir is None:
            self.exit(VoiceprintQualityDecision(True, changes))
            return
        if not changes:
            self._set_status(tr("No staged quality changes to save.", "没有待保存的声纹质量变更。"))
            return
        for sample_id, status in changes.items():
            update_voiceprint_sample_status(sample_id, status, get_voiceprint_db_path(self.store_dir))
        self._reload_report(status=tr(f"Saved {len(changes)} change(s) and refreshed quality scores.", f"已保存 {len(changes)} 个变更，并刷新质量评分。"))

    def action_refresh_quality(self) -> None:
        """Reload quality scores from SQLite."""
        if _changed_statuses(self.report, self.statuses):
            self._set_status(tr("Save staged changes before refreshing quality scores.", "刷新质量评分前请先保存暂存变更。"))
            return
        self._reload_report(status=tr("Quality scores refreshed.", "质量评分已刷新。"))

    def action_show_shortcuts(self) -> None:
        """Show shortcut help."""
        self.push_screen(VoiceprintQualityHelpScreen())

    def action_quit(self) -> None:
        """Exit without saving changes."""
        self.exit(VoiceprintQualityDecision(False, {}))

    def _mark_sample(self, status: str) -> None:
        """Mark selected sample with a lifecycle status."""
        sample = self._sample()
        if sample is None:
            return
        self.statuses[sample.sample_public_id] = status
        self._set_status(tr(f"Set {sample.sample_public_id} to {status}.", f"已把 {sample.sample_public_id} 标记为 {status}。"))
        self._refresh()

    def _move_person(self, delta: int) -> None:
        """Move selected person."""
        if not self.report.people:
            return
        self.selected_person_index = (self.selected_person_index + delta) % len(self.report.people)
        self._refresh()

    def _move_sample(self, delta: int) -> None:
        """Move selected sample."""
        person = self._person()
        if person is None or not person.samples:
            return
        current = self.selected_sample_indices[person.speaker_public_id]
        self.selected_sample_indices[person.speaker_public_id] = max(0, min(len(person.samples) - 1, current + delta))
        self._refresh()

    def _refresh(self) -> None:
        """Refresh visible panes."""
        self._refresh_focus_styles()
        self.query_one("#overview", Static).update(self._overview())
        self.query_one("#people", Static).update(self._people_pane())
        self.query_one("#samples", Static).update(self._samples_pane())

    def _reload_report(self, *, status: str) -> None:
        """Reload quality report while preserving the selected person when possible."""
        previous_person_id = self._person().speaker_public_id if self._person() is not None else None
        self.report = analyze_voiceprint_quality(store_dir=self.store_dir, speaker=self.speaker, model=self.model)
        self.selected_sample_indices = {person.speaker_public_id: 0 for person in self.report.people}
        self.statuses = _initial_statuses(self.report)
        self.selected_person_index = _person_index(self.report, previous_person_id)
        self._refresh()
        self._set_status(status)

    def _refresh_focus_styles(self) -> None:
        """Make the focused pane visually obvious."""
        for column in COLUMNS:
            pane = self.query_one(f"#{column}", Static)
            focused = column == self.focused_column
            pane.set_class(focused, FOCUSED_PANE_CLASS)
            pane.set_class(not focused, UNFOCUSED_PANE_CLASS)

    def _overview(self) -> str:
        """Render report overview."""
        person = self._person()
        sample = self._sample()
        changed = len(_changed_statuses(self.report, self.statuses))
        return "\n".join(
            [
                tr("[b]VOICEPRINT QUALITY[/b]", "[b]声纹质量检查[/b]"),
                f"DB      {escape(str(self.report.db_path))}",
                f"Model   {escape(self.report.model)}",
                tr(
                    f"Total   people {len(self.report.people)} | samples {self.report.sample_count} | suspicious {self.report.suspicious_count} | critical {self.report.critical_count}",
                    f"总览    人员 {len(self.report.people)} | 样本 {self.report.sample_count} | 可疑 {self.report.suspicious_count} | 严重 {self.report.critical_count}",
                ),
                tr(f"Focus   {_person_summary(person)}", f"当前    {_person_summary(person)}"),
                tr(f"Sample  {_sample_summary(sample, self.statuses)}", f"样本    {_sample_summary(sample, self.statuses)}"),
                tr(f"Changes {changed} staged", f"变更    已暂存 {changed} 个"),
            ]
        )

    def _people_pane(self) -> str:
        """Render people list."""
        lines = [tr("[b]People[/b]", "[b]人员[/b]")]
        if not self.report.people:
            lines.append(tr("[yellow]No embedded voiceprint samples.[/]", "[yellow]没有已 embedding 的声纹样本。[/]"))
            return "\n".join(lines)
        for index, person in enumerate(self.report.people):
            marker = ">" if index == self.selected_person_index else " "
            style = _person_style(person)
            score = "-" if person.mean_score is None else f"{person.mean_score:.3f}"
            line = f"{marker} {person.speaker_name} {person.speaker_public_id} mean={score} suspicious={person.suspicious_count}/{person.sample_count}"
            lines.append(f"[{style}]{escape(line)}[/]" if style else escape(line))
        return "\n".join(lines)

    def _samples_pane(self) -> str:
        """Render samples for selected person."""
        person = self._person()
        if person is None:
            return tr("[b]Samples[/b]\n[yellow]No person selected.[/]", "[b]样本[/b]\n[yellow]未选择人员。[/]")
        lines = [tr(f"[b]{person.speaker_name} samples[/b]", f"[b]{person.speaker_name} 样本[/b]")]
        selected_index = self.selected_sample_indices[person.speaker_public_id]
        for index, sample in enumerate(person.samples):
            marker = ">" if index == selected_index else " "
            status = self.statuses[sample.sample_public_id]
            score = "-" if sample.score is None else f"{sample.score:.3f}"
            style = _sample_style(sample, status)
            line = f"{marker} {sample.sample_public_id} score={score} {sample.label} -> {status} | {_clip_time(sample)}"
            lines.append(f"[{style}]{escape(line)}[/]" if style else escape(line))
            if index == selected_index:
                lines.append(f"  [dim]{escape(sample.reason)}[/]")
                lines.append(f"  [dim]{escape(trim_text(sample.transcript_text, 120))}[/]")
                lines.append(f"  [dim]{escape(str(sample.clip_path))}[/]")
        return "\n".join(lines)

    def _person(self) -> VoiceprintQualityPerson | None:
        """Return selected person."""
        if not self.report.people:
            return None
        return self.report.people[self.selected_person_index]

    def _sample(self) -> VoiceprintQualitySample | None:
        """Return selected sample."""
        person = self._person()
        if person is None or not person.samples:
            return None
        return person.samples[self.selected_sample_indices[person.speaker_public_id]]

    def _set_status(self, message: str) -> None:
        """Update bottom status line."""
        self.query_one("#status", Static).update(message)

    def _stop_playback(self) -> None:
        """Stop current playback if needed."""
        process = self.playback_process
        self.playback_process = None
        if process is None or process.poll() is not None:
            return
        process.terminate()
        try:
            process.wait(timeout=2)
        except subprocess.TimeoutExpired:
            process.kill()

    def _is_playing(self) -> bool:
        """Return whether playback is running."""
        process = self.playback_process
        return process is not None and process.poll() is None


def run_voiceprint_quality_tui(report: VoiceprintQualityReport) -> VoiceprintQualityDecision:
    """
    Run the voiceprint quality TUI.

    Args:
        report: Quality report.

    Returns:
        User decision.
    """
    return VoiceprintQualityApp(report).run()


def run_voiceprint_quality_review_tui(
    report: VoiceprintQualityReport,
    *,
    store_dir: Path | None,
    speaker: str | None,
    model: str | None,
) -> VoiceprintQualityDecision:
    """
    Run the quality review TUI with in-place save and refresh enabled.

    Args:
        report: Quality report.
        store_dir: Optional voiceprint store directory.
        speaker: Optional speaker filter.
        model: Optional model key.

    Returns:
        User decision when the app exits.
    """
    return VoiceprintQualityApp(report, store_dir=store_dir, speaker=speaker, model=model).run()


def persist_quality_decision(decision: VoiceprintQualityDecision, *, store_dir: Path | None) -> dict[str, str]:
    """
    Persist sample status changes from quality review.

    Args:
        decision: TUI decision.
        store_dir: Optional voiceprint store directory.

    Returns:
        Changed sample statuses.
    """
    if not decision.saved:
        return {}
    db_path = get_voiceprint_db_path(store_dir)
    for sample_id, status in decision.statuses.items():
        update_voiceprint_sample_status(sample_id, status, db_path)
    return dict(decision.statuses)


def _initial_statuses(report: VoiceprintQualityReport) -> dict[str, str]:
    """Return sample id to current status."""
    return {sample.sample_public_id: sample.status for person in report.people for sample in person.samples}


def _changed_statuses(report: VoiceprintQualityReport, statuses: dict[str, str]) -> dict[str, str]:
    """Return staged status changes only."""
    changes = {}
    for person in report.people:
        for sample in person.samples:
            status = statuses[sample.sample_public_id]
            if status != sample.status:
                changes[sample.sample_public_id] = status
    return changes


def _person_index(report: VoiceprintQualityReport, person_public_id: str | None) -> int:
    """Return the best selected person index after a report refresh."""
    if person_public_id is None:
        return 0
    for index, person in enumerate(report.people):
        if person.speaker_public_id == person_public_id:
            return index
    return 0


def _status_text() -> str:
    """Return bottom status help."""
    return tr(
        "j/k move | h/l pane | space play/stop | x toggle quarantine | a active | r quarantine | s save+refresh | u refresh | q quit",
        "j/k 移动 | h/l 切栏 | space 播放/停止 | x 切换隔离 | a 保留 | r 隔离 | s 保存并刷新 | u 刷新 | q 退出",
    )


def _help_text() -> str:
    """Return shortcut help text."""
    return tr(
        """\
[b]Voiceprint Quality Review[/b]

Left pane        People sorted by suspicious sample count
Right pane       Samples sorted by quality risk
space            Play or stop selected WAV sample
x                Toggle selected sample active/quarantined
a                Mark selected sample active
r                Mark selected sample quarantined
s                Save status changes and refresh scores
u                Refresh quality scores from SQLite
q / Esc          Quit without saving

Quarantined samples stay in the library but are excluded from future matching.
""",
        """\
[b]声纹质量检查[/b]

左侧             按可疑样本数量排序的人员
右侧             按风险排序的样本
space            播放或停止当前 WAV 样本
x                在 active/quarantined 之间切换
a                保留当前样本，参与后续匹配
r                隔离当前样本，不参与后续匹配
s                保存状态变更并刷新评分
u                从 SQLite 刷新质量评分
q / Esc          不保存退出

被隔离的样本仍保留在声纹库里，但后续匹配不再使用。
""",
    )


def _person_summary(person: VoiceprintQualityPerson | None) -> str:
    """Return compact person summary."""
    if person is None:
        return "-"
    mean = "-" if person.mean_score is None else f"{person.mean_score:.3f}"
    return f"{person.speaker_name} {person.speaker_public_id} mean={mean} suspicious={person.suspicious_count}"


def _sample_summary(sample: VoiceprintQualitySample | None, statuses: dict[str, str]) -> str:
    """Return compact sample summary."""
    if sample is None:
        return "-"
    score = "-" if sample.score is None else f"{sample.score:.3f}"
    return f"{sample.sample_public_id} score={score} {sample.label} status={statuses[sample.sample_public_id]}"


def _person_style(person: VoiceprintQualityPerson) -> str:
    """Return markup style for a person row."""
    if person.critical_count:
        return "bold red"
    if person.suspicious_count:
        return "yellow"
    return "green"


def _sample_style(sample: VoiceprintQualitySample, status: str) -> str:
    """Return markup style for a sample row."""
    if status != VOICEPRINT_SAMPLE_STATUS_ACTIVE:
        return "dim"
    if sample.label == "critical":
        return "bold red"
    if sample.label == "warning":
        return "yellow"
    if sample.label == "ok":
        return "green"
    return ""


def _clip_time(sample: VoiceprintQualitySample) -> str:
    """Return a best-effort clip time label from the store row."""
    return (
        f"{sample.project_id} "
        f"{format_ms_timestamp(sample.source_begin_time_ms)}-{format_ms_timestamp(sample.source_end_time_ms)}"
    )


def trim_text(text: str, limit: int) -> str:
    """Return compact one-line transcript text."""
    normalized = " ".join(text.split())
    if len(normalized) <= limit:
        return normalized
    return f"{normalized[: max(0, limit - 1)]}…"
