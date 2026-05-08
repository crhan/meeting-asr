"""Rich renderer for human-facing doctor diagnostics."""

from __future__ import annotations

import textwrap
from typing import Protocol

from rich import box
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from app.presentation.cli.i18n import current_cli_language
from app.presentation.cli.output import cli_console


class DoctorCheck(Protocol):
    """Read-only shape consumed by the doctor renderer."""

    name: str
    status: str
    detail: str
    fix_prompt: str | None

    @property
    def needs_attention(self) -> bool:
        """Return whether this check should print repair guidance."""
        ...


LABELS = {
    "en": {
        "title": "Meeting-ASR Doctor",
        "summary": "Summary",
        "checks": "Checks",
        "status": "Status",
        "check": "Check",
        "detail": "Detail",
        "repair": "Repair Prompts",
        "prompt": "Prompt",
        "mode": "Mode",
        "basic": "Basic",
        "full": "Full",
        "full_hint": "For complete integration checks, run `meeting-asr doctor --full`.",
    },
    "zh": {
        "title": "Meeting-ASR 诊断",
        "summary": "汇总",
        "checks": "检查项",
        "status": "状态",
        "check": "检查",
        "detail": "详情",
        "repair": "修复提示",
        "prompt": "提示词",
        "mode": "模式",
        "basic": "基础",
        "full": "完整",
        "full_hint": "需要完整集成检查时，运行 `meeting-asr doctor --full`。",
    },
}
STATUS_LABELS = {
    "en": {"ok": "OK", "warn": "WARN", "fail": "FAIL"},
    "zh": {"ok": "正常", "warn": "警告", "fail": "失败"},
}
STATUS_STYLES = {"ok": "green", "warn": "yellow", "fail": "red"}
DETAIL_REPLACEMENTS_ZH = (
    ("installed:", "已安装："),
    ("missing standard packages:", "缺少标准依赖："),
    ("dependencies installed", "依赖已安装"),
    ("not found in PATH; install ffmpeg", "PATH 中找不到；请安装 ffmpeg"),
    ("not found; install mpv or IINA for speaker review", "未找到；请安装 mpv 或 IINA 用于 speaker review"),
    ("bucket metadata request succeeded", "bucket 元数据请求成功"),
    ("no object uploaded", "未上传对象"),
    ("put_object + signed GET succeeded", "put_object + signed GET 成功"),
)
PROMPT_REPLACEMENTS_ZH = (
    ("You are fixing `meeting-asr doctor` output.", "你正在修复 `meeting-asr doctor` 的诊断问题。"),
    ("Problem:", "问题："),
    ("Repair:", "修复："),
    ("Verify:", "验证："),
    ("Do not print or commit secrets.", "不要打印或提交密钥。"),
)


def render_doctor_report(checks: list[DoctorCheck], *, full: bool = False) -> None:
    """
    Render doctor checks for humans.

    Args:
        checks: Diagnostic checks from the doctor command.
        full: Whether this report includes all integration checks.

    Returns:
        None.
    """
    lang = current_cli_language()
    console = cli_console()
    console.print(_summary_panel(checks, lang, full=full))
    console.print(_checks_table(checks, lang))
    _print_full_hint(full=full, lang=lang, console=console)
    _print_repair_prompts(checks, lang, console)


def _summary_panel(checks: list[DoctorCheck], lang: str, *, full: bool) -> Panel:
    """Build the top-level doctor summary panel."""
    counts = _counts(checks)
    grid = Table.grid(expand=True)
    for status in ("ok", "warn", "fail"):
        grid.add_column(justify="center")
    grid.add_column(justify="center")
    grid.add_row(
        *(_summary_cell(status, counts[status], lang) for status in ("ok", "warn", "fail")),
        _mode_cell(full, lang),
    )
    return Panel(grid, title=LABELS[lang]["title"], subtitle=LABELS[lang]["summary"], border_style="dim")


def _checks_table(checks: list[DoctorCheck], lang: str) -> Table:
    """Build a scan-friendly checks table."""
    table = Table(title=LABELS[lang]["checks"], box=box.SIMPLE_HEAVY, expand=True)
    table.add_column(LABELS[lang]["status"], no_wrap=True)
    table.add_column(LABELS[lang]["check"], no_wrap=True, style="cyan")
    table.add_column(LABELS[lang]["detail"], ratio=1, overflow="fold")
    for check in checks:
        table.add_row(_status_text(check.status, lang), check.name, _detail_text(check.detail, lang))
    return table


def _print_repair_prompts(checks: list[DoctorCheck], lang: str, console: Console) -> None:
    """Print repair prompts when doctor found warnings or failures."""
    problem_checks = [check for check in checks if check.needs_attention]
    if not problem_checks:
        return
    table = Table(box=box.SIMPLE_HEAVY, expand=True)
    table.add_column(LABELS[lang]["check"], no_wrap=True, style="cyan")
    table.add_column(LABELS[lang]["prompt"], ratio=1, overflow="fold")
    for check in problem_checks:
        table.add_row(check.name, _prompt_text(check.fix_prompt or "", lang))
    console.print(Panel(table, title=LABELS[lang]["repair"], border_style="yellow"))


def _print_full_hint(*, full: bool, lang: str, console: Console) -> None:
    """Print a discoverability hint for the side-effecting full check."""
    if full:
        return
    console.print(Panel(Text(LABELS[lang]["full_hint"]), border_style="blue"))


def _counts(checks: list[DoctorCheck]) -> dict[str, int]:
    """Count checks by status."""
    return {
        "ok": sum(1 for check in checks if check.status == "ok"),
        "warn": sum(1 for check in checks if check.status == "warn"),
        "fail": sum(1 for check in checks if check.status == "fail"),
    }


def _summary_cell(status: str, count: int, lang: str) -> Text:
    """Build one colored summary cell."""
    text = Text(str(count), style=f"bold {STATUS_STYLES[status]}")
    text.append(f" {STATUS_LABELS[lang][status]}", style=STATUS_STYLES[status])
    return text


def _mode_cell(full: bool, lang: str) -> Text:
    """Build the doctor mode cell."""
    mode = LABELS[lang]["full"] if full else LABELS[lang]["basic"]
    separator = "：" if lang == "zh" else ": "
    return Text(f"{LABELS[lang]['mode']}{separator}{mode}", style="bold blue")


def _status_text(status: str, lang: str) -> Text:
    """Build one colored status label."""
    return Text(STATUS_LABELS[lang].get(status, status.upper()), style=STATUS_STYLES.get(status, "white"))


def _detail_text(detail: str, lang: str) -> Text:
    """Build wrapped detail text."""
    localized = _localize_detail(detail, lang)
    return Text("\n".join(_format_detail_lines(localized, width=92)))


def _prompt_text(prompt: str, lang: str) -> Text:
    """Build wrapped repair prompt text."""
    localized = _localize_prompt(prompt, lang)
    return Text("\n".join(_wrap_text(localized, width=92)))


def _localize_detail(detail: str, lang: str) -> str:
    """Localize common doctor detail fragments while preserving technical values."""
    if lang != "zh":
        return detail
    localized = detail
    for source, target in DETAIL_REPLACEMENTS_ZH:
        localized = localized.replace(source, target)
    return localized


def _localize_prompt(prompt: str, lang: str) -> str:
    """Localize doctor prompt wrapper text."""
    if lang != "zh":
        return prompt
    localized = prompt
    for source, target in PROMPT_REPLACEMENTS_ZH:
        localized = localized.replace(source, target)
    return localized


def _format_detail_lines(detail: str, *, width: int) -> list[str]:
    """Split semicolon-heavy detail text before wrapping."""
    lines: list[str] = []
    for part in detail.split("; "):
        lines.extend(_wrap_text(part, width=width))
    return lines


def _wrap_text(value: str, *, width: int) -> list[str]:
    """Wrap terminal text while preserving explicit line breaks and commands."""
    lines: list[str] = []
    for raw_line in value.splitlines() or [""]:
        if _looks_like_command(raw_line):
            lines.append(raw_line)
            continue
        wrapped = textwrap.wrap(raw_line, width=width, break_long_words=False, break_on_hyphens=False)
        lines.extend(wrapped or [""])
    return lines


def _looks_like_command(value: str) -> bool:
    """Return whether a line should stay copyable as one command."""
    stripped = value.strip()
    return stripped.startswith(("meeting-asr ", "uv ", "brew ", "python ", "scripts/"))
