"""Rich render helpers for TUI text diffs."""

from __future__ import annotations

import difflib
import re
from collections.abc import Callable

from rich.text import Text

DIFF_TOKEN_RE = re.compile(
    r"[A-Za-z0-9_+.#-]+|[\u4e00-\u9fff]+|\s+|[^\w\s]", re.UNICODE
)


def styled_unified_diff(diff_text: str) -> Text:
    """Return token-styled unified diff text."""
    if not diff_text:
        return Text("(empty diff)", style="dim")
    rendered = Text(no_wrap=False)
    lines = diff_text.splitlines()
    index = 0
    while index < len(lines):
        line = lines[index]
        if _is_removed_diff_line(line):
            removed, index = _collect_diff_lines(lines, index, _is_removed_diff_line)
            added, index = _collect_diff_lines(lines, index, _is_added_diff_line)
            append_changed_line_block(rendered, removed, added)
            continue
        if _is_added_diff_line(line):
            added, index = _collect_diff_lines(lines, index, _is_added_diff_line)
            append_changed_line_block(rendered, [], added)
            continue
        rendered.append(line, style=_diff_line_style(line))
        rendered.append("\n")
        index += 1
    return rendered


def styled_before_after(original_text: str, corrected_text: str) -> Text:
    """Return a token-styled before/after correction preview."""
    original_segments, corrected_segments = word_diff_segments(
        original_text, corrected_text
    )
    rendered = Text(no_wrap=False)
    append_segmented_line(rendered, "Before: ", original_segments, removed=True)
    append_segmented_line(rendered, "After:  ", corrected_segments, removed=False)
    return rendered


def append_changed_line_block(
    rendered: Text, removed: list[str], added: list[str]
) -> None:
    """Append one removed/added line block with token-level highlights."""
    count = max(len(removed), len(added))
    for index in range(count):
        old_line = removed[index] if index < len(removed) else None
        new_line = added[index] if index < len(added) else None
        if old_line is not None and new_line is not None:
            old_segments, new_segments = word_diff_segments(old_line[1:], new_line[1:])
            append_segmented_line(rendered, "-", old_segments, removed=True)
            append_segmented_line(rendered, "+", new_segments, removed=False)
            continue
        if old_line is not None:
            rendered.append(old_line, style="red")
            rendered.append("\n")
        if new_line is not None:
            rendered.append(new_line, style="green")
            rendered.append("\n")


def append_segmented_line(
    rendered: Text,
    prefix: str,
    segments: list[tuple[str, bool]],
    *,
    removed: bool,
) -> None:
    """Append one diff line with changed token highlights."""
    rendered.append(prefix, style="bold red" if removed else "bold green")
    for token, changed in segments:
        rendered.append(token, style=_diff_token_style(changed, removed=removed))
    rendered.append("\n")


def word_diff_segments(
    old_text: str, new_text: str
) -> tuple[list[tuple[str, bool]], list[tuple[str, bool]]]:
    """Return old/new token segments marked as changed or unchanged."""
    old_tokens = _diff_tokens(old_text)
    new_tokens = _diff_tokens(new_text)
    matcher = difflib.SequenceMatcher(a=old_tokens, b=new_tokens, autojunk=False)
    old_segments: list[tuple[str, bool]] = []
    new_segments: list[tuple[str, bool]] = []
    for tag, old_start, old_end, new_start, new_end in matcher.get_opcodes():
        changed = tag != "equal"
        old_segments.append(("".join(old_tokens[old_start:old_end]), changed))
        new_segments.append(("".join(new_tokens[new_start:new_end]), changed))
    return old_segments, new_segments


def _collect_diff_lines(
    lines: list[str], start: int, predicate: Callable[[str], bool]
) -> tuple[list[str], int]:
    """Collect a contiguous diff line run."""
    collected: list[str] = []
    index = start
    while index < len(lines) and predicate(lines[index]):
        collected.append(lines[index])
        index += 1
    return collected, index


def _diff_tokens(text: str) -> list[str]:
    """Split diff text into readable word-ish tokens."""
    return DIFF_TOKEN_RE.findall(text) or [text]


def _is_removed_diff_line(line: str) -> bool:
    """Return whether a unified diff line is a removed content line."""
    return line.startswith("-") and not line.startswith("---")


def _is_added_diff_line(line: str) -> bool:
    """Return whether a unified diff line is an added content line."""
    return line.startswith("+") and not line.startswith("+++")


def _diff_token_style(changed: bool, *, removed: bool) -> str:
    """Return the style for one diff token."""
    if changed:
        return "bold red" if removed else "bold green"
    return "dim red" if removed else "dim green"


def _diff_line_style(line: str) -> str:
    """Return the style for one unified diff line."""
    if line.startswith("@@"):
        return "bold cyan"
    if line.startswith("---") or line.startswith("+++"):
        return "bold white on dark_blue"
    if line.startswith("-"):
        return "red"
    if line.startswith("+"):
        return "green"
    if line.startswith("\\"):
        return "yellow"
    return "dim"
