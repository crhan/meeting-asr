"""Agent-facing self-discovery commands for the Meeting-ASR CLI."""

from __future__ import annotations

import difflib
import platform
import re
from importlib.metadata import PackageNotFoundError
from importlib.metadata import version as package_version
from typing import Any

import click
import typer

from app.presentation.cli.json_output import emit_json

SCHEMA_VERSION = 1
SIDE_EFFECT_ENUM = (
    "fs-read",
    "fs-write",
    "config-write",
    "network-io",
    "oss-upload",
    "llm-call",
    "subprocess",
    "audio-playback",
    "destructive",
)

COMMANDS_META: list[dict[str, Any]] = [
    {
        "name": "agent-guide",
        "group": "agent",
        "summary": "Runtime guide for LLM agents; supports section slicing.",
        "args": [],
        "supports_json": True,
        "side_effects": [],
        "needs_sudo": False,
        "interactive": False,
        "exit_codes": [0, 2],
        "examples": [
            "meeting-asr agent-guide",
            "meeting-asr agent-guide --list-sections",
            "meeting-asr agent-guide --section workflow --json",
        ],
    },
    {
        "name": "commands",
        "group": "agent",
        "summary": "Machine-readable command metadata, including side effects.",
        "args": [],
        "supports_json": True,
        "side_effects": [],
        "needs_sudo": False,
        "interactive": False,
        "exit_codes": [0],
        "examples": [
            "meeting-asr commands --json",
            "meeting-asr commands --schema",
        ],
    },
    {
        "name": "version",
        "group": "agent",
        "summary": "Version and supported feature flags.",
        "args": [],
        "supports_json": True,
        "side_effects": [],
        "needs_sudo": False,
        "interactive": False,
        "exit_codes": [0],
        "examples": ["meeting-asr version", "meeting-asr version --json"],
    },
    {
        "name": "doctor",
        "group": "diagnostic",
        "summary": "Check runtime dependencies and global configuration.",
        "args": [],
        "supports_json": True,
        "side_effects": ["fs-read"],
        "conditional_side_effects": {
            "--oss-upload-probe": ["network-io", "oss-upload"],
        },
        "needs_sudo": False,
        "interactive": False,
        "exit_codes": [0, 1, 2],
        "examples": [
            "meeting-asr doctor",
            "meeting-asr doctor --full --json",
        ],
    },
    {
        "name": "paths",
        "group": "diagnostic",
        "summary": "Print XDG config, data, cache, project, voiceprint, and lexicon paths.",
        "args": [],
        "supports_json": True,
        "side_effects": ["fs-read"],
        "needs_sudo": False,
        "interactive": False,
        "exit_codes": [0],
        "examples": ["meeting-asr paths", "meeting-asr paths --json"],
    },
    {
        "name": "project run",
        "group": "project",
        "summary": "Create or reuse a project, prepare audio, run ASR, match speakers, and summarize.",
        "args": [{"name": "input_file", "required": True}],
        "supports_json": False,
        "side_effects": [
            "fs-read",
            "fs-write",
            "network-io",
            "oss-upload",
            "llm-call",
            "subprocess",
        ],
        "side_effect_notes": [
            "--no-summarize and --no-polish reduce LLM/network calls.",
            "--oss-upload=false disables OSS upload when a usable URL is already available.",
        ],
        "needs_sudo": False,
        "interactive": False,
        "exit_codes": [0, 1, 2, 130],
        "examples": [
            "meeting-asr project run ~/Downloads/meeting.mp4 --no-progress --agent-log",
            "meeting-asr project run ~/Downloads/meeting.mp4 --variant experiment-a",
        ],
    },
    {
        "name": "project prepare",
        "group": "project",
        "summary": "Extract reusable project audio without submitting ASR.",
        "args": [{"name": "project", "required": True}],
        "supports_json": False,
        "side_effects": ["fs-read", "fs-write", "subprocess"],
        "needs_sudo": False,
        "interactive": False,
        "exit_codes": [0, 1, 2],
        "examples": ["meeting-asr project prepare <project-id> --no-progress"],
    },
    {
        "name": "project transcribe",
        "group": "project",
        "summary": "Run ASR for an existing project and write structured artifacts.",
        "args": [{"name": "project", "required": True}],
        "supports_json": False,
        "side_effects": ["fs-read", "fs-write", "network-io", "oss-upload"],
        "needs_sudo": False,
        "interactive": False,
        "exit_codes": [0, 1, 2, 130],
        "examples": ["meeting-asr project transcribe <project-id> --no-progress"],
    },
    {
        "name": "project list",
        "group": "project",
        "summary": "List projects in the XDG project store.",
        "args": [],
        "supports_json": True,
        "side_effects": ["fs-read"],
        "needs_sudo": False,
        "interactive": False,
        "exit_codes": [0],
        "examples": [
            "meeting-asr project list",
            "meeting-asr project list --json",
        ],
    },
    {
        "name": "project show",
        "group": "project",
        "summary": "Show one project manifest, artifact paths, and speaker state.",
        "args": [{"name": "project", "required": True}],
        "supports_json": True,
        "side_effects": ["fs-read"],
        "needs_sudo": False,
        "interactive": False,
        "exit_codes": [0, 1, 2],
        "examples": [
            "meeting-asr project show p-292d10c1232b79a0",
            "meeting-asr project show p-292d10c1232b79a0 --json",
        ],
    },
    {
        "name": "project status",
        "group": "project",
        "summary": "Print a compact project status payload.",
        "args": [{"name": "project", "required": True}],
        "supports_json": True,
        "side_effects": ["fs-read"],
        "needs_sudo": False,
        "interactive": False,
        "exit_codes": [0, 1, 2],
        "examples": ["meeting-asr project status <project-id> --json"],
    },
    {
        "name": "project review",
        "group": "review",
        "summary": "Open the recommended human review TUI for speaker and transcript correction.",
        "args": [{"name": "project", "required": False}],
        "supports_json": False,
        "side_effects": ["fs-read", "fs-write", "audio-playback"],
        "needs_sudo": False,
        "interactive": True,
        "exit_codes": [0, 1, 2, 130],
        "examples": ["meeting-asr project review <project-id>"],
    },
    {
        "name": "project speakers match",
        "group": "speaker",
        "summary": "Match project speakers against the global voiceprint registry.",
        "args": [{"name": "project", "required": True}],
        "supports_json": True,
        "side_effects": ["fs-read", "fs-write"],
        "needs_sudo": False,
        "interactive": False,
        "exit_codes": [0, 1, 2],
        "examples": ["meeting-asr project speakers match <project-id> --json"],
    },
    {
        "name": "project speakers apply",
        "group": "speaker",
        "summary": "Apply confirmed speaker mappings; merge mode is the default.",
        "args": [{"name": "project", "required": True}],
        "supports_json": False,
        "side_effects": ["fs-read", "fs-write"],
        "conditional_side_effects": {"--replace": ["fs-write"]},
        "needs_sudo": False,
        "interactive": False,
        "exit_codes": [0, 1, 2],
        "examples": [
            'meeting-asr project speakers apply <project-id> --map "1=Alice"',
        ],
    },
    {
        "name": "project speakers review",
        "group": "speaker",
        "summary": "Open interactive speaker identity review.",
        "args": [{"name": "project", "required": True}],
        "supports_json": False,
        "side_effects": ["fs-read", "fs-write", "audio-playback"],
        "needs_sudo": False,
        "interactive": True,
        "exit_codes": [0, 1, 2, 130],
        "examples": ["meeting-asr project speakers review <project-id>"],
    },
    {
        "name": "project transcript list",
        "group": "transcript",
        "summary": "List transcript artifacts for a project.",
        "args": [{"name": "project", "required": True}],
        "supports_json": True,
        "side_effects": ["fs-read"],
        "needs_sudo": False,
        "interactive": False,
        "exit_codes": [0, 1, 2],
        "examples": ["meeting-asr project transcript list <project-id> --json"],
    },
    {
        "name": "project transcript show",
        "group": "transcript",
        "summary": "Print a transcript artifact such as corrected text or SRT.",
        "args": [{"name": "project", "required": True}],
        "supports_json": False,
        "side_effects": ["fs-read"],
        "needs_sudo": False,
        "interactive": False,
        "exit_codes": [0, 1, 2],
        "examples": [
            "meeting-asr project transcript show <project-id> --kind corrected",
        ],
    },
    {
        "name": "project correct edit",
        "group": "transcript",
        "summary": "Prepare or apply transcript correction proposals.",
        "args": [{"name": "project", "required": True}],
        "supports_json": False,
        "side_effects": ["fs-read", "fs-write", "llm-call", "subprocess"],
        "side_effect_notes": ["--no-ai disables the LLM call."],
        "needs_sudo": False,
        "interactive": True,
        "exit_codes": [0, 1, 2, 130],
        "examples": ["meeting-asr project correct edit <project-id> --no-ai"],
    },
    {
        "name": "project delete",
        "group": "project",
        "summary": "Move a project to trash; --permanent physically deletes it.",
        "args": [{"name": "project", "required": True}],
        "supports_json": False,
        "side_effects": ["fs-read", "fs-write"],
        "conditional_side_effects": {"--permanent": ["destructive"]},
        "needs_sudo": False,
        "interactive": True,
        "exit_codes": [0, 1, 2, 130],
        "examples": [
            "meeting-asr project delete <project-id> --yes",
            "meeting-asr project delete <project-id> --permanent --yes",
        ],
    },
    {
        "name": "voiceprint review",
        "group": "voiceprint",
        "summary": "Review candidate samples and global voiceprint matches in a TUI.",
        "args": [{"name": "project", "required": False}],
        "supports_json": False,
        "side_effects": ["fs-read", "fs-write", "audio-playback"],
        "needs_sudo": False,
        "interactive": True,
        "exit_codes": [0, 1, 2, 130],
        "examples": ["meeting-asr voiceprint review <project-id>"],
    },
    {
        "name": "voiceprint capture",
        "group": "voiceprint",
        "summary": "Capture voiceprint samples from confirmed project speakers.",
        "args": [{"name": "project", "required": True}],
        "supports_json": False,
        "side_effects": ["fs-read", "fs-write", "subprocess"],
        "needs_sudo": False,
        "interactive": True,
        "exit_codes": [0, 1, 2, 130],
        "examples": ["meeting-asr voiceprint capture <project-id>"],
    },
    {
        "name": "voiceprint embed",
        "group": "voiceprint",
        "summary": "Generate or rebuild local voiceprint embeddings.",
        "args": [],
        "supports_json": False,
        "side_effects": ["fs-read", "fs-write"],
        "needs_sudo": False,
        "interactive": False,
        "exit_codes": [0, 1, 2],
        "examples": [
            "meeting-asr voiceprint embed",
            "meeting-asr voiceprint embed --rebuild",
        ],
    },
    {
        "name": "voiceprint list",
        "group": "voiceprint",
        "summary": "List speakers known to the global voiceprint registry.",
        "args": [],
        "supports_json": True,
        "side_effects": ["fs-read"],
        "needs_sudo": False,
        "interactive": False,
        "exit_codes": [0],
        "examples": ["meeting-asr voiceprint list --json"],
    },
    {
        "name": "lexicon list",
        "group": "lexicon",
        "summary": "List accepted correction lexicon entries.",
        "args": [],
        "supports_json": True,
        "side_effects": ["fs-read"],
        "needs_sudo": False,
        "interactive": False,
        "exit_codes": [0],
        "examples": ["meeting-asr lexicon list --json"],
    },
    {
        "name": "lexicon hotwords sync",
        "group": "lexicon",
        "summary": "Sync local hotword candidates to DashScope ASR hotwords.",
        "args": [],
        "supports_json": True,
        "side_effects": ["fs-read", "fs-write", "network-io"],
        "supports_dry_run": True,
        "needs_sudo": False,
        "interactive": False,
        "exit_codes": [0, 1, 2],
        "examples": ["meeting-asr lexicon hotwords sync --dry-run --json"],
    },
    {
        "name": "oss upload",
        "group": "oss",
        "summary": "Upload a local file to OSS and print a signed URL.",
        "args": [{"name": "file", "required": True}],
        "supports_json": False,
        "side_effects": ["fs-read", "network-io", "oss-upload"],
        "needs_sudo": False,
        "interactive": False,
        "exit_codes": [0, 1, 2],
        "examples": ["meeting-asr oss upload ./audio.wav"],
    },
    {
        "name": "oss lifecycle set",
        "group": "oss",
        "summary": "Create or update an OSS lifecycle rule.",
        "args": [],
        "supports_json": False,
        "side_effects": ["network-io", "config-write"],
        "needs_sudo": False,
        "interactive": False,
        "exit_codes": [0, 1, 2],
        "examples": ["meeting-asr oss lifecycle set --prefix meeting-asr/ --days 7"],
    },
    {
        "name": "config set",
        "group": "config",
        "summary": "Write one global config key.",
        "args": [
            {"name": "key", "required": True},
            {"name": "value", "required": True},
        ],
        "supports_json": False,
        "side_effects": ["config-write"],
        "needs_sudo": False,
        "interactive": False,
        "exit_codes": [0, 1, 2],
        "examples": ['meeting-asr config set ui.editor "code --wait"'],
    },
    {
        "name": "completion install",
        "group": "agent",
        "summary": "Install a generated shell completion script.",
        "args": [{"name": "shell", "required": True}],
        "supports_json": False,
        "side_effects": ["fs-write"],
        "needs_sudo": False,
        "interactive": False,
        "exit_codes": [0, 1, 2],
        "examples": ["meeting-asr completion install zsh"],
    },
    {
        "name": "help",
        "group": "agent",
        "summary": "Render localized root or nested command help.",
        "args": [{"name": "command_path", "required": False, "variadic": True}],
        "supports_json": False,
        "side_effects": [],
        "needs_sudo": False,
        "interactive": False,
        "exit_codes": [0, 2],
        "examples": ["meeting-asr help", "meeting-asr help project run"],
    },
]

COMMANDS_DATA_SCHEMA: dict[str, Any] = {
    "$schema": "https://json-schema.org/draft/2020-12/schema",
    "title": "meeting-asr commands --json data schema",
    "type": "object",
    "required": ["schema_version", "version", "commands", "side_effect_enum"],
    "properties": {
        "schema_version": {"const": SCHEMA_VERSION},
        "version": {"type": "string"},
        "side_effect_enum": {
            "type": "array",
            "items": {"enum": list(SIDE_EFFECT_ENUM)},
        },
        "commands": {
            "type": "array",
            "items": {
                "type": "object",
                "required": [
                    "name",
                    "group",
                    "summary",
                    "supports_json",
                    "side_effects",
                    "needs_sudo",
                    "interactive",
                    "exit_codes",
                    "examples",
                ],
                "properties": {
                    "name": {"type": "string"},
                    "group": {"type": "string"},
                    "summary": {"type": "string"},
                    "args": {"type": "array"},
                    "supports_json": {"type": "boolean"},
                    "supports_dry_run": {"type": "boolean"},
                    "side_effects": {
                        "type": "array",
                        "items": {"enum": list(SIDE_EFFECT_ENUM)},
                    },
                    "conditional_side_effects": {
                        "type": "object",
                        "additionalProperties": {
                            "type": "array",
                            "items": {"enum": list(SIDE_EFFECT_ENUM)},
                        },
                    },
                    "side_effect_notes": {
                        "type": "array",
                        "items": {"type": "string"},
                    },
                    "needs_sudo": {"type": "boolean"},
                    "interactive": {"type": "boolean"},
                    "exit_codes": {
                        "type": "array",
                        "items": {"type": "integer", "minimum": 0},
                    },
                    "examples": {
                        "type": "array",
                        "items": {"type": "string"},
                    },
                },
                "additionalProperties": True,
            },
        },
    },
    "additionalProperties": True,
}

AGENT_GUIDE_TEMPLATE = """# meeting-asr Agent Guide

This document is generated at runtime by `meeting-asr agent-guide`.
Repository development rules live in `AGENTS.md`; this file is the CLI contract.

## Onboarding

1. `meeting-asr agent-guide` - read this runtime contract.
2. `meeting-asr version --json` - check `data.supported_features`.
3. `meeting-asr commands --json` - inspect command metadata and side effects.
4. `meeting-asr commands --schema` - validate the metadata shape if needed.
5. `meeting-asr doctor --full --json` - verify local dependencies and config.
6. `meeting-asr project list --json` or `meeting-asr paths --json` - locate state.

Use `uv run meeting-asr ...` when validating source-code changes in this checkout.
Use the installed `meeting-asr` command when validating the user-facing editable tool.

## Workflow

Default non-interactive run:

```bash
meeting-asr project run <video> --no-progress --agent-log
```

`--agent-log` prints structured stage and heartbeat lines. It is the clean path for
long ASR jobs because agents can tell whether extraction, upload, ASR, matching, or
summarization is still progressing.

After a run, inspect:

```bash
meeting-asr project show <project-id> --json
meeting-asr project status <project-id> --json
meeting-asr project transcript list <project-id> --json
```

For human correction, prefer `meeting-asr project review <project-id>`.

## Identity And Paths

Project IDs are content based: `p-<sha16>`. The same source media should reuse the
same project; deliberate experiments must use `--variant <name>`.

Important paths are discoverable through:

```bash
meeting-asr paths --json
meeting-asr project show <project-id> --json
```

Project artifacts normally live under the XDG data directory. The project-managed
video copy can be pruned after reusable audio exists; the original outside the
project is not the identity source after creation. Re-runs should use the stored
project audio when available.

## JSON And Discovery

Discovery commands use a small envelope:

```json
{
  "schema_version": __SCHEMA_VERSION__,
  "cmd": "commands",
  "ok": true,
  "data": {},
  "error": null,
  "code": 0,
  "hints": []
}
```

Stable discovery entrypoints:

- `meeting-asr version --json`
- `meeting-asr commands --json`
- `meeting-asr commands --schema`
- `meeting-asr agent-guide --section <name> --json`

Existing business commands keep their historical payloads. Do not assume every
command uses the discovery envelope; check `commands --json`.

## Side Effects

`commands --json` exposes `side_effects`, `conditional_side_effects`,
`interactive`, and `needs_sudo` for each important command.

Side effect enum:

```
__SIDE_EFFECT_ENUM__
```

High-risk commands:

- `project run` writes project artifacts and may call DashScope, OSS, ffmpeg, and LLM summarization.
- `project delete --permanent` physically removes project data.
- `voiceprint capture/embed` writes the global voiceprint store.
- `lexicon hotwords sync` writes remote DashScope hotword state.
- `oss upload` uploads bytes to OSS.

## Non-Interactive

For long jobs, combine `--no-progress --agent-log` where the command supports it.
Avoid TUI commands unless the user explicitly wants interactive review. Commands
marked `interactive=true` can open Textual UI, an editor, or audio playback.

Never log secrets from config or environment. `config show` hides secrets unless
the user explicitly asks to reveal them.

## Troubleshooting

Fast baseline:

```bash
meeting-asr doctor --full --json
meeting-asr paths --json
meeting-asr project show <project-id> --json
```

If two CLI instances behave differently, check both versions and import roots:

```bash
which meeting-asr
meeting-asr version --json
uv run meeting-asr version --json
```

For project run hangs, rerun with `--no-progress --agent-log` before claiming root
cause. The stage log is the observable contract.

## Completion

Root completion is intentionally custom. Do not reintroduce static command lists.
Generate shell scripts from the Typer command tree through:

```bash
meeting-asr completion zsh
meeting-asr completion bash
meeting-asr completion fish
```
"""


def agent_guide_command(
    section: str | None = typer.Option(
        None,
        "--section",
        help="Print only one guide section. Matching is case-insensitive and fuzzy.",
    ),
    list_sections: bool = typer.Option(
        False, "--list-sections", help="List available guide sections."
    ),
    as_json: bool = typer.Option(False, "--json", help="Print JSON envelope."),
) -> None:
    """Print the runtime agent guide."""
    guide = _build_agent_guide()
    sections = _split_agent_guide_sections(guide)
    section_names = list(sections)
    if list_sections:
        data = {"available_sections": section_names, "section_count": len(sections)}
        if as_json:
            emit_json(envelope("agent-guide", data=data))
            return
        _print_sections(section_names)
        return
    if section:
        _print_requested_section(section, sections, section_names, as_json=as_json)
        return
    if as_json:
        emit_json(
            envelope(
                "agent-guide",
                data={"markdown": guide, "available_sections": section_names},
            )
        )
        return
    typer.echo(guide)


def commands_command(
    as_json: bool = typer.Option(False, "--json", help="Print JSON envelope."),
    schema: bool = typer.Option(
        False, "--schema", help="Print the JSON schema for commands --json data."
    ),
) -> None:
    """Print command metadata for agents and scripts."""
    if schema:
        payload = COMMANDS_DATA_SCHEMA
        emit_json(envelope("commands", data=payload) if as_json else payload)
        return
    payload = commands_payload()
    if as_json:
        emit_json(envelope("commands", data=payload))
        return
    _print_commands_table(payload["commands"])


def version_command(
    as_json: bool = typer.Option(False, "--json", help="Print JSON envelope."),
) -> None:
    """Print the installed Meeting-ASR version."""
    if as_json:
        emit_json(version_envelope())
        return
    typer.echo(f"meeting-asr {_installed_version()}")


def commands_payload() -> dict[str, Any]:
    """
    Build the command metadata payload.

    Returns:
        Machine-readable metadata under the schema advertised by ``--schema``.
    """
    return {
        "schema_version": SCHEMA_VERSION,
        "version": _installed_version(),
        "side_effect_enum": list(SIDE_EFFECT_ENUM),
        "commands": COMMANDS_META,
    }


def version_envelope() -> dict[str, Any]:
    """
    Build the version probe envelope.

    Returns:
        JSON envelope with feature flags that agents can use for compatibility.
    """
    data = {
        "version": _installed_version(),
        "schema_version": SCHEMA_VERSION,
        "python": platform.python_version(),
        "platform": platform.system().lower(),
        "supported_features": {
            "agent_guide": True,
            "agent_guide_sections": True,
            "commands_json": True,
            "commands_schema": True,
            "version_json": True,
            "version_subcommand": True,
            "discovery_envelope": True,
            "side_effects_enum": True,
            "localized_help": True,
            "help_subcommand": True,
            "project_agent_log": True,
            "project_id_content_hash": True,
            "reusable_project_audio": True,
            "oss_audio_reuse": True,
        },
        "entrypoints": {
            "guide": "meeting-asr agent-guide",
            "commands": "meeting-asr commands --json",
            "commands_schema": "meeting-asr commands --schema",
            "version": "meeting-asr version --json",
            "doctor": "meeting-asr doctor --full --json",
        },
    }
    return envelope("version", data=data)


def root_version_json_requested(args: list[str]) -> bool:
    """
    Return whether argv requests ``meeting-asr --version --json``.

    Args:
        args: Command-line arguments after the program name.

    Returns:
        True only when root-level version flags are present.
    """
    normalized = _root_flag_tokens(args)
    return "--version" in normalized and "--json" in normalized


def envelope(
    cmd: str,
    *,
    data: dict[str, Any] | None,
    ok: bool = True,
    error: str | None = None,
    code: int = 0,
    hints: list[str] | None = None,
) -> dict[str, Any]:
    """
    Build the stable discovery envelope.

    Args:
        cmd: Command name for log correlation.
        data: Command payload for successful results.
        ok: Whether the command succeeded.
        error: Human-readable error message.
        code: Intended process exit code.
        hints: Optional next-step commands or repair hints.

    Returns:
        JSON-compatible envelope.
    """
    return {
        "schema_version": SCHEMA_VERSION,
        "cmd": cmd,
        "ok": ok,
        "data": data,
        "error": error,
        "code": code,
        "hints": hints or [],
    }


def _installed_version() -> str:
    """Return the installed package version."""
    try:
        return package_version("meeting-asr")
    except PackageNotFoundError:
        return "0.0.0+local"


def _build_agent_guide() -> str:
    """Build markdown guidance for runtime agent consumption."""
    return AGENT_GUIDE_TEMPLATE.replace(
        "__SCHEMA_VERSION__", str(SCHEMA_VERSION)
    ).replace("__SIDE_EFFECT_ENUM__", ", ".join(SIDE_EFFECT_ENUM))


def _split_agent_guide_sections(markdown: str) -> dict[str, str]:
    """Split the guide into normalized H2 sections."""
    sections: dict[str, str] = {}
    current_name: str | None = None
    current_lines: list[str] = []
    for line in markdown.splitlines():
        match = re.match(r"^##\s+(.+?)\s*$", line)
        if match:
            if current_name is None and current_lines:
                sections["introduction"] = "\n".join(current_lines).rstrip() + "\n"
            elif current_name is not None:
                sections[current_name] = "\n".join(current_lines).rstrip() + "\n"
            current_name = _normalize_section_name(match.group(1))
            current_lines = [line]
            continue
        current_lines.append(line)
    if current_name is None:
        sections["introduction"] = "\n".join(current_lines).rstrip() + "\n"
    else:
        sections[current_name] = "\n".join(current_lines).rstrip() + "\n"
    return sections


def _normalize_section_name(title: str) -> str:
    """Normalize a markdown heading into a stable section name."""
    lowered = title.lower().replace("&", " and ")
    cleaned = re.sub(r"[^a-z0-9]+", "-", lowered).strip("-")
    return cleaned or "section"


def _match_section(requested: str, section_names: list[str]) -> str | None:
    """Return the best matching section name."""
    normalized = _normalize_section_name(requested)
    if normalized in section_names:
        return normalized
    matches = difflib.get_close_matches(normalized, section_names, n=1, cutoff=0.35)
    return matches[0] if matches else None


def _print_requested_section(
    requested: str,
    sections: dict[str, str],
    section_names: list[str],
    *,
    as_json: bool,
) -> None:
    """Print one requested section or fail with a helpful error."""
    match = _match_section(requested, section_names)
    if match is None:
        hints = [f"available sections: {', '.join(section_names)}"]
        suggestions = difflib.get_close_matches(
            _normalize_section_name(requested), section_names, n=3, cutoff=0.25
        )
        if suggestions:
            hints.insert(0, f"did you mean: {suggestions[0]}")
        if as_json:
            emit_json(
                envelope(
                    "agent-guide",
                    data=None,
                    ok=False,
                    error=f"unknown section: {requested}",
                    code=2,
                    hints=hints,
                )
            )
            raise typer.Exit(code=2)
        raise click.UsageError(f"Unknown section: {requested}. {hints[0]}")
    chunk = sections[match]
    if as_json:
        emit_json(
            envelope(
                "agent-guide",
                data={
                    "section": match,
                    "markdown": chunk,
                    "available_sections": section_names,
                },
            )
        )
        return
    typer.echo(chunk)


def _print_sections(section_names: list[str]) -> None:
    """Print available guide sections as plain lines."""
    typer.echo(f"Available sections ({len(section_names)}):")
    for name in section_names:
        typer.echo(f"  {name}")
    typer.echo("Use: meeting-asr agent-guide --section <name>")


def _print_commands_table(commands: list[dict[str, Any]]) -> None:
    """Print a compact human-readable command metadata table."""
    groups: dict[str, list[dict[str, Any]]] = {}
    for command in commands:
        groups.setdefault(command["group"], []).append(command)
    for group, rows in groups.items():
        typer.echo(group)
        for row in rows:
            flags = _command_flags(row)
            suffix = f" [{', '.join(flags)}]" if flags else ""
            typer.echo(f"  {row['name']:<28} {row['summary']}{suffix}")
        typer.echo()
    typer.echo("Machine-readable: meeting-asr commands --json")


def _command_flags(command: dict[str, Any]) -> list[str]:
    """Return short badges for one command metadata row."""
    flags: list[str] = []
    if command.get("supports_json"):
        flags.append("--json")
    if command.get("supports_dry_run"):
        flags.append("--dry-run")
    if command.get("interactive"):
        flags.append("interactive")
    if command.get("needs_sudo"):
        flags.append("sudo")
    side_effects = command.get("side_effects") or []
    flags.extend(side_effects)
    for trigger, effects in (command.get("conditional_side_effects") or {}).items():
        flags.append(f"{trigger}={'+'.join(effects)}")
    return flags


def _root_flag_tokens(args: list[str]) -> list[str]:
    """Return root flag tokens only, or an empty list when a subcommand is present."""
    tokens: list[str] = []
    index = 0
    while index < len(args):
        token = args[index]
        if token == "--lang":
            tokens.append(token)
            index += 2
            continue
        if token in {"--version", "--json", "--no-color", "--verbose", "-v"}:
            tokens.append(token)
            index += 1
            continue
        return []
    return tokens
