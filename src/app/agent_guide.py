"""Runtime guide text for LLM agents using Meeting-ASR."""

from __future__ import annotations

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

## Rerun And Caching

Same-source runs reuse the content-based project id unless `--variant <name>` is
set. Use variants only for deliberate experiments.

Retry paths:

```bash
meeting-asr project rerun <project-id> --no-progress
meeting-asr project run <same-video> --no-progress --agent-log
```

Reusable audio is the durable ASR input. If `project show --json` reports an
audio path, later ASR retries should use it instead of extracting from video
again. `project rerun` is the explicit command for that path; `project transcribe`
is the lower-level compatible command kept for existing scripts. The
project-managed copy under `source/` may be removed after audio is prepared; do
not assume the staged video still exists. Never delete the user's original file
outside the project directory.

OSS object keys are stable per project audio object. Let Meeting-ASR decide
whether it can reuse an existing OSS object or needs to upload. Do not handcraft
signed URLs or log secret-bearing URLs.

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

## Review And Voiceprints

Speaker review state has two different ignore concepts:

- Project ignored speakers live in `speakers/speaker_ignore.json` and
  `manifest.speakers.ignored`. Treat them as intentionally ignored review noise,
  not unresolved identities.
- Voiceprint samples have lifecycle statuses: `active`, `verified-active`,
  `quarantined`, `rejected`, and `invalidated`. Matching uses only `active` and
  `verified-active`. `quarantined` and `rejected` samples are retained for audit
  but must not participate in matching. `invalidated` marks samples whose
  sentence was reassigned to another speaker; rows, clips, and embeddings are
  kept so the status can be restored instead of re-capturing audio.

Use `meeting-asr project speakers match <project-id> --json` for machine checks.
Use `meeting-asr voiceprint review <project-id>` only when the user wants an
interactive review session.

Current-project voiceprint evaluation and historical reverse checks have
different meanings. A current-project `changed-best` result usually means new
embeddings improved the target project. A historical reverse-check warning means
the same new embedding may regress an older project and should be treated as
risk.

## Non-Interactive

For long jobs, combine `--no-progress --agent-log` where the command supports it.
Avoid TUI commands unless the user explicitly wants interactive review. Commands
marked `interactive=true` can open Textual UI, an editor, or audio playback.

Never log secrets from config or environment. `config show` hides secrets unless
the user explicitly asks to reveal them.

## Reporting Back

When handing results to a user, include concrete evidence, not just "done":

- project id and project directory
- command path used (`uv run meeting-asr` vs installed `meeting-asr`)
- final transcript/subtitle paths when relevant
- unresolved speaker count or review status when relevant
- tests or smoke commands run

Do not claim ASR, review, or voiceprint work is complete from filesystem presence
alone. Check `project show --json`, `project status --json`, or the command's
final summary.

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


def build_agent_guide(*, schema_version: int, side_effect_enum: tuple[str, ...]) -> str:
    """
    Build the rendered agent guide text.

    Args:
        schema_version: Discovery envelope schema version.
        side_effect_enum: Stable side-effect labels used by ``commands --json``.

    Returns:
        Markdown guide with runtime constants inserted.
    """
    return AGENT_GUIDE_TEMPLATE.replace(
        "__SCHEMA_VERSION__", str(schema_version)
    ).replace("__SIDE_EFFECT_ENUM__", ", ".join(side_effect_enum))
