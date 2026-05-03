# AGENTS.md

## Completion Notes

- Root CLI uses `add_completion=False`; keep `completion_init()` in `src/app/cli.py`.
- Typer completion scripts use `complete_bash` / `complete_zsh` style instructions.
- Without Typer's completion classes registered at startup, runtime completion can emit `plain,xxx` values or reject the shell instruction.
- Do not reintroduce hand-written static command lists for bash/zsh/fish; generate scripts from the Typer/Click command tree.

## Speaker Preview Notes

- IINA may ignore CLI-provided `--mpv-sub-file` for external subtitles.
- Its mpv log showed only `Loading external files in .../source/` and no open of `exports/subtitle.srt`.
- For IINA preview, stage a same-stem `.srt` next to the source video so IINA/mpv auto-loads it as a sidecar subtitle.

## Project TUI Notes

- `src/app/commands/project.py` is already oversized; do not put Textual UI implementations there.
- Keep project selection/review UI modules separate, such as `src/app/project_tui.py` and `src/app/speaker_tui.py`, and leave command modules as thin Typer adapters.

## Project Identity Notes

- Do not build new project identity from creation date or title. That created duplicate projects for the same video and made IDs change across runs.
- New project IDs are content-based (`p-<sha16>`). `project create` / `project run` should reuse an existing project for the same source video when no explicit `--project-dir` is provided.
- Existing date/title IDs must keep resolving for backward compatibility; do not rewrite old manifests unless a migration command is added.
- Generated next-step commands must use stable `project_id` or paths, not project list `No.` values. Numeric `No.` is only a list shortcut and must not be printed as `Project No.` in command output.

## Install And Verification Notes

- Agent-side development and verification must run through `uv run ...` in the current checkout or worktree. Do not use the global `meeting-asr` binary to validate code you just edited.
- User-facing validation on the main checkout should use the global editable tool installed by `scripts/install-tool.sh`.
- Before claiming a checkout is active in the global `meeting-asr`, verify `which meeting-asr`, imported `app.__file__`, and `meeting_asr-*.dist-info/direct_url.json`.
- `direct_url.json` without `{"editable": true}` means the global command imports a packaged wheel snapshot, not live checkout code.
- If working from a temporary worktree, do not repoint the global editable install to that worktree unless the user explicitly asks for it.
- `scripts/install-tool.sh` defaults to editable mode for local development. `scripts/install-tool.sh --wheel` is only for release or formal user-install simulation.
- Historical memory entries that mention raw `uv tool install --editable . --force` are stale. Use `scripts/install-tool.sh`; only pass `--force` for executable conflicts.

## File Copy Notes

- Do not validate destructive project workflows on `rsync --link-dest` or other hardlink-based copies. Project outputs are rewritten in place during ASR, and hardlinked test copies can mutate the user's real project artifacts.
- If a non-destructive validation copy is needed, use a plain copy for writable metadata/output files (`project.json`, `asr/`, `exports/`, `speakers/`) and only link immutable large media (`source/`, `audio/`) when the command will not rewrite them.
