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

## Install Verification Notes

- The user-facing `meeting-asr` may be a uv tool wrapper under `~/.local/bin`, importing packaged code from `~/.local/share/uv/tools/meeting-asr/.../site-packages`.
- Before claiming a worktree change is active in the CLI, verify `which meeting-asr`, `app.__file__`, and `meeting_asr-*.dist-info/direct_url.json`.
- `direct_url.json` without `{"editable": true}` means the global command is not editable, even if it was installed from a local checkout.
