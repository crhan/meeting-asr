# AGENTS.md

## Completion Notes

- Root CLI uses `add_completion=False`; keep `completion_init()` in `src/app/cli.py`.
- Typer completion scripts use `complete_bash` / `complete_zsh` style instructions.
- Without Typer's completion classes registered at startup, runtime completion can emit `plain,xxx` values or reject the shell instruction.
- Do not reintroduce hand-written static command lists for bash/zsh/fish; generate scripts from the Typer/Click command tree.

## Typer / Click Notes

- Typer 0.26 vendored Click into the private `typer._click` package and dropped the external `click` dependency (`click` is no longer installed). The CLI presentation layer (help, parse-error panels, completion, exit codes) uses Typer's PUBLIC API only: `typer.core.Typer{Group,Command,Argument,Option}`, `typer.main.get_command`, `typer.Context`, `typer.BadParameter`/`Exit`/`Abort`. Never `import click`, and never import `typer._click` (private, no `__all__`, no stability promise).
- `src/app/presentation/cli/errors.py` discriminates parse exceptions (NoSuchOption / MissingParameter / BadParameter) by class name + attribute shape, and `typer_context.py` recognizes usage errors by `ctx`/`format_message`/`exit_code` shape — not by `isinstance`, because Typer no longer exposes those exception classes. This duck-typing is deliberate; do not "fix" it to `isinstance` + private imports, it would silently break localized errors.
- en/zh bilingual help is rendered by our own renderer (driven by locale / `--lang` / `MEETING_ASR_LANG`), independent of Typer; the 0.26 upgrade preserved it at zero cost. Keep it bilingual unless explicitly asked to go Chinese-only.

## Speaker Preview Notes

- IINA may ignore CLI-provided `--mpv-sub-file` for external subtitles.
- Its mpv log showed only `Loading external files in .../source/` and no open of `exports/subtitle.srt`.
- For IINA preview, stage a same-stem `.srt` next to the source video so IINA/mpv auto-loads it as a sidecar subtitle.

## Speaker Apply Notes

- `meeting-asr project speakers apply --map N=Name` is a patch operation by default. It must merge into saved speaker names instead of replacing `speaker_map.json`.
- Destructive replacement requires the explicit `--replace` flag.
- When merging, preserve `speaker_person_map.json` entries only for unchanged names. If a speaker name changes without a new person id, drop that speaker's stale voiceprint link.

## ASR Postprocess Notes

- Filler-only speaker removal happens in `src/app/postprocess.py` during `parse_transcription_result()`.
- It only affects newly normalized ASR output. Existing project artifacts keep old `asr/sentences.json` and `project.json` speaker IDs until reprocessed from `asr/raw_result.json` or retranscribed.
- Backchannel-heavy tracks can contain short fragments like `对对对` or `就是可以再理一下了`; treat them as low-information tracks instead of requiring every sentence to match the exact filler word list.
- A real attendee may appear only as short backchannel utterances. Before declaring that a person is absent after rerun, compare `asr/raw_result.json` speaker IDs against normalized `asr/sentences.json`; `load_transcript_result()` and `project show` can still hide a raw speaker through the low-information filter.

## Project TUI Notes

- `src/app/commands/project.py` is already oversized; do not put Textual UI implementations there.
- Keep project selection/review UI modules separate, such as `src/app/project_tui.py` and `src/app/speaker_tui.py`, and leave command modules as thin Typer adapters.
- Project Review inline text correction mutates in-memory `SentenceSegment.text` before final save. Keep a separate loaded-text baseline for correction diffs; otherwise re-editing a staged sentence compares version 2 -> version 3 instead of original -> version 3.
- Terminal IME composition may not reach Textual text widgets even though bracketed paste works. Keep paste-friendly editing and an external-editor fallback for Chinese correction text.
- Voiceprint Review result rendering has two different meanings: current-project score changes are the expected result of new embeddings, so `changed-best` should read as green success there; historical reverse checks are regression risk and should keep warning/critical colors.

## Project Identity Notes

- Do not build new project identity from creation date or title. That created duplicate projects for the same video and made IDs change across runs.
- Project IDs are content-based (`p-<sha16>`) and default project directories use the same string. `project create` / `project run` must reuse an existing project for the same source video even when an explicit `--project-dir` is provided; `--project-dir` is only the desired path for a brand-new source.
- For deliberate same-source experiments, use `--variant <name>` so the project id becomes `p-<sha16>-v-<name>`. Do not create multiple directories with the same `project_id` to compare ASR settings.
- `project_id` is the only project identity printed by `project list` and generated next-step commands. Do not add numeric project list shortcuts back.

## Install And Verification Notes

- Agent-side development and verification must run through `uv run ...` in the current checkout or worktree. Do not use the global `meeting-asr` binary to validate code you just edited.
- User-facing validation on the main checkout should use the global editable tool installed by `scripts/install-tool.sh`.
- Before claiming a checkout is active in the global `meeting-asr`, verify `which meeting-asr`, imported `app.__file__`, and `meeting_asr-*.dist-info/direct_url.json`.
- `direct_url.json` without `{"editable": true}` means the global command imports a packaged wheel snapshot, not live checkout code.
- If working from a temporary worktree, do not repoint the global editable install to that worktree unless the user explicitly asks for it.
- `scripts/install-tool.sh` defaults to editable mode for local development. `scripts/install-tool.sh --wheel` is only for release or formal user-install simulation.
- Historical memory entries that mention raw `uv tool install --editable . --force` are stale. Use `scripts/install-tool.sh`; only pass `--force` for executable conflicts.
- Local voiceprint embedding uses `local-speechbrain` as the default provider. SpeechBrain, torch, and torchaudio are standard dependencies, not a `local-voiceprint` extra; do not suggest `uv sync --extra local-voiceprint`.

## File Copy Notes

- Do not validate destructive project workflows on `rsync --link-dest` or other hardlink-based copies. Project outputs are rewritten in place during ASR, and hardlinked test copies can mutate the user's real project artifacts.
- If a non-destructive validation copy is needed, use a plain copy for writable metadata/output files (`project.json`, `asr/`, `exports/`, `speakers/`) and only link immutable large media (`source/`, `audio/`) when the command will not rewrite them.
- Project run may prune only the managed copy under `project/source/` after `audio/audio.*` exists; never delete `manifest.source.original_path`. Reruns must prefer existing `audio/audio.*`, and OSS reuse must verify the object still exists before skipping upload.

## Worktree / Merge Notes

- Worktree 工作的交付单位是「分支 + PR」,不是「直接改 main」。触发条件是「人在 worktree / 非 main 分支,且要把活送进 main」,**不是「调用过 `EnterWorktree`」**——会话可能一启动 CWD 就被钉在 worktree 里(born-in-worktree),那时 `EnterWorktree` 从没被调用、`ExitWorktree` 对它是 no-op(只管本会话 `EnterWorktree` 建的)、`EnterWorktree({path})` 也够不到主 checkout(只收 `.claude/worktrees/` 下的 worktree),工具链结构上没法把你送回 main 本地合,所以更要走 PR。
- 干完 → 推分支 → `gh pr create`(本仓库 remote 是 GitHub `crhan/meeting-asr`)→ 让用户合 PR。**绝不 `git push origin HEAD:main` 或任何形式直接推 main。**
- 为什么不直接推 main:① 绕过评审;② 悄悄推进 origin/main,主 checkout `/Users/ruohan.chen/project/meeting-asr` 的 main 立刻 stale,得手动 `git pull` 才同步;③ 一旦不是干净 fast-forward,要么被拒、要么诱导 `--force`(灾难)。"碰巧能跑"≠正确。
- 唯一可本地合的情形:会话**从主 checkout 起**、用 `EnterWorktree` 进 worktree、再 `ExitWorktree({action:"keep"})` 回到主 checkout,此时在主 checkout 里 `git merge --ff-only` 合本地 main 是干净的。除此之外一律 PR。
- `ExitWorktree({action:"remove"})` 只删本会话 `EnterWorktree` 建的 worktree;born-in-worktree 的目录不会自动清,也不会在会话结束时弹 keep/remove 提示,需在主 checkout 手动 `git worktree remove` + `git branch -d`。
