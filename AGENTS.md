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

## Project Merge Notes

- `project merge` 是无状态纯函数（实现在 `src/app/transcript_merge.py`，命令是 `project.py` 里的薄适配器）：读 N 个 project 产物 → 重建单一 `TranscriptResult` → 复用现有渲染器输出。**绝不回写任何 project**；`merge.json` 是本次输出包的只读清单，住 `--out` 目录内，不是 session 状态、不进 `projects_dir`。否决过 `session` 概念，因为它会再造一个身份层与 content-based `project_id` 打架（见 Project Identity Notes）。
- **必须从 `asr/sentences.json` 重渲染，不能拼接 `exports/transcript_named.txt`**。拼接预渲染文本无法重新归属说话人，跨段声纹归一就做不成。
- 跨段说话人归一身份键优先级：`vpp`（`speaker_person_map.json`，跨段最强主键，手动 apply 与 `project run` 自动稳定化都会写）> 显示名（`speaker_map.json`，排除占位名）> per-segment 匿名（绝不跨段合并）。归一做**两级坍缩**：先项目内（实测同一段内会有多个 local id 同 vpp / 同名）、再跨项目。权威人名优先取声纹库 `get_voiceprint_person(vpp).name`，库缺则回退段内本地名。
- name→vpp 提升默认开：某段仅命名、另一段同名带 vpp 时，把 name-only 折叠到该 vpp（实测同名→同 vpp 0 冲突）。同名落到多个 vpp 时**不提升**并 warn。`--no-name-to-vpp` 关闭。审计轨写在 `merge.json` 的 `identities[].promoted_from_name`。
- 占位名（`待确认发言人2` 等，正则在 `_PLACEHOLDER_NAME_RE`）视为无名，落匿名分支，绝不跨段误并。`name_fold` 做 NFKC + 去 IME 窄空格(U+202F/U+00A0 等) + 折叠空白；括号别名（`张辉洲(尺木)`）整串作 key，全/半角括号经 NFKC 归一。
- 时间轴是**连续打包**：第 k 段时间戳整体偏移 `Σ(前段 audio.duration_seconds)`，单调不重叠。真实墙钟（`meeting_time`）只进段界 header 和 `merge.json`，不进时间轴——否则中场休息会在字幕里punch 出几十分钟空洞。`render_named_srt` 的序号硬编码从 1 起，所以 SRT 必须用整段合并后的单一 `TranscriptResult` 一次性渲染，不能逐段拼。
- 段排序按 `meeting_time` 升序，但**实测数据 naive/aware 时区混存**，直接 `sorted(fromisoformat)` 抛 `TypeError`；解析时把 naive 一律当 +08:00（`_parse_timestamp`）。无法解析则回退 `created_at`，再不行回退命令行顺序并 warn（`--keep-order` 显式保持命令行序）。
- `ignored` speaker（`speaker_ignore.json`）按**匿名保留**处理（与单段命名导出语义一致：仍出现，但不具名），不丢弃、即使带 vpp 也不归并；`merge.json` 记 `ignored_speaker_count`。低信息过滤口径跨段统一（默认 `include_low_information=False`），`--include-low-information` 透传。
- 单段（N=1）走同一管线退化为直接命名导出（无段界 header）；重复传同一 project ref 去重并 warn，时间轴不翻倍。
