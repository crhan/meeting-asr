# AGENTS.md

## Business Data vs Code (配置 vs 代码) — 必须时刻遵守

- **业务相关内容绝不进代码库。** 具体专名、人名、产品名、术语映射（例如 `iSee` 是我们负责的平台、`IC` 是独立开发者，二者语境完全不同）属于**用户数据**，不是代码。它们只能存在于配置 / 跨项目纠错词库（`meeting-asr lexicon`，落盘在 `$XDG_DATA_HOME/meeting-asr/lexicon/lexicon.sqlite`），**永远不要**把它们硬编码成源码里的字面量或映射表。
- **判断方法（每次新增逻辑都先问）：** 这是「通用机制」还是「具体业务知识」？机制（怎么纠错、怎么判别语境、怎么调用 LLM）→ 进代码；具体的词、含义、映射、谁是谁 → 进配置/lexicon。任何要写 `"IC" -> "iSee"` 之类字面量的冲动，都是信号：它该进 lexicon，不该进 `src/`。
- **已证伪的死路（别重走）：** DashScope fun-asr 的自定义热词 vocabulary 对 `iSee` **完全不生效**。受控实验（同一段音频，无 vocab / `{"text":"iSee","weight":4}` / 追加 `"lang":"en"` 三条件）输出逐字一致，仍出 `IC`。`lang=en` 也无效。因此 ASR 提交阶段的热词偏置治不了「IC/iSee」同音错误。`corrections/asr_hotwords.json` 现在只用于**记录本次提交了哪些热词**（可观测性），不要再指望它纠正识别结果。
- **正确的修复层是后处理（polish / local_correction），且必须配置化：** 用 LLM 结合上下文语境判别（IC=人 vs iSee=平台，语境差异大、易分辨），判别依据来自 lexicon 的词条 + 别名 + context，不得把具体词写进代码。

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

## Speaker Resplit Notes（ASR under-split 救援）

- ASR diarizer 会 **under-split**：多个真人塌缩进同一条 speaker track。`speaker_stabilization` 原本只能在**已存在的项目 speaker 之间**挪句子（`_target_speaker_by_person` 只收 `assigned_person_id` 非空的人），所以「库里有声纹、但本项目还没建 track 的人」（例如某段会议里只露几句的第三人）永远救不出来。`src/app/speaker_resplit.py` 补这个缺口，作为 `stabilize_project_speakers` 的**一次性前置阶段**（迭代前跑一次，避免跨轮 ping-pong / 重复 mint），受 `project run --speaker-resplit/--no-speaker-resplit` 控制（默认开），well-split 项目下分析为空、纯 no-op。
- **聚类优先，锚定干净库向量，绝不用本 track 质心判离群**：under-split 的 track 质心是多人混合体，用它判「谁离群」是循环论证。正确做法：把句子按**逐句最佳库匹配**分组，再用**该组质心**（多句平均、低噪）去验证。promotion 与 residue 判据**刻意非对称**——promotion 有强正证据（质心贴某库人 + 领先当前指派人），2 句即可（可靠性看**总时长**不看句数，故 `min_group_seconds` 才是真闸门）；residue 只有弱负证据（谁都不像），`residue_match_floor` 压到 0.40，且整簇质心去噪后仍不匹配任何库人才入桶（防把「曹仁音质差的句子」误拉）。阈值都在 `ResplitParams`，dry-run 校准过。
- 落地：promotion 给库内新人 **mint 新 speaker_id**（`max(detected)+1`，单调不复用）并**显式 seed 名字 + person_map**；residue 全部塞进**一个匿名 unknown 桶**（新 id，不 seed，靠 review 看见——`speaker_id=None` 会被 review UI 过滤掉，所以桶**必须**是真实整数 id）。seed 之所以不被随后的 rematch 覆盖，是靠 `apply_project_speakers` 的**合并语义**（`_resolve_speaker_person_mapping` 在「名字未变」时保留既有 person_map）；unknown 桶的 probe 必 < rematch 0.75 阈值，故 rematch 不会给它命名。只读审计落 `speakers/speaker_resplit.json`。
- **血泪坑（验证时差点踩）**：`apply_project_sentence_reassignments` → `_invalidate_overlapping_voiceprint_samples` 会按 `project_id` + 原 speaker + 时间重叠**删全局声纹库的样本**。project_id 是内容寻址的，所以**拷贝项目做端到端验证 + `--apply` 时，若 store 指向全局库，会误删真实库样本**。务必把 voiceprints.sqlite 也拷一份，并用 `resplit --apply --store-dir <拷贝>` 隔离（匹配只读库 BLOB，不需要 clips/normalized 目录）。dry-run（不加 `--apply`）只读不删，安全。
- 预览/手动应用：`meeting-asr project speakers resplit <proj>` 默认 dry-run（打印 promotions / unknown 桶 / near-miss + 证据分数，零写盘），`--apply` 调 `apply_project_resplit` 直接修一个已处理完的项目（不必重跑 `project run`）；拷贝项目上务必配 `--store-dir` 隔离声纹库。**空声纹库 / 选错 model 导致库为空时整个分析 no-op**（否则"谁都不像"会把正常 track 误塞进桶）。
- **dry-run「零写盘」靠 `analyze_project_resplit(read_only=True)` 兑现，别把它当默认行为删掉**：嵌入本身要落 probe clip（embed 必须读音频文件）+ persist `tmp/speaker_cluster/clip_embeddings.json`，默认会写进项目 tmp/。read_only 路径把**暖缓存仍从真实项目读**进内存复用，但把 clip 抽取 + cache persist 整体重定向到一次性 `TemporaryDirectory`（`_ClusterContext.project_root` 换成 scratch，`source` 仍指真实音频；clip 持有的是向量不是文件路径，嵌完即弃）。CLI dry-run 传 `read_only=True`，`project run` / `--apply` 不传（那两条**该**写 cache 暖后续嵌入）。谁要把 read_only 去掉或让 dry-run 也 persist，就破坏了预览契约（实测参考项目 dry-run 后 635 个 tmp/speakers/exports 文件 mtime 零变化）。
