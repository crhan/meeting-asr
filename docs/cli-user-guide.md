# Meeting-ASR CLI 用户手册

## 1. 检查环境

```bash
meeting-asr doctor
meeting-asr doctor --full
meeting-asr doctor --full --json
```

`doctor` 默认是基础检查：本地环境、基础配置、编辑器、预览播放器和声纹依赖提示。
它不做网络写入，适合随手跑。`--full` 是完整集成检查：要求 OSS 配置完整，上传一个极小
文本对象，签 URL 读回，再删除，并且严格检查声纹 embedding。给 agent 或脚本用
`meeting-asr doctor --full --json`，JSON 字段和值保持英文稳定，不做本地化。
默认 provider 是 `local-speechbrain`，只检查本地依赖；切到 `bailian` 后才检查阿里云 endpoint 和 OSS。
`doctor` 遇到 fail/warn 会输出 `Repair prompts`，可以直接交给大模型继续修复。

运行其他 CLI 命令时，如果遇到配置、OSS、DashScope、ffmpeg 或声纹依赖类错误，CLI 会在
`Error:` 后给出对应的 `meeting-asr doctor ...` 命令。网络超时、限流、5xx 这类临时错误会先自动
重试；如果重试后仍失败，再按提示重新运行或交给 `doctor` 排查本地配置。

配置上传目录 7 天过期删除：

```bash
meeting-asr oss lifecycle set --prefix meeting-asr/ --days 7
```

这个规则按对象年龄删除，不是按最后访问时间删除。阿里云 OSS 的 last-access lifecycle 不能删除对象。

## 2. Shell Completion

安装补全：

```bash
meeting-asr completion install zsh
```

支持 `bash`、`zsh`、`fish`、`powershell` 和 `pwsh` 安装。只想查看脚本时：

```bash
meeting-asr completion zsh
meeting-asr completion bash
```

补全由 CLI 命令树动态生成，会覆盖子命令、选项，以及配置 key、OSS 上传模式、
音频格式等有限值。

## 2.1 Help 和稳定输出

`meeting-asr` 是 git-like 多级命令。除了 `--help`，也可以用 `help` 子命令查看任意层级：

```bash
meeting-asr
meeting-asr --help
meeting-asr -h
meeting-asr help
meeting-asr help project list
meeting-asr help project transcript show
```

Help 语言支持 `en` 和 `zh`。默认 `auto`：先看 `MEETING_ASR_LANG`，再看
`LC_ALL`、`LC_MESSAGES`、`LANG`，中文 locale 自动显示中文。也可以临时传全局选项：

```bash
LC_ALL=zh_CN.UTF-8 meeting-asr
meeting-asr --lang zh help project list
MEETING_ASR_LANG=zh meeting-asr help project list
```

当前 root 空命令、root `--help`、root `-h`、`meeting-asr help ...`，以及子命令原生
`project list --help` / `project list -h` 都会走 Meeting-ASR 的 i18n renderer。

人类默认看 Rich 表格；脚本优先用 `--json`。如果只需要稳定、可 grep/awk/cut 的行文本，
列表类命令提供 `--plain`：

```bash
meeting-asr project list --plain
meeting-asr project transcript list PROJECT_ID --plain
meeting-asr voiceprint list --plain
meeting-asr lexicon list --plain
meeting-asr project trash list --plain
```

## 3. 创建项目

```bash
meeting-asr project create "/path/to/meeting.mp4" \
  --title "供应商管理AI治理" \
  --meeting-time "2026-04-29T15:07:42+08:00"
```

成功后 CLI 会输出可复制命令：

```bash
meeting-asr project transcribe PROJECT_ID
meeting-asr project status PROJECT_ID
meeting-asr project review PROJECT_ID
```

同一个源视频再次创建时，CLI 会复用已有项目，不会因为日期变化生成新项目。
新项目的 `project_id` 基于源文件内容 hash，形如 `p-...`，不依赖创建时间。

默认项目目录遵循 XDG：`~/.local/share/meeting-asr/projects`。
列出默认项目目录：

```bash
meeting-asr project list
meeting-asr project list --projects-dir "/path/to/projects"
```

`project list` 的 `State` 不是内部 manifest status，而是从实际文件推导出的当前项目阶段。
表格第一列就是稳定 Project ID，后续命令应复制这个 ID。下一步命令、`Artifacts`、
项目目录或原始内部 status 放在 `project status PROJECT_ID` 或 `--json` 里看。

更新 project 元数据：

```bash
meeting-asr project update PROJECT_ID --title "新的会议标题"
meeting-asr project update PROJECT_ID --meeting-time "2026-05-02T10:00:00+08:00"
```

删除 project：

```bash
meeting-asr project delete PROJECT_ID
meeting-asr project trash list
meeting-asr project trash restore TRASH_REF
meeting-asr project trash purge TRASH_REF --yes
meeting-asr project trash cleanup --older-than-days 30 --yes
meeting-asr project delete PROJECT_ID --permanent --yes
```

默认删除是安全删除：项目会移动到
`~/.local/share/meeting-asr/trash/projects/`，不再出现在 `project list`。
误删后用 `project trash list` 查看 trash，再用 `project trash restore TRASH_REF`
恢复到原目录。`TRASH_REF` 可以是 Project ID 或 Trash Dir。确认不需要时，
用 `project trash purge TRASH_REF --yes` 删除单个项目，
或用 `project trash cleanup --older-than-days 30 --yes` 清理超过 30 天的 trash。
Meeting-ASR 不会自动清理 trash；只有 `purge`、`cleanup` 或 `delete --permanent --yes`
会物理删除项目目录。

## 4. 转写

全自动入口优先用：

```bash
meeting-asr project run "/path/to/meeting.mp4"
```

它会创建或复用项目、转写、声纹匹配，并自动应用 accepted 的 speaker 匹配。
转写完成后还会调用 DashScope 文本模型生成会议标题和摘要。默认模型来自
`dashscope.summary_model`，可用 `--summary-model` 临时覆盖；如果不想生成摘要，用
`--no-summarize`。
如果还有未确认 speaker，输出会给出 `meeting-asr project review PROJECT_ID`。

如果只想给已经转写完成的 project 补摘要：

```bash
meeting-asr project summarize PROJECT_ID
```

如果 OSS 已配置，默认使用 private OSS signed URL：

```bash
meeting-asr project transcribe
```

交互式终端默认显示进度 UI；输出重定向或非 TTY 环境不会污染 stdout。所有耗时命令都可用
`--no-progress` 关闭。

如果终端或日志系统不适合彩色输出，根命令加 `--no-color`；设置 `NO_COLOR` 或 `TERM=dumb`
时也会自动禁用 Rich 颜色。
需要看依赖库和内部诊断日志时，根命令加 `--verbose` 或 `-v`；默认只显示 warning 及以上日志。

`project run` 有两类动态 ETA baseline。OSS 上传阶段会按实际上传字节回调刷新进度，并在完成后
记录一条吞吐样本；下一次会按文件大小估算上传 ETA。DashScope 等待阶段会在远程 ASR 等待结束后
记录一条耗时样本；下一次会按音频时长估算等待 ETA。默认数据库：
`~/.local/share/meeting-asr/metrics/runtime.sqlite`。没有历史样本时会显示
`baseline: collecting`。

如果你已经有公网可访问音频 URL：

```bash
meeting-asr project transcribe \
  --file-url "https://example.com/audio.flac" \
  --oss-upload false
```

## 4.1 词汇纠错

专有名词、人名昵称和系统名可以通过编辑器纠错：

```bash
meeting-asr project correct edit PROJECT_ID
meeting-asr project correct edit PROJECT_ID --editor "code --wait"
meeting-asr project correct edit PROJECT_ID --model qwen-plus
meeting-asr project correct accept PROJECT_ID
meeting-asr lexicon list
meeting-asr lexicon show iSee
meeting-asr lexicon add iSee --category system --alias 艾赛
meeting-asr lexicon stats
meeting-asr lexicon export --output lexicon.json
meeting-asr lexicon hotwords list
meeting-asr lexicon hotwords status
meeting-asr lexicon hotwords export
meeting-asr lexicon hotwords sync --target-model fun-asr
meeting-asr lexicon hotwords remote-list
meeting-asr project transcribe PROJECT_ID --asr-hotwords auto
meeting-asr config set ui.editor "code --wait"
meeting-asr config set dashscope.correction_model qwen-plus
meeting-asr project correct edit PROJECT_ID --no-open
meeting-asr project transcript show PROJECT_ID --kind corrected
```

`correct edit` 会生成 `tmp/corrections/review_*.md`，每句前面有
`meeting-asr` HTML 锚点。只修改转写正文，保留锚点；退出编辑器后，CLI 会通过前后
对比识别样例改动，再调用 DashScope 文本模型生成全篇 proposal。你会先看到：

```text
tmp/corrections/proposal_*.md
tmp/corrections/proposal_*.diff
tmp/corrections/proposal_*.json
```

确认 proposal 后，才会输出最终产物：

```text
asr/sentences_corrected.json
exports/transcript_corrected.txt
exports/transcript_named_corrected.txt
exports/subtitle_named_corrected.srt
corrections/asr_hotwords.json
corrections/applied.json
```

原始 `asr/sentences.json`、`exports/transcript.txt` 和
`exports/transcript_named.txt` 不会被覆盖。可学习的替换会写入跨项目词汇库：

```text
~/.local/share/meeting-asr/lexicon/lexicon.sqlite
```

`meeting-asr lexicon list/show/add/delete/stats/import/export` 管理的是本地词库本体：
标准词、别名和纠错上下文。它不直接调用远端服务，也不等同于 ASR 热词表。

如果没有传 `--editor`，编辑器优先级是 `ui.editor`、`VISUAL`、`EDITOR`、`code --wait`、`vim`。
纠错模型优先使用 `--model`，否则使用 `dashscope.correction_model`。如果 DashScope 不可用，
会退回本地替换规则，并在 proposal 里显示 fallback 原因。已经编辑过的 review 文件可以用
`--review-file tmp/corrections/review_*.md` 复用。

`corrections/asr_hotwords.json` 是这次 correction 理解直接产出的热词表。跨项目累计热词可以用
`meeting-asr lexicon hotwords list/status/export` 投影和查看，用
`meeting-asr lexicon hotwords sync --target-model fun-asr` 同步成 DashScope `vocabulary_id`。
远端表可用 `remote-list`、`remote-show`、`remote-delete --yes` 管理；本地缓存错了用
`clear-cache` 清掉。`project transcribe/run` 默认 `--asr-hotwords auto`：如果配置了
`dashscope.asr_vocabulary_id` 就直接使用；否则根据跨项目词库同步热词后再提交 ASR。传
`--asr-hotwords off` 可关闭，传 `--asr-hotwords vocab-...` 可指定已有热词表。

## 5. 自动匹配 + 人工确认 speaker

```bash
meeting-asr project speakers match
meeting-asr project speakers inspect
meeting-asr project review PROJECT
meeting-asr project speakers apply
meeting-asr project speakers preview PROJECT
meeting-asr project speakers preview PROJECT --speaker-id 3
meeting-asr project transcript show PROJECT
```

这一步的核心原则：`match` 只给建议，`apply` 才真正写名字。

`match` 会给当前 project 的 speaker 生成候选：

```bash
meeting-asr project speakers match
```

它只写 `speakers/speaker_matches.json`，不会修改转写文本和字幕。没有声纹库也可以跑；
这时每个 speaker 都会是 `unknown score=0.000 review`，表示当前没有可匹配的人。

`inspect` 用来 review。它会同时显示 speaker 样例和声纹建议：

```text
Speaker E (speaker_id=4)
  Voiceprint match: 敬悦 score=0.775 accepted
```

优先跑 project 层 `review` 进入 TUI 完成确认和人工补足。`PROJECT` 可以是 project
目录、AutoRun 输出的 `Project ID`，也可以是唯一匹配的标题：

```bash
meeting-asr project review PROJECT
```

如果忘了 Project ID，直接运行：

```bash
meeting-asr project review
```

它会先打开 project list TUI，看到历史项目后按 Enter 进入选中项目的 review。

`review` 是新的键盘式入口：

- 顶部 `Project` 是项目事实：标题、项目 id、转写时长、speaker 数量、project 状态。
- 顶部 `Steps` 是流程进度：`1 Match` 自动声纹匹配、`2 Names` 人工姓名确认、
  `3 Capture` 声纹片段采集、`4 Embed` 声纹 embedding。
- 顶部 `Auto` 是自动匹配质量：accepted/review/unknown 数量、平均分、最高分。
- 顶部 `Check` 是需要人工注意的问题：conflict/mismatch，以及当前选中的 speaker 状态。
- 顶部 `Output` 是项目最终产物：人名版文本 `exports/transcript_named.txt` 和人名版字幕 `exports/subtitle_named.srt`。
- 顶部 `Next/Done` 是状态结论：`Next` 说明还缺什么命令；`Done` 说明产物已经就绪，并给出 preview 和查看文本的命令。
- 初始进入是浏览模式，不会要求输入人名；先看 speaker 和样例。
- 默认只有两栏：左边 speaker，右边样例；姓名候选不会常驻占用空间。
- `h/l` 和 `left/right` 切换当前关注列。
- `j/k` 和 `up/down` 在当前关注列内上下移动。
- 关注左列时上下切 speaker；关注右列时上下切样例。
- 当前关注列会用更重的边框和背景高亮；高亮行表示当前选中项。
- `PageUp/PageDown` 翻样例页；`[` 和 `]` 也可以翻页。
- `space` 播放当前选中的单条样例；播放中再次按 `space` 停止。
- `?` 弹出快捷键指引，`Esc`、`q` 或再次按 `?` 关闭。
- `a` 接受当前声纹 match。
- `/` 才打开底部姓名输入/搜索面板；可以输入新人名。
- `Tab` 接受当前搜索结果里的第一个人名建议。
- `i` 明确忽略当前 speaker：保留匿名 label，在顶部显示为 `ignored`，后续 `voiceprint capture` 会跳过它。
- `s` 保存并写出 named transcript/SRT。

如果当前终端不能打开 TUI，可以先看队列：

```bash
meeting-asr project review PROJECT --summary
```

如果只想列出可进入的历史 project：

```bash
meeting-asr project review --summary
```

纯终端 prompt 入口仍然是 `apply`：

```bash
meeting-asr project speakers apply
```

`apply` 默认交互式逐个 speaker 提示输入人名：

- 如果声纹匹配已 accepted，匹配到的人名会作为默认值，直接回车确认。
- 如果没有匹配成功，默认值是 `Speaker A`、`Speaker B` 这类匿名 label，需要手动输入真实姓名。
- 如果样例还不足以确认，在姓名提示处输入 `/more` 会继续输出更多样例。
- 输入 `/audio` 会把当前终端上显示的样例批次合成一段音频播放，并自动停止；也可以用 Space/P 暂停、Q/Esc 或 Ctrl-C 提前结束。
- 输入过 `/more` 或 `/audio` 后，可以在下一次提示里按上方向键召回命令。
- 如果最终仍不知道是谁，直接回车保留匿名 label。

`apply` 成功后会写入：

```text
speakers/speaker_map.json
exports/transcript_named.txt
exports/subtitle_named.srt
```

写完后用 preview 和 transcript 复核：

```bash
meeting-asr project speakers preview
meeting-asr project transcript show
```

`preview` 只是检查字幕和视频是否对齐，不是新的产物。项目完成后的核心产物是：

```text
exports/transcript_named.txt
exports/subtitle_named.srt
```

如果要脚本化执行，仍可使用：

```bash
meeting-asr project speakers apply --map 0=欧丁 --map 1=敬悦
```

`match --apply` 只适合完全信任自动匹配时使用：

```bash
meeting-asr project speakers match --apply
```

它只会应用 accepted match；如果还有没匹配成功的人，不要用它替代交互式 `apply`。

## 6. 记录跨项目声纹

人工确认完成后，再把这个 project 里已确认的人写进跨项目声纹库。声纹库不放在当前
project 目录。默认存放位置遵循 XDG：
`~/.local/share/meeting-asr/voiceprints/`。

```bash
meeting-asr voiceprint capture
meeting-asr voiceprint embed
meeting-asr voiceprint browse
meeting-asr voiceprint list
meeting-asr voiceprint show 1
meeting-asr voiceprint play 1 --sample 1
meeting-asr voiceprint path
```

`capture` 会从当前 project 的 `asr/sentences.json` 和
`speakers/speaker_map.json` 选择已确认姓名的 speaker 参考片段，WAV 写入
`voiceprints/clips/`，索引写入 `voiceprints/voiceprints.sqlite`。
仍然是 `Speaker A`、`Speaker C` 这种匿名 label 的人会跳过，不进入声纹库。
`list` 会显示 speaker ID，并按 speaker 汇总样本数、项目数和 embedding 覆盖率。

`browse` 是声纹库 TUI：

- 左边是跨项目声纹库里已有的人。
- 右边是当前人的 WAV 样本，包含来源 project、project 内 speaker id、时间戳和转写文本。
- `h/l` 或左右方向键切换关注列。
- `j/k` 或上下方向键在当前列移动。
- `PageUp/PageDown` 或 `[`、`]` 翻样本页。
- `space` 播放当前样本；播放中再次按 `space` 停止。
- `?` 查看快捷键。
- 删除样本或整个人仍用显式 CLI，不放在浏览 TUI 里，避免误删。

`embed` 默认使用本地 `local-speechbrain` provider。先安装本地声纹依赖：

```bash
uv sync --extra local-voiceprint
```

如果 CLI 是 `uv tool install` 安装的，改用：

```bash
scripts/install-tool.sh
```

这个脚本是独立安装入口，不是 `meeting-asr` 子命令。它显式传
`uv tool install --python 3.14 --editable`，避免 uv tool 默认解释器落到不满足
`Python>=3.14` 的版本。本地开发默认 editable，源码修改会直接生效。
如果要模拟正式用户安装或发布验证，使用 `scripts/install-tool.sh --wheel`。
项目已配置 `tool.uv.cache-keys` 跟踪 `src/**/*.py`，wheel 模式下源码变化会触发本地
wheel 重建。
脚本安装后会比对当前 checkout 和实际安装包的源码指纹；如果不一致会直接失败。

默认配置：

```bash
meeting-asr config set voiceprint.embedding_provider "local-speechbrain"
meeting-asr doctor --full
```

如果要使用百炼/AnalyticDB 声纹检索 provider，切换 provider 并配置 endpoint：

```bash
meeting-asr config set voiceprint.embedding_provider "bailian"
meeting-asr config set voiceprint.embedding_endpoint "http://<adb-ai-app-host>:8100/audio/embedding"
meeting-asr doctor --full
```

这里的 endpoint 不是本机要安装的东西，也不是 `tongyi-embedding-vision-*`
视觉多模态 embedding 模型名。它是 AnalyticDB MySQL 声纹检索服务暴露的音频
embedding API 地址，官方 API 形状是 `http://addr:8100/audio/embedding`。

获取方式：

1. 声纹检索当前是邀测能力；如果你的 AnalyticDB 集群没有开通，先提交阿里云工单联系技术支持。
2. 开通或部署完成后，进入 AnalyticDB MySQL 控制台，选择目标地域和集群。
3. 在左侧进入 `AI 应用`，打开 `应用管理`，查看目标应用服务的 `调用信息`。
4. 从调用信息里拿到调用地址或 host，配置成 `http://<addr>:8100/audio/embedding`。

然后生成 embedding 并匹配新项目：

```bash
meeting-asr voiceprint embed
meeting-asr project speakers match
meeting-asr project speakers inspect
meeting-asr project speakers apply
```

也可以用 `--provider` 临时覆盖全局配置，方便评测不同后端：

```bash
meeting-asr voiceprint embed --provider bailian --rebuild
meeting-asr project speakers match --provider bailian
```

如果只想看会切哪些片段，不写文件和数据库：

```bash
meeting-asr voiceprint capture --dry-run
```

删除样本或整个人：

```bash
meeting-asr voiceprint delete-sample 1 --sample 1
meeting-asr voiceprint delete-speaker 1 --yes
```

先用 `voiceprint list` 看 speaker ID。`show`、`play`、`delete-sample`
和 `delete-speaker` 都接受姓名或 ID；先用 `voiceprint show 1` 看样本编号，
再用同一个编号播放或删除。

## 7. 最终文件

直接查看结果：

```bash
meeting-asr project transcript list
meeting-asr project transcript show
meeting-asr project transcript show --kind plain
meeting-asr project transcript path --kind srt
meeting-asr project transcript open --kind named
```

- `exports/transcript.txt`：纯文本
- `exports/transcript_speakers.txt`：匿名 speaker 文本
- `exports/transcript_named.txt`：人名版文本
- `exports/subtitle.srt`：匿名字幕
- `exports/subtitle_named.srt`：人名版字幕
- `exports/meeting_summary.md`：会议标题和摘要
- `exports/meeting_summary.json`：结构化会议摘要
- `asr/raw_result.json`：DashScope 原始结果
- `asr/sentences.json`：标准化逐句结果
