# Meeting-ASR CLI 用户手册

先读 [快速开始](quick-start.md)。这里记录命令细节和排障入口。

## 1. 健康检查和配置

```bash
meeting-asr doctor
meeting-asr doctor --full
meeting-asr doctor --full --json
```

- `doctor`：本地环境、基础配置、编辑器、播放器、声纹标准依赖提示；不做网络写入。
- `doctor --full`：完整集成检查，会验证 OSS 上传/签名 URL/读回/删除，并严格检查声纹 embedding。
- `doctor --full --json`：给 agent 或脚本用，字段和值保持英文稳定。

配置入口：

```bash
meeting-asr config set dashscope.api_key "<dashscope-api-key>"
meeting-asr config set dashscope.summary_model qwen-plus
meeting-asr config set dashscope.correction_model qwen-plus
meeting-asr config set dashscope.model_endpoints '{"qwen3.6-*":"multimodal"}'
meeting-asr config set oss.access_key_id "<oss-access-key-id>"
meeting-asr config set oss.access_key_secret "<oss-access-key-secret>"
meeting-asr config set oss.bucket_name "<bucket>"
meeting-asr config set oss.region "<region>"
meeting-asr config set oss.endpoint "<oss-endpoint>"
meeting-asr config set ui.editor "code --wait"
```

XDG 默认路径：

```text
~/.config/meeting-asr/config.json
~/.local/share/meeting-asr/projects
~/.local/share/meeting-asr/voiceprints
~/.local/share/meeting-asr/metrics/runtime.sqlite
```

`dashscope.model_endpoints` 用于覆盖模型到调用端点的路由，值是 JSON 对象。支持的端点是
`generation`、`multimodal`、`compatible`；key 可以是精确模型名或通配符。内置路由已覆盖
`qwen3.6-plus`、`qwen3.6-flash`、`qwen3.5-plus`、`qwen-vl-*` 等多模态模型，一般不需要手工配置。
如果希望所有 Qwen3.6 调用都走 OpenAI-compatible，可以设置：

```bash
meeting-asr config set dashscope.model_endpoints '{"qwen3.6-*":"compatible"}'
```

要临时使用另一套项目库，设置 XDG 数据目录：

```bash
XDG_DATA_HOME="/path/to/data-home" meeting-asr project list
```

## 2. Help、completion 和稳定输出

```bash
meeting-asr
meeting-asr help project run
meeting-asr --lang zh help project list
MEETING_ASR_LANG=zh meeting-asr help project list
```

Help 语言默认 `auto`：先看 `MEETING_ASR_LANG`，再看 `LC_ALL`、`LC_MESSAGES`、`LANG`。

安装 shell completion：

```bash
meeting-asr completion install zsh
meeting-asr completion zsh
meeting-asr completion bash
```

人类默认看 Rich 表格；脚本优先用 `--json`。需要稳定行文本时用 `--plain`：

```bash
meeting-asr project list --plain
meeting-asr project transcript list PROJECT_ID --plain
meeting-asr voiceprint list --plain
meeting-asr lexicon list --plain
meeting-asr project trash list --plain
```

## 3. Project 生命周期

创建项目：

```bash
meeting-asr project create "/path/to/meeting.mp4" \
  --title "供应商管理AI治理" \
  --meeting-time "2026-04-29T15:07:42+08:00"
```

要点：

- Project ID 基于源文件内容 hash，形如 `p-...`。
- 同一个源视频再次 create/run 会复用已有项目。
- 后续命令使用 Project ID、项目目录或唯一标题；不需要 `cd` 到项目目录。

查看项目：

```bash
meeting-asr project list
meeting-asr project show PROJECT_ID
meeting-asr project status PROJECT_ID
```

`project list` 只做列表，默认只显示 Project ID、状态、标题和关键词；会议时间会规范化进标题，不再单独显示 meeting/update 时间列。项目阶段、下一步、产物、runtime 状态、错误恢复命令看 `project show`。

更新元数据：

```bash
meeting-asr project update PROJECT_ID --title "新的会议标题"
meeting-asr project update PROJECT_ID --meeting-time "2026-05-02T10:00:00+08:00"
```

带 meeting time 的项目标题会统一成 `YYYY-MM-DD HH:MM 标题`。

删除和恢复：

```bash
meeting-asr project delete PROJECT_ID
meeting-asr project trash list
meeting-asr project trash restore TRASH_REF
meeting-asr project trash purge TRASH_REF --yes
meeting-asr project trash cleanup --older-than-days 30 --yes
meeting-asr project delete PROJECT_ID --permanent --yes
```

默认删除会移动到 `~/.local/share/meeting-asr/trash/projects/`。Meeting-ASR 不自动清理 trash；只有 `purge`、`cleanup` 或 `delete --permanent --yes` 会物理删除。

## 4. 转写和长任务进度

推荐入口：

```bash
meeting-asr project run "/path/to/meeting.mp4"
```

`project run` 会创建或复用项目、抽取音频、上传 private OSS、提交 ASR、下载转写、应用已入库的本地词汇订正、生成回忆索引、生成 transcript polish proposal、声纹匹配，并自动应用 accepted speaker。

如果只想对已有项目执行转写：

```bash
meeting-asr project transcribe PROJECT_ID
```

如果已有公网音频 URL：

```bash
meeting-asr project transcribe PROJECT_ID \
  --file-url "https://example.com/audio.flac" \
  --oss-upload false
```

长任务可观测性：

- 交互式终端显示多步骤进度。
- 默认人类输出只显示进度和最终摘要，不打印 `stage=` / `heartbeat=` 结构化日志。
- 长轮询会更新当前步骤的 elapsed、ETA、poll 状态或 batch index。
- 结构化日志只在显式 `--agent-log` 开启时输出，供 Agent、CI 或日志系统诊断。
- Agent 推荐使用 `--agent-log --no-progress`，这样 stdout/stderr 只有稳定的 stage/heartbeat 文本和最终摘要。
- 日志只包含非敏感标识，例如 `dashscope_task_id`、`oss_object_key`、`signed_url_ready`；signed URL query、token、secret、access key 不输出。

Agent 诊断入口：

```bash
meeting-asr project run "/path/to/meeting.mp4" --agent-log --no-progress
```

中断或怀疑卡住：

```bash
meeting-asr project show PROJECT_ID
```

它会显示 `Current stage`、`Stage updated`、`External IDs`、`Last error`、缺失产物和恢复命令。

关闭进度或颜色：

```bash
meeting-asr project run "/path/to/meeting.mp4" --no-progress
meeting-asr --no-color project list
```

`project run` 会记录 OSS 上传和 DashScope ASR 等待的动态 ETA baseline。没有历史样本时显示 `baseline: collecting`。

## 5. 本地词汇订正、Transcript polish 和人工纠错

`project run` 默认会先应用已经入库的本地词汇订正规则。这一步不调用模型，不猜新规则，只把已经确认过的错词规则应用到全文，例如 `IC -> iSee`。

如果只想保留远端 ASR 原文，显式关闭本地订正：

```bash
meeting-asr project run "/path/to/meeting.mp4" --no-local-correction
```

`project run` 还会运行 transcript polish。默认保持兼容模式：只生成 proposal，不自动接受。Agent 自动化场景建议开启配置，让 strict polish 通过内置 guard 后直接写回 corrected 产物：

```bash
meeting-asr config set correction.polish_auto_accept true
```

看状态：

```bash
meeting-asr project show PROJECT_ID
```

如果显示 `Transcript polish: accepted`，`exports/transcript_named_corrected.txt` 已经可用。如果显示 `Transcript polish: proposal ready`，可以直接接受；需要抽查质量时再看 diff：

```bash
meeting-asr project correct accept PROJECT_ID --proposal /path/to/proposal.json
meeting-asr project correct diff PROJECT_ID --proposal /path/to/proposal.json
```

手工教系统纠错：

```bash
meeting-asr project correct edit PROJECT_ID
meeting-asr project correct edit PROJECT_ID --editor "code --wait"
meeting-asr project correct edit PROJECT_ID --model qwen-plus
```

流程：

1. 生成 `tmp/corrections/review_*.md` 并打开编辑器。
2. 你只改正文，保留锚点。
3. CLI 用前后 diff 学习样例改动。
4. DashScope 文本模型生成全篇 proposal。
5. 接受后写出 corrected transcript/subtitle；原始文件不覆盖。

纠错产物：

```text
asr/sentences_corrected.json
exports/transcript_corrected.txt
exports/transcript_named_corrected.txt
exports/subtitle_named_corrected.srt
corrections/asr_hotwords.json
corrections/applied.json
```

词库和热词：

```bash
meeting-asr lexicon list
meeting-asr lexicon show TERM_OR_ID
meeting-asr lexicon add iSee --category system --alias IC
meeting-asr lexicon add iSee --category system --alias 艾赛
meeting-asr lexicon stats
meeting-asr lexicon export --output lexicon.json
meeting-asr lexicon hotwords list
meeting-asr lexicon hotwords status
meeting-asr lexicon hotwords sync --target-model fun-asr
```

`lexicon` 管本地词库本体；`corrections/asr_hotwords.json` 是某个项目 correction 理解产出的热词；`lexicon hotwords` 是跨项目词库投影。

## 6. Speaker review

首选入口：

```bash
meeting-asr project review PROJECT_ID
```

不传 Project ID 会先打开 project list TUI：

```bash
meeting-asr project review
```

TUI 常用键：

```text
h/l 或 ←/→     切换左右列
j/k 或 ↑/↓     当前列上下移动
PageUp/PageDown 翻样例页
space          播放/停止当前 sample
a              接受当前声纹 match
i              忽略当前 speaker
/              搜索或输入人名
e              修改当前 sample 文本并进入转写纠错
c              同 e，保留为隐藏别名
t              切换「分组视图 / 时间轴视图」
r              在时间轴视图下，把当前句子改给另一个 speaker
p              切换 Project
v              进入 Voiceprint Review
m              用最新声纹库重新匹配当前项目
b              对已采集声纹补做 embedding
s              保存
?              快捷键
```

时间轴视图（`t`）按 ASR 切分的真实时间顺序展示所有句子，便于边听边核对。如果听出某句话其实是另一个人讲的，光标停在该句上按 `r` 选目标 speaker。

按 `s` 保存若存在归属变更，会自动跑后链路：

- 写回 `asr/sentences.json`（以及 `asr/sentences_corrected.json`，如果存在）
- 重新生成命名 transcript / 字幕（`transcript_named.txt` / `subtitle_named.srt` 及对应 corrected 版本）
- 重新生成匿名分组 transcript（`transcript_speakers.txt`）
- 删除已采集的声纹样本中、与归属变更句子区间重叠且属于原 speaker 的那些（其它项目和其它 speaker 的样本不受影响），用户需要后续重新采集
- 重新跑 `project speakers match`，刷新 `speakers/speaker_matches.json`

只改 speaker 姓名（没有 reassign）不会触发声纹失效或 rematch。

如果发现 ASR 把多个人合并成同一个 speaker（典型表现：某个 speaker 声纹评分异常低，且每个 speaker 的样本里听到不同的人），先按估算的真实人数重跑 ASR：

```bash
meeting-asr project transcribe PROJECT_ID --speaker-count 6
```

`--speaker-count` 也可以加在 `project run` 上。重跑会作废下游产物（corrections / named transcript / SRT），声纹样本不变；之后可重新做 voiceprint match。

只读诊断：

```bash
meeting-asr project review PROJECT_ID --summary
meeting-asr project speakers inspect PROJECT_ID
```

脚本化映射，不是人类首选路径：

```bash
meeting-asr project speakers apply PROJECT_ID --map 0=欧丁 --map 1=敬悦
meeting-asr project speakers match PROJECT_ID --apply
```

`match --apply` 只应用 accepted match；低分、无候选、冲突仍应进 `project review`。

保存 speaker 后会写出：

```text
speakers/speaker_map.json
speakers/speaker_ignore.json
speakers/speaker_person_map.json
exports/transcript_named.txt
exports/subtitle_named.srt
```

## 7. 声纹库

声纹是跨项目数据，默认位置：

```text
~/.local/share/meeting-asr/voiceprints/
  voiceprints.sqlite
  clips/<project-id>/speaker_<id>/clip_001.wav
```

项目内采样：

```bash
meeting-asr voiceprint review PROJECT_ID
meeting-asr project speakers match PROJECT_ID
```

首选流程是在 `project review` TUI 中按 `v` 进入 Voiceprint Review。确认样本后按 `s`，它会采样、生成 embedding，并做当前项目和历史项目反向评测。评测发现风险时，先看提示的 `meeting-asr project review PROJECT_ID`，确认后再接受 embedding。

`meeting-asr voiceprint embed` 仍保留给脚本和补救场景，例如已有样本但 embedding 缺失时使用。
它会先使用归一化后的派生音频提取 embedding，不覆盖原始 clip。只想重建归一化音频时：

```bash
meeting-asr voiceprint normalize --rebuild
meeting-asr voiceprint embed --rebuild
```

全局声纹库浏览：

```bash
meeting-asr voiceprint review
meeting-asr voiceprint quality --review
meeting-asr voiceprint list
meeting-asr voiceprint show PERSON_ID_OR_NAME
meeting-asr voiceprint play PERSON_ID_OR_NAME --sample 1
meeting-asr voiceprint delete-sample PERSON_ID_OR_NAME --sample 1
meeting-asr voiceprint delete-speaker PERSON_ID_OR_NAME --yes
meeting-asr voiceprint people list
```

`voiceprint review PROJECT_ID` 左侧是项目待采样候选，右侧是样本；`Tab` 切换视图，`p` 看项目候选样本，`g` 看全局库，`y` 看质量检查。没有 Project ID 时直接进入全局库视图。质量页可播放样本、隔离离群样本、标记人工确认，并按 `u` 重新计算质量。删除操作保留在显式 CLI，避免 TUI 里误删。

声纹 embedding 只保留本地 `local-speechbrain`，不再提供远端 provider 配置。

## 8. 最终文件

查看：

```bash
meeting-asr project transcript list PROJECT_ID
meeting-asr project transcript show PROJECT_ID --kind named
meeting-asr project transcript show PROJECT_ID --kind corrected
meeting-asr project transcript path PROJECT_ID --kind srt
meeting-asr project transcript open PROJECT_ID --kind named
```

主要产物：

```text
exports/transcript.txt                  # 纯文本
exports/transcript_speakers.txt         # 匿名 speaker 文本
exports/transcript_named.txt            # 人名版文本
exports/transcript_named_corrected.txt  # 人名 + 纠错文本
exports/subtitle.srt                    # 匿名字幕
exports/subtitle_named.srt              # 人名版字幕
exports/subtitle_named_corrected.srt    # 人名 + 纠错字幕
exports/meeting_summary.md              # 会议标题和回忆索引
exports/meeting_summary.json            # 结构化回忆索引
asr/raw_result.json                      # DashScope 原始结果
asr/sentences.json                       # 标准化逐句结果
```

## 9. OSS lifecycle

每次上传都会自动兜底：若 bucket 上还没有 `meeting-asr-auto-delete` 规则，就补一条默认规则（`meeting-asr/` 前缀，1 天后删除）。已存在同 id 规则时不覆盖，凭证缺少 lifecycle 权限时降级为 warning、不阻塞上传。

手动调整过期时长（覆盖同 id 规则）：

```bash
meeting-asr oss lifecycle set --prefix meeting-asr/ --days 7
```

这个规则按对象年龄删除，不是按最后访问时间删除；OSS 按天粒度后台批量执行，实际删除可能滞后到期后最多 24 小时。配置时只 upsert 指定 rule，不覆盖 bucket 里其他 lifecycle rule。
