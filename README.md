# Meeting-ASR

`meeting-asr` 是一个项目化 CLI：从本地 MP4/MOV/MKV 创建项目，抽取 mono 16kHz 音频，上传 private OSS 并签出 URL，调用阿里云 DashScope / 百炼 Fun-ASR 异步转写，最后生成文本、字幕和 speaker 人工标注结果。

## 架构说明

代码分层见 `docs/architecture.md`。新代码优先放入 `core/`、`infra/`、`presentation/cli` 或 `presentation/tui`，旧的顶层 import wrapper 只用于兼容。

## 快速开始

如果只想知道怎么跑会议转写，先看 [快速开始：两条路径](docs/quick-start.md)。

下面是开发和安装入口。

```bash
uv venv
uv sync --all-groups
uv run meeting-asr --help
```

安装成可直接运行的命令：

```bash
scripts/install-tool.sh
meeting-asr completion install zsh
exec zsh
```

`scripts/install-tool.sh` 是独立安装入口，不属于业务 CLI。它固定使用：

- `uv tool install --python 3.14 --force --reinstall --refresh`
- 默认安装 `local-voiceprint` extra
- 安装后验证 `meeting-asr` wrapper 实际使用的 Python 和包来源

`uv` 可以使用 pyenv 提供的 Python；这里显式传 `--python 3.14` 是为了避免
`uv tool install` 默认解释器落到不满足本项目 `Python>=3.14` 的版本。

如果只想看当前安装状态：

```bash
scripts/install-tool.sh --check
```

如果要使用默认的本地声纹 embedding provider，安装本地声纹依赖：

```bash
uv sync --extra local-voiceprint
```

如果是 `uv tool install` 安装方式：

```bash
scripts/install-tool.sh
```

`completion install` 支持 `bash`、`zsh`、`fish`、`powershell` 和 `pwsh`；也可以用
`meeting-asr completion zsh` 这类命令直接输出补全脚本。

配置遵循 XDG Base Directory：

- 配置文件：`$XDG_CONFIG_HOME/meeting-asr/config.json`，默认 `~/.config/meeting-asr/config.json`
- 默认项目：`$XDG_DATA_HOME/meeting-asr/projects`，默认 `~/.local/share/meeting-asr/projects`

```bash
meeting-asr config set dashscope.api_key "<your-dashscope-api-key>"
meeting-asr config set dashscope.base_url "https://dashscope.aliyuncs.com/api/v1"
meeting-asr config set dashscope.summary_model "qwen-plus"
meeting-asr config set oss.access_key_id "<your-oss-access-key-id>"
meeting-asr config set oss.access_key_secret "<your-oss-access-key-secret>"
meeting-asr config set oss.bucket_name "<your-bucket>"
meeting-asr config set oss.region "<your-region>"
meeting-asr config set oss.endpoint "<your-oss-endpoint>"
meeting-asr config set voiceprint.embedding_provider "local-speechbrain"
meeting-asr doctor --require-voiceprint-embedding
```

声纹 embedding 支持多个 provider：

```bash
meeting-asr config set voiceprint.embedding_provider "local-speechbrain"
meeting-asr config set voiceprint.embedding_provider "bailian"
```

`local-speechbrain` 是默认值，使用本地 SpeechBrain ECAPA speaker embedding 模型，不依赖阿里云声纹服务。
`bailian` 保留为阿里云 AnalyticDB 声纹检索 provider，申请开通后再配置 endpoint：

```bash
meeting-asr config set voiceprint.embedding_provider "bailian"
meeting-asr config set voiceprint.embedding_endpoint "http://<adb-ai-app-host>:8100/audio/embedding"
meeting-asr doctor --require-oss --require-voiceprint-embedding
```

`voiceprint.embedding_endpoint` 不是本机要安装的东西，也不是
`tongyi-embedding-vision-*` 这类视觉多模态模型名。它是 AnalyticDB MySQL
声纹检索服务暴露的音频 embedding API 地址，官方 API 形状是
`http://addr:8100/audio/embedding`。

这个地址从 AnalyticDB 来：

1. 声纹检索当前是邀测能力；如果你的 AnalyticDB 集群没有开通，先提交阿里云工单联系技术支持。
2. 开通或部署完成后，进入 AnalyticDB MySQL 控制台，选择目标地域和集群。
3. 在左侧进入 `AI 应用`，打开 `应用管理`，查看目标应用服务的 `调用信息`。
4. 从调用信息里拿到调用地址或 host，配置成 `http://<addr>:8100/audio/embedding`。

`doctor` 遇到 fail/warn 会输出 `Repair prompts`，这段可以直接交给大模型继续修复。

## 主流程

一条命令创建或复用项目、转写、生成会议标题/摘要、声纹匹配，并自动应用 accepted 的 speaker 匹配：

```bash
meeting-asr project run "/path/to/meeting.mp4" \
  --meeting-time "2026-04-29T15:07:42+08:00"
```

分步执行：

```bash
meeting-asr project create "/path/to/meeting.mp4" --title "供应商管理AI治理"
meeting-asr project list
meeting-asr project transcribe PROJECT_NO
meeting-asr project speakers match PROJECT_NO --apply
meeting-asr project review PROJECT_NO
meeting-asr project speakers preview PROJECT_NO
meeting-asr project transcript show PROJECT_NO
meeting-asr voiceprint capture
meeting-asr voiceprint embed
meeting-asr voiceprint browse
```

`project create` 会复制源视频到 `source/`，后续命令只需要项目目录，不需要再次传视频路径。
`project run` 不需要手工输入会议标题；转写完成后会调用 DashScope 文本模型生成标题和摘要，
写入 `exports/meeting_summary.md` 和 `exports/meeting_summary.json`。如果显式传了
`--title`，模型仍会生成摘要，但不会覆盖手工标题。
同一个源视频再次创建或 run 会复用已有项目；新项目 ID 基于源文件内容 hash，形如 `p-...`，
不依赖创建日期。
AutoRun、create 和 `project list` 会打印短数字 `Project No.`，后续命令优先传这个数字；
也仍然可以传 project path、project id 或 project title，不需要先 `cd`。在项目目录内执行时，
项目路径参数仍默认是当前目录。不记得 Project No. 时，跑 `meeting-asr project list` 看表格，
或直接跑 `meeting-asr project review` 打开 project list TUI，选中历史 project 后进入 review。
`project list` 默认列出 XDG 项目目录，也可以用 `--projects-dir` 指定项目父目录。
交互式终端会在 stderr 显示 Rich 进度；脚本、管道和测试输出保持纯文本。需要关闭时加
`--no-progress`。

`project run` 会在远程 ASR 等待完成后记录一条耗时样本，并刷新动态 ETA baseline。
记录按 provider、service、model、endpoint 分组，默认写入
`~/.local/share/meeting-asr/metrics/runtime.sqlite`。后续同一后端会根据音频时长显示
DashScope 等待阶段的预计耗时；没有历史样本时显示 `baseline: collecting`。

Project 元数据和删除：

```bash
meeting-asr project update PROJECT_NO --title "新的会议标题"
meeting-asr project update PROJECT_NO --meeting-time "2026-05-02T10:00:00+08:00"
meeting-asr project delete PROJECT_NO
meeting-asr project delete PROJECT_NO --permanent --yes
```

`project delete` 默认不会物理删除，会移动到
`~/.local/share/meeting-asr/trash/projects/`；只有显式传 `--permanent --yes`
才会直接删除项目目录。

Speaker 命名分两步：`speakers match` 只写声纹候选到 `speakers/speaker_matches.json`，
不改转写结果；`speakers apply` 才会把自动候选和人工输入合并，写入
`speakers/speaker_map.json`、`exports/transcript_named.txt` 和
`exports/subtitle_named.srt`。
这个项目的终点不是 preview，而是人名版文本和字幕已经写出。`preview` 只是用播放器检查
`exports/subtitle_named.srt` 是否和视频对得上；看完没问题就可以直接用
`meeting-asr project transcript show` 或 `meeting-asr project transcript open --kind named`
读取最终结果。

没有声纹库也可以跑 `speakers match`；这时所有 speaker 都会是
`unknown score=0.000 review`。这不是错误，只表示当前声纹库没有可匹配的人。

推荐流程：

1. 先跑 `meeting-asr project speakers match`。有声纹库时会生成候选；没有声纹库时会生成全 unknown 的 review 结果。
2. 用 `meeting-asr project speakers inspect` 查看每个 speaker 的样例和声纹建议。
3. 优先跑 `meeting-asr project review PROJECT_NO` 进入 project 层 TUI；如果不传
   `PROJECT_NO`，会先打开 project list。进入 review 后，它顶部会显示 project 概况、
   match/manual/capture/embed 进度、match 分数和 conflict/mismatch；下方两栏用于切
   speaker/sample、样例翻页、播放/停止当前样例、接受 match、输入新名字并保存；按 `?`
   可查看快捷键。
   顶部 `Output` 会直接列出最终项目产物；如果顶部信息太多，先看 `Next/Done` 行。
   `Next` 表示还没完成，按它给的命令继续；`Done` 表示产物已就绪，并给出 preview 和查看文本的命令。
4. 如果只想用纯终端 prompt，仍可跑 `meeting-asr project speakers apply`。已 accepted 的 match 会作为默认值，直接回车确认；未匹配的人手动输入姓名；不确定就输入 `/more` 继续看样例，或用 `/audio` 播放当前可见样例的短预览。
5. 用 `meeting-asr project speakers preview` 复核带名字的字幕。
6. 确认无误后，最终项目产物就是 `exports/transcript_named.txt` 和 `exports/subtitle_named.srt`。
7. 如果有新确认的人，再跑 `meeting-asr voiceprint capture && meeting-asr voiceprint embed`，把他们补进跨项目声纹库。

`meeting-asr project speakers match --apply` 只是快捷路径：它只会应用已 accepted
的匹配，适合你确定自动结果已经足够时使用；如果还有未匹配的人，应使用交互式
`speakers apply` 补足。

转写结果属于 project，用 project 子命令查看：

```bash
meeting-asr project transcript list
meeting-asr project transcript show
meeting-asr project transcript path --kind srt
meeting-asr project transcript open --kind named
```

声纹是跨项目数据，不写在单个 project 里。默认存放位置遵循 XDG：

```text
~/.local/share/meeting-asr/voiceprints/
  voiceprints.sqlite
  clips/<project-id>/speaker_<id>/clip_001.wav
```

常用命令：

```bash
meeting-asr voiceprint capture
meeting-asr voiceprint embed
meeting-asr voiceprint browse
meeting-asr voiceprint list
meeting-asr voiceprint show 1
meeting-asr voiceprint play 1 --sample 1
meeting-asr voiceprint delete-sample 1 --sample 1
meeting-asr voiceprint delete-speaker 1 --yes
meeting-asr voiceprint path
```

`voiceprint capture` 只记录已确认姓名的 speaker；仍是 `Speaker A`、`Speaker C`
这种匿名 label 的人会跳过。`voiceprint list` 会显示 speaker ID，并按 speaker
汇总样本数、项目数和 embedding 覆盖率；`show`、`play`、`delete-sample` 和
`delete-speaker` 可用姓名或 ID 引用同一个人。`show` 会显示样本编号，`play` 和
`delete-sample` 都按这个编号精确操作。
`voiceprint browse` 是声纹库 TUI：左边选人，右边看这个人的 WAV 样本、来源项目、
时间戳和转写文本；`space` 播放/停止当前样本，`?` 看快捷键。删除仍用显式 CLI，
避免在浏览界面里误删。

声纹 embedding 默认走 `local-speechbrain`。生成 embedding 后，可以匹配新项目：

```bash
meeting-asr doctor --require-voiceprint-embedding
meeting-asr voiceprint embed
meeting-asr project speakers match
meeting-asr project speakers inspect
meeting-asr project speakers apply
```

如果要临时对比阿里云 provider，不改全局配置也可以传参数：

```bash
meeting-asr voiceprint embed --provider bailian --rebuild
meeting-asr project speakers match --provider bailian
```

## 输出结构

```text
project/
  project.json
  source/<video>
  source/original.path
  audio/audio.flac
  asr/raw_result.json
  asr/sentences.json
  speakers/speaker_map.json
  exports/transcript.txt
  exports/transcript_speakers.txt
  exports/transcript_named.txt
  exports/subtitle.srt
  exports/subtitle_named.srt
  notes.md
```

## 关键约束

- DashScope 录音文件识别只能接收公网 HTTP/HTTPS URL，不能直接传本地文件。
- 本工具默认用 private OSS 上传后签出临时 GET URL，不要求 bucket public read。
- `meeting-asr oss lifecycle set` 配置的是 OSS 前缀对象按对象年龄过期删除；阿里云 OSS 基于最后访问时间的生命周期规则不能用于删除对象。
- speaker diarization 只适用于单声道音频，因此本地预处理固定为 mono 16kHz s16。
- `speaker_count` 只是参考值，不能假设平台严格返回这个人数。
- 工具只做匿名 speaker 聚合和人工映射，不自动识别人名。
- `transcription_url` 会过期，任务完成后必须立刻下载保存。

## 文档

- [快速开始：两条路径](docs/quick-start.md)
- [CLI 用户手册](docs/cli-user-guide.md)
- [开发者指南](docs/developer-guide.md)
