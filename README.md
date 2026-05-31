# Meeting-ASR

`meeting-asr` 是一个项目化会议转写 CLI：输入本地视频，创建稳定 Project ID，抽取音频，上传 private OSS 签名 URL，调用 DashScope/Fun-ASR 转写，生成会议回忆索引、转写文本、字幕，并通过 TUI 完成 speaker、词汇纠错和声纹库维护。

## 先看这里

只想跑一次会议转写，读 [快速开始](docs/quick-start.md)。

常用路径只有两条：

```bash
# 1. 全自动：创建/复用项目、转写、回忆索引、声纹匹配、输出产物
meeting-asr project run "/path/to/meeting.mp4"

# 2. 人工兜底：处理未匹配 speaker、词汇纠错、声纹采样
meeting-asr project review PROJECT_ID
```

如果忘了 `PROJECT_ID`：

```bash
meeting-asr project list
meeting-asr project show PROJECT_ID
```

最终最常用的产物：

```text
exports/transcript_named_corrected.txt  # 如果本地词汇订正或人工纠错已生效
exports/transcript_named.txt
exports/subtitle_named.srt
exports/meeting_summary.md
```

查看结果：

```bash
meeting-asr project transcript show PROJECT_ID --kind auto
meeting-asr project speakers preview PROJECT_ID
```

## 安装和配置

普通用户直接从 PyPI 安装全局命令：

```bash
uv tool install meeting-asr --python 3.14
meeting-asr --version
meeting-asr completion install zsh
```

升级到 PyPI 最新版本：

```bash
uv tool install meeting-asr --python 3.14 --reinstall --refresh
meeting-asr --version
```

开发环境：

```bash
uv venv
uv sync --all-groups
uv run meeting-asr --help
uv run pytest -q
```

本地开发需要全局 editable 命令时再使用：

```bash
scripts/install-tool.sh
scripts/install-tool.sh --check
```

配置遵循 XDG：

```text
~/.config/meeting-asr/config.json
~/.local/share/meeting-asr/projects
~/.local/share/meeting-asr/voiceprints
```

最小配置入口：

```bash
meeting-asr config set dashscope.api_key "<dashscope-api-key>"
meeting-asr config set oss.access_key_id "<oss-access-key-id>"
meeting-asr config set oss.access_key_secret "<oss-access-key-secret>"
meeting-asr config set oss.bucket_name "<bucket>"
meeting-asr config set oss.region "<region>"
meeting-asr config set oss.endpoint "<oss-endpoint>"
meeting-asr doctor --full
```

本地声纹 embedding 默认使用 `local-speechbrain`，SpeechBrain/Torch 是标准依赖。正式安装缺依赖或需要刷新 wheel 时重新安装 PyPI 包：

```bash
uv tool install meeting-asr --python 3.14 --reinstall --refresh
```

`doctor` 发现配置或依赖问题时会输出 `Repair prompts`，可直接交给 agent 继续修复。

## 核心命令

```bash
meeting-asr project run "/path/to/meeting.mp4" --meeting-time "2026-04-29T15:07:42+08:00"
meeting-asr project review PROJECT_ID
meeting-asr project transcript show PROJECT_ID --kind auto
meeting-asr project correct diff PROJECT_ID
meeting-asr project correct accept PROJECT_ID
meeting-asr project correct eval-polish
meeting-asr project merge P1 P2 ... --out ./merged   # 把同一场会的多段闪记合并成单一转写包
meeting-asr voiceprint review PROJECT_ID
meeting-asr voiceprint review
meeting-asr voiceprint quality --review
```

`project run` 默认显示长任务进度，并把当前阶段、外部 task id、最近错误、本地词汇订正和 polish 状态写进 `project.json`。如果命令中断或怀疑卡住，先跑：

```bash
meeting-asr project show PROJECT_ID
```

Transcript polish 有独立评测集，不再只看 proposal diff 多少。改 prompt、模型或 guard 前后先跑：

```bash
uv run meeting-asr project correct eval-polish
uv run meeting-asr project correct eval-polish --model qwen3.6-plus
```

评测集说明见 [docs/polish-eval.md](docs/polish-eval.md)。

删除项目默认进 trash，不会直接物理删除：

```bash
meeting-asr project delete PROJECT_ID
meeting-asr project trash list
meeting-asr project trash restore TRASH_REF
meeting-asr project trash purge TRASH_REF --yes
```

## 文档地图

- [快速开始](docs/quick-start.md)：只讲全自动路径和 TUI 兜底路径。
- [项目介绍](docs/project-introduction.md)：面向推广的短文，说明核心功能和解决的痛点。
- [发布流程](docs/release.md)：PyPI Trusted Publishing 和 GitHub Release 发布步骤。
- [CLI 用户手册](docs/cli-user-guide.md)：命令、参数、产物、故障排查。
- [架构说明](docs/architecture.md)：分层结构和新代码放置规则。
- [开发者指南](docs/developer-guide.md)：安装、测试、completion 验证。
- [TUI 测试](docs/tui-testing.md)：Textual headless 测试约定。

## 关键边界

- DashScope ASR 只能接收公网 HTTP/HTTPS URL；本工具默认走 private OSS + 临时 signed GET URL。
- signed URL、token、secret、access key 不写入日志或 `project.json`。
- 默认项目身份是内容 hash 生成的 `p-...`，同一个源视频会复用同一个项目。
- 人类修正 speaker 的首选入口是 `meeting-asr project review PROJECT_ID`；`speakers apply --map` 是脚本化接口。
- 声纹是跨项目数据，属于稳定 person ID，不用姓名做主键。
- 文档只记录已验证路径；未验证的远端声纹 provider 不写成用户教程。
