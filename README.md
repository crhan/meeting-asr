# Meeting-ASR

`meeting-asr` 是一个项目化会议转写 CLI：输入本地视频，创建稳定 Project ID，抽取音频，上传 private OSS 签名 URL，调用 DashScope/Fun-ASR 转写，生成会议摘要、转写文本、字幕，并通过 TUI 完成 speaker、词汇纠错和声纹库维护。

## 先看这里

只想跑一次会议转写，读 [快速开始](docs/quick-start.md)。

常用路径只有两条：

```bash
# 1. 全自动：创建/复用项目、转写、摘要、声纹匹配、输出产物
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
exports/transcript_named.txt
exports/subtitle_named.srt
exports/meeting_summary.md
```

查看结果：

```bash
meeting-asr project transcript show PROJECT_ID --kind named
meeting-asr project speakers preview PROJECT_ID
```

## 安装和配置

开发环境：

```bash
uv venv
uv sync --all-groups
uv run meeting-asr --help
uv run pytest -q
```

安装全局可执行命令（本地开发）：

```bash
scripts/install-tool.sh
scripts/install-tool.sh --check
meeting-asr completion install zsh
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

本地声纹 embedding 默认使用 `local-speechbrain`。如果全局命令缺依赖，重新运行：

```bash
scripts/install-tool.sh
```

`doctor` 发现配置或依赖问题时会输出 `Repair prompts`，可直接交给 agent 继续修复。

## 核心命令

```bash
meeting-asr project run "/path/to/meeting.mp4" --meeting-time "2026-04-29T15:07:42+08:00"
meeting-asr project review PROJECT_ID
meeting-asr project transcript show PROJECT_ID --kind named
meeting-asr project correct diff PROJECT_ID
meeting-asr project correct accept PROJECT_ID
meeting-asr voiceprint review PROJECT_ID
meeting-asr voiceprint embed
meeting-asr voiceprint review
```

`project run` 默认显示长任务进度，并把当前阶段、外部 task id、最近错误和 polish 状态写进 `project.json`。如果命令中断或怀疑卡住，先跑：

```bash
meeting-asr project show PROJECT_ID
```

删除项目默认进 trash，不会直接物理删除：

```bash
meeting-asr project delete PROJECT_ID
meeting-asr project trash list
meeting-asr project trash restore TRASH_REF
meeting-asr project trash purge TRASH_REF --yes
```

## 文档地图

- [快速开始](docs/quick-start.md)：只讲全自动路径和 TUI 兜底路径。
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
