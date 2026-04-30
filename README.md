# Meeting-ASR

`meeting-asr` 是一个项目化 CLI：从本地 MP4/MOV/MKV 创建项目，抽取 mono 16kHz 音频，上传 private OSS 并签出 URL，调用阿里云 DashScope / 百炼 Fun-ASR 异步转写，最后生成文本、字幕和 speaker 人工标注结果。

## 快速开始

```bash
uv venv
uv sync --all-groups
uv run meeting-asr --help
```

安装成可直接运行的命令：

```bash
uv tool install --editable . --force
meeting-asr completion install zsh
exec zsh
```

`completion install` 支持 `bash`、`zsh`、`fish`、`powershell` 和 `pwsh`；也可以用
`meeting-asr completion zsh` 这类命令直接输出补全脚本。

配置遵循 XDG Base Directory：

- 配置文件：`$XDG_CONFIG_HOME/meeting-asr/config.json`，默认 `~/.config/meeting-asr/config.json`
- 默认项目：`$XDG_DATA_HOME/meeting-asr/projects`，默认 `~/.local/share/meeting-asr/projects`

```bash
meeting-asr config set dashscope.api_key "<your-dashscope-api-key>"
meeting-asr config set dashscope.base_url "https://dashscope.aliyuncs.com/api/v1"
meeting-asr config set oss.access_key_id "<your-oss-access-key-id>"
meeting-asr config set oss.access_key_secret "<your-oss-access-key-secret>"
meeting-asr config set oss.bucket_name "<your-bucket>"
meeting-asr config set oss.region "<your-region>"
meeting-asr config set oss.endpoint "<your-oss-endpoint>"
meeting-asr doctor --require-oss
```

## 主流程

一条命令创建项目并转写：

```bash
meeting-asr project run "/path/to/meeting.mp4" \
  --title "供应商管理AI治理" \
  --meeting-time "2026-04-29T15:07:42+08:00"
```

分步执行：

```bash
meeting-asr project create "/path/to/meeting.mp4" --title "供应商管理AI治理"
cd "<Project created 输出的路径>"
meeting-asr project transcribe
meeting-asr project speakers inspect
meeting-asr project speakers preview
meeting-asr project speakers apply
meeting-asr transcript show
```

`project create` 会复制源视频到 `source/`，后续命令只需要项目目录，不需要再次传视频路径。
在项目目录内执行时，项目路径参数默认是当前目录；在其他目录执行时仍可显式传项目路径。

转写结果可以用独立 CLI 查看：

```bash
meeting-asr transcript list
meeting-asr transcript show
meeting-asr transcript path --kind srt
meeting-asr transcript open --kind named
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

- [CLI 用户手册](docs/cli-user-guide.md)
- [开发者指南](docs/developer-guide.md)
