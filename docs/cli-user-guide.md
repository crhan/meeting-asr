# Meeting-ASR CLI 用户手册

## 1. 检查环境

```bash
meeting-asr doctor --require-oss
meeting-asr doctor --oss-upload-probe
```

`--require-oss` 只检查配置是否存在；`--oss-upload-probe` 会上传一个极小文本对象，签 URL 读回，再删除。

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

## 3. 创建项目

```bash
meeting-asr project create "/path/to/meeting.mp4" \
  --title "供应商管理AI治理" \
  --meeting-time "2026-04-29T15:07:42+08:00"
```

成功后 CLI 会输出可复制命令：

```bash
cd "/path/to/project"
meeting-asr project transcribe
meeting-asr project status
```

默认项目目录遵循 XDG：`~/.local/share/meeting-asr/projects`。

## 4. 转写

如果 OSS 已配置，默认使用 private OSS signed URL：

```bash
meeting-asr project transcribe
```

如果你已经有公网可访问音频 URL：

```bash
meeting-asr project transcribe \
  --file-url "https://example.com/audio.flac" \
  --oss-upload false
```

## 5. 人工确认 speaker

```bash
meeting-asr project speakers inspect
meeting-asr project speakers preview
meeting-asr project speakers preview --speaker-id 3
```

确认后写入映射：

```bash
meeting-asr project speakers apply
```

`apply` 默认交互式逐个 speaker 提示输入人名，并展示该 speaker 的样例文本。
如果样例还不足以确认，在姓名提示处输入 `/more` 会继续输出更多样例。
输入过 `/more` 后，可以在下一次提示里按上方向键召回 `/more`。
如果要脚本化执行，仍可使用 `--map 0=欧丁 --map 1=敬悦`。

## 6. 记录跨项目声纹

声纹库是跨项目的，不放在当前 project 目录。默认存放位置遵循 XDG：
`~/.local/share/meeting-asr/voiceprints/`。

```bash
meeting-asr voiceprint capture
meeting-asr voiceprint list
meeting-asr voiceprint show "欧丁"
meeting-asr voiceprint path
```

`capture` 会从当前 project 的 `asr/sentences.json` 和
`speakers/speaker_map.json` 选择每个 speaker 的参考片段，WAV 写入
`voiceprints/clips/`，索引写入 `voiceprints/voiceprints.sqlite`。

如果只想看会切哪些片段，不写文件和数据库：

```bash
meeting-asr voiceprint capture --dry-run
```

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
- `asr/raw_result.json`：DashScope 原始结果
- `asr/sentences.json`：标准化逐句结果
