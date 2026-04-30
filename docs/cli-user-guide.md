# Meeting-ASR CLI 用户手册

## 1. 检查环境

```bash
meeting-asr doctor --require-oss
meeting-asr doctor --require-oss --require-voiceprint-embedding
meeting-asr doctor --require-voiceprint-embedding
meeting-asr doctor --oss-upload-probe
```

`--require-oss` 只检查配置是否存在；`--oss-upload-probe` 会上传一个极小文本对象，签 URL 读回，再删除。
`--require-voiceprint-embedding` 会按当前 `voiceprint.embedding_provider` 检查声纹 embedding。
默认 provider 是 `local-speechbrain`，只检查本地依赖；切到 `bailian` 后才检查阿里云 endpoint 和 OSS。
`doctor` 遇到 fail/warn 会输出 `Repair prompts`，可以直接交给大模型继续修复。

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
meeting-asr voiceprint embed
meeting-asr voiceprint list
meeting-asr voiceprint show "欧丁"
meeting-asr voiceprint play "欧丁" --sample 1
meeting-asr voiceprint path
```

`capture` 会从当前 project 的 `asr/sentences.json` 和
`speakers/speaker_map.json` 选择已确认姓名的 speaker 参考片段，WAV 写入
`voiceprints/clips/`，索引写入 `voiceprints/voiceprints.sqlite`。
仍然是 `Speaker A`、`Speaker C` 这种匿名 label 的人会跳过，不进入声纹库。

`embed` 默认使用本地 `local-speechbrain` provider。先安装本地声纹依赖：

```bash
uv sync --extra local-voiceprint
```

如果 CLI 是 `uv tool install` 安装的，改用：

```bash
uv tool install --editable ".[local-voiceprint]" --force
```

默认配置：

```bash
meeting-asr config set voiceprint.embedding_provider "local-speechbrain"
meeting-asr doctor --require-voiceprint-embedding
```

如果要使用百炼/AnalyticDB 声纹检索 provider，切换 provider 并配置 endpoint：

```bash
meeting-asr config set voiceprint.embedding_provider "bailian"
meeting-asr config set voiceprint.embedding_endpoint "http://<adb-ai-app-host>:8100/audio/embedding"
meeting-asr doctor --require-oss --require-voiceprint-embedding
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
meeting-asr project speakers match --apply
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
meeting-asr voiceprint delete-sample "欧丁" --sample 1
meeting-asr voiceprint delete-speaker "欧丁" --yes
```

先用 `voiceprint show "欧丁"` 看样本编号，再用同一个编号播放或删除。

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
