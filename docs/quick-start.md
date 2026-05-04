# Meeting-ASR 快速开始

这份文档只讲两条使用路径：全自动路径和 TUI 兜底路径。完整参数说明见
`docs/cli-user-guide.md`。

## 1. 全自动路径

适合已经积累了声纹库，并且希望让命令自己跑到可用结果。

```bash
meeting-asr project run "/path/to/meeting.mp4"
```

`project run` 会自动执行：

1. 创建或复用项目。同一个源视频不会重复创建新项目。
2. 抽取音频并提交 DashScope 转写。
3. 下载并写出匿名转写结果。
4. 用 DashScope 文本模型生成会议标题和摘要。
5. 用声纹库匹配 speaker。
6. 自动应用 accepted 的 speaker 匹配。

看项目 ID：

```bash
meeting-asr project list
```

查看结果：

```bash
meeting-asr project transcript list PROJECT_ID
meeting-asr project transcript show PROJECT_ID --kind named
meeting-asr project speakers preview PROJECT_ID
```

如果输出是 `Project automation completed.`，说明所有 speaker 都已自动确认。
最终重点产物是：

- `exports/transcript_named.txt`
- `exports/subtitle_named.srt`
- `exports/meeting_summary.md`

如果输出是 `Project automation needs review.`，说明至少有一个 speaker 没有自动确认，
进入 TUI 路径。

## 2. TUI 兜底路径

适合处理未匹配、低分、冲突或需要人工确认的 speaker。

```bash
meeting-asr project review PROJECT_ID
```

如果不想复制 Project ID：

```bash
meeting-asr project review
```

这会先打开项目列表，选中后进入 review TUI。

TUI 里主要看三处：

- `Next/Done`：下一步做什么，或者是否已经完成。
- speaker/sample 区：逐个确认 speaker，播放样例，接受 match 或输入姓名。
- `Output`：最终产物是否已经生成。

确认完成后再检查结果：

```bash
meeting-asr project transcript show PROJECT_ID --kind named
meeting-asr project speakers preview PROJECT_ID
```

如果这次人工确认了新人，把他们补进跨项目声纹库：

```bash
meeting-asr voiceprint review PROJECT_ID
meeting-asr voiceprint embed
```

## 3. 日常判断

- 先跑 `project run`。
- 如果自动完成，直接看 `transcript_named.txt`。
- 如果需要 review，跑 `project review PROJECT_ID`。
- 以后同一个视频再次运行，会复用已有项目，不会因为日期变化生成新 ID。
- 如果命令失败，先看 `Next step` 提示；配置/环境问题会给出对应的 `meeting-asr doctor ...` 命令。
- 想一次性做完整集成检查，运行 `meeting-asr doctor --full`；给 agent 用 `meeting-asr doctor --full --json`。
