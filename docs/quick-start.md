# Meeting-ASR 快速开始

这份文档只回答一个问题：拿到一个会议视频后怎么跑完。完整参数见 [CLI 用户手册](cli-user-guide.md)。

## 路径 1：全自动

适合已有声纹库，或者先接受自动结果再人工补足。

```bash
meeting-asr project run "/path/to/meeting.mp4"
```

这个命令会：

1. 创建或复用项目；同一个源视频不会重复生成新 ID。
2. 抽取音频、上传 private OSS、提交 DashScope ASR。
3. 下载并标准化转写结果。
4. 应用已入库的本地词汇订正规则，例如把 `IC` 订正为 `iSee`。
5. 生成会议标题和回忆索引。
6. 生成 transcript polish proposal。
7. 用全局声纹库匹配 speaker。
8. 自动应用 accepted speaker 匹配并写出 named transcript/SRT。

看项目：

```bash
meeting-asr project list
meeting-asr project show PROJECT_ID
```

看结果：

```bash
meeting-asr project transcript show PROJECT_ID --kind auto
meeting-asr project speakers preview PROJECT_ID
```

最终重点产物：

```text
exports/transcript_named_corrected.txt  # 有本地词汇订正或人工纠错时
exports/transcript_named.txt
exports/subtitle_named.srt
exports/meeting_summary.md
```

如果 `project show PROJECT_ID` 显示 `Transcript polish: proposal ready`，先看 diff，再接受：

```bash
meeting-asr project correct diff PROJECT_ID
meeting-asr project correct accept PROJECT_ID
```

如果输出是 `Project automation needs review.`，进入 TUI 路径。

## 路径 2：TUI 人工兜底

适合处理低分、未匹配、speaker 冲突、需要忽略的 speaker、词汇纠错和声纹采样。

```bash
meeting-asr project review PROJECT_ID
```

不知道 Project ID 时直接运行：

```bash
meeting-asr project review
```

TUI 里重点看：

- 顶部 `Next/Done`：下一步或完成状态。
- 左侧 speaker：match 状态、分数、是否 ignored。
- 右侧 sample：逐条播放、确认或排除。
- `?`：快捷键。

常用键：

```text
j/k 或 ↑/↓     当前列上下移动
h/l 或 ←/→     切换左右列
space          播放/停止当前 sample
a              接受当前声纹匹配
i              忽略当前 speaker
/              搜索或输入人名
c              修改转写文本并生成 correction proposal
v              进入 Voiceprint Review，采样、embedding、评测
s              保存当前 review
```

保存后检查：

```bash
meeting-asr project transcript show PROJECT_ID --kind named
meeting-asr project speakers preview PROJECT_ID
```

如果这次确认了新人，在 Project Review 里按 `v` 进入声纹采样；或者用 CLI：

```bash
meeting-asr voiceprint review PROJECT_ID
meeting-asr voiceprint embed
```

## 故障时

```bash
meeting-asr project show PROJECT_ID
meeting-asr doctor --full
meeting-asr doctor --full --json
```

`project show` 看当前阶段、外部 task id、缺失产物和恢复命令。`doctor --full --json` 给 agent 或脚本用，字段和值保持英文稳定。
