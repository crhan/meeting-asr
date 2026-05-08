# Meeting-ASR：一条命令把钉钉会议录音变成可用转写

会议录音最常见的问题不是“有没有摘要”，而是：录音听不动、转写要等、Speaker A/B/C 看不懂、人名和专业词经常错。`meeting-asr` 只瞄准一个核心结果：把钉钉会议录音变成可校对、可搜索、知道谁说了什么的转写文本。

Summary、标题、成本统计都只是辅助信息。真正重要的产物是最终转写：

```text
exports/transcript_named.txt          # 带发言人姓名的最终转写
exports/transcript_named_corrected.txt # 纠错后的最终转写，如有纠错
exports/subtitle_named.srt            # 带发言人姓名的字幕
```

项目地址：[https://github.com/crhan/meeting-asr](https://github.com/crhan/meeting-asr)

## 第一部分：痛点与便捷性

### 痛点：会议录音不是会议记录

拿到一段钉钉会议录音后，真正麻烦的是后面的链路：

```text
抽音频 -> 上传 -> 等 ASR -> 下载结果 -> 识别发言人 -> 修人名 -> 修错词 -> 导出最终文本
```

这条链路如果靠人手动做，很容易卡在三类问题上：

- 不知道当前跑到哪一步，长任务像是卡住了。
- ASR 只能给出 `Speaker A`，读者不知道是谁在说话。
- 专业词、人名、系统名被识别错，同类错误会在全文反复出现。

`meeting-asr` 的设计是让 Agent 和人都用同一个稳定入口：先自动跑，能自动接受的直接接受，不确定的再进入 review。

### 一个命令先拿到结果

常用入口只有一个：

```bash
meeting-asr project run meeting.mp4
```

它会创建或复用 Project，完成音频处理、OSS 上传、DashScope ASR、说话人匹配、最终产物写出。用户不需要先理解中间目录，也不需要手动拼一串命令。

典型输出应该让人立刻知道三件事：项目是什么、最终转写在哪里、是否需要 review。

```text
Project automation completed.
Project ID: p-xxxxxxxxxxxxxxxx
Title: 维修样板间目标对齐
Detected speakers: 6
Voiceprint matches: 5/6 accepted

Outputs:
  exports/transcript_named.txt
  exports/subtitle_named.srt

Next:
  meeting-asr project review p-xxxxxxxxxxxxxxxx
```

> 截图位 1：`project run` 完成输出  
> 建议截图内容：Project ID、speaker 匹配结果、最终转写路径、下一步 review 命令。  
> 建议文件名：`docs/assets/meeting-asr-project-run.png`

### Agent 为什么会更方便

对 Agent 来说，`project run` 的关键不是“炫功能”，而是输出结构稳定：

- 有稳定的 `Project ID`，后续命令不用依赖当前目录。
- 有明确的最终转写路径，Agent 可以直接读取产物。
- 如果有异常，会给出 `project review PROJECT_ID` 这种可执行下一步。
- 长任务状态会写入 Project，用户中断后还能 `project show` 查看。

```bash
meeting-asr project show p-xxxxxxxxxxxxxxxx
meeting-asr project transcript show p-xxxxxxxxxxxxxxxx --kind named
```

> 截图位 2：`project show` 项目状态页  
> 建议截图内容：会议时间、当前阶段、最终产物、speaker 状态、下一步命令。  
> 建议文件名：`docs/assets/meeting-asr-project-show.png`

### 这里自然会出现一个问题

如果一条命令就能给出最终转写，那发言人姓名是怎么来的？

答案不是“让用户每次手填”。`meeting-asr` 把 speaker review、声纹采样、embedding、历史项目评测放进同一个流程：

- 低置信度 speaker 不自动写名字，只展示最佳候选和分数。
- 用户在 TUI 里播放片段、确认人名、忽略无效 speaker。
- 确认后的样本进入全局声纹库，生成本地 embedding。
- 后续项目用声纹库自动匹配，越用越少手工确认。

下面用一个完整流程演示这件事。

## 第二部分：全流程演示

### 1. 从 Project Run 开始

先把会议视频交给自动流程：

```bash
meeting-asr project run meeting.mp4
```

如果所有 speaker 都能自动接受，用户可以直接看最终转写：

```bash
meeting-asr project transcript show p-xxxxxxxxxxxxxxxx --kind named
```

如果存在低于阈值、无候选、或需要人工确认的 speaker，输出会引导进入 review：

```bash
meeting-asr project review p-xxxxxxxxxxxxxxxx
```

> 截图位 3：自动流程提示需要 review  
> 建议截图内容：below-threshold speaker、best candidate、score、recommended next step。  
> 建议文件名：`docs/assets/meeting-asr-run-needs-review.png`

### 2. 进入 Project Review 浏览 speaker

Project Review 是人工介入的主入口。它把当前项目、speaker 列表、样本片段、匹配分数和下一步动作放在一个 TUI 里。

常用浏览操作：

```text
↑/↓ 或 j/k       在当前区域上下移动
←/→ 或 h/l       在 speaker 列和 sample 列之间切换
PageUp/PageDown  翻页查看更多 sample
?                查看快捷键
```

> 截图位 4：Project Review 总览  
> 建议截图内容：顶部项目状态、左侧 speaker 列表、右侧 sample 列表、分数和状态。  
> 建议文件名：`docs/assets/meeting-asr-review-overview.png`

### 3. 听片段，确认或忽略 speaker

不要只看名字，要先听片段。TUI 支持直接播放当前 sample，再决定接受候选、改名或忽略。

```text
space            播放或停止当前 sample
/                修改当前 speaker 的人名
a                接受当前声纹候选
i                忽略无效 speaker，例如全是语气词或噪音
m                用最新全局声纹库重新匹配当前项目
s                保存 speaker 映射和命名产物
```

保存后会写出：

```text
exports/transcript_named.txt
exports/subtitle_named.srt
```

> 截图位 5：确认 speaker 人名  
> 建议截图内容：候选人列表、score、当前 speaker sample、确认后的姓名状态。  
> 建议文件名：`docs/assets/meeting-asr-edit-speaker-name.png`

### 4. 修改转写错误，并把经验沉淀为热词

如果样本里有错词，直接在 TUI 里改当前句子。这个修改不是只改一行；系统会基于这次修改理解上下文，扫描全文并提出 correction proposal。

```text
e                修改当前 sample 文本
s                保存 review，并生成全文 correction proposal
d                查看 correction diff
x                include/exclude 当前修改项
a                应用已选修改
Esc              返回上一级
```

接受后会生成纠错产物和热词材料：

```text
exports/transcript_named_corrected.txt
corrections/asr_hotwords.json
```

这样下一次 ASR 可以带着热词再跑，减少同类专业词错误。

> 截图位 6：词级 correction diff  
> 建议截图内容：Before/After、词级高亮、include/exclude 状态、应用按钮提示。  
> 建议文件名：`docs/assets/meeting-asr-correction-diff.png`

### 5. 进入 Voiceprint Review 做声纹采样

speaker 名字确认后，就可以把可靠样本沉淀到全局声纹库。入口在 Project Review 里：

```text
v                进入 Voiceprint Review
```

Voiceprint Review 里先看候选 sample。不是所有 sample 都应该进入声纹库：如果片段里混入多人说话、背景噪音太重，应该排除。

```text
space            播放或停止当前 sample
x                勾选或排除当前 sample
d                取消当前人的全部 sample
s                采集已选 sample、生成 embedding，并做评测
```

> 截图位 7：Voiceprint Review 采样页  
> 建议截图内容：候选 speaker、person id、score、sample 勾选状态、当前播放片段。  
> 建议文件名：`docs/assets/meeting-asr-voiceprint-sampling.png`

### 6. 评测 embedding，再决定是否接受

embedding 不是直接覆盖。系统会先评测这批新增声纹对当前项目和历史项目的影响：

- 当前项目 score 是否上升。
- 历史项目是否出现 score 明显下降。
- 是否出现人名变化。
- 是否跌破自动接受阈值。

黄色代表需要注意，红色代表风险较高。用户确认后才接受；不确认就回滚。

```text
a                接受本次 embedding
r 或 Esc/q       回滚本次 embedding
```

> 截图位 8：Voiceprint embedding 评测页  
> 建议截图内容：当前项目分数变化、历史项目反向评测、黄色/红色风险、接受/回滚提示。  
> 建议文件名：`docs/assets/meeting-asr-voiceprint-evaluation.png`

### 7. 回到 Project，重新匹配并导出最终转写

声纹库更新后，可以回到 Project Review 重新匹配当前项目：

```text
m                用最新声纹库重新匹配当前项目
s                保存最终结果
```

最终读者只需要看转写：

```bash
meeting-asr project transcript show p-xxxxxxxxxxxxxxxx --kind corrected
```

如果没有做纠错，就看命名转写：

```bash
meeting-asr project transcript show p-xxxxxxxxxxxxxxxx --kind named
```

## 最终结果

`meeting-asr` 的目标不是生成一堆中间文件，而是把会议录音变成一份可靠转写：

- 知道每句话是谁说的。
- 能忽略无意义 speaker。
- 能修专业词、人名、系统名。
- 能把人名和声纹经验沉淀下来，下次少做重复劳动。
- Agent 可以拿 Project ID 和稳定命令继续处理，不依赖人工记路径。

这就是它解决的实际问题：让会议录音从“只能回放”变成“可以直接使用的文本资产”。

## 项目链接和手册

如果想实际试用或查看实现，可以从这里开始：

- 项目主页：[GitHub - crhan/meeting-asr](https://github.com/crhan/meeting-asr)
- 快速开始：[docs/quick-start.md](https://github.com/crhan/meeting-asr/blob/main/docs/quick-start.md)
- CLI 用户手册：[docs/cli-user-guide.md](https://github.com/crhan/meeting-asr/blob/main/docs/cli-user-guide.md)
- 架构说明：[docs/architecture.md](https://github.com/crhan/meeting-asr/blob/main/docs/architecture.md)
- 开发者手册：[docs/developer-guide.md](https://github.com/crhan/meeting-asr/blob/main/docs/developer-guide.md)
- TUI 测试说明：[docs/tui-testing.md](https://github.com/crhan/meeting-asr/blob/main/docs/tui-testing.md)
