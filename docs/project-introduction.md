# Meeting-ASR：把钉钉会议录音变成可校对、可复用的会议记录

会议录音本身不是结果。真正麻烦的是三件事：转写要等、人名要猜、专业词总被识别错。`meeting-asr` 做的是一条从钉钉会议录音到可用会议材料的流水线：先自动跑完，再把不确定的部分交给人快速确认，确认结果还能沉淀到下一次使用。

## 核心功能

### 会议录音转写

输入一段会议录音或视频，工具会创建稳定的 Project，抽取音频，上传 private OSS 签名 URL，调用 DashScope/Fun-ASR 转写，并写出最终产物：

```text
exports/transcript_named.txt      # 带发言人姓名的转写文本
exports/subtitle_named.srt        # 带发言人姓名的字幕
exports/meeting_summary.md        # 会议标题和摘要
```

用户不用关心中间目录，也不用反复手动拼接命令。常用入口只有一个：

```bash
meeting-asr project run meeting.mp4
```

### 说话人识别

ASR 只给出 Speaker A、Speaker B 还不够。会议记录真正可读，必须知道每句话是谁说的。`meeting-asr` 会把 speaker 分段、样本片段和最终文本放进同一个 Project，并提供 review 入口处理低置信度 speaker、噪音 speaker 和需要忽略的片段。

```bash
meeting-asr project review PROJECT_ID
```

### 声纹匹配

人工确认过的人不会每次从零开始。声纹样本会进入跨项目声纹库，后续项目可以直接用本地 embedding 做匹配。匹配结果会保留分数和阈值：高置信度自动接受，低于阈值只给候选，不偷偷写名字。

这解决的是一个很具体的问题：同一个人反复出现在不同会议里，系统应该越来越会认，而不是每次都让人重新手填。

## 用户体验优化

### 一条命令跑完整流程

`project run` 会串起项目创建、音频处理、OSS 上传、ASR、摘要、声纹匹配和产物写出。长任务会显示当前阶段；中断后也可以用 `project show` 查看卡在哪一步。

```bash
meeting-asr project show PROJECT_ID
```

### TUI 里完成说话人校对

低分 speaker 不应该逼用户手写 `0=张三 1=李四`。TUI 会把 speaker、候选人、匹配分数和样本片段放在一起：用户可以播放片段、确认人名、忽略无效 speaker，或者进入声纹采样。

> 配图位 1：Project Review TUI  
> 建议截图：左侧 speaker 列表、右侧 sample 播放区域、顶部状态栏和下一步提示。  
> 建议文件名：`docs/assets/meeting-asr-project-review.png`

### 声纹采集和评测

确认 speaker 之后，可以直接进入 Voiceprint Review。这里不是盲目采样：用户先听候选 sample，排除混入多人说话或质量差的片段，再写入全局声纹库并生成 embedding。

新 embedding 生效前会做评测：看当前项目分数是否上升，也看历史项目有没有明显下降或人名变化。异常会标红，避免一次坏样本污染整个声纹库。

> 配图位 2：Voiceprint Review TUI  
> 建议截图：候选 sample、score、已选/未选状态、历史项目反向评测结果。  
> 建议文件名：`docs/assets/meeting-asr-voiceprint-review.png`

### 转写错误修正和热词同步

专业词、人名、系统名经常被 ASR 识别错。`meeting-asr` 支持在 review 里直接修改转写文本，然后生成全篇 correction proposal。用户可以看 diff、逐条接受或排除。

接受后的纠错经验会进入本地词库，并能投影成 DashScope ASR 热词表。下一次转写时，系统会带着这些热词再跑，减少同类错误重复出现。

> 配图位 3：Correction Diff TUI  
> 建议截图：词级 diff、include/exclude 选择、proposal 汇总。  
> 建议文件名：`docs/assets/meeting-asr-correction-diff.png`

### 本地化声纹处理

当前默认使用本地 `local-speechbrain` 声纹 embedding。声纹采样、embedding、匹配都可以在本机完成，不依赖额外远端声纹服务。对 Agent 工作流来说，这意味着少一个外部申请环节，也少一个不稳定的配置点。

## 典型使用路径

```bash
# 自动跑完整会议
meeting-asr project run meeting.mp4

# 如果有人名、声纹或转写需要人工确认
meeting-asr project review PROJECT_ID

# 查看最终转写
meeting-asr project transcript show PROJECT_ID --kind named
```

最终目标很简单：会议结束后，用户拿到的不是一段录音，而是一份能看、能搜、知道谁说了什么、并且会随着使用变准的会议记录。
