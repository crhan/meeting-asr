# 更新日志

本项目的所有重要变更都会记录在这个文件中。

格式参考 [Keep a Changelog](https://keepachangelog.com/zh-CN/1.1.0/)，
并遵循 [Semantic Versioning](https://semver.org/lang/zh-CN/spec/v2.0.0.html)。

## [未发布]

### 新增

- 新增统一的声纹质量检查 TUI，可在全局声纹库中检查样本质量、播放单个样本、原地刷新评分并修改样本状态。
- 新增声纹样本生命周期状态 `verified-active`，用于标记“人工确认是本人”的样本：继续参与匹配，但不再作为质量风险提示。
- 新增声纹质量原因的中英文说明，让 TUI 中的离群、低分、一致性等判断更容易理解。
- 新增声纹采样候选池：采样规划时每个 speaker 最多展示 12 个候选样本，并只把请求数量内的 top 样本标记为 `recommended`。
- 新增采样候选的可解释信息，包括 `recommended` / `candidate`、选择分数，以及 duration/text/boundary 三类评分细节。
- 新增基于 embedding 中心性的最终样本选择：真正写入声纹库前，优先保留更接近该 speaker 候选簇中心的样本。

### 变更

- 声纹采样不再只取最长的转写片段，而是按时长、文本信息量、speaker 边界安全性综合评分。
- 声纹采样现在会优先选择时间上更分散的样本，并避开低信息量的语气词片段。
- 声纹 embedding 默认使用标准化后的音频片段，减少音量差异对 embedding 的影响。
- 声纹质量检查播放样本时优先播放标准化音频。
- 项目 speaker 匹配现在会缓存项目侧 probe embedding，并并行执行匹配，减少重复计算。
- Voiceprint Review 和 Voiceprint Quality 的 TUI 显示更清晰，质量状态变更后可以在页面内刷新，不需要退出重进。

### 修复

- 修复重复历史项目导致同一段音频被重复采集进声纹库的问题；同一 speaker 下相同音频 hash 的样本会被去重。
- 修复声纹质量 TUI 中 Rich markup 被当作普通文本显示的问题，例如 `[dim]` / `[cyan]`。
- 修复修改样本状态后质量检查页面状态不刷新的问题。
- 修复历史项目反向评测中 unchanged 分数被误标为严重风险的问题。

## [0.1.0] - 2026-05-09

### 新增

- 首个公开版本，提供基于 project 的 Meeting-ASR CLI。
- 新增项目创建、会议转写、转写导出、speaker review、声纹匹配、词汇纠错 review，以及 GitHub Actions 发布基础能力。

[未发布]: https://github.com/crhan/meeting-asr/compare/v0.1.0...HEAD
[0.1.0]: https://github.com/crhan/meeting-asr/releases/tag/v0.1.0
