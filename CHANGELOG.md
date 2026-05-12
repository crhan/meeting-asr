# 更新日志

本项目的所有重要变更都会记录在这个文件中。

格式参考 [Keep a Changelog](https://keepachangelog.com/zh-CN/1.1.0/)，
并遵循 [Semantic Versioning](https://semver.org/lang/zh-CN/spec/v2.0.0.html)。

## [未发布]

## [0.4.0] - 2026-05-12

### 新增

- `meeting-asr project show --json` 新增 `ignored_speakers` 字段以及 `speakers[]` 数组（含 `speaker_id` / `label` / `name` / `status` / `sample_count` / `match`），`status` 取值为 `matched | below-threshold | no-candidate | ignored | unnamed`，下游 agent 可直接判断 speaker 是否被忽略，不必再读 `speakers/speaker_ignore.json`。
- 共享 `effective_match_status` 与 `MATCH_STATUS_IGNORED`：CLI 渲染会把 `speaker_ignore.json` 中的 speaker 一律视为 `ignored`，不再误报为 below-threshold。
- Project Review TUI 时间轴视图支持对当前 sample 执行 speaker 归属重指派，保存后会同步刷新命名 transcript、字幕和 voiceprint 匹配状态。

### 变更

- `project speakers inspect` 对 ignored speaker 显示 `Status: ignored`，并跳过 voiceprint match 行；只在仍有非 ignored 的 below-threshold / no-candidate speaker 时才输出 “Recommended next step”。
- `project speakers review --summary`、`project speakers match`、`project run` 的 unresolved 计数与下一步推荐都会跳过 ignored speaker。
- Project Review TUI 会保留已命名的低信息 speaker，避免 review 入口把真实短反馈 speaker 直接隐藏。
- `apply_project_speakers()` 生成命名 transcript、字幕和 manifest 时继续过滤低信息 speaker，避免低信息 speaker 重新污染普通输出。
- Strict polish 批处理恢复可见进度，并在批次运行时持续写入 heartbeat，长任务不再表现为静默卡住。

### 修复

- DashScope strict polish 部分批次失败时会保留已经通过 guard 的修正结果，不再因为单个批次失败丢弃整轮可用输出。

## [0.3.0] - 2026-05-09

### 新增

- Project Review TUI 新增「时间轴视图」（`t` 键切换）：按 ASR 切分的真实时间顺序展示所有句子，便于边听边核对。
- 在时间轴视图下按 `r` 可把当前句子重新指派给另一个 speaker。
- 按 `s` 保存若存在归属变更，会自动跑后链路：写回 `asr/sentences.json` / `sentences_corrected.json`，重新生成命名 transcript 与字幕、匿名 `transcript_speakers.txt`，删除被归属变更覆盖的声纹样本，并重跑 voiceprint 匹配（`speaker_matches.json`）。
- Voiceprint Review 播放样本时会在状态栏显示播放进度，并在当前 sample 行标记 `PLAY`。
- Polish proposal 中每条改动会带上 `change_type`（typo / term / case / punct / dup / filler / restart / emphasis），并在 markdown 中按类型分组展示。
- `project correct polish accept` 新增 `--select`（按编号或区间挑选）和 `--types`（按 change_type 过滤），可只接受需要的类别而不是全量。
- Polish 每次运行会写出 `polish_strict_meta_<ts>_<model>.json` sidecar，包含所有候选的 LLM 输出、change_type 和 guard 判定，便于离线分析。

### 变更

- 声纹采样默认勾选策略从“最高分前 N 个”调整为“分数达标后按时间分散选择”，降低单一说话状态过拟合的风险。
- Polish 默认改为面向下游摘要 agent 的严格模式：聚焦 ASR 噪声（重复 / 语气词 / 重启 / 强调）和 typo/术语/大小写/标点修正，禁止跨句借用、ASCII 幻觉、以及删除 `我觉得` / `可能` / `或许` / `对吧` 等承载事实信号的修饰词。`project correct polish` 与 `project run` 都默认走严格 polish，可用 `--legacy-polish` 回退到旧版重写行为。
- 严格 polish 在 LLM 之后增加确定性 guard：长度比 / 长度差 / ASCII 编辑距离幻觉 / 保护词删除 / 跨句借用直扫，全部失败时按旧路径抛出 `model_error`，部分批次失败时通过 `Model fallback` 信息提示用户。
- Release workflow 默认安装 ffmpeg，发布构建环境与本地保持一致。

## [0.2.0] - 2026-05-09

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

[未发布]: https://github.com/crhan/meeting-asr/compare/v0.4.0...HEAD
[0.4.0]: https://github.com/crhan/meeting-asr/compare/v0.3.0...v0.4.0
[0.3.0]: https://github.com/crhan/meeting-asr/compare/v0.2.0...v0.3.0
[0.2.0]: https://github.com/crhan/meeting-asr/compare/v0.1.0...v0.2.0
[0.1.0]: https://github.com/crhan/meeting-asr/releases/tag/v0.1.0
