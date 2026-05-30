# 更新日志

本项目的所有重要变更都会记录在这个文件中。

格式参考 [Keep a Changelog](https://keepachangelog.com/zh-CN/1.1.0/)，
并遵循 [Semantic Versioning](https://semver.org/lang/zh-CN/spec/v2.0.0.html)。

## [未发布]

### 新增

- 新增 `meeting-asr project merge <p1> <p2> ...`：把同一场会被钉钉拆成多段闪记（各自一个 project）的转写合并成单一转写包，原生支持中场休息分段的场景。按 `meeting_time` 时间序拼接，跨段**按声纹人 public id（`vpp`）归一发言人**——同一个人在不同段即使本地 speaker_id 不同、甚至某段没命名，也会对齐成同一发言人并取声纹库权威名；仅命名未连声纹的发言人默认按同名提升对齐到声纹人（`--no-name-to-vpp` 关闭）。时间轴连续打包（各段按音频时长偏移、单调不重叠），段界 header 保留各段原始会议时间/时长/句数。产出 `transcript_merged.txt` / `_corrected.txt`、`subtitle_merged.srt` / `_corrected.srt` 和结构化只读清单 `merge.json`（含段元信息与发言人归一审计轨）。单段退化为直接导出；合并为无状态操作，绝不回写原 project。
- `project run` / `project rerun` / `project transcribe` 在 ASR 提交后把本次随 DashScope 任务一起提交的热词表写入项目 `corrections/asr_hotwords.json`（含 `dashscope_vocabulary` 与逐条 hotword）。此前该文件只有 `project correct` 流程写，新转写完的项目看到的是空文件，让人误以为没给识别引擎喂热词——其实 lexicon vocabulary 一直在随任务提交。现在每次转写都会落地“本次实际提交了哪些专名”，便于核对 iSee / CLI / SKU 等热词是否生效。文件在下游产物失效（invalidation）之后写，避免被重跑清空；`--asr-hotwords off` 时记为空表。

## [0.8.0] - 2026-05-30

### 新增

- `project speakers apply --map <id>=@vpp-<public_id>` 支持按声纹库稳定人员 public id 绑定发言人：apply 时把 person 引用写入项目 `speaker_person_map.json`，capture 直接归到已有 person，并从声纹库取该 person 的显示名渲染转写。这避免了手工命名时因花名与库内“真名(花名)”不一致而给同一个人新建重复声纹条目。`--map <id>=<name>` 旧用法保持不变；`apply` 新增 `--store-dir` 用于解析 `@vpp` 的显示名。
- 新增 `meeting-asr voiceprint people merge <from_id> <into_id>`：把源声纹人员的样本并入目标人员（音频相同的样本按 clip 去重丢弃），随后删除清空的源人员，用于合并历史上同一个人被建成的多条声纹条目。带确认提示，`--yes` 跳过。
- 句级声纹改判扩展到未命名（低于命名阈值）的 speaker 簇。此前逐句声纹核对只覆盖已命名 speaker，未命名簇的句子被整批跳过——而这恰恰是最容易混入他人的场景。现在未命名簇里若某句明显且稳定地匹配到本会议中另一位**已确认** speaker（分数达到 foreign 阈值 0.55、且明显领先次选），会被标记为 `identity-foreign` 并交由稳定化流程改判到该 speaker；匹配到未确认身份（包括该簇自身真实说话人）的句子保持原状，避免整簇误判。

### 变更

- 升级到 typer 0.26：typer 把 Click 源码内置（vendored）并移除了对外部 click 包的依赖。CLI 表现层（本地化 help、解析错误面板、shell 补全、退出码）相应改用 typer 公共 API 实现，不再直接依赖 click，也不使用 typer 的私有内部模块；命令行的可见行为保持不变。
- 工作流进度条改用整个终端宽度自适应渲染，不再固定 120 列上限；窄终端下进度条自动收窄给描述让位，宽终端下描述与进度条都充分展开，显示更舒适。

## [0.7.0] - 2026-05-21

### 新增

- `project run` / `project transcribe` 会优先复用项目内已提取的音频，完整流程完成后可清理项目内视频副本，减少重复提取和磁盘占用。
- 项目 ASR 上传支持复用稳定的项目 OSS object，仍可用时只刷新签名 URL，避免重跑时重复上传同一份音频。
- 新增 `meeting-asr project rerun <project>` 作为显式 ASR 重跑入口，复用已有项目音频和 OSS 状态；`project transcribe` 保持兼容。
- 新增 Agent 自发现入口：`agent-guide`、`commands --json`、`commands --schema`、`version --json`，暴露 side effects、interactive、feature flags 和运行时指南。

### 变更

- `agent-guide` 增补重跑缓存、声纹样本状态、非交互运行、交付回报等 LLM Agent 指南。
- `project rerun` 和 ASR 失败恢复提示统一指向显式重跑命令。

## [0.6.2] - 2026-05-21

### 修复

- 修复 Project Review 里同一句反复编辑时 diff 基准漂移的问题，二次修改仍按加载时原文对比最终文本，并提供外部编辑器入口以兜底终端中文输入法兼容问题。
- 修复 Voiceprint Review 当前项目分数检查的颜色语义：接受 embedding 后的 `changed-best` 属于预期改善，显示为绿色；历史反向评测仍保留风险颜色。

## [0.6.1] - 2026-05-21

### 修复

- 项目转写复用已上传的项目音频 OSS 对象，仅刷新签名 URL，避免同一项目重复上传音频；如果重新签名失败，则回退到原上传路径。

## [0.6.0] - 2026-05-21

### 新增

- 新增 speaker 聚类质量诊断，并在 Project Review 中展示聚类状态、离群样本和混桶风险。
- 新增全量 speaker cluster 行级评分，支持逐句定位 speaker 样本离群。
- 新增逐句声纹身份诊断，可对每个句子判断是否更像另一个已知 speaker。
- `project run` 默认接入两轮逐句 speaker 稳定化：刷新诊断、自动改写高置信归属冲突、重新计算声纹分数。
- `project speakers sample-match` 支持 `--workers`，逐句 embedding 和声纹匹配可并发执行。
- Project Review 增加样本筛选能力，便于在大量句子中聚焦异常样本。

### 变更

- 统一声纹 embedding 音频预处理，减少源音量和格式差异对匹配结果的影响。
- Project Review 样本播放改为精确抽取句子片段，并调整样本双行布局与诊断命名。
- 时间戳敏感的预览、声纹匹配、聚类诊断和采样流程优先使用项目 ASR 音频，避免原始 source 与 ASR 音频时长不一致导致字幕和播放错位。

### 修复

- 修复显式 `--project-dir` 可能绕过同源项目复用、创建重复项目的问题。
- 修复 Project Review 保存后声纹诊断未刷新，导致页面继续展示过期诊断的问题。
- 修复 Project Review 预览缓存只看 mtime/size，可能复用错误音频来源缓存的问题。
- 修复 `project speakers apply` 可能覆盖已有说话人映射的问题。

## [0.5.0] - 2026-05-19

### 新增

- 新增 transcript polish 评测命令与评测用例集，覆盖 Qwen3.6 适配效果。
- 新增统一 DashScope chat 调用层，按模型配置路由不同端点。
- `project run` 支持通过配置自动接受 transcript polish 结果。

### 变更

- `project list` 输出更精简，并规范会议标题中的时间前缀，减少重复和不可区分标题。
- polish 接受后的项目运行状态会正确刷新，避免后续流程继续看到过期状态。

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

[未发布]: https://github.com/crhan/meeting-asr/compare/v0.7.0...HEAD
[0.7.0]: https://github.com/crhan/meeting-asr/compare/v0.6.2...v0.7.0
[0.6.2]: https://github.com/crhan/meeting-asr/compare/v0.6.1...v0.6.2
[0.6.1]: https://github.com/crhan/meeting-asr/compare/v0.6.0...v0.6.1
[0.6.0]: https://github.com/crhan/meeting-asr/compare/v0.5.0...v0.6.0
[0.5.0]: https://github.com/crhan/meeting-asr/compare/v0.4.0...v0.5.0
[0.4.0]: https://github.com/crhan/meeting-asr/compare/v0.3.0...v0.4.0
[0.3.0]: https://github.com/crhan/meeting-asr/compare/v0.2.0...v0.3.0
[0.2.0]: https://github.com/crhan/meeting-asr/compare/v0.1.0...v0.2.0
[0.1.0]: https://github.com/crhan/meeting-asr/releases/tag/v0.1.0
