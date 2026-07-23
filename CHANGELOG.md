# 更新日志

本项目的所有重要变更都会记录在这个文件中。

格式参考 [Keep a Changelog](https://keepachangelog.com/zh-CN/1.1.0/)，
并遵循 [Semantic Versioning](https://semver.org/lang/zh-CN/spec/v2.0.0.html)。

## [0.19.0] - 2026-07-23

### 新增

- **新增本地声纹嵌入 provider `local-campp`（3D-Speaker CAM++ 中文模型）**：网络结构从 modelscope/3D-Speaker（Apache-2.0）vendor 进 `src/app/infra/campplus.py`，复用既有 torch/torchaudio，零新增依赖；28MB checkpoint 首次使用时从 ModelScope 下载并按 sha256 校验（缓存文件每次加载前复验，损坏自动重下），缓存于 `$XDG_CACHE_HOME/meeting-asr/models/campplus/`。真实库实测对中文说话人分离度显著优于 SpeechBrain ECAPA（同人最低分 0.450→0.546、逐句配对同人均值 0.609→0.701，异人分布相近）。
- **新增 `voiceprint.provider` 配置键**（env `MEETING_ASR_VOICEPRINT_PROVIDER`）选择默认声纹 provider；`voiceprint embed` 新增 `--provider`；仅有 `--model` 的命令按 model key 前缀自动反推 provider，显式 provider 与可识别 model key 冲突时报错拒绝，防止向量命名空间被污染。doctor / web doctor 按配置的 provider 检查依赖。

### 变更

- **默认声纹 provider 切换为 `local-campp`**：老库无缝迁移——match / sample-match 入口自动为当前模型增量回填缺失的库向量（每样本约 1 秒，一次性），speechbrain 向量原样保留在库中；`config set voiceprint.provider local-speechbrain` 随时切回。两个模型的向量按 model key 并存，可用 `--model` 指定任一模型对比。
- 后台向量回填遇到 clip 文件已删除的陈旧登记行时跳过并告警，不再使整次匹配失败；`voiceprint embed` 命令保持严格报错语义。

### 修复

- web `/api/doctor` 不再返回两条同名的 voiceprint-embedding 检查项。
- `project run` 匹配阶段的运行记录不再硬编码 provider 名，按实际解析的 provider 记录。

## [0.18.0] - 2026-07-23

### 新增

- **OSS 上传自动兜底生命周期规则**：每次上传前检查 bucket 上是否存在 `meeting-asr-auto-delete` 规则，缺失时自动补一条（`meeting-asr/` 前缀）；已存在同 id 规则（含手动 `oss lifecycle set --days N` 调整过的）绝不覆盖。凭证缺少 `PutBucketLifecycle` 权限时降级为 warning，不阻塞上传；自定义 `--object-name` 落在前缀外时显式警告"不会自动过期"，不静默假装有兜底。

### 变更

- **OSS 对象默认过期时长从 7 天降到 1 天**：OSS lifecycle 粒度最小 1 天（无对象级 TTL），1 天已是可达最小保留期；`oss lifecycle set` 的 CLI 默认值与上传兜底共用同一组常量。

## [0.17.0] - 2026-07-23

### 新增

- **`project run` 支持按产物续跑**：运行流程改为产物门控的阶段管线；已有转写、润色或纪要产物会直接复用，中断后重跑只补缺失阶段，避免重复调用付费服务。
- **ASR 任务支持服务端接回与墙钟超时**：持久化任务仍在运行时可重新接回轮询；默认等待预算按音频时长计算，超时后给出可恢复提示，不再无限挂起。新增 `--force-asr` 保留显式强制重转写能力。
- **新增只读声纹阈值校准命令**：`voiceprint calibrate` 基于当前声纹库计算本人和冒名分数分布、等错误率阈值及低误纳阈值，为人工调整匹配阈值提供证据。
- **运行结果新增身份一致度信号**：声纹稳定化完成后汇总逐句身份的一致、冲突和模糊数量，避免仅凭流程完成状态掩盖身份质量问题。

### 变更

- 声纹命名与稳定化提前到润色和纪要之前；润色可结合说话人身份判断语境歧义，纪要优先读取纠错稿并使用已确认的人名。
- probe、cluster 与 sample matching 共用内容寻址的逐片段嵌入缓存；稳定化在零重指派时提前收敛，减少重复计算。
- 声纹匹配、重匹配、逐句身份和簇质量阈值统一收口到单一参数模块，保持现有数值不变并明确跨阶段契约。
- 声纹采集在目标人物已有足够样本时增加身份预检；与人物质心明显不一致的新片段进入隔离状态，保留复核但不污染匹配池。

### 修复

- 句子重指派导致的旧声纹样本从物理删除改为 `invalidated` 软失效；数据库记录、音频片段和嵌入均保留，可恢复且不会因复制项目误连全局库而造成不可逆删除。
- 修复重复运行项目时无条件重新提交 DashScope 任务、重复执行润色和纪要，以及固定跑满声纹稳定化轮次造成的额外费用和等待。

## [0.16.0] - 2026-07-16

### 新增

- **声纹采集支持按项目 speaker 精确筛选**：`voiceprint capture` 新增可重复的 `--speaker-id`、逗号形式 `--speaker-ids`、`--only-needed` 与 `--min-samples`；可只给指定或样本不足的人采集，未传筛选参数时保持原有全场采集行为。
- **声纹采集新增结构化预览与结果输出**：支持 `--dry-run --json`，逐 speaker 返回现有样本数、capture/skip/fail 决策及原因；实际执行还返回稳定 sample ID、clip、person、speaker 与 embedding 状态。
- **新增指定 speaker 的声纹学习闭环**：`project speakers learn` 可对确认身份的 speaker 执行定向采集、仅为新增样本生成 embedding、重新匹配并输出前后分数；只有身份和阈值校验通过才安全 apply，否则返回 `needs_review` 和非零退出码。
- **声纹样本支持稳定 ID 删除**：新增 `voiceprint delete-sample --sample-id vps-...`，兼容 `--keep-clip`，调用方无需再依赖删除后会变化的列表 index。
- **Web 项目主流程补齐**：项目页新增搜索与状态筛选、待复核标记、下一步指引、多输入 Run、生成纪要、项目合并预览与执行、标题编辑，以及最终产物的页内预览和下载。
- **Web 全局任务中心与复核能力增强**：长任务支持跨页面查看、SSE 重挂、幂等去重、取消及排队原因说明；speaker review 新增时间轴、新建 speaker、独立重匹配、Top 3 身份候选与项目深链；纠错页新增字符级 diff、批量选择及 accept/discard 完整生命周期；词库页新增详情编辑、歧义管理、停用恢复和永久删除；声纹库补上人物合并入口。

### 变更

- Web 声纹采集统一接入任务进度与失败恢复流程；Modal 补齐焦点圈闭、焦点归还、叠层管理和 IME 防误提交，危险操作默认聚焦取消。
- Web 导航、错误反馈、token 失效重认证、音频 seek、响应式布局和任务终态恢复统一增强；服务重启或任务消失时不再无限等待。
- Lexicon upsert 的可选字段改为“未提供即保留原值”，添加别名或恢复词条时不再覆盖既有类别、描述和停用状态。

### 修复

- **保证声纹采集写入一致性**：直接多 speaker 采集按 speaker 独立原子提交，交互 review 批次整体原子提交；speaker 无效、切片或数据库写入失败时恢复原 clip，不遗留半套 sample。
- 修复确定性 clip 路径覆盖与重复音频 hash 交叉时，文件、数据库元数据和 embedding 可能不一致的问题；批次落库按最终状态去重，并清除失效 embedding。
- 修复 Web speaker review 在导航、跨项目定位、重匹配和弹窗快捷键下可能静默丢失暂存修改，以及声纹采集结果被动关闭时可能意外回滚的问题。
- 修复任务取消后立即重跑仍挂接旧任务、迟到快照把终态翻回运行中、损坏的单个 speaker 匹配文件拖垮整个项目列表等稳定性问题。
- 修复纠错提案应用后仍重复出现、放弃动作缺失，以及 Web 添加词库别名时可能清空词条元数据的问题。
- 修复完整 `uv build` 在从 sdist 构建 wheel 时重复收录 Web SPA 文件、导致发布构建失败的问题；生成的静态资源改用 Hatchling build artifact 规则按自然包路径收录。

## [0.15.0] - 2026-07-02

### 变更

- **声纹匹配打分改用更稳健的项目级质心**：候选打分优先用同一 person 在单个项目内的声纹质心（要求该项目至少 2 个样本），只有在项目质心分高于全局人物质心时才采用，避免个别项目里的偏差样本拖累跨项目匹配；`speaker_matches.json` 新增 `score_source`（`project-centroid` / `person-centroid`）标注打分来源。
- **声纹探针样本改为质量评分挑选，不再只挑最长句子**：新增 `voiceprint_segment_selection` 模块，按文本质量打分并在说话人时间轴上均匀取样（避免扎堆在同一段长独白），质量不足时回退到旧的“最长句子”策略保底。
- **below-threshold 候选新增强 margin 兜底接受**：候选分数虽低于常规阈值，但明显显著领先第二名（`score >= 0.65` 且领先 `>= 0.25`）时视为安全接受，减少因阈值一刀切造成的漏识别；接受原因（`threshold` / `strong-margin`）记入 `accept_reason` 供审计。
- **CLI 输出暴露声纹匹配诊断信息**：`project speakers match` 等命令的匹配行现在附带 `reason=` / `margin=` / `source=` 诊断后缀，便于排查为什么某个 speaker 被判定匹配或未匹配。

## [0.14.0] - 2026-06-29

### 变更

- **Web UI 视觉整体重塑为「录音棚控制台」暗色风格**：把原本扁平、通用的手写暗色皮肤升级成有层次、有签名的设计语言——真正的 ink 色阶 + elevation 阴影让表格 / 面板 / 卡片成为独立悬浮层，信号蓝 accent + topbar 信号线 + 均衡器 wordmark 形成品牌识别，状态点改为会发光的「信号点」，时间码 / 句号 / 分数统一用等宽字体作为「仪表数据」，并补齐过渡动效、键盘焦点环（可访问性）和按压反馈。全部落在共享 token 与基础组件层，所有页面自动继承；class 名与结构尺寸不变，颜色语义（绿=匹配 / 黄=存疑 / 红=冲突 / 品红=串场 / 蓝=活动）一律保留。
- **声纹质量修复工作流改进**：`verified-active`（确认身份）现在表示「确认这是本人且**保留在匹配中**」，但不再无条件标为可信、也不掩盖差的嵌入质量——一个分数很低的已确认样本仍会按质心被标为存疑 / 严重（reason 前缀「身份已确认」），与「停用」（移出匹配）形成明确区分；声纹库新增「确认但排除」等更细的状态与质量标签。
- **Web 删除 / 丢弃确认改用应用内样式化确认框**：删除样本 / 人物、删词条、语言切换丢弃未保存改动等确认，不再弹出浏览器原生 `window.confirm`（露出 `127.0.0.1 显示` 之类、与暗色 UI 脱节），改为复用应用内 Modal 的统一确认框，销毁性操作用红色按钮。
- Web speaker review 默认布局更紧凑；逐句身份诊断改为悬停时在独立浮窗展示，不再与正文重叠；人物列表与逐句分数措辞统一；Web capture review 控件与 TUI 对齐。

### 新增

- **Web speaker review 支持逐句文字编辑**：可直接在网页里修正单句转写文本，复用既有暂存 / 保存链路。
- **Web speaker 支持合并与空人物删除**，以及从自由命名的 speaker 身份直接创建声纹人物。
- **项目级 sentence locator**：新增按项目作用域的句子定位控件，便于精确跳转 / 引用具体句子。
- Web review 现在逐句解释身份诊断依据（身份分数 + 疑似错人 / 身份接近 / 低于阈值等原因），默认隐藏、悬停展开。

### 修复

- 修复项目列表 ID / 会议时间 / 产物列在连字符处难看断行（`corrected-srt` 断成 `corrected- / srt`、`p-xxxx` 断成 `p- / xxxx`）：原子值整行不断，多项列表只在项之间断行。
- 修复 Web UI 重塑引入的键盘焦点回归：全局 `:focus-visible` 兜底可见轮廓，避免部分控件 Tab 后无焦点指示。
- 修复声纹复核播放进度条布局，以及 speaker review 悬停诊断重叠。

## [0.13.2] - 2026-06-24

### 变更

- Web 声纹库合并“库”和“质量”两个入口：同一页展示人物、样本、质量分、状态、播放和文本；新增按质量问题、姓名拼音、样本数排序，以及按全部、有问题、已停用、可信、未嵌入筛选。默认只读，进入编辑模式后才显示新建人物、改名、删除人物、修改样本状态和删除样本，降低误操作风险。
- Web 声纹样本状态文案改为面向使用语义：`active` 显示为“参与匹配”，`verified-active` 显示为“可信样本”，`quarantined` 显示为“已停用”，避免“活跃/确认/隔离”语义混淆。
- Web speaker review 的逐句身份分数改为显示“身份 0.xx + 疑似错人/身份接近/低于阈值”等原因，人物列表分数改为“匹配 <姓名> 0.xx”，并让逐句“归属”按钮常驻显示，方便直接修正疑似错桶片段。

### 新增

- Web 文字纠错页现在为每条有有效时间窗的建议提供原音频播放按钮和时间点，review polish 提案时可直接听原声判断是否接受改写。

## [0.13.1] - 2026-06-24

### 变更

- Web UI 的 Python 服务依赖现在进入默认安装面：`fastapi`、`uvicorn[standard]`、`sse-starlette`、`python-multipart` 不再挂在 `web` extra 下。正式用户执行 `uv tool install meeting-asr --python 3.14` 后即可运行 `meeting-asr web`，不需要再安装 `meeting-asr[web]`。
- `scripts/install-tool.sh` 不再生成 `.[web]` 安装目标；`--web/--no-web` 仅控制本地 SPA 静态资源是否构建，Python 依赖始终来自默认包依赖。相关文档和缺依赖提示同步改为“重装默认包”，避免继续引导用户使用额外 extra。

## [0.13.0] - 2026-06-23

### 新增

- 新增 `meeting-asr project speakers rerun <project>`：从项目内 `asr/raw_result.json` 重建 speaker 产物，不重新提交 ASR。它会先把 speaker 切分恢复到原始 ASR 输出，再重新跑声纹匹配、可选 crosstalk 标记、speaker stabilization 和 under-split rescue，最后重渲染命名 transcript / SRT。用于声纹库、speaker 稳定化或分桶逻辑更新后，对既有项目做 speaker-only 复算；命令保留 `--store-dir`，方便在项目拷贝和隔离声纹库上验证，避免误动真实库。
- Project Review TUI 新增复制快捷键：`y`、`Ctrl+C`、macOS 终端可传入时的 `Cmd+C` 都会复制当前选区；没有选区时复制当前高亮 sample / 时间轴行，包含时间戳、speaker 标签、当前名称和句子文本。复制优先走 Textual 内置剪贴板，再兜底系统剪贴板命令，便于从 TUI 里直接摘取待核对片段。
- Project Review TUI 新增 `d` 快捷键接受明确的错桶诊断：当前高亮句子若分桶诊断给出 `疑似错桶` 且带有具体 `更像=Speaker X` 目标，可一键移动到建议 speaker，并复用现有 reassignment 保存链路；边界近、身份接近或没有具体目标的样本不会自动移动，仍需按 `r` 手动选择。

### 修复

- 修复 under-split rescue 在未绑定身份的 speaker track 上可能过度抽离的问题：当候选片段虽然略像库内某人、但仍高度贴近当前 track 自身质心时，不再把它 promotion 或塞入 unknown bucket，避免把一个一致的源 speaker 拆碎成伪新 speaker。resplit 审计 payload 现在记录 `source_score`，便于解释这类拒绝原因。

## [0.12.0] - 2026-06-08

### 新增

- 新增 **Web UI（`meeting-asr web`）**：把原本的 Textual TUI 全部功能搬上浏览器。`meeting-asr web` 启动一个本地 FastAPI 服务（默认 `127.0.0.1:8765`，单用户、loopback 免鉴权，非 loopback 强制 bearer token），前端是 React + TypeScript + Vite SPA（构建产物 force-include 进 wheel，发布时 hatch 钩子自动构建，安装无需 node）。覆盖：项目列表与完整 ASR 摄入管线控制台（运行管线 / summarize / merge，经后台任务 + SSE 实时进度）、**speaker review**（双栏、逐句音频播放、改名 / 接受声纹匹配 / 忽略 / 重指派 / 疑点过滤 / 保存）、**声纹采集**（候选 clip 勾选 → 采集+嵌入 → 本项目与历史项目逐发言人分数变化对比 → 接受/回滚）、**声纹库**（浏览 / 质量离群改判 / 人物 CRUD）、**文字纠错**（polish 提案逐条勾选应用）、**纠错词库**（词条/消歧/热词）、**设置**（配置凭证、环境诊断）。架构上**零业务逻辑重写**：危险落盘路径（speaker_map 合并、会删全局声纹样本的重指派 + rematch、声纹采集事务）抽进 `core/speaker_review_service.py` / `core/voiceprint_review_service.py`，CLI 与 web 共用同一条路径；长任务进度复用现有 `emit_progress`，经 `call_soon_threadsafe` 推 SSE；音频经带 HTTP Range 的端点给浏览器 `<audio>`；并发用单 worker + per-project/per-store 异步锁 + 任务串行（不引入 Redis/Celery）。绑定到非 loopback 时除强制 bearer token 外还内置一组部署安全默认：校验 `Host` 头防 DNS rebinding、联网 500 响应体脱敏、clip 路径按库内相对路径隔离防穿越。用法与架构见 `docs/web-ui.md`。
- Web UI **token 模式完整可用**（修复 PR #24 review）：`require_auth` 现在同时接受 `Authorization: Bearer` 头与 `?token=` query 参数（常数时间比较），因为 `EventSource`（SSE 进度）和 `<audio>`（clip 播放）这类浏览器托管请求无法设置请求头，否则在 token 保护的绑定下会全 401。启动时打印并自动打开带 `?token=` 的 URL，SPA 首屏读取后存入 localStorage 并抹掉地址栏；裸 URL / token 失效时弹输入框兜底（探针 `/api/auth/check`）。
- Web UI **声纹采集事务对全局库独占**（修复 PR #24 review）：一次采集会对全局声纹库留一份回滚快照直到 accept/rollback；在它待决期间任何会改库的写（人物增删改/合并、样本状态、再发起采集）都会被拒（HTTP 409），避免一次 rollback 把这期间的编辑静默还原。

### 修复

- 修复 ASR 热词不遵循配置词库的 **store 隔离**漏洞：`project run` / `project transcribe` 指定非默认词库（`--lexicon-db`，以及 web 的 `--store-dir`）时，热词此前仍从默认 XDG 词库读取并快照、忽略用户指定的库，与 local_correction / polish 早已遵守的隔离口径不一致；现在热词与纠错 / polish 走同一个配置词库（CLI 不带 override 时仍回退 XDG），识别偏置与 `corrections/asr_hotwords.json` 记录才真正反映指定库。
- 修复 TUI **Project Review 保存失败时弹窗只显示空白 / 晦涩错误**：共享保存路径会把异常转成 `typer.Exit` 并打到已被 TUI 接管的 console，导致用户看不到失败原因；现在 TUI 弹窗能拿到真实异常消息，CLI 调用边界仍保留本地化错误面板。
- 修复**声纹删除会误删真实库文件**的数据安全问题：删除声纹样本 / 说话人此前按数据库里记录的绝对 `clip_path` 删文件，当 `--store-dir` 指向拷贝库（文档推荐的安全验证流程）时会越界删掉真实库的 clip、破坏隔离；现在删除按配置库的库内相对路径重定位并拒绝越界路径，默认库行为逐字节不变，拷贝库验证场景改为安全结果。

## [0.11.0] - 2026-06-05

### 新增

- `project run` 支持**多输入拼接成单一连续项目**：`meeting-asr project run a.mp4 b.mp4 c.mp4 ...` 把多段媒体先拼成一条连续音频，再在全量上**只跑一次** ASR、diarization 与声纹匹配，所以跨段说话的同一个人天然只有一条 speaker（不必事后归一）。典型场景是钉钉把同一场会拆成两段闪记——这与事后合并各自已转写产物的 `project merge` 是两条正交路径。拼接用 ffmpeg 把每段归一到 16k mono s16 中间件、按 concat demuxer `-c copy` 无损无缝零漂移地接起来；时间轴本就统一连续，无需 offset 数学。项目身份守恒：N=1 与今日逐字节一致（同 `project_id`、同 manifest）；N>1 的 `project_id` 是有序各段 sha 的组合内容哈希，顺序敏感、同序重跑可复用；多段 manifest 故意把 `source.original_path` 置空，使单/多输入在复用时绝不互撞，逐段溯源落 `audio.segments`。`project show` 展示分段来源；`--file-url` 与多输入互斥。
- 新增**声纹低置信 crosstalk/噪音放行档**。会尾常混入另一拨人的零碎串场（样本极少、声纹分数极低、候选对不上），以前这种 cluster 卡在 below-threshold，逼下游瞎猜名或整场绕过。crosstalk 档只给它打一个 **advisory 标记**:speaker 仍是匿名 `Speaker N`、句子一字不改、不移动、不改名，只是不再阻塞主流程、下游可选择放行落地。判据刻意保守且非对称——`sample_count ≤ 3` **且** `0 < best_score < 0.5` **且**候选不集中才标;要求 best_score 大于 0(库里有弱候选但对不上)是关键,空库/选错 model 时绝不把正常 speaker 全标噪音;有清晰弱领先者仍判为「真人只是低于阈值」。标志持久化进 `speaker_matches.json` 的 `crosstalk` 字段,在 run / `speakers match` / `project show` 三处未决门禁里与 matched 同列放行,match 表渲染为 magenta。`project run` 与 `project speakers match` 新增 `--crosstalk/--no-crosstalk`、`--crosstalk-max-samples`、`--crosstalk-score-floor`(默认开),设定落 `manifest.speakers["crosstalk"]`,后续 rematch 不会偷偷把关掉的档重新打开。与 resplit 的 unknown 桶正交可叠加。
- `lexicon show` / `lexicon list` 现在**展示别名消歧状态**(issue #18)。`disambiguate` 把语境指引写进别名(NULL=盲替;非 NULL=排除盲替、由 polish 逐句判别),但此前读侧完全看不见,配完无法复核只能手写 SQL 查库。现在 `lexicon show` 给消歧别名标 `[ambiguous]` 并打印指引全文,`lexicon list` 在 Aliases 列旁标 `(N ambiguous)`,`--json` 一并带 `disambiguation` / `ambiguous_alias_count` 字段。

### 修复

- `project speakers apply --map` 支持**逗号分隔的多重映射**并**拒绝垃圾名**。此前 `--map '0=武一,2=欧丁,3=墨泪'` 会把整串糊进 speaker 0 的名字、静默丢掉后两条;现在单个 `--map` 值按 `,` 拆分(等价于重复 `--map`),且含 `,` 或 `=` 的名字会被明确报错而非写进 `speaker_map.json`。另修:整体为空的 `--map` 值不再被静默跳过而是报错;`--variant` 在按 `--project-dir` 复用多输入项目时被正确尊重。
- `lexicon` 消歧字段的 export/import **往返不再静默丢失**:此前导出再导入会把所有消歧别名降级回盲替。现在往返保留指引,且 import 能区分「缺消歧键」与「显式 null」、blanket 来源会清掉过期消歧标记。
- crosstalk-only(全员判串场)的非阻塞运行现在**补渲染匿名命名产物**:`accepted_mapping` 为空、`apply` 被跳过时,stabilization 后仍补出匿名 `Speaker N` 的 `transcript_named.txt` / `subtitle_named.srt`,让 run summary 报的「ready」名副其实;`--no-crosstalk` 经一次 rematch 重指派后不再失效,crosstalk 在声纹 CLI 行也显式渲染(不再 fall through 成误导性的 `no-candidate`)。

## [0.10.0] - 2026-06-04

### 新增

- 新增 ASR under-split（欠分割）救援。DashScope diarizer 有时把多个真人塌进同一条 speaker track（典型如整场会只切出两条 track，第三人混在其中只露几句）；此前的说话人稳定化只能在**已存在的项目 speaker 之间**挪句子，所以“声纹库里有、但本项目还没建 track 的人”永远救不出来。新增 `meeting-asr project speakers resplit <project>`：把拥挤 track 的句子按**逐句声纹身份**重新聚类——把“确属声纹库内另一个人”的句子组**提升为独立 speaker track**（自动分配新 id 并按库内权威名命名），把“不匹配任何库内人”的离群簇收进 **review 可见的 unknown 桶**交人工确认。为避免误判：聚类锚定干净的库向量而非被污染的 track 质心；promotion 要强正证据（质心贴某库人且明显领先当前指派人）、residue 要整簇去噪后仍谁都不像；且**覆盖 track 半数以上的主簇绝不被抽离**（防止把干净单人 track 整条搬走）。默认 dry-run 预览，零写盘（探针音频与嵌入缓存重定向到临时目录，不碰项目文件）；`--apply` 落地；在项目拷贝上用 `--store-dir` 隔离声纹库，避免误删真实库样本。
- `project run` / `project rerun` 的说话人稳定化阶段现在**自动跑 under-split 救援**（迭代前一次性前置，well-split 项目下为 no-op），`--no-speaker-resplit` 关闭；即便本次声纹聚合匹配整体未达阈值，高置信 promotion 仍会跑（只跳过需要项目内锚点的迭代轮）。

## [0.9.0] - 2026-05-31

### 新增

- 新增 `meeting-asr project merge <p1> <p2> ...`：把同一场会被钉钉拆成多段闪记（各自一个 project）的转写合并成单一转写包，原生支持中场休息分段的场景。按 `meeting_time` 时间序拼接，跨段**按声纹人 public id（`vpp`）归一发言人**——同一个人在不同段即使本地 speaker_id 不同、甚至某段没命名，也会对齐成同一发言人并取声纹库权威名；仅命名未连声纹的发言人默认按同名提升对齐到声纹人（`--no-name-to-vpp` 关闭）。时间轴连续打包（各段按音频时长偏移、单调不重叠），段界 header 保留各段原始会议时间/时长/句数。产出 `transcript_merged.txt` / `_corrected.txt`、`subtitle_merged.srt` / `_corrected.srt` 和结构化只读清单 `merge.json`（含段元信息与发言人归一审计轨）。单段退化为直接导出；合并为无状态操作，绝不回写原 project。
- `project run` / `project rerun` / `project transcribe` 在 ASR 提交后把本次随 DashScope 任务一起提交的热词表写入项目 `corrections/asr_hotwords.json`（含 `dashscope_vocabulary` 与逐条 hotword）。此前该文件只有 `project correct` 流程写，新转写完的项目看到的是空文件，让人误以为没给识别引擎喂热词——其实 lexicon vocabulary 一直在随任务提交。现在每次转写都会落地“本次实际提交了哪些专名”，便于核对 iSee / CLI / SKU 等热词是否生效。文件在下游产物失效（invalidation）之后写，避免被重跑清空；`--asr-hotwords off` 时记为空表。
- 新增 `meeting-asr lexicon disambiguate <term> <alias> <guidance>`：把同音歧义的 ASCII 别名（典型如 `IC`——既可能是平台 iSee、也可能是个人贡献者角色）标成「按语境判别」。被标记的别名从确定性盲替规则中剔除，改由 polish 阶段的 LLM 按本句语境逐句判断；判别依据（guidance 文本）作为业务知识只存在 lexicon 配置里，不进代码库。
- 新增 `evals/restore_eval.py`：「语义异常检测 + 语境还原」的模型能力评测。不喂 wrong_text→right 映射，只给 lexicon 权威词库 + 语境签名，量化模型「检测 ASR 误识别专名 → 音近+语境还原、且不过度还原正常词」的召回/精确，并把切片长度（chunk）与重叠（overlap）作为变量探注意力涣散拐点。gold 从本机真实项目 `raw→corrected` diff 动态提取、记分牌落 `evals/local/`（均不进库），换模型时重跑对比能力漂移。

### 变更

- transcript polish 严格模式新增「术语消歧」prompt 区块，由 lexicon 的歧义别名 guidance 动态驱动，让 LLM 按语境决定是否纠正（而不是无脑替换）。同时移除了 strict polish prompt 里硬编码的 `把IC→把 iSee` 标尺样例：业务专名属于配置，不应写死在代码里；该样例也是盲替，会把「个人贡献者 IC」误改成平台名。背景：受控 A/B 实验证明 DashScope fun-asr 的自定义热词 vocabulary 对 `iSee` 完全不生效（无 vocab / 带 vocab / 带 `lang=en` 输出逐字一致），ASR 提交侧治不了同音错词，只能在 polish 阶段按语境修。

### 修复

- 确定性 lexicon 纠错此前对 `asr_error` 别名做无条件全替换，会把歧义词（如指「个人贡献者」的 `IC`）也错改成平台名 `iSee`。现在带 disambiguation 标记的别名跳过盲替，避免误伤。
- transcript polish 的确定性 guard 此前会误拒大量合法的去口癖（de-stutter）润色：相邻重复折叠（如 `可以，可以`→`可以`）或仅调整空格的纯去口癖改写，会被判成删除保护词而退回，拉低 polish 采纳率。现在 guard 先放行纯去口癖改写、保护词计数改在去口癖后的文本上比对、并豁免相邻重复保护词的去重；ASCII guard 也改为词表感知（已验证的专名不再被当成乱码拦下）。受控统计上被采纳的润色从 391 条恢复到 518 条，并补了确定性测试锁定该行为。

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

[未发布]: https://github.com/crhan/meeting-asr/compare/v0.12.0...HEAD
[0.12.0]: https://github.com/crhan/meeting-asr/compare/v0.11.0...v0.12.0
[0.11.0]: https://github.com/crhan/meeting-asr/compare/v0.10.0...v0.11.0
[0.10.0]: https://github.com/crhan/meeting-asr/compare/v0.9.0...v0.10.0
[0.9.0]: https://github.com/crhan/meeting-asr/compare/v0.8.0...v0.9.0
[0.8.0]: https://github.com/crhan/meeting-asr/compare/v0.7.0...v0.8.0
[0.7.0]: https://github.com/crhan/meeting-asr/compare/v0.6.2...v0.7.0
[0.6.2]: https://github.com/crhan/meeting-asr/compare/v0.6.1...v0.6.2
[0.6.1]: https://github.com/crhan/meeting-asr/compare/v0.6.0...v0.6.1
[0.6.0]: https://github.com/crhan/meeting-asr/compare/v0.5.0...v0.6.0
[0.5.0]: https://github.com/crhan/meeting-asr/compare/v0.4.0...v0.5.0
[0.4.0]: https://github.com/crhan/meeting-asr/compare/v0.3.0...v0.4.0
[0.3.0]: https://github.com/crhan/meeting-asr/compare/v0.2.0...v0.3.0
[0.2.0]: https://github.com/crhan/meeting-asr/compare/v0.1.0...v0.2.0
[0.1.0]: https://github.com/crhan/meeting-asr/releases/tag/v0.1.0
