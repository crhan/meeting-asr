# Web UI

`meeting-asr web` 启动一个本地 Web 服务，把原本的 Textual TUI 全部功能搬到浏览器：项目管理、
speaker review、声纹库（采集 / 浏览 / 质量）、文字纠错、纠错词库，以及完整的 ASR 摄入管线控制台。

## 快速开始

```bash
# 安装（默认带 Web UI Python 依赖；脚本会构建前端）
scripts/install-tool.sh
# 或在 checkout 里直接跑
uv run meeting-asr web --port 8765
```

浏览器会自动打开 `http://127.0.0.1:8765/`。

### 常用参数

| 参数 | 说明 |
| --- | --- |
| `--host` | 绑定地址，默认 `127.0.0.1`（仅本机）。非 loopback 会**强制要求 token**。 |
| `--port` | 端口，默认 `8765`。被占用时会在启动前给出明确提示。 |
| `--projects-dir` | 项目父目录，默认 XDG 数据目录。 |
| `--store-dir` | 声纹/词库 store 目录。**实验时指向一份拷贝**以保护真实声纹库。 |
| `--token` | 非 loopback 绑定所需的 bearer token（不填则自动生成并打印一次）。 |
| `--no-open` | 不自动打开浏览器。 |

## 功能地图

- **Projects** `/projects`：项目列表 + 状态徽章；「运行管线」从一个服务器端媒体路径发起完整转写
  （create → ASR → summarize → match），SSE 实时进度。点项目进入 speaker review。
- **Speaker review** `/projects/:ref/speakers`：双栏（发言人 / 逐句转写），逐句音频播放，改名 /
  接受声纹匹配 / 忽略 / 重指派，疑点/低分过滤，保存。头部可跳「采集声纹」「文字纠错」。
- **采集声纹** `/projects/:ref/capture`：候选 clip 勾选 → 采集+嵌入（后台 job）→ 结果对比
  （本项目与历史项目逐发言人分数变化）→ 接受 / 回滚。
- **Voiceprints** `/voiceprints`：库浏览（人物 + 样本 + 音频）、质量（离群样本 + 状态改判）、
  人物 CRUD。
- **文字纠错** `/projects/:ref/corrections`：生成 polish 提案（LLM job）→ 逐条勾选 diff → 应用。
- **Lexicon** `/lexicon`：纠错词条增删搜、消歧、ASR 热词。
- **Settings** `/settings`：配置（DashScope/OSS 凭证，密钥默认脱敏）、环境诊断（等同 `doctor`）。

## 架构

```
浏览器 SPA (React+TS+Vite) ──HTTP/JSON + SSE──▶ FastAPI（单 worker）
                                                  ├─ routers/*       薄 HTTP 适配
                                                  ├─ JobManager      run_in_executor 跑阻塞业务，
                                                  │                  per-project 串行，SSE 扇出
                                                  ├─ progress_bridge CliProgressEvent → SSE
                                                  │                  （call_soon_threadsafe）
                                                  ├─ LockRegistry    per-project + per-store 锁
                                                  └─ core/*_service  CLI 与 web 共用的中立业务编排
                                                        └─ 复用 project_manager / speaker_matching /
                                                           voiceprints / *_store 等现有纯业务入口
```

设计要点：

- **零业务逻辑重写**：所有领域逻辑复用现有纯函数。speaker review 的危险落盘路径（speaker_map
  合并、会删全局声纹样本的重指派 + rematch）抽进 `core/speaker_review_service.py`，CLI 和 web 共用
  同一条路径；声纹采集事务抽进 `core/voiceprint_review_service.py`。
- **进度复用**：所有长任务本就走 `emit_progress(reporter, CliProgressEvent)`，web 接一个把事件经
  `call_soon_threadsafe` 推到 SSE 的 reporter（emit 在 executor 线程，必须 threadsafe）。
- **音频走 HTTP**：clip 用 ffmpeg 切成 WAV 落盘，经带 HTTP Range 的端点给浏览器 `<audio>`，可 seek。
- **并发**：单 worker + per-project 与 per-store 的 `asyncio` 锁（排序取锁防死锁）+ 任务串行；
  不引入 Redis/Celery。
- **声纹采集事务独占**：一次采集会对**全局声纹库**留一份回滚快照，直到用户 accept/rollback。
  在它待决期间，任何会改库的写（人物增删改/合并、样本状态、再发起采集）都会被拒（HTTP 409），
  否则一次 rollback 会把这期间的编辑静默还原。前端据此引导用户先接受或回滚采集结果。

## 安全

单用户本地工具：loopback 绑定免鉴权。`--host` 指向非 loopback 时强制 token（自动生成或
`--token` 指定）。

注意：「loopback 免鉴权」的边界是**主机**而不是用户——在多用户共享主机（跳板机、共用开发机）上，
同机的其他本地用户也能访问 `127.0.0.1`，等于能读你的项目与配置。这种环境请显式加 `--token`。

**Token 交接**：启动时控制台会打印一条带 `?token=` 的 URL（浏览器也会自动用它打开）。SPA 首屏
读取 `?token=`、存入 localStorage、并从地址栏抹掉。之后：

- 普通 `fetch` 调用带 `Authorization: Bearer <token>` 头；
- `EventSource`（SSE 进度）和 `<audio>`（clip 播放）**无法设置请求头**，改用 `?token=` query 参数
  携带——后端 `require_auth` 同时接受 header 和 query 两种凭证（常数时间比较）。
- 没带 token 打开裸 URL 时，前端弹出 token 输入框作为兜底（探针 `/api/auth/check`）。

权衡：query-param token 可能出现在访问日志里；对单用户 LAN 工具这是可接受的取舍，未引入
cookie/CSRF 面。仍不强制 HTTPS。

## 开发

```bash
# 后端
uv run meeting-asr web --port 8765
# 前端（dev server，代理 /api 到 8765）
cd web && npm install && npm run dev   # http://localhost:5173
# 构建（产物落 src/app/web/static/，被 wheel 的 build-artifact 规则收录）
cd web && npm run build
```

发布 / wheel 安装验证时由 `MEETING_ASR_BUILD_WEB=1` 触发（CI、`install-tool.sh --wheel` 都会设置）：
`hatch_build.py` 在 wheel 构建阶段**无条件重建** SPA（绝不信任可能过期的旧 `static/`），缺 npm 或
构建后仍无产物会直接报错，不会静默发出没有 UI 的 wheel。不设该变量则是 base CLI 构建路径：
不碰 npm，有现成 `static/` 就带上，没有就发一个无 Web UI 的合法 wheel。
