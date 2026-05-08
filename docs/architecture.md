# Meeting-ASR 架构说明

目标：业务逻辑和展示层分开，CLI/TUI 各自只负责交互和渲染。

## 分层

```text
src/app/
  core/              # 领域状态、工作流事件、运行时 baseline
  infra/             # 外部系统适配：DashScope、ffmpeg、OSS 等
  presentation/
    cli/             # Typer/Rich 输出、help、进度、错误渲染
    tui/             # Textual screen、TUI view state、键盘交互
  commands/          # Typer 命令适配层，应该保持薄
```

## 依赖规则

- `core/` 不 import `presentation/`。
- `infra/` 不 import `presentation/`。
- CLI/TUI 可以 import `core/`、`infra/`，再把结果渲染给用户。
- 新 UI 不放进 `commands/project.py`。
- 旧 wrapper 例如 `app.cli_ui`、`app.asr_client`、`app.speaker_tui` 只为兼容旧 import；新代码直接使用分层路径。

## 当前主要模块

- `core/asr_wait.py`：DashScope ASR 等待 ETA 和进度事件。
- `core/oss_upload.py`：OSS 上传 ETA 和进度事件。
- `core/progress.py`：展示无关的进度事件模型。
- `infra/dashscope_asr.py`：DashScope ASR submit/fetch/download。
- `infra/ffmpeg.py`：ffmpeg/ffprobe 适配。
- `presentation/cli/help.py`：Meeting-ASR 自己的 i18n help renderer。
- `presentation/cli/project_list.py`：project list 的 Rich/plain 输出。
- `presentation/cli/errors.py`：CLI 错误和 doctor 引导。
- `presentation/tui/`：project、speaker、voiceprint、correction 相关 Textual UI。

## 迁移方向

1. 命令模块只解析参数和调用用例。
2. 继续把 `project_manager.py` 里的工作流逻辑拆到 `core/` / `infra/`。
3. core workflow 返回 typed result/event，不返回 UI 文案。
4. 兼容 wrapper 先保留，等内部 import 和测试全部迁完再删。
