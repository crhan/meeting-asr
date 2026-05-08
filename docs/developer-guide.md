# Meeting-ASR 开发者指南

## 开发环境

```bash
uv venv
uv sync --all-groups
uv run meeting-asr --help
uv run pytest -q
```

代码使用 `src` 布局，包入口是 `src/app`。本地开发和验证必须用 `uv run ...`，不要用全局 `meeting-asr` 验证刚改的代码。

## 全局安装

全局命令用独立脚本刷新，不做成业务 CLI 子命令：

```bash
scripts/install-tool.sh
scripts/install-tool.sh --check
```

脚本默认执行：

```bash
uv tool install --python 3.14 --editable .[local-voiceprint]
```

关键原因：

- `uv tool install` 可以使用 pyenv 的 Python，但必须显式指定 `--python 3.14`，否则可能选到不满足 `Python>=3.14` 的解释器。
- 本地开发默认 editable，源码修改直接生效。
- `scripts/install-tool.sh --wheel` 只用于发布验证或模拟正式用户安装。
- wheel 模式依赖 `tool.uv.cache-keys` 跟踪 `src/**/*.py`，避免复用旧 wheel。
- 安装后脚本会验证 wrapper、Python、源码路径和源码指纹；不一致直接失败。
- 只有已有非 uv 可执行文件冲突时才用 `scripts/install-tool.sh --force`。

## 验证

```bash
uv run ruff check src tests
uv run pytest -q
uv run python -m compileall -q src tests
git diff --check
```

改 completion 时额外跑：

```bash
uv run meeting-asr completion zsh >/tmp/meeting-asr.zsh
zsh -n /tmp/meeting-asr.zsh
uv run meeting-asr completion bash >/tmp/meeting-asr.bash
bash -n /tmp/meeting-asr.bash
env _MEETING_ASR_COMPLETE=complete_bash \
  COMP_WORDS='meeting-asr project transcribe --' \
  COMP_CWORD=3 \
  uv run meeting-asr
```

根 CLI 关闭了 Typer 的 `add_completion`，所以启动时必须调用 `completion_init()`。否则 completion 脚本可能存在，但运行时会出现 `plain,xxx` 前缀或 shell 指令顺序错误。

## 设计边界

- 公开转写入口只允许 `meeting-asr project ...`。
- `project create` 复制源视频到项目目录；后续命令只依赖 Project ID 或项目目录。
- 全局配置、项目库、声纹库遵循 XDG Base Directory。
- API key、secret、signed URL 不写日志、不写仓库、不写 `project.json`。
- 新 UI 放 `presentation/cli` 或 `presentation/tui`；不要继续膨胀 `commands/project.py`。
