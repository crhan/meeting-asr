# Meeting-ASR 开发者指南

## 开发环境

```bash
uv venv
uv sync --all-groups
uv run meeting-asr --help
uv run pytest -q
```

代码使用 `src` 布局，包入口是 `src/app`。

## 全局安装

全局命令用独立脚本刷新，不做成 `meeting-asr` 子命令：

```bash
scripts/install-tool.sh
scripts/install-tool.sh --check
```

这个脚本显式执行 `uv tool install --python 3.14`。
原因：

- `uv tool install` 可以使用 pyenv 提供的 Python，但要通过 `--python 3.14`
  或 `UV_PYTHON=$(pyenv which python3.14)` 明确指定。
- 不指定 `--python` 时，uv tool 的默认解释器可能落到 uv managed Python 3.13，
  与本项目 `Python>=3.14` 冲突。
- uv 对本地目录的默认缓存只跟踪 `pyproject.toml` / `setup.py` / `setup.cfg`，
  不会因为普通源码文件变化自动重建 wheel。
- 项目通过 `tool.uv.cache-keys` 显式跟踪 `src/**/*.py`，源码变化会触发重建。
- 安装后会比对当前 checkout 和实际 `site-packages/app` 的源码指纹；不一致就失败。
- 只有已有非 uv 可执行文件冲突时才传 `scripts/install-tool.sh --force`。
- completion 只能把 `~/.local/bin` 这类用户命令目录加入 PATH，不能把
  `~/.local/share/uv/tools/meeting-asr/bin` 加进去；后者会泄漏 tool 私有
  `python/python3` 到用户 shell。

## 验证

```bash
uv run pytest -q
uv run python -m compileall src tests
git diff --check
```

如果改了 completion：

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

注意：根 CLI 关闭了 Typer 的 `add_completion`，所以必须在启动时调用
`completion_init()`。否则生成的 completion 脚本看似存在，但运行时可能出现
`plain,xxx` 前缀或 shell 指令顺序不匹配。

## 设计边界

- 公开转写入口只允许 `meeting-asr project ...`。
- `project create` 复制源视频到项目目录，后续命令只依赖项目目录。
- 全局配置和默认项目目录遵循 XDG Base Directory。
- 不把 API key、secret、signed URL 写入日志或仓库。
- signed URL 只短暂传给 DashScope，不写入 `project.json`。
