# Meeting-ASR 开发者指南

## 开发环境

```bash
uv venv
uv sync --all-groups
uv run meeting-asr --help
uv run pytest -q
```

代码使用 `src` 布局，包入口是 `src/app`。

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
```

## 设计边界

- 公开转写入口只允许 `meeting-asr project ...`。
- `project create` 复制源视频到项目目录，后续命令只依赖项目目录。
- 全局配置和默认项目目录遵循 XDG Base Directory。
- 不把 API key、secret、signed URL 写入日志或仓库。
- signed URL 只短暂传给 DashScope，不写入 `project.json`。
