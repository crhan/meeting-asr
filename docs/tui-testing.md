# TUI 测试

Meeting-ASR 的 TUI 用 Textual headless 测试，重点测行为和状态，不做默认截图快照。

## 规则

- 用 `App.run_test()` 启动 TUI。
- 用 `Pilot.press()` 驱动键盘流程。
- 用 `Pilot.resize_terminal()` 测响应式布局。
- 断言应用状态和关键 rendered markup。
- 不依赖真实播放器、编辑器或外部网络；这些能力用 monkeypatch 隔离。

## 测试分层

- `tests/test_speaker_tui_status.py`：纯状态渲染、conflict/mismatch 规则。
- `tests/test_speaker_tui.py`：Project Review TUI 的浏览、编辑、保存、播放目标、列导航、分页、resize、声纹入口。
- `tests/test_voiceprint_review_tui.py`：Voiceprint Review 的项目候选、全局库、采样选择、颜色语义、退出行为。
- `tests/test_voiceprint_review_workflow.py`：声纹采样、embedding、评测、回滚事务。

## 常用命令

```bash
uv run pytest tests/test_speaker_tui.py tests/test_speaker_tui_status.py -q
uv run pytest tests/test_voiceprint_review_tui.py tests/test_voiceprint_review_workflow.py -q
uv run pytest -q
```

如果以后布局复杂到需要视觉回归，再引入 `pytest-textual-snapshot`。快照更新必须人工 review，不能无脑覆盖 baseline。
