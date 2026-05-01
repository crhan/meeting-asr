# TUI Testing

Speaker review TUI tests follow Textual's official headless testing model:

- Use `App.run_test()` to run the app without opening a terminal UI.
- Use `Pilot.press()` for keyboard flows.
- Use `Pilot.resize_terminal()` for responsive layout behavior.
- Assert application state and rendered markup after each interaction.

Current test layers:

- `tests/test_speaker_tui_status.py`: pure status rendering and conflict/mismatch rules.
- `tests/test_speaker_tui.py`: Textual Pilot flows for browse/edit/save, playback targeting, column navigation,
  pagination, resize behavior, and project session loading.

Run focused TUI tests:

```bash
uv run pytest tests/test_speaker_tui.py tests/test_speaker_tui_status.py -q
```

Run the full suite before committing:

```bash
uv run pytest -q
```

Snapshot testing is intentionally not enabled by default. The current risk is behavior and workflow state, so headless
semantic tests are more stable than pixel snapshots. If layout styling becomes complex enough to justify visual
regression tests, add `pytest-textual-snapshot` as a dev dependency and keep snapshots reviewed manually before updating
baselines.
