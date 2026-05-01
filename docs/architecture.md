# Meeting-ASR Architecture

Meeting-ASR now uses a layered layout. New code should pick the target layer first, then import only inward or sideways through stable adapters.

## Layers

```text
src/app/
  core/              # domain state, workflow events, runtime baselines
  infra/             # external systems: DashScope, ffmpeg/ffprobe, storage adapters
  presentation/
    cli/             # Typer/Rich output, CLI progress and error rendering
    tui/             # Textual screens and TUI-specific view state
  commands/          # current Typer command adapters; keep thin
```

## Dependency Rules

- `core/` must not import `presentation/`.
- `infra/` must not import `presentation/`.
- CLI/TUI code may import `core/` and `infra/`, then render the result.
- Compatibility wrappers such as `app.cli_ui`, `app.asr_client`, and `app.speaker_tui` exist only to avoid breaking existing imports. New code should use the layered paths directly.
- `project_manager.py` is still a transitional workflow module. Do not add new UI or external-service details there; extract new logic into `core/`, `infra/`, or `presentation/` first.

## Current Split

- `core/asr_metrics.py`: SQLite observations and precomputed ASR ETA baselines.
- `core/asr_wait.py`: provider-scoped ETA estimate/record helpers and progress event generation.
- `core/oss_metrics.py`: SQLite upload throughput observations and precomputed OSS ETA baselines.
- `core/oss_upload.py`: provider-scoped OSS upload ETA estimate/record helpers and progress events.
- `core/progress.py`: presentation-neutral progress event dataclasses.
- `infra/dashscope_asr.py`: DashScope ASR submit/fetch/download adapter.
- `infra/ffmpeg.py`: ffmpeg/ffprobe media adapter.
- `presentation/cli/progress.py`: Rich progress renderer.
- `presentation/cli/errors.py`: CLI error rendering and doctor guidance.
- `presentation/tui/*`: Textual project, speaker, and voiceprint screens.

## Migration Direction

1. Keep command modules as thin Typer adapters.
2. Move project lifecycle and transcription workflow code out of `project_manager.py` gradually.
3. Replace UI-flavored return strings in core workflows with typed result/event objects.
4. Keep old import wrappers until all internal imports and tests use layered paths.
