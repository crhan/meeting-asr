"""Serve transcript audio clips over HTTP for the browser ``<audio>`` element.

The TUI shells out to mpv/afplay; the browser cannot. Instead we extract the same WAV
clip (``extract_audio_clip``) into a per-project cache and serve it. Starlette's
``FileResponse`` handles HTTP Range requests, so the player can seek. Clips are keyed by
time range and the project audio is immutable for a given transcript, so cached clips are
safe to reuse.
"""

from __future__ import annotations

import os
import tempfile
from pathlib import Path

from fastapi import APIRouter, Depends, Query
from fastapi.responses import FileResponse

from app.infra.ffmpeg import extract_audio_clip
from app.project_manager import load_manifest, project_paths, resolve_project_audio_path
from app.web.deps import get_settings, require_auth, resolve_web_project_ref
from app.web.settings import WebSettings

router = APIRouter(
    prefix="/api/projects", tags=["audio"], dependencies=[Depends(require_auth)]
)

# Small lead-in and tail padding so a clip is comfortable to listen to (mirrors the TUI).
_LEAD_IN_SECONDS = 0.25
_TAIL_PAD_SECONDS = 0.5


@router.get("/{project_ref}/clip")
def get_clip(
    project_ref: str,
    begin_ms: int = Query(..., ge=0),
    end_ms: int = Query(..., ge=0),
    settings: WebSettings = Depends(get_settings),
) -> FileResponse:
    """Extract (or reuse a cached) WAV clip for one transcript time range."""
    if end_ms <= begin_ms:
        raise ValueError("end_ms must be greater than begin_ms.")
    project_dir = resolve_web_project_ref(project_ref, settings)
    manifest = load_manifest(project_dir)
    source = resolve_project_audio_path(project_dir, manifest)

    cache_dir = project_paths(project_dir).root / "tmp" / "web_clips"
    cache_path = cache_dir / f"{begin_ms}_{end_ms}.wav"
    if not cache_path.is_file():
        start = max(0.0, begin_ms / 1000.0 - _LEAD_IN_SECONDS)
        duration = (end_ms - begin_ms) / 1000.0 + _LEAD_IN_SECONDS + _TAIL_PAD_SECONDS
        # Extract to a unique temp file then atomically rename into place. Sync routes run
        # in a thread pool, so two requests for the same uncached clip would otherwise both
        # run ffmpeg over the same path -- and worse, is_file() turns True the instant ffmpeg
        # *creates* the file, so a concurrent request could serve a half-written WAV. An
        # atomic os.replace makes cache_path appear only when fully written; concurrent
        # extractions each write their own temp and the last rename wins, no corruption.
        cache_dir.mkdir(parents=True, exist_ok=True)
        # The suffix MUST end in .wav: extract_audio_clip does not force -f wav, so ffmpeg
        # infers the output container from the final extension. A ".tmp" tail makes it fail
        # with "Unable to choose an output format", breaking every uncached clip.
        fd, tmp_name = tempfile.mkstemp(dir=cache_dir, suffix=".tmp.wav")
        os.close(fd)
        tmp_path = Path(tmp_name)
        try:
            extract_audio_clip(
                source, tmp_path, start_seconds=start, duration_seconds=duration
            )
            os.replace(tmp_path, cache_path)
        finally:
            tmp_path.unlink(missing_ok=True)
    return FileResponse(
        cache_path,
        media_type="audio/wav",
        headers={"Cache-Control": "private, max-age=3600"},
    )
