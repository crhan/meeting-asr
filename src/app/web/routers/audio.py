"""Serve transcript audio clips over HTTP for the browser ``<audio>`` element.

The TUI shells out to mpv/afplay; the browser cannot. Instead we extract the same WAV
clip (``extract_audio_clip``) into a per-project cache and serve it. Starlette's
``FileResponse`` handles HTTP Range requests, so the player can seek. Clips are keyed by
time range and the project audio is immutable for a given transcript, so cached clips are
safe to reuse.
"""

from __future__ import annotations

from fastapi import APIRouter, Depends, Query
from fastapi.responses import FileResponse

from app.core.project_refs import resolve_project_ref
from app.infra.ffmpeg import extract_audio_clip
from app.project_manager import load_manifest, project_paths, resolve_project_audio_path
from app.web.deps import get_settings, require_auth
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
    project_dir = resolve_project_ref(project_ref, settings.projects_dir)
    manifest = load_manifest(project_dir)
    source = resolve_project_audio_path(project_dir, manifest)

    cache_dir = project_paths(project_dir).root / "tmp" / "web_clips"
    cache_path = cache_dir / f"{begin_ms}_{end_ms}.wav"
    if not cache_path.is_file():
        start = max(0.0, begin_ms / 1000.0 - _LEAD_IN_SECONDS)
        duration = (end_ms - begin_ms) / 1000.0 + _LEAD_IN_SECONDS + _TAIL_PAD_SECONDS
        extract_audio_clip(
            source, cache_path, start_seconds=start, duration_seconds=duration
        )
    return FileResponse(
        cache_path,
        media_type="audio/wav",
        headers={"Cache-Control": "private, max-age=3600"},
    )
