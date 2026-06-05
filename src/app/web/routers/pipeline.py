"""Ingestion pipeline over the web: full run, summarize, and merge.

These reuse the exact CLI orchestration so the web is a true "control console":

* ``run`` calls ``_run_project_workflow`` (the same orchestrator behind ``project run``)
  as a background job, streaming its multi-step progress over SSE.
* ``summarize`` calls ``summarize_project``.
* ``merge`` calls ``merge_projects`` / ``write_merge_outputs``.

Inputs are server-side file paths (this is a local single-user tool; the media lives on
the same machine). Long, external-service-calling runs (DashScope ASR + LLM) run in the
executor under the job manager.
"""

from __future__ import annotations

import asyncio
from pathlib import Path

from fastapi import APIRouter, Depends

from app.commands.project import _run_project_workflow  # reuse the run orchestrator
from app.core.project_models import ProjectTranscribeOptions
from app.lexicon_store import get_lexicon_db_path
from app.project_manager import (
    create_or_reuse_project,
    resolve_run_project_dir,
    summarize_project,
)
from app.transcript_merge import merge_projects, write_merge_outputs
from app.voiceprint_people import get_voiceprint_person
from app.voiceprint_store import get_voiceprint_db_path
from app.web.deps import (
    get_jobs,
    get_locks,
    get_settings,
    require_auth,
    resolve_web_project_ref,
)
from app.web.jobs import JobManager
from app.web.locks import LockRegistry, project_lock_key
from app.web.schemas import (
    JobRef,
    MergeApplyIn,
    MergePreviewIn,
    RunPipelineIn,
    SummarizeIn,
)
from app.web.settings import WebSettings

router = APIRouter(
    prefix="/api/pipeline", tags=["pipeline"], dependencies=[Depends(require_auth)]
)


def _require_file(path_str: str) -> Path:
    path = Path(path_str).expanduser()
    if not path.is_file():
        raise FileNotFoundError(f"Input file does not exist: {path}")
    return path.resolve()


@router.post("/run", response_model=JobRef)
async def run_pipeline(
    payload: RunPipelineIn,
    settings: WebSettings = Depends(get_settings),
    jobs: JobManager = Depends(get_jobs),
    locks: LockRegistry = Depends(get_locks),
) -> JobRef:
    """Run the full project pipeline as a background job (create -> ASR -> summarize)."""
    input_path = _require_file(payload.input_path)
    extra_inputs = [_require_file(p) for p in payload.extra_inputs]
    options = ProjectTranscribeOptions(
        speaker_count=payload.speaker_count,
        language=payload.language,
        model=payload.model,
        oss_upload=payload.oss_upload,
        file_url=None,
        generate_srt=True,
        timestamp_alignment=True,
        disfluency_removal=False,
        audio_format=payload.audio_format,
        asr_hotwords=payload.asr_hotwords,
    )
    lexicon_db = get_lexicon_db_path(settings.store_dir)
    loop = asyncio.get_running_loop()

    # Resolve the project under a lock keyed by its content-addressed identity, so two
    # concurrent first-time runs of the same media can't both create the same directory.
    # create_or_reuse then yields the project's ACTUAL directory -- which may be a
    # non-canonical path for a project originally made with --project-dir -- and the job is
    # keyed by that real path, the same key inline speaker/correction saves take, so a run
    # and an inline edit of the same project serialize instead of racing.
    create_key = await loop.run_in_executor(
        None,
        lambda: resolve_run_project_dir(
            input_path,
            extra_inputs=extra_inputs,
            projects_dir=settings.projects_dir,
            variant=payload.variant,
        ),
    )
    async with locks.acquire(project_lock_key(str(create_key))):
        created = await loop.run_in_executor(
            None,
            lambda: create_or_reuse_project(
                input_path,
                title=payload.title,
                projects_dir=settings.projects_dir,
                project_dir=None,
                meeting_time=payload.meeting_time,
                hash_source=True,
                variant=payload.variant,
                extra_inputs=extra_inputs,
            ),
        )
    project_dir = created.project_dir

    def work(reporter) -> dict[str, object]:
        summary = _run_project_workflow(
            input_path,
            extra_inputs=extra_inputs,
            title=payload.title,
            projects_dir=settings.projects_dir,
            project_dir=project_dir,
            meeting_time=payload.meeting_time,
            variant=payload.variant,
            options=options,
            store_dir=settings.store_dir,
            lexicon_db=lexicon_db,
            voiceprint_model=None,
            match_threshold=payload.match_threshold,
            summarize=payload.summarize,
            summary_model=None,
            polish=payload.polish,
            local_correction=payload.local_correction,
            correction_model=None,
            polish_concurrency=None,
            progress=reporter,
        )
        return {
            "project_id": summary.project.manifest.project_id,
            "project_dir": str(summary.project.project_dir),
            "detected_speaker_count": summary.transcription.detected_speaker_count,
            "sentence_count": summary.transcription.sentence_count,
            "applied_speaker_count": len(summary.applied_mapping),
            "has_summary": summary.meeting_summary is not None,
            "polished": summary.correction_summary is not None,
        }

    job = jobs.submit("pipeline-run", work, project_id=str(project_dir))
    return JobRef(job_id=job.id, kind=job.kind, status=job.status)


@router.post("/summarize/{project_ref}", response_model=JobRef)
def summarize(
    project_ref: str,
    payload: SummarizeIn,
    settings: WebSettings = Depends(get_settings),
    jobs: JobManager = Depends(get_jobs),
) -> JobRef:
    """Generate meeting memory-index artifacts for a project (background job)."""
    project_dir = resolve_web_project_ref(project_ref, settings)

    def work(reporter) -> dict[str, object]:
        result = summarize_project(
            project_dir,
            model=payload.model,
            update_title=payload.update_title,
            progress=reporter,
        )
        return {"project_dir": str(result.project_dir)}

    job = jobs.submit("pipeline-summarize", work, project_id=str(project_dir))
    return JobRef(job_id=job.id, kind=job.kind, status=job.status)


def _merge_result_payload(result) -> dict[str, object]:
    merged = result.merged_corrected or result.merged_raw
    return {
        "order_source": result.order_source,
        "use_corrected": result.use_corrected,
        "identity_count": len(result.identities),
        "speaker_count": len(result.mapping),
        "sentence_count": len(merged.sentences),
        "names": sorted(set(result.mapping.values())),
        "warnings": list(result.warnings),
    }


def _name_resolver(store_dir: Path | None):
    """Resolve a voiceprint person's authoritative name by vpp public id."""
    db_path = get_voiceprint_db_path(store_dir)

    def resolve(vpp: str) -> str | None:
        person = get_voiceprint_person(vpp, db_path)
        return person.name if person else None

    return resolve


@router.post("/merge-preview")
async def merge_preview(
    payload: MergePreviewIn,
    settings: WebSettings = Depends(get_settings),
) -> dict[str, object]:
    """Merge several projects into one transcript (preview only, no write)."""
    dirs = [resolve_web_project_ref(r, settings) for r in payload.project_refs]
    loop = asyncio.get_running_loop()
    result = await loop.run_in_executor(
        None,
        lambda: merge_projects(
            dirs,
            use_corrected=payload.use_corrected,
            name_to_vpp=payload.name_to_vpp,
            include_low_information=payload.include_low_information,
            keep_order=payload.keep_order,
            store_dir=settings.store_dir,
            title=payload.title,
            vpp_name_resolver=_name_resolver(settings.store_dir),
        ),
    )
    return _merge_result_payload(result)


@router.post("/merge")
async def merge_apply(
    payload: MergeApplyIn,
    settings: WebSettings = Depends(get_settings),
) -> dict[str, object]:
    """Merge several projects and write the output bundle to ``out_dir``."""
    dirs = [resolve_web_project_ref(r, settings) for r in payload.project_refs]
    out_dir = Path(payload.out_dir).expanduser().resolve()
    loop = asyncio.get_running_loop()

    def work() -> dict[str, object]:
        result = merge_projects(
            dirs,
            use_corrected=payload.use_corrected,
            name_to_vpp=payload.name_to_vpp,
            include_low_information=payload.include_low_information,
            keep_order=payload.keep_order,
            store_dir=settings.store_dir,
            title=payload.title,
            vpp_name_resolver=_name_resolver(settings.store_dir),
        )
        outputs = write_merge_outputs(result, out_dir, force=True)
        payload_out = _merge_result_payload(result)
        payload_out["out_dir"] = str(outputs.out_dir)
        payload_out["written"] = [
            str(p)
            for p in (
                outputs.transcript,
                outputs.transcript_corrected,
                outputs.subtitle,
                outputs.subtitle_corrected,
                outputs.manifest,
            )
            if p is not None
        ]
        return payload_out

    return await loop.run_in_executor(None, work)
