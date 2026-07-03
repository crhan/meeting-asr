"""Cross-project correction lexicon: terms CRUD, stats, disambiguations, hotwords.

The lexicon is its own global SQLite store (separate from the voiceprint store). Its db
path is resolved from ``settings.store_dir`` so an isolated ``--store-dir`` copy is honored
for both reads and writes -- otherwise testing against a copy would silently mutate the
real correction dictionary. Writes run in the executor under a dedicated lexicon lock.
"""

from __future__ import annotations

import asyncio

from fastapi import APIRouter, Depends, Query

from app.lexicon_store import (
    delete_lexicon_term,
    get_lexicon_db_path,
    get_lexicon_term,
    lexicon_stats,
    list_asr_hotwords,
    list_lexicon_disambiguations,
    list_lexicon_terms,
    set_alias_disambiguation,
    upsert_lexicon_term,
)
from app.web.deps import get_locks, get_settings, require_auth
from app.web.locks import LockRegistry, store_lock_key
from app.web.schemas import (
    DisambiguationOut,
    HotwordOut,
    LexiconAliasOut,
    LexiconContextOut,
    LexiconStatsOut,
    LexiconTermDetailOut,
    LexiconTermOut,
    LexiconTermsOut,
    SetDisambiguationIn,
    UpsertTermIn,
)
from app.web.settings import WebSettings

router = APIRouter(
    prefix="/api/lexicon", tags=["lexicon"], dependencies=[Depends(require_auth)]
)

_LEXICON_LOCK = store_lock_key("lexicon")


def _term_out(term) -> LexiconTermOut:
    return LexiconTermOut(
        term_id=term.term_id,
        public_id=term.public_id,
        canonical=term.canonical,
        category=term.category,
        description=term.description,
        status=term.status,
        alias_count=term.alias_count,
        context_count=term.context_count,
        ambiguous_alias_count=term.ambiguous_alias_count,
        created_at=term.created_at,
        updated_at=term.updated_at,
    )


async def _run(locks: LockRegistry, fn):
    loop = asyncio.get_running_loop()
    async with locks.acquire(_LEXICON_LOCK):
        return await loop.run_in_executor(None, fn)


@router.get("/terms", response_model=LexiconTermsOut)
def get_terms(
    query: str | None = Query(default=None),
    category: str | None = Query(default=None),
    status: str = Query(default="active"),
    limit: int = Query(default=200, ge=1, le=1000),
    settings: WebSettings = Depends(get_settings),
) -> LexiconTermsOut:
    """List lexicon terms, optionally filtered by text/category/status."""
    terms = list_lexicon_terms(
        status=status,
        category=category,
        query=query,
        limit=limit,
        db_path=get_lexicon_db_path(settings.store_dir),
    )
    return LexiconTermsOut(terms=[_term_out(t) for t in terms])


@router.get("/stats", response_model=LexiconStatsOut)
def get_stats(settings: WebSettings = Depends(get_settings)) -> LexiconStatsOut:
    """Return aggregate lexicon statistics."""
    stats = lexicon_stats(db_path=get_lexicon_db_path(settings.store_dir))
    return LexiconStatsOut(
        active_terms=stats.active_terms,
        inactive_terms=stats.inactive_terms,
        aliases=stats.aliases,
        contexts=stats.contexts,
        hotwords=stats.hotwords,
        cached_vocabularies=stats.cached_vocabularies,
    )


@router.get("/disambiguations", response_model=list[DisambiguationOut])
def get_disambiguations(
    settings: WebSettings = Depends(get_settings),
) -> list[DisambiguationOut]:
    """List context-dependent aliases with user guidance."""
    rows = list_lexicon_disambiguations(db_path=get_lexicon_db_path(settings.store_dir))
    return [
        DisambiguationOut(
            alias=row.alias,
            canonical=row.canonical,
            category=row.category,
            guidance=row.guidance,
        )
        for row in rows
    ]


@router.post("/disambiguations", response_model=DisambiguationOut | None)
async def set_disambiguation(
    payload: SetDisambiguationIn,
    settings: WebSettings = Depends(get_settings),
    locks: LockRegistry = Depends(get_locks),
) -> DisambiguationOut | None:
    """Mark an alias as context-ambiguous so polish resolves it per sentence, or clear it.

    Mirrors ``meeting-asr lexicon disambiguate``: an empty ``guidance`` returns the alias to
    deterministic blanket replacement (response is null). Without this, a web-only user could
    add aliases but never mark business-ambiguous ones (e.g. a surface that means a product in
    one context and a person in another), leaving them wrongly blanket-replaced.
    """
    db_path = get_lexicon_db_path(settings.store_dir)
    entry = await _run(
        locks,
        lambda: set_alias_disambiguation(
            term=payload.term,
            alias=payload.alias,
            guidance=payload.guidance,
            db_path=db_path,
        ),
    )
    if entry is None:
        return None
    return DisambiguationOut(
        alias=entry.alias,
        canonical=entry.canonical,
        category=entry.category,
        guidance=entry.guidance,
    )


@router.get("/hotwords", response_model=list[HotwordOut])
def get_hotwords(
    limit: int = Query(default=500, ge=1, le=2000),
    settings: WebSettings = Depends(get_settings),
) -> list[HotwordOut]:
    """List ASR hotwords derived from accepted corrections."""
    rows = list_asr_hotwords(
        limit=limit, db_path=get_lexicon_db_path(settings.store_dir)
    )
    return [
        HotwordOut(
            text=row.text, weight=row.weight, category=row.category, source=row.source
        )
        for row in rows
    ]


@router.get("/terms/{ref}", response_model=LexiconTermDetailOut)
def get_term_detail(
    ref: str, settings: WebSettings = Depends(get_settings)
) -> LexiconTermDetailOut:
    """One term's full detail: aliases (with disambiguation) + recent contexts.

    ``ref`` resolves like the CLI/DELETE: term id, canonical text, or alias
    (LookupError -> 404).
    """
    detail = get_lexicon_term(ref, db_path=get_lexicon_db_path(settings.store_dir))
    return LexiconTermDetailOut(
        term=_term_out(detail.term),
        aliases=[
            LexiconAliasOut(
                alias=alias.alias,
                alias_type=alias.alias_type,
                disambiguation=alias.disambiguation,
            )
            for alias in detail.aliases
        ],
        contexts=[
            LexiconContextOut(
                wrong_text=row.wrong_text,
                corrected_text=row.corrected_text,
                left_context=row.left_context,
                right_context=row.right_context,
                speaker_name=row.speaker_name,
                project_id=row.project_id,
                sentence_id=row.sentence_id,
                source=row.source,
                created_at=row.created_at,
            )
            for row in detail.contexts
        ],
    )


@router.post("/terms", response_model=LexiconTermOut)
async def create_or_update_term(
    payload: UpsertTermIn,
    settings: WebSettings = Depends(get_settings),
    locks: LockRegistry = Depends(get_locks),
) -> LexiconTermOut:
    """Create or update a lexicon term (merges aliases)."""
    db_path = get_lexicon_db_path(settings.store_dir)
    detail = await _run(
        locks,
        lambda: upsert_lexicon_term(
            canonical=payload.canonical,
            category=payload.category,
            description=payload.description,
            aliases=tuple(payload.aliases),
            status=payload.status,
            db_path=db_path,
        ),
    )
    return _term_out(detail.term)


@router.delete("/terms/{ref}")
async def remove_term(
    ref: str,
    permanent: bool = Query(default=False),
    settings: WebSettings = Depends(get_settings),
    locks: LockRegistry = Depends(get_locks),
) -> dict[str, str]:
    """Delete (or deactivate) a lexicon term."""
    db_path = get_lexicon_db_path(settings.store_dir)
    detail = await _run(
        locks, lambda: delete_lexicon_term(ref, permanent=permanent, db_path=db_path)
    )
    return {"deleted_public_id": detail.term.public_id}
