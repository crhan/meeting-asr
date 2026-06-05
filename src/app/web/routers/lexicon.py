"""Cross-project correction lexicon: terms CRUD, stats, disambiguations, hotwords.

The lexicon is its own global SQLite store (separate from the voiceprint store), so these
endpoints use the default lexicon db. Writes run in the executor under a dedicated lexicon
store lock.
"""

from __future__ import annotations

import asyncio

from fastapi import APIRouter, Depends, Query

from app.lexicon_store import (
    delete_lexicon_term,
    lexicon_stats,
    list_asr_hotwords,
    list_lexicon_disambiguations,
    list_lexicon_terms,
    upsert_lexicon_term,
)
from app.web.deps import get_locks, require_auth
from app.web.locks import LockRegistry, store_lock_key
from app.web.schemas import (
    DisambiguationOut,
    HotwordOut,
    LexiconStatsOut,
    LexiconTermOut,
    LexiconTermsOut,
    UpsertTermIn,
)

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
) -> LexiconTermsOut:
    """List lexicon terms, optionally filtered by text/category/status."""
    terms = list_lexicon_terms(
        status=status, category=category, query=query, limit=limit
    )
    return LexiconTermsOut(terms=[_term_out(t) for t in terms])


@router.get("/stats", response_model=LexiconStatsOut)
def get_stats() -> LexiconStatsOut:
    """Return aggregate lexicon statistics."""
    stats = lexicon_stats()
    return LexiconStatsOut(
        active_terms=stats.active_terms,
        inactive_terms=stats.inactive_terms,
        aliases=stats.aliases,
        contexts=stats.contexts,
        hotwords=stats.hotwords,
        cached_vocabularies=stats.cached_vocabularies,
    )


@router.get("/disambiguations", response_model=list[DisambiguationOut])
def get_disambiguations() -> list[DisambiguationOut]:
    """List context-dependent aliases with user guidance."""
    rows = list_lexicon_disambiguations()
    return [
        DisambiguationOut(
            alias=row.alias,
            canonical=row.canonical,
            category=row.category,
            guidance=row.guidance,
        )
        for row in rows
    ]


@router.get("/hotwords", response_model=list[HotwordOut])
def get_hotwords(limit: int = Query(default=500, ge=1, le=2000)) -> list[HotwordOut]:
    """List ASR hotwords derived from accepted corrections."""
    rows = list_asr_hotwords(limit=limit)
    return [
        HotwordOut(
            text=row.text, weight=row.weight, category=row.category, source=row.source
        )
        for row in rows
    ]


@router.post("/terms", response_model=LexiconTermOut)
async def create_or_update_term(
    payload: UpsertTermIn, locks: LockRegistry = Depends(get_locks)
) -> LexiconTermOut:
    """Create or update a lexicon term (merges aliases)."""
    detail = await _run(
        locks,
        lambda: upsert_lexicon_term(
            canonical=payload.canonical,
            category=payload.category,
            description=payload.description,
            aliases=tuple(payload.aliases),
            status=payload.status,
        ),
    )
    return _term_out(detail.term)


@router.delete("/terms/{ref}")
async def remove_term(
    ref: str,
    permanent: bool = Query(default=False),
    locks: LockRegistry = Depends(get_locks),
) -> dict[str, str]:
    """Delete (or deactivate) a lexicon term."""
    detail = await _run(locks, lambda: delete_lexicon_term(ref, permanent=permanent))
    return {"deleted_public_id": detail.term.public_id}
