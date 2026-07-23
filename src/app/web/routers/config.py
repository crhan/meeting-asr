"""Global configuration + environment diagnostics over the web.

Lets a web user set DashScope/OSS credentials and run the same checks as
``meeting-asr doctor`` without touching the terminal. Secret values are masked unless
``reveal=true`` is requested (loopback-only by default, so reveal is acceptable locally).
"""

from __future__ import annotations

import asyncio

from fastapi import APIRouter, Depends, HTTPException, Query

from app.commands.doctor import (
    _check_editor,
    _check_ffmpeg,
    _check_preview_player,
    _check_python,
    _check_python_packages,
    _check_settings,
    _check_voiceprint_embedding_settings,
)
from app.config import (
    CONFIG_KEYS,
    get_config_path,
    load_config_values,
    set_config_value,
    unset_config_value,
)
from app.web.deps import get_locks, get_settings, require_auth
from app.web.locks import LockRegistry, store_lock_key
from app.web.schemas import (
    ConfigKeyOut,
    ConfigOut,
    DoctorCheckOut,
    DoctorOut,
    SetConfigIn,
)
from app.web.settings import WebSettings

router = APIRouter(tags=["config"], dependencies=[Depends(require_auth)])

_CONFIG_LOCK = store_lock_key("config")


@router.get("/api/config", response_model=ConfigOut)
def get_config(
    reveal: bool = Query(default=False),
    settings: WebSettings = Depends(get_settings),
) -> ConfigOut:
    """List config keys with current values (secrets masked unless reveal=true).

    Revealing plaintext secrets is loopback-only: a bearer token alone must not let a
    networked client exfiltrate DashScope/OSS credentials (we do not force HTTPS, so the
    response could also be sniffed). On a non-loopback bind, ``reveal=true`` is refused.
    """
    if reveal and not settings.is_local:
        raise HTTPException(
            status_code=403,
            detail="Revealing secret values is only permitted from a loopback bind.",
        )
    values = load_config_values()
    keys = []
    for key in CONFIG_KEYS:
        # load_config_values keys by PUBLIC name (e.g. "dashscope.api_key"), not the
        # dataclass field_name -- looking up field_name always missed, so every key showed
        # as unset and reveal returned nothing.
        raw = values.get(key.name)
        is_set = bool(raw)
        shown = raw if (raw and (reveal or not key.secret)) else None
        keys.append(
            ConfigKeyOut(
                name=key.name,
                env_name=key.env_name,
                secret=key.secret,
                is_set=is_set,
                value=shown,
            )
        )
    return ConfigOut(config_file=str(get_config_path()), keys=keys)


@router.patch("/api/config")
async def set_config(
    payload: SetConfigIn, locks: LockRegistry = Depends(get_locks)
) -> dict[str, str]:
    """Set one config key."""
    loop = asyncio.get_running_loop()
    async with locks.acquire(_CONFIG_LOCK):
        key, _path = await loop.run_in_executor(
            None, lambda: set_config_value(payload.key, payload.value)
        )
    return {"key": key}


@router.delete("/api/config/{key}")
async def unset_config(
    key: str, locks: LockRegistry = Depends(get_locks)
) -> dict[str, str]:
    """Unset one config key."""
    loop = asyncio.get_running_loop()
    async with locks.acquire(_CONFIG_LOCK):
        resolved, _path = await loop.run_in_executor(
            None, lambda: unset_config_value(key)
        )
    return {"key": resolved}


@router.get("/api/doctor", response_model=DoctorOut)
async def doctor(oss: bool = Query(default=False)) -> DoctorOut:
    """Run environment diagnostics (same checks as `meeting-asr doctor`)."""

    def run_checks():
        results = [
            _check_python(),
            _check_python_packages(require_oss=oss),
            _check_ffmpeg(),
            _check_preview_player(),
            _check_editor(),
            _check_settings(require_oss=oss),
            _check_voiceprint_embedding_settings(required=False),
        ]
        return results

    loop = asyncio.get_running_loop()
    checks = await loop.run_in_executor(None, run_checks)
    out = [
        DoctorCheckOut(
            name=c.name, status=c.status, detail=c.detail, fix_prompt=c.fix_prompt
        )
        for c in checks
    ]
    return DoctorOut(ok=not any(c.status == "fail" for c in out), checks=out)
