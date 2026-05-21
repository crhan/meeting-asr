"""Prefixed object id helpers."""

from __future__ import annotations

import secrets
from collections.abc import Callable


def new_prefixed_id(
    prefix: str, exists: Callable[[str], bool], *, bytes_count: int = 8
) -> str:
    """Generate a collision-free prefixed object id."""
    while True:
        object_id = f"{prefix}{secrets.token_hex(bytes_count)}"
        if not exists(object_id):
            return object_id
