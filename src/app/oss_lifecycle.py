"""OSS lifecycle configuration helpers."""

from __future__ import annotations

from typing import Any

from app.config import Settings
from app.uploader import build_oss_bucket, import_oss2


def set_lifecycle_rule(
    settings: Settings, *, prefix: str, days: int, rule_id: str
) -> None:
    """
    Configure an OSS lifecycle rule that deletes objects after N days.

    Args:
        settings: Runtime settings with OSS values.
        prefix: Object prefix.
        days: Expiration days.
        rule_id: Lifecycle rule id.
    """
    oss2 = import_oss2()
    bucket = build_oss_bucket(settings)
    rules = _load_lifecycle_rules(bucket, oss2)
    rule = oss2.models.LifecycleRule(
        rule_id,
        prefix=prefix,
        status="Enabled",
        expiration=oss2.models.LifecycleExpiration(days=days),
    )
    preserved_rules = [
        existing_rule
        for existing_rule in rules
        if getattr(existing_rule, "id", None) != rule_id
    ]
    lifecycle = oss2.models.BucketLifecycle([*preserved_rules, rule])
    bucket.put_bucket_lifecycle(lifecycle)


def _load_lifecycle_rules(bucket: Any, oss2: Any) -> list[Any]:
    """Load current lifecycle rules, treating missing lifecycle as empty."""
    try:
        lifecycle = bucket.get_bucket_lifecycle()
    except oss2.exceptions.NoSuchLifecycle:
        return []
    return list(getattr(lifecycle, "rules", []))
