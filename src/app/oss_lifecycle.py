"""OSS lifecycle configuration helpers."""

from __future__ import annotations

from typing import Any

from app.config import Settings
from app.uploader import build_oss_bucket, import_oss2

DEFAULT_LIFECYCLE_PREFIX = "meeting-asr/"
DEFAULT_LIFECYCLE_DAYS = 1
DEFAULT_LIFECYCLE_RULE_ID = "meeting-asr-auto-delete"


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


def ensure_bucket_lifecycle_rule(
    bucket: Any,
    *,
    prefix: str = DEFAULT_LIFECYCLE_PREFIX,
    days: int = DEFAULT_LIFECYCLE_DAYS,
    rule_id: str = DEFAULT_LIFECYCLE_RULE_ID,
) -> bool:
    """
    Ensure an auto-delete lifecycle rule exists on the bucket.

    Unlike :func:`set_lifecycle_rule`, this never overwrites an existing rule
    with the same id, so a manually tuned ``lifecycle set --days N`` survives
    subsequent uploads.

    Args:
        bucket: ``oss2.Bucket`` instance.
        prefix: Object prefix.
        days: Expiration days used only when the rule is missing.
        rule_id: Lifecycle rule id.

    Returns:
        True when a new rule was created, False when it already existed.
    """
    oss2 = import_oss2()
    rules = _load_lifecycle_rules(bucket, oss2)
    if any(getattr(existing_rule, "id", None) == rule_id for existing_rule in rules):
        return False
    rule = oss2.models.LifecycleRule(
        rule_id,
        prefix=prefix,
        status="Enabled",
        expiration=oss2.models.LifecycleExpiration(days=days),
    )
    bucket.put_bucket_lifecycle(oss2.models.BucketLifecycle([*rules, rule]))
    return True


def _load_lifecycle_rules(bucket: Any, oss2: Any) -> list[Any]:
    """Load current lifecycle rules, treating missing lifecycle as empty."""
    try:
        lifecycle = bucket.get_bucket_lifecycle()
    except oss2.exceptions.NoSuchLifecycle:
        return []
    return list(getattr(lifecycle, "rules", []))
