"""OSS lifecycle configuration helpers."""

from __future__ import annotations

from app.config import Settings
from app.uploader import build_oss_bucket


def set_lifecycle_rule(settings: Settings, *, prefix: str, days: int, rule_id: str) -> None:
    """
    Configure an OSS lifecycle rule that deletes objects after N days.

    Args:
        settings: Runtime settings with OSS values.
        prefix: Object prefix.
        days: Expiration days.
        rule_id: Lifecycle rule id.
    """
    import oss2
    from oss2.models import LifecycleExpiration, LifecycleRule, BucketLifecycle

    bucket = build_oss_bucket(settings)
    rule = LifecycleRule(rule_id, prefix=prefix, status="Enabled", expiration=LifecycleExpiration(days=days))
    lifecycle = BucketLifecycle([rule])
    bucket.put_bucket_lifecycle(lifecycle)
