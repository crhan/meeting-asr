"""Tests for OSS lifecycle helpers."""

from __future__ import annotations

from typing import Any

from app.oss_lifecycle import (
    DEFAULT_LIFECYCLE_DAYS,
    DEFAULT_LIFECYCLE_PREFIX,
    DEFAULT_LIFECYCLE_RULE_ID,
    ensure_bucket_lifecycle_rule,
)
from app.uploader import import_oss2

oss2 = import_oss2()


class FakeLifecycleBucket:
    """Fake oss2 Bucket exposing only lifecycle methods."""

    def __init__(self, rules: list[Any] | None = None) -> None:
        """
        Create a fake bucket with optional preexisting lifecycle rules.

        Args:
            rules: Existing lifecycle rules; None means no lifecycle at all.
        """
        self.rules = rules
        self.put_calls: list[Any] = []

    def get_bucket_lifecycle(self):
        """Return existing rules or raise like a lifecycle-less bucket."""
        if self.rules is None:
            raise oss2.exceptions.NoSuchLifecycle(404, {}, b"", {})
        return oss2.models.BucketLifecycle(list(self.rules))

    def put_bucket_lifecycle(self, lifecycle) -> None:
        """Record the lifecycle configuration written to the bucket."""
        self.put_calls.append(lifecycle)


def test_ensure_creates_rule_when_bucket_has_no_lifecycle() -> None:
    """A lifecycle-less bucket should gain the default auto-delete rule."""
    bucket = FakeLifecycleBucket(rules=None)

    created = ensure_bucket_lifecycle_rule(bucket)

    assert created is True
    assert len(bucket.put_calls) == 1
    (rule,) = bucket.put_calls[0].rules
    assert rule.id == DEFAULT_LIFECYCLE_RULE_ID
    assert rule.prefix == DEFAULT_LIFECYCLE_PREFIX
    assert rule.status == "Enabled"
    assert rule.expiration.days == DEFAULT_LIFECYCLE_DAYS


def test_ensure_keeps_existing_rule_untouched() -> None:
    """A preexisting rule with the same id must never be overwritten."""
    existing = oss2.models.LifecycleRule(
        DEFAULT_LIFECYCLE_RULE_ID,
        prefix=DEFAULT_LIFECYCLE_PREFIX,
        status="Enabled",
        expiration=oss2.models.LifecycleExpiration(days=30),
    )
    bucket = FakeLifecycleBucket(rules=[existing])

    created = ensure_bucket_lifecycle_rule(bucket)

    assert created is False
    assert bucket.put_calls == []


def test_ensure_preserves_unrelated_rules() -> None:
    """Creating our rule must keep other bucket rules intact."""
    unrelated = oss2.models.LifecycleRule(
        "other-team-rule",
        prefix="other/",
        status="Enabled",
        expiration=oss2.models.LifecycleExpiration(days=90),
    )
    bucket = FakeLifecycleBucket(rules=[unrelated])

    created = ensure_bucket_lifecycle_rule(bucket)

    assert created is True
    (lifecycle,) = bucket.put_calls
    rule_ids = [rule.id for rule in lifecycle.rules]
    assert rule_ids == ["other-team-rule", DEFAULT_LIFECYCLE_RULE_ID]
