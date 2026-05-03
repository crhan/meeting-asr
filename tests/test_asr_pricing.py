"""Tests for ASR cost estimation."""

from __future__ import annotations

from app.asr_pricing import asr_cost_from_dict, estimate_asr_cost, format_asr_cost


def test_estimate_fun_asr_mainland_cost_rounds_to_billable_seconds() -> None:
    """Fun-ASR cost estimates should use endpoint region and ceiling seconds."""
    estimate = estimate_asr_cost(
        model="fun-asr",
        base_url="https://dashscope.aliyuncs.com/api/v1",
        audio_duration_seconds=4553.2,
    )

    assert estimate is not None
    assert estimate.pricing_region == "chinese_mainland"
    assert estimate.billing_seconds == 4554
    assert estimate.currency == "CNY"
    assert estimate.unit_price_per_second == 0.00022
    assert estimate.estimated_cost == 1.00188
    assert format_asr_cost(estimate) == "¥1.001880 estimated (4554s x ¥0.000220/s, Chinese Mainland)"


def test_estimate_returns_none_for_unknown_model() -> None:
    """Unknown model pricing should not produce a fake cost."""
    estimate = estimate_asr_cost(
        model="unknown-asr",
        base_url="https://dashscope.aliyuncs.com/api/v1",
        audio_duration_seconds=60.0,
    )

    assert estimate is None
    assert format_asr_cost(estimate) == "unavailable"


def test_legacy_mainland_cost_metadata_is_repriced_as_cny() -> None:
    """Older manifests used USD field names for mainland DashScope costs."""
    estimate = asr_cost_from_dict(
        {
            "model": "fun-asr",
            "pricing_region": "chinese_mainland",
            "billing_seconds": 4554,
            "audio_duration_seconds": 4553.2,
            "unit_price_usd_per_second": 0.000032,
            "estimated_cost_usd": 0.145728,
            "currency": "USD",
        }
    )

    assert estimate.currency == "CNY"
    assert format_asr_cost(estimate) == "¥1.001880 estimated (4554s x ¥0.000220/s, Chinese Mainland)"
