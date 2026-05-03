"""Tests for ASR cost estimation."""

from __future__ import annotations

from app.asr_pricing import estimate_asr_cost, format_asr_cost


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
    assert estimate.unit_price_usd_per_second == 0.000032
    assert estimate.estimated_cost_usd == 0.145728
    assert format_asr_cost(estimate) == "$0.145728 estimated (4554s x $0.000032/s, Chinese Mainland)"


def test_estimate_returns_none_for_unknown_model() -> None:
    """Unknown model pricing should not produce a fake cost."""
    estimate = estimate_asr_cost(
        model="unknown-asr",
        base_url="https://dashscope.aliyuncs.com/api/v1",
        audio_duration_seconds=60.0,
    )

    assert estimate is None
    assert format_asr_cost(estimate) == "unavailable"
