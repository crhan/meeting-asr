"""ASR pricing helpers for cost estimates."""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass

MAINLAND_REGION = "chinese_mainland"
INTERNATIONAL_REGION = "international"
US_REGION = "us"

_REGION_LABELS = {
    MAINLAND_REGION: "Chinese Mainland",
    INTERNATIONAL_REGION: "International",
    US_REGION: "US",
}

_PRICE_USD_PER_SECOND = {
    MAINLAND_REGION: {
        "fun-asr": 0.000032,
        "fun-asr-2025-11-07": 0.000032,
        "fun-asr-2025-08-25": 0.000032,
        "fun-asr-mtl": 0.000032,
        "fun-asr-mtl-2025-08-25": 0.000032,
        "qwen3-asr-flash-filetrans": 0.000032,
        "qwen3-asr-flash-filetrans-2025-11-17": 0.000032,
        "qwen3-asr-flash": 0.000032,
        "qwen3-asr-flash-2026-02-10": 0.000032,
        "qwen3-asr-flash-2025-09-08": 0.000032,
        "paraformer-v2": 0.000012,
        "paraformer-8k-v2": 0.000012,
    },
    INTERNATIONAL_REGION: {
        "fun-asr": 0.000035,
        "fun-asr-2025-11-07": 0.000035,
        "fun-asr-2025-08-25": 0.000035,
        "fun-asr-mtl": 0.000035,
        "fun-asr-mtl-2025-08-25": 0.000035,
        "qwen3-asr-flash-filetrans": 0.000035,
        "qwen3-asr-flash-filetrans-2025-11-17": 0.000035,
        "qwen3-asr-flash": 0.000035,
        "qwen3-asr-flash-2026-02-10": 0.000035,
        "qwen3-asr-flash-2025-09-08": 0.000035,
    },
    US_REGION: {
        "qwen3-asr-flash-us": 0.000035,
        "qwen3-asr-flash-2025-09-08-us": 0.000035,
    },
}


@dataclass(frozen=True, slots=True)
class AsrCostEstimate:
    """One ASR billing estimate."""

    model: str
    pricing_region: str
    billing_seconds: int
    audio_duration_seconds: float
    unit_price_usd_per_second: float
    estimated_cost_usd: float

    def to_dict(self) -> dict[str, str | int | float]:
        """
        Convert the estimate to JSON-ready metadata.

        Args:
            None.

        Returns:
            Dictionary suitable for ``project.json``.
        """
        payload = asdict(self)
        payload["currency"] = "USD"
        return payload


def estimate_asr_cost(
    *,
    model: str,
    base_url: str,
    audio_duration_seconds: float | None,
) -> AsrCostEstimate | None:
    """
    Estimate ASR input-audio cost from model, region, and duration.

    Args:
        model: DashScope ASR model id.
        base_url: DashScope base URL used for the request.
        audio_duration_seconds: Input audio duration in seconds.

    Returns:
        Cost estimate when both duration and model price are known.
    """
    if audio_duration_seconds is None or audio_duration_seconds <= 0:
        return None
    normalized_model = model.strip()
    pricing_region = pricing_region_from_base_url(base_url)
    unit_price = _PRICE_USD_PER_SECOND.get(pricing_region, {}).get(normalized_model)
    if unit_price is None:
        return None
    billing_seconds = max(1, math.ceil(audio_duration_seconds))
    return AsrCostEstimate(
        model=normalized_model,
        pricing_region=pricing_region,
        billing_seconds=billing_seconds,
        audio_duration_seconds=float(audio_duration_seconds),
        unit_price_usd_per_second=unit_price,
        estimated_cost_usd=round(billing_seconds * unit_price, 6),
    )


def pricing_region_from_base_url(base_url: str) -> str:
    """
    Infer the DashScope pricing region from the configured endpoint.

    Args:
        base_url: DashScope base URL.

    Returns:
        Pricing region key.
    """
    lowered = base_url.lower()
    if "dashscope-intl" in lowered:
        return INTERNATIONAL_REGION
    if "dashscope-us" in lowered:
        return US_REGION
    return MAINLAND_REGION


def format_asr_cost(estimate: AsrCostEstimate | None) -> str:
    """
    Render a human-readable ASR cost estimate.

    Args:
        estimate: Optional ASR cost estimate.

    Returns:
        Display string for CLI summaries.
    """
    if estimate is None:
        return "unavailable"
    region_label = _REGION_LABELS.get(estimate.pricing_region, estimate.pricing_region)
    return (
        f"${estimate.estimated_cost_usd:.6f} estimated "
        f"({estimate.billing_seconds}s x ${estimate.unit_price_usd_per_second:.6f}/s, {region_label})"
    )
