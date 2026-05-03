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

_CURRENCY_SYMBOLS = {
    "CNY": "¥",
    "USD": "$",
}

_REGION_CURRENCY = {
    MAINLAND_REGION: "CNY",
    INTERNATIONAL_REGION: "USD",
    US_REGION: "USD",
}

_PRICE_PER_SECOND = {
    MAINLAND_REGION: {
        "fun-asr": 0.00022,
        "fun-asr-2025-11-07": 0.00022,
        "fun-asr-2025-08-25": 0.00022,
        "fun-asr-mtl": 0.00022,
        "fun-asr-mtl-2025-08-25": 0.00022,
        "paraformer-v2": 0.00008,
        "paraformer-8k-v2": 0.00008,
    },
    INTERNATIONAL_REGION: {
        "fun-asr": 0.000035,
        "fun-asr-2025-11-07": 0.000035,
        "fun-asr-2025-08-25": 0.000035,
        "fun-asr-mtl": 0.000035,
        "fun-asr-mtl-2025-08-25": 0.000035,
    },
    US_REGION: {},
}


@dataclass(frozen=True, slots=True)
class AsrCostEstimate:
    """One ASR billing estimate."""

    model: str
    pricing_region: str
    billing_seconds: int
    audio_duration_seconds: float
    unit_price_per_second: float
    estimated_cost: float
    currency: str

    @property
    def unit_price_usd_per_second(self) -> float:
        """
        Return the unit price for legacy callers.

        Returns:
            Unit price in the estimate currency.
        """
        return self.unit_price_per_second

    @property
    def estimated_cost_usd(self) -> float:
        """
        Return the estimated cost for legacy callers.

        Returns:
            Estimated cost in the estimate currency.
        """
        return self.estimated_cost

    def to_dict(self) -> dict[str, str | int | float]:
        """
        Convert the estimate to JSON-ready metadata.

        Args:
            None.

        Returns:
            Dictionary suitable for ``project.json``.
        """
        payload = asdict(self)
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
    unit_price = _PRICE_PER_SECOND.get(pricing_region, {}).get(normalized_model)
    if unit_price is None:
        return None
    billing_seconds = max(1, math.ceil(audio_duration_seconds))
    currency = _REGION_CURRENCY.get(pricing_region, "USD")
    return AsrCostEstimate(
        model=normalized_model,
        pricing_region=pricing_region,
        billing_seconds=billing_seconds,
        audio_duration_seconds=float(audio_duration_seconds),
        unit_price_per_second=unit_price,
        estimated_cost=round(billing_seconds * unit_price, 6),
        currency=currency,
    )


def asr_cost_from_dict(payload: dict[str, object]) -> AsrCostEstimate:
    """
    Build an ASR cost estimate from manifest metadata.

    Args:
        payload: Serialized cost metadata.

    Returns:
        Parsed ASR cost estimate.
    """
    if estimate := _legacy_mainland_estimate(payload):
        return estimate
    if "unit_price_per_second" in payload and "estimated_cost" in payload:
        unit_price = float(payload["unit_price_per_second"])
        estimated_cost = float(payload["estimated_cost"])
    else:
        unit_price = float(payload["unit_price_usd_per_second"])
        estimated_cost = float(payload["estimated_cost_usd"])
    return AsrCostEstimate(
        model=str(payload["model"]),
        pricing_region=str(payload["pricing_region"]),
        billing_seconds=int(payload["billing_seconds"]),
        audio_duration_seconds=float(payload["audio_duration_seconds"]),
        unit_price_per_second=unit_price,
        estimated_cost=estimated_cost,
        currency=str(payload.get("currency") or "USD"),
    )


def _legacy_mainland_estimate(payload: dict[str, object]) -> AsrCostEstimate | None:
    """Re-price legacy mainland manifests that were written with USD field names."""
    region = str(payload.get("pricing_region") or "")
    model = str(payload.get("model") or "")
    if "unit_price_per_second" in payload or region != MAINLAND_REGION:
        return None
    unit_price = _PRICE_PER_SECOND.get(region, {}).get(model)
    if unit_price is None:
        return None
    billing_seconds = int(payload["billing_seconds"])
    return AsrCostEstimate(
        model=model,
        pricing_region=region,
        billing_seconds=billing_seconds,
        audio_duration_seconds=float(payload["audio_duration_seconds"]),
        unit_price_per_second=unit_price,
        estimated_cost=round(billing_seconds * unit_price, 6),
        currency=_REGION_CURRENCY[region],
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
    symbol = _CURRENCY_SYMBOLS.get(estimate.currency, f"{estimate.currency} ")
    return (
        f"{symbol}{estimate.estimated_cost:.6f} estimated "
        f"({estimate.billing_seconds}s x {symbol}{estimate.unit_price_per_second:.6f}/s, {region_label})"
    )
