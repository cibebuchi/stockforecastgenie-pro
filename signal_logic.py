from __future__ import annotations


def volatility_regime(realized_vol20_pct: float) -> str:
    """Classify 20-day annualized realized volatility for the deployment model."""
    v = float(realized_vol20_pct)
    if v < 12:
        return "Low"
    if v <= 22:
        return "Normal"
    return "High"


def signal_details(pred_return: float, confidence: float, realized_vol20_pct: float) -> dict:
    """Return the model signal plus the exact thresholds used to create it.

    The deployed pretrained model uses realized volatility rather than VIX.
    Thresholds are intentionally conservative and are expressed as decimal
    returns (for example, 0.007 = 0.7%).
    """
    regime = volatility_regime(realized_vol20_pct)
    directional_threshold = {"Low": 0.005, "Normal": 0.007, "High": 0.010}[regime]
    strong_threshold = 2.0 * directional_threshold
    directional_confidence_threshold = 0.55
    strong_confidence_threshold = 0.65

    r = float(pred_return)
    c = float(confidence)
    magnitude = abs(r)

    qualifies_directional = magnitude >= directional_threshold and c >= directional_confidence_threshold
    qualifies_strong = magnitude >= strong_threshold and c >= strong_confidence_threshold

    if not qualifies_directional:
        signal = "Neutral / No Clear Edge"
    elif r > 0:
        signal = "Strong Positive Model Signal" if qualifies_strong else "Positive Model Signal"
    elif r < 0:
        signal = "Strong Negative Model Signal" if qualifies_strong else "Negative Model Signal"
    else:
        signal = "Neutral / No Clear Edge"

    return {
        "signal": signal,
        "regime": regime,
        "forecast_return": r,
        "forecast_magnitude": magnitude,
        "confidence": c,
        "realized_vol20_pct": float(realized_vol20_pct),
        "directional_threshold": directional_threshold,
        "strong_threshold": strong_threshold,
        "directional_confidence_threshold": directional_confidence_threshold,
        "strong_confidence_threshold": strong_confidence_threshold,
        "qualifies_directional": qualifies_directional,
        "qualifies_strong": qualifies_strong,
    }


def model_signal(pred_return: float, confidence: float, realized_vol20_pct: float) -> tuple[str, str]:
    """Backward-compatible signal function used by the Streamlit app."""
    details = signal_details(pred_return, confidence, realized_vol20_pct)
    return details["signal"], details["regime"]
