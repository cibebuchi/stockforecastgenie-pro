from __future__ import annotations


def volatility_regime(realized_vol20_pct: float) -> str:
    v = float(realized_vol20_pct)
    if v < 12:
        return "Low"
    if v <= 22:
        return "Normal"
    return "High"


def model_signal(pred_return: float, confidence: float, realized_vol20_pct: float) -> tuple[str, str]:
    regime = volatility_regime(realized_vol20_pct)
    threshold = {"Low": 0.005, "Normal": 0.007, "High": 0.010}[regime]
    r = float(pred_return)
    c = float(confidence)
    if abs(r) < threshold or c < 0.55:
        return "Neutral / No Clear Edge", regime
    if r > 0:
        return ("Strong Positive Model Signal" if abs(r) >= 2 * threshold and c >= 0.65 else "Positive Model Signal"), regime
    return ("Strong Negative Model Signal" if abs(r) >= 2 * threshold and c >= 0.65 else "Negative Model Signal"), regime
