from typing import Dict, Optional


def directional_confidence(pred_return: float, prob_up: Optional[float]) -> float:
    """Confidence in the direction implied by the regression forecast."""
    if prob_up is not None:
        p = float(min(max(prob_up, 0.0), 1.0))
        return p if pred_return >= 0 else 1.0 - p
    magnitude = abs(float(pred_return))
    return float(min(0.50 + (magnitude / 0.02) * 0.20, 0.75))


def vix_regime(vix_value: Optional[float]) -> Dict:
    if vix_value is None or vix_value != vix_value:
        return {"regime": "Unknown", "edge_threshold": 0.005, "strong_edge_threshold": 0.010, "confidence_threshold": 0.58}
    if vix_value >= 25:
        return {"regime": "High volatility", "edge_threshold": 0.008, "strong_edge_threshold": 0.012, "confidence_threshold": 0.63}
    if vix_value >= 15:
        return {"regime": "Normal volatility", "edge_threshold": 0.005, "strong_edge_threshold": 0.010, "confidence_threshold": 0.58}
    return {"regime": "Low volatility", "edge_threshold": 0.003, "strong_edge_threshold": 0.007, "confidence_threshold": 0.55}


def classify_signal(
    pred_return: float,
    prob_up: Optional[float],
    recent_naive_return: float,
    vix_value: Optional[float],
    signal_buffer: float = 0.002,
) -> Dict:
    confidence = directional_confidence(pred_return, prob_up)
    regime = vix_regime(vix_value)

    recent_naive_return = float(recent_naive_return or 0.0)
    positive_reference = max(0.0, recent_naive_return)
    negative_reference = min(0.0, recent_naive_return)
    positive_edge = float(pred_return - positive_reference)
    negative_edge = float(pred_return - negative_reference)

    edge_threshold = regime["edge_threshold"] + signal_buffer
    strong_threshold = regime["strong_edge_threshold"] + signal_buffer
    base_conf = max(0.50, regime["confidence_threshold"] - 0.03)

    if positive_edge >= strong_threshold and confidence >= regime["confidence_threshold"]:
        signal, tone = "Strong Positive", "positive"
    elif positive_edge >= edge_threshold and confidence >= base_conf:
        signal, tone = "Positive", "positive"
    elif negative_edge <= -strong_threshold and confidence >= regime["confidence_threshold"]:
        signal, tone = "Strong Negative", "negative"
    elif negative_edge <= -edge_threshold and confidence >= base_conf:
        signal, tone = "Negative", "negative"
    else:
        signal, tone = "Neutral / No Clear Edge", "neutral"

    return {
        "signal": signal,
        "tone": tone,
        "confidence": confidence,
        "regime": regime["regime"],
        "predicted_return_pct": pred_return * 100.0,
        "edge_vs_flat_pct": pred_return * 100.0,
        "edge_vs_recent_pct": (pred_return - recent_naive_return) * 100.0,
        "required_edge_pct": edge_threshold * 100.0,
        "signal_buffer_pct": signal_buffer * 100.0,
        "vix_value": None if vix_value is None or vix_value != vix_value else float(vix_value),
    }


def market_context(current_price: float, reference_price: Optional[float]) -> Dict:
    if reference_price is None or reference_price != reference_price or reference_price == 0:
        return {"bias": "Unknown", "trend_pct": None}
    trend = float((current_price - reference_price) / reference_price)
    if trend >= 0.03:
        bias = "Bullish"
    elif trend <= -0.03:
        bias = "Bearish"
    else:
        bias = "Neutral"
    return {"bias": bias, "trend_pct": trend * 100.0}


def signal_alignment(signal: str, bias: str) -> str:
    if signal in {"Positive", "Strong Positive"} and bias == "Bullish":
        return "Aligned"
    if signal in {"Negative", "Strong Negative"} and bias == "Bearish":
        return "Aligned"
    if signal in {"Positive", "Strong Positive"} and bias == "Bearish":
        return "Mixed"
    if signal in {"Negative", "Strong Negative"} and bias == "Bullish":
        return "Mixed"
    return "Neutral"
