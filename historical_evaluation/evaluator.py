"""Retrospective *fixed-model* evaluation for the StockForecastGenie conference app.

This module calls the deployed pretrained models only: no training, weights updates,
market-data persistence, or changes to the live inference code. Each replay passes
only observations through the requested forecast date to predict_frozen().

IMPORTANT: The frozen bundles were trained through 2026-10-02. Forecast dates in
September and early October are included for descriptive exploration, but these
are NOT out-of-sample historical forecasts. This is explicitly labeled per row.
FRED history fetched today is not guaranteed to be a genuine point-in-time vintage.
"""
from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd

from frozen_inference import load_bundle, predict_frozen
from signal_logic import signal_details

ALLOWED_TARGETS = ("DJIA", "SP500")
ALLOWED_HORIZONS = (1, 3, 5)
FIRST_ALLOWED_DATE = pd.Timestamp("2026-09-01")
LAST_ALLOWED_DATE = pd.Timestamp("2026-11-10")
MAX_BATCH_DATES = 80


class HistoricalEvaluationError(ValueError):
    """An evaluation input cannot be evaluated."""


def normalize_market_data(prices: pd.DataFrame) -> pd.DataFrame:
    """Work with genuine available trading closes, without filling market holidays."""
    if not {"Date", "Close"}.issubset(prices.columns):
        raise HistoricalEvaluationError("Historical data needs Date and Close columns.")
    out = prices[["Date", "Close"]].copy()
    out["Date"] = pd.to_datetime(out["Date"], errors="coerce").dt.tz_localize(None).dt.normalize()
    out["Close"] = pd.to_numeric(out["Close"], errors="coerce")
    out = out.replace([np.inf, -np.inf], np.nan).dropna()
    out = out.drop_duplicates("Date", keep="last").sort_values("Date").reset_index(drop=True)
    if out.empty:
        raise HistoricalEvaluationError("No observed market closes were returned.")
    if (out["Close"] <= 0).any():
        raise HistoricalEvaluationError("Market close observations must be positive.")
    return out


def read_training_cutoff(models_dir: Path, target: str, horizon: int) -> pd.Timestamp:
    """Retrieve the training-end date for a deployed bundle from its manifest."""
    path = Path(models_dir) / "MODEL_MANIFEST.csv"
    if not path.is_file():
        raise HistoricalEvaluationError("models/MODEL_MANIFEST.csv was not found.")
    manifest = pd.read_csv(path)
    required = {"target", "lead", "training_end"}
    if not required.issubset(manifest.columns):
        raise HistoricalEvaluationError("Model manifest lacks target, lead, or training_end.")
    matching = manifest.loc[
        (manifest["target"].astype(str) == target) &
        (pd.to_numeric(manifest["lead"], errors="coerce") == horizon)
    ]
    if len(matching) != 1:
        raise HistoricalEvaluationError(f"Expected one model manifest row for {target}/{horizon}d.")
    cutoff = pd.to_datetime(matching.iloc[0]["training_end"], errors="coerce")
    if pd.isna(cutoff):
        raise HistoricalEvaluationError("Invalid model training-end date in manifest.")
    return pd.Timestamp(cutoff).normalize()


def point_at_or_before(obs: pd.DataFrame | None, when) -> float | None:
    """Historical reference close; return None if absent or outdated (>7 days)."""
    if obs is None or obs.empty or pd.isna(when):
        return None
    d = normalize_market_data(obs)
    wanted = pd.Timestamp(when).normalize()
    sub = d.loc[d["Date"] <= wanted]
    if sub.empty or (wanted - sub.iloc[-1]["Date"]).days > 7:
        return None
    return float(sub.iloc[-1]["Close"])


def vix_regime(value: float | None) -> str:
    """VIX context-only regimes (different from price-derived model regimes)."""
    if value is None or not np.isfinite(value):
        return "Unavailable"
    if value < 15:
        return "Low"
    if value < 25:
        return "Normal"
    return "High"


def evaluation_category(asof: pd.Timestamp, cutoff: pd.Timestamp, horizon: int,
                        available_dates: pd.Series) -> str:
    """Describe evaluation provenance rather than claiming in-sample skill."""
    if asof <= cutoff:
        return "Training-period replay (not out-of-sample)"
    mature_sessions = int(((available_dates > cutoff) & (available_dates < asof)).sum())
    if mature_sessions < horizon:
        return "Training-boundary replay (not independent)"
    return "Post-training replay"


def evaluate_one(prices: pd.DataFrame, asof, target: str, horizon: int,
                 models_dir: Path, vix: pd.DataFrame | None = None,
                 bundle: dict | None = None) -> dict:
    if target not in ALLOWED_TARGETS or horizon not in ALLOWED_HORIZONS:
        raise HistoricalEvaluationError("Unsupported market or forecast horizon.")
    asof = pd.Timestamp(asof).normalize()
    if not FIRST_ALLOWED_DATE <= asof <= LAST_ALLOWED_DATE:
        raise HistoricalEvaluationError("Select a date from September 1 to November 10, 2026.")
    data = normalize_market_data(prices)
    matching_pos = data.index[data["Date"] == asof].tolist()
    if len(matching_pos) != 1:
        raise HistoricalEvaluationError(f"{asof.date()} is not an available observed market close (holiday/missing data).")
    current_pos = matching_pos[0]
    history = data.iloc[:current_pos + 1].copy()  # Never provide later market closes as features.
    if len(history) < 80:
        raise HistoricalEvaluationError("At least 80 market-close observations are needed before the selected date.")
    frozen = bundle if bundle is not None else load_bundle(models_dir, target, horizon)
    forecast = predict_frozen(frozen, history)
    issue = pd.Timestamp(forecast["runtime"]).normalize()
    if issue != asof:
        raise HistoricalEvaluationError(f"Forecast used {issue.date()} instead of requested date {asof.date()}.")

    close = float(forecast["current_close"])
    predicted = float(forecast["forecast_close"])
    lower = float(forecast["interval_low"])
    upper = float(forecast["interval_high"])
    predicted_return_pct = float(forecast["forecast_return"]) * 100.0
    details = signal_details(
        float(forecast["forecast_return"]),
        float(forecast["direction_confidence"]),
        float(forecast["realized_vol20"]),
    )
    outcome_pos = current_pos + horizon
    # Outcome dates are counted by observed trading sessions, not calendar days.
    outcome_date = data.iloc[outcome_pos]["Date"] if outcome_pos < len(data) else pd.NaT
    actual = float(data.iloc[outcome_pos]["Close"]) if outcome_pos < len(data) else np.nan
    has_outcome = np.isfinite(actual)
    actual_return_pct = ((actual / close) - 1) * 100.0 if has_outcome else np.nan
    cutoff = read_training_cutoff(models_dir, target, horizon)
    vix_before = point_at_or_before(vix, asof)
    vix_after = point_at_or_before(vix, outcome_date) if has_outcome else None
    model_category = evaluation_category(asof, cutoff, horizon, data["Date"])

    return {
        "Market": target,
        "Forecast date": asof.date().isoformat(),
        "Horizon": int(horizon),
        "Outcome date": outcome_date.date().isoformat() if has_outcome else "Pending",
        "Starting close": close,
        "Predicted close": predicted,
        "Actual close": actual,
        "Predicted gain (points)": predicted - close,
        "Predicted return (%)": predicted_return_pct,
        "Actual return (%)": actual_return_pct,
        "Absolute error (points)": abs(predicted - actual) if has_outcome else np.nan,
        "No-change error (points)": abs(close - actual) if has_outcome else np.nan,
        "Direction correct": bool(np.sign(predicted - close) == np.sign(actual - close)) if has_outcome else None,
        "Forecast range low": lower,
        "Forecast range high": upper,
        "Actual inside illustrative range": bool(lower <= actual <= upper) if has_outcome else None,
        "Signal": details["signal"],
        "Model realized-volatility regime": details["regime"],
        "20-day realized volatility (%)": float(details["realized_vol20_pct"]),
        "Model direction score (%)": float(details["confidence"]) * 100.0,
        "Directional threshold (%)": float(details["directional_threshold"]) * 100.0,
        "Strong-signal threshold (%)": float(details["strong_threshold"]) * 100.0,
        "Minimum direction score (%)": float(details["directional_confidence_threshold"]) * 100.0,
        "Minimum strong score (%)": float(details["strong_confidence_threshold"]) * 100.0,
        "VIX at forecast": vix_before,
        "VIX regime at forecast (context)": vix_regime(vix_before),
        "VIX at outcome": vix_after,
        "VIX regime at outcome (context)": vix_regime(vix_after),
        "VIX change": vix_after - vix_before if vix_after is not None and vix_before is not None else np.nan,
        "Model training through": cutoff.date().isoformat(),
        "Replay classification": model_category,
        "Status": "Evaluated" if has_outcome else "Outcome pending",
    }


def run_evaluation(prices: pd.DataFrame, dates, target: str, horizons,
                   models_dir: Path, vix: pd.DataFrame | None = None) -> tuple[pd.DataFrame, list[dict]]:
    """Process at most 80 historical session dates, reusing each loaded model."""
    requested_dates = sorted(set(pd.Timestamp(d).normalize() for d in dates))
    requested_horizons = sorted(set(int(h) for h in horizons))
    if not requested_dates or not requested_horizons:
        raise HistoricalEvaluationError("Choose at least one date and one forecast horizon.")
    if len(requested_dates) > MAX_BATCH_DATES:
        raise HistoricalEvaluationError(f"Choose at most {MAX_BATCH_DATES} observed market dates per run.")
    if target not in ALLOWED_TARGETS or any(h not in ALLOWED_HORIZONS for h in requested_horizons):
        raise HistoricalEvaluationError("Only DJIA/SP500 and 1-, 3-, or 5-trading-day horizons are supported.")
    bundles = {h: load_bundle(models_dir, target, h) for h in requested_horizons}
    results, skipped = [], []
    for day in requested_dates:
        for h in requested_horizons:
            try:
                results.append(evaluate_one(prices, day, target, h, models_dir, vix, bundles[h]))
            except (HistoricalEvaluationError, ValueError, KeyError, IndexError) as exc:
                skipped.append({"Date": day.date().isoformat(), "Horizon": h, "Reason": str(exc)})
    return pd.DataFrame(results), skipped
