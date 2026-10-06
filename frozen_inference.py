from __future__ import annotations
from pathlib import Path
from typing import Dict
import joblib
import numpy as np
import pandas as pd

from frozen_features import latest_feature_row


def bundle_path(models_dir: Path, target: str, lead: int) -> Path:
    return Path(models_dir) / f"frozen_{target}_lead{int(lead)}.joblib"


def load_bundle(models_dir: Path, target: str, lead: int) -> dict:
    path = bundle_path(models_dir, target, lead)
    if not path.exists():
        raise FileNotFoundError(
            f"Frozen model not found: {path.name}. Run the offline Yahoo freezing script locally first; "
            "the deployed app intentionally cannot train a model."
        )
    bundle = joblib.load(path)
    required = {"base_models", "meta_reg", "meta_cls", "feature_columns", "training_source", "residual_q05", "residual_q95"}
    missing = required.difference(bundle.keys())
    if missing:
        raise ValueError(f"Frozen model bundle is incomplete; missing: {sorted(missing)}")
    return bundle


def predict_frozen(bundle: dict, recent_prices: pd.DataFrame) -> Dict[str, float]:
    row = latest_feature_row(recent_prices, bundle["feature_columns"])
    X = row[bundle["feature_columns"]]
    base_preds = []
    for name, model in bundle["base_models"].items():
        base_preds.append(float(model.predict(X)[0]))
    meta_X = np.asarray(base_preds, dtype=float).reshape(1, -1)
    gain = float(bundle["meta_reg"].predict(meta_X)[0])
    if hasattr(bundle["meta_cls"], "predict_proba"):
        prob_up = float(bundle["meta_cls"].predict_proba(meta_X)[0, 1])
    else:
        prob_up = 0.5
    current = float(row["Close"].iloc[0])
    pred_close = current + gain
    pred_return = gain / current if current else np.nan
    confidence = prob_up if gain >= 0 else 1.0 - prob_up
    q05 = float(bundle.get("residual_q05", 0.0))
    q95 = float(bundle.get("residual_q95", 0.0))
    low_close = current + gain + q05
    high_close = current + gain + q95
    if low_close > high_close:
        low_close, high_close = high_close, low_close
    return {
        "runtime": pd.Timestamp(row["Date"].iloc[0]),
        "current_close": current,
        "forecast_close": pred_close,
        "forecast_gain": gain,
        "forecast_return": pred_return,
        "prob_up": prob_up,
        "direction_confidence": confidence,
        "interval_low": low_close,
        "interval_high": high_close,
        "realized_vol20": float(row["vol_20"].iloc[0]) * 100.0,
        "base_preds": dict(zip(bundle["base_models"].keys(), base_preds)),
        "training_source": str(bundle.get("training_source", "Non-FRED historical data")),
        "training_period": str(bundle.get("training_period", "")),
        "model_created": str(bundle.get("model_created", "")),
    }
