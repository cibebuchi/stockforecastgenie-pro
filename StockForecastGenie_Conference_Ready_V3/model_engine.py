import importlib.util
from functools import lru_cache
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import HuberRegressor, LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from config import BASE_MODELS, FEATURE_SET, OBJECTIVE, STACK_SCRIPT_PATH, TOP_K, TRAIN_MONTHS


@lru_cache(maxsize=1)
def load_stack_module(stack_script_path: str = str(STACK_SCRIPT_PATH)):
    path = Path(stack_script_path)
    spec = importlib.util.spec_from_file_location("stockforecastgenie_runtime", str(path))
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load forecasting module from {path}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def to_outputs(pred_value: float, current_price: float, objective: str = OBJECTIVE) -> Dict[str, float]:
    if objective == "price":
        pred_price = float(pred_value)
        pred_gain = float(pred_price - current_price)
        pred_return = float(pred_gain / current_price) if current_price else np.nan
    elif objective == "gain":
        pred_gain = float(pred_value)
        pred_price = float(current_price + pred_gain)
        pred_return = float(pred_gain / current_price) if current_price else np.nan
    else:
        pred_return = float(pred_value)
        pred_price = float(current_price * (1.0 + pred_return))
        pred_gain = float(pred_price - current_price)
    return {"pred_price": pred_price, "pred_gain": pred_gain, "pred_return": pred_return}


def make_meta_reg() -> Pipeline:
    return Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        ("model", HuberRegressor(epsilon=1.35, alpha=1e-4, max_iter=3000)),
    ])


def make_meta_cls() -> Pipeline:
    return Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        ("model", LogisticRegression(max_iter=2000, class_weight="balanced", random_state=42)),
    ])


def _blocked_meta_split(train_df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    n = len(train_df)
    split = max(20, int(n * 0.8))
    split = min(split, n - 10) if n > 30 else max(10, n - 5)
    split = max(10, split)
    early, late = train_df.iloc[:split].copy(), train_df.iloc[split:].copy()
    if len(late) < 8:
        split = max(8, int(n * 0.7))
        early, late = train_df.iloc[:split].copy(), train_df.iloc[split:].copy()
    return early, late


def _pick_features(stack_mod, train_df: pd.DataFrame, feature_cols: List[str], target: str) -> List[str]:
    candidates = stack_mod.filter_feature_candidates(feature_cols, target, TRAIN_MONTHS, FEATURE_SET)
    y = train_df["target_gain"].to_numpy(dtype=float)
    return stack_mod.select_features_train_only(train_df[candidates], y, candidates, target, TOP_K)


def _base_predictions(stack_mod, train_df: pd.DataFrame, test_row: pd.DataFrame, cols: List[str]) -> Dict[str, float]:
    y = train_df["target_gain"].to_numpy(dtype=float)
    out = {}
    for model_name in BASE_MODELS:
        pred, _ = stack_mod.fit_predict_reg(model_name, train_df[cols], y, test_row[cols])
        out[model_name] = float(pred[0])
    return out


def _stack_prediction(stack_mod, train_df: pd.DataFrame, test_row: pd.DataFrame, cols: List[str], current_price: float) -> Dict:
    early, late = _blocked_meta_split(train_df)

    if len(early) < 15 or len(late) < 8:
        base_test = _base_predictions(stack_mod, train_df, test_row, cols)
        pred_gain = float(np.median(list(base_test.values())))
        outputs = to_outputs(pred_gain, current_price)
        base_returns = [to_outputs(v, current_price)["pred_return"] for v in base_test.values()]
        return {"pred_value": pred_gain, "prob_up": float(np.mean(np.asarray(base_returns) > 0)), "base_preds": base_test, **outputs}

    base_meta = {}
    for model_name in BASE_MODELS:
        pred_meta, _ = stack_mod.fit_predict_reg(
            model_name,
            early[cols],
            early["target_gain"].to_numpy(dtype=float),
            late[cols],
        )
        base_meta[model_name] = np.asarray(pred_meta, dtype=float)

    meta_x = pd.DataFrame(base_meta, index=late.index)
    meta_reg = make_meta_reg()
    meta_reg.fit(meta_x, late["target_gain"].to_numpy(dtype=float))

    base_test = _base_predictions(stack_mod, train_df, test_row, cols)
    test_meta = pd.DataFrame([{m: base_test[m] for m in BASE_MODELS}])
    pred_gain = float(meta_reg.predict(test_meta)[0])

    prob_up: Optional[float] = None
    dir_y = (late["target_gain"].to_numpy(dtype=float) > 0).astype(int)
    if len(np.unique(dir_y)) >= 2:
        meta_cls = make_meta_cls()
        meta_cls.fit(meta_x, dir_y)
        prob_up = float(meta_cls.predict_proba(test_meta)[0, 1])

    return {"pred_value": pred_gain, "prob_up": prob_up, "base_preds": base_test, **to_outputs(pred_gain, current_price)}


def _extract_vix(row: pd.DataFrame) -> float:
    for col in ["VIXCLS_current", "VIXCLS", "VIXCLS_lag_1"]:
        if col in row.columns and pd.notna(row[col].iloc[0]):
            return float(row[col].iloc[0])
    return np.nan


def _training_rows(supervised: pd.DataFrame, runtime: pd.Timestamp) -> pd.DataFrame:
    """Operationally valid training rows: labels must be known by the runtime cutoff."""
    start = runtime - pd.DateOffset(months=TRAIN_MONTHS)
    run_day = pd.to_datetime(supervised["run_day"])
    target_day = pd.to_datetime(supervised["target_day"])
    mask = (run_day >= start) & (run_day < runtime) & (target_day <= runtime)
    return supervised.loc[mask].sort_values("run_day").reset_index(drop=True)


def _reference_price(train_df: pd.DataFrame) -> float:
    if train_df.empty:
        return np.nan
    return float(train_df["current_price"].iloc[0])


def _naive(current_price: float, recent_gain: float) -> Dict[str, Dict[str, float]]:
    recent_gain = 0.0 if pd.isna(recent_gain) else float(recent_gain)
    return {
        "flat": {"pred_price": current_price, "pred_gain": 0.0, "pred_return": 0.0},
        "recent": {
            "pred_price": current_price + recent_gain,
            "pred_gain": recent_gain,
            "pred_return": recent_gain / current_price if current_price else np.nan,
        },
    }


def prepare_supervised(raw: pd.DataFrame, target: str, lead: int) -> Tuple[pd.DataFrame, List[str]]:
    stack_mod = load_stack_module()
    supervised, feature_cols = stack_mod.build_supervised(raw.copy(), target, int(lead))
    supervised["run_day"] = pd.to_datetime(supervised["run_day"])
    supervised["target_day"] = pd.to_datetime(supervised["target_day"])
    return supervised.sort_values("run_day").reset_index(drop=True), feature_cols


def historical_evaluation(raw: pd.DataFrame, target: str, lead: int, runtime: pd.Timestamp) -> Dict:
    stack_mod = load_stack_module()
    supervised, feature_cols = prepare_supervised(raw, target, lead)
    runtime = pd.Timestamp(runtime)
    test = supervised.loc[supervised["run_day"] == runtime].copy()
    if test.empty:
        raise RuntimeError("No evaluable row exists for that date. Choose another market date.")
    test = test.iloc[[0]].copy()
    train = _training_rows(supervised, runtime)
    min_rows = int(stack_mod.dynamic_min_train_rows(TRAIN_MONTHS))
    if len(train) < min_rows:
        raise RuntimeError(f"Not enough operationally available training rows. Have {len(train)}, need {min_rows}.")
    cols = _pick_features(stack_mod, train, feature_cols, target)
    current = float(test["current_price"].iloc[0])
    pred = _stack_prediction(stack_mod, train, test, cols, current)
    recent_gain = float(test["naive_last_h_gain"].iloc[0]) if "naive_last_h_gain" in test.columns else 0.0
    naive = _naive(current, recent_gain)
    return {
        "runtime": runtime,
        "target_day": pd.Timestamp(test["target_day"].iloc[0]),
        "current_price": current,
        "reference_price": _reference_price(train),
        "vix_value": _extract_vix(test),
        "recent_naive_return": naive["recent"]["pred_return"],
        "selected_feature_count": len(cols),
        "prediction": pred,
        "naive": naive,
        "actual_price": float(test["target_price"].iloc[0]),
        "actual_gain": float(test["target_gain"].iloc[0]),
        "actual_return": float(test["target_return"].iloc[0]),
    }


def _live_feature_frame(stack_mod, raw: pd.DataFrame, target: str, lead: int) -> pd.DataFrame:
    d = stack_mod.add_target_features(raw.copy(), target)
    d = stack_mod.add_cross_target_features(d, target)
    d = stack_mod.add_exog_features(d)
    d = stack_mod.add_time_features(d)
    d["current_price"] = d[target]
    d["run_day"] = pd.to_datetime(d["Date"])
    d["naive_last_h_gain"] = d["current_price"] - d["current_price"].shift(int(lead))
    return d.sort_values("run_day").reset_index(drop=True)


def _future_trading_day(raw: pd.DataFrame, runtime: pd.Timestamp, lead: int) -> pd.Timestamp:
    # Future exchange calendars are not part of the model inputs; use a weekday convention
    # and label it as an estimated target date in the UI.
    return pd.bdate_range(runtime + pd.offsets.BDay(1), periods=int(lead))[-1]


def live_forecast(raw_updated: pd.DataFrame, target: str, lead: int) -> Dict:
    stack_mod = load_stack_module()
    supervised, feature_cols = prepare_supervised(raw_updated, target, lead)
    live_frame = _live_feature_frame(stack_mod, raw_updated, target, lead)
    candidates = live_frame.loc[live_frame[target].notna()].copy()
    if candidates.empty:
        raise RuntimeError(f"No current {target} close is available in the refreshed data.")
    test = candidates.iloc[[-1]].copy()
    runtime = pd.Timestamp(test["run_day"].iloc[0])
    train = _training_rows(supervised, runtime)
    min_rows = int(stack_mod.dynamic_min_train_rows(TRAIN_MONTHS))
    if len(train) < min_rows:
        raise RuntimeError(f"Not enough recent labeled history for live training. Have {len(train)}, need {min_rows}.")
    cols = _pick_features(stack_mod, train, feature_cols, target)
    current = float(test["current_price"].iloc[0])
    pred = _stack_prediction(stack_mod, train, test, cols, current)
    recent_gain = float(test["naive_last_h_gain"].iloc[0]) if "naive_last_h_gain" in test.columns else 0.0
    naive = _naive(current, recent_gain)
    return {
        "runtime": runtime,
        "target_day": _future_trading_day(raw_updated, runtime, lead),
        "current_price": current,
        "reference_price": _reference_price(train),
        "vix_value": _extract_vix(test),
        "recent_naive_return": naive["recent"]["pred_return"],
        "selected_feature_count": len(cols),
        "prediction": pred,
        "naive": naive,
        "data_last_date": pd.Timestamp(raw_updated["Date"].max()),
    }
