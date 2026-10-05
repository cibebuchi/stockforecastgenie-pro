import os
import re
import math
import warnings
import argparse
from typing import List, Dict, Tuple, Any

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge, HuberRegressor, LogisticRegression
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor
from sklearn.model_selection import train_test_split

import xgboost as xgb

warnings.filterwarnings("ignore")

# ============================================================
# DEFAULTS
# ============================================================
CSV_IN = r"C:\Users\chied\Downloads\market_daily_data_2015_to_present.csv"
OUT_DIR = r"C:\Users\chied\Downloads\STOCK_STACK_PLUS_OUTPUTS"
TARGETS = ("SP500",)
TRAIN_MONTHS = [4]
LEAD_TIMES = [1]
BUSINESS_DAYS_ONLY = True
BASE_MODELS = ["et", "rf", "xgb_native"]
FEATURE_SET = "core"  # core or full
TOP_K_FEATURES = 40
SEED = 42
MIN_TRAIN_ROWS_FLOOR = 20
VAL_TAIL_ROWS = 63
TIME_FEATURES = {
    "is_monday", "is_friday", "is_month_start", "is_month_end",
    "dow_sin", "dow_cos", "month_sin", "month_cos", "doy_sin", "doy_cos"
}
EXOG_COLS = (
    "UNRATE", "FEDFUNDS", "INDPRO", "CSUSHPINSA", "UMCSENT",
    "DCOILWTICO", "M2SL", "DGS10", "CPIAUCSL", "VIXCLS"
)
CORE_EXOGS = {"VIXCLS", "DGS10", "DCOILWTICO", "FEDFUNDS", "UNRATE"}
VALID_FEATURE_SETS = {"core", "full"}

XGB_PARAMS = {
    "objective": "reg:squarederror",
    "eval_metric": "mae",
    "tree_method": "hist",
    "eta": 0.04,
    "max_depth": 4,
    "subsample": 0.9,
    "colsample_bytree": 0.9,
    "lambda": 1.0,
    "alpha": 0.0,
    "min_child_weight": 1.0,
    "gamma": 0.0,
    "seed": SEED,
    "verbosity": 0,
}
NUM_BOOST_ROUND = 800
EARLY_STOPPING_ROUNDS = 50

np.random.seed(SEED)


# ============================================================
# METRICS
# ============================================================
def rmse(y_true, y_pred) -> float:
    return float(math.sqrt(mean_squared_error(y_true, y_pred)))


def mape(y_true, y_pred, eps: float = 1e-8) -> float:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    denom = np.maximum(np.abs(y_true), eps)
    return float(np.mean(np.abs((y_true - y_pred) / denom)) * 100.0)


def directional_accuracy(y_true_gain, y_pred_gain) -> float:
    yt = np.sign(np.asarray(y_true_gain, dtype=float))
    yp = np.sign(np.asarray(y_pred_gain, dtype=float))
    return float(np.mean(yt == yp))


def return_directional_accuracy(y_true_return, y_pred_return) -> float:
    yt = np.sign(np.asarray(y_true_return, dtype=float))
    yp = np.sign(np.asarray(y_pred_return, dtype=float))
    return float(np.mean(yt == yp))


def down_recall(y_true_gain, y_pred_gain) -> float:
    yt = np.asarray(y_true_gain, dtype=float)
    yp = np.asarray(y_pred_gain, dtype=float)
    mask = yt < 0
    if mask.sum() == 0:
        return np.nan
    return float(np.mean(yp[mask] < 0))


# ============================================================
# HELPERS
# ============================================================
def parse_int_list(values: List[str]) -> List[int]:
    out: List[int] = []
    for v in values:
        for part in str(v).split(","):
            part = part.strip()
            if part:
                out.append(int(part))
    return out


def parse_str_list(values: List[str]) -> List[str]:
    out: List[str] = []
    for v in values:
        for part in str(v).split(","):
            part = part.strip()
            if part:
                out.append(part)
    return out


def nan_report_quick(df: pd.DataFrame, name: str):
    n = int(df.isna().sum().sum())
    print(f"[NaN] {name}: total NaNs = {n:,} {'✅' if n == 0 else '❌'}")


def lag_cap_for_train_months(train_months: int) -> int:
    if train_months <= 1:
        return 10
    if train_months <= 2:
        return 15
    if train_months <= 3:
        return 21
    if train_months <= 6:
        return 42
    return 63


def dynamic_min_train_rows(train_months: int) -> int:
    cap = lag_cap_for_train_months(train_months)
    return max(MIN_TRAIN_ROWS_FLOOR, cap + 5)


# ============================================================
# LOAD + CLEAN
# ============================================================
def load_data(path: str, business_days_only: bool = True) -> pd.DataFrame:
    df = pd.read_csv(path, parse_dates=["Date"])
    df = df.sort_values("Date").drop_duplicates("Date", keep="last").reset_index(drop=True)

    for c in df.columns:
        if c != "Date":
            df[c] = pd.to_numeric(df[c], errors="coerce")

    if business_days_only:
        df = df[df["Date"].dt.dayofweek < 5].copy().reset_index(drop=True)

    slow_cols = [c for c in EXOG_COLS if c in df.columns]
    if slow_cols:
        df[slow_cols] = df[slow_cols].ffill().bfill()

    nan_report_quick(df, "LOADED DATA")
    return df


# ============================================================
# FEATURE ENGINEERING
# ============================================================
def add_time_features(df: pd.DataFrame) -> pd.DataFrame:
    d = df.copy()
    t = pd.to_datetime(d["Date"])
    dow = t.dt.dayofweek
    month = t.dt.month
    doy = t.dt.dayofyear
    d["is_monday"] = (dow == 0).astype(int)
    d["is_friday"] = (dow == 4).astype(int)
    d["is_month_start"] = t.dt.is_month_start.astype(int)
    d["is_month_end"] = t.dt.is_month_end.astype(int)
    d["dow_sin"] = np.sin(2 * np.pi * dow / 7.0)
    d["dow_cos"] = np.cos(2 * np.pi * dow / 7.0)
    d["month_sin"] = np.sin(2 * np.pi * month / 12.0)
    d["month_cos"] = np.cos(2 * np.pi * month / 12.0)
    d["doy_sin"] = np.sin(2 * np.pi * doy / 365.25)
    d["doy_cos"] = np.cos(2 * np.pi * doy / 365.25)
    return d


def add_target_features(df: pd.DataFrame, target: str) -> pd.DataFrame:
    d = df.copy()
    d[f"{target}_ret_1d"] = d[target].pct_change(1)
    d[f"{target}_logret_1d"] = np.log(d[target]).diff(1)
    d[f"{target}_gain_1d"] = d[target].diff(1)

    lags = [1, 2, 3, 5, 10, 21, 42, 63]
    wins = [5, 10, 21, 42, 63]
    for L in lags:
        d[f"{target}_lag_{L}"] = d[target].shift(L)
        d[f"{target}_ret_lag_{L}"] = d[f"{target}_ret_1d"].shift(L)
        d[f"{target}_gain_lag_{L}"] = d[f"{target}_gain_1d"].shift(L)
    for W in wins:
        d[f"{target}_ret_rollmean_{W}"] = d[f"{target}_ret_1d"].rolling(W).mean()
        d[f"{target}_ret_rollstd_{W}"] = d[f"{target}_ret_1d"].rolling(W).std()
        d[f"{target}_gain_rollmean_{W}"] = d[f"{target}_gain_1d"].rolling(W).mean()
        d[f"{target}_gain_rollstd_{W}"] = d[f"{target}_gain_1d"].rolling(W).std()
        d[f"{target}_mom_{W}"] = d[target] / d[target].shift(W) - 1.0
        d[f"{target}_ma_gap_{W}"] = d[target] / d[target].rolling(W).mean() - 1.0
        d[f"{target}_rollmin_{W}"] = d[target].rolling(W).min()
        d[f"{target}_rollmax_{W}"] = d[target].rolling(W).max()
    d[f"{target}_drawdown_63"] = d[target] / d[target].rolling(63).max() - 1.0
    return d


def add_cross_target_features(df: pd.DataFrame, target: str) -> pd.DataFrame:
    d = df.copy()
    other = "SP500" if target == "DJIA" else "DJIA"
    if other not in d.columns:
        return d
    d[f"{other}_ret_1d"] = d[other].pct_change(1)
    d[f"{other}_gain_1d"] = d[other].diff(1)
    for L in [1, 2, 3, 5, 10, 21, 42, 63]:
        d[f"{other}_lag_{L}"] = d[other].shift(L)
        d[f"{other}_ret_lag_{L}"] = d[f"{other}_ret_1d"].shift(L)
        d[f"{other}_gain_lag_{L}"] = d[f"{other}_gain_1d"].shift(L)
    for W in [5, 10, 21, 42, 63]:
        d[f"{other}_ret_rollmean_{W}"] = d[f"{other}_ret_1d"].rolling(W).mean()
        d[f"{other}_ret_rollstd_{W}"] = d[f"{other}_ret_1d"].rolling(W).std()
        d[f"{other}_mom_{W}"] = d[other] / d[other].shift(W) - 1.0
    return d


def add_exog_features(df: pd.DataFrame) -> pd.DataFrame:
    d = df.copy()
    for c in EXOG_COLS:
        if c not in d.columns:
            continue
        d[f"{c}_current"] = d[c]
        for L in [1, 2, 5, 10, 21, 42, 63]:
            d[f"{c}_lag_{L}"] = d[c].shift(L)
        d[f"{c}_chg_1"] = d[c].diff(1)
        d[f"{c}_chg_5"] = d[c].diff(5)
        d[f"{c}_pctchg_21"] = d[c].pct_change(21)
        for W in [5, 10, 21, 42, 63]:
            d[f"{c}_rollmean_{W}"] = d[c].rolling(W).mean()
            d[f"{c}_rollstd_{W}"] = d[c].rolling(W).std()
    return d


def build_supervised(raw: pd.DataFrame, target: str, lead: int) -> Tuple[pd.DataFrame, List[str]]:
    d = raw.copy()
    d = add_target_features(d, target)
    d = add_cross_target_features(d, target)
    d = add_exog_features(d)
    d = add_time_features(d)

    d["current_price"] = d[target]
    d["target_price"] = d[target].shift(-lead)
    d["target_gain"] = d["target_price"] - d["current_price"]
    d["target_return"] = d["target_price"] / d["current_price"] - 1.0
    d["target_up"] = (d["target_gain"] > 0).astype(float)
    d["run_day"] = d["Date"]
    d["target_day"] = d["Date"].shift(-lead)
    d[f"naive_last_h_gain"] = d["current_price"] - d["current_price"].shift(lead)

    feature_cols = [
        c for c in d.columns
        if c not in {
            "Date", "run_day", "target_day",
            "target_price", "target_gain", "target_return", "target_up"
        }
    ]

    d = d.dropna(subset=["target_price", "target_gain", "target_return", "target_up"]).reset_index(drop=True)
    return d, feature_cols


# ============================================================
# FEATURE FILTERING / SELECTION (TRAIN-ONLY)
# ============================================================
def extract_last_numeric_token(col: str):
    m = re.search(r"_(\d+)(?:d)?$", col)
    return int(m.group(1)) if m else None


def feature_allowed_by_cap(col: str, lag_cap: int) -> bool:
    if col in {"current_price", "lead_days", "naive_last_h_gain"}:
        return True
    n = extract_last_numeric_token(col)
    if n is None:
        return True
    return n <= lag_cap


def core_feature_allowed(col: str, target: str) -> bool:
    other = "SP500" if target == "DJIA" else "DJIA"
    if col in TIME_FEATURES or col == "current_price" or col == "naive_last_h_gain":
        return True
    if col.startswith(f"{target}_") or col.startswith(f"{other}_"):
        return True
    for ex in CORE_EXOGS:
        if col == f"{ex}_current":
            return True
        if ex in {"VIXCLS", "DGS10", "DCOILWTICO"} and col.startswith(f"{ex}_"):
            return True
    return False


def filter_feature_candidates(feature_cols: List[str], target: str, train_months: int, feature_set: str) -> List[str]:
    lag_cap = lag_cap_for_train_months(train_months)
    cols = [c for c in feature_cols if feature_allowed_by_cap(c, lag_cap)]
    if feature_set == "core":
        cols = [c for c in cols if core_feature_allowed(c, target)]
    return cols


def default_always_keep(target: str, candidates: List[str]) -> List[str]:
    other = "SP500" if target == "DJIA" else "DJIA"
    must = [
        "current_price",
        f"{target}_lag_1",
        f"{target}_ret_lag_1",
        f"{target}_gain_lag_1",
        f"{target}_ret_rollstd_5",
        f"{target}_ret_rollstd_10",
        f"{target}_mom_5",
        f"{target}_ma_gap_5",
        f"{other}_ret_lag_1",
        f"{other}_gain_lag_1",
        "VIXCLS_current",
        "VIXCLS_chg_1",
        "DGS10_current",
        "DGS10_chg_1",
        "DCOILWTICO_current",
        "DCOILWTICO_chg_1",
        "FEDFUNDS_current",
        "UNRATE_current",
    ]
    keep = [c for c in must if c in candidates]
    keep += [c for c in TIME_FEATURES if c in candidates]
    return sorted(set(keep))


def select_features_train_only(
    Xtr: pd.DataFrame,
    ytr: np.ndarray,
    candidate_cols: List[str],
    target: str,
    top_k: int,
) -> List[str]:
    if len(candidate_cols) <= top_k:
        return candidate_cols

    always_keep = default_always_keep(target, candidate_cols)
    remaining = [c for c in candidate_cols if c not in always_keep]
    if not remaining:
        return always_keep[:top_k]

    X0 = Xtr[candidate_cols].copy()
    med = X0.median(numeric_only=True)
    X0 = X0.fillna(med)
    y0 = np.asarray(ytr, dtype=float)

    scores = []
    for c in remaining:
        s = pd.to_numeric(X0[c], errors="coerce").fillna(med.get(c, 0.0)).to_numpy(dtype=float)
        if np.nanstd(s) < 1e-12:
            score = 0.0
        else:
            cc = np.corrcoef(s, y0)[0, 1]
            score = 0.0 if np.isnan(cc) else abs(float(cc))
        scores.append((c, score))

    scores = sorted(scores, key=lambda x: x[1], reverse=True)
    slots = max(0, top_k - len(always_keep))
    picked = [c for c, _ in scores[:slots]]
    selected = always_keep + picked
    return [c for c in candidate_cols if c in selected]


# ============================================================
# MODEL FACTORIES
# ============================================================
def _reg_pipeline(model) -> Pipeline:
    return Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("model", model),
    ])


def _scaled_reg_pipeline(model) -> Pipeline:
    return Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        ("model", model),
    ])


def make_reg_model(name: str):
    name = name.lower().strip()
    if name == "et":
        return _reg_pipeline(ExtraTreesRegressor(
            n_estimators=300, random_state=SEED, n_jobs=-1, min_samples_leaf=2
        ))
    if name == "rf":
        return _reg_pipeline(RandomForestRegressor(
            n_estimators=250, random_state=SEED, n_jobs=-1, min_samples_leaf=2
        ))
    if name == "ridge":
        return _scaled_reg_pipeline(Ridge(alpha=2.0, random_state=SEED))
    if name == "huber":
        return _scaled_reg_pipeline(HuberRegressor(epsilon=1.35, alpha=1e-4, max_iter=3000))
    raise ValueError(f"Unknown reg model: {name}")


def xgb_native_fit_predict(Xtr: np.ndarray, ytr: np.ndarray, Xte: np.ndarray) -> Tuple[np.ndarray, int]:
    n = len(ytr)
    val_rows = min(max(20, n // 5), VAL_TAIL_ROWS)
    val_rows = min(val_rows, max(10, n - 10))

    if n <= 30:
        dtr = xgb.DMatrix(Xtr, label=ytr, missing=np.nan)
        dte = xgb.DMatrix(Xte, missing=np.nan)
        booster = xgb.train(
            params=XGB_PARAMS,
            dtrain=dtr,
            num_boost_round=min(300, NUM_BOOST_ROUND),
            verbose_eval=False,
        )
        return booster.predict(dte), min(300, NUM_BOOST_ROUND) - 1

    Xtr0, ytr0 = Xtr[:-val_rows], ytr[:-val_rows]
    Xva, yva = Xtr[-val_rows:], ytr[-val_rows:]
    dtr = xgb.DMatrix(Xtr0, label=ytr0, missing=np.nan)
    dva = xgb.DMatrix(Xva, label=yva, missing=np.nan)
    dte = xgb.DMatrix(Xte, missing=np.nan)
    booster = xgb.train(
        params=XGB_PARAMS,
        dtrain=dtr,
        num_boost_round=NUM_BOOST_ROUND,
        evals=[(dva, "val")],
        early_stopping_rounds=EARLY_STOPPING_ROUNDS,
        verbose_eval=False,
    )
    best_it = getattr(booster, "best_iteration", None)
    if best_it is None:
        best_it = NUM_BOOST_ROUND - 1
    yhat = booster.predict(dte, iteration_range=(0, int(best_it) + 1))
    return yhat, int(best_it)


def fit_predict_reg(name: str, Xtr: pd.DataFrame, ytr: np.ndarray, Xte: pd.DataFrame) -> Tuple[np.ndarray, Any]:
    name = name.lower().strip()
    if name == "xgb_native":
        return xgb_native_fit_predict(Xtr.to_numpy(dtype=float), ytr, Xte.to_numpy(dtype=float))
    mdl = make_reg_model(name)
    mdl.fit(Xtr, ytr)
    return mdl.predict(Xte), "ok"


# ============================================================
# META MODELS
# ============================================================
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
        ("model", LogisticRegression(C=1.0, random_state=SEED, max_iter=2000, class_weight="balanced")),
    ])


# ============================================================
# OPERATIONAL WINDOWS
# ============================================================
def get_train_idx(run_days: pd.Series, current_day: pd.Timestamp, train_months: int) -> np.ndarray:
    start_day = pd.Timestamp(current_day) - pd.DateOffset(months=int(train_months))
    mask = (run_days >= start_day) & (run_days < pd.Timestamp(current_day))
    return np.where(mask.to_numpy())[0]


# ============================================================
# BASE REGRESSION PREDICTIONS
# ============================================================
def run_base_regression_panel(
    full: pd.DataFrame,
    feature_cols: List[str],
    target_name: str,
    lead: int,
    train_months: int,
    base_models: List[str],
    feature_set: str,
    top_k_features: int,
    out_dir: str,
) -> str:
    d = full.sort_values("run_day").reset_index(drop=True).copy()
    run_days = pd.to_datetime(d["run_day"])
    unique_days = list(run_days.sort_values().unique())

    X_all = d[feature_cols].copy()
    y_all = d["target_gain"].to_numpy(dtype=float)
    min_rows = dynamic_min_train_rows(train_months)

    rows = []
    candidate_cols = filter_feature_candidates(feature_cols, target_name, train_months, feature_set)

    for day in unique_days:
        test_idx = np.where(run_days.to_numpy() == np.datetime64(day))[0]
        if len(test_idx) != 1:
            continue

        train_idx = get_train_idx(run_days, pd.Timestamp(day), train_months)
        if len(train_idx) < min_rows:
            continue

        Xtr_raw = X_all.iloc[train_idx]
        ytr = y_all[train_idx]
        Xte_raw = X_all.iloc[test_idx]

        selected_cols = select_features_train_only(
            Xtr=Xtr_raw,
            ytr=ytr,
            candidate_cols=candidate_cols,
            target=target_name,
            top_k=top_k_features,
        )

        Xtr = Xtr_raw[selected_cols]
        Xte = Xte_raw[selected_cols]

        row = {
            "target": target_name,
            "lead_days": int(lead),
            "train_months": int(train_months),
            "run_day": pd.Timestamp(day),
            "target_day": pd.Timestamp(d.loc[test_idx[0], "target_day"]),
            "current_price": float(d.loc[test_idx[0], "current_price"]),
            "y_true_price": float(d.loc[test_idx[0], "target_price"]),
            "y_true_gain": float(d.loc[test_idx[0], "target_gain"]),
            "y_true_return": float(d.loc[test_idx[0], "target_return"]),
            "y_true_up": int(d.loc[test_idx[0], "target_up"]),
            "naive_last_h_gain": float(d.loc[test_idx[0], "naive_last_h_gain"])
            if pd.notna(d.loc[test_idx[0], "naive_last_h_gain"]) else np.nan,
            "selected_feature_count": int(len(selected_cols)),
            "selected_features": "|".join(selected_cols),
        }

        for m in base_models:
            yhat, meta = fit_predict_reg(m, Xtr, ytr, Xte)
            row[f"pred_gain_{m}"] = float(yhat[0])
            row[f"pred_price_{m}"] = row["current_price"] + float(yhat[0])
            row[f"pred_return_{m}"] = float(yhat[0]) / row["current_price"] if row["current_price"] != 0 else np.nan
            row[f"meta_{m}"] = meta

        rows.append(row)

    if not rows:
        raise RuntimeError(
            f"No base predictions generated for target={target_name}, lead={lead}, train_months={train_months}."
        )

    out = pd.DataFrame(rows).sort_values("run_day").reset_index(drop=True)
    tag = f"{target_name}__lead{lead}d__train{train_months}m__stackplus_base_panel"
    path = os.path.join(out_dir, f"{tag}.csv")
    out.to_csv(path, index=False)
    print(f"Saved base panel: {path}")
    return path


# ============================================================
# THRESHOLD TUNING (TRAIN-ONLY)
# ============================================================
def tune_direction_threshold(y_true_gain_val: np.ndarray, prob_up_val: np.ndarray) -> Tuple[float, pd.DataFrame]:
    y_true_gain_val = np.asarray(y_true_gain_val, dtype=float)
    prob_up_val = np.asarray(prob_up_val, dtype=float)

    grid = np.concatenate([
        np.linspace(0.35, 0.65, 13),
        np.array([0.50])
    ])
    grid = np.unique(np.round(grid, 4))

    rows = []
    for thr in grid:
        pred_gain_sign = np.where(prob_up_val >= thr, 1.0, -1.0)
        da = directional_accuracy(y_true_gain_val, pred_gain_sign)
        dr = down_recall(y_true_gain_val, pred_gain_sign)
        rows.append({"threshold": float(thr), "directional_accuracy": da, "down_recall": dr})

    tab = pd.DataFrame(rows).sort_values(["directional_accuracy", "down_recall", "threshold"], ascending=[False, False, True])
    best_thr = float(tab.iloc[0]["threshold"])
    return best_thr, tab


# ============================================================
# STACKING (NO LEAKAGE)
# ============================================================
def run_stack_plus_from_base(
    base_path: str,
    target_name: str,
    lead: int,
    train_months: int,
    base_models: List[str],
    out_dir: str,
) -> Tuple[str, str]:
    base = pd.read_csv(base_path, parse_dates=["run_day", "target_day"]).sort_values("run_day").reset_index(drop=True)
    run_days = pd.to_datetime(base["run_day"])
    unique_days = list(run_days.sort_values().unique())

    meta_feats = [f"pred_gain_{m}" for m in base_models]
    y_reg_all = base["y_true_gain"].to_numpy(dtype=float)
    y_cls_all = base["y_true_up"].to_numpy(dtype=int)
    min_rows = dynamic_min_train_rows(train_months)

    preds_rows = []
    fold_rows = []
    thr_rows = []

    for day in unique_days:
        test_idx = np.where(run_days.to_numpy() == np.datetime64(day))[0]
        if len(test_idx) != 1:
            continue

        train_idx = get_train_idx(run_days, pd.Timestamp(day), train_months)
        if len(train_idx) < min_rows:
            continue

        Xtr_full = base.iloc[train_idx][meta_feats]
        ytr_reg = y_reg_all[train_idx]
        ytr_cls = y_cls_all[train_idx]
        Xte = base.iloc[test_idx][meta_feats]

        val_rows = min(max(20, len(train_idx) // 5), VAL_TAIL_ROWS)
        val_rows = min(val_rows, max(10, len(train_idx) - 10))
        inner_tr_idx = train_idx[:-val_rows] if len(train_idx) > (val_rows + 5) else train_idx
        inner_va_idx = train_idx[-val_rows:] if len(train_idx) > (val_rows + 5) else train_idx

        Xtr_inner = base.iloc[inner_tr_idx][meta_feats]
        ytr_reg_inner = y_reg_all[inner_tr_idx]
        ytr_cls_inner = y_cls_all[inner_tr_idx]
        Xva = base.iloc[inner_va_idx][meta_feats]
        yva_gain = y_reg_all[inner_va_idx]

        reg_inner = make_meta_reg()
        reg_inner.fit(Xtr_inner, ytr_reg_inner)
        cls_inner = make_meta_cls()
        cls_inner.fit(Xtr_inner, ytr_cls_inner)
        prob_up_val = cls_inner.predict_proba(Xva)[:, 1]
        best_thr, thr_tab = tune_direction_threshold(yva_gain, prob_up_val)

        reg = make_meta_reg()
        reg.fit(Xtr_full, ytr_reg)
        cls = make_meta_cls()
        cls.fit(Xtr_full, ytr_cls)

        pred_gain_raw = float(reg.predict(Xte)[0])
        prob_up = float(cls.predict_proba(Xte)[:, 1][0])
        pred_sign = 1.0 if prob_up >= best_thr else -1.0
        pred_gain_final = pred_sign * abs(pred_gain_raw)

        current_price = float(base.iloc[test_idx[0]]["current_price"])
        true_price = float(base.iloc[test_idx[0]]["y_true_price"])
        true_gain = float(base.iloc[test_idx[0]]["y_true_gain"])
        true_return = float(base.iloc[test_idx[0]]["y_true_return"])

        pred_price_final = current_price + pred_gain_final
        pred_return_raw = pred_gain_raw / current_price if current_price != 0 else np.nan
        pred_return_final = pred_gain_final / current_price if current_price != 0 else np.nan

        preds_rows.append({
            "target": target_name,
            "lead_days": int(lead),
            "objective": "gain",
            "model": f"stackplus__{'+'.join(base_models)}__meta_huber_logit",
            "run_day": pd.Timestamp(day),
            "target_day": pd.Timestamp(base.iloc[test_idx[0]]["target_day"]),
            "train_months": int(train_months),
            "current_price": current_price,
            "y_true_price": true_price,
            "y_true_gain": true_gain,
            "y_true_return": true_return,
            "y_pred_gain_raw": pred_gain_raw,
            "y_pred_return_raw": pred_return_raw,
            "prob_up": prob_up,
            "direction_threshold": best_thr,
            "pred_sign": pred_sign,
            "y_pred_gain": pred_gain_final,
            "y_pred_price": pred_price_final,
            "y_pred_return": pred_return_final,
            "naive_last_h_gain": float(base.iloc[test_idx[0]]["naive_last_h_gain"])
            if pd.notna(base.iloc[test_idx[0]]["naive_last_h_gain"]) else np.nan,
            **{f"base_pred_{m}": float(base.iloc[test_idx[0]][f"pred_gain_{m}"]) for m in base_models},
            **{f"base_pred_return_{m}": float(base.iloc[test_idx[0]][f"pred_return_{m}"]) for m in base_models},
        })

        fold_rows.append({
            "target": target_name,
            "lead_days": int(lead),
            "model": f"stackplus__{'+'.join(base_models)}__meta_huber_logit",
            "run_day": pd.Timestamp(day),
            "train_months": int(train_months),
            "MAE": float(abs(true_gain - pred_gain_final)),
            "RMSE": float(abs(true_gain - pred_gain_final)),
            "Price_AE": float(abs(true_price - pred_price_final)),
            "Return_AE": float(abs(true_return - pred_return_final)),
            "Return_SE": float((true_return - pred_return_final) ** 2) if pd.notna(pred_return_final) else np.nan,
            "Directional_Accuracy": float(np.sign(true_gain) == np.sign(pred_gain_final)),
            "Return_Directional_Accuracy": float(np.sign(true_return) == np.sign(pred_return_final)) if pd.notna(pred_return_final) else np.nan,
            "Down_Recall": float((pred_gain_final < 0)) if true_gain < 0 else np.nan,
            "Threshold": best_thr,
            "Prob_Up": prob_up,
        })

        thr_rows.append({
            "run_day": pd.Timestamp(day),
            "target": target_name,
            "lead_days": int(lead),
            "train_months": int(train_months),
            "selected_threshold": best_thr,
            "validation_rows": len(inner_va_idx),
        })

    if not preds_rows:
        raise RuntimeError(
            f"No stacked predictions generated for target={target_name}, lead={lead}, train_months={train_months}."
        )

    preds = pd.DataFrame(preds_rows).sort_values("run_day").reset_index(drop=True)
    folds = pd.DataFrame(fold_rows).sort_values("run_day").reset_index(drop=True)
    thrs = pd.DataFrame(thr_rows).sort_values("run_day").reset_index(drop=True)

    tag = f"{target_name}__lead{lead}d__train{train_months}m__gain__stackplus__{'+'.join(base_models)}"
    preds_path = os.path.join(out_dir, f"{tag}__preds.csv")
    folds_path = os.path.join(out_dir, f"{tag}__fold_metrics.csv")
    thrs_path = os.path.join(out_dir, f"{tag}__thresholds.csv")

    preds.to_csv(preds_path, index=False)
    folds.to_csv(folds_path, index=False)
    thrs.to_csv(thrs_path, index=False)

    print(f"Saved stack preds: {preds_path}")
    print(f"Saved stack folds: {folds_path}")
    print(f"Saved thresholds: {thrs_path}")
    return preds_path, folds_path


# ============================================================
# MATCHED NAIVE COMPARISON
# ============================================================
def build_naive_from_model_timeline(model_preds: pd.DataFrame, baseline_name: str) -> pd.DataFrame:
    d = model_preds.copy()
    d["model"] = baseline_name
    if baseline_name == "flat":
        d["y_pred_gain"] = 0.0
        d["y_pred_price"] = d["current_price"]
        d["y_pred_return"] = 0.0
    elif baseline_name == "last_h_gain":
        d["y_pred_gain"] = d["naive_last_h_gain"].fillna(0.0)
        d["y_pred_price"] = d["current_price"] + d["y_pred_gain"]
        d["y_pred_return"] = d["y_pred_gain"] / d["current_price"]
    elif baseline_name == "always_up":
        d["y_pred_gain"] = 1.0
        d["y_pred_price"] = d["current_price"] + d["y_pred_gain"]
        d["y_pred_return"] = d["y_pred_gain"] / d["current_price"]
    elif baseline_name == "always_down":
        d["y_pred_gain"] = -1.0
        d["y_pred_price"] = d["current_price"] + d["y_pred_gain"]
        d["y_pred_return"] = d["y_pred_gain"] / d["current_price"]
    else:
        raise ValueError(f"Unknown baseline: {baseline_name}")
    return d


def summarize_preds(d: pd.DataFrame) -> Dict[str, Any]:
    return {
        "target": d["target"].iloc[0],
        "lead_days": int(d["lead_days"].iloc[0]),
        "objective": "gain",
        "model": d["model"].iloc[0],
        "train_months": int(d["train_months"].iloc[0]),
        "n_forecasts": len(d),
        "price_MAE": float(mean_absolute_error(d["y_true_price"], d["y_pred_price"])),
        "price_RMSE": rmse(d["y_true_price"], d["y_pred_price"]),
        "price_MAPE": mape(d["y_true_price"], d["y_pred_price"]),
        "gain_MAE": float(mean_absolute_error(d["y_true_gain"], d["y_pred_gain"])),
        "gain_RMSE": rmse(d["y_true_gain"], d["y_pred_gain"]),
        "gain_Directional_Accuracy": directional_accuracy(d["y_true_gain"], d["y_pred_gain"]),
        "return_MAE": float(mean_absolute_error(d["y_true_return"], d["y_pred_return"])),
        "return_RMSE": rmse(d["y_true_return"], d["y_pred_return"]),
        "return_Directional_Accuracy": return_directional_accuracy(d["y_true_return"], d["y_pred_return"]),
        "down_recall": down_recall(d["y_true_gain"], d["y_pred_gain"]),
    }


def save_model_and_naive_compare(preds_path: str, out_dir: str):
    model_preds = pd.read_csv(preds_path, parse_dates=["run_day", "target_day"]).sort_values("run_day").reset_index(drop=True)
    rows = [summarize_preds(model_preds)]

    for baseline in ["flat", "last_h_gain", "always_up", "always_down"]:
        b = build_naive_from_model_timeline(model_preds, baseline)
        rows.append(summarize_preds(b))

    comp = pd.DataFrame(rows)
    comp["sort_primary"] = -comp["gain_Directional_Accuracy"].fillna(-999)
    comp["sort_secondary"] = -comp["return_Directional_Accuracy"].fillna(-999)
    comp["sort_tertiary"] = comp["price_MAE"]
    comp = comp.sort_values(["sort_primary", "sort_secondary", "sort_tertiary"]).drop(columns=["sort_primary", "sort_secondary", "sort_tertiary"]).reset_index(drop=True)

    compare_base = os.path.basename(preds_path).replace("__preds.csv", "__matched_compare.csv")
    out_path = os.path.join(out_dir, compare_base)
    comp.to_csv(out_path, index=False)

    print("\nMatched model-vs-naive comparison:")
    print(comp.to_string(index=False))
    print(f"Saved comparison: {out_path}")
    return out_path


# ============================================================
# MAIN
# ============================================================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv_in", type=str, default=CSV_IN)
    parser.add_argument("--out_dir", type=str, default=OUT_DIR)
    parser.add_argument("--targets", nargs="+", default=list(TARGETS))
    parser.add_argument("--train_months", nargs="+", default=[str(x) for x in TRAIN_MONTHS])
    parser.add_argument("--lead_times", nargs="+", default=[str(x) for x in LEAD_TIMES])
    parser.add_argument("--business_days_only", type=int, default=1)
    parser.add_argument("--base_models", nargs="+", default=BASE_MODELS)
    parser.add_argument("--feature_set", type=str, default=FEATURE_SET)
    parser.add_argument("--top_k_features", type=int, default=TOP_K_FEATURES)
    args = parser.parse_args()

    feature_set = str(args.feature_set).lower().strip()
    if feature_set not in VALID_FEATURE_SETS:
        raise ValueError(f"Unknown feature_set '{feature_set}'. Choose one of {sorted(VALID_FEATURE_SETS)}")

    targets = parse_str_list(args.targets)
    train_months_list = parse_int_list(args.train_months)
    lead_times = parse_int_list(args.lead_times)
    base_models = [m.lower().strip() for m in parse_str_list(args.base_models)]
    out_dir = args.out_dir
    os.makedirs(out_dir, exist_ok=True)

    print("CSV_IN:", args.csv_in)
    print("OUT_DIR:", out_dir)
    print("TARGETS:", targets)
    print("TRAIN_MONTHS:", train_months_list)
    print("LEAD_TIMES:", lead_times)
    print("BUSINESS_DAYS_ONLY:", bool(int(args.business_days_only)))
    print("BASE_MODELS:", base_models)
    print("FEATURE_SET:", feature_set)
    print("TOP_K_FEATURES:", args.top_k_features)

    raw = load_data(args.csv_in, business_days_only=bool(int(args.business_days_only)))
    print("Data start:", raw["Date"].min())
    print("Data end  :", raw["Date"].max())
    print("Rows      :", len(raw))

    all_outputs = []

    for target_name in targets:
        if target_name not in raw.columns:
            print(f"Skipping target '{target_name}' because it is not present in the data.")
            continue

        for lead in lead_times:
            full, feature_cols = build_supervised(raw, target_name, int(lead))
            nan_report_quick(full, f"ENGINEERED FULL ({target_name}, lead={lead})")

            for train_months in train_months_list:
                base_path = run_base_regression_panel(
                    full=full,
                    feature_cols=feature_cols,
                    target_name=target_name,
                    lead=int(lead),
                    train_months=int(train_months),
                    base_models=base_models,
                    feature_set=feature_set,
                    top_k_features=int(args.top_k_features),
                    out_dir=out_dir,
                )
                preds_path, folds_path = run_stack_plus_from_base(
                    base_path=base_path,
                    target_name=target_name,
                    lead=int(lead),
                    train_months=int(train_months),
                    base_models=base_models,
                    out_dir=out_dir,
                )
                compare_path = save_model_and_naive_compare(preds_path, out_dir)
                all_outputs.extend([base_path, preds_path, folds_path, compare_path])

    print("\nDONE ✅")
    print("Generated files:")
    for p in all_outputs:
        print(" -", p)


if __name__ == "__main__":
    main()
