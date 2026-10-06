from __future__ import annotations
import numpy as np
import pandas as pd

FEATURE_COLUMNS = [
    "ret_1", "ret_2", "ret_3", "ret_5", "ret_10", "ret_20",
    "mom_5", "mom_10", "mom_20",
    "sma_gap_5", "sma_gap_10", "sma_gap_20", "sma_gap_50",
    "vol_5", "vol_10", "vol_20",
    "range_10", "drawdown_20", "rsi_14",
    "dow_sin", "dow_cos", "month_sin", "month_cos",
]


def _rsi(close: pd.Series, window: int = 14) -> pd.Series:
    delta = close.diff()
    gain = delta.clip(lower=0).rolling(window, min_periods=window).mean()
    loss = (-delta.clip(upper=0)).rolling(window, min_periods=window).mean()
    rs = gain / loss.replace(0, np.nan)
    out = 100 - (100 / (1 + rs))
    return out.fillna(50.0)


def build_features(price_df: pd.DataFrame) -> pd.DataFrame:
    """Create deterministic price-only features used by both offline training and live inference."""
    d = price_df[["Date", "Close"]].copy()
    d["Date"] = pd.to_datetime(d["Date"], errors="coerce").dt.tz_localize(None).dt.normalize()
    d["Close"] = pd.to_numeric(d["Close"], errors="coerce")
    d = d.dropna().drop_duplicates("Date", keep="last").sort_values("Date").reset_index(drop=True)
    c = d["Close"]
    r = c.pct_change()

    for n in [1, 2, 3, 5, 10, 20]:
        d[f"ret_{n}"] = c.pct_change(n)
    for n in [5, 10, 20]:
        d[f"mom_{n}"] = c / c.shift(n) - 1.0
    for n in [5, 10, 20, 50]:
        ma = c.rolling(n, min_periods=n).mean()
        d[f"sma_gap_{n}"] = c / ma - 1.0
    for n in [5, 10, 20]:
        d[f"vol_{n}"] = r.rolling(n, min_periods=n).std() * np.sqrt(252.0)

    hi10 = c.rolling(10, min_periods=10).max()
    lo10 = c.rolling(10, min_periods=10).min()
    d["range_10"] = (hi10 - lo10) / c.replace(0, np.nan)
    peak20 = c.rolling(20, min_periods=20).max()
    d["drawdown_20"] = c / peak20 - 1.0
    d["rsi_14"] = _rsi(c, 14) / 100.0

    dow = d["Date"].dt.dayofweek.astype(float)
    month = d["Date"].dt.month.astype(float)
    d["dow_sin"] = np.sin(2 * np.pi * dow / 5.0)
    d["dow_cos"] = np.cos(2 * np.pi * dow / 5.0)
    d["month_sin"] = np.sin(2 * np.pi * (month - 1.0) / 12.0)
    d["month_cos"] = np.cos(2 * np.pi * (month - 1.0) / 12.0)
    return d


def latest_feature_row(price_df: pd.DataFrame, feature_columns=None) -> pd.DataFrame:
    cols = list(feature_columns or FEATURE_COLUMNS)
    d = build_features(price_df)
    usable = d.dropna(subset=cols)
    if usable.empty:
        raise ValueError("Not enough recent observations to compute the frozen model feature vector.")
    return usable.iloc[[-1]].copy()


def make_supervised(price_df: pd.DataFrame, lead: int, feature_columns=None) -> pd.DataFrame:
    cols = list(feature_columns or FEATURE_COLUMNS)
    d = build_features(price_df)
    d["target_gain"] = d["Close"].shift(-int(lead)) - d["Close"]
    d["target_return"] = d["Close"].shift(-int(lead)) / d["Close"] - 1.0
    d["target_day"] = d["Date"].shift(-int(lead))
    return d.dropna(subset=cols + ["target_gain", "target_return", "target_day"]).reset_index(drop=True)
