"""Refresh the repository's FRED snapshot for fast Streamlit deployment.

This script is designed for GitHub Actions.  It updates the core deployment
series in market_daily_data_2015_to_present.csv and leaves the broader frozen
research columns intact.  A FRED API key is preferred but the public CSV route
is retained as a fallback.
"""
from __future__ import annotations

import os
from pathlib import Path

import pandas as pd

from config import DATA_PATH, FRED_SERIES, TARGET_SERIES
from data_utils import fetch_recent_fred_panel


def main() -> None:
    path = Path(DATA_PATH)
    old = pd.read_csv(path, parse_dates=["Date"]).sort_values("Date")
    old = old.drop_duplicates("Date", keep="last").reset_index(drop=True)

    last_date = pd.to_datetime(old["Date"]).max()
    # Re-fetch a short overlap so revisions / late releases are incorporated.
    start_date = (last_date - pd.Timedelta(days=14)).strftime("%Y-%m-%d")
    end_date = pd.Timestamp.today().normalize().strftime("%Y-%m-%d")
    api_key = os.getenv("FRED_API_KEY", "").strip() or None

    fresh = fetch_recent_fred_panel(start_date, end_date, api_key=api_key)

    # Preserve all research columns.  Fresh values replace overlapping core cells.
    idx = old.set_index("Date")
    fresh_idx = fresh.set_index("Date")
    all_dates = idx.index.union(fresh_idx.index).sort_values()
    out = idx.reindex(all_dates)
    for col in FRED_SERIES:
        if col not in out.columns:
            out[col] = pd.NA
        if col in fresh_idx.columns:
            out.loc[fresh_idx.index, col] = fresh_idx[col]

    out = out.reset_index().rename(columns={"index": "Date"})
    out["Date"] = pd.to_datetime(out["Date"])
    out = out[out["Date"].dt.dayofweek < 5].copy()

    # Keep actual equity-index sessions. Macro series are aligned to these sessions
    # and carried forward from the latest released value.
    target_cols = [c for c in TARGET_SERIES if c in out.columns]
    out = out[out[target_cols].notna().any(axis=1)].copy()

    non_targets = [c for c in FRED_SERIES if c not in TARGET_SERIES and c in out.columns]
    out[non_targets] = out[non_targets].ffill().bfill()

    out = out.sort_values("Date").drop_duplicates("Date", keep="last").reset_index(drop=True)
    out.to_csv(path, index=False, date_format="%Y-%m-%d")

    latest = out.loc[out[target_cols].notna().any(axis=1), "Date"].max()
    print(f"Updated {path.name}: {len(out):,} rows; latest market date {latest.date()}")


if __name__ == "__main__":
    main()
