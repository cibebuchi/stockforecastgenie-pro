from concurrent.futures import ThreadPoolExecutor, as_completed
from io import StringIO

import pandas as pd
import requests

from config import DATA_PATH, FRED_SERIES, TARGET_SERIES


def load_local_history() -> pd.DataFrame:
    """Load the frozen local research archive and keep weekday observations only."""
    df = pd.read_csv(DATA_PATH, parse_dates=["Date"])
    df = df.sort_values("Date").drop_duplicates("Date", keep="last").reset_index(drop=True)
    for col in df.columns:
        if col != "Date":
            df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df[df["Date"].dt.dayofweek < 5].copy().reset_index(drop=True)

    # FRED macro/context series have mixed reporting frequencies.  The production
    # snapshot is aligned to market sessions, so carry the most recently available
    # non-target observation forward exactly as the original research loader did.
    non_targets = [c for c in df.columns if c not in {"Date", *TARGET_SERIES}]
    if non_targets:
        df[non_targets] = df[non_targets].ffill().bfill()
    return df


def _clean_fred_frame(df: pd.DataFrame, series_id: str) -> pd.DataFrame:
    date_candidates = ["observation_date", "DATE", "Date", "date"]
    date_col = next((c for c in date_candidates if c in df.columns), None)
    if date_col is None or series_id not in df.columns:
        raise RuntimeError(f"Unexpected FRED response structure for {series_id}.")
    out = df[[date_col, series_id]].rename(columns={date_col: "Date"})
    out["Date"] = pd.to_datetime(out["Date"], errors="coerce")
    out[series_id] = pd.to_numeric(out[series_id], errors="coerce")
    return out.dropna(subset=["Date"]).sort_values("Date").drop_duplicates("Date", keep="last")


def fetch_fred_series(
    series_id: str,
    start_date: str,
    end_date: str,
    api_key: str | None = None,
    timeout: int = 8,
) -> pd.DataFrame:
    """Fetch one FRED series. Prefer the authenticated API; fall back to public graph CSV."""
    key = (api_key or "").strip()
    headers = {"User-Agent": "StockForecastGeniePro/ConferenceEdition"}

    if key:
        try:
            response = requests.get(
                "https://api.stlouisfed.org/fred/series/observations",
                params={
                    "series_id": series_id,
                    "api_key": key,
                    "file_type": "json",
                    "observation_start": start_date,
                    "observation_end": end_date,
                },
                headers=headers,
                timeout=timeout,
            )
            response.raise_for_status()
            observations = response.json().get("observations", [])
            df = pd.DataFrame(
                {
                    "Date": [row.get("date") for row in observations],
                    series_id: [row.get("value") for row in observations],
                }
            )
            if not df.empty:
                return _clean_fred_frame(df, series_id)
        except Exception:
            # Never expose a URL containing the API key. The public endpoint below is the safe fallback.
            pass

    try:
        response = requests.get(
            "https://fred.stlouisfed.org/graph/fredgraph.csv",
            params={"id": series_id, "cosd": start_date, "coed": end_date},
            headers=headers,
            timeout=timeout,
        )
        response.raise_for_status()
        df = pd.read_csv(StringIO(response.text), na_values=["."])
        return _clean_fred_frame(df, series_id)
    except Exception as exc:
        raise RuntimeError(f"FRED refresh failed for {series_id}.") from exc


def fetch_recent_fred_panel(start_date: str, end_date: str, api_key: str | None = None) -> pd.DataFrame:
    """Fetch required FRED series in parallel. A server-side API key is optional."""
    frames = {}
    errors = []
    with ThreadPoolExecutor(max_workers=6) as pool:
        futures = {pool.submit(fetch_fred_series, sid, start_date, end_date, api_key): sid for sid in FRED_SERIES}
        for future in as_completed(futures):
            sid = futures[future]
            try:
                frames[sid] = future.result()
            except Exception as exc:
                errors.append(f"{sid}: {exc}")

    # Market targets are essential for a live forecast. Other series can fall back to local history.
    missing_targets = [sid for sid in TARGET_SERIES if sid not in frames]
    if missing_targets:
        detail = "; ".join(errors[:3])
        raise RuntimeError(f"Could not refresh required market series: {', '.join(missing_targets)}. {detail}")

    panel = None
    for sid in FRED_SERIES:
        frame = frames.get(sid)
        if frame is None:
            continue
        panel = frame if panel is None else panel.merge(frame, on="Date", how="outer")
    if panel is None or panel.empty:
        raise RuntimeError("FRED refresh returned no observations.")
    for sid in FRED_SERIES:
        if sid not in panel.columns:
            panel[sid] = pd.NA
    return panel[["Date", *FRED_SERIES]].sort_values("Date").reset_index(drop=True)


def build_live_raw_panel(raw_history: pd.DataFrame, train_months: int = 6, extra_months_buffer: int = 4, api_key: str | None = None) -> pd.DataFrame:
    """Refresh recent data and return a trading-session panel for live inference."""
    history = raw_history.copy()
    history["Date"] = pd.to_datetime(history["Date"])

    end_date = pd.Timestamp.today().normalize()
    start_date = end_date - pd.DateOffset(months=max(train_months + extra_months_buffer, 10))

    hist = history.loc[history["Date"] >= start_date].copy()
    fresh = fetch_recent_fred_panel(start_date.strftime("%Y-%m-%d"), end_date.strftime("%Y-%m-%d"), api_key=api_key)

    merged = hist.merge(fresh, on="Date", how="outer", suffixes=("_hist", "_fresh"))
    for col in FRED_SERIES:
        h = f"{col}_hist"
        f = f"{col}_fresh"
        if f in merged.columns and h in merged.columns:
            merged[col] = merged[f].combine_first(merged[h])
        elif f in merged.columns:
            merged[col] = merged[f]
        elif h in merged.columns:
            merged[col] = merged[h]
        else:
            merged[col] = pd.NA

    merged = merged[["Date", *FRED_SERIES]].sort_values("Date").reset_index(drop=True)
    for c in FRED_SERIES:
        merged[c] = pd.to_numeric(merged[c], errors="coerce")

    # Keep only actual market sessions when at least one target index has a reported close.
    merged = merged[merged[list(TARGET_SERIES)].notna().any(axis=1)].copy()

    # Carry macro/context variables across trading sessions, but never fabricate index closes.
    non_targets = [c for c in FRED_SERIES if c not in TARGET_SERIES]
    merged[non_targets] = merged[non_targets].ffill().bfill()
    return merged.sort_values("Date").reset_index(drop=True)


def nearest_runtime(supervised_df: pd.DataFrame, runtime_date) -> pd.Timestamp:
    runtime_date = pd.Timestamp(runtime_date)
    run_days = pd.to_datetime(supervised_df["run_day"]).sort_values().drop_duplicates()
    eligible = run_days[run_days <= runtime_date]
    if len(eligible) == 0:
        return pd.Timestamp(run_days.iloc[0])
    return pd.Timestamp(eligible.iloc[-1])
