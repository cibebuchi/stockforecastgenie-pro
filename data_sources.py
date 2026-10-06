from __future__ import annotations
from datetime import date, timedelta
import html
import urllib.parse
import xml.etree.ElementTree as ET
import pandas as pd
import requests

FRED_OBS_URL = "https://api.stlouisfed.org/fred/series/observations"
BLS_URL = "https://api.bls.gov/publicAPI/v2/timeseries/data/"


class DataSourceError(RuntimeError):
    pass


def fetch_fred_index_history(series_id: str, api_key: str, lookback_days: int = 540) -> pd.DataFrame:
    """Fetch FRED observations transiently for frozen-model inference.

    Deliberately no cache, no disk write, and no model fitting. Caller must supply the
    end user's own FRED API key for the current session.
    """
    if not str(api_key or "").strip():
        raise DataSourceError("A FRED API key is required for the live inference feature.")
    end = date.today()
    start = end - timedelta(days=int(lookback_days))
    params = {
        "series_id": series_id,
        "api_key": str(api_key).strip(),
        "file_type": "json",
        "observation_start": start.isoformat(),
        "observation_end": end.isoformat(),
        "sort_order": "asc",
    }
    try:
        r = requests.get(FRED_OBS_URL, params=params, timeout=18, headers={"User-Agent": "StockForecastGeniePro/1.0"})
        r.raise_for_status()
        payload = r.json()
    except Exception as exc:
        raise DataSourceError(f"FRED request failed: {exc}") from exc
    if isinstance(payload, dict) and payload.get("error_message"):
        raise DataSourceError(str(payload.get("error_message")))
    obs = payload.get("observations", []) if isinstance(payload, dict) else []
    rows = []
    for item in obs:
        val = item.get("value")
        if val in (None, "."):
            continue
        try:
            rows.append((pd.Timestamp(item["date"]), float(val)))
        except Exception:
            continue
    out = pd.DataFrame(rows, columns=["Date", "Close"])
    if len(out) < 80:
        raise DataSourceError(f"Only {len(out)} usable observations were returned for {series_id}.")
    return out.drop_duplicates("Date", keep="last").sort_values("Date").reset_index(drop=True)


def fetch_bls_context() -> dict:
    """Retrieve latest U.S. unemployment and CPI directly from BLS for context only."""
    try:
        payload = {"seriesid": ["LNS14000000", "CUSR0000SA0"]}
        r = requests.post(BLS_URL, json=payload, timeout=10, headers={"Content-Type": "application/json"})
        r.raise_for_status()
        data = r.json()
        result = {}
        for series in data.get("Results", {}).get("series", []):
            sid = series.get("seriesID")
            values = [x for x in series.get("data", []) if str(x.get("period", "")).startswith("M") and x.get("period") != "M13"]
            if not values:
                continue
            item = values[0]
            key = "unemployment_rate" if sid == "LNS14000000" else "cpi"
            result[key] = float(item["value"])
            result[key + "_period"] = f"{item.get('periodName', '')} {item.get('year', '')}".strip()
        return result
    except Exception:
        return {}


def fetch_google_news(query: str, limit: int = 10) -> list[dict]:
    q = urllib.parse.quote_plus(query)
    url = f"https://news.google.com/rss/search?q={q}&hl=en-US&gl=US&ceid=US:en"
    try:
        r = requests.get(url, timeout=10, headers={"User-Agent": "Mozilla/5.0"})
        r.raise_for_status()
        root = ET.fromstring(r.content)
    except Exception as exc:
        raise DataSourceError(f"News request failed: {exc}") from exc
    items = []
    for node in root.findall(".//item")[: int(limit)]:
        title = html.unescape(node.findtext("title") or "")
        link = node.findtext("link") or ""
        pub = node.findtext("pubDate") or ""
        source_node = node.find("source")
        source = source_node.text if source_node is not None else ""
        items.append({"title": title, "link": link, "pubDate": pub, "source": source})
    return items
