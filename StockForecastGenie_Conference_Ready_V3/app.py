import html
import os
import re
import xml.etree.ElementTree as ET
from urllib.parse import quote_plus

import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st
import requests

from config import (
    ALLOWED_LEADS,
    ALLOWED_TARGETS,
    APP_SUBTITLE,
    APP_TITLE,
    CONFERENCE_DATES,
    CONFERENCE_LOCATION,
    CONFERENCE_NAME,
    DEFAULT_LEAD,
    DEFAULT_TARGET,
    DOI_URL,
    LOGO_PATH,
    PAPER_URL,
    PUBLISHED_METRICS,
    TARGET_LABELS,
    TRAIN_MONTHS,
)
from data_utils import build_live_raw_panel, load_local_history, nearest_runtime
from model_engine import historical_evaluation, live_forecast, prepare_supervised
from signal_logic import classify_signal, market_context, signal_alignment

st.set_page_config(page_title=APP_TITLE, page_icon="📈", layout="wide", initial_sidebar_state="collapsed")

st.markdown(
    """
<style>
:root {
  --ink:#111827; --muted:#5B6472; --line:#E5E7EB; --paper:#FFFFFF;
  --soft:#F7F8FA; --brand:#F47B20; --brand2:#C85E0C; --navy:#172033;
  --green:#15803D; --red:#B91C1C; --amber:#9A6700;
}
.stApp { background: var(--soft); color: var(--ink); }
.main .block-container { max-width: 1120px; padding-top: 1.2rem; padding-bottom: 2rem; }
[data-testid="stHeader"] { background: rgba(0,0,0,0); }
h1,h2,h3,h4,p,label,span,div { color: var(--ink); }
.hero { background: linear-gradient(135deg,#fff 0%,#FFF8F2 100%); border:1px solid var(--line); border-radius:24px; padding:24px 26px; box-shadow:0 12px 36px rgba(17,24,39,.06); margin-bottom:16px; }
.eyebrow { color:var(--brand)!important; font-weight:900; letter-spacing:.05em; text-transform:uppercase; font-size:.78rem; }
.hero-title { font-size:2.15rem; line-height:1.08; font-weight:900; margin:.25rem 0 .4rem 0; color:var(--navy)!important; }
.hero-sub { color:var(--muted)!important; font-size:1.05rem; line-height:1.5; max-width:850px; }
.badge { display:inline-block; background:#FFF0E4; border:1px solid #FFD4B4; padding:6px 10px; border-radius:999px; font-weight:800; font-size:.82rem; margin:8px 6px 0 0; color:#8A3B06!important; }
.card { background:var(--paper); border:1px solid var(--line); border-radius:18px; padding:17px 18px; box-shadow:0 6px 18px rgba(17,24,39,.04); height:100%; }
.card-label { color:var(--muted)!important; font-size:.84rem; font-weight:800; text-transform:uppercase; letter-spacing:.03em; }
.card-value { font-size:1.65rem; font-weight:900; margin-top:4px; line-height:1.15; color:var(--navy)!important; }
.card-sub { color:var(--muted)!important; font-size:.9rem; margin-top:5px; }
.signal { border-radius:20px; padding:20px; margin:12px 0 16px 0; border:1px solid var(--line); background:#fff; }
.signal.positive { border-left:7px solid var(--green); }
.signal.negative { border-left:7px solid var(--red); }
.signal.neutral { border-left:7px solid var(--amber); }
.signal-name { font-size:1.8rem; font-weight:950; margin:.2rem 0 .45rem 0; }
.signal-copy { color:var(--muted)!important; line-height:1.55; }
.disclaimer { background:#FFF8E7; border:1px solid #F0D999; border-radius:14px; padding:11px 14px; color:#684E00!important; font-size:.9rem; margin:8px 0 14px 0; }
.small { color:var(--muted)!important; font-size:.9rem; }
div[data-testid="stMetric"] { background:#fff; border:1px solid var(--line); border-radius:16px; padding:10px 14px; }
.stButton>button { border-radius:12px; background:var(--brand); color:#fff; border:0; font-weight:900; min-height:46px; }
.stButton>button:hover { background:var(--brand2); color:#fff; }
.stTabs [data-baseweb="tab-list"] { gap:6px; }
.stTabs [data-baseweb="tab"] { background:#fff; border:1px solid var(--line); border-radius:10px 10px 0 0; font-weight:800; padding:9px 14px; }
.stTabs [data-baseweb="tab"][aria-selected="true"] { background:#FFF0E4!important; border-color:#F47B20!important; box-shadow:inset 0 -3px 0 #F47B20; }
.stTabs [data-baseweb="tab"][aria-selected="true"] p,
.stTabs [data-baseweb="tab"][aria-selected="true"] span { color:#8A3B06!important; }
.news-card { background:#fff; border:1px solid var(--line); border-radius:16px; padding:15px 16px; margin-bottom:12px; box-shadow:0 4px 16px rgba(17,24,39,.04); }
.news-title { font-size:1.04rem; font-weight:900; line-height:1.38; margin-bottom:5px; color:var(--navy)!important; }
.news-meta { color:var(--muted)!important; font-size:.84rem; font-weight:700; margin-bottom:7px; }
.news-summary { color:#475569!important; line-height:1.5; margin-bottom:8px; }
.news-tag { display:inline-block; padding:4px 9px; border-radius:999px; background:#EFF6FF; color:#1D4ED8!important; font-size:.78rem; font-weight:800; margin:0 5px 5px 0; }
.footer { margin-top:28px; padding:18px 0 4px; border-top:1px solid var(--line); color:var(--muted)!important; font-size:.84rem; text-align:center; line-height:1.55; }
@media (max-width: 700px) {
  .main .block-container { padding-left:.85rem; padding-right:.85rem; }
  .hero { padding:19px 17px; border-radius:18px; }
  .hero-title { font-size:1.72rem; }
  .hero-sub { font-size:.98rem; }
  .card-value { font-size:1.35rem; }
}
</style>
""",
    unsafe_allow_html=True,
)


@st.cache_data(show_spinner=False)
def cached_history():
    return load_local_history()


def server_fred_api_key() -> str:
    """Read a server-side FRED key without ever asking conference visitors for it."""
    env_key = os.getenv("FRED_API_KEY", "").strip()
    try:
        secret_key = str(st.secrets.get("FRED_API_KEY", "")).strip()
    except Exception:
        secret_key = ""
    return secret_key or env_key


@st.cache_data(ttl=900, show_spinner=False)
def cached_live_panel(api_key: str = ""):
    return build_live_raw_panel(cached_history(), train_months=TRAIN_MONTHS, api_key=api_key or None)


@st.cache_data(show_spinner=False)
def cached_supervised(target: str, lead: int):
    return prepare_supervised(cached_history(), target, lead)[0]


def target_name(code: str) -> str:
    return TARGET_LABELS[code]

def target_price_history(raw: pd.DataFrame, target: str) -> pd.DataFrame:
    """Return a clean Date/Close frame for one index from either local or refreshed data."""
    if raw is None or raw.empty or "Date" not in raw.columns or target not in raw.columns:
        return pd.DataFrame(columns=["Date", "Close"])
    out = raw[["Date", target]].copy().rename(columns={target: "Close"})
    out["Date"] = pd.to_datetime(out["Date"], errors="coerce")
    out["Close"] = pd.to_numeric(out["Close"], errors="coerce")
    return out.dropna(subset=["Date", "Close"]).sort_values("Date").drop_duplicates("Date", keep="last").reset_index(drop=True)


def render_recent_forecast_chart(result: dict, target: str, price_source: pd.DataFrame, historical: bool = False, lookback: int = 30):
    """Show the recent observed path and extend it to the model forecast date."""
    price_df = target_price_history(price_source, target)
    runtime = pd.Timestamp(result["runtime"])
    target_day = pd.Timestamp(result["target_day"])
    recent = price_df.loc[price_df["Date"] <= runtime].tail(int(lookback)).copy()
    if len(recent) < 2:
        return

    recent["Daily return"] = recent["Close"].pct_change() * 100
    pred_price = float(result["prediction"]["pred_price"])
    current_price = float(result["current_price"])

    fig = make_subplots(
        rows=2,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.07,
        row_heights=[0.76, 0.24],
    )
    fig.add_trace(
        go.Scatter(
            x=recent["Date"], y=recent["Close"], mode="lines", name="Observed close",
            line=dict(width=3), hovertemplate="%{x|%b %d, %Y}<br>Close: %{y:,.2f}<extra></extra>",
        ), row=1, col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=[runtime, target_day], y=[current_price, pred_price], mode="lines+markers",
            name="Model forecast path", line=dict(width=4, dash="dash"), marker=dict(size=10),
            hovertemplate="%{x|%b %d, %Y}<br>Index level: %{y:,.2f}<extra></extra>",
        ), row=1, col=1,
    )
    if historical and result.get("actual_price") is not None:
        fig.add_trace(
            go.Scatter(
                x=[target_day], y=[float(result["actual_price"])], mode="markers", name="Observed target close",
                marker=dict(size=13, symbol="diamond"),
                hovertemplate="%{x|%b %d, %Y}<br>Observed close: %{y:,.2f}<extra></extra>",
            ), row=1, col=1,
        )
    fig.add_trace(
        go.Bar(
            x=recent["Date"], y=recent["Daily return"].fillna(0), name="Daily % change", showlegend=False,
            hovertemplate="%{x|%b %d, %Y}<br>Daily change: %{y:.2f}%<extra></extra>",
        ), row=2, col=1,
    )
    fig.add_vline(x=runtime, line_dash="dot", line_width=1, row=1, col=1)
    fig.update_layout(
        title=f"{target_name(target)} recent trend and forecast",
        height=560,
        margin=dict(l=10, r=10, t=55, b=10),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
        hovermode="x unified",
    )
    fig.update_yaxes(title_text="Index close", row=1, col=1)
    fig.update_yaxes(title_text="Daily %", row=2, col=1)
    st.plotly_chart(fig, use_container_width=True)
    st.caption("Solid line = recent observed closes. Dashed segment = the model's forecast path from the latest completed close to the estimated forecast date.")


def render_market_trends():
    st.markdown("## Market trends")
    st.markdown(
        '<div class="small">Explore the latest completed daily index history used for conference context. '
        'The app refreshes FRED automatically; visitors do not need an API key.</div>',
        unsafe_allow_html=True,
    )
    a, b = st.columns(2)
    with a:
        target = st.selectbox("Index", ALLOWED_TARGETS, index=ALLOWED_TARGETS.index(DEFAULT_TARGET), format_func=target_name, key="trend_target")
    with b:
        period = st.selectbox("Time window", [30, 90, 252, 756], index=1, format_func=lambda n: {30:"30 trading days",90:"90 trading days",252:"1 year",756:"3 years"}[n], key="trend_period")

    try:
        panel = cached_live_panel(server_fred_api_key())
        price = target_price_history(panel, target).tail(int(period)).copy()
        if len(price) < 2:
            st.info("Not enough observations are available for this view.")
            return
        price["Daily return"] = price["Close"].pct_change() * 100
        change = price["Close"].iloc[-1] / price["Close"].iloc[0] - 1
        c1, c2, c3 = st.columns(3)
        with c1:
            metric_card("Latest close", f"{price['Close'].iloc[-1]:,.2f}", str(price["Date"].iloc[-1].date()))
        with c2:
            metric_card("Window return", f"{change:+.2%}", f"{len(price)} observations")
        with c3:
            metric_card("Data source", "FRED", "Latest completed daily observation")

        fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.07, row_heights=[0.76, 0.24])
        fig.add_trace(go.Scatter(x=price["Date"], y=price["Close"], mode="lines", name="Close", line=dict(width=3)), row=1, col=1)
        fig.add_trace(go.Bar(x=price["Date"], y=price["Daily return"].fillna(0), name="Daily % change", showlegend=False), row=2, col=1)
        fig.update_layout(title=f"{target_name(target)} time-series trend", height=570, margin=dict(l=10, r=10, t=55, b=10), hovermode="x unified", showlegend=False)
        fig.update_yaxes(title_text="Index close", row=1, col=1)
        fig.update_yaxes(title_text="Daily %", row=2, col=1)
        st.plotly_chart(fig, use_container_width=True)
        st.caption("FRED daily index observations are not intraday quotes. During an open trading session, the latest completed close will normally be the previous trading day's value.")
    except Exception as exc:
        st.warning("The live trend could not be refreshed right now. The historical research archive remains available in the Historical Demo tab.")
        with st.expander("Technical detail"):
            st.code(str(exc))



def metric_card(label: str, value: str, sub: str = ""):
    st.markdown(
        f'<div class="card"><div class="card-label">{html.escape(label)}</div>'
        f'<div class="card-value">{html.escape(value)}</div>'
        f'<div class="card-sub">{html.escape(sub)}</div></div>',
        unsafe_allow_html=True,
    )


def signal_explanation(result: dict, signal: dict, context: dict, lead: int) -> str:
    direction = "higher" if result["prediction"]["pred_return"] > 0 else "lower"
    conf = signal["confidence"] * 100
    vix_text = signal["regime"].lower()
    return (
        f"The ensemble estimates a {direction} index level over the next {lead} trading day"
        f"{'s' if lead != 1 else ''}. After comparing the forecast with simple baselines and applying "
        f"the {vix_text} threshold, the result is <b>{signal['signal']}</b>. "
        f"Directional confidence score: <b>{conf:.1f}%</b>. The broader six-month market context is "
        f"<b>{context['bias']}</b>; it is shown for interpretation and does not overwrite the short-horizon signal."
    )


def render_result(result: dict, target: str, lead: int, historical: bool = False, price_source: pd.DataFrame | None = None):
    pred = result["prediction"]
    sig = classify_signal(
        pred_return=pred["pred_return"],
        prob_up=pred.get("prob_up"),
        recent_naive_return=result.get("recent_naive_return", 0.0),
        vix_value=result.get("vix_value"),
    )
    ctx = market_context(result["current_price"], result.get("reference_price"))
    alignment = signal_alignment(sig["signal"], ctx["bias"])

    c1, c2, c3 = st.columns(3)
    with c1:
        metric_card("Current close", f"{result['current_price']:,.2f}", f"Runtime: {result['runtime'].date()}")
    with c2:
        metric_card("Forecast close", f"{pred['pred_price']:,.2f}", f"Estimated target date: {result['target_day'].date()}")
    with c3:
        sign = "+" if pred["pred_return"] >= 0 else ""
        metric_card("Forecast move", f"{sign}{pred['pred_return']*100:.2f}%", f"{sign}{pred['pred_gain']:,.2f} index points")

    st.markdown(
        f'<div class="signal {sig["tone"]}"><div class="card-label">Model signal</div>'
        f'<div class="signal-name">{html.escape(sig["signal"])}</div>'
        f'<div class="signal-copy">{signal_explanation(result, sig, ctx, lead)}</div></div>',
        unsafe_allow_html=True,
    )

    m1, m2, m3, m4 = st.columns(4)
    with m1:
        metric_card("VIX", "—" if pd.isna(result.get("vix_value")) else f"{result['vix_value']:.2f}", sig["regime"])
    with m2:
        metric_card("Directional confidence", f"{sig['confidence']*100:.1f}%", "Model score, not a guarantee")
    with m3:
        metric_card("Market context", ctx["bias"], "Six-month training-window trend")
    with m4:
        metric_card("Signal / context", alignment, "Agreement with broader trend")

    if price_source is not None:
        st.markdown("### Recent time-series trend")
        render_recent_forecast_chart(result, target, price_source, historical=historical, lookback=30)

    st.markdown("### Forecast versus baselines")
    comp = pd.DataFrame([
        {"Method": "Stacked ensemble", "Forecast close": pred["pred_price"], "Forecast return": pred["pred_return"]},
        {"Method": "No-change baseline", "Forecast close": result["naive"]["flat"]["pred_price"], "Forecast return": 0.0},
        {"Method": "Recent-move baseline", "Forecast close": result["naive"]["recent"]["pred_price"], "Forecast return": result["naive"]["recent"]["pred_return"]},
    ])
    if historical:
        comp["Actual close"] = result["actual_price"]
        comp["Absolute error"] = (comp["Forecast close"] - result["actual_price"]).abs()
        st.dataframe(
            comp.style.format({"Forecast close": "{:,.2f}", "Forecast return": "{:.2%}", "Actual close": "{:,.2f}", "Absolute error": "{:,.2f}"}),
            hide_index=True,
            use_container_width=True,
        )
        st.caption(f"Observed return over the selected horizon: {result['actual_return']:.2%}.")
    else:
        st.dataframe(
            comp.style.format({"Forecast close": "{:,.2f}", "Forecast return": "{:.2%}"}),
            hide_index=True,
            use_container_width=True,
        )



def render_live():
    st.markdown("## Try the model now")
    st.markdown('<div class="small">Designed for conference use: choose an index and forecast horizon, then run the published stacked-ensemble framework. The 5-day horizon is selected by default because it performed best in the 2025 rolling evaluation. Market data are refreshed automatically from FRED; visitors do not need an API key.</div>', unsafe_allow_html=True)
    a, b = st.columns(2)
    with a:
        target = st.selectbox("Index", ALLOWED_TARGETS, index=ALLOWED_TARGETS.index(DEFAULT_TARGET), format_func=target_name, key="live_target")
    with b:
        lead = st.selectbox("Forecast horizon", ALLOWED_LEADS, index=ALLOWED_LEADS.index(DEFAULT_LEAD), format_func=lambda x: f"{x} trading day{'s' if x != 1 else ''}", key="live_lead")

    with st.expander("How is the latest data obtained?"):
        st.markdown(
            "The deployed app automatically requests the newest available daily observations from **Federal Reserve Economic Data (FRED)**. "
            "A private server-side FRED API key is used when configured; otherwise the app falls back to FRED's public CSV endpoint. "
            "No visitor credentials are required. Because these are daily observations rather than intraday quotes, the latest completed index close may be the previous trading day while the market is still open."
        )

    if st.button("Generate latest forecast", type="primary", use_container_width=True):
        with st.spinner("Refreshing market data and fitting the ensemble…"):
            try:
                panel = cached_live_panel(server_fred_api_key())
                result = live_forecast(panel, target, int(lead))
                age = (pd.Timestamp.today().normalize() - result["runtime"].normalize()).days
                if age > 4:
                    st.warning(f"The latest usable {target_name(target)} close in the refreshed source is {result['runtime'].date()}. Interpret this as a dated demonstration, not a current-market call.")
                render_result(result, target, int(lead), historical=False, price_source=panel)
            except Exception as exc:
                st.error("Live refresh is temporarily unavailable. The historical demo and published evidence below remain available.")
                with st.expander("Technical detail"):
                    st.code(str(exc))



def strip_html(value: str) -> str:
    if value is None:
        return ""
    value = html.unescape(str(value))
    value = re.sub(r"<[^>]+>", " ", value)
    return re.sub(r"\s+", " ", value).strip()


def short_text(value: str, max_chars: int = 250) -> str:
    value = strip_html(value)
    if len(value) <= max_chars:
        return value
    return value[: max_chars - 1].rsplit(" ", 1)[0] + "…"


def infer_news_tag(text_value: str) -> str:
    text_l = (text_value or "").lower()
    if any(k in text_l for k in ["war", "conflict", "missile", "attack", "airstrike", "ceasefire", "invasion", "military"]):
        return "Geopolitical risk"
    if any(k in text_l for k in ["sanction", "export control", "embargo", "tariff", "trade war", "supply chain"]):
        return "Trade / policy risk"
    if any(k in text_l for k in ["oil", "brent", "wti", "opec", "natural gas", "energy"]):
        return "Energy"
    if any(k in text_l for k in ["fed", "federal reserve", "rate", "treasury", "yield", "bond"]):
        return "Rates / Fed"
    if any(k in text_l for k in ["inflation", "cpi", "pce"]):
        return "Inflation"
    if any(k in text_l for k in ["vix", "volatility", "selloff", "risk appetite"]):
        return "Volatility"
    if any(k in text_l for k in ["earnings", "profit", "revenue", "guidance"]):
        return "Earnings"
    if any(k in text_l for k in ["jobs", "payroll", "unemployment", "labor"]):
        return "Labor / Macro"
    return "Market sentiment"


def infer_market_relevance(text_value: str) -> str:
    text_l = (text_value or "").lower()
    if any(k in text_l for k in ["oil", "brent", "wti", "opec", "gas", "energy"]):
        return "Energy-price changes can affect inflation expectations, transport costs, margins, and sector rotation."
    if any(k in text_l for k in ["sanction", "export control", "embargo", "tariff", "trade war", "supply chain"]):
        return "Trade-policy or supply-chain shocks can affect company revenues, margins, inflation, and risk appetite."
    if any(k in text_l for k in ["war", "conflict", "missile", "attack", "invasion", "military"]):
        return "Geopolitical stress can move risk sentiment, defense and energy shares, yields, and volatility."
    if any(k in text_l for k in ["fed", "federal reserve", "rate", "treasury", "yield", "inflation", "cpi", "pce"]):
        return "Macro and rates news can change discount rates, valuation, and broad equity-index direction."
    if any(k in text_l for k in ["earnings", "profit", "revenue", "guidance"]):
        return "Large-company earnings can influence index-level sentiment and expected growth."
    return "This headline may affect broad market sentiment or investor positioning."


def news_query(target: str, focus: str, extra_terms: str = "") -> str:
    base = '("Dow Jones" OR "US stocks" OR "Wall Street")' if target == "DJIA" else '("S&P 500" OR "US stocks" OR "Wall Street")'
    if focus == "Geopolitical risk":
        context = '(war OR conflict OR sanctions OR tariff OR "trade war" OR "supply chain" OR oil OR OPEC OR China OR Taiwan OR Russia OR Ukraine)'
    elif focus == "Macro / Fed":
        context = '("Federal Reserve" OR inflation OR CPI OR PCE OR "Treasury yields" OR jobs OR unemployment OR VIX)'
    else:
        context = '("Federal Reserve" OR inflation OR earnings OR VIX OR oil OR geopolitical OR sanctions OR tariffs OR "Treasury yields")'
    q = f"{base} {context}"
    if extra_terms.strip():
        q = f"{q} {extra_terms.strip()}"
    return q


@st.cache_data(show_spinner=False, ttl=900)
def fetch_google_news_rss(query: str, max_items: int = 8):
    encoded = quote_plus(query)
    url = f"https://news.google.com/rss/search?q={encoded}&hl=en-US&gl=US&ceid=US:en"
    try:
        response = requests.get(
            url,
            headers={"User-Agent": "Mozilla/5.0 StockForecastGeniePro/ConferenceEdition"},
            timeout=12,
        )
        response.raise_for_status()
        root = ET.fromstring(response.content)
    except Exception as exc:
        raise RuntimeError("The news feed could not be refreshed right now.") from exc

    channel = root.find("channel")
    if channel is None:
        return []
    rows = []
    for item in channel.findall("item")[: int(max_items)]:
        title = strip_html(item.findtext("title"))
        link = item.findtext("link") or ""
        published = strip_html(item.findtext("pubDate"))
        summary = short_text(item.findtext("description"), 250)
        source = strip_html(item.findtext("source")) or "News source"
        combined = f"{title} {summary}"
        if title:
            rows.append({
                "title": title,
                "link": link,
                "published": published,
                "summary": summary,
                "source": source,
                "tag": infer_news_tag(combined),
                "relevance": infer_market_relevance(combined),
            })
    return rows


def render_news_card(article: dict):
    title = html.escape(article.get("title", "Untitled"))
    source = html.escape(article.get("source", "News source"))
    published = html.escape(article.get("published", ""))
    summary = html.escape(article.get("summary", ""))
    tag = html.escape(article.get("tag", "Market context"))
    relevance = html.escape(article.get("relevance", ""))
    link = html.escape(article.get("link", ""), quote=True)
    link_html = f'<a href="{link}" target="_blank">Open article ↗</a>' if link else ""
    meta = f"{source} • {published}" if published else source
    st.markdown(
        f'<div class="news-card"><div class="news-title">{title}</div>'
        f'<div class="news-meta">{meta}</div><div class="news-summary">{summary}</div>'
        f'<span class="news-tag">{tag}</span>'
        f'<div class="news-summary"><b>Why it may matter:</b> {relevance}</div>{link_html}</div>',
        unsafe_allow_html=True,
    )


def render_market_news():
    st.markdown("## Market news & event context")
    st.markdown(
        '<div class="small">Recent headlines are shown as interpretation context only. '
        '<b>News is not used as a predictor in the published forecasting model</b>, so the conference demo remains faithful to the paper.</div>',
        unsafe_allow_html=True,
    )
    a, b = st.columns(2)
    with a:
        target = st.selectbox("Index context", ALLOWED_TARGETS, index=ALLOWED_TARGETS.index(DEFAULT_TARGET), format_func=target_name, key="news_target")
    with b:
        focus = st.selectbox("News focus", ["Market + macro + geopolitical", "Macro / Fed", "Geopolitical risk"], key="news_focus")
    extra = st.text_input("Optional topic", placeholder="e.g., oil, tariffs, AI, inflation", key="news_topic")
    if st.button("Refresh recent news", use_container_width=True, key="news_refresh"):
        with st.spinner("Fetching recent market context…"):
            try:
                articles = fetch_google_news_rss(news_query(target, focus, extra), max_items=8)
                if not articles:
                    st.info("No recent articles were returned for this query. Try removing the optional topic.")
                else:
                    st.caption("Source: Google News RSS search. Headlines and summaries belong to their respective publishers.")
                    for article in articles:
                        render_news_card(article)
            except Exception as exc:
                st.warning(str(exc))


def render_historical():
    st.markdown("## Historical case explorer")
    st.markdown('<div class="small">Pick a date that is already in the research archive. The model is refit using only labels whose outcome date was available by that runtime.</div>', unsafe_allow_html=True)
    a, b = st.columns(2)
    with a:
        target = st.selectbox("Index", ALLOWED_TARGETS, format_func=target_name, key="hist_target")
    with b:
        lead = st.selectbox("Forecast horizon", ALLOWED_LEADS, index=2, format_func=lambda x: f"{x} trading day{'s' if x != 1 else ''}", key="hist_lead")
    supervised = cached_supervised(target, int(lead))
    days = pd.to_datetime(supervised["run_day"]).sort_values().drop_duplicates()
    default_idx = max(0, len(days) - 45)
    chosen = st.date_input("Runtime date", value=days.iloc[default_idx].date(), min_value=days.iloc[0].date(), max_value=days.iloc[-1].date())
    runtime = nearest_runtime(supervised, chosen)
    if runtime.date() != pd.Timestamp(chosen).date():
        st.info(f"Using the nearest available market date on or before your choice: {runtime.date()}.")
    if st.button("Run historical case", use_container_width=True):
        with st.spinner("Refitting the ensemble for this historical cutoff…"):
            try:
                result = historical_evaluation(cached_history(), target, int(lead), runtime)
                render_result(result, target, int(lead), historical=True, price_source=cached_history())
            except Exception as exc:
                st.error(str(exc))


def render_evidence():
    st.markdown("## Published 2025 rolling evaluation")
    st.markdown("The paper reports that performance improved materially at longer horizons. The five-day stacked ensemble reached **70.9% directional accuracy for the S&P 500** and **73.2% for the Dow**, while reducing mean absolute error versus a no-change benchmark by **22.4%** and **24.3%**, respectively.")

    rows = []
    for target in ALLOWED_TARGETS:
        for lead in ALLOWED_LEADS:
            metric = PUBLISHED_METRICS[target][lead]
            rows.append({"Index": target_name(target), "Horizon": f"{lead} day", "Directional accuracy (%)": metric["directional_accuracy"], "MAE improvement vs no-change (%)": metric["mae_improvement"]})
    df = pd.DataFrame(rows)
    st.dataframe(df, hide_index=True, use_container_width=True)

    fig = go.Figure()
    for target in ALLOWED_TARGETS:
        vals = [PUBLISHED_METRICS[target][h]["directional_accuracy"] for h in ALLOWED_LEADS]
        fig.add_trace(go.Bar(name=target_name(target), x=[f"{h}-day" for h in ALLOWED_LEADS], y=vals))
    fig.add_hline(y=50, line_dash="dash", annotation_text="50% reference")
    fig.update_layout(barmode="group", yaxis_title="Directional accuracy (%)", height=390, margin=dict(l=10, r=10, t=20, b=10))
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("### What the system does")
    x, y, z = st.columns(3)
    with x:
        metric_card("1 · Forecast", "Stacked ensemble", "Extra Trees + Random Forest + XGBoost with a robust meta-learner")
    with y:
        metric_card("2 · Context", "Volatility-aware", "VIX changes the minimum forecast edge required for a directional signal")
    with z:
        metric_card("3 · Explain", "Plain-language signal", "Strong Negative → Neutral / No Clear Edge → Strong Positive")

    l1, l2 = st.columns(2)
    with l1:
        st.link_button("Read the conference paper", PAPER_URL, use_container_width=True)
    with l2:
        st.link_button("Open DOI", DOI_URL, use_container_width=True)


def render_method():
    st.markdown("## Method & responsible use")
    st.markdown(
        "StockForecastGenie Pro forecasts short-horizon movements in the S&P 500 and Dow Jones Industrial Average. "
        "The conference interface fixes the main research specification to a **six-month rolling training window**, "
        "**point gain** as the supervised target, **core market/macro features**, and the **stacked ensemble**. "
        "Only the published 1-, 3-, and 5-day horizons are exposed to visitors."
    )
    st.markdown(
        "For operational use, the app also applies a label-availability safeguard: a training row is eligible only when "
        "its forecast outcome date has already occurred by the runtime cutoff. This prevents future target values from "
        "entering a live fit."
    )
    st.markdown("### Signal interpretation")
    st.markdown(
        "The signal is **decision support, not a trading instruction**. Forecast magnitude is compared with no-change "
        "and recent-move baselines, then filtered through VIX-dependent thresholds. The broader six-month market trend "
        "is displayed separately as context and does not overwrite the short-horizon signal."
    )
    st.markdown("### Data")
    st.markdown(
        "The model uses market and macroeconomic series from Federal Reserve Economic Data (FRED), including index "
        "levels, VIX, Treasury yields, oil prices, policy rates, unemployment, industrial production, consumer sentiment, "
        "money supply, inflation, and housing-price information. The deployed app can use a server-side FRED API key when configured, "
        "with a public FRED CSV fallback so visitors never need to enter credentials."
    )
    st.markdown("### News context")
    st.markdown("The Market News tab retrieves recent market, macroeconomic, and geopolitical headlines for interpretation only. News is deliberately kept outside the forecasting feature set so the conference app remains aligned with the published model.")
    st.markdown('<div class="disclaimer"><b>Research demonstration only.</b> Forecasts are uncertain, may be wrong, and should not be interpreted as personalized investment advice or a recommendation to buy or sell any security.</div>', unsafe_allow_html=True)


def main():
    left, right = st.columns([4.7, 1.3], vertical_alignment="center")
    with left:
        st.markdown(
            f'<div class="hero"><div class="eyebrow">Conference research demo</div>'
            f'<div class="hero-title">{APP_TITLE}</div><div class="hero-sub">{APP_SUBTITLE}</div>'
            f'<span class="badge">{CONFERENCE_NAME}</span><span class="badge">{CONFERENCE_LOCATION}</span>'
            f'<span class="badge">{CONFERENCE_DATES}</span></div>',
            unsafe_allow_html=True,
        )
    with right:
        if LOGO_PATH.exists():
            st.image(str(LOGO_PATH), use_container_width=True)

    st.markdown('<div class="disclaimer"><b>Research demonstration — not investment advice.</b> The app presents model signals and uncertainty-aware context rather than direct trading instructions.</div>', unsafe_allow_html=True)

    tab_live, tab_trends, tab_news, tab_hist, tab_evidence, tab_method = st.tabs(["Live Demo", "Market Trends", "Market News", "Historical Demo", "Published Evidence", "Method & About"])
    with tab_live:
        render_live()
    with tab_trends:
        render_market_trends()
    with tab_news:
        render_market_news()
    with tab_hist:
        render_historical()
    with tab_evidence:
        render_evidence()
    with tab_method:
        render_method()

    st.markdown(
        '<div class="footer">Chibuike Chiedozie Ibebuchi · Department of Mathematics, Morgan State University<br>'
        'Presented at the 7th National HBCU Blockchain, FinTech, and AI Conference · Nashville, Tennessee · November 8–10, 2026<br>'
        'Market and macroeconomic data: FRED, Federal Reserve Bank of St. Louis. This research prototype is not endorsed or certified by the Federal Reserve Bank of St. Louis.</div>',
        unsafe_allow_html=True,
    )


if __name__ == "__main__":
    main()
