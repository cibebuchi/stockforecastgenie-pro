from __future__ import annotations
import math
from pathlib import Path
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from config import (
    APP_TITLE, APP_SUBTITLE, CONFERENCE_DATES, CONFERENCE_LOCATION, CONFERENCE_NAME,
    DEFAULT_LEAD, DEFAULT_TARGET, DOI_URL, FRED_KEY_URL, FRED_LOOKBACK_DAYS,
    FRED_TERMS_URL, LOGO_PATH, MODELS_DIR, PAPER_URL, PUBLISHED_METRICS, TARGETS,
)
from data_sources import DataSourceError, fetch_bls_context, fetch_fred_index_history, fetch_google_news
from frozen_inference import load_bundle, predict_frozen
from signal_logic import model_signal

st.set_page_config(page_title=APP_TITLE, page_icon="📈", layout="wide", initial_sidebar_state="collapsed")

st.markdown("""
<style>
.block-container{max-width:1180px;padding-top:1.15rem;padding-bottom:3rem}
.hero{border:1px solid #ead9c8;border-radius:18px;padding:20px 22px;background:linear-gradient(135deg,#fffaf5,#ffffff);margin-bottom:15px}
.hero h1{margin:0;font-size:2.05rem}.muted{color:#616161}.tiny{font-size:.82rem;color:#666}
.badge{display:inline-block;padding:5px 10px;border-radius:999px;background:#fff0e1;color:#9a4e00;border:1px solid #ffd2a4;font-weight:700;font-size:.78rem}
.card{border:1px solid #e7e7e7;border-radius:15px;padding:16px;background:white;height:100%}
[data-testid="stMetric"]{border:1px solid #ececec;border-radius:14px;padding:10px 14px;background:#fff}
div[role="radiogroup"]{gap:.35rem;border-bottom:1px solid #eee;padding-bottom:.3rem}
div[role="radiogroup"] label{border-radius:10px;padding:.25rem .55rem}
div[role="radiogroup"] label:has(input:checked){background:#fff0e1!important;color:#7f3f00!important;border:1px solid #ffb66d}
.stButton>button{border-radius:10px;font-weight:700}
.frednotice{padding:12px 14px;border:1px solid #dedede;border-radius:12px;background:#fafafa;font-size:.88rem}
</style>
""", unsafe_allow_html=True)

st.markdown(f"""
<div class="hero">
<span class="badge">Conference edition · pretrained model</span>
<h1>{APP_TITLE}</h1>
<div>{APP_SUBTITLE}</div>
<div class="muted" style="margin-top:7px">Developed by <strong>Chibuike Chiedozie Ibebuchi</strong> · Department of Mathematics, Morgan State University</div>
<div class="muted" style="margin-top:4px">{CONFERENCE_NAME} · {CONFERENCE_LOCATION} · {CONFERENCE_DATES}</div>
</div>
""", unsafe_allow_html=True)

page = st.radio("Navigation", ["Live Demo", "Market Trends", "Market News", "Published Evidence", "Method & Terms"], horizontal=True, label_visibility="collapsed")


def fred_notice():
    st.markdown(
        f"""<div class="frednotice">
        <strong>FRED® notice:</strong> This product uses the FRED® API but is not endorsed or certified by the Federal Reserve Bank of St. Louis.
        FRED observations are retrieved for the current forecast, not for model training.
        <a href="{FRED_TERMS_URL}" target="_blank">FRED API Terms of Use</a>.
        </div>""",
        unsafe_allow_html=True,
    )


def fred_credentials():
    st.markdown("#### Your FRED API access")
    st.caption("FRED's current API documentation says users of an application should use their own API key. The key below is held only in this Streamlit session and is not written to disk or GitHub.")
    key = st.text_input("FRED API key", type="password", value=st.session_state.get("fred_key", ""), placeholder="Paste your personal FRED API key")
    agree = st.checkbox("I confirm this is my FRED API key and I agree to the FRED API Terms of Use for this session.", value=st.session_state.get("fred_agree", False))
    if key:
        st.session_state["fred_key"] = key
    st.session_state["fred_agree"] = agree
    st.markdown(f"<span class='tiny'>Need a key? <a href='{FRED_KEY_URL}' target='_blank'>FRED API key page</a>.</span>", unsafe_allow_html=True)
    return key, agree


def forecast_chart(prices: pd.DataFrame, result: dict, lead: int, label: str):
    recent = prices.tail(90)
    target_date = pd.bdate_range(result["runtime"] + pd.offsets.BDay(1), periods=int(lead))[-1]
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=recent["Date"], y=recent["Close"], mode="lines", name="Observed index"))
    fig.add_trace(go.Scatter(
        x=[result["runtime"], target_date], y=[result["current_close"], result["forecast_close"]],
        mode="lines+markers", name="Pretrained forecast", line={"dash":"dash"}
    ))
    fig.add_trace(go.Scatter(
        x=[target_date, target_date], y=[result["interval_low"], result["interval_high"]],
        mode="lines", name="Illustrative forecast range", line={"width":6}
    ))
    fig.update_layout(title=f"{label}: recent index path and {lead}-day forecast", xaxis_title="Date", yaxis_title="Index level", height=430, margin=dict(l=25,r=20,t=55,b=25), legend_orientation="h")
    st.plotly_chart(fig, use_container_width=True)


if page == "Live Demo":
    st.subheader("Explore a forecast using a pretrained model")
    st.info("The models were trained before deployment using Yahoo Finance historical index prices. During this demonstration, the models are used only to generate forecasts. No additional training or model updates take place.")
    st.caption("**How this differs from the conference paper:** This pretrained demonstration uses price history and 20-day realized volatility. VIX, Treasury yields, and the other macroeconomic predictors used in the published research are not inputs to this deployed model. Its results should not be treated as a replication of the paper's reported accuracy.")
    fred_notice()
    st.caption("A FRED API key does not by itself grant additional rights to third-party copyrighted index data. Use the data only as permitted by FRED and the relevant index owners.")
    key, agree = fred_credentials()

    c1, c2 = st.columns([1.25, 1])
    with c1:
        target = st.selectbox("Market", list(TARGETS.keys()), index=list(TARGETS.keys()).index(DEFAULT_TARGET), format_func=lambda x: TARGETS[x]["label"])
    with c2:
        lead = st.selectbox("Forecast horizon", [1,3,5], index=[1,3,5].index(DEFAULT_LEAD), format_func=lambda x: f"{x} trading day{'s' if x != 1 else ''}")

    if st.button("Generate latest forecast", type="primary", use_container_width=True):
        if not key or not agree:
            st.warning("Enter your own FRED API key and accept the FRED API Terms for this session first.")
        else:
            try:
                with st.spinner("Fetching only the recent observations required for inference..."):
                    bundle = load_bundle(MODELS_DIR, target, lead)
                    prices = fetch_fred_index_history(TARGETS[target]["fred_series"], key, FRED_LOOKBACK_DAYS)
                    result = predict_frozen(bundle, prices)
                signal, regime = model_signal(result["forecast_return"], result["direction_confidence"], result["realized_vol20"])
                st.session_state["last_forecast"] = {"target":target,"lead":lead,"result":result}

                m1,m2,m3,m4 = st.columns(4)
                m1.metric("Current close", f"{result['current_close']:,.2f}")
                m2.metric("Forecast close", f"{result['forecast_close']:,.2f}", f"{result['forecast_gain']:+,.2f}")
                m3.metric("Expected move", f"{result['forecast_return']*100:+.2f}%")
                m4.metric("Directional model score", f"{result['direction_confidence']*100:.1f}%")
                st.markdown(f"### {signal}")
                st.caption(f"Latest observation: {result['runtime'].date()} · historical-volatility regime: {regime} · illustrative forecast range: {result['interval_low']:,.2f} to {result['interval_high']:,.2f}")
                st.caption("The directional score is a model output, not a verified probability of being correct. The illustrative range is derived from historical model residuals and has not been independently calibrated to 90% coverage.")
                forecast_chart(prices, result, lead, TARGETS[target]["label"])

                ctx = fetch_bls_context()
                if ctx:
                    st.markdown("#### Independent macro context")
                    a,b = st.columns(2)
                    if "unemployment_rate" in ctx:
                        a.metric("U.S. unemployment rate (BLS)", f"{ctx['unemployment_rate']:.1f}%", help=ctx.get("unemployment_rate_period",""))
                    if "cpi" in ctx:
                        b.metric("CPI-U, seasonally adjusted (BLS)", f"{ctx['cpi']:.1f}", help=ctx.get("cpi_period",""))
                    st.caption("These government statistics provide additional economic context. They do not change the pretrained forecast.")

                with st.expander("How this pretrained model works", expanded=False):
                    st.write(f"**Original model training data:** {result['training_source']}")
                    st.write(f"**Training period:** {result['training_period']}")
                    st.write("**During your visit:** The previously trained model produces a forecast. Its parameters are not changed, and no additional model training takes place.")
                    st.write("**Current data:** Recent FRED observations are used only to calculate the forecast inputs and display the chart. The application does not add those observations to a training dataset or store them in the repository or a database.")
                    st.write("**Volatility signal:** Uses price-derived realized volatility, not VIX. This makes the interactive deployment different from the published research model.")
            except (DataSourceError, FileNotFoundError, ValueError) as exc:
                st.error(str(exc))

elif page == "Market Trends":
    st.subheader("Explore recent market movements")
    st.caption("Recent FRED index observations are retrieved for this request; the application does not save them to a historical data file.")
    fred_notice()
    key, agree = fred_credentials()
    c1,c2 = st.columns(2)
    target = c1.selectbox("Market", list(TARGETS.keys()), key="trend_target", format_func=lambda x: TARGETS[x]["label"])
    window = c2.selectbox("View", ["30 trading days","90 trading days","1 year"], index=1)
    if st.button("Load market trend", use_container_width=True):
        if not key or not agree:
            st.warning("Enter your own FRED API key and accept the FRED API Terms first.")
        else:
            try:
                prices = fetch_fred_index_history(TARGETS[target]["fred_series"], key, FRED_LOOKBACK_DAYS)
                n = {"30 trading days":30,"90 trading days":90,"1 year":252}[window]
                view = prices.tail(n)
                fig = go.Figure(go.Scatter(x=view["Date"], y=view["Close"], mode="lines", name=TARGETS[target]["label"]))
                fig.update_layout(title=f"{TARGETS[target]['label']} — {window}", xaxis_title="Date", yaxis_title="Index level", height=460, margin=dict(l=25,r=20,t=55,b=25))
                st.plotly_chart(fig, use_container_width=True)
                st.caption(f"Latest available observation in this session: {view['Date'].max().date()}. Source: FRED®. No observation history is persisted by the app.")
            except DataSourceError as exc:
                st.error(str(exc))

elif page == "Market News":
    st.subheader("Recent market context")
    st.caption("News is for market context only. Headlines do not change the pretrained forecasts.")
    market = st.selectbox("News focus", ["S&P 500", "Dow Jones", "Federal Reserve inflation Treasury yields", "U.S. stock market volatility"])
    topic = st.text_input("Optional additional topic", placeholder="e.g., tariffs, oil, AI, earnings")
    if st.button("Get recent headlines"):
        query = market + (f" {topic}" if topic.strip() else "")
        try:
            items = fetch_google_news(query, 12)
            for item in items:
                st.markdown(f"**[{item['title']}]({item['link']})**")
                st.caption(" · ".join(x for x in [item.get("source",""), item.get("pubDate","")] if x))
        except DataSourceError as exc:
            st.error(str(exc))

elif page == "Published Evidence":
    st.subheader("Published 2025 rolling evaluation")
    st.warning("These are the published results for the original rolling-training research system, not performance estimates for the pretrained demonstration available in the Live Demo. The two systems use different inputs and evaluation designs.")
    rows=[]
    for target, by_lead in PUBLISHED_METRICS.items():
        for lead, m in by_lead.items():
            rows.append({"Market":TARGETS[target]["label"],"Horizon":f"{lead} day","Directional accuracy (%)":m["directional_accuracy"],"MAE improvement vs no-change (%)":m["mae_improvement"]})
    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
    st.markdown(f"[Read the conference paper]({PAPER_URL}) · [DOI]({DOI_URL})")

else:
    st.subheader("Method, provenance, and terms")
    st.markdown("""
**Pretrained demonstration: how it works**

1. Model training is completed **before deployment** on a separate computer. The resulting models are saved as fixed, pretrained files.
2. Training history comes from Yahoo Finance index closing prices. No FRED data were used for that model training.
3. The web app uses those pretrained models to make predictions; it does not continue training them.
4. A user-provided FRED API key retrieves recent S&P 500 or Dow observations needed to calculate the current forecast inputs.
5. Current FRED observations are used for that request and are not written to GitHub, a model file, or a historical database.
6. Bureau of Labor Statistics data and market news are shown as context and are not model inputs.

**Comparison with the paper:** The conference paper evaluated a rolling-training ensemble with macroeconomic inputs and VIX-based signals. This web demonstration uses a fixed pretrained ensemble and price-based realized volatility. They share the same forecasting targets and broad ensemble approach, but are **not methodologically identical**. The published accuracy figures must not be attributed to this demonstration.

**Performance note:** The training script’s MODEL_MANIFEST.csv contains diagnostic scores from the data used to fit the ensemble’s combination layer. These are **not independent, out-of-sample test results**; a separate holdout or walk-forward evaluation is needed before making performance claims about these pretrained models.
""")
    fred_notice()
    st.markdown(f"- [FRED API Terms of Use]({FRED_TERMS_URL})\n- [Conference paper]({PAPER_URL})\n- [Paper DOI]({DOI_URL})")
    st.warning("S&P 500 and Dow Jones series available through FRED may be owned by third parties and subject to additional copyright or use restrictions. FRED access does not override those rights. Keep use within permissions applicable to the data and event setting.")
    st.info("Research and educational decision support only. Forecasts are uncertain and are not investment advice, a solicitation, or a recommendation to buy or sell securities.")

st.divider()
footer_text, footer_logo = st.columns([5, 1])
with footer_text:
    st.caption("StockForecastGenie Pro · Developed by Chibuike Chiedozie Ibebuchi · Department of Mathematics, Morgan State University")
    st.caption("Pretrained-model conference demonstration · Research/educational use only · Not investment advice")
with footer_logo:
    if Path(LOGO_PATH).exists():
        st.image(str(LOGO_PATH), width=105)

