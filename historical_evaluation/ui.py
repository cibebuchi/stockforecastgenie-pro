"""Isolated Streamlit page: fixed-model historical replay (Sep 1–Nov 10, 2026).

No live page is modified. Uses the repository's existing inference and signal logic.
"""
from __future__ import annotations

from datetime import date
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from config import MODELS_DIR, TARGETS, FRED_LOOKBACK_DAYS, FRED_KEY_URL, FRED_TERMS_URL
from data_sources import DataSourceError, fetch_fred_index_history
from historical_evaluation.evaluator import (
    HistoricalEvaluationError, normalize_market_data, run_evaluation,
)

_START = date(2026, 9, 1)
_END = date(2026, 11, 10)


def _single_date_chart(data: pd.DataFrame, label: str):
    """Actual close vs forecast with *illustrative*, not calibrated, ranges."""
    rows = data.sort_values("Horizon")
    x = [f"{h} trading day{'s' if h != 1 else ''}" for h in rows["Horizon"]]
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=x, y=rows["Predicted close"], name="Forecast close",
        mode="markers+lines", marker={"size": 11},
        error_y={
            "type": "data", "symmetric": False,
            "array": (rows["Forecast range high"] - rows["Predicted close"]).clip(lower=0).tolist(),
            "arrayminus": (rows["Predicted close"] - rows["Forecast range low"]).clip(lower=0).tolist(),
            "visible": True,
        },
    ))
    actual = rows[rows["Status"] == "Evaluated"]
    if not actual.empty:
        x_actual = [f"{h} trading day{'s' if h != 1 else ''}" for h in actual["Horizon"]]
        fig.add_trace(go.Scatter(x=x_actual, y=actual["Actual close"],
                                 name="Actual close", mode="markers", marker={"size": 12, "symbol": "diamond"}))
    fig.update_layout(title=f"{label}: predicted vs actual index close",
                      xaxis_title="Forecast horizon", yaxis_title="Index level",
                      height=410, margin=dict(l=15, r=15, t=55, b=20), legend_orientation="h")
    st.plotly_chart(fig, width="stretch")
    st.caption("Error bars show the illustrative forecast range.")


def _range_chart(data: pd.DataFrame, label: str):
    evaluated = data[data["Status"] == "Evaluated"]
    if evaluated.empty:
        st.info("No actual outcomes have become available for these selections.")
        return
    available_horizons = sorted(evaluated["Horizon"].unique().tolist())
    horizon = st.selectbox("Compare forecast and actual across dates (horizon)",
                           available_horizons, index=0, key="hist_chart_horizon",
                           format_func=lambda h: f"{h} trading day{'s' if h > 1 else ''}")
    sub = evaluated[evaluated["Horizon"] == horizon].sort_values("Forecast date")
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=sub["Forecast date"], y=sub["Predicted close"],
        name="Predicted index close", mode="lines+markers",
    ))
    fig.add_trace(go.Scatter(
        x=sub["Forecast date"], y=sub["Actual close"],
        name="Actual index close", mode="lines+markers",
    ))
    fig.update_layout(title=f"{label}: predicted versus actual ({horizon}-day horizon)",
                      xaxis_title="Forecast issue date", yaxis_title="Index close",
                      height=420, margin=dict(l=15, r=15, t=55, b=20), legend_orientation="h")
    st.plotly_chart(fig, width="stretch")
    st.caption("Each actual close belongs to the corresponding forecast's outcome date, "
               "not the issue date shown along the horizontal axis.")



def _details(data: pd.DataFrame):
    """Keep the page focused on forecast, observed outcome, and a short signal explanation."""
    st.markdown("#### Forecast vs. actual")
    single_date = data["Forecast date"].nunique() == 1
    if not single_date:
        dates = sorted(data["Forecast date"].unique().tolist(), reverse=True)
        detail_date = st.selectbox("Show forecast details for", dates, key="hist_detail_date")
        data = data[data["Forecast date"] == detail_date]

    for _, row in data.sort_values("Horizon").iterrows():
        h = int(row["Horizon"])
        evaluated = row["Status"] == "Evaluated"
        label = f"{h}-trading-day forecast · {row['Signal']}"
        with st.expander(label, expanded=(single_date or h == int(data["Horizon"].min()))):
            a, b, c, d = st.columns(4)
            a.metric("Predicted close", f"{row['Predicted close']:,.2f}")
            b.metric("Actual close", f"{row['Actual close']:,.2f}" if evaluated else "Pending")
            c.metric("Predicted move", f"{row['Predicted return (%)']:+.2f}%")
            d.metric("Actual move", f"{row['Actual return (%)']:+.2f}%" if evaluated else "Pending")

            if evaluated:
                correct = bool(row["Direction correct"])
                inside = bool(row["Actual inside illustrative range"])
                msg = ("Forecast direction matched the observed movement." if correct
                       else "Forecast direction did not match the observed movement.")
                msg += (" Actual close was within the displayed range." if inside
                        else " Actual close was outside the displayed range.")
                if correct:
                    st.success(msg, icon="✅")
                else:
                    st.warning(msg, icon="⚠️")
                st.caption(f"Absolute forecast error: {row['Absolute error (points)']:,.2f} index points.")
            else:
                st.info("Actual outcome pending for this forecast horizon.")

            st.markdown(
                f"**Illustrative forecast range:** {row['Forecast range low']:,.2f} – "
                f"{row['Forecast range high']:,.2f}"
            )
            st.caption(
                f"**Model signal:** {row['Signal']} · "
                f"**Direction score:** {row['Model direction score (%)']:.1f}% "
                f"(minimum {row['Minimum direction score (%)']:.0f}%)"
            )
            st.caption(
                f"**Volatility:** {row['Model realized-volatility regime']} "
                f"({row['20-day realized volatility (%)']:.1f}% annualized) · "
                f"**Movement thresholds:** {row['Directional threshold (%)']:.2f}% regular / "
                f"{row['Strong-signal threshold (%)']:.2f}% strong"
            )
            if pd.notna(row["VIX at forecast"]):
                vix_note = (f"VIX: {row['VIX at forecast']:.2f} "
                            f"({row['VIX regime at forecast (context)']}) at forecast")
                if pd.notna(row["VIX at outcome"]):
                    vix_note += f" → {row['VIX at outcome']:.2f} at outcome"
                st.caption(vix_note + " · VIX is context; thresholds use realized volatility.")

def render():
    st.subheader("Model Evaluation")
    st.caption("Compare forecasts with actual closes for September 1–November 10, 2026.")
    st.markdown(f"[Get a FRED API key]({FRED_KEY_URL}) · [FRED terms]({FRED_TERMS_URL})")
    key = st.text_input("FRED API key", value=st.session_state.get("fred_key", ""),
                        type="password", key="historical_fred_key")
    agree = st.checkbox("I am using my own FRED API key and agree to the applicable terms.",
                        key="historical_fred_agree")
    c1, c2 = st.columns(2)
    market = c1.selectbox("Equity index", list(TARGETS.keys()), index=0,
                          format_func=lambda t: TARGETS[t]["label"], key="hist_target")
    scope = c2.radio("Selection mode", ["Single date", "Date range"],
                     horizontal=True, key="hist_scope")
    max_date = min(date.today(), _END)
    if max_date < _START:
        st.info("The evaluation window has not begun.")
        return
    if scope == "Single date":
        chosen = st.date_input("Forecast date (observed market close)",
                               value=max_date, min_value=_START,
                               max_value=max_date, key="hist_date")
        first = last = chosen
    else:
        left, right = st.columns(2)
        first = left.date_input("From", value=_START, min_value=_START,
                                max_value=max_date, key="hist_first")
        last = right.date_input("To", value=max_date, min_value=_START,
                                max_value=max_date, key="hist_last")
    horizons = st.multiselect("Forecast horizons (trading days)", [1, 3, 5],
                              default=[1, 3, 5], format_func=lambda h: f"{h} day{'s' if h > 1 else ''}",
                              key="hist_horizons")
    st.caption("Unavailable trading days (weekends or market holidays) are excluded. "
               "For recent forecast dates, 3-day or 5-day **actual outcomes may still be pending**.")

    selection = (market, scope, str(first), str(last), tuple(sorted(horizons)))
    if st.button("Evaluate selected historical dates", type="primary", key="hist_go"):
        st.session_state.pop("hist_output", None)
        if not key or not agree:
            st.warning("Enter your own FRED API key and accept the relevant terms first.")
        elif not horizons:
            st.warning("Select at least one forecast horizon.")
        elif first > last:
            st.warning("The start date must not be later than the end date.")
        else:
            try:
                with st.spinner("Fetching observed closes and replaying frozen models..."):
                    # Existing data-source function retrieves observations through today.
                    # No additional package dependency and no data saved on the server.
                    prices = normalize_market_data(fetch_fred_index_history(
                        TARGETS[market]["fred_series"], key, max(int(FRED_LOOKBACK_DAYS), (date.today() - first).days + 365)
                    ))
                    vix = None
                    vix_error = None
                    try:
                        vix = normalize_market_data(fetch_fred_index_history(
                            "VIXCLS", key, max(int(FRED_LOOKBACK_DAYS), (date.today() - first).days + 365)
                        ))
                    except DataSourceError as exc:
                        vix_error = str(exc)
                    dates = prices.loc[prices["Date"].between(pd.Timestamp(first), pd.Timestamp(last)), "Date"].tolist()
                    if not dates:
                        raise HistoricalEvaluationError("No available observed index closes within the selected dates.")
                    results, skipped = run_evaluation(prices, dates, market, horizons, MODELS_DIR, vix)
                    if results.empty:
                        raise HistoricalEvaluationError(
                            "No forecasts could be produced. " + (skipped[0]["Reason"] if skipped else "")
                        )
                # Retain computed results only across Streamlit reruns; no raw price panels.
                st.session_state["hist_output"] = (selection, results, skipped, vix_error)
            except (DataSourceError, HistoricalEvaluationError, FileNotFoundError, ValueError, KeyError,
                    ImportError, AttributeError) as exc:
                st.error(f"Historical evaluation could not be completed: {exc}")

    stored = st.session_state.get("hist_output")
    if stored is None or stored[0] != selection:
        return
    _, results, skipped, vix_error = stored
    if vix_error:
        st.caption("VIX context unavailable for this selection; forecasts can still be compared.")
    # Lead with what users came here to see: forecasts beside actual outcomes.
    completed = results[(results["Status"] == "Evaluated") &
                        (results["Replay classification"] == "Post-training replay")]
    if not completed.empty and results["Forecast date"].nunique() > 1:
        m1, m2, m3 = st.columns(3)
        m1.metric("Post-training forecasts evaluated", len(completed))
        m2.metric("Mean absolute error", f"{completed['Absolute error (points)'].mean():,.2f} points")
        m3.metric("Directional accuracy", f"{100 * completed['Direction correct'].mean():.1f}%")

    _details(results)
    if results["Forecast date"].nunique() == 1:
        _single_date_chart(results, TARGETS[market]["label"])
    else:
        st.markdown("#### All selected dates")
        summary = results[["Forecast date", "Horizon", "Outcome date", "Predicted close", "Actual close",
                           "Predicted return (%)", "Actual return (%)", "Absolute error (points)",
                           "Direction correct", "Signal"]].copy()
        summary["Direction correct"] = results.apply(
            lambda r: ("✅ Matched" if bool(r["Direction correct"]) else "— Did not match")
            if r["Status"] == "Evaluated" else "Pending", axis=1)
        st.dataframe(summary, hide_index=True, width="stretch", column_config={
            "Predicted close": st.column_config.NumberColumn("Predicted close", format="%.2f"),
            "Actual close": st.column_config.NumberColumn("Actual close", format="%.2f"),
            "Predicted return (%)": st.column_config.NumberColumn("Forecast move (%)", format="%+.2f"),
            "Actual return (%)": st.column_config.NumberColumn("Actual move (%)", format="%+.2f"),
            "Absolute error (points)": st.column_config.NumberColumn("Error (points)", format="%.2f"),
            "Direction correct": "Direction",
        })
        _range_chart(results, TARGETS[market]["label"])

    if skipped:
        with st.expander(f"{len(skipped)} unavailable forecast(s) and reasons"):
            st.dataframe(pd.DataFrame(skipped), hide_index=True, width="stretch")
    st.download_button("Download historical evaluation (CSV)", data=results.to_csv(index=False).encode("utf-8"),
                       file_name=f"StockForecastGenie_{market}_{first}_{last}.csv", mime="text/csv",
                       key="hist_download")
