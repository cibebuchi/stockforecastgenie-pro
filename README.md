# StockForecastGenie Pro — Conference Edition

Conference-ready Streamlit interface for the paper **“StockForecastGenie Pro: A Volatility-Aware Machine Learning Decision-Support System for Short-Horizon Equity Index Forecasting.”**

## What changed from the original local app

- Mobile-first, QR-friendly interface with the live demo first.
- Visitors choose only **S&P 500 / Dow** and **1 / 3 / 5 trading days**.
- The published main specification is fixed: six-month rolling window, point-gain target, core features, stacked ensemble.
- No visitor API key is requested. A server-side FRED API key can be configured in Streamlit Secrets; if absent, the app falls back to the public FRED graph-export endpoint.
- Added a **Market News** tab using Google News RSS for recent market, macroeconomic, and geopolitical context. News remains outside the model predictors.
- Public-facing labels are **Strong Negative, Negative, Neutral / No Clear Edge, Positive, Strong Positive**, matching the paper rather than Buy/Sell commands.
- Broader market context is shown separately and does **not** overwrite the short-horizon model signal.
- Live/historical fitting applies an operational label-availability safeguard: a training label is used only after its target date has occurred.
- All Windows-only absolute paths were replaced with portable relative paths.
- Added conference branding, published evidence, responsible-use language, and links to the official paper and DOI.

## Local run

```bash
python -m venv .venv
# Windows: .venv\\Scripts\\activate
# macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
streamlit run app.py
```

## Streamlit Community Cloud deployment

1. Create a GitHub repository and put the contents of this folder at the repository root.
2. In Streamlit Community Cloud, choose **New app**.
3. Select the repository, branch, and `app.py`.
4. In **App settings → Secrets**, optionally add:

```toml
FRED_API_KEY = "YOUR_FRED_API_KEY"
```

   This key stays server-side and is never shown to conference visitors. Do **not** put the real key in GitHub or in `app.py`. If no key is configured, the app uses the public FRED CSV fallback.
5. Deploy.
6. Open the public URL on a phone and test the **Live Demo**, **Market News**, and **Historical Demo** tabs before creating the QR code.
7. Use the final public URL to generate the QR code placed on the presentation slide.

## Pre-conference checks

- Test on Wi-Fi and cellular data.
- Run S&P 500 and Dow at the 5-day horizon.
- Confirm the latest runtime shown is recent; if not, the app automatically warns users.
- Click **Refresh recent news** and confirm headlines load on both Wi-Fi and cellular data.
- Keep the paper URL and a screenshot of the app on the slide as fallback if venue internet is poor.

## Important note

This is a research demonstration, not personalized investment advice. Forecasts can be wrong.


## News design

The Market News tab retrieves recent Google News RSS results related to the selected index, Federal Reserve/macroeconomic conditions, and geopolitical risks. It tags each headline with a likely market channel and a short explanation. News is **context only** and is not fed into the published forecasting model.

## V3 conference usability refinements

- Restored the recent **time-series trend + forecast path** directly in Live Demo results.
- Added a dedicated **Market Trends** tab with 30-day, 90-day, 1-year, and 3-year views.
- Changed the active-tab styling from dark/navy to a light FinTech-orange treatment.
- Added an in-app explanation of how fresh FRED data are obtained and why the latest completed daily close can be the previous trading day during market hours.
- Visitors still never need to enter an API key. A private server-side `FRED_API_KEY` is used when configured; otherwise FRED's public CSV endpoint is used.
