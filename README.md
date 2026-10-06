# StockForecastGenie Pro — V8.1 Pretrained Conference Edition

This update polishes the text displayed in the working V8 application and keeps the already-trained six models unchanged. **Do not retrain** and **do not replace the six model files** with this patch.

## What the deployment actually does

- The interactive forecasts use previously trained model files created offline from Yahoo Finance index closing prices. The deployed application never trains or updates them.
- Recent actual S&P 500 / Dow Jones index observations are requested from FRED **only for current forecast inference**, using a user-supplied FRED API key. The app does not store or maintain a FRED historical dataset.
- The demo uses price-derived return, momentum, trend, and realized-volatility features. **It does not use VIX or macroeconomic predictors to forecast.** Bureau of Labor Statistics context and Google News are displayed separately.
- It shares the two index targets, 1/3/5 day horizons, and ensemble-model family with the conference research, but **does not reproduce the paper's complete feature set, rolling 2025 evaluation, or published performance.**
- The illustrative forecast band is *not* a calibrated 90% prediction interval. Directional classifier scores are *not* established success probabilities.
- Scores in the original offline `MODEL_MANIFEST.csv` are fit-stage diagnostics, because the ensemble combination layer was trained on those same rows. **They are not independent holdout performance estimates.** Independent walk-forward or holdout testing is required before reporting deployment-model accuracy.

## What this package includes

`app.py` and `config.py` are the two updated runtime files. `deploy_v8_1_to_github.py` copies the app and your **existing six models** from your local V8 working directory to the GitHub repository folder and removes obsolete FRED-snapshot workflows/files. It checks both locations before copying, and optionally commits and pushes. It does **not** contact FRED, Yahoo, or train any model.

### Windows Command Prompt commands

From your inner V8 folder (the one containing `app.py` and `models`):

```bat
copy /Y "C:\Users\chied\Downloads\StockForecastGenie_V8_1_Pretrained_Deployment_Patch\app.py" app.py
copy /Y "C:\Users\chied\Downloads\StockForecastGenie_V8_1_Pretrained_Deployment_Patch\config.py" config.py
python -m streamlit run app.py
```

After testing, stop Streamlit with Ctrl+C and run the deploy helper from the extracted patch directory:

```bat
cd /d "C:\Users\chied\Downloads\StockForecastGenie_V8_1_Pretrained_Deployment_Patch"
python deploy_v8_1_to_github.py
```

This copies the prepared deployment to your *local GitHub repository folder* but does not push. Review `git status` in that repo, then:

```bat
cd /d "C:\Users\chied\Downloads\StockForecastGenie_Conference_Ready_V3"
git add -A
git commit -m "Deploy pretrained inference app and update research disclosures"
git push origin main
```

Or run `python deploy_v8_1_to_github.py --push` to do the push after checking the filenames.

**Security/legal note:** Keeping GitHub private or freezing a model is not proof of the right to use Yahoo historical data or redistribute third-party S&P/Dow observations. Check any applicable third-party data permissions before providing access to conference attendees.


## V8.2 presentation polish

The interface now identifies Chibuike Chiedozie Ibebuchi as the developer and shows the Morgan State FinTech Center logo as a small footer mark rather than a large header graphic.
