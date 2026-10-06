# FRED data and model-training disclosure

This conference demonstration uses previously trained models. The Streamlit application does not train, retrain, or change model parameters during a session.

- Model training was carried out separately using Yahoo Finance historical index closing prices.
- FRED observations are accessed using the visitor's FRED API key, to calculate recent features and show the corresponding index trend.
- FRED observations are not added to the model's training history or written to the application's GitHub repository, persistent cache, or database.
- Bureau of Labor Statistics statistics and Google News headlines are informational context, not predictive features.
- The pretrained models use price-derived realized volatility, **not the VIX and macroeconomic input framework in the published paper**.
- Published 2025 evaluation results belong to the paper's original model and must not be presented as performance of the pretrained demo.

The following notice is shown in the app: "This product uses the FRED® API but is not endorsed or certified by the Federal Reserve Bank of St. Louis."

FRED's terms and third-party index-owner permissions remain relevant to use of the data. A FRED key and private GitHub repository do not themselves establish additional index display or training rights.
