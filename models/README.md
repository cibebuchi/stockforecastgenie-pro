# Frozen model artifacts

The deployed application **will not train a model**. It expects these six files:

- `frozen_SP500_lead1.joblib`
- `frozen_SP500_lead3.joblib`
- `frozen_SP500_lead5.joblib`
- `frozen_DJIA_lead1.joblib`
- `frozen_DJIA_lead3.joblib`
- `frozen_DJIA_lead5.joblib`

Create them once, locally, from non-FRED historical data by running:

```bash
pip install -r requirements-freeze.txt
python offline_training/freeze_models_yahoo.py --start 2015-01-01 --end 2026-10-01 --confirm-yahoo-training YES
```

Review `MODEL_MANIFEST.csv`, then commit the frozen `.joblib` files to the private repository.

Do not run the training utility from Streamlit, GitHub Actions, or any FRED workflow.
