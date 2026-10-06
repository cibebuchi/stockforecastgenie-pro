from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
MODELS_DIR = BASE_DIR / "models"
LOGO_PATH = BASE_DIR / "logo.png"

APP_TITLE = "StockForecastGenie Pro"
APP_SUBTITLE = "Pretrained-model decision support for short-horizon U.S. equity-index forecasting"
CONFERENCE_NAME = "7th National HBCU Blockchain, FinTech, and AI Conference"
CONFERENCE_LOCATION = "Nashville, Tennessee"
CONFERENCE_DATES = "November 8–10, 2026"
PAPER_URL = "https://fintech.morgan.edu/publications/2026-conference-proceedings/paper-05/"
DOI_URL = "https://doi.org/10.68414/NKSA6726"
FRED_TERMS_URL = "https://fred.stlouisfed.org/docs/api/terms_of_use.html"
FRED_KEY_URL = "https://fredaccount.stlouisfed.org/apikeys"

TARGETS = {
    "SP500": {"label": "S&P 500", "fred_series": "SP500", "yahoo_symbol": "^GSPC"},
    "DJIA": {"label": "Dow Jones Industrial Average", "fred_series": "DJIA", "yahoo_symbol": "^DJI"},
}
ALLOWED_LEADS = [1, 3, 5]
DEFAULT_TARGET = "SP500"
DEFAULT_LEAD = 5
FRED_LOOKBACK_DAYS = 540

PUBLISHED_METRICS = {
    "SP500": {
        1: {"directional_accuracy": 48.7, "mae_improvement": -2.2},
        3: {"directional_accuracy": 64.8, "mae_improvement": 7.6},
        5: {"directional_accuracy": 70.9, "mae_improvement": 22.4},
    },
    "DJIA": {
        1: {"directional_accuracy": 52.1, "mae_improvement": -0.5},
        3: {"directional_accuracy": 60.2, "mae_improvement": 6.4},
        5: {"directional_accuracy": 73.2, "mae_improvement": 24.3},
    },
}
