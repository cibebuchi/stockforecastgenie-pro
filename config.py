from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "market_daily_data_2015_to_present.csv"
STACK_SCRIPT_PATH = BASE_DIR / "stock_stack_upgraded_matched_naive_with_return.py"
LOGO_PATH = BASE_DIR / "logo.png"

APP_TITLE = "StockForecastGenie Pro"
APP_SUBTITLE = "Volatility-aware machine-learning decision support for short-horizon equity-index forecasting"
CONFERENCE_NAME = "7th National HBCU Blockchain, FinTech, and AI Conference"
CONFERENCE_LOCATION = "Nashville, Tennessee"
CONFERENCE_DATES = "November 8–10, 2026"
PAPER_URL = "https://fintech.morgan.edu/publications/2026-conference-proceedings/paper-05/"
DOI_URL = "https://doi.org/10.68414/NKSA6726"

TARGET_LABELS = {
    "SP500": "S&P 500",
    "DJIA": "Dow Jones Industrial Average",
}
ALLOWED_TARGETS = ["SP500", "DJIA"]
ALLOWED_LEADS = [1, 3, 5]
DEFAULT_TARGET = "SP500"
DEFAULT_LEAD = 5
TRAIN_MONTHS = 6
OBJECTIVE = "gain"
FEATURE_SET = "core"
TOP_K = 40
BASE_MODELS = ["et", "rf", "xgb_native"]

FRED_SERIES = [
    # Core live/deployment inputs used by the conference specification.
    # The broader frozen research archive still contains the additional paper variables.
    "SP500", "DJIA", "VIXCLS", "DCOILWTICO", "DGS10", "UNRATE", "FEDFUNDS",
]
TARGET_SERIES = {"SP500", "DJIA"}

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
