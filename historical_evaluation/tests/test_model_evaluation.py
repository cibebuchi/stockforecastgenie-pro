"""Offline tests. No FRED/network calls and no actual model weights are needed."""
from __future__ import annotations
import sys
import types
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd

# Make integration dependencies explicit so these tests run without downloading
# private/proprietary pretrained model binaries or calling the live data API.
if "frozen_inference" not in sys.modules:
    module = types.ModuleType("frozen_inference")
    module.load_bundle = lambda *args, **kwargs: {"fake": "bundle"}
    module.predict_frozen = lambda *args, **kwargs: None
    sys.modules["frozen_inference"] = module
if "signal_logic" not in sys.modules:
    module = types.ModuleType("signal_logic")
    module.signal_details = lambda ret, score, vol: {
        "signal": "Neutral / No Clear Edge", "regime": "Normal",
        "realized_vol20_pct": vol, "confidence": score,
        "directional_threshold": 0.007, "strong_threshold": 0.014,
        "directional_confidence_threshold": 0.55, "strong_confidence_threshold": 0.65,
    }
    sys.modules["signal_logic"] = module

from historical_evaluation import evaluator as ev


def fixture_price_data():
    dates = pd.bdate_range("2025-02-01", "2026-11-13")
    n = len(dates)
    return pd.DataFrame({"Date": dates, "Close": 37000 + np.arange(n) * 17.0})


class ModelEvaluationTests(unittest.TestCase):
    def setUp(self):
        self.prices = fixture_price_data()
        self.temp = tempfile.TemporaryDirectory()
        self.models_dir = Path(self.temp.name)
        rows = [
            (t, h, "2026-10-02")
            for t in ("SP500", "DJIA") for h in (1, 3, 5)
        ]
        pd.DataFrame(rows, columns=["target", "lead", "training_end"]).to_csv(
            self.models_dir / "MODEL_MANIFEST.csv", index=False)
        self.seen = []

    def tearDown(self):
        self.temp.cleanup()

    def _predict(self, bundle, historical):
        self.seen.append(historical.copy())
        return {
            "runtime": historical["Date"].iloc[-1],
            "current_close": float(historical["Close"].iloc[-1]),
            "forecast_close": float(historical["Close"].iloc[-1] + 40),
            "forecast_return": 40 / float(historical["Close"].iloc[-1]),
            "direction_confidence": 0.57, "realized_vol20": 18.3,
            "interval_low": float(historical["Close"].iloc[-1] - 30),
            "interval_high": float(historical["Close"].iloc[-1] + 100),
        }

    def test_forecast_only_sees_selected_date_and_actual_future(self):
        with patch.object(ev, "predict_frozen", side_effect=self._predict), patch.object(ev, "signal_details", side_effect=sys.modules["signal_logic"].signal_details):
            d = ev.evaluate_one(self.prices, "2026-11-04", "DJIA", 3, self.models_dir, bundle={})
        self.assertEqual(self.seen[-1]["Date"].iloc[-1], pd.Timestamp("2026-11-04"))
        self.assertEqual(d["Outcome date"], "2026-11-09")
        self.assertEqual(d["Horizon"], 3)
        self.assertTrue(np.isfinite(d["Actual close"]))
        self.assertIn("Forecast range low", d)
        self.assertIn("Directional threshold (%)", d)
        self.assertEqual(d["Replay classification"], "Post-training replay")

    def test_pretrained_overlap_labeled(self):
        with patch.object(ev, "predict_frozen", side_effect=self._predict), patch.object(ev, "signal_details", side_effect=sys.modules["signal_logic"].signal_details):
            d = ev.evaluate_one(self.prices, "2026-09-15", "SP500", 1, self.models_dir, bundle={})
        self.assertIn("Training-period", d["Replay classification"])

    def test_missing_outcome_is_pending(self):
        # Cutoff feature history through Nov 10, so 5-day results can't be seen yet.
        prices = self.prices.loc[self.prices["Date"] <= "2026-11-11"]
        with patch.object(ev, "predict_frozen", side_effect=self._predict), patch.object(ev, "signal_details", side_effect=sys.modules["signal_logic"].signal_details):
            d = ev.evaluate_one(prices, "2026-11-10", "DJIA", 5, self.models_dir, bundle={})
        self.assertEqual(d["Status"], "Outcome pending")
        self.assertTrue(np.isnan(d["Actual close"]))

    def test_batch_loads_each_frozen_model_once(self):
        with patch.object(ev, "load_bundle", return_value={}) as load, \
             patch.object(ev, "predict_frozen", side_effect=self._predict), \
             patch.object(ev, "signal_details", side_effect=sys.modules["signal_logic"].signal_details):
            data, errors = ev.run_evaluation(self.prices, ["2026-11-03", "2026-11-04"],
                                             "DJIA", [1, 3, 5], self.models_dir)
        self.assertEqual(load.call_count, 3)
        self.assertEqual(len(data), 6)
        self.assertEqual(errors, [])

    def test_weekends_not_synthetic_sessions(self):
        with patch.object(ev, "predict_frozen", side_effect=self._predict), patch.object(ev, "signal_details", side_effect=sys.modules["signal_logic"].signal_details):
            with self.assertRaisesRegex(ev.HistoricalEvaluationError, "not an available"):
                ev.evaluate_one(self.prices, "2026-11-07", "SP500", 1, self.models_dir, bundle={})

    def test_vix_is_context_only(self):
        vix = pd.DataFrame({"Date": ["2026-11-04", "2026-11-05"],
                            "Close": [17.0, 29.0]})
        with patch.object(ev, "predict_frozen", side_effect=self._predict), patch.object(ev, "signal_details", side_effect=sys.modules["signal_logic"].signal_details):
            d = ev.evaluate_one(self.prices, "2026-11-04", "DJIA", 1, self.models_dir,
                                vix=vix, bundle={})
        self.assertEqual(d["VIX change"], 12.0)
        self.assertEqual(d["VIX regime at forecast (context)"], "Normal")
        self.assertEqual(d["VIX regime at outcome (context)"], "High")
        self.assertEqual(d["Model realized-volatility regime"], "Normal")

    def test_evaluation_ui_is_compact_and_keeps_key_results(self):
        import ast
        ui_path = Path(__file__).resolve().parents[1] / "ui.py"
        code = ui_path.read_text(encoding="utf-8")
        ast.parse(code, filename="ui.py")
        self.assertIn('st.subheader("Model Evaluation")', code)
        self.assertIn('a.metric("Predicted close"', code)
        self.assertIn('b.metric("Actual close"', code)
        self.assertIn('st.success(msg, icon="✅")', code)
        self.assertIn('st.selectbox("Show forecast details for"', code)
        self.assertIn('Unavailable trading days (weekends or market holidays) are excluded.', code)
        self.assertNotIn('Full diagnostics and evaluation provenance', code)
        self.assertNotIn('How to interpret earlier dates:', code)
        self.assertNotIn('#### Historical data access', code)
        self.assertNotIn('Fixed-model replay (dates through Oct. 2 overlap training)', code)

    def test_full_application_contains_isolated_evaluation_tab(self):
        import ast
        app = Path(__file__).resolve().parents[2] / "app.py"
        code = app.read_text(encoding="utf-8")
        ast.parse(code, filename="app.py")
        self.assertIn('"Live Demo", "Model Evaluation", "Market Trends"', code)
        self.assertEqual(code.count('elif page == "Model Evaluation":'), 1)
        self.assertIn('from historical_evaluation.ui import render as render_historical_evaluation', code)


if __name__ == "__main__":
    unittest.main()
