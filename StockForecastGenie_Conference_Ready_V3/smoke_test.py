"""Offline smoke test for the historical model path. Does not require Streamlit or internet."""
import pandas as pd
from data_utils import load_local_history
from model_engine import historical_evaluation, prepare_supervised

raw = load_local_history()
target = "SP500"
lead = 5
supervised, _ = prepare_supervised(raw, target, lead)
# Use a late date that leaves a known future outcome in the frozen archive.
runtime = pd.Timestamp(supervised["run_day"].iloc[-40])
result = historical_evaluation(raw, target, lead, runtime)
print("runtime:", result["runtime"].date())
print("target_day:", result["target_day"].date())
print("current:", round(result["current_price"], 3))
print("forecast:", round(result["prediction"]["pred_price"], 3))
print("actual:", round(result["actual_price"], 3))
print("selected_features:", result["selected_feature_count"])
print("OK")
