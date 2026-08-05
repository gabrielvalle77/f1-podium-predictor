"""
Predict podium probabilities for an upcoming race given a starting grid.

Usage: edit the GRID list below (driver code + starting position) and run:
    python3 predict_race.py

It pulls each driver's latest known form (rolling averages, career podium
rate, constructor form) from model_dataset.csv and combines it with the
grid position you give it.
"""

import pandas as pd
import pickle

with open("podium_model.pkl", "rb") as f:
    saved = pickle.load(f)
    clf, FEATURES = saved["model"], saved["features"]

data = pd.read_csv("model_dataset.csv")
data["date"] = pd.to_datetime(data["date"])

# ---------------------------------------------------------------------------
# EDIT THIS: starting grid for the race you want to predict.
# Use each driver's 3-letter code (as in drivers.csv 'code' column) and
# their grid position for this race.
# ---------------------------------------------------------------------------
GRID = [
    ("VER", 1), ("NOR", 2), ("PIA", 3), ("LEC", 4), ("HAM", 5),
    ("RUS", 6), ("ANT", 7), ("ALO", 8), ("HAD", 9), ("GAS", 10),
]

rows = []
for code, grid_pos in GRID:
    driver_history = data[data["code"] == code].sort_values("date")
    if driver_history.empty:
        print(f"  (!) No history found for driver code '{code}', skipping")
        continue
    latest = driver_history.iloc[-1].copy()  # most recent known form
    latest["quali_position"] = grid_pos
    latest["grid"] = grid_pos
    latest["grid_quali_delta"] = 0
    rows.append(latest)

pred_df = pd.DataFrame(rows)
pred_df["pred_podium_proba"] = clf.predict_proba(pred_df[FEATURES])[:, 1]
pred_df = pred_df.sort_values("pred_podium_proba", ascending=False)

print("Predicted podium probability (based on grid position + recent form):\n")
for _, r in pred_df.iterrows():
    bar = "#" * int(r.pred_podium_proba * 30)
    print(f"  P{int(r.grid):<2} {r.forename} {r.surname:<14s} {r.pred_podium_proba:>5.1%}  {bar}")
