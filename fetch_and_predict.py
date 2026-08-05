"""
Fetch a real race's current qualifying results directly from the Jolpica F1
API and predict both podium probabilities and the full finishing order.

Usage:
    python3 fetch_and_predict.py                  # predicts the next/most recent race automatically
    python3 fetch_and_predict.py 2026 12           # predicts a specific season+round

Notes:
- Uses the free, no-auth-required Jolpica API (Ergast's official successor):
  https://api.jolpi.ca/ergast/f1/
- If qualifying for the requested race hasn't happened yet, this falls back
  to the most recently completed qualifying session and tells you so.
"""

import sys
import pickle
import requests
import pandas as pd

BASE_URL = "https://api.jolpi.ca/ergast/f1"


def fetch_qualifying(season=None, round_=None):
    """Fetch qualifying results for a given season/round, or the latest
    available qualifying session if not specified."""
    if season and round_:
        url = f"{BASE_URL}/{season}/{round_}/qualifying/"
    else:
        # 'current/last' returns the most recently completed race weekend
        url = f"{BASE_URL}/current/last/qualifying/"

    resp = requests.get(url, timeout=15)
    resp.raise_for_status()
    data = resp.json()

    races = data["MRData"]["RaceTable"]["Races"]
    if not races:
        raise ValueError(
            f"No qualifying data found for {season or 'current'}/{round_ or 'last'} yet. "
            "Try an earlier round, or check back after Saturday qualifying."
        )
    race = races[0]
    grid = [
        (q["Driver"]["code"], int(q["position"]))
        for q in race["QualifyingResults"]
    ]
    return race["raceName"], race.get("season"), race.get("round"), grid


def predict(grid, race_label):
    with open("podium_model.pkl", "rb") as f:
        saved = pickle.load(f)
        clf, FEATURES = saved["model"], saved["features"]
    with open("full_order_model.pkl", "rb") as f:
        saved_reg = pickle.load(f)
        reg, _ = saved_reg["model"], saved_reg["features"]

    data = pd.read_csv("model_dataset.csv")

    rows = []
    for code, grid_pos in grid:
        history = data[data["code"] == code].sort_values("date")
        if history.empty:
            print(f"  (!) No historical form found for '{code}' (rookie or name mismatch), skipping")
            continue
        latest = history.iloc[-1].copy()
        latest["quali_position"] = grid_pos
        latest["grid"] = grid_pos
        latest["grid_quali_delta"] = 0
        rows.append(latest)

    pred_df = pd.DataFrame(rows)
    pred_df["pred_podium_proba"] = clf.predict_proba(pred_df[FEATURES])[:, 1]
    pred_df["pred_position"] = reg.predict(pred_df[FEATURES])
    pred_df["predicted_rank"] = pred_df["pred_position"].rank(method="first").astype(int)

    print(f"\n=== {race_label}: predicted podium probability ===")
    for _, r in pred_df.sort_values("pred_podium_proba", ascending=False).iterrows():
        bar = "#" * int(r.pred_podium_proba * 30)
        print(f"  grid P{int(r.grid):<2} {r.forename} {r.surname:<16s} {r.pred_podium_proba:>5.1%}  {bar}")

    print(f"\n=== {race_label}: predicted full finishing order ===")
    for _, r in pred_df.sort_values("predicted_rank").iterrows():
        print(f"  predicted P{r.predicted_rank:<2d} {r.forename} {r.surname:<16s} (started grid P{int(r.grid)})")


if __name__ == "__main__":
    season = sys.argv[1] if len(sys.argv) > 1 else None
    round_ = sys.argv[2] if len(sys.argv) > 2 else None

    race_name, season_used, round_used, grid = fetch_qualifying(season, round_)
    label = f"{race_name} {season_used} (round {round_used})"
    print(f"Fetched real qualifying results for: {label}")
    predict(grid, label)
