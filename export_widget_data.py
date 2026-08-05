"""
Exports the trained model's predictions + historical stats to a single JSON
file consumed by the F1 Predictor web artifact (static HTML/JS dashboard).

Run this whenever the models are retrained (after f1_predictor.py) to refresh
the artifact's data:
    python export_widget_data.py
"""

import json
import pickle
import numpy as np
import pandas as pd

with open("podium_model.pkl", "rb") as f:
    saved = pickle.load(f)
    clf, FEATURES = saved["model"], saved["features"]

with open("full_order_model.pkl", "rb") as f:
    saved_reg = pickle.load(f)
    reg = saved_reg["model"]

data = pd.read_csv("model_dataset.csv")
data["date"] = pd.to_datetime(data["date"])

# Latest row per driver code = their current team + most recent known form
latest_idx = data.groupby("code")["date"].idxmax()
latest = data.loc[latest_idx].copy()

# Only keep drivers active in the last 2 seasons (drop long-retired codes)
current_year = data["year"].max()
latest = latest[latest["year"] >= current_year - 1].sort_values("surname")

TEAM_COLORS = {
    "McLaren": "#FF8000",
    "Ferrari": "#E8002D",
    "Red Bull": "#3671C6",
    "Mercedes": "#27F4D2",
    "Aston Martin": "#229971",
    "Alpine F1 Team": "#00A1E8",
    "Williams": "#1868DB",
    "RB F1 Team": "#6C98FF",
    "Haas F1 Team": "#B6BABD",
    "Sauber": "#52C832",
    "Audi": "#D50000",
    "Cadillac": "#001E50",
}

grid_positions = list(range(1, 21))
drivers_out = []

for _, row in latest.iterrows():
    proba_by_grid = []
    predpos_by_grid = []
    for g in grid_positions:
        r = row.copy()
        r["quali_position"] = g
        r["grid"] = g
        r["grid_quali_delta"] = 0
        X = pd.DataFrame([r[FEATURES]])
        proba_by_grid.append(round(float(clf.predict_proba(X)[:, 1][0]), 4))
        predpos_by_grid.append(round(float(reg.predict(X)[0]), 2))

    drivers_out.append({
        "code": row["code"],
        "name": f"{row['forename']} {row['surname']}",
        "team": row["constructor_name"],
        "color": TEAM_COLORS.get(row["constructor_name"], "#888888"),
        "careerRaces": int(row["career_races"]),
        "careerPodiumRate": round(float(row["career_podium_rate"]) if not pd.isna(row["career_podium_rate"]) else 0, 4),
        "avgPosLast5": round(float(row["position_avg_last5"]), 2) if not pd.isna(row["position_avg_last5"]) else None,
        "avgPtsLast5": round(float(row["points_avg_last5"]), 2) if not pd.isna(row["points_avg_last5"]) else None,
        "constructorAvgPosLast5": round(float(row["constructor_avg_pos_last5"]), 2) if not pd.isna(row["constructor_avg_pos_last5"]) else None,
        "probaByGrid": proba_by_grid,
        "predPosByGrid": predpos_by_grid,
    })

# ---------------------------------------------------------------------------
# All-time career leaderboard (uses full historical dataset, all seasons)
# ---------------------------------------------------------------------------
results = pd.read_csv("results.csv").replace("\\N", np.nan)
results["position"] = pd.to_numeric(results["position"], errors="coerce")
drivers_csv = pd.read_csv("drivers.csv")

career = results.merge(drivers_csv[["driverId", "forename", "surname", "code"]], on="driverId")
career["podium"] = (career["position"] <= 3).astype(int)
career["win"] = (career["position"] == 1).astype(int)

leaderboard = (
    career.groupby(["forename", "surname"])
    .agg(races=("raceId", "count"), podiums=("podium", "sum"), wins=("win", "sum"))
    .reset_index()
    .sort_values("podiums", ascending=False)
    .head(10)
)
leaderboard_out = [
    {
        "name": f"{r.forename} {r.surname}",
        "races": int(r.races),
        "podiums": int(r.podiums),
        "wins": int(r.wins),
    }
    for r in leaderboard.itertuples()
]

# ---------------------------------------------------------------------------
# Model performance headline stats (static, from README / last training run)
# ---------------------------------------------------------------------------
model_stats = {
    "rocAuc": 0.919,
    "mae": 3.58,
    "trainSeasons": f"1950-{int(current_year) - 1}",
    "totalRaceEntries": int(len(data)),
}

output = {
    "generatedFrom": f"model_dataset.csv (through season {int(current_year)})",
    "drivers": drivers_out,
    "leaderboard": leaderboard_out,
    "modelStats": model_stats,
}

with open("widget_data.json", "w", encoding="utf-8") as f:
    json.dump(output, f, ensure_ascii=False, indent=2)

print(f"Exported {len(drivers_out)} active drivers, {len(leaderboard_out)} leaderboard entries -> widget_data.json")
