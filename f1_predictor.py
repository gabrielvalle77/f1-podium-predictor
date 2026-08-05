"""
F1 Podium Predictor
====================
Predicts the probability that a driver finishes on the podium (top 3) in a
Grand Prix, using historical qualifying position, driver/constructor recent
form, and career stats.

Data source: Ergast-schema F1 dataset (1950-2026), mirrored from Kaggle by
TracingInsights/RaceData on GitHub. Covers races, results, qualifying,
drivers, constructors and standings.
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import roc_auc_score, classification_report, log_loss, mean_absolute_error
import warnings
warnings.filterwarnings("ignore")

DATA_DIR = "."

# ---------------------------------------------------------------------------
# 1. Load raw tables
# ---------------------------------------------------------------------------
races = pd.read_csv(f"{DATA_DIR}/races.csv")
results = pd.read_csv(f"{DATA_DIR}/results.csv")
qualifying = pd.read_csv(f"{DATA_DIR}/qualifying.csv")
drivers = pd.read_csv(f"{DATA_DIR}/drivers.csv")
constructors = pd.read_csv(f"{DATA_DIR}/constructors.csv")

races["date"] = pd.to_datetime(races["date"], errors="coerce")

# Only keep races that have actually happened (have a finishing position)
results = results.replace("\\N", np.nan)
results["position"] = pd.to_numeric(results["position"], errors="coerce")
results["grid"] = pd.to_numeric(results["grid"], errors="coerce")
results["points"] = pd.to_numeric(results["points"], errors="coerce")

qualifying = qualifying.replace("\\N", np.nan)
qualifying["position"] = pd.to_numeric(qualifying["position"], errors="coerce")

# ---------------------------------------------------------------------------
# 2. Build the main race-entry table
# ---------------------------------------------------------------------------
df = results.merge(races[["raceId", "year", "round", "circuitId", "date", "name"]], on="raceId")
df = df.merge(
    qualifying[["raceId", "driverId", "position"]].rename(columns={"position": "quali_position"}),
    on=["raceId", "driverId"], how="left"
)
df = df.merge(drivers[["driverId", "driverRef", "code", "forename", "surname"]], on="driverId")
df = df.merge(constructors[["constructorId", "name"]].rename(columns={"name": "constructor_name"}), on="constructorId")

# If official qualifying position is missing (older seasons), fall back to grid position
df["quali_position"] = df["quali_position"].fillna(df["grid"])

# Only keep entries where we know how the race actually finished
df = df.dropna(subset=["position"]).sort_values(["date", "raceId"]).reset_index(drop=True)

df["podium"] = (df["position"] <= 3).astype(int)
df["win"] = (df["position"] == 1).astype(int)

# ---------------------------------------------------------------------------
# 3. Feature engineering: rolling "form" features computed WITHOUT leakage
#    (only using data strictly before the current race)
# ---------------------------------------------------------------------------
df = df.sort_values(["driverId", "date"])

def rolling_form(group, col, windows=(3, 5, 10)):
    shifted = group[col].shift(1)  # exclude current race
    out = {}
    for w in windows:
        out[f"{col}_avg_last{w}"] = shifted.rolling(w, min_periods=1).mean()
    return pd.DataFrame(out, index=group.index)

driver_form = df.groupby("driverId", group_keys=False).apply(
    lambda g: rolling_form(g, "position")
)
driver_points_form = df.groupby("driverId", group_keys=False).apply(
    lambda g: rolling_form(g, "points")
)
df = pd.concat([df, driver_form, driver_points_form], axis=1)

# Driver career podium rate up to (not including) this race
df["career_podium_rate"] = (
    df.groupby("driverId")["podium"].transform(lambda s: s.shift(1).expanding().mean())
)
df["career_races"] = df.groupby("driverId").cumcount()

# Constructor recent form (avg finishing position of BOTH cars, last 5 races)
df = df.sort_values(["constructorId", "date"])
constructor_form = df.groupby("constructorId", group_keys=False).apply(
    lambda g: pd.DataFrame({
        "constructor_avg_pos_last5": g["position"].shift(1).rolling(5, min_periods=1).mean(),
        "constructor_avg_points_last5": g["points"].shift(1).rolling(5, min_periods=1).mean(),
    }, index=g.index)
)
df = pd.concat([df.drop(columns=[c for c in constructor_form.columns if c in df.columns]), constructor_form], axis=1)

df = df.sort_values(["date", "raceId"])

# Grid vs quali delta (penalty situations, e.g. grid drop)
df["grid_quali_delta"] = df["grid"] - df["quali_position"]

FEATURES = [
    "quali_position", "grid",
    "position_avg_last3", "position_avg_last5", "position_avg_last10",
    "points_avg_last3", "points_avg_last5", "points_avg_last10",
    "career_podium_rate", "career_races",
    "constructor_avg_pos_last5", "constructor_avg_points_last5",
    "grid_quali_delta",
]

model_df = df.dropna(subset=FEATURES + ["podium"]).copy()

# ---------------------------------------------------------------------------
# 4. Train / test split by SEASON (never train on the future)
# ---------------------------------------------------------------------------
test_year = model_df["year"].max()
# make sure test year actually has completed races (has 'Finished'-type results)
while model_df[model_df.year == test_year].shape[0] < 40:
    test_year -= 1

train_df = model_df[model_df["year"] < test_year]
test_df = model_df[model_df["year"] == test_year]

X_train, y_train = train_df[FEATURES], train_df["podium"]
X_test, y_test = test_df[FEATURES], test_df["podium"]

clf = RandomForestClassifier(
    n_estimators=400, max_depth=7, min_samples_leaf=15,
    class_weight="balanced", random_state=42, n_jobs=-1
)
clf.fit(X_train, y_train)

proba = clf.predict_proba(X_test)[:, 1]
pred = (proba >= 0.5).astype(int)

print(f"Training on seasons < {test_year} ({len(train_df)} entries)")
print(f"Testing on season {test_year} ({len(test_df)} entries)\n")
print(f"ROC AUC:  {roc_auc_score(y_test, proba):.3f}")
print(f"LogLoss:  {log_loss(y_test, proba):.3f}")
print()
print(classification_report(y_test, pred, target_names=["Not podium", "Podium"]))

print("Feature importances:")
for feat, imp in sorted(zip(FEATURES, clf.feature_importances_), key=lambda x: -x[1]):
    print(f"  {feat:28s} {imp:.3f}")

# ---------------------------------------------------------------------------
# 5. Per-race demo: rank predicted podium probability against reality for a
#    handful of races in the test season
# ---------------------------------------------------------------------------
demo_races = test_df["raceId"].drop_duplicates().tail(3)
for rid in demo_races:
    race_rows = test_df[test_df["raceId"] == rid].copy()
    race_rows["pred_proba"] = clf.predict_proba(race_rows[FEATURES])[:, 1]
    race_rows = race_rows.sort_values("pred_proba", ascending=False)
    race_name = races.loc[races.raceId == rid, "name"].values[0]
    race_year = races.loc[races.raceId == rid, "year"].values[0]
    print(f"\n=== {race_name} {race_year} — Top 5 predicted podium probability ===")
    for _, r in race_rows.head(5).iterrows():
        actual = f"P{int(r.position)}" if not np.isnan(r.position) else "DNF"
        print(f"  {r.forename} {r.surname:15s} pred={r.pred_proba:.2f}   actual={actual}")

# ---------------------------------------------------------------------------
# 6. Full finishing-order model: regress predicted finishing position, then
#    rank drivers within each race by that prediction to get a full order.
# ---------------------------------------------------------------------------
y_train_pos, y_test_pos = train_df["position"], test_df["position"]

reg = RandomForestRegressor(
    n_estimators=400, max_depth=8, min_samples_leaf=10,
    random_state=42, n_jobs=-1
)
reg.fit(X_train, y_train_pos)

pos_pred = reg.predict(X_test)
print(f"\nFull-order model MAE (predicted vs actual finishing position): {mean_absolute_error(y_test_pos, pos_pred):.2f} positions")

# Demo: reconstruct the predicted full order for one race and compare to reality
demo_rid = test_df["raceId"].drop_duplicates().iloc[-1]
race_rows = test_df[test_df["raceId"] == demo_rid].copy()
race_rows["pred_position"] = reg.predict(race_rows[FEATURES])
race_rows["predicted_rank"] = race_rows["pred_position"].rank(method="first").astype(int)
race_rows = race_rows.sort_values("predicted_rank")
race_name = races.loc[races.raceId == demo_rid, "name"].values[0]
race_year = races.loc[races.raceId == demo_rid, "year"].values[0]
print(f"\n=== {race_name} {race_year} — full predicted order vs actual ===")
for _, r in race_rows.iterrows():
    actual = f"P{int(r.position)}" if not np.isnan(r.position) else "DNF"
    print(f"  predicted P{r.predicted_rank:<2d} {r.forename} {r.surname:<16s} actual={actual}")

# ---------------------------------------------------------------------------
# 7. Save both trained models + the processed dataset for reuse
# ---------------------------------------------------------------------------
import pickle
with open(f"{DATA_DIR}/podium_model.pkl", "wb") as f:
    pickle.dump({"model": clf, "features": FEATURES}, f)

with open(f"{DATA_DIR}/full_order_model.pkl", "wb") as f:
    pickle.dump({"model": reg, "features": FEATURES}, f)

model_df.to_csv(f"{DATA_DIR}/model_dataset.csv", index=False)
print("\nSaved podium_model.pkl, full_order_model.pkl, and model_dataset.csv")
