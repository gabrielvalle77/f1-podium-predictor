# F1 Podium Predictor

A machine learning model that predicts the probability of a driver finishing
on the podium (top 3), trained on real historical F1 data (1950–2026).

## Data

CSV files in Ergast schema (races, results, qualifying, drivers, constructors,
standings), sourced from `TracingInsights/RaceData` on GitHub, which mirrors
the Kaggle "Formula 1 World Championship" dataset and auto-updates from the
Jolpica API (the Ergast API's successor).

## Files

- `f1_predictor.py` — loads the data, engineers features, trains a
  `RandomForestClassifier` (podium yes/no) AND a `RandomForestRegressor`
  (predicted finishing position, used to reconstruct a full predicted race
  order), evaluates both on the most recent season, and saves the trained
  models (`podium_model.pkl`, `full_order_model.pkl`) + processed dataset
  (`model_dataset.csv`).
- `predict_race.py` — loads the saved models and predicts podium probability
  + full finishing order for a custom starting grid you type in by hand
  (edit the `GRID` list at the top).
- `fetch_and_predict.py` — **fully automatic**: pulls the real, current
  qualifying results straight from the live Jolpica F1 API and predicts that
  race for you, no manual data entry needed.
  ```bash
  python3 fetch_and_predict.py                # most recently completed qualifying
  python3 fetch_and_predict.py 2026 12         # a specific season + round
  ```
  Note: this calls `https://api.jolpi.ca/ergast/f1/` directly over the
  internet using `requests`, so it needs normal (unrestricted) network
  access — it will work fine on your own machine.

## How it works

For every driver in every historical race, the model uses only information
that would have been known *before* that race (no data leakage):

- Qualifying / grid position
- Rolling average finishing position over their last 3/5/10 races
- Rolling average points over their last 3/5/10 races
- Career podium rate and number of career races so far
- Constructor's average finishing position/points over its last 5 races
  (both cars combined)
- Grid vs. qualifying delta (penalties, grid drops)

It trains on all seasons before the most recent one and tests on the most
recent completed season, so the reported accuracy reflects genuine
out-of-sample performance.

## Running it

```bash
pip install -r requirements.txt
python3 f1_predictor.py         # (re)trains + evaluates + saves both models
python3 predict_race.py         # predicts a grid you type in yourself
python3 fetch_and_predict.py    # predicts using a REAL, auto-fetched grid
```

The trained models (`podium_model.pkl`, `full_order_model.pkl`) and
`model_dataset.csv` are already included, so `predict_race.py` and
`fetch_and_predict.py` work immediately without retraining.

## Current performance

- **Podium classifier:** ~0.92 ROC AUC on held-out season data — quite good
  at ranking who's *likely* to podium.
- **Full-order regressor:** ~3.6 positions mean absolute error on held-out
  data — it reliably identifies the front-runners and correctly picked the
  actual race winner in testing, but exact midfield order is inherently
  noisy (that's motorsport — safety cars, strategy, mechanical failures, and
  one bad lap can flip everything).

## Ideas to extend it

- Add circuit-specific features (some drivers/teams are stronger at certain
  tracks)
- Model DNF risk separately (reliability varies a lot by team/era)
- Try gradient boosting (XGBoost/LightGBM) instead of Random Forest
- Predict exact finishing position instead of just podium/not
- Pull this season's live qualifying results automatically from the
  Jolpica API (https://api.jolpi.ca/ergast/f1/) instead of typing them by hand
