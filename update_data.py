"""
Re-download the latest F1 dataset CSVs so the model can be retrained on
up-to-date race results.

Source: TracingInsights/RaceData on GitHub, which mirrors the Kaggle F1
dataset and is auto-updated from the Jolpica API after each race weekend.
https://github.com/TracingInsights/RaceData

Usage:
    python3 update_data.py

Then retrain with:
    python3 f1_predictor.py
"""

import requests

BASE_URL = "https://raw.githubusercontent.com/TracingInsights/RaceData/main/data"

FILES = [
    "races", "results", "qualifying", "drivers", "constructors",
    "circuits", "driver_standings", "constructor_standings", "status",
]


def update_data():
    print("Updating F1 dataset...\n")
    for name in FILES:
        url = f"{BASE_URL}/{name}.csv"
        try:
            resp = requests.get(url, timeout=20)
            resp.raise_for_status()
        except requests.RequestException as e:
            print(f"  (!) Failed to download {name}.csv: {e}")
            continue

        with open(f"{name}.csv", "wb") as f:
            f.write(resp.content)

        n_lines = resp.text.count("\n")
        print(f"  {name}.csv updated  ({n_lines} lines)")

    print("\nDone. Run 'python f1_predictor.py' to retrain on the fresh data.")


if __name__ == "__main__":
    update_data()
