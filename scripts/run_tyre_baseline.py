"""Run the leave-one-race-out tyre baseline on all downloaded races."""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from vista.models.tyre_baseline import add_features, clean_laps, loro_eval  # noqa: E402


def main() -> None:
    files = sorted(Path("data/processed/laps").glob("*.parquet"))
    if len(files) < 2:
        raise SystemExit("Need at least 2 races. Run: make data")
    df = pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)
    df = add_features(clean_laps(df))
    res = loro_eval(df)
    print(res.round(3).to_string(index=False))
    print("\nMean over held-out races:")
    print(res.groupby("model")[["mae_s", "rmse_s"]].mean().round(3))
    Path("reports").mkdir(exist_ok=True)
    res.to_csv("reports/tyre_baseline_loro.csv", index=False)


if __name__ == "__main__":
    main()
