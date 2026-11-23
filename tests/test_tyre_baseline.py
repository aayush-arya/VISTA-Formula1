import numpy as np
import pandas as pd

from vista.models.tyre_baseline import add_features, clean_laps, loro_eval


def synthetic_laps(n_races=3, laps=50, seed=0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows = []
    for r in range(n_races):
        base = 85 + 5 * r
        for drv, comp in [("AAA", "SOFT"), ("BBB", "MEDIUM")]:
            rate = 0.10 if comp == "SOFT" else 0.05
            for lap in range(1, laps + 1):
                rows.append({
                    "RaceId": f"race{r}", "Driver": drv, "LapNumber": lap,
                    "Compound": comp, "TyreLife": lap, "IsPitLap": False,
                    "IsAccurate": True, "Deleted": False, "TrackStatus": "1",
                    "TrackTemp": 30 + rng.normal(), "Rainfall": False,
                    "LapTimeS": base + rate * lap - 0.03 * lap + rng.normal(0, 0.15),
                })
    return pd.DataFrame(rows)


def test_clean_laps_drops_pit_and_flagged_laps():
    df = synthetic_laps()
    df.loc[0, "IsPitLap"] = True
    df.loc[1, "TrackStatus"] = "4"
    assert len(clean_laps(df)) == len(df) - 2


def test_loro_models_beat_constant_baseline():
    df = add_features(clean_laps(synthetic_laps()))
    res = loro_eval(df)
    assert set(res["model"]) == {"constant", "linear", "gbr"}
    assert res["held_out_race"].nunique() == 3
    mean = res.groupby("model")["mae_s"].mean()
    assert mean["linear"] < mean["constant"]
    assert mean["gbr"] < mean["constant"]
