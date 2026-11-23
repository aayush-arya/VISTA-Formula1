"""Tyre degradation baselines evaluated with leave-one-race-out validation.

Target: lap time delta (seconds) relative to a per-race reference pace.
Splitting by race, never by lap, avoids leakage between correlated laps.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error

COMPOUNDS = ["SOFT", "MEDIUM", "HARD", "INTERMEDIATE", "WET"]
BASE_FEATURES = ["TyreLife", "RaceProgress"] + [f"is_{c}" for c in COMPOUNDS]
OPTIONAL_FEATURES = ["TrackTemp", "Rainfall"]


def clean_laps(df: pd.DataFrame) -> pd.DataFrame:
    """Keep representative racing laps: green flag, accurate, not in/out laps."""
    ok = df["LapTimeS"].notna() & df["Compound"].isin(COMPOUNDS)
    ok &= ~df["IsPitLap"].astype(bool)
    if "IsAccurate" in df:
        ok &= df["IsAccurate"].fillna(False).astype(bool)
    if "Deleted" in df:
        ok &= ~df["Deleted"].fillna(False).astype(bool)
    if "TrackStatus" in df:
        ok &= df["TrackStatus"].astype(str).eq("1")
    return df[ok].reset_index(drop=True)


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    d = df.copy()
    d["RaceProgress"] = d["LapNumber"] / d.groupby("RaceId")["LapNumber"].transform("max")

    def reference(s: pd.Series) -> float:
        return float(s.nsmallest(max(int(len(s) * 0.25), 1)).median())

    refs = d.groupby("RaceId")["LapTimeS"].apply(reference)
    d["RefLapS"] = d["RaceId"].map(refs)
    d["DeltaS"] = d["LapTimeS"] - d["RefLapS"]
    for c in COMPOUNDS:
        d[f"is_{c}"] = (d["Compound"] == c).astype(int)
    if "Rainfall" in d:
        d["Rainfall"] = d["Rainfall"].astype(float)
    return d


def feature_list(df: pd.DataFrame) -> list[str]:
    return BASE_FEATURES + [c for c in OPTIONAL_FEATURES if c in df.columns]


def _model(kind: str):
    if kind == "linear":
        return LinearRegression()
    if kind == "gbr":
        return GradientBoostingRegressor(
            n_estimators=200, max_depth=3, learning_rate=0.05, random_state=0
        )
    raise ValueError(kind)


def loro_eval(df: pd.DataFrame, kinds=("constant", "linear", "gbr")) -> pd.DataFrame:
    """Leave-one-race-out evaluation. Returns MAE and RMSE per held-out race."""
    feats = feature_list(df)
    rows = []
    for held in sorted(df["RaceId"].unique()):
        train, test = df[df["RaceId"] != held], df[df["RaceId"] == held]
        med = train[feats].median()
        Xtr, Xte = train[feats].fillna(med), test[feats].fillna(med)
        ytr, yte = train["DeltaS"], test["DeltaS"]
        for kind in kinds:
            if kind == "constant":
                pred = np.full(len(yte), ytr.mean())
            else:
                pred = _model(kind).fit(Xtr, ytr).predict(Xte)
            rows.append({
                "held_out_race": held,
                "model": kind,
                "mae_s": mean_absolute_error(yte, pred),
                "rmse_s": mean_squared_error(yte, pred) ** 0.5,
                "n_laps": len(yte),
            })
    return pd.DataFrame(rows)
