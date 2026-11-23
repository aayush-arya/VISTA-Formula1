"""Load real race lap data from FastF1 into a flat, analysis-ready table."""
from __future__ import annotations

from pathlib import Path

import pandas as pd

LAP_COLUMNS = [
    "Driver", "DriverNumber", "Team", "LapNumber", "LapTime", "Stint", "Compound",
    "TyreLife", "FreshTyre", "PitInTime", "PitOutTime", "Position", "TrackStatus",
    "IsAccurate", "Deleted", "LapStartTime", "Time",
]
TIME_COLUMNS = ["PitInTime", "PitOutTime", "LapStartTime", "Time"]


def enable_cache(cache_dir: str = "data/cache/fastf1") -> None:
    import fastf1

    Path(cache_dir).mkdir(parents=True, exist_ok=True)
    fastf1.Cache.enable_cache(cache_dir)


def race_id(year: int, event: str) -> str:
    return f"{year}_{event.replace(' ', '_')}"


def load_race_laps(year: int, event: str, session_name: str = "R") -> pd.DataFrame:
    """Return one row per driver lap, with times in seconds and weather attached."""
    import fastf1

    session = fastf1.get_session(year, event, session_name)
    session.load(laps=True, telemetry=False, weather=True, messages=True)
    laps = session.laps

    cols = [c for c in LAP_COLUMNS if c in laps.columns]
    df = pd.DataFrame(laps[cols]).copy()
    df["LapTimeS"] = df["LapTime"].dt.total_seconds()
    df = df.drop(columns=["LapTime"])
    for c in TIME_COLUMNS:
        if c in df.columns:
            df[c] = df[c].dt.total_seconds()
    df["IsPitLap"] = df["PitInTime"].notna() | df["PitOutTime"].notna()

    try:
        weather = laps.get_weather_data().reset_index(drop=True)
        for c in ["AirTemp", "TrackTemp", "Humidity", "Rainfall", "WindSpeed"]:
            if c in weather.columns:
                df[c] = weather[c].to_numpy()
    except Exception as exc:  # weather is optional; do not fail the whole load
        print(f"warning: weather not attached for {year} {event}: {exc}")

    df["Year"] = year
    df["Event"] = event
    df["RaceId"] = race_id(year, event)
    return df.reset_index(drop=True)


def save_laps(df: pd.DataFrame, out_dir: str = "data/processed/laps") -> Path:
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    path = Path(out_dir) / f"{df['RaceId'].iloc[0]}.parquet"
    df.to_parquet(path, index=False)
    return path
