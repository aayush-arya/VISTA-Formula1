"""Download and cache lap tables for every race listed in a config file."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from vista.data.fastf1_loader import enable_cache, load_race_laps, save_laps  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/races.yaml")
    args = ap.parse_args()

    cfg = yaml.safe_load(Path(args.config).read_text())
    enable_cache()
    for r in cfg["races"]:
        print(f"loading {r['year']} {r['event']} ...")
        df = load_race_laps(r["year"], r["event"], cfg.get("session", "R"))
        print(f"  {len(df)} laps -> {save_laps(df)}")


if __name__ == "__main__":
    main()
