"""Build the park inputs for the hitter table (#433): data/hitter_ros/parks_<name>.parquet.

One row per table row: park factors of his team's home park this season and of the
parks he batted in (this season so far, last season, the last three), from the
seasons before each row's season only (see hitter_ros/parks.py). Rebuild whenever the
table or the store changes; train_hitter_ros.py --parks <name> refuses a file that
doesn't match the table.

Usage:
    python scripts/fetch_pitch_data.py --start 2008 --end 2026 --only milb   # stores ballparks
    python scripts/build_hitter_ros_parks.py --name p3   # the default park inputs
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from fantasy_baseball.hitter_ros.features import ROW_KEYS
from fantasy_baseball.hitter_ros.parks import (
    PARK_FEATURES,
    build_options,
    build_park_features,
    load_player_games,
    load_team_games,
    parks_path,
)
from fantasy_baseball.pitch_data.store import connect

TABLE = PROJECT_ROOT / "data" / "hitter_ros" / "table.parquet"
STORE = PROJECT_ROOT / "data" / "pitch_data"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--name", required=True, help="names the file: parks_<name>.parquet")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    table = pd.read_parquet(TABLE, columns=[*ROW_KEYS, "as_of", "team_id"])
    conn = connect(STORE)
    features = build_park_features(table, load_team_games(conn), load_player_games(conn))
    out = parks_path(TABLE.parent, args.name)
    features.to_parquet(out, index=False)
    out.with_suffix(".json").write_text(json.dumps(build_options(), indent=2))
    known = features[PARK_FEATURES].notna().mean().round(3)
    logging.info("wrote %s: %d rows, %d features", out, len(features), len(PARK_FEATURES))
    logging.info(
        "share known: %s", known[[f"park_{w}_avg" for w in ("home", "std", "p1", "p3")]].to_dict()
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
