"""Build the prospect-pedigree inputs for the hitter table (#433):
data/hitter_ros/pedigree_<name>.parquet.

One row per table row: MLB Pipeline list values and June-draft inputs, from lists and
drafts before the row's season (see hitter_ros/pedigree_features.py). The build options
are saved next to it as pedigree_<name>.json. Rebuild whenever the table or the
pedigree data changes; train_hitter_ros.py --pedigree <name> refuses a file that
doesn't match the table.

Usage:
    python scripts/fetch_pedigree.py   # once
    python scripts/build_hitter_ros_pedigree.py --name rookies
    python scripts/build_hitter_ros_pedigree.py --name all
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
from fantasy_baseball.hitter_ros.pedigree_features import (
    PEDIGREE_FEATURES,
    PEDIGREE_PRESETS,
    build_pedigree_features,
    pedigree_path,
)
from fantasy_baseball.pitch_data.store import connect

TABLE = PROJECT_ROOT / "data" / "hitter_ros" / "table.parquet"
STORE = PROJECT_ROOT / "data" / "pitch_data"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--name", required=True, choices=sorted(PEDIGREE_PRESETS))
    args = parser.parse_args()
    options = PEDIGREE_PRESETS[args.name]
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    table = pd.read_parquet(
        TABLE, columns=[*ROW_KEYS, "as_of", "car_pa", "std_pa", "car_seasons_in_store"]
    )
    conn = connect(STORE)
    features = build_pedigree_features(
        table,
        rankings=conn.execute("SELECT * FROM prospect_rankings").df(),
        draft=conn.execute("SELECT * FROM draft").df(),
        vet_min_pa=options["vets_blank_from"],
    )
    out = pedigree_path(TABLE.parent, args.name)
    features.to_parquet(out, index=False)
    out.with_suffix(".json").write_text(json.dumps(options, indent=2))
    logging.info("wrote %s: %d rows, %d features", out, len(features), len(PEDIGREE_FEATURES))
    known = features[PEDIGREE_FEATURES].notna().mean().round(3).to_dict()
    logging.info("share of rows with each input: %s", known)
    return 0


if __name__ == "__main__":
    sys.exit(main())
