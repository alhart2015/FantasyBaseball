"""Build the minor-league inputs for the hitter table (#435): data/hitter_ros/milb_<name>.parquet.

One row per table row: a graded minor-league line for this season so far, last season,
the last three and the career (see hitter_ros/milb_features.py). The build options are
saved next to it as milb_<name>.json. Rebuild whenever the table or the minor-league
store changes; train_hitter_ros.py --milb <name> refuses a file that doesn't match the
table.

Usage:
    python scripts/fetch_pitch_data.py --start 2008 --end 2026 --only milb   # once
    python scripts/build_hitter_ros_milb.py --name rookies-s100   # the default (a preset)
    python scripts/build_hitter_ros_milb.py --name raw
    python scripts/build_hitter_ros_milb.py --name s100 --shrink-pa 100
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
from fantasy_baseball.hitter_ros.milb_features import (
    MILB_FEATURES,
    MILB_PRESETS,
    build_milb_features,
    milb_path,
)
from fantasy_baseball.hitter_ros.milb_grade import load_milb_lines, load_mlb_lines, season_totals
from fantasy_baseball.pitch_data.store import connect

TABLE = PROJECT_ROOT / "data" / "hitter_ros" / "table.parquet"
STORE = PROJECT_ROOT / "data" / "pitch_data"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--name",
        required=True,
        help=f"names the file: milb_<name>.parquet; the presets {sorted(MILB_PRESETS)} "
        "set the options below",
    )
    parser.add_argument(
        "--shrink-pa",
        type=float,
        help="pull each window's graded rates this many PA toward its levels' average (0)",
    )
    parser.add_argument(
        "--vets-blank-from",
        type=float,
        help="no minor-league inputs for rows with at least this many MLB PA when projected "
        "(or too little history to tell)",
    )
    args = parser.parse_args()
    given = {"shrink_pa": args.shrink_pa, "vets_blank_from": args.vets_blank_from}
    preset = MILB_PRESETS.get(args.name)
    if preset is not None:
        clash = {k: v for k, v in given.items() if v is not None and v != preset[k]}
        if clash:
            parser.error(f"{args.name} is a preset {preset}; {clash} would change it")
        options = dict(preset)
    else:
        options = {"shrink_pa": args.shrink_pa or 0.0, "vets_blank_from": args.vets_blank_from}
    shrink_pa = options["shrink_pa"] or 0.0
    if shrink_pa < 0:
        parser.error("--shrink-pa must be >= 0")
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    table = pd.read_parquet(
        TABLE, columns=[*ROW_KEYS, "as_of", "car_pa", "std_pa", "car_seasons_in_store"]
    )
    conn = connect(STORE)
    window_lines = load_milb_lines(conn, by_window=True)
    features = build_milb_features(
        table,
        season_lines=season_totals(window_lines),
        window_lines=window_lines,
        mlb_lines=load_mlb_lines(conn),
        shrink_pa=shrink_pa,
        vet_min_pa=options["vets_blank_from"],
    )
    out = milb_path(TABLE.parent, args.name)
    features.to_parquet(out, index=False)
    out.with_suffix(".json").write_text(json.dumps(options, indent=2))
    has = features[[f"milb_{w}_log_pa" for w in ("std", "p1", "p3", "car")]].gt(0).mean()
    logging.info("wrote %s: %d rows, %d features", out, len(features), len(MILB_FEATURES))
    logging.info("rows with minor-league PA, by window: %s", has.round(3).to_dict())
    return 0


if __name__ == "__main__":
    sys.exit(main())
