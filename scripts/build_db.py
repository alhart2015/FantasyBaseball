#!/usr/bin/env python3
"""Rebuild the SQLite database from source files."""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from fantasy_baseball.config import load_config
from fantasy_baseball.data.db import (
    DB_PATH,
    create_tables,
    get_connection,
    load_blended_projections,
    load_draft_results,
    load_raw_projections,
    load_standings,
    load_weekly_rosters,
)

CONFIG_PATH = PROJECT_ROOT / "config" / "league.yaml"
PROJECTIONS_DIR = PROJECT_ROOT / "data" / "projections"
DRAFTS_PATH = PROJECT_ROOT / "data" / "historical_drafts_resolved.json"
STANDINGS_PATH = PROJECT_ROOT / "data" / "historical_standings.json"
ROSTERS_DIR = PROJECT_ROOT / "data" / "rosters"


def main():
    config = load_config(CONFIG_PATH)
    db_path = DB_PATH
    print(f"Building database: {db_path}")

    if db_path.exists():
        db_path.unlink()
        print("  Deleted existing database")

    conn = get_connection(db_path)
    create_tables(conn)
    print("  Created tables")

    load_raw_projections(conn, PROJECTIONS_DIR)
    raw_count = conn.execute("SELECT COUNT(*) FROM raw_projections").fetchone()[0]
    print(f"  Loaded {raw_count} raw projection rows")

    load_blended_projections(
        conn, PROJECTIONS_DIR,
        config.projection_systems, config.projection_weights,
    )
    blended_count = conn.execute("SELECT COUNT(*) FROM blended_projections").fetchone()[0]
    print(f"  Loaded {blended_count} blended projection rows")

    if DRAFTS_PATH.exists():
        load_draft_results(conn, DRAFTS_PATH)
        draft_count = conn.execute("SELECT COUNT(*) FROM draft_results").fetchone()[0]
        print(f"  Loaded {draft_count} draft picks")

    if STANDINGS_PATH.exists():
        load_standings(conn, STANDINGS_PATH)
        standings_count = conn.execute("SELECT COUNT(*) FROM standings").fetchone()[0]
        print(f"  Loaded {standings_count} standings rows")

    if ROSTERS_DIR.exists():
        load_weekly_rosters(conn, ROSTERS_DIR)
        roster_count = conn.execute("SELECT COUNT(*) FROM weekly_rosters").fetchone()[0]
        print(f"  Loaded {roster_count} roster entries")

    conn.close()
    print("Done!")


if __name__ == "__main__":
    main()
