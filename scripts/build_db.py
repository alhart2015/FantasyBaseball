#!/usr/bin/env python3
"""Rebuild the SQLite database from source files."""

import json
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
    load_positions,
    load_raw_projections,
    load_ros_projections,
    load_standings,
    load_weekly_rosters,
)
from fantasy_baseball.data.yahoo_players import load_positions_cache

CONFIG_PATH = PROJECT_ROOT / "config" / "league.yaml"
PROJECTIONS_DIR = PROJECT_ROOT / "data" / "projections"
DRAFTS_PATH = PROJECT_ROOT / "data" / "historical_drafts_resolved.json"
STANDINGS_PATH = PROJECT_ROOT / "data" / "historical_standings.json"
ROSTERS_DIR = PROJECT_ROOT / "data" / "rosters"
POSITIONS_PATH = PROJECT_ROOT / "data" / "player_positions.json"
GAME_LOGS_PATH = PROJECT_ROOT / "data" / "game_logs_2026.json"
WEEKLY_ROSTERS_PATH = PROJECT_ROOT / "data" / "weekly_rosters_2026.json"
STANDINGS_2026_PATH = PROJECT_ROOT / "data" / "standings_2026.json"


def main():
    config = load_config(CONFIG_PATH)
    db_path = DB_PATH
    print(f"Building database: {db_path}")

    conn = get_connection(db_path)
    create_tables(conn)
    print("  Created tables")

    load_raw_projections(conn, PROJECTIONS_DIR)
    raw_count = conn.execute("SELECT COUNT(*) FROM raw_projections").fetchone()[0]
    print(f"  Loaded {raw_count} raw projection rows")

    if DRAFTS_PATH.exists():
        load_draft_results(conn, DRAFTS_PATH)
        draft_count = conn.execute("SELECT COUNT(*) FROM draft_results").fetchone()[0]
        print(f"  Loaded {draft_count} draft picks")

    if STANDINGS_PATH.exists():
        load_standings(conn, STANDINGS_PATH)
        standings_count = conn.execute("SELECT COUNT(*) FROM standings").fetchone()[0]
        print(f"  Loaded {standings_count} standings rows")

    # Load rosters BEFORE projections so roster names are available for quality checks
    roster_names = None
    if ROSTERS_DIR.exists():
        load_weekly_rosters(conn, ROSTERS_DIR)
        roster_count = conn.execute("SELECT COUNT(*) FROM weekly_rosters").fetchone()[0]
        print(f"  Loaded {roster_count} roster entries")
        from fantasy_baseball.data.db import get_roster_names
        roster_names = get_roster_names(conn)
        if roster_names:
            print(f"  Found {len(roster_names)} rostered players for quality checks")

    load_blended_projections(
        conn, PROJECTIONS_DIR,
        config.projection_systems, config.projection_weights,
        roster_names=roster_names, progress_cb=print,
    )
    blended_count = conn.execute("SELECT COUNT(*) FROM blended_projections").fetchone()[0]
    print(f"  Loaded {blended_count} blended projection rows")

    load_ros_projections(
        conn, PROJECTIONS_DIR,
        config.projection_systems, config.projection_weights,
        roster_names=roster_names, progress_cb=print,
    )
    ros_count = conn.execute("SELECT COUNT(*) FROM ros_blended_projections").fetchone()[0]
    print(f"  Loaded {ros_count} ROS projection rows")

    if POSITIONS_PATH.exists():
        positions = load_positions_cache(POSITIONS_PATH)
        load_positions(conn, positions)
        pos_count = conn.execute("SELECT COUNT(*) FROM positions").fetchone()[0]
        print(f"  Loaded {pos_count} player positions")

    # Load in-season snapshot data (accumulated across refreshes)
    for path, table, label in [
        (WEEKLY_ROSTERS_PATH, "weekly_rosters", "roster snapshots"),
        (STANDINGS_2026_PATH, "standings", "standings snapshots"),
        (GAME_LOGS_PATH, "game_logs", "game log entries"),
    ]:
        if not path.exists():
            continue
        rows = json.loads(path.read_text(encoding="utf-8"))
        if not rows:
            continue
        cols = list(rows[0].keys())
        placeholders = ", ".join("?" for _ in cols)
        col_names = ", ".join(cols)
        conn.executemany(
            f"INSERT OR IGNORE INTO {table} ({col_names}) VALUES ({placeholders})",
            [tuple(r[c] for c in cols) for r in rows],
        )
        conn.commit()
        count = conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
        print(f"  Loaded {count} {label}")

    conn.close()
    print("Done!")


if __name__ == "__main__":
    main()
