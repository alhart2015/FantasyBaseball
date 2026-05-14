#!/usr/bin/env python3
"""Rebuild the SQLite database from source files.

Usage:
    python scripts/build_db.py            # Build/update the database
    python scripts/build_db.py --export   # Export in-season data to JSON for git
"""

import argparse
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
    load_draft_results,
    load_positions,
    load_raw_projections,
    load_weekly_rosters,
)
from fantasy_baseball.data.yahoo_players import load_positions_cache

CONFIG_PATH = PROJECT_ROOT / "config" / "league.yaml"
PROJECTIONS_DIR = PROJECT_ROOT / "data" / "projections"
DRAFTS_PATH = PROJECT_ROOT / "data" / "historical_drafts_resolved.json"
ROSTERS_DIR = PROJECT_ROOT / "data" / "rosters"
POSITIONS_PATH = PROJECT_ROOT / "data" / "player_positions.json"
WEEKLY_ROSTERS_PATH = PROJECT_ROOT / "data" / "weekly_rosters_2026.json"


SNAPSHOT_TABLES = [
    (
        "weekly_rosters",
        WEEKLY_ROSTERS_PATH,
        "roster snapshots",
        "SELECT * FROM weekly_rosters WHERE snapshot_date >= '2026-'",
    ),
]


def export_snapshots():
    """Export in-season data from SQLite to JSON files for git."""
    conn = get_connection(DB_PATH)
    conn.row_factory = __import__("sqlite3").Row
    for _table, path, label, query in SNAPSHOT_TABLES:
        rows = conn.execute(query).fetchall()
        data = [dict(r) for r in rows]
        path.write_text(json.dumps(data, indent=2), encoding="utf-8")
        print(f"  Exported {len(data)} {label} -> {path.name}")
    conn.close()
    print("Done! Commit the updated JSON files to git.")


def main():
    parser = argparse.ArgumentParser(description="Build or export the SQLite database.")
    parser.add_argument(
        "--export", action="store_true", help="Export in-season snapshot data to JSON for git"
    )
    args = parser.parse_args()

    if args.export:
        export_snapshots()
        return

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

    from fantasy_baseball.data.kv_store import get_kv
    from fantasy_baseball.data.projections import blend_projections
    from fantasy_baseball.data.redis_store import set_blended_projections
    from fantasy_baseball.utils.time_utils import local_today

    current_year = local_today().year
    year_dir = PROJECTIONS_DIR / str(current_year)
    if year_dir.exists():
        hitters_df, pitchers_df, _ = blend_projections(
            year_dir,
            config.projection_systems,
            config.projection_weights,
            roster_names=roster_names,
            progress_cb=print,
        )
        client = get_kv()
        set_blended_projections(client, "hitters", hitters_df.to_dict(orient="records"))
        set_blended_projections(client, "pitchers", pitchers_df.to_dict(orient="records"))
        print(f"  Loaded {len(hitters_df)} hitters + {len(pitchers_df)} pitchers to Redis")
    else:
        raise FileNotFoundError(
            f"Preseason projections directory not found: {year_dir}. "
            f"Expected data/projections/{current_year}/ with {{system}}-hitters.csv / "
            f"{{system}}-pitchers.csv files for each configured system."
        )

    if POSITIONS_PATH.exists():
        positions = load_positions_cache(POSITIONS_PATH)
        load_positions(conn, positions)
        pos_count = conn.execute("SELECT COUNT(*) FROM positions").fetchone()[0]
        print(f"  Loaded {pos_count} player positions")

    # Load in-season snapshot data (accumulated across refreshes)
    for table, path, label, _query in SNAPSHOT_TABLES:
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
