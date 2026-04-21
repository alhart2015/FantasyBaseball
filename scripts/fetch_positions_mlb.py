"""Fill position gaps using the MLB Stats API (free, no auth required).

Reads projection CSVs for MLBAMIDs, checks which players are missing
from the position cache, then queries the MLB API for their positions.
Merges results into player_positions.json.

Usage:
    python scripts/fetch_positions_mlb.py [--year YEAR]
"""
import argparse
import json
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from fantasy_baseball.utils.name_utils import normalize_name

CACHE_PATH = PROJECT_ROOT / "data" / "player_positions.json"
PROJECTIONS_DIR = PROJECT_ROOT / "data" / "projections"

MLB_API_BASE = "https://statsapi.mlb.com/api/v1"

# Map MLB API position abbreviations to Yahoo-style positions
MLB_POS_MAP = {
    "C": "C",
    "1B": "1B",
    "2B": "2B",
    "3B": "3B",
    "SS": "SS",
    "LF": "OF",
    "CF": "OF",
    "RF": "OF",
    "OF": "OF",
    "DH": "Util",
    "SP": "SP",
    "RP": "RP",
    "P": "P",
    "TWP": "SP",  # two-way player
}


def load_projection_mlbamids(proj_dir=None):
    """Load all player names and MLBAMIDs from projection CSVs."""
    if proj_dir is None:
        proj_dir = PROJECTIONS_DIR
    players = {}  # name -> mlbamid
    for csv_file in proj_dir.glob("*.csv"):
        df = pd.read_csv(csv_file, encoding="utf-8-sig")
        if "Name" not in df.columns or "MLBAMID" not in df.columns:
            continue
        for _, row in df.iterrows():
            name = row["Name"]
            mlbamid = row.get("MLBAMID")
            if pd.notna(name) and pd.notna(mlbamid) and name:
                players[name] = int(mlbamid)
    return players


def fetch_positions_from_mlb_api(mlbam_ids):
    """Fetch positions from MLB Stats API in batches.

    Returns dict of mlbamid -> list of positions.
    """
    results = {}
    ids_list = list(mlbam_ids)
    batch_size = 100

    for i in range(0, len(ids_list), batch_size):
        batch = ids_list[i:i + batch_size]
        ids_str = ",".join(str(x) for x in batch)
        url = f"{MLB_API_BASE}/people?personIds={ids_str}"

        try:
            req = urllib.request.Request(url)
            req.add_header("User-Agent", "FantasyBaseball/1.0")
            with urllib.request.urlopen(req, timeout=15) as resp:
                data = json.loads(resp.read().decode("utf-8"))
        except (urllib.error.URLError, urllib.error.HTTPError, TimeoutError) as e:
            print(f"  API error for batch {i}-{i+len(batch)}: {e}")
            continue

        for person in data.get("people", []):
            pid = person.get("id")
            full_name = person.get("fullName", "")
            primary_pos = person.get("primaryPosition", {})
            pos_abbr = primary_pos.get("abbreviation", "")

            yahoo_pos = MLB_POS_MAP.get(pos_abbr, pos_abbr)
            positions = [yahoo_pos] if yahoo_pos else []

            # Add IF eligibility for infielders
            if yahoo_pos in ("1B", "2B", "3B", "SS"):
                positions.append("IF")
            # Add Util for all hitters
            if yahoo_pos in ("C", "1B", "2B", "3B", "SS", "OF"):
                positions.append("Util")

            results[pid] = {"name": full_name, "positions": positions}

        # Rate limit
        if i + batch_size < len(ids_list):
            time.sleep(0.5)

    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--year", type=int, default=2026, help="Season year (default: 2026)")
    args = parser.parse_args()
    proj_dir = PROJECTIONS_DIR / str(args.year)

    # Load current position cache
    if CACHE_PATH.exists():
        with open(CACHE_PATH) as f:
            cache = json.load(f)
        print(f"Current cache: {len(cache)} players")
    else:
        cache = {}
        print("No existing cache found")

    norm_cache = {normalize_name(k) for k in cache}

    # Load projection players with MLBAMIDs
    proj_players = load_projection_mlbamids(proj_dir)
    print(f"Projection players with MLBAMID: {len(proj_players)}")

    # Find players missing from cache
    missing = {}
    for name, mlbamid in proj_players.items():
        if normalize_name(name) not in norm_cache:
            missing[name] = mlbamid

    print(f"Players missing from cache: {len(missing)}")

    if not missing:
        print("No gaps to fill!")
        return

    # Deduplicate by MLBAMID
    id_to_names = {}
    for name, mid in missing.items():
        if mid not in id_to_names:
            id_to_names[mid] = []
        id_to_names[mid].append(name)

    print(f"Unique MLBAMIDs to query: {len(id_to_names)}")
    print("Fetching from MLB Stats API...")

    api_results = fetch_positions_from_mlb_api(id_to_names.keys())
    print(f"Got positions for {len(api_results)} players")

    # Merge into cache using the projection name (not MLB API name)
    # so name matching with projections works
    added = 0
    for mid, names in id_to_names.items():
        if mid in api_results:
            api_data = api_results[mid]
            positions = api_data["positions"]
            if positions:
                for name in names:
                    if name not in cache:
                        cache[name] = positions
                        added += 1

    print(f"Added {added} players to cache")

    # Save updated cache
    with open(CACHE_PATH, "w") as f:
        json.dump(cache, f, indent=2)
    print(f"Updated cache saved: {len(cache)} total players")

    # Show some examples of what was added
    print("\nSample additions:")
    count = 0
    for mid, names in id_to_names.items():
        if mid in api_results and count < 15:
            api_data = api_results[mid]
            for name in names:
                if count < 15:
                    print(f"  {name:<30} -> {cache.get(name, '?')}")
                    count += 1


if __name__ == "__main__":
    main()
