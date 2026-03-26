"""Fetch actual MLB season stats from the MLB Stats API.

Pulls hitter and pitcher stats for specified seasons, saves as CSVs
in data/stats/ with column names matching our projection format.

Usage:
    python scripts/fetch_actual_stats.py                    # Fetch 2022-2025
    python scripts/fetch_actual_stats.py --years 2024 2025  # Specific years
"""
import argparse
import json
import sys
import time
import urllib.request
import urllib.error
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
STATS_DIR = PROJECT_ROOT / "data" / "stats"

MLB_API_BASE = "https://statsapi.mlb.com/api/v1"
BATCH_SIZE = 500  # MLB API max per request


def fetch_stats(season: int, group: str) -> list[dict]:
    """Fetch all player stats for a season from the MLB Stats API.

    Args:
        season: MLB season year (e.g. 2024)
        group: 'hitting' or 'pitching'

    Returns:
        List of player stat dicts
    """
    sort_stat = "plateAppearances" if group == "hitting" else "inningsPitched"
    all_splits = []
    offset = 0

    while True:
        url = (
            f"{MLB_API_BASE}/stats"
            f"?stats=season&sportId=1&season={season}"
            f"&group={group}&gameType=R"
            f"&limit={BATCH_SIZE}&offset={offset}"
            f"&sortStat={sort_stat}&order=desc"
            f"&playerPool=ALL"
        )
        req = urllib.request.Request(url, headers={"User-Agent": "FantasyBaseball/1.0"})
        try:
            with urllib.request.urlopen(req, timeout=30) as resp:
                data = json.loads(resp.read())
        except (urllib.error.URLError, urllib.error.HTTPError, TimeoutError) as e:
            print(f"  API error at offset {offset}: {e}")
            break

        stats_block = data.get("stats", [{}])[0]
        splits = stats_block.get("splits", [])
        if not splits:
            break

        all_splits.extend(splits)
        total = stats_block.get("totalSplits", 0)

        if offset + BATCH_SIZE >= total:
            break
        offset += BATCH_SIZE
        time.sleep(0.3)  # Rate limit

    return all_splits


def parse_ip(ip_str: str) -> float:
    """Parse innings pitched string (e.g. '208.2') to float (208.667)."""
    if "." in str(ip_str):
        parts = str(ip_str).split(".")
        whole = int(parts[0])
        thirds = int(parts[1]) if len(parts) > 1 else 0
        return whole + thirds / 3.0
    return float(ip_str)


def splits_to_hitter_rows(splits: list[dict], min_pa: int = 50) -> list[dict]:
    """Convert MLB API splits to hitter CSV rows."""
    rows = []
    for s in splits:
        stat = s["stat"]
        pa = stat.get("plateAppearances", 0)
        if pa < min_pa:
            continue

        player = s.get("player", {})
        team = s.get("team", {})
        ab = stat.get("atBats", 0)
        h = stat.get("hits", 0)
        avg = h / ab if ab > 0 else 0

        rows.append({
            "Name": player.get("fullName", ""),
            "Team": team.get("name", "").split()[-1] if team.get("name") else "",
            "G": stat.get("gamesPlayed", 0),
            "PA": pa,
            "AB": ab,
            "H": h,
            "HR": stat.get("homeRuns", 0),
            "R": stat.get("runs", 0),
            "RBI": stat.get("rbi", 0),
            "SB": stat.get("stolenBases", 0),
            "AVG": round(avg, 3),
            "MLBAMID": player.get("id", ""),
        })
    return rows


def splits_to_pitcher_rows(splits: list[dict], min_ip: float = 10.0) -> list[dict]:
    """Convert MLB API splits to pitcher CSV rows."""
    rows = []
    for s in splits:
        stat = s["stat"]
        ip = parse_ip(stat.get("inningsPitched", "0"))
        if ip < min_ip:
            continue

        player = s.get("player", {})
        team = s.get("team", {})
        er = stat.get("earnedRuns", 0)
        bb = stat.get("baseOnBalls", 0)
        h_allowed = stat.get("hits", 0)
        era = er * 9 / ip if ip > 0 else 0
        whip = (bb + h_allowed) / ip if ip > 0 else 0

        rows.append({
            "Name": player.get("fullName", ""),
            "Team": team.get("name", "").split()[-1] if team.get("name") else "",
            "W": stat.get("wins", 0),
            "L": stat.get("losses", 0),
            "SV": stat.get("saves", 0),
            "G": stat.get("gamesPitched", stat.get("gamesPlayed", 0)),
            "GS": stat.get("gamesStarted", 0),
            "IP": round(ip, 1),
            "SO": stat.get("strikeOuts", 0),
            "ER": er,
            "BB": bb,
            "H": h_allowed,
            "ERA": round(era, 2),
            "WHIP": round(whip, 2),
            "MLBAMID": player.get("id", ""),
        })
    return rows


def write_csv(rows: list[dict], path: Path):
    """Write rows to CSV."""
    if not rows:
        print(f"  No data to write for {path.name}")
        return
    cols = list(rows[0].keys())
    with open(path, "w", encoding="utf-8", newline="") as f:
        f.write(",".join(cols) + "\n")
        for row in rows:
            vals = []
            for c in cols:
                v = row[c]
                if isinstance(v, str) and ("," in v or '"' in v):
                    v = f'"{v}"'
                vals.append(str(v))
            f.write(",".join(vals) + "\n")
    print(f"  Wrote {len(rows)} rows to {path.name}")


def main():
    parser = argparse.ArgumentParser(description="Fetch actual MLB stats from MLB API")
    parser.add_argument(
        "--years", type=int, nargs="+", default=[2022, 2023, 2024],
        help="Seasons to fetch (default: 2022 2023 2024)",
    )
    parser.add_argument(
        "--min-pa", type=int, default=50,
        help="Minimum PA for hitters (default: 50)",
    )
    parser.add_argument(
        "--min-ip", type=float, default=10.0,
        help="Minimum IP for pitchers (default: 10)",
    )
    args = parser.parse_args()

    STATS_DIR.mkdir(parents=True, exist_ok=True)

    for year in args.years:
        print(f"\n=== {year} ===")

        # Hitters
        print(f"  Fetching {year} hitting stats...")
        h_splits = fetch_stats(year, "hitting")
        print(f"  Got {len(h_splits)} hitter records")
        h_rows = splits_to_hitter_rows(h_splits, min_pa=args.min_pa)
        write_csv(h_rows, STATS_DIR / f"hitters-{year}.csv")

        time.sleep(0.5)

        # Pitchers
        print(f"  Fetching {year} pitching stats...")
        p_splits = fetch_stats(year, "pitching")
        print(f"  Got {len(p_splits)} pitcher records")
        p_rows = splits_to_pitcher_rows(p_splits, min_ip=args.min_ip)
        write_csv(p_rows, STATS_DIR / f"pitchers-{year}.csv")

        time.sleep(0.5)

    print("\nDone.")


if __name__ == "__main__":
    main()
