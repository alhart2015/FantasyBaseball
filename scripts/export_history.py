"""Dump Redis history hashes to local JSON for offline analysis.

Usage:
    python scripts/export_history.py              # dumps to data/history/YYYY-MM-DD/
    python scripts/export_history.py --out PATH   # custom output dir
"""

from __future__ import annotations

import argparse
import json
from datetime import date
from pathlib import Path

from fantasy_baseball.data.kv_store import get_kv
from fantasy_baseball.data.redis_store import (
    get_standings_history,
    get_weekly_roster_history,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("data/history") / date.today().isoformat(),
    )
    args = parser.parse_args()

    client = get_kv()

    args.out.mkdir(parents=True, exist_ok=True)

    rosters = get_weekly_roster_history(client)
    (args.out / "weekly_rosters_history.json").write_text(
        json.dumps(rosters, indent=2), encoding="utf-8"
    )
    print(f"Wrote {len(rosters)} roster snapshots -> {args.out / 'weekly_rosters_history.json'}")

    standings = get_standings_history(client)
    (args.out / "standings_history.json").write_text(
        json.dumps(standings, indent=2), encoding="utf-8"
    )
    print(f"Wrote {len(standings)} standings snapshots -> {args.out / 'standings_history.json'}")


if __name__ == "__main__":
    main()
