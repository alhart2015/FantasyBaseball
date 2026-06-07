"""READ-ONLY probe: what snapshot data exists for position-aware replacement levels.

Checks the weekly roster-history dates (to pick post-draft + now), the preseason
projection pool (blended_projections), and the current full-season pool. Reads only.
"""

import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))


def _load_dotenv() -> None:
    env = Path(__file__).resolve().parent.parent / ".env"
    if not env.exists():
        return
    for line in env.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, v = line.split("=", 1)
        os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))


def main() -> None:
    _load_dotenv()
    from fantasy_baseball.data.kv_store import build_explicit_upstash_kv
    from fantasy_baseball.data.redis_store import (
        get_blended_projections,
        get_weekly_roster_history,
    )

    kv = build_explicit_upstash_kv()

    rosters = get_weekly_roster_history(kv)
    dates = sorted(d for d in rosters if d.startswith("2026-"))
    print(f"weekly_rosters_history: {len(dates)} snapshots in 2026")
    if dates:
        print(f"  earliest (post-draft?) = {dates[0]}   latest (now) = {dates[-1]}")
        print(f"  all dates: {dates}")
        first = rosters[dates[0]]
        teams = sorted({e['team'] for e in first})
        print(f"  earliest snapshot: {len(first)} roster entries across {len(teams)} teams")
        print(f"  sample entry: {first[0]}")

    for kind in ("hitters", "pitchers"):
        try:
            pool = get_blended_projections(kv, kind)
            n = len(pool) if pool else 0
            sample = pool[0] if pool else None
            print(f"\nblended_projections:{kind} (preseason pool): {n} players")
            if sample:
                print(f"  sample keys: {sorted(sample.keys())}")
        except Exception as e:  # noqa: BLE001
            print(f"\nblended_projections:{kind}: ERROR {e}")


if __name__ == "__main__":
    main()
