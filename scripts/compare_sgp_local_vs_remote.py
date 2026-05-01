"""One-shot comparison of roster_audit player_sgp between local and Upstash.

Reads the LOCAL ``cache:roster_audit`` payload from the local SQLite KV
(``data/local.db``) and pulls the REMOTE same key from Upstash for a
side-by-side diff. Intended to show how SGP shifts after a config/math
change before deploying. Run a local refresh first
(``python scripts/run_lineup.py``) so the local KV is populated.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from fantasy_baseball.data.cache_keys import CacheKey, redis_key
from fantasy_baseball.data.kv_store import build_explicit_upstash_kv, get_kv


def main() -> None:
    local_kv = get_kv()
    local_raw = local_kv.get(redis_key(CacheKey.ROSTER_AUDIT))
    if local_raw is None:
        sys.exit(
            "Local cache:roster_audit is empty. Run "
            "``python scripts/run_lineup.py`` first to populate the local KV."
        )
    local = json.loads(local_raw)

    remote_kv = build_explicit_upstash_kv()
    raw = remote_kv.get(redis_key(CacheKey.ROSTER_AUDIT))
    remote = json.loads(raw) if raw else []

    local_by_id = {entry["player_id"]: entry for entry in local if entry.get("player_id")}
    remote_by_id = {entry["player_id"]: entry for entry in remote if entry.get("player_id")}

    header = f"{'player':<28}{'slot':<6}{'type':<9}{'local':>8}{'remote':>8}{'delta':>8}{'%':>8}"

    by_type: dict[str, list[tuple]] = {"hitter": [], "pitcher": []}
    missing = []
    for pid, L in local_by_id.items():
        R = remote_by_id.get(pid)
        if R is None:
            missing.append(L["player"])
            continue
        ls = L.get("player_sgp") or 0.0
        rs = R.get("player_sgp") or 0.0
        delta = ls - rs
        pct = (delta / rs * 100) if rs != 0 else 0.0
        by_type[L["player_type"]].append(
            (L["player"], L["slot"], L["player_type"], ls, rs, delta, pct)
        )

    print(header)
    print("-" * len(header))
    for t in ("hitter", "pitcher"):
        rows = sorted(by_type[t], key=lambda x: -abs(x[5]))
        for name, slot, ptype, ls, rs, delta, pct in rows:
            print(
                f"{name[:27]:<28}{slot:<6}{ptype:<9}{ls:>8.2f}{rs:>8.2f}{delta:>+8.2f}{pct:>+7.1f}%"
            )
        print()

    print("Totals (player_sgp sum):")
    for t in ("hitter", "pitcher"):
        rows = by_type[t]
        total_local = sum(r[3] for r in rows)
        total_remote = sum(r[4] for r in rows)
        print(
            f"  {t:<8}: local={total_local:.2f}  "
            f"remote={total_remote:.2f}  delta={total_local - total_remote:+.2f}"
        )
    grand_l = sum(r[3] for rows in by_type.values() for r in rows)
    grand_r = sum(r[4] for rows in by_type.values() for r in rows)
    print(f"  TOTAL   : local={grand_l:.2f}  remote={grand_r:.2f}  delta={grand_l - grand_r:+.2f}")
    if missing:
        print()
        print(f"Note: {len(missing)} local players missing from remote: {missing}")


if __name__ == "__main__":
    main()
