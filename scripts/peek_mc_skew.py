"""READ-ONLY: pull the live Monte Carlo + projected standings from prod Upstash
and inspect leader skew (does mean sit below or above the median?).

Only calls .get() -- never .set(). Settles the mean-vs-median direction with
live data instead of argument.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from fantasy_baseball.data.cache_keys import CacheKey, redis_key  # noqa: E402
from fantasy_baseball.data.kv_store import build_explicit_upstash_kv  # noqa: E402

kv = build_explicit_upstash_kv()


def get_unwrapped(key: CacheKey):
    raw = kv.get(redis_key(key))
    if raw is None:
        return None
    obj = json.loads(raw) if isinstance(raw, str) else raw
    if isinstance(obj, dict) and "_data" in obj:
        return obj["_data"], obj.get("_meta", {})
    return obj, {}


mc, mc_meta = get_unwrapped(CacheKey.MONTE_CARLO) or (None, {})
if mc is None:
    print("NO monte_carlo cache in prod Upstash")
    sys.exit(0)

print("monte_carlo _meta:", json.dumps(mc_meta)[:300])
print("monte_carlo top keys:", list(mc.keys()) if isinstance(mc, dict) else type(mc))

ros = mc.get("rest_of_season") if isinstance(mc, dict) else None
if not ros:
    print("no rest_of_season block; keys:", list(mc.keys()))
    sys.exit(0)

tr = ros.get("team_results", {})
print(f"\nrest_of_season MC (use_management=False), {len(tr)} teams")
print(f"{'team':<30}{'p10':>6}{'median':>8}{'p90':>6}{'lo=med-p10':>11}{'hi=p90-med':>11}{'skew':>7}{'1st%':>6}")
rows = []
for team, d in tr.items():
    med = d.get("median_pts")
    p10 = d.get("p10")
    p90 = d.get("p90")
    if med is None or p10 is None or p90 is None:
        continue
    lo = med - p10
    hi = p90 - med
    skew = "LEFT" if lo > hi else ("right" if hi > lo else "sym")
    rows.append((med, team, p10, p90, lo, hi, skew, d.get("first_pct")))
for med, team, p10, p90, lo, hi, skew, first in sorted(rows, reverse=True):
    print(f"{team:<30}{p10:>6}{med:>8}{p90:>6}{lo:>11.1f}{hi:>11.1f}{skew:>7}{first:>6}")

print("\nLEFT skew => mean < median (the median OVERSTATES that team's expectation).")
print("right skew => mean > median (the median UNDERSTATES it).")
