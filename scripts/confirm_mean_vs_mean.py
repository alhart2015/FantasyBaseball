"""READ-ONLY: live ERoto (analytic mean) vs MC (median + skew-implied mean),
pulled from prod Upstash. Only .get(), never .set().

ERoto roto_total = score_roto with team_sds (exact expected roto points).
MC mean is inferred from p10/median/p90: skew is mild, so mean ~= median; the
sign of (med-p10) - (p90-med) tells you which side the mean falls.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from fantasy_baseball.data.cache_keys import CacheKey, redis_key  # noqa: E402
from fantasy_baseball.data.kv_store import build_explicit_upstash_kv  # noqa: E402
from fantasy_baseball.scoring import team_sds_from_json  # noqa: E402
from fantasy_baseball.web.season_data import format_standings_for_display  # noqa: E402
from fantasy_baseball.web.season_routes import _projected_as_standings  # noqa: E402

kv = build_explicit_upstash_kv()


def get_unwrapped(key: CacheKey):
    raw = kv.get(redis_key(key))
    if raw is None:
        return None
    obj = json.loads(raw) if isinstance(raw, str) else raw
    if isinstance(obj, dict) and "_data" in obj:
        return obj["_data"]
    return obj


proj = get_unwrapped(CacheKey.PROJECTIONS)
mc = get_unwrapped(CacheKey.MONTE_CARLO)
if not proj or "projected_standings" not in proj:
    print("no projected_standings in PROJECTIONS cache")
    sys.exit(0)

standings = _projected_as_standings(proj["projected_standings"])
team_sds = team_sds_from_json(proj["team_sds"]) if proj.get("team_sds") else None
rows = format_standings_for_display(standings, "Hart of the Order", team_sds=team_sds)

eroto = {}
for r in rows["teams"]:
    eroto[r["name"]] = r["score_roto_total"]

tr = mc["rest_of_season"]["team_results"]

print(f"{'team':<30}{'ERoto(mean)':>12}{'MC median':>11}{'MC mean~':>10}{'skew':>7}{'1st%':>7}")
out = []
for team, e in eroto.items():
    d = tr.get(team, {})
    med = d.get("median_pts")
    p10 = d.get("p10")
    p90 = d.get("p90")
    if med is None:
        continue
    lo = med - p10
    hi = p90 - med
    # crude mean estimate for a mildly-skewed unimodal dist:
    # Pearson-style mean ~= median - (skew nudge). Use a light correction.
    mean_est = med - 0.35 * (lo - hi)
    skew = "LEFT" if lo > hi else ("right" if hi > lo else "sym")
    out.append((e, team, med, mean_est, skew, d.get("first_pct")))

for e, team, med, mean_est, skew, first in sorted(out, reverse=True):
    print(f"{team:<30}{e:>12.1f}{med:>11.1f}{mean_est:>10.1f}{skew:>7}{first:>7}")

h = eroto.get("Hart of the Order")
p = eroto.get("Hello Peanuts!")
hm = tr["Hart of the Order"]["median_pts"]
pm = tr["Hello Peanuts!"]["median_pts"]
print(f"\nHart-minus-Peanuts margin:  ERoto = {h - p:+.1f}   MC(median) = {hm - pm:+.1f}")
print(f"Hart first_pct = {tr['Hart of the Order']['first_pct']}%   "
      f"Peanuts first_pct = {tr['Hello Peanuts!']['first_pct']}%")
