"""Read-only: stress-test category fragility under the validators' wider SDs.

Both the methodology audit (sqrt(fraction_remaining) over-shrink + within-team
independence) and the baseball-realism scout concluded the persisted team_sds
are too TIGHT. This recomputes P(1st)/P(finish 5th+) for the user's team under
(a) the model's current SDs and (b) SDs inflated per the scout's realistic
full-season-spread targets, applied to EVERY team symmetrically.

The inflation factors are approximate (the scout's realistic-spread estimates
relative to the model's implied spread), NOT a precise recalibration -- this is
a sensitivity check on how much the fragility conclusion depends on the SDs.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from fantasy_baseball.category_odds import _GH_NODES, _GH_WEIGHTS, _SQRT_2PI, _prob_opp_above  # noqa: E402
from fantasy_baseball.data.cache_keys import CacheKey, redis_key  # noqa: E402
from fantasy_baseball.data.kv_store import build_explicit_upstash_kv  # noqa: E402
from fantasy_baseball.scoring import team_sds_from_json  # noqa: E402
from fantasy_baseball.utils.constants import ALL_CATEGORIES, INVERSE_STATS, Category  # noqa: E402

TEAM = "Hart of the Order"

# Scout's realistic-spread multipliers on the model SD (1.0 = model is fine).
INFLATE: dict[Category, float] = {
    Category.R: 1.6,
    Category.HR: 1.3,
    Category.RBI: 1.7,
    Category.SB: 1.0,
    Category.AVG: 1.4,
    Category.W: 1.4,
    Category.K: 1.3,
    Category.ERA: 1.2,
    Category.WHIP: 1.1,
    Category.SV: 1.0,
}


def _read(kv, key):
    raw = kv.get(redis_key(key))
    obj = json.loads(raw)
    return obj["_data"] if isinstance(obj, dict) and "_data" in obj else obj


def rank_pmf(means, sds, ui, *, hib):
    sign = 1.0 if hib else -1.0
    mu = [sign * m for m in means]
    opps = [(mu[i], sds[i]) for i in range(len(means)) if i != ui]
    full = [0.0] * len(means)
    for node, weight in zip(_GH_NODES, _GH_WEIGHTS, strict=True):
        x = mu[ui] + sds[ui] * node
        qs = [_prob_opp_above(x, mo, so) for mo, so in opps]
        pmf = [1.0]
        for q in qs:
            nxt = [0.0] * (len(pmf) + 1)
            for c, p in enumerate(pmf):
                nxt[c] += p * (1 - q)
                nxt[c + 1] += p * q
            pmf = nxt
        w = weight / _SQRT_2PI
        for k, p in enumerate(pmf):
            full[k] += w * p
    return full


def main() -> None:
    kv = build_explicit_upstash_kv()
    proj = _read(kv, CacheKey.PROJECTIONS)
    rows = proj["projected_standings"]["teams"]
    means = {r["name"]: {Category(k): float(v) for k, v in r["stats"].items()} for r in rows}
    sds = team_sds_from_json(proj.get("team_sds") or {})
    teams = [r["name"] for r in rows]
    ui = teams.index(TEAM)

    print(f"{'cat':<5} | {'P(1st)':>17} | {'P(finish 5th+)':>22}")
    print(f"{'':5} | {'model':>7} {'wider':>8} | {'model':>9} {'wider':>11}  x")
    print("-" * 56)
    for cat in ALL_CATEGORIES:
        hib = cat not in INVERSE_STATS
        m = [means[t][cat] for t in teams]
        s_model = [sds.get(t, {}).get(cat, 0.0) for t in teams]
        s_wide = [x * INFLATE[cat] for x in s_model]

        pmf_m = rank_pmf(m, s_model, ui, hib=hib)
        pmf_w = rank_pmf(m, s_wide, ui, hib=hib)
        p1_m, p1_w = pmf_m[0] * 100, pmf_w[0] * 100
        p5_m, p5_w = sum(pmf_m[4:]) * 100, sum(pmf_w[4:]) * 100
        mult = f"{INFLATE[cat]:.1f}x" if INFLATE[cat] != 1.0 else "--"
        print(f"{cat.value:<5} | {p1_m:>6.0f}% {p1_w:>7.0f}% | {p5_m:>8.1f}% {p5_w:>10.1f}%  {mult}")


if __name__ == "__main__":
    main()
