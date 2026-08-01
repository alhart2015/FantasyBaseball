"""READ-ONLY test of the load-bearing claim: is the ERoto-vs-MC leader gap an
SD-WIDTH effect? Feed ERoto's OWN live means + SDs into a hard-ranking sampler
(must reproduce score_roto's analytic total at k=1), then scale every SD by k and
see whether widening ALONE drives Hart 84.8 -> 76.7 (the live MC mean).

If a plausible k reproduces 76.7, SD width is a SUFFICIENT explanation.
If no plausible k gets there, the gap is NOT mainly an SD effect -> claim refuted.

Only .get() on Upstash. numpy RNG seeded for reproducibility.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from fantasy_baseball.data.cache_keys import CacheKey, redis_key  # noqa: E402
from fantasy_baseball.data.kv_store import build_explicit_upstash_kv  # noqa: E402
from fantasy_baseball.scoring import (  # noqa: E402
    ALL_CATS as ALL_CATEGORIES,
    INVERSE_CATS,
    score_roto,
    team_sds_from_json,
)
from fantasy_baseball.web.season_routes import _projected_as_standings  # noqa: E402

kv = build_explicit_upstash_kv()


def get_unwrapped(key: CacheKey):
    raw = kv.get(redis_key(key))
    obj = json.loads(raw) if isinstance(raw, str) else raw
    return obj["_data"] if isinstance(obj, dict) and "_data" in obj else obj


proj = get_unwrapped(CacheKey.PROJECTIONS)
standings = _projected_as_standings(proj["projected_standings"])
team_sds = team_sds_from_json(proj["team_sds"])

teams = [e.team_name for e in standings.entries]
cats = list(ALL_CATEGORIES)
n_teams, n_cats = len(teams), len(cats)

mu = np.array([[e.stats[c] for c in cats] for e in standings.entries], dtype=float)  # (T, C)
sd = np.array([[team_sds.get(e.team_name, {}).get(c, 0.0) for c in cats]
               for e in standings.entries], dtype=float)
inverse = np.array([c in INVERSE_CATS for c in cats])  # True => lower is better
hart_i = teams.index("Hart of the Order")
peanut_i = teams.index("Hello Peanuts!")

# Analytic ground truth from score_roto (what the dashboard shows).
analytic = score_roto(standings, team_sds=team_sds)
print(f"Analytic score_roto totals:  Hart={analytic['Hart of the Order'].total:.1f}  "
      f"Peanuts={analytic['Hello Peanuts!'].total:.1f}")
print(f"Live MC mean targets:        Hart~76.7  Peanuts~72.3\n")


def sim_mean_totals(k: float, n_iter: int = 4000, seed: int = 7) -> np.ndarray:
    """Hard-rank MC sampling each team-cat value ~ Normal(mu, k*sd). Returns
    mean total roto points per team over n_iter iterations."""
    rng = np.random.default_rng(seed)
    totals = np.zeros(n_teams)
    sd_k = sd * k
    for _ in range(n_iter):
        draw = rng.normal(mu, sd_k)  # (T, C)
        # For inverse cats, negate so 'higher is better' ranking is uniform.
        vals = np.where(inverse, -draw, draw)
        # points in a cat = 1 + (# teams you strictly beat). argsort-rank.
        order = np.argsort(vals, axis=0)  # ascending; worst first
        rank = np.empty_like(order)
        for c in range(n_cats):
            rank[order[:, c], c] = np.arange(n_teams)  # 0=worst .. T-1=best
        pts = rank + 1  # 1..T
        totals += pts.sum(axis=1)
    return totals / n_iter


# Validation: k=1 must reproduce the analytic totals.
v = sim_mean_totals(1.0)
print(f"k=1.00 sampler:  Hart={v[hart_i]:.1f}  Peanuts={v[peanut_i]:.1f}   "
      f"(should match analytic above -> validates the sampler)\n")

print(f"{'k':>6}{'Hart':>8}{'Peanuts':>9}{'margin':>8}")
for k in [1.0, 1.25, 1.5, 1.75, 2.0, 2.5, 3.0]:
    t = sim_mean_totals(k)
    print(f"{k:>6.2f}{t[hart_i]:>8.1f}{t[peanut_i]:>9.1f}{t[hart_i] - t[peanut_i]:>8.1f}")

print("\nRead: the k where Hart ~= 76.7 is how much WIDER the MC's effective spread")
print("would have to be than ERoto's analytic SD for SD-width to fully explain the gap.")
