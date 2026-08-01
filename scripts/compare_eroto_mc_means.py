"""READ-ONLY: reconstruct the EXACT live MC inputs from prod Upstash and run the
MC loop capturing per-team per-CATEGORY means, then diff against ERoto's
per-category projected means. Isolates whether the two engines see different
INPUTS (and where) vs just different variance.

Same rosters feed both engines (verified: refresh_pipeline all_team_rosters ==
opp_rosters + matched). This measures what the MC's processing
(full-season-minus-YTD, per-iteration active-lineup argmax, max(0) clamp,
mean_scale haircut, YTD-AB fallback) does to the per-category MEANS.

Only .get(). seed=42, n=1000 to match the live pipeline; validates against the
cached median/first_pct.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import yaml  # noqa: E402

from fantasy_baseball.data.cache_keys import CacheKey, redis_key  # noqa: E402
from fantasy_baseball.data.kv_store import build_explicit_upstash_kv  # noqa: E402
from fantasy_baseball.models.positions import BENCH_SLOTS  # noqa: E402
from fantasy_baseball.scoring import ALL_CATS, score_roto_dict  # noqa: E402
from fantasy_baseball.simulation import simulate_remaining_season  # noqa: E402
from fantasy_baseball.web.season_routes import _projected_as_standings  # noqa: E402

kv = build_explicit_upstash_kv()


def g(key: CacheKey):
    raw = kv.get(redis_key(key))
    if raw is None:
        return None
    o = json.loads(raw) if isinstance(raw, str) else raw
    return o["_data"] if isinstance(o, dict) and "_data" in o else o


USER = "Hart of the Order"
cfg = yaml.safe_load((ROOT / "config" / "league.yaml").read_text())
slots = cfg["roster_slots"]
non_hitter = {str(s) for s in BENCH_SLOTS} | {"P", "BN", "IL"}
h_slots = sum(v for k, v in slots.items() if k not in non_hitter and k != "P")
p_slots = slots.get("P", 9)

opp = g(CacheKey.OPP_ROSTERS)
user_roster = g(CacheKey.ROSTER)
standings = g(CacheKey.STANDINGS)
proj = g(CacheKey.PROJECTIONS)
mc_cached = g(CacheKey.MONTE_CARLO)["rest_of_season"]["team_results"]
fr = proj["fraction_remaining"]

team_rosters = {USER: user_roster, **opp}

actual_standings_dict: dict[str, dict] = {}
for t in standings["teams"]:
    row = dict(t["stats"])
    ip = t.get("extras", {}).get("IP")
    if ip is not None:
        row["IP"] = float(ip)
    actual_standings_dict[t["name"]] = row

print(f"h_slots={h_slots} p_slots={p_slots} fr={fr:.4f}  teams={len(team_rosters)}")

# --- ERoto per-category means from cached projected_standings ---
eroto_std = _projected_as_standings(proj["projected_standings"])
eroto_cat = {e.team_name: {c: e.stats[c] for c in ALL_CATS} for e in eroto_std.entries}

# --- Run the MC loop, accumulate per-team per-cat sums + roto totals ---
n_iter = 1000
rng = np.random.default_rng(42)
flat = {tk: [_p for _p in players] for tk, players in team_rosters.items()}
cat_keys = [c.value for c in ALL_CATS]
sums = {t: {c: 0.0 for c in cat_keys} for t in team_rosters}
roto_tot = {t: 0.0 for t in team_rosters}
wins = {t: 0 for t in team_rosters}

# simulate_remaining_season expects full-season-flattened dicts; it reads nested
# full_season_projection via _apply_variance? No -- run_ros_monte_carlo flattens
# first. Replicate that flatten here.
from fantasy_baseball.simulation import _flatten_full_season  # noqa: E402

flat = {tk: [_flatten_full_season(p) for p in players] for tk, players in team_rosters.items()}

for _ in range(n_iter):
    sim_stats, _inj = simulate_remaining_season(
        actual_standings_dict, flat, fr, rng, h_slots, p_slots
    )
    for t, s in sim_stats.items():
        for c in cat_keys:
            sums[t][c] += s.get(c, 0.0)
    roto = score_roto_dict(sim_stats)
    ranked = sorted(roto.items(), key=lambda x: x[1]["total"], reverse=True)
    for rank, (name, pts) in enumerate(ranked, 1):
        roto_tot[name] += pts["total"]
        if rank == 1:
            wins[name] += 1

mc_cat = {t: {c: sums[t][c] / n_iter for c in cat_keys} for t in team_rosters}
mc_total = {t: roto_tot[t] / n_iter for t in team_rosters}

# --- Validate reconstruction against cached medians/first_pct ---
print("\nVALIDATION (my run vs cached live MC):")
for t in [USER, "Hello Peanuts!"]:
    print(f"  {t:<20} my mean_total={mc_total[t]:5.1f} my 1st%={100*wins[t]/n_iter:5.1f}"
          f"   cached median={mc_cached[t]['median_pts']} cached 1st%={mc_cached[t]['first_pct']}")

INV = {"ERA", "WHIP"}
print("\nPER-CATEGORY MEANS  ERoto vs MC  (diff = MC - ERoto; *=helps that team's rank)")
for t in [USER, "Hello Peanuts!"]:
    print(f"\n== {t} ==   ERoto roto={sum(1 for _ in [0]) and ''}", end="")
    print(f"ERoto total={sum(eroto_cat[t][c] for c in []) or ''}")
    print(f"  {'cat':>5}{'ERoto':>10}{'MC':>10}{'diff':>9}")
    for c in ALL_CATS:
        e = eroto_cat[t][c]
        m = mc_cat[t][c.value]
        print(f"  {c.value:>5}{e:>10.2f}{m:>10.2f}{m - e:>+9.2f}")
