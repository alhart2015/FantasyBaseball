"""READ-ONLY experiment: quantify the MC playing-time fixes.

Reconstructs the exact ROS Monte Carlo inputs from prod cache (rosters with
full-season projections, actual YTD standings, fraction_remaining, slot counts),
reproduces the live baseline, then sweeps two levers in isolation:

  1. PLAYING_TIME_MAX_SCALE  -- the over-performance clip ceiling (currently 2.0).
  2. Replacement backfill     -- zeroing REPLACEMENT_* removes the per-player
                                 injury fill so depth comes only from top-N.

Only reads from Upstash; never writes. Monkeypatches are process-local.
"""

import json
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


def _unwrap(raw):
    if raw is None:
        return None
    obj = json.loads(raw)
    if isinstance(obj, dict) and "_data" in obj and "_meta" in obj:
        return obj["_data"]
    return obj


def main() -> None:
    _load_dotenv()
    from fantasy_baseball import simulation
    from fantasy_baseball.data.cache_keys import CacheKey, redis_key
    from fantasy_baseball.data.kv_store import build_explicit_upstash_kv
    from fantasy_baseball.models.player import Player
    from fantasy_baseball.models.positions import BENCH_SLOTS
    from fantasy_baseball.models.standings import Standings
    from fantasy_baseball.simulation import run_ros_monte_carlo
    from fantasy_baseball.utils.constants import AB_PER_PA, OpportunityStat
    from fantasy_baseball.web.season_routes import _load_config

    kv = build_explicit_upstash_kv()

    def read(key):
        return _unwrap(kv.get(redis_key(key)))

    opp_raw = read(CacheKey.OPP_ROSTERS) or {}
    user_raw = read(CacheKey.ROSTER) or []
    proj = read(CacheKey.PROJECTIONS) or {}
    standings_raw = read(CacheKey.STANDINGS)

    config = _load_config()
    user_name = config.team_name

    # --- Reconstruct rosters as Player objects (full_season_projection intact) ---
    team_rosters = {tname: [Player.from_dict(p) for p in plist] for tname, plist in opp_raw.items()}
    team_rosters[user_name] = [Player.from_dict(p) for p in user_raw]

    # --- fraction_remaining + slots ---
    fraction_remaining = float(proj["fraction_remaining"])
    non_hitter_slots = BENCH_SLOTS | {"P"}
    h_slots = sum(v for k, v in config.roster_slots.items() if k not in non_hitter_slots)
    p_slots = config.roster_slots.get("P", 9)

    # --- actual YTD standings dict (same construction as refresh_pipeline) ---
    standings = Standings.from_json(standings_raw)
    actual_standings = {}
    for e in standings.entries:
        row = e.stats.to_dict()
        ip = e.extras.get(OpportunityStat.IP)
        pa = e.extras.get(OpportunityStat.PA)
        if ip is not None:
            row["IP"] = float(ip)
        if pa is not None:
            row["AB"] = float(pa) * AB_PER_PA
        actual_standings[e.team_name] = row

    print(
        f"teams={len(team_rosters)} h_slots={h_slots} p_slots={p_slots} "
        f"frac_remaining={fraction_remaining:.3f}"
    )
    print(f"user roster size={len(team_rosters[user_name])}")
    print()

    # Current projected ERoto totals (the "cur_pts" lead) for side-by-side context.
    from fantasy_baseball.scoring import score_roto
    from fantasy_baseball.web.season_routes import _projected_as_standings, _team_sds_from_cache

    cur_std = _projected_as_standings(proj["projected_standings"])
    cur_sds = _team_sds_from_cache(proj["team_sds"])
    cur_tot = {team: cp.total for team, cp in score_roto(cur_std, team_sds=cur_sds).items()}

    # Reference win% from earlier model stages (same rosters, n=1000, seed=42):
    #   flat-2.0-clip PT + flat backfill : Hart 49.5%, Hello Peanuts 30.3%
    #   empirical-shape PT + flat backfill: Hart 54.6%, Hello Peanuts 29.5%
    PRIOR = {user_name: "49.5 -> 54.6", "Hello Peanuts!": "30.3 -> 29.5"}

    def run(label):
        res = run_ros_monte_carlo(
            team_rosters=team_rosters,
            actual_standings=actual_standings,
            fraction_remaining=fraction_remaining,
            h_slots=h_slots,
            p_slots=p_slots,
            user_team_name=user_name,
            n_iterations=1000,
            use_management=False,
        )
        tr = res["team_results"]
        print(f"\n{label}")
        print(f"  {'team':30s} {'cur_pts':>8s} {'MC_win%':>8s} {'MC_med':>7s}   prior win%")
        print("  " + "-" * 70)
        for t in sorted(tr, key=lambda t: cur_tot.get(t, -1), reverse=True):
            v = tr[t]
            print(
                f"  {t[:30]:30s} {cur_tot.get(t, float('nan')):8.1f} "
                f"{v.get('first_pct'):7.1f}% {v.get('median_pts'):7.1f}   {PRIOR.get(t, '')}"
            )
        return tr

    run("Empirical PT shape + POSITION-AWARE replacement backfill (shipped)")


if __name__ == "__main__":
    main()
