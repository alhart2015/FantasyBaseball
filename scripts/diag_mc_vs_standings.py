"""READ-ONLY diagnostic: reconcile projected roto standings vs Monte Carlo win%.

Pulls cache:projections and cache:monte_carlo from prod Upstash, then prints,
for each team, the score_roto expected-points total (preseason + current) next
to the MC first-place probability (preseason base + live rest-of-season).

This script ONLY reads. It never writes to Upstash.
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


def _unwrap(raw: str | None):
    if raw is None:
        return None
    obj = json.loads(raw)
    if isinstance(obj, dict) and "_data" in obj and "_meta" in obj:
        return obj["_data"], obj.get("_meta", {})
    return obj, {}


def main() -> None:
    _load_dotenv()
    from fantasy_baseball.data.cache_keys import CacheKey, redis_key
    from fantasy_baseball.data.kv_store import build_explicit_upstash_kv
    from fantasy_baseball.scoring import score_roto
    from fantasy_baseball.web.season_routes import (
        _projected_as_standings,
        _team_sds_from_cache,
    )

    kv = build_explicit_upstash_kv()

    proj_payload, proj_meta = _unwrap(kv.get(redis_key(CacheKey.PROJECTIONS))) or (None, {})
    mc_payload, mc_meta = _unwrap(kv.get(redis_key(CacheKey.MONTE_CARLO))) or (None, {})

    print("=== cache:projections _meta ===")
    print(json.dumps(proj_meta, indent=2))
    print("=== cache:monte_carlo _meta ===")
    print(json.dumps(mc_meta, indent=2))

    def roto_totals(standings_json, sds_json):
        if not standings_json:
            return {}
        standings = _projected_as_standings(standings_json)
        sds = _team_sds_from_cache(sds_json)
        scored = score_roto(standings, team_sds=sds)
        return {team: cp.total for team, cp in scored.items()}

    cur_tot = roto_totals(
        proj_payload.get("projected_standings") if proj_payload else None,
        proj_payload.get("team_sds") if proj_payload else None,
    )
    pre_tot = roto_totals(
        proj_payload.get("preseason_standings") if proj_payload else None,
        proj_payload.get("preseason_team_sds") if proj_payload else None,
    )

    def mc_block(payload_key):
        if not mc_payload:
            return {}
        block = mc_payload.get(payload_key) or {}
        return block.get("team_results") or {}

    pre_tr = mc_block("base")
    ros_tr = mc_block("rest_of_season")

    teams = set(cur_tot) | set(pre_tot) | set(pre_tr) | set(ros_tr)
    # sort by current projected total desc
    ordered = sorted(teams, key=lambda t: cur_tot.get(t, -1), reverse=True)

    def g(d, t, k):
        v = (d.get(t) or {}).get(k)
        return v if v is not None else float("nan")

    print()
    print(
        "ANALYTIC score_roto total (cur_pts) vs MC median/p10/p90 of total roto pts (live ROS MC)"
    )
    hdr = (
        f"{'team':30s} {'cur_pts':>8s} {'rosMC%':>7s} "
        f"{'mc_med':>7s} {'mc_p10':>7s} {'mc_p90':>7s} {'med-cur':>8s}"
    )
    print(hdr)
    print("-" * len(hdr))
    for t in ordered:
        med = g(ros_tr, t, "median_pts")
        cur = cur_tot.get(t, float("nan"))
        print(
            f"{t[:30]:30s} "
            f"{cur:8.1f} "
            f"{g(ros_tr, t, 'first_pct'):7.1f} "
            f"{med:7.1f} "
            f"{g(ros_tr, t, 'p10'):7.1f} "
            f"{g(ros_tr, t, 'p90'):7.1f} "
            f"{(med - cur):8.1f}"
        )

    print()
    print("Preseason MC for reference:")
    hdr2 = f"{'team':30s} {'pre_pts':>8s} {'preMC%':>7s} {'mc_med':>7s}"
    print(hdr2)
    print("-" * len(hdr2))
    for t in ordered:
        print(
            f"{t[:30]:30s} "
            f"{pre_tot.get(t, float('nan')):8.1f} "
            f"{g(pre_tr, t, 'first_pct'):7.1f} "
            f"{g(pre_tr, t, 'median_pts'):7.1f}"
        )

    # Per-category: score_roto expected points vs MC median points for the user.
    user = "Hart of the Order"
    cr = (mc_payload.get("rest_of_season") or {}).get("category_risk") if mc_payload else None
    proj_std = _projected_as_standings(proj_payload["projected_standings"])
    sds = _team_sds_from_cache(proj_payload["team_sds"])
    scored = score_roto(proj_std, team_sds=sds)
    user_cp = scored[user]
    print()
    print(f"PER-CATEGORY for {user}: analytic score_roto E[pts] vs MC median pts (live ROS)")
    h = f"{'cat':6s} {'analytic_pts':>12s} {'team_sd':>9s} {'mc_med':>7s} {'mc_p10':>7s} {'mc_p90':>7s}"
    print(h)
    print("-" * len(h))
    a_tot = 0.0
    m_tot = 0.0
    for cat, pts in user_cp.values.items():
        cat_name = cat.value if hasattr(cat, "value") else str(cat)
        mc_cat = (cr or {}).get(cat_name, {}) if cr else {}
        sd = (sds.get(user, {}) or {}).get(cat, float("nan")) if sds else float("nan")
        med = mc_cat.get("median_pts", float("nan"))
        a_tot += pts
        if med == med:
            m_tot += med
        print(
            f"{cat_name:6s} {pts:12.2f} {sd:9.3f} "
            f"{med:7.1f} {mc_cat.get('p10', float('nan')):7.1f} "
            f"{mc_cat.get('p90', float('nan')):7.1f}"
        )
    print("-" * len(h))
    print(f"{'TOTAL':6s} {a_tot:12.2f} {'':9s} {m_tot:7.1f}")


if __name__ == "__main__":
    main()
