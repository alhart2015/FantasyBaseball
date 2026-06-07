"""Read-only: per-category finish-rank distribution for the user's team.

Uses the SAME Gaussian model the dashboard's Category Bars view draws from
(category_odds: Gauss-Hermite quadrature over the user's projected normal,
Poisson-binomial over opponents), but extends it from P(1st)/P(top3) to the
FULL rank distribution so we can read P(finish 5th-or-worse) -- the fragility
of a lead.

Data source is prod Upstash cache:projections (projected_standings + team_sds
+ fraction_remaining) -- the identical blob season_routes feeds the bars view,
so these odds agree with what the chart shows. Nothing is written.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

# Reuse the bars-view model internals so our numbers match the chart exactly.
from fantasy_baseball.category_odds import (  # noqa: E402
    _GH_NODES,
    _GH_WEIGHTS,
    _SQRT_2PI,
    _prob_opp_above,
    category_finish_odds,
)
from fantasy_baseball.data.cache_keys import CacheKey, redis_key  # noqa: E402
from fantasy_baseball.data.kv_store import build_explicit_upstash_kv  # noqa: E402
from fantasy_baseball.data.redis_store import get_latest_standings  # noqa: E402
from fantasy_baseball.scoring import score_roto, team_sds_from_json  # noqa: E402
from fantasy_baseball.utils.constants import (  # noqa: E402
    ALL_CATEGORIES,
    INVERSE_STATS,
    RATE_STATS,
    Category,
)

TEAM = "Hart of the Order"


def _read_prod_cache(kv, key: CacheKey):
    raw = kv.get(redis_key(key))
    if raw is None:
        return None
    obj = json.loads(raw)
    # Unwrap provenance envelope ({_meta, _data}); bare payloads pass through.
    if isinstance(obj, dict) and "_meta" in obj and "_data" in obj:
        return obj["_data"]
    return obj


def rank_distribution(means, sds, user_index, *, higher_is_better):
    """Full finish-rank pmf for the user via the category_odds Gaussian model.

    Returns ``pmf`` where ``pmf[k]`` = P(exactly k opponents beat the user),
    so finish position = k + 1. Integrates the user's N(mu, sd) draw via
    Gauss-Hermite, and at each draw runs the exact O(n^2) Poisson-binomial DP
    over opponents' beat-probabilities (same machinery as the bars view, just
    the whole pmf instead of only k<=2).
    """
    n = len(means)
    sign = 1.0 if higher_is_better else -1.0
    mu = [sign * m for m in means]
    mu_u = mu[user_index]
    sd_u = sds[user_index]
    opps = [(mu[i], sds[i]) for i in range(n) if i != user_index]

    full = [0.0] * n  # index = number of opponents that beat the user (0..n-1)
    for node, weight in zip(_GH_NODES, _GH_WEIGHTS, strict=True):
        x = mu_u + sd_u * node
        qs = [_prob_opp_above(x, mo, so) for (mo, so) in opps]
        # Poisson-binomial pmf over opponents.
        pmf = [1.0]
        for q in qs:
            nxt = [0.0] * (len(pmf) + 1)
            for c, p in enumerate(pmf):
                nxt[c] += p * (1.0 - q)
                nxt[c + 1] += p * q
            pmf = nxt
        w = weight / _SQRT_2PI
        for k, p in enumerate(pmf):
            full[k] += w * p
    return full


def _fmt(cat: Category, v: float) -> str:
    if cat in RATE_STATS:
        return f"{v:.4f}" if cat == Category.AVG else f"{v:.3f}"
    return f"{v:.1f}"


def main() -> None:
    kv = build_explicit_upstash_kv()

    proj_cache = _read_prod_cache(kv, CacheKey.PROJECTIONS) or {}
    ps_dict = proj_cache.get("projected_standings") or {}
    ps_rows = ps_dict.get("teams") if isinstance(ps_dict, dict) else None
    team_sds = team_sds_from_json(proj_cache.get("team_sds") or {})
    fr = proj_cache.get("fraction_remaining")
    if not ps_rows:
        print("No projected_standings teams in cache:projections.")
        return
    print(f"projected_standings effective_date: {ps_dict.get('effective_date')}")

    # each row: {name, stats:{R:..}, total_ab, total_ip}
    proj_means = {row["name"]: {Category(k): float(v) for k, v in row["stats"].items()} for row in ps_rows}
    teams = [row["name"] for row in ps_rows]
    if TEAM not in teams:
        print(f"{TEAM} not in projected standings: {teams}")
        return

    cur = get_latest_standings(kv)
    cur_pts = score_roto(cur) if cur else None
    cur_by_team = cur.by_team() if cur else {}
    n = len(teams)

    print(f"fraction_remaining = {fr}  (so ~{(1 - float(fr)) * 100:.0f}% of season played)" if fr is not None else "fraction_remaining: MISSING")
    print(f"teams = {n}\n")

    header = (
        f"{'cat':<5} {'curVal':>8} {'curRk':>5} | {'projVal':>8} {'projRk':>6} "
        f"{'E[fin]':>6} | {'P1st':>6} {'Ptop3':>6} {'Ptop5':>6} {'P>=5th':>7} {'P>=6th':>7} | sd"
    )
    print(header)
    print("-" * len(header))

    rows = []
    for cat in ALL_CATEGORIES:
        hib = cat not in INVERSE_STATS
        means = [proj_means[t][cat] for t in teams]
        sds = [team_sds.get(t, {}).get(cat, 0.0) for t in teams]
        ui = teams.index(TEAM)

        pmf = rank_distribution(means, sds, ui, higher_is_better=hib)
        p_first = pmf[0] * 100
        p_top3 = sum(pmf[:3]) * 100
        p_top5 = sum(pmf[:5]) * 100
        p_5plus = sum(pmf[4:]) * 100   # finish 5th or worse
        p_6plus = sum(pmf[5:]) * 100   # finish 6th or worse
        e_fin = sum((k + 1) * p for k, p in enumerate(pmf))

        # projected rank (deterministic, by mean)
        order = sorted(teams, key=lambda t: proj_means[t][cat], reverse=hib)
        proj_rk = order.index(TEAM) + 1

        # current value + rank
        if cur and TEAM in cur_by_team:
            cur_val = cur_by_team[TEAM].stats[cat]
            cur_order = sorted(
                cur.entries, key=lambda e: e.stats[cat], reverse=hib
            )
            cur_rk = [e.team_name for e in cur_order].index(TEAM) + 1
        else:
            cur_val, cur_rk = float("nan"), 0

        rows.append((cat, p_5plus, p_first))
        print(
            f"{cat.value:<5} {_fmt(cat, cur_val):>8} {cur_rk:>5} | "
            f"{_fmt(cat, means[ui]):>8} {proj_rk:>6} {e_fin:>6.2f} | "
            f"{p_first:>5.1f}% {p_top3:>5.1f}% {p_top5:>5.1f}% {p_5plus:>6.1f}% {p_6plus:>6.1f}% | "
            f"{sds[ui]:.3f}"
        )

    # Cross-check vs the library's category_finish_odds (P1st/Ptop3 must match).
    print("\n-- consistency check vs category_finish_odds (P1st, Ptop3) --")
    for cat in ALL_CATEGORIES:
        hib = cat not in INVERSE_STATS
        means = [proj_means[t][cat] for t in teams]
        sds = [team_sds.get(t, {}).get(cat, 0.0) for t in teams]
        ui = teams.index(TEAM)
        o = category_finish_odds(means, sds, ui, higher_is_better=hib)
        print(f"{cat.value:<5} lib=({o.first_pct:5.1f},{o.top3_pct:5.1f}) clearWins={o.clear_wins}/{o.opponents}")

    print("\n=== FRAGILITY RANK (most fragile lead first, by P(finish 5th or worse)) ===")
    for cat, p5, p1 in sorted(rows, key=lambda r: -r[1]):
        print(f"{cat.value:<5} P(>=5th)={p5:5.1f}%  P(1st)={p1:5.1f}%")


if __name__ == "__main__":
    main()
