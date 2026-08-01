"""READ-ONLY decomposition of the MC's per-team roto, isolating WHY the MC
treats Hart vs Peanuts differently than ERoto. Reuses the real sim primitives.

Toggles:
  reoptimize=True  -> argmax best-12/9 by SIMULATED performance each iter (real MC)
  reoptimize=False -> active lineup FIXED ONCE by projection (management-aware)
  pt_variance=True -> real per-player injury/playing-time sampling (real MC)
  pt_variance=False-> deterministic eff_mean (keeps mean haircut, removes injury RISK)

Comparisons:
  (baseline) - (no-reoptimize)  = management-blindness / argmax "perfect deployment" tax
  (no-PT-variance) - (baseline) = how much injury RISK costs each team  <-- star-power theory
Plus a star-concentration metric (share of above-replacement value in top players).
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
from fantasy_baseball.models.player import PlayerType  # noqa: E402
from fantasy_baseball.scoring import score_roto_dict  # noqa: E402
from fantasy_baseball.simulation import (  # noqa: E402
    CLOSER_SV_THRESHOLD,
    HITTER_COV,
    HITTER_IDX,
    PITCHER_COV,
    PITCHER_IDX,
    _flatten_full_season,
    _projected_volume,
    _replacement_line,
)
from fantasy_baseball.utils.constants import (  # noqa: E402
    HITTING_COUNTING,
    PITCHING_COUNTING,
    REPLACEMENT_BY_POSITION,
)
from fantasy_baseball.utils.playing_time import playing_time_params  # noqa: E402

kv = build_explicit_upstash_kv()


def g(key: CacheKey):
    raw = kv.get(redis_key(key))
    o = json.loads(raw) if isinstance(raw, str) else raw
    return o["_data"] if isinstance(o, dict) and "_data" in o else o


USER = "Hart of the Order"
FR = 0.5946
H_SLOTS, P_SLOTS = 12, 9
opp = g(CacheKey.OPP_ROSTERS)
user = g(CacheKey.ROSTER)
standings = g(CacheKey.STANDINGS)
team_rosters = {USER: user, **opp}
flat = {t: [_flatten_full_season(p) for p in ps] for t, ps in team_rosters.items()}

actuals_by_team = {}
for t in standings["teams"]:
    row = dict(t["stats"])
    ip = t.get("extras", {}).get("IP")
    if ip is not None:
        row["IP"] = float(ip)
    actuals_by_team[t["name"]] = row

_TYP_AB, _TYP_IP = 5500, 1450


def eff_mean(p, is_hitter):
    ms, _ = playing_time_params(PlayerType.HITTER if is_hitter else PlayerType.PITCHER,
                                _projected_volume(p, is_hitter))
    return 1.0 - (1.0 - ms) * FR


def apply_var(players, ptype, rng, pt_variance):
    is_hitter = ptype == PlayerType.HITTER
    cols = HITTING_COUNTING if is_hitter else PITCHING_COUNTING
    idx = HITTER_IDX if is_hitter else PITCHER_IDX
    cov = (HITTER_COV if is_hitter else PITCHER_COV) * FR
    n = len(players)
    if n == 0:
        return []
    if pt_variance:
        us = rng.random(n)
        from fantasy_baseball.utils.playing_time import playing_time_shape, scale_from_uniform
        scales = np.empty(n)
        for i, p in enumerate(players):
            vol = _projected_volume(p, is_hitter)
            ms, cv = playing_time_params(ptype, vol)
            scales[i] = scale_from_uniform(ms, cv, playing_time_shape(ptype, vol), float(us[i]), FR)
    else:
        scales = np.array([eff_mean(p, is_hitter) for p in players])
    draws = rng.multivariate_normal(np.zeros(len(idx)), cov, size=n)
    out = []
    for i, p in enumerate(players):
        scale = float(scales[i])
        fm = max(0.0, 1.0 - scale)
        repl = _replacement_line(p, is_hitter)
        row = {}
        for col in cols:
            base = float(p.get(col, 0) or 0)
            rc = repl.get(col, 0) * fm
            if col in idx:
                row[col] = base * max(0, 1.0 + draws[i][idx[col]]) * scale + rc
            else:
                row[col] = base * scale + rc
        row["name"] = p.get("name", "?")
        out.append(row)
    return out


def hsort_key(h):
    return h["r"] + h["hr"] + h["rbi"] + h["sb"]


def psort_key(p):
    return (p.get("sv", 0) >= CLOSER_SV_THRESHOLD, p["w"] + p["k"] + p.get("sv", 0))


def run(reoptimize, pt_variance, n_iter=1000, seed=42):
    rng = np.random.default_rng(seed)
    teams = list(flat.keys())
    # Pre-pick fixed lineup by PROJECTION when not reoptimizing.
    fixed_active = {}
    if not reoptimize:
        for t, players in flat.items():
            H = [p for p in players if p.get("player_type") == PlayerType.HITTER]
            P = [p for p in players if p.get("player_type") == PlayerType.PITCHER]
            ah = {p["name"] for p in sorted(H, key=hsort_key, reverse=True)[:H_SLOTS]}
            ap = {p["name"] for p in sorted(P, key=psort_key, reverse=True)[:P_SLOTS]}
            fixed_active[t] = (ah, ap)
    roto_tot = {t: 0.0 for t in teams}
    wins = {t: 0 for t in teams}
    for _ in range(n_iter):
        sim_stats = {}
        for t, players in flat.items():
            actuals = actuals_by_team.get(t, {})
            H = [p for p in players if p.get("player_type") == PlayerType.HITTER]
            P = [p for p in players if p.get("player_type") == PlayerType.PITCHER]
            ah = apply_var(H, PlayerType.HITTER, rng, pt_variance)
            ap = apply_var(P, PlayerType.PITCHER, rng, pt_variance)
            if reoptimize:
                ah.sort(key=hsort_key, reverse=True)
                ap.sort(key=psort_key, reverse=True)
                act_h, act_p = ah[:H_SLOTS], ap[:P_SLOTS]
            else:
                fh, fp = fixed_active[t]
                act_h = [h for h in ah if h["name"] in fh]
                act_p = [p for p in ap if p["name"] in fp]
            ab = sum(h["ab"] for h in act_h)
            hh = sum(h["h"] for h in act_h)
            ip = sum(p["ip"] for p in act_p)
            er = sum(p["er"] for p in act_p)
            bb = sum(p["bb"] for p in act_p)
            ha = sum(p["h_allowed"] for p in act_p)
            a_ab = actuals.get("AB", _TYP_AB * (1 - FR))
            a_ip = actuals.get("IP", _TYP_IP * (1 - FR))
            a_h = actuals.get("AVG", 0) * a_ab
            a_er = actuals.get("ERA", 0) * a_ip / 9
            a_hbb = actuals.get("WHIP", 0) * a_ip
            t_ab = a_ab + max(0, ab - a_ab)
            t_h = a_h + max(0, hh - a_h)
            t_ip = a_ip + max(0, ip - a_ip)
            t_er = a_er + max(0, er - a_er)
            t_hbb = a_hbb + max(0, (bb + ha) - a_hbb)
            sim_stats[t] = {
                "R": actuals.get("R", 0) + max(0, sum(h["r"] for h in act_h) - actuals.get("R", 0)),
                "HR": actuals.get("HR", 0) + max(0, sum(h["hr"] for h in act_h) - actuals.get("HR", 0)),
                "RBI": actuals.get("RBI", 0) + max(0, sum(h["rbi"] for h in act_h) - actuals.get("RBI", 0)),
                "SB": actuals.get("SB", 0) + max(0, sum(h["sb"] for h in act_h) - actuals.get("SB", 0)),
                "AVG": t_h / t_ab if t_ab > 0 else 0,
                "W": actuals.get("W", 0) + max(0, sum(p["w"] for p in act_p) - actuals.get("W", 0)),
                "K": actuals.get("K", 0) + max(0, sum(p["k"] for p in act_p) - actuals.get("K", 0)),
                "SV": actuals.get("SV", 0) + max(0, sum(p.get("sv", 0) for p in act_p) - actuals.get("SV", 0)),
                "ERA": 9 * t_er / t_ip if t_ip > 0 else 99,
                "WHIP": t_hbb / t_ip if t_ip > 0 else 99,
            }
        roto = score_roto_dict(sim_stats)
        ranked = sorted(roto.items(), key=lambda x: x[1]["total"], reverse=True)
        for rk, (nm, pts) in enumerate(ranked, 1):
            roto_tot[nm] += pts["total"]
            if rk == 1:
                wins[nm] += 1
    return ({t: roto_tot[t] / n_iter for t in teams}, {t: 100 * wins[t] / n_iter for t in teams})


base, base_w = run(True, True)
noreopt, _ = run(False, True)
nopt, nopt_w = run(True, False)

print(f"{'team':<22}{'BASE':>7}{'noReopt':>9}{'argmaxTax':>10}{'noInjury':>9}{'injuryCost':>11}{'1st%':>7}")
for t in [USER, "Hello Peanuts!"]:
    tax = base[t] - noreopt[t]          # how much the argmax inflates this team
    inj = nopt[t] - base[t]             # how much injury RISK costs this team (>0 = hurt by injuries)
    print(f"{t:<22}{base[t]:>7.1f}{noreopt[t]:>9.1f}{tax:>10.1f}{nopt[t]:>9.1f}{inj:>11.1f}{base_w[t]:>7.1f}")

print("\n-- Star concentration: above-replacement value per player (full-season) --")
def arepl(p, is_h):
    repl = _replacement_line(p, is_h)
    if is_h:
        return max(0, (p.get("r",0)+p.get("hr",0)+p.get("rbi",0)+p.get("sb",0))
                   - (repl.get("r",0)+repl.get("hr",0)+repl.get("rbi",0)+repl.get("sb",0))*FR)
    return max(0, (p.get("k",0)+5*p.get("w",0)+3*p.get("sv",0))
               - (repl.get("k",0)+5*repl.get("w",0)+3*repl.get("sv",0))*FR)
for t in [USER, "Hello Peanuts!"]:
    vals = sorted((arepl(p, p.get("player_type")==PlayerType.HITTER) for p in flat[t]), reverse=True)
    tot = sum(vals)
    top3 = sum(vals[:3])
    print(f"  {t:<22} total AR={tot:6.0f}  top3 AR={top3:6.0f} ({100*top3/tot:4.1f}%)  top5={100*sum(vals[:5])/tot:4.1f}%")
