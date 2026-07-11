"""Backtest: is ERoto's analytic team-category SD calibrated to reality?

For 2022-2025, blend steamer+zips projections, build realistic synthetic rosters
(13 everyday hitters + 9 pitchers), and for each counting category compute the
standardized residual

    z = (actual_team_total - projected_team_total) / eroto_SD

where eroto_SD^2 = sum_players (negbin_perf_variance(stat, proj) + proj^2 * cv_pt^2)
-- exactly the per-player quadrature ProjectedStandings/score_roto uses, fed by the
unified NegBin dispersion (STAT_DISPERSION) that the MC also samples. If SD(z) ~= 1,
ERoto's NegBin-based SD matches realized variance. If
SD(z) ~= 2, ERoto is 2x too tight -> the leader's category sweeps are far less
certain than ERoto thinks -> the MC's wider/lower picture is right. If SD(z) ~= 1,
ERoto's SD is fine and the MC's roto deflation is an argmax artifact, not variance.

Local files only (data/projections, data/stats). No network.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from fantasy_baseball.models.player import PlayerType
from fantasy_baseball.sgp.closer_mixture import sv_role_variance
from fantasy_baseball.utils.dispersion import negbin_perf_variance
from fantasy_baseball.utils.playing_time import playing_time_params

YEARS = [2022, 2023, 2024, 2025]
PROJ = ROOT / "data" / "projections"
STATS = ROOT / "data" / "stats"
N_TEAMS = 500
H_PA_MIN, P_IP_MIN = 450, 60
N_H, N_P = 13, 9
# Saves come only from save-relevant pitchers. A random 9-pitcher team is dominated by
# projected-~0-SV arms whose occasional fluke saves the multiplicative mixture (mu = s *
# a_k) structurally cannot produce from s ~= 0 -- a known unmodeled phenomenon (a small
# projection-independent save-hazard floor would capture it; deferred). Including them
# makes the full-pool z-score look under-dispersed by an artifact of the measurement, not
# a closer-variance failure, so measure the SV category on the save-relevant pool.
SV_POOL_MIN_PROJ = 5.0
rng = np.random.default_rng(11)

H_CATS = [("R", "r"), ("HR", "hr"), ("RBI", "rbi"), ("SB", "sb")]
P_CATS = [("W", "w"), ("SO", "k"), ("SV", "sv")]  # actual col, STAT_DISPERSION key


def _read(path):
    return pd.read_csv(path, encoding="utf-8-sig")


def proj_path(year, system, kind):
    a = PROJ / str(year) / f"{system}-{kind}.csv"
    b = PROJ / str(year) / f"{system}-{kind}-{year}.csv"
    return a if a.exists() else b


def blend(year, kind, cols):
    frames = []
    for sysname in ("steamer", "zips"):
        p = proj_path(year, sysname, kind)
        if not p.exists():
            continue
        df = _read(p)
        df = df[df["MLBAMID"].notna()].copy()
        df["MLBAMID"] = df["MLBAMID"].astype(int)
        keep = ["MLBAMID"] + [c for c in cols if c in df.columns]
        frames.append(df[keep])
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames).groupby("MLBAMID", as_index=False).mean()


def build_year(year):
    """Return (hitter_merged, pitcher_merged). Every actuals file carries the standard
    counting template (W/SO/SV), so all pitcher categories are available every year."""
    # Hitters
    hp = blend(year, "hitters", ["PA", "R", "HR", "RBI", "SB"])
    ha = _read(STATS / f"hitters-{year}.csv")
    ha["MLBAMID"] = ha["MLBAMID"].astype(int)
    hp = hp[hp["PA"] >= H_PA_MIN]
    hm = hp.merge(
        ha[["MLBAMID", "R", "HR", "RBI", "SB"]], on="MLBAMID", how="left", suffixes=("_p", "_a")
    )
    # Pitchers
    acols = [acol for acol, _ in P_CATS]
    pa = _read(STATS / f"pitchers-{year}.csv")
    pp = blend(year, "pitchers", ["IP", *acols])
    pa["MLBAMID"] = pa["MLBAMID"].astype(int)
    pp = pp[pp["IP"] >= P_IP_MIN]
    pm = pp.merge(pa[["MLBAMID", *acols]], on="MLBAMID", how="left", suffixes=("_p", "_a"))
    return hm, pm


def cv_pt(vol, is_hitter):
    return playing_time_params(PlayerType.HITTER if is_hitter else PlayerType.PITCHER, vol)[1]


def team_z(pool, cats, vol_col, is_hitter, dnp_zero):
    """One synthetic team's per-category standardized residual."""
    idx = rng.choice(len(pool), size=(N_H if is_hitter else N_P), replace=False)
    t = pool.iloc[idx]
    out = {}
    for acol, key in cats:
        proj = t[f"{acol}_p"].to_numpy(dtype=float)
        act = t[f"{acol}_a"].to_numpy(dtype=float)
        if dnp_zero:
            act = np.nan_to_num(act, nan=0.0)
        else:
            mask = ~np.isnan(act)
            proj, act = proj[mask], act[mask]
        if len(proj) == 0:
            out[acol] = np.nan
            continue
        if key == "sv":
            var = float(np.sum(sv_role_variance(proj)))  # role-switch mixture (full-season)
        else:
            cvp = np.array([cv_pt(v, is_hitter) for v in t[vol_col].to_numpy(dtype=float)])
            if not dnp_zero:
                cvp = cvp[mask]
            var = np.sum(negbin_perf_variance(key, proj) + proj**2 * cvp**2)
        sd = np.sqrt(var)
        out[acol] = (act.sum() - proj.sum()) / sd if sd > 0 else np.nan
    return out


def run(dnp_zero):
    zs = {acol: [] for acol, _ in H_CATS + P_CATS}
    for year in YEARS:
        hm, pm = build_year(year)
        # SV is measured on the save-relevant pool (see SV_POOL_MIN_PROJ); W/K/hitters
        # on the full pool.
        wk_cats = [(a, k) for a, k in P_CATS if a != "SV"]
        sv_cats = [(a, k) for a, k in P_CATS if a == "SV"]
        sv_pool = pm[pm["SV_p"] >= SV_POOL_MIN_PROJ]
        for _ in range(N_TEAMS):
            for acol, v in team_z(hm, H_CATS, "PA", True, dnp_zero).items():
                zs[acol].append(v)
            for acol, v in team_z(pm, wk_cats, "IP", False, dnp_zero).items():
                zs[acol].append(v)
            if len(sv_pool) >= N_P:
                for acol, v in team_z(sv_pool, sv_cats, "IP", False, dnp_zero).items():
                    zs[acol].append(v)
    return zs


def main():
    print(
        f"Synthetic teams: {N_TEAMS}/yr x {len(YEARS)} yrs; "
        f"hitters PA>={H_PA_MIN}, pitchers IP>={P_IP_MIN}"
    )
    print(
        "z = (actual_team_total - projected) / eroto_SD.  SD(z)=1 -> calibrated; "
        ">1 -> ERoto too TIGHT by that factor.\n"
    )
    for label, dnp in [
        ("MATCHED-ONLY (excludes DNP/bust tail -> lower bound on variance)", False),
        ("DNP=0 (rostered-but-absent counted as zero -> includes bust tail)", True),
    ]:
        zs = run(dnp)
        print(f"== {label} ==")
        print(f"  {'cat':>5}{'mean z':>9}{'SD(z)':>8}{'n':>7}   verdict")
        all_z = []
        for acol, _ in H_CATS + P_CATS:
            arr = np.array([z for z in zs[acol] if z == z])
            all_z.extend(arr.tolist())
            sd = arr.std()
            v = "calibrated" if 0.8 <= sd <= 1.25 else ("TOO TIGHT" if sd > 1.25 else "too wide")
            print(f"  {acol:>5}{arr.mean():>9.2f}{sd:>8.2f}{len(arr):>7}   {v} ({sd:.1f}x)")
        a = np.array(all_z)
        print(
            f"  POOLED  mean={a.mean():.2f}  SD(z)={a.std():.2f}  -> ERoto SD is "
            f"{'~calibrated' if 0.8 <= a.std() <= 1.25 else f'{a.std():.1f}x too tight'}\n"
        )


if __name__ == "__main__":
    main()
