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
from fantasy_baseball.utils.dispersion import negbin_perf_variance
from fantasy_baseball.utils.playing_time import playing_time_params

YEARS = [2022, 2023, 2024, 2025]
PROJ = ROOT / "data" / "projections"
STATS = ROOT / "data" / "stats"
N_TEAMS = 500
H_PA_MIN, P_IP_MIN = 450, 60
N_H, N_P = 13, 9
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


def _derive_so(pa):
    """Reconstruct SO for actuals files that ship K/9 instead of raw SO (2025 is a
    rate/advanced export). IP is baseball thirds-notation (195.1 == 195 1/3), so
    convert before dividing -- the naive K/9*IP/9 misrounds ~9% of pitchers."""
    if "SO" not in pa.columns and {"K/9", "IP"}.issubset(pa.columns):
        ip = pa["IP"].to_numpy(dtype=float)
        ip_true = np.floor(ip) + (ip - np.floor(ip)) * 10.0 / 3.0
        pa = pa.copy()
        pa["SO"] = np.round(pa["K/9"].to_numpy(dtype=float) * ip_true / 9.0)
    return pa


def build_year(year):
    """Return (hitter_merged, pitcher_merged_or_None, pitcher_cats). Pitcher cats are
    per-year: 2025 actuals lack raw SO (derived above), so admit whatever is present."""
    # Hitters
    hp = blend(year, "hitters", ["PA", "R", "HR", "RBI", "SB"])
    ha = _read(STATS / f"hitters-{year}.csv")
    ha["MLBAMID"] = ha["MLBAMID"].astype(int)
    hp = hp[hp["PA"] >= H_PA_MIN]
    hm = hp.merge(
        ha[["MLBAMID", "R", "HR", "RBI", "SB"]], on="MLBAMID", how="left", suffixes=("_p", "_a")
    )
    # Pitchers -- per-year category set from whichever actual cols are present.
    # W and SV are in every actuals file; SO is derived for rate-only years (2025).
    pa = _derive_so(_read(STATS / f"pitchers-{year}.csv"))
    p_cats = [(acol, key) for acol, key in P_CATS if acol in pa.columns]
    if not p_cats:
        return hm, None, []
    acols = [acol for acol, _ in p_cats]
    pp = blend(year, "pitchers", ["IP", *acols])
    pa["MLBAMID"] = pa["MLBAMID"].astype(int)
    pp = pp[pp["IP"] >= P_IP_MIN]
    pm = pp.merge(pa[["MLBAMID", *acols]], on="MLBAMID", how="left", suffixes=("_p", "_a"))
    return hm, pm, p_cats


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
        hm, pm, p_cats = build_year(year)
        for _ in range(N_TEAMS):
            for acol, v in team_z(hm, H_CATS, "PA", True, dnp_zero).items():
                zs[acol].append(v)
            if pm is not None:
                for acol, v in team_z(pm, p_cats, "IP", False, dnp_zero).items():
                    zs[acol].append(v)
    return zs


print(
    f"Synthetic teams: {N_TEAMS}/yr x {len(YEARS)} yrs; hitters PA>={H_PA_MIN}, pitchers IP>={P_IP_MIN}"
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
