"""Backtest: are blended hitter projections too spread out? (#389)

For each season with archived projections and actuals, blend the projection
systems the same way the draft pipeline does, then compare projected per-PA
rates (R, HR, RBI, SB) to actual per-PA rates among regulars.

Shrink model, per stat s and season:

    shrunk_rate = mu + k_s * (projected_rate - mu)

where mu is the PA-weighted projected rate of the regulars pool (projected
PA >= --min-pa). k_s = 1 is today's behavior. k_s is fit leave-one-season-out
(pooled within-season OLS slope of actual rate on projected rate) and scored on
the held-out season, so no season grades a factor it helped fit.

Scored two ways on the held-out season:
  1. Rate error (MAE) among regulars who also reached --min-pa actual PA.
  2. Draft value: hitter SGP (R/HR/RBI/SB counts + AVG) for the top --pool
     hitters by raw projected SGP. Reports rank correlation with actual SGP and
     the calibration slope of actual SGP on projected SGP (1.0 = value spread is
     right; below 1 = stars are over-valued relative to the rest of the pool).
     Run twice: at projected PA, and at actual PA (rate spread only).

Usage:
    python scripts/backtest_projection_shrink.py
    python scripts/backtest_projection_shrink.py --min-pa 300 --pool 150
"""

import argparse
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from fantasy_baseball.data.projections import blend_projections
from fantasy_baseball.sgp.denominators import get_sgp_denominators
from fantasy_baseball.sgp.player_value import REPLACEMENT_AVG, calculate_hitting_rate_sgp
from fantasy_baseball.utils.constants import DEFAULT_TEAM_AB, Category

PROJECTIONS_DIR = PROJECT_ROOT / "data" / "projections"
STATS_DIR = PROJECT_ROOT / "data" / "stats"

# Systems actually archived per season (2022-2025 have steamer + zips only).
SEASON_SYSTEMS: dict[int, list[str]] = {
    2022: ["steamer", "zips"],
    2023: ["steamer", "zips"],
    2024: ["steamer", "zips"],
    2025: ["steamer", "zips"],
    2026: ["steamer", "zips", "atc", "the-bat-x", "oopsy"],
}

STATS = ["r", "hr", "rbi", "sb"]
STAT_CATEGORY = {"r": Category.R, "hr": Category.HR, "rbi": Category.RBI, "sb": Category.SB}


def load_season(year: int) -> pd.DataFrame:
    """Blended projections joined to actuals on mlbam_id (actual_* columns).

    Actuals files only list players with >= 50 PA; anyone missing gets zeros,
    which is the right draft-value outcome for a player who barely played.
    """
    hitters, _, _ = blend_projections(PROJECTIONS_DIR / str(year), SEASON_SYSTEMS[year])
    hitters = hitters[hitters["mlbam_id"].notna()].copy()
    hitters["mlbam_id"] = hitters["mlbam_id"].astype(int)

    actual = pd.read_csv(STATS_DIR / f"hitters-{year}.csv")
    actual = actual.rename(columns={c: c.lower() for c in actual.columns})
    actual = actual.rename(columns={"mlbamid": "mlbam_id"})
    keep = ["mlbam_id", "pa", "ab", "h", *STATS]
    actual = actual[keep].add_prefix("actual_").rename(columns={"actual_mlbam_id": "mlbam_id"})

    df = hitters.merge(actual, on="mlbam_id", how="left")
    for c in actual.columns:
        if c != "mlbam_id":
            df[c] = df[c].fillna(0.0)
    df["year"] = year
    return df


def pool_means(df: pd.DataFrame, min_pa: float) -> dict[str, float]:
    """PA-weighted projected per-PA rate of the regulars pool, per stat."""
    reg = df[df["pa"] >= min_pa]
    return {s: reg[s].sum() / reg["pa"].sum() for s in STATS}


def within_season_slope(frames: list[pd.DataFrame], stat: str) -> float:
    """Pooled OLS slope of actual rate on projected rate, centered per season."""
    num = den = 0.0
    for f in frames:
        x = f[stat] / f["pa"]
        y = f[f"actual_{stat}"] / f["actual_pa"]
        xc, yc = x - x.mean(), y - y.mean()
        num += float((xc * yc).sum())
        den += float((xc * xc).sum())
    return num / den


def shrink(df: pd.DataFrame, mu: dict[str, float], k: dict[str, float]) -> pd.DataFrame:
    """Shrink R/HR/RBI/SB per-PA rates toward mu by k; PA is unchanged."""
    out = df.copy()
    for s in STATS:
        rate = out[s] / out["pa"].where(out["pa"] > 0)
        out[s] = ((mu[s] + k[s] * (rate - mu[s])) * out["pa"]).fillna(0.0).clip(lower=0.0)
    return out


def hitter_sgp(df: pd.DataFrame, prefix: str = "") -> pd.Series:
    denoms = get_sgp_denominators()
    total = sum(df[f"{prefix}{s}"] / denoms[STAT_CATEGORY[s]] for s in STATS)
    ab = df[f"{prefix}ab"]
    avg = (df[f"{prefix}h"] / ab.where(ab > 0)).fillna(0.0)
    total = total + calculate_hitting_rate_sgp(
        player_avg=avg,
        player_ab=ab,
        replacement_avg=REPLACEMENT_AVG,
        sgp_denominator=denoms[Category.AVG],
        team_ab=DEFAULT_TEAM_AB,
    )
    return total


def at_actual_pa(df: pd.DataFrame) -> pd.DataFrame:
    """Rescale projected counts to actual PA, keeping projected rates.

    Removes playing-time misses so the value check grades the rate spread alone.
    """
    out = df.copy()
    scale = (out["actual_pa"] / out["pa"].where(out["pa"] > 0)).fillna(0.0)
    for c in [*STATS, "h", "ab"]:
        out[c] = out[c] * scale
    return out


def value_metrics(df: pd.DataFrame, pool_idx: pd.Index) -> dict[str, float]:
    """Rank correlation, calibration slope and top-tier gap of projected SGP.

    ``pool_idx`` fixes the player pool, so raw and shrunk are graded on the
    same players.
    """
    top = df.loc[pool_idx].copy()
    top["proj_sgp"] = hitter_sgp(top)
    top["act_sgp"] = hitter_sgp(top, prefix="actual_")
    rho = spearmanr(top["proj_sgp"], top["act_sgp"]).statistic
    x = top["proj_sgp"] - top["proj_sgp"].mean()
    y = top["act_sgp"] - top["act_sgp"].mean()
    slope = float((x * y).sum() / (x * x).sum())
    fifth = len(top) // 5
    tier = top.nlargest(fifth, "proj_sgp")
    rest = top.nsmallest(fifth, "proj_sgp")
    gap_proj = tier["proj_sgp"].mean() - rest["proj_sgp"].mean()
    gap_act = tier["act_sgp"].mean() - rest["act_sgp"].mean()
    return {"rho": float(rho), "slope": slope, "gap_proj": gap_proj, "gap_act": gap_act}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--min-pa", type=float, default=300.0)
    parser.add_argument("--pool", type=int, default=150, help="hitters in the draft-value pool")
    args = parser.parse_args()
    logging.disable(logging.WARNING)  # blend quality warnings are known/expected here

    seasons = {y: load_season(y) for y in SEASON_SYSTEMS}
    regulars = {
        y: df[(df["pa"] >= args.min_pa) & (df["actual_pa"] >= args.min_pa)]
        for y, df in seasons.items()
    }

    print(
        f"In-season slope of actual on projected per-PA rate (proj & actual PA >= {args.min_pa:.0f})"
    )
    print(f"{'year':<6}{'n':>5}" + "".join(f"{s.upper() + '/PA':>9}" for s in STATS))
    for y, reg in regulars.items():
        row = "".join(f"{within_season_slope([reg], s):>9.2f}" for s in STATS)
        print(f"{y:<6}{len(reg):>5}{row}")
    pooled = "".join(f"{within_season_slope(list(regulars.values()), s):>9.2f}" for s in STATS)
    print(f"{'all':<6}{sum(len(r) for r in regulars.values()):>5}{pooled}")

    print("\nLeave-one-season-out: fit k on the other seasons, score the held-out one")
    header = (
        f"{'year':<6}"
        + "".join(f"{'k_' + s:>7}" for s in STATS)
        + "".join(f"{'mae_' + s:>14}" for s in STATS)
    )
    print(header + "   (MAE per 600 PA: raw -> shrunk)")
    value_rows = []
    for y in seasons:
        train = [regulars[t] for t in regulars if t != y]
        k = {s: within_season_slope(train, s) for s in STATS}
        mu = pool_means(seasons[y], args.min_pa)
        raw, shr = seasons[y], shrink(seasons[y], mu, k)
        reg_idx = regulars[y].index
        maes = []
        for s in STATS:
            act = raw.loc[reg_idx, f"actual_{s}"] / raw.loc[reg_idx, "actual_pa"] * 600
            m_raw = (raw.loc[reg_idx, s] / raw.loc[reg_idx, "pa"] * 600 - act).abs().mean()
            m_shr = (shr.loc[reg_idx, s] / shr.loc[reg_idx, "pa"] * 600 - act).abs().mean()
            maes.append(f"{m_raw:>6.2f}->{m_shr:<6.2f}")
        print(
            f"{y:<6}" + "".join(f"{k[s]:>7.2f}" for s in STATS) + "".join(f"{m:>14}" for m in maes)
        )
        pool_idx = hitter_sgp(raw).nlargest(args.pool).index
        value_rows.append((y, raw, shr, pool_idx))

    for title, view in [
        ("projected PA", lambda df: df),
        ("actual PA (rate spread only)", at_actual_pa),
    ]:
        print(f"\nDraft value, top {args.pool} hitters by raw projected SGP, at {title}")
        print(
            f"{'year':<6}{'rank corr':>16}{'value slope':>16}"
            f"{'top-bottom fifth gap: proj':>32}{'actual':>8}"
        )
        for y, raw, shr, pool_idx in value_rows:
            a = value_metrics(view(raw), pool_idx)
            b = value_metrics(view(shr), pool_idx)
            print(
                f"{y:<6}{a['rho']:>7.3f} -> {b['rho']:<6.3f}"
                f"{a['slope']:>7.2f} -> {b['slope']:<6.2f}"
                f"{a['gap_proj']:>20.2f} -> {b['gap_proj']:<6.2f}{a['gap_act']:>9.2f}"
            )


if __name__ == "__main__":
    main()
