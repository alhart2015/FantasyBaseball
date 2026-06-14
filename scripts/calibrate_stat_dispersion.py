"""Calibrate per-stat Negative-Binomial dispersion from projection-vs-actual
residuals, conditional on realized playing time (2022-2024).

Mirrors scripts/calibrate_playing_time.py's data handling. Emits a
STAT_DISPERSION dict (per-stat r, with a Poisson sentinel) for paste into
src/fantasy_baseball/utils/constants.py, plus a leave-one-season-out
interval-coverage table that gates the shipped values. Performance dispersion
is measured conditional on realized PT so it does NOT double-count the
playing-time model's variance.

Usage:
    python scripts/calibrate_stat_dispersion.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import minimize_scalar
from scipy.stats import nbinom, poisson

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

PROJ_DIR = PROJECT_ROOT / "data" / "projections"
STATS_DIR = PROJECT_ROOT / "data" / "stats"
YEARS = [2022, 2023, 2024]

# model stat key -> CSV column name (verified). k=SO, h_allowed=H.
HITTER_COLS = {"r": "R", "hr": "HR", "rbi": "RBI", "sb": "SB", "h": "H"}
PITCHER_COLS = {"w": "W", "k": "SO", "sv": "SV", "er": "ER", "bb": "BB", "h_allowed": "H"}

# Sentinel for "no overdispersion -> use Poisson" (NegBin r -> inf).
POISSON_SENTINEL = float("inf")

# Bounds for the log-r search. exp(13.8) ~ 1e6: at the upper bound the NegBin is
# indistinguishable from Poisson, so we treat hitting it as the Poisson floor.
_LOG_R_LO = np.log(1e-3)
_LOG_R_HI = np.log(1e6)


def fit_dispersion(x: np.ndarray, mu: np.ndarray) -> float:
    """MLE of a single NegBin dispersion r for counts x with per-obs means mu.

    Each observation is x_i ~ NegBin(mean=mu_i, dispersion=r) with a shared r
    (heteroscedastic means, one dispersion). Returns POISSON_SENTINEL when the
    optimizer yields r_hat >= 200: for genuinely Poisson data the MLE drifts
    toward large r rather than pinning to a specific value, so a large r_hat is
    the Poisson signature. The threshold is then applied as a clamp.
    """
    x = np.asarray(x, dtype=float)
    mu = np.asarray(mu, dtype=float)
    mask = mu > 0
    x, mu = x[mask], mu[mask]

    def nll(log_r: float) -> float:
        r = np.exp(log_r)
        p = r / (r + mu)
        return -float(np.sum(nbinom.logpmf(x, r, p)))

    res = minimize_scalar(nll, bounds=(_LOG_R_LO, _LOG_R_HI), method="bounded")
    r_hat = float(np.exp(res.x))
    # r >= 200: for genuinely Poisson data the MLE drifts toward the upper
    # search bound rather than pinning, so r_hat >> 200 is the Poisson
    # signature. 200 is a conservative cutoff -- at typical stat volumes
    # (mu < 20) the excess variance over Poisson is under 10%, and real
    # baseball dispersions are r ~ 1-20, well below it.
    if r_hat >= 200.0:
        return POISSON_SENTINEL
    return r_hat


def interval_coverage(x: np.ndarray, mu: np.ndarray, r: float, level: float) -> float:
    """Fraction of x inside the central `level` predictive interval of the model.

    r == POISSON_SENTINEL uses Poisson(mu); otherwise NegBin(mean=mu, disp=r).
    """
    x = np.asarray(x, dtype=float)
    mu = np.asarray(mu, dtype=float)
    lo_q, hi_q = (1.0 - level) / 2.0, (1.0 + level) / 2.0
    if r == POISSON_SENTINEL:
        lo = poisson.ppf(lo_q, mu)
        hi = poisson.ppf(hi_q, mu)
    else:
        p = r / (r + mu)
        lo = nbinom.ppf(lo_q, r, p)
        hi = nbinom.ppf(hi_q, r, p)
    return float(np.mean((x >= lo) & (x <= hi)))


def _read(path: Path, cols: list[str]) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    df = pd.read_csv(path, encoding="utf-8-sig")
    if "MLBAMID" not in df.columns:
        return pd.DataFrame()
    # dropna MUST precede astype(int): a NaN MLBAMID would crash the int cast.
    df = df.dropna(subset=["MLBAMID"]).copy()
    df["MLBAMID"] = df["MLBAMID"].astype(int)
    keep = ["MLBAMID", *[c for c in cols if c in df.columns]]
    return df[keep]


def _find_proj(year: int, system: str, kind: str) -> Path | None:
    d = PROJ_DIR / str(year)
    # Match calibrate_playing_time.py: try both separator forms before globbing.
    for name in (f"{system}-{kind}.csv", f"{system}_{kind}.csv"):
        if (d / name).exists():
            return d / name
    matches = sorted(d.glob(f"{system}-{kind}*.csv"))
    return matches[0] if matches else None


def _blend_proj(year: int, kind: str, cols: list[str]) -> pd.DataFrame:
    s = _read(_find_proj(year, "steamer", kind) or Path("x"), cols)
    z = _read(_find_proj(year, "zips", kind) or Path("x"), cols)
    if s.empty or z.empty:
        return pd.DataFrame()
    m = s.merge(z, on="MLBAMID", suffixes=("_s", "_z"))
    out = pd.DataFrame({"MLBAMID": m["MLBAMID"]})
    for c in cols:
        cs, cz = f"{c}_s", f"{c}_z"
        if cs in m.columns and cz in m.columns:
            out[c] = (m[cs] + m[cz]) / 2.0
    return out


def build_residuals(kind: str) -> dict[str, pd.DataFrame]:
    """Per-stat DataFrame of {year, actual, mu} conditioned on realized PT.

    mu = (proj_count / proj_PT) * actual_PT. Rows with actual_PT <= 0 (the
    PT-loss tail owned by the playing-time model) are excluded.
    """
    is_hitter = kind == "hitters"
    pt = "PA" if is_hitter else "IP"
    colmap = HITTER_COLS if is_hitter else PITCHER_COLS
    proj_cols = [pt, *colmap.values()]
    actual_cols = [pt, *colmap.values()]

    per_stat: dict[str, list[pd.DataFrame]] = {k: [] for k in colmap}
    for year in YEARS:
        proj = _blend_proj(year, kind, proj_cols)
        actual = _read(STATS_DIR / f"{kind}-{year}.csv", actual_cols)
        if proj.empty or actual.empty:
            print(f"  {kind} {year}: projection or actuals missing, skipping")
            continue
        m = proj.merge(actual, on="MLBAMID", suffixes=("_proj", "_act"))
        m = m[m[f"{pt}_act"] > 0]
        # proj_rate = proj_count / proj_PT would be inf/NaN if a projection row
        # has 0 PA/IP (FanGraphs can emit 0-PT prospect lines in some vintages);
        # the NaN mu then fails the `mu > 0` filter and the row vanishes
        # silently. This makes projection-side filtering explicit and symmetric
        # with the actual-side filter above.
        m = m[m[f"{pt}_proj"] > 0]
        for key, col in colmap.items():
            # Guard: a projection system may lack a column; merge only suffixes
            # overlapping names, so f"{col}_proj" can be absent. Skip rather than
            # KeyError (and real data has all columns, so this rarely triggers).
            if f"{col}_proj" not in m.columns or f"{col}_act" not in m.columns:
                continue
            proj_rate = m[f"{col}_proj"] / m[f"{pt}_proj"]
            mu = proj_rate * m[f"{pt}_act"]
            df = pd.DataFrame(
                {
                    "year": year,
                    "actual": m[f"{col}_act"].astype(float),
                    "mu": mu.astype(float),
                }
            )
            df = df[df["mu"] > 0]
            per_stat[key].append(df)
    return {
        k: pd.concat(v, ignore_index=True) if v else pd.DataFrame() for k, v in per_stat.items()
    }
