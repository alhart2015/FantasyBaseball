"""Calibrate per-stat Negative-Binomial dispersion from projection-vs-actual
residuals, conditional on realized playing time (2022-2024), and the correlation
between a player's stats that the Monte Carlo's Gaussian copula draws with
(HITTER_CORRELATION / PITCHER_CORRELATION, #198).

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
from scipy.special import ndtri
from scipy.stats import nbinom, poisson

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from fantasy_baseball.utils.constants import CLOSER_SV_THRESHOLD, STAT_DISPERSION
from fantasy_baseball.utils.dispersion import resolve_dispersion_r

PROJ_DIR = PROJECT_ROOT / "data" / "projections"
STATS_DIR = PROJECT_ROOT / "data" / "stats"
YEARS = [2022, 2023, 2024]
# Held out of the fit and printed as a check on the correlation matrices.
HOLDOUT_YEARS = [2025]

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


def role_stable_sv(df: pd.DataFrame, threshold: float = CLOSER_SV_THRESHOLD) -> pd.DataFrame:
    """Keep only pitcher-seasons that were closers in BOTH projection and reality.

    Restricting the SV dispersion fit to role-stable closers keeps the single
    NegBin r from being inflated by job-loss events it cannot shape-model (the
    role-mixture tail is a documented, deferred limitation).
    """
    stable = (df["proj_sv"] >= threshold) & (df["actual_sv"] >= threshold)
    return df[stable].reset_index(drop=True)


def fit_banded_dispersion(
    df: pd.DataFrame, n_bands: int = 4, min_per_band: int = 150
) -> list[tuple[float, float]]:
    """Fit a separate r per qcut(mu) band; return [(mu_upper, r), ...] last=inf.

    The number of bands is capped so each holds at least ``min_per_band`` rows;
    a thin stat (e.g. role-stable saves, n~43) collapses to a single band -- a
    robust scalar -- rather than overfitting noisy per-band r's.
    """
    n_eff = max(1, min(n_bands, len(df) // min_per_band))
    d = df.copy()
    d["bin"] = pd.qcut(d["mu"], q=n_eff, labels=False, duplicates="drop")
    bins = sorted(d["bin"].unique())
    bands: list[tuple[float, float]] = []
    for i, b in enumerate(bins):
        g = d[d["bin"] == b]
        r = fit_dispersion(g["actual"].to_numpy(), g["mu"].to_numpy())
        upper = float("inf") if i == len(bins) - 1 else float(g["mu"].max())
        bands.append((upper, r))
    return bands


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


def build_joint(kind: str, years: list[int] | None = None) -> pd.DataFrame:
    """One row per player-season with every stat's actual, mu and projected count:
    ``<key>_x``, ``<key>_mu`` (= projected rate x REALIZED PT, so performance is
    measured given playing time), ``<key>_proj``. Rows with no realized or no
    projected PT are dropped (the PT-loss tail is the playing-time model's; a 0-PT
    projection row would give an inf/NaN rate). A stat a vintage lacks is left out
    (its columns absent) rather than raising."""
    is_hitter = kind == "hitters"
    pt = "PA" if is_hitter else "IP"
    colmap = HITTER_COLS if is_hitter else PITCHER_COLS
    cols = [pt, *colmap.values()]
    out = []
    for year in YEARS if years is None else years:
        proj = _blend_proj(year, kind, cols)
        actual = _read(STATS_DIR / f"{kind}-{year}.csv", cols)
        if proj.empty or actual.empty:
            print(f"  {kind} {year}: projection or actuals missing, skipping")
            continue
        m = proj.merge(actual, on="MLBAMID", suffixes=("_proj", "_act"))
        m = m[(m[f"{pt}_act"] > 0) & (m[f"{pt}_proj"] > 0)]
        d = pd.DataFrame({"year": year}, index=m.index)
        for key, col in colmap.items():
            if f"{col}_proj" not in m.columns or f"{col}_act" not in m.columns:
                continue
            d[f"{key}_x"] = m[f"{col}_act"].astype(float)
            d[f"{key}_mu"] = (m[f"{col}_proj"] / m[f"{pt}_proj"] * m[f"{pt}_act"]).astype(float)
            d[f"{key}_proj"] = m[f"{col}_proj"].astype(float)
        out.append(d)
    return pd.concat(out, ignore_index=True) if out else pd.DataFrame()


def build_residuals(kind: str) -> dict[str, pd.DataFrame]:
    """Per-stat DataFrame of {year, actual, mu} (``build_joint``'s rows, mu > 0), plus
    the projected/actual SV counts for ``role_stable_sv`` (rate x PT mu loses the
    count)."""
    joint = build_joint(kind)
    colmap = HITTER_COLS if kind == "hitters" else PITCHER_COLS
    out: dict[str, pd.DataFrame] = {}
    for key in colmap:
        if f"{key}_mu" not in joint.columns:
            out[key] = pd.DataFrame()
            continue
        df = pd.DataFrame(
            {"year": joint["year"], "actual": joint[f"{key}_x"], "mu": joint[f"{key}_mu"]}
        )
        if key == "sv":
            df["proj_sv"] = joint["sv_proj"]
            df["actual_sv"] = joint["sv_x"]
        out[key] = df[df["mu"] > 0].reset_index(drop=True)
    return out


def copula_score_draws(
    x: np.ndarray, mu: np.ndarray, r: np.ndarray, rng: np.random.Generator, n_draws: int
) -> np.ndarray:
    """``n_draws`` x rows of normal scores of randomized PITs under NegBin(mu, r)
    (r inf = Poisson): the latent z the MC's copula maps through each stat's
    marginal. The CDF step is computed once; only the jitter inside it is drawn."""
    x = np.asarray(x, float)
    lo, hi = np.empty_like(x), np.empty_like(x)
    pois = np.isinf(r)
    hi[pois] = poisson.cdf(x[pois], mu[pois])
    lo[pois] = poisson.cdf(x[pois] - 1, mu[pois])
    p = r[~pois] / (r[~pois] + mu[~pois])
    hi[~pois] = nbinom.cdf(x[~pois], r[~pois], p)
    lo[~pois] = nbinom.cdf(x[~pois] - 1, r[~pois], p)
    u = lo + rng.random((n_draws, len(x))) * (hi - lo)
    return np.asarray(ndtri(np.clip(u, 1e-6, 1 - 1e-6)))


def copula_correlation(
    joint: pd.DataFrame,
    keys: list[str],
    n_draws: int = 20,
    seed: int = 198,
    min_rows: int = 30,
) -> tuple[np.ndarray, np.ndarray]:
    """Correlation of the copula scores for each pair of stats, and each pair's row
    count. Scores use the SHIPPED marginals (``STAT_DISPERSION``), the ones the MC
    maps the latents through. A row counts for a stat when its mean is positive and
    its actual is known; SV only on role-stable closers (``role_stable_sv``'s
    rule): for other relievers a save count is too small to measure a correlation
    from, and a closer who lost the job is the role mixture's draw, not the
    copula's. A pair with fewer than ``min_rows`` shared rows is left at 0. The
    matrix is then nudged to the nearest positive-definite one so the MC's Cholesky
    works (pairwise rows differ)."""
    rng = np.random.default_rng(seed)
    z: dict[str, np.ndarray] = {}
    for key in keys:
        if f"{key}_mu" not in joint.columns:
            continue
        mu = joint[f"{key}_mu"].to_numpy()
        x = joint[f"{key}_x"].to_numpy()
        ok = (mu > 0) & ~np.isnan(x)
        if key == "sv":
            ok &= (joint["sv_proj"].to_numpy() >= CLOSER_SV_THRESHOLD) & (x >= CLOSER_SV_THRESHOLD)
        zk = np.full((n_draws, len(joint)), np.nan)
        r = resolve_dispersion_r(STAT_DISPERSION[key], mu[ok])
        zk[:, ok] = copula_score_draws(x[ok], mu[ok], r, rng, n_draws)
        z[key] = zk
    k = len(keys)
    c, n = np.eye(k), np.zeros((k, k), int)
    for i in range(k):
        for j in range(i + 1, k):
            if keys[i] not in z or keys[j] not in z:
                continue
            both = ~np.isnan(z[keys[i]][0]) & ~np.isnan(z[keys[j]][0])
            n[i, j] = n[j, i] = int(both.sum())
            if both.sum() < min_rows:
                continue
            rho = np.mean(
                [
                    np.corrcoef(zi[both], zj[both])[0, 1]
                    for zi, zj in zip(z[keys[i]], z[keys[j]], strict=True)
                ]
            )
            c[i, j] = c[j, i] = rho
    w, v = np.linalg.eigh(c)
    if w.min() < 1e-6:
        c = v @ np.diag(np.clip(w, 1e-6, None)) @ v.T
        d = np.sqrt(np.diag(c))
        c = c / np.outer(d, d)
    return c, n


def _emit_correlation(name: str, keys: list[str], c: np.ndarray, *, paste: bool) -> None:
    """Print a matrix; paste-ready only when ``paste`` (a diagnostic is commented out
    so it can't be pasted over a constant by mistake)."""
    lead = "" if paste else "# "
    print(f"\n{lead}{name}: list[list[float]] = [")
    print(f"{lead}    #  " + "  ".join(f"{k:>6}" for k in keys))
    for i, key in enumerate(keys):
        row = ", ".join(f"{c[i, j]:+.3f}" for j in range(len(keys)))
        print(f"{lead}    [{row}],  # {key}")
    print(f"{lead}]")


def loso_coverage(data: pd.DataFrame, levels: tuple[float, ...] = (0.50, 0.80)) -> pd.DataFrame:
    """Leave-one-season-out coverage: fit r on the other years, test on the held-out."""
    rows = []
    years = sorted(data["year"].unique())
    for held in years:
        train = data[data["year"] != held]
        test = data[data["year"] == held]
        if train.empty or test.empty:
            continue
        r = fit_dispersion(train["actual"].to_numpy(), train["mu"].to_numpy())
        row = {"held_out": held, "r": r, "n": len(test)}
        for lv in levels:
            row[f"cov_{lv:.2f}"] = interval_coverage(
                test["actual"].to_numpy(), test["mu"].to_numpy(), r, lv
            )
        rows.append(row)
    return pd.DataFrame(rows)


def bucket_diagnostic(df: pd.DataFrame, r, n_bins: int = 4) -> pd.DataFrame:
    """Per projected-count bucket: Pearson dispersion of standardized residuals.

    r may be a scalar (broadcast) or a per-element array aligned to df rows.
    Implied conditional variance per row is mu + mu^2/r (mu when r is the Poisson
    sentinel inf). The Pearson statistic mean((actual-mu)^2 / implied) is ~1.0
    when the dispersion fits that bucket's mean range; it uses PER-OBSERVATION mu
    so it is immune to within-bucket mu spread. Systematic departure from 1.0
    across buckets means a single scalar r does not fit (escalate to banded).
    """
    d = df.copy()
    d["r_elem"] = r  # scalar broadcasts; array aligns positionally
    d["bin"] = pd.qcut(d["mu"], q=n_bins, labels=False, duplicates="drop")
    out = []
    for b, g in d.groupby("bin"):
        implied = np.where(np.isinf(g["r_elem"]), g["mu"], g["mu"] + g["mu"] ** 2 / g["r_elem"])
        pearson = float((((g["actual"] - g["mu"]) ** 2) / implied).mean())
        out.append(
            {"bin": int(b), "n": len(g), "mu_med": float(g["mu"].median()), "pearson": pearson}
        )
    return pd.DataFrame(out)


def _emit_dispersion(dispersion: dict) -> None:
    print("\nSTAT_DISPERSION = {")
    for k, v in dispersion.items():
        if isinstance(v, list):
            parts = ", ".join(
                f'(float("inf"), {r:.3f})' if b == float("inf") else f"({b:.1f}, {r:.3f})"
                for b, r in v
            )
            print(f'    "{k}": [{parts}],')
        elif v == POISSON_SENTINEL:
            print(f'    "{k}": float("inf"),')
        else:
            print(f'    "{k}": {v:.3f},')
    print("}")


def main() -> None:
    from fantasy_baseball.utils.constants import (
        HITTER_CORR_STATS,
        PITCHER_CORR_STATS,
    )

    print("=" * 72)
    print("STAT DISPERSION CALIBRATION (NegBin r), years:", YEARS)
    dispersion: dict[str, float | list[tuple[float, float]]] = {}
    for kind, keys in (("hitters", HITTER_CORR_STATS), ("pitchers", PITCHER_CORR_STATS)):
        res = build_residuals(kind)
        for key in keys:
            df = res.get(key, pd.DataFrame())
            if df.empty:
                print(f"  {key}: no data, skipping")
                continue
            if key == "sv":
                df = role_stable_sv(df)
                if df.empty:
                    print("  sv: no role-stable closers, skipping")
                    continue

            actual = df["actual"].to_numpy()
            mu = df["mu"].to_numpy()
            r_scalar = fit_dispersion(actual, mu)
            scalar_diag = bucket_diagnostic(df, r_scalar)

            if scalar_diag["pearson"].between(0.75, 1.35).all():
                value: float | list[tuple[float, float]] = r_scalar
                diag = scalar_diag
                kind_label = "scalar"
            else:
                value = fit_banded_dispersion(df)
                if len(value) == 1:
                    value = value[0][1]  # thin data -> single robust scalar
                r_elem = resolve_dispersion_r(value, mu)
                diag = bucket_diagnostic(df, r_elem)
                kind_label = "banded" if isinstance(value, list) else "scalar (thin)"
            dispersion[key] = value

            print(f"\n  {key}: {kind_label}  n={len(df)}")
            if isinstance(value, list):
                print(f"  bands (mu_upper, r): {value}")
            else:
                print(f"  r: {'Poisson' if value == POISSON_SENTINEL else round(value, 3)}")
            # LOSO only for scalar, non-sv stats; banded and role-stable-sv pools
            # are too thin to split by fold, so the in-sample bucket diagnostic
            # (Pearson ~1.0) is the acceptance signal there.
            if kind_label == "scalar" and key != "sv":
                print("  LOSO coverage:")
                print(loso_coverage(df).to_string(index=False))
            else:
                print(
                    "  (in-sample bucket diagnostic is the acceptance; LOSO skipped -- thin/banded)"
                )
            print("  bucket diagnostic (Pearson dispersion of standardized residuals):")
            print(diag.to_string(index=False))

    _emit_dispersion(dispersion)

    print("\n" + "=" * 72)
    print("COPULA CORRELATIONS (normal scores of PITs under the shipped STAT_DISPERSION)")
    if dispersion != STAT_DISPERSION:
        print(
            "  WARNING: the dispersion fitted above differs from STAT_DISPERSION; the "
            "correlations below use STAT_DISPERSION (what the MC samples). Paste both or "
            "neither."
        )
    for kind, keys, name, paste in (
        ("pitchers", PITCHER_CORR_STATS, "PITCHER_CORRELATION", True),
        # The hitter matrix stays a team-level stand-in until #538 (see constants.py),
        # so this one is a diagnostic, printed commented-out.
        ("hitters", HITTER_CORR_STATS, "HITTER_CORRELATION (DIAGNOSTIC, not shipped)", False),
    ):
        joint = build_joint(kind)
        c, n = copula_correlation(joint, keys)
        print(f"\n  {kind}: {len(joint)} player-seasons, {YEARS[0]}-{YEARS[-1]}")
        _emit_correlation(name, keys, c, paste=paste)
        pairs = [(i, j) for i in range(len(keys)) for j in range(i + 1, len(keys))]
        print("  rows per pair: " + ", ".join(f"{keys[i]}-{keys[j]} {n[i, j]}" for i, j in pairs))
        hold = build_joint(kind, HOLDOUT_YEARS)
        if not hold.empty:
            ch, _ = copula_correlation(hold, keys)
            print(
                f"  held out {HOLDOUT_YEARS}, {len(hold)} player-seasons: "
                + ", ".join(f"{keys[i]}-{keys[j]} {ch[i, j]:+.3f}" for i, j in pairs)
            )


if __name__ == "__main__":
    main()
