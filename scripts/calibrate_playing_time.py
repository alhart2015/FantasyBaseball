"""Calibrate the playing-time model from historical projections vs actuals.

The Monte Carlo's old injury model modeled realized playing time as a one-sided
haircut: a player either plays his full projected PA/IP or, with some
probability, loses a uniform-random chunk. Validation (see the exploration
scripts) showed that model was 2-3x too tight and structurally wrong (it can
never exceed the projection, but real players beat their projected PA/IP all
the time).

This script measures the real distribution of ``actual_PT / projected_PT``
and emits a two-sided, volume-scaled playing-time model intended to be the
single source of truth for BOTH:

  - the MC sampler (draw a per-player playing-time multiplier), and
  - ERoto's analytical SDs (multiply the central projection by mean_scale,
    add CV_pt in quadrature to the per-stat sigma).

Method (the "core fixes" agreed after the methodology + scout review):
  - Condition on PROJECTED playing time only (the deployment population --
    you only roster projected-relevant players); let actual roam free.
  - P1 fix: a projected player with NO actuals row played ~0, NOT dropped.
    (The old exploration used ``float(x) or 0``, and ``NaN or 0`` is NaN,
    silently dropping the season-ending-injury / lost-job tail -- the exact
    ``x or default`` numeric-falsy trap CLAUDE.md warns about.)
  - Split pitchers SP vs RP by a projected-IP threshold (PitcherStats has no
    GS field, so IP is the only role signal available where the model is
    applied; their PT processes differ: a 40-70 IP reliever is role-volatile,
    a 180 IP starter is not).
  - Fit MONOTONE volume curves (isotonic): mean_scale increases with
    projected volume, CV_pt decreases. Avoids overfitting noisy small-n bands.

Only PA / IP are read, so the missing-rate-column issue in the 2025 pitcher
actuals file is irrelevant here.

Usage:
    python scripts/calibrate_playing_time.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from fantasy_baseball.utils.constants import STARTER_IP_THRESHOLD

PROJ_DIR = PROJECT_ROOT / "data" / "projections"
STATS_DIR = PROJECT_ROOT / "data" / "stats"

YEARS = [2022, 2023, 2024, 2025]

# Population = REAL rosterable players, not every prospect FanGraphs assigns a
# nominal line. Steamer/ZiPS project hundreds of AAA players a ~200 PA / ~50 IP
# line who never reach MLB; including them (as 0 actual) craters the low-volume
# bands with phantoms. The MLB-appearance rate by projected-volume band shows a
# sharp knee: hitters jump 26% -> 91% played crossing ~300 PA; SP reach >=94%
# played by ~100 IP. Above the knee, an UNMATCHED player genuinely is an injury
# (count as 0 PT -- the tail we want); below it, unmatched is a phantom.
#
# Relievers are the exception: real setup men ARE projected only 40-70 IP, so a
# volume floor can't separate them from phantom arms. For RP we instead require
# an MLB appearance (drops phantoms; keeps the demoted/injured-reliever tail,
# who DID pitch). This is a deliberate, documented mild survivorship choice for
# RP only -- a projected reliever who throws zero MLB pitches all year is rare.
#
# SP/RP is split by a projected-IP threshold, NOT projected starts: PitcherStats
# carries no GS field, so projected IP is the only role signal available where
# the model is actually applied. Calibrating on the same signal keeps the curve
# lookup consistent between here and simulation.py / scoring.py.
HITTER_MIN_PA = 350
# Reuse the deployment threshold so calibration and the runtime curve lookup
# (utils.playing_time) can never silently drift. Also the SP volume floor.
SP_IP_THRESHOLD = STARTER_IP_THRESHOLD

# Floor for the projection merge (kept low; group filters below do the real work).
MERGE_MIN_PA = 200
MERGE_MIN_IP = 40

# Bins per group for the reviewable band table (balanced-n via qcut).
N_BINS = {"hitters": 5, "SP": 4, "RP": 3}

# Quantile levels for the empirical SHAPE ladder. The MC sampler draws
# u ~ Uniform(0,1), clamps to [first, last], and interpolates a standardized
# z = (ratio - band_mean) / band_sd at u. p99 is the realistic over-performance
# ceiling (replaces the flat PLAYING_TIME_MAX_SCALE = 2.0 clip); p01 carries the
# season-ending-injury tail. Stored as z (mean 0, sd 1) so the runtime can apply
# the volume-curve's mean_scale/cv_pt as location/scale and stay moment-consistent
# with ERoto, which keeps using just those two numbers.
QUANTILE_LEVELS = [0.01, 0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95, 0.99]


def _load(path: Path | None, cols: list[str]) -> pd.DataFrame:
    """Load a projection/actuals CSV, keep MLBAMID + requested columns.

    FanGraphs exports are UTF-8 with a BOM and may suffix the pitcher
    strikeout column differently; we only need PA/IP/GS/G here so no
    stat-name normalization is required.
    """
    if path is None or not path.exists():
        return pd.DataFrame()
    df = pd.read_csv(path, encoding="utf-8-sig")
    if "MLBAMID" not in df.columns:
        return pd.DataFrame()
    df = df.dropna(subset=["MLBAMID"]).copy()
    df["MLBAMID"] = df["MLBAMID"].astype(int)
    keep = ["MLBAMID", *[c for c in cols if c in df.columns]]
    return df[keep]


def _find_proj(year: int, system: str, kind: str) -> Path | None:
    d = PROJ_DIR / str(year)
    for name in (f"{system}-{kind}.csv", f"{system}_{kind}.csv"):
        if (d / name).exists():
            return d / name
    matches = sorted(d.glob(f"{system}-{kind}*.csv"))
    return matches[0] if matches else None


def _blend(year: int, kind: str, cols: list[str]) -> pd.DataFrame:
    """Steamer+ZiPS 50/50 blend for players present in BOTH systems."""
    s = _load(_find_proj(year, "steamer", kind), cols)
    z = _load(_find_proj(year, "zips", kind), cols)
    if s.empty or z.empty:
        return pd.DataFrame()
    merged = s.merge(z, on="MLBAMID", suffixes=("_s", "_z"))
    out = pd.DataFrame({"MLBAMID": merged["MLBAMID"]})
    for c in cols:
        cs, cz = f"{c}_s", f"{c}_z"
        if cs in merged.columns and cz in merged.columns:
            out[c] = (merged[cs] + merged[cz]) / 2.0
    return out


def build_table(kind: str) -> pd.DataFrame:
    """Per-player-season projected vs actual playing time, P1-fixed.

    Returns one row per projected-relevant player-season with the blended
    projected volume, the (possibly zero) actual volume, the ratio, and --
    for pitchers -- the projected start share used to assign SP/RP.
    """
    is_hitter = kind == "hitters"
    pt = "PA" if is_hitter else "IP"
    min_proj = MERGE_MIN_PA if is_hitter else MERGE_MIN_IP
    proj_cols = [pt]

    frames = []
    for year in YEARS:
        proj = _blend(year, kind, proj_cols)
        actual = _load(STATS_DIR / f"{kind}-{year}.csv", [pt])
        if proj.empty or actual.empty:
            print(f"  {kind} {year}: projection or actuals missing, skipping")
            continue
        proj = proj[proj[pt] >= min_proj]
        merged = proj.merge(
            actual.rename(columns={pt: "pt_actual"}),
            on="MLBAMID",
            how="left",
        )
        merged["matched"] = merged["pt_actual"].notna()
        # P1 fix: a projected player absent from actuals played ~0, not NaN.
        merged["pt_actual"] = merged["pt_actual"].where(merged["pt_actual"].notna(), 0.0)
        merged["year"] = year
        merged = merged.rename(columns={pt: "pt_proj"})
        frames.append(merged)

    if not frames:
        return pd.DataFrame()
    df = pd.concat(frames, ignore_index=True)
    df["ratio"] = df["pt_actual"] / df["pt_proj"]

    if not is_hitter:
        df["role"] = np.where(df["pt_proj"] >= SP_IP_THRESHOLD, "SP", "RP")
    else:
        df["role"] = "hitters"
    return df


def _isotonic(x: np.ndarray, y: np.ndarray, w: np.ndarray, increasing: bool) -> np.ndarray:
    model = IsotonicRegression(increasing=increasing, out_of_bounds="clip")
    return model.fit_transform(x, y, sample_weight=w)


def calibrate_group(df: pd.DataFrame, label: str, n_bins: int) -> list[dict]:
    """Bin by projected volume, summarize, and monotone-smooth.

    Returns the calibrated curve as a list of {vol, mean_scale, cv_pt, n}
    points (band centers). mean_scale is forced non-decreasing in volume,
    cv_pt non-increasing -- the relationships the data and baseball both
    support, which protects against noisy small-n bands.
    """
    df = df.copy()
    df["bin"] = pd.qcut(df["pt_proj"], q=n_bins, labels=False, duplicates="drop")
    rows = []
    for _b, g in df.groupby("bin"):
        r = g["ratio"]
        mean = float(r.mean())
        sd = float(r.std())
        # Standardized SHAPE ladder: z = (ratio - mean) / sd, so mean 0, sd 1.
        # The runtime re-applies the (monotone) mean_scale/cv_pt as location/scale.
        z = (r - mean) / sd if sd > 0 else r * 0.0
        rows.append(
            {
                "vol": float(g["pt_proj"].median()),
                "n": len(g),
                "played": float(g["matched"].mean()),
                "raw_mean": mean,
                "raw_sd": sd,
                "p10": float(r.quantile(0.10)),
                "p90": float(r.quantile(0.90)),
                "z_ladder": [float(z.quantile(q)) for q in QUANTILE_LEVELS],
            }
        )
    rows.sort(key=lambda d: d["vol"])

    vol = np.array([d["vol"] for d in rows])
    n = np.array([d["n"] for d in rows], dtype=float)
    mean_mono = _isotonic(vol, np.array([d["raw_mean"] for d in rows]), n, increasing=True)
    sd_mono = _isotonic(vol, np.array([d["raw_sd"] for d in rows]), n, increasing=False)

    # Global no-intercept slope as an alternative single-number anchor.
    x = df["pt_proj"].to_numpy()
    y = df["pt_actual"].to_numpy()
    slope = float(np.sum(x * y) / np.sum(x * x))

    print(f"\n  {label}  (n={len(df)}, no-intercept slope actual~proj = {slope:.3f})")
    print(
        f"  {'vol':>6} {'n':>5} {'played%':>8} {'raw_mean':>9} {'mono_mean':>10} "
        f"{'raw_sd':>8} {'mono_cv':>8} {'p10':>6} {'p90':>6}"
    )
    print("  " + "-" * 72)
    curve = []
    for i, d in enumerate(rows):
        print(
            f"  {d['vol']:>6.0f} {d['n']:>5} {d['played']:>7.0%} {d['raw_mean']:>9.3f} "
            f"{mean_mono[i]:>10.3f} {d['raw_sd']:>8.3f} {sd_mono[i]:>8.3f} "
            f"{d['p10']:>6.2f} {d['p90']:>6.2f}"
        )
        curve.append(
            {
                "vol": round(d["vol"], 1),
                "mean_scale": round(float(mean_mono[i]), 4),
                "cv_pt": round(float(sd_mono[i]), 4),
                "z_ladder": [round(z, 4) for z in d["z_ladder"]],
                "n": d["n"],
            }
        )
    return curve


def main() -> None:
    print("=" * 72)
    print("PLAYING-TIME MODEL CALIBRATION")
    print(f"Years: {YEARS}  |  condition: projected only (P1-fixed, no actual floor)")
    print("=" * 72)

    hitters = build_table("hitters")
    pitchers = build_table("pitchers")

    # Rosterability filters (see population note at top): volume floor at the
    # >=90%-played knee for hitters/SP (unmatched there = injury, kept as 0);
    # MLB-appearance requirement for RP (volume can't separate phantom arms).
    hitters = hitters[hitters["pt_proj"] >= HITTER_MIN_PA]
    sp = pitchers[pitchers["role"] == "SP"]
    rp = pitchers[(pitchers["role"] == "RP") & pitchers["matched"]]

    curves: dict[str, list[dict]] = {}

    print(f"\nHITTERS (volume = projected PA, floor {HITTER_MIN_PA}):")
    curves["hitters"] = calibrate_group(hitters, "hitters", N_BINS["hitters"])

    print(f"\nPITCHERS / SP (projected IP >= {SP_IP_THRESHOLD}, n={len(sp)}):")
    curves["SP"] = calibrate_group(sp, "pitchers/SP", N_BINS["SP"])

    print(
        f"\nPITCHERS / RP (projected IP < {SP_IP_THRESHOLD}, MLB-appearance required, n={len(rp)}):"
    )
    curves["RP"] = calibrate_group(rp, "pitchers/RP", N_BINS["RP"])

    print("\n" + "=" * 72)
    print("CALIBRATED CURVES (band centers; model interpolates between them)")
    print("=" * 72)
    print("\nPLAYING_TIME_CURVES = {")
    for key, pts in curves.items():
        print(f"    {key!r}: [")
        for p in pts:
            print(
                f"        {{'vol': {p['vol']}, 'mean_scale': {p['mean_scale']}, "
                f"'cv_pt': {p['cv_pt']}}},  # n={p['n']}"
            )
        print("    ],")
    print("}")
    print(
        "\nNote: 'mean_scale' is the multiplicative haircut on projected PA/IP; "
        "'cv_pt' is\nthe SD of actual/projected at that volume. Both monotone-smoothed."
    )

    print("\n" + "=" * 72)
    print("EMPIRICAL SHAPE LADDERS (standardized z = (ratio - mean) / sd per band)")
    print(f"QUANTILE_LEVELS = {QUANTILE_LEVELS}")
    print("=" * 72)
    print("\nPLAYING_TIME_SHAPE = {")
    for key, pts in curves.items():
        print(f"    {key!r}: [")
        for p in pts:
            print(f"        {{'vol': {p['vol']}, 'z': {p['z_ladder']}}},  # n={p['n']}")
        print("    ],")
    print("}")
    print(
        "\nNote: at runtime draw u ~ Uniform(0,1), clamp to [first, last] level,\n"
        "interpolate z at u over QUANTILE_LEVELS, then scale = mean_scale + z * cv_pt\n"
        "(with the fraction_remaining damping applied to mean_scale/cv_pt)."
    )


if __name__ == "__main__":
    main()
