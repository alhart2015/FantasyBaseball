"""Characterize the SHAPE of realized/projected playing time (not just mean+cv).

Reuses the data loader + rosterability filters from calibrate_playing_time.py
and reports, per group (hitters / SP / RP) and per projected-volume band:

  - full quantile ladder (p1..p99, max) of ratio = actual_PT / projected_PT
  - skewness and the up/down spread asymmetry (sd above vs below the median)
  - the empirical over-performance ceiling (p95/p99/max) -- the realistic
    replacement for the flat PLAYING_TIME_MAX_SCALE = 2.0 clip
  - how the ceiling moves with projected volume

Read-only analysis; writes nothing.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from calibrate_playing_time import (
    HITTER_MIN_PA,
    build_table,
)

QUANTS = [0.01, 0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95, 0.99]


def _describe(r: pd.Series, label: str) -> None:
    med = r.median()
    above = r[r >= med] - med
    below = med - r[r < med]
    # RMS spread above vs below the median = asymmetry signal.
    sd_up = float(np.sqrt((above**2).mean())) if len(above) else 0.0
    sd_dn = float(np.sqrt((below**2).mean())) if len(below) else 0.0
    qs = r.quantile(QUANTS)
    print(f"\n{label}  (n={len(r)})")
    print(f"  mean={r.mean():.3f}  sd={r.std():.3f}  skew={r.skew():+.2f}  max={r.max():.2f}")
    print(
        f"  spread above median={sd_up:.3f}  below median={sd_dn:.3f}  (down/up={sd_dn / sd_up:.1f}x)"
    )
    print("  quantiles: " + "  ".join(f"p{int(q * 100)}={qs[q]:.2f}" for q in QUANTS))
    for thr in (1.0, 1.10, 1.25, 1.50, 2.00):
        print(f"    frac >= {thr:.2f}: {float((r >= thr).mean()):6.1%}", end="")
    print()


def _by_band(df: pd.DataFrame, label: str, n_bins: int) -> None:
    df = df.copy()
    df["bin"] = pd.qcut(df["pt_proj"], q=n_bins, labels=False, duplicates="drop")
    print(f"\n  {label}: over-performance ceiling by projected-volume band")
    print(f"  {'vol':>6} {'n':>5} {'median':>7} {'p90':>6} {'p95':>6} {'p99':>6} {'max':>6}")
    print("  " + "-" * 50)
    for _b, g in sorted(df.groupby("bin"), key=lambda kv: kv[1]["pt_proj"].median()):
        r = g["ratio"]
        print(
            f"  {g['pt_proj'].median():>6.0f} {len(g):>5} {r.median():>7.2f} "
            f"{r.quantile(0.90):>6.2f} {r.quantile(0.95):>6.2f} "
            f"{r.quantile(0.99):>6.2f} {r.max():>6.2f}"
        )


def main() -> None:
    hitters = build_table("hitters")
    pitchers = build_table("pitchers")
    hitters = hitters[hitters["pt_proj"] >= HITTER_MIN_PA]
    sp = pitchers[pitchers["role"] == "SP"]
    rp = pitchers[(pitchers["role"] == "RP") & pitchers["matched"]]

    print("=" * 74)
    print("PLAYING-TIME SHAPE: ratio = actual_PT / projected_PT  (2022-2025)")
    print("=" * 74)

    _describe(hitters["ratio"], "HITTERS (proj PA >= 350)")
    _by_band(hitters, "HITTERS", 5)

    _describe(sp["ratio"], "SP (proj IP >= 100)")
    _by_band(sp, "SP", 4)

    _describe(rp["ratio"], "RP (proj IP < 100, MLB appearance)")
    _by_band(rp, "RP", 3)


if __name__ == "__main__":
    main()
