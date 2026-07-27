"""The fold: shrink a rate residual, gate on realized playing time, reconstruct a line.

Pure functions -- no I/O, no config. Increment 2 reuses this module unchanged.

Two rules here are load-bearing and were both wrong in earlier spec drafts:
  * The shrink applies to RATE residuals only. Applying it to playing time would
    damp an injury signal in proportion to the playing time the injury suppressed.
  * Each rate multiplies its OWN denominator, with AB derived from PA first.
    Multiplying H/AB by PA inflates AVG by 1/0.8977 (a .250 hitter scores .278).
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def shrink(n: pd.Series, n0: float) -> pd.Series:
    """Sample-size shrink in [0, 1): n / (n + n0). Never amplifies.

    `n` is REALIZED playing time -- only observed opportunities carry sampling
    noise, so a blended full-season figure would understate it (spec 5.3).
    """
    filled = n.fillna(0.0).clip(lower=0.0)
    result: pd.Series = filled / (filled + n0)
    return result


def gate_mask(realized_pt: pd.Series, threshold: float) -> pd.Series:
    """True where the player has enough realized MLB playing time to be folded.

    NaN (absent from the MLB leaderboard -- i.e. in the minors) is False: absence
    is not an observation of zero, and folding it would gut the projection.
    """
    result: pd.Series = realized_pt.fillna(0.0) >= threshold
    return result


def fold_rates(
    base: pd.DataFrame, residual: pd.DataFrame, weight: pd.Series, k: float
) -> pd.DataFrame:
    """base + k * weight * residual, per rate column, floored at 0."""
    out = base.copy()
    for col in base.columns:
        moved = base[col] + k * weight * residual[col].fillna(0.0)
        out[col] = moved.clip(lower=0.0)
    return out


def _guarded(numer: pd.Series, denom: pd.Series) -> pd.Series:
    result: pd.Series = numer.divide(denom.where(denom > 0, other=np.nan)).fillna(0.0)
    return result


def reconstruct_hitter(rates: pd.DataFrame, pa: pd.Series) -> pd.DataFrame:
    ab = (pa * rates["ab_pa"]).clip(lower=0.0)
    h = ab * rates["h_ab"]
    return pd.DataFrame(
        {
            "pa": pa,
            "ab": ab,
            "h": h,
            "avg": _guarded(h, ab),
            "hr": pa * rates["hr_pa"],
            "r": pa * rates["r_pa"],
            "rbi": pa * rates["rbi_pa"],
            "sb": pa * rates["sb_pa"],
        },
        index=rates.index,
    )


def reconstruct_pitcher(rates: pd.DataFrame, ip: pd.Series) -> pd.DataFrame:
    er = ip * rates["er_ip"]
    bb = ip * rates["bb_ip"]
    hits = ip * rates["h_ip"]
    return pd.DataFrame(
        {
            "ip": ip,
            "er": er,
            "bb": bb,
            "h_allowed": hits,
            "era": _guarded(9.0 * er, ip),
            "whip": _guarded(bb + hits, ip),
            "k": ip * rates["k_ip"],
            "w": ip * rates["w_ip"],
        },
        index=rates.index,
    )
