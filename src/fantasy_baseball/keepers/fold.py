"""The fold: shrink a rate residual, gate on realized playing time, reconstruct a line.

Pure functions -- no I/O, no config. Increment 2 reuses this module unchanged.

Two rules here are load-bearing and were both wrong in earlier spec drafts:
  * The shrink applies to RATE residuals only. Applying it to playing time would
    damp an injury signal in proportion to the playing time the injury suppressed.
  * Each rate multiplies its OWN denominator, with AB derived from PA first.
    Multiplying H/AB by PA inflates AVG by 1/0.8977 (a .250 hitter scores .278).
"""

from __future__ import annotations

from collections.abc import Mapping

import pandas as pd

from fantasy_baseball.keepers.actuals import safe_ratio


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
    base: pd.DataFrame,
    residual: pd.DataFrame,
    weight: pd.Series | Mapping[str, pd.Series],
    k: float | Mapping[str, float],
) -> pd.DataFrame:
    """base + k * weight * residual, per column, floored at 0.

    BOTH `k` and `weight` may be per-column mappings, and both have to be, because
    the study calibrated them per column:

      * `k` differs by column (er_ip 0.343, ab_pa 0.687, k_ip 1.0, ...).
      * `weight` differs by KIND of column. The eleven rate coefficients were fit
        at `w = shrink(pt, n0)`; the playing-time coefficient was fit UNSHRUNK at
        `w = 1` (spec 5.3 -- damping the PT residual by a function of the playing
        time an injury suppressed is circular). Passing one weight for all columns
        therefore cannot reproduce the calibrated model, and folding PT at the
        rate weight silently attenuates the playing-time move.

    Build the weights with `coefficients.FoldPolicy.serve_weights`, which composes
    them correctly; do not assemble them by hand. A column missing from `k` is not
    folded. A NaN residual means "no observation" and passes through unmoved -- it
    must never read as a move of zero-minus-base.
    """
    cols = base.columns
    if isinstance(k, Mapping):
        factors = pd.Series([float(k.get(col, 0.0)) for col in cols], index=cols)
    else:
        factors = pd.Series(float(k), index=cols)
    per_col = weight if isinstance(weight, Mapping) else dict.fromkeys(cols, weight)
    weights = pd.DataFrame({col: per_col[col] for col in cols}, index=base.index)
    moved: pd.DataFrame = base + residual[cols].fillna(0.0).mul(weights).mul(factors, axis=1)
    return moved.clip(lower=0.0)


def _guarded(numer: pd.Series, denom: pd.Series) -> pd.Series:
    """`safe_ratio` with 0/0 resolved to 0.0 rather than NaN.

    Same divide-by-zero guard as the input side; only the post-NaN policy differs.
    On the OUTPUT side a reconstructed line needs a number the scoring path can
    consume, and a player with zero playing time genuinely has a zero line.
    """
    result: pd.Series = safe_ratio(numer, denom).fillna(0.0)
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


def gate_ramp(realized_pt: pd.Series, threshold: float, width: float) -> pd.Series:
    """A linear on-ramp for the gate, in [0, 1], replacing the hard cliff.

    `gate_mask` is fully on or fully off at `threshold`. Because the playing-time
    term is unshrunk (spec 5.3), that boundary is a large discontinuity: a regular
    lost to a May injury with a 120-PA line against a 400-PA ZiPS 2026 and a 380-PA
    ZiPS 2027 passes through at 380 PA just below the gate and folds to
    380 + k*(120 - 400) PA just above it -- a drop of tens of percent across a
    couple of plate appearances. Spec 5.4 requires increment 1 either publish that
    magnitude or specify a ramp; this is the ramp.

    Ramps linearly from 0 at `threshold` to 1 at `threshold + width`. Below the
    threshold a player is unfolded exactly as before, so the passthrough rule and
    the "absence is not zero" rule (NaN -> 0.0 -> unfolded) are unchanged; the
    ramp only removes the step at the boundary.
    """
    if width <= 0:
        raise ValueError(f"ramp width must be positive, got {width}")
    filled = realized_pt.fillna(0.0)
    result: pd.Series = ((filled - threshold) / width).clip(lower=0.0, upper=1.0)
    return result
