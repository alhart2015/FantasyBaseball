"""How much of a season's deviation from projection carries into the next season.

The keeper question this answers: a player beat (or missed) his 2026 projection by
some gap G. How much of G was a real change in what he is, and how much was noise
that will regress? Call the persisting share **S**, so the part worth carrying
forward is `B = S * G` -- the breakout coefficient.

`S` is not asserted here. It is FIT, per statistic, from historical triples of
(projection for year Y, actual year Y, actual year Y+1):

    (actual_{Y+1} - projection_Y) = a + S * (actual_Y - projection_Y)

Read the regression as: "the projection missed by G this year; next year it misses
by `a + S*G`." An `S` of 0 means the gap was pure noise and the original projection
was already the best guess for next year. An `S` of 1 means the gap was entirely
real and permanent. Everything useful is in between, and the fitted value IS the
optimal shrinkage -- a noisy gap attenuates the slope on its own, so a statistic
whose single-season gap is mostly sampling error earns a low `S` without anyone
deciding that it should.

**The intercept is load-bearing; do not drop it.** `projection_Y` is a one-year-stale
baseline when used against year Y+1, so `a` absorbs the average year-over-year drift
(aging plus league run environment) that the stale projection does not know about.
It is a POPULATION average, though: it applies the same drift to a 23-year-old and a
36-year-old. Per-player aging is what the out-year ZiPS files are for, and folding
them in is the step after this one.

Rates and volume are fit SEPARATELY, and that separation is the point
-------------------------------------------------------------------
A counting stat is `volume * rate`. Its gap therefore mixes two things with very
different persistence: "he played more than we thought" and "he was better per plate
appearance than we thought." Fitting `S` on the counting stat alone estimates one
blended coefficient for both, and because counting stats are volume-dominated, that
coefficient mostly measures playing-time persistence -- a previous version of this
subsystem was deleted for exactly that failure (see
`docs/keeper-value-teardown-2026-08-01.md`).

So the canonical path fits `S` per RATE (`hr_pa`, `k_ip`, ...) and separately for
playing time (`pa`, `ip`). `fit_counting_share` fits the blended coefficient too,
not because it is the one to use but so the difference between them is a measured
number rather than an assumption.

Everything here is pure and I/O-free. `scripts/keeper_persistence.py` loads the data
and drives it.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

# The 5x5 categories, expressed in the canonical rate/PT schema of `keepers.actuals`.
# AVG is `h_ab` over `ab_pa * pa`; ERA is `er_ip * 9`; WHIP is `bb_ip + h_ip`. Every
# category is reachable from a playing-time column plus these rates.
HITTER_COUNTING: dict[str, str] = {"R": "r_pa", "HR": "hr_pa", "RBI": "rbi_pa", "SB": "sb_pa"}
PITCHER_COUNTING: dict[str, str] = {"W": "w_ip", "SV": "sv_ip", "SO": "k_ip"}


@dataclass(frozen=True)
class Share:
    """A fitted persistence coefficient for one column.

    `share` is S: the fraction of this year's gap that repeats next year. `intercept`
    is the average drift the stale baseline misses. `r2` is against the mean of the
    NEXT-year gap, so it answers "how much of next year's miss did knowing this
    year's miss explain" -- expect small values; single seasons are noisy and a
    genuinely predictive S can sit under 0.10.
    """

    column: str
    share: float
    intercept: float
    n: int
    r2: float
    # Standard error of `share`. The honest read on whether a coefficient is
    # separable from 0 (pure noise) or from 1 (fully persistent) at this sample size.
    stderr: float

    @property
    def separable_from_zero(self) -> bool:
        """Is S more than two standard errors above 0? If not, the gap carries no
        measurable signal for this stat and the projection alone is the better guess."""
        return self.share > 2 * self.stderr


def gap(observed: pd.Series, projected: pd.Series) -> pd.Series:
    """`observed - projected`, aligned on the shared index. NaN on either side stays NaN."""
    return observed.subtract(projected)


def fit_share(
    gap_now: pd.Series,
    gap_next: pd.Series,
    *,
    column: str,
    weights: pd.Series | None = None,
) -> Share:
    """Least-squares fit of `gap_next = intercept + share * gap_now`.

    `weights` (playing time, normally) targets the population we actually decide over:
    unweighted, the fit is dominated by low-volume players whose gap is mostly sampling
    error, which drags `share` toward zero for a reason that has nothing to do with the
    stat's real persistence.

    Rows with NaN on either side, or a non-positive weight, are dropped pairwise --
    a player missing one rate still informs every other rate.
    """
    frame = pd.DataFrame({"x": gap_now, "y": gap_next})
    if weights is not None:
        frame["w"] = weights
    else:
        frame["w"] = 1.0
    frame = frame.replace([np.inf, -np.inf], np.nan).dropna()
    frame = frame.loc[frame["w"] > 0]
    n = len(frame)
    if n < 3:
        raise ValueError(f"{column}: need at least 3 usable pairs to fit, got {n}")

    x, y, w = frame["x"].to_numpy(), frame["y"].to_numpy(), frame["w"].to_numpy()
    # A degenerate regressor (every gap identical) has no slope to estimate. Report it
    # as "no signal" rather than dividing by zero into a nan/inf that reads as a fit.
    wsum = w.sum()
    xbar, ybar = (w * x).sum() / wsum, (w * y).sum() / wsum
    sxx = (w * (x - xbar) ** 2).sum()
    if sxx <= 0:
        return Share(
            column=column, share=0.0, intercept=float(ybar), n=n, r2=0.0, stderr=float("inf")
        )

    sxy = (w * (x - xbar) * (y - ybar)).sum()
    slope = sxy / sxx
    intercept = ybar - slope * xbar

    resid = y - (intercept + slope * x)
    ss_res = (w * resid**2).sum()
    ss_tot = (w * (y - ybar) ** 2).sum()
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0
    # Weighted OLS slope SE. dof = n-2 for the two fitted parameters.
    stderr = float(np.sqrt(ss_res / (n - 2) / sxx)) if n > 2 else float("inf")

    return Share(
        column=column,
        share=float(slope),
        intercept=float(intercept),
        n=n,
        r2=float(r2),
        stderr=stderr,
    )


def apply_share(projected: pd.Series, gap_now: pd.Series, fit: Share) -> pd.Series:
    """The breakout-adjusted forecast: `projection + intercept + S * gap`.

    This is `B = S * G` from the design, plus the drift term. Where the gap is NaN
    (no observation), the projection passes through with only the drift applied --
    an unobserved player is not evidence of a zero gap.
    """
    return projected + fit.intercept + fit.share * gap_now.fillna(0.0)


def rmse(prediction: pd.Series, truth: pd.Series, weights: pd.Series | None = None) -> float:
    """Weighted root-mean-square error over rows where both sides are present."""
    frame = pd.DataFrame({"p": prediction, "t": truth})
    frame["w"] = 1.0 if weights is None else weights
    frame = frame.replace([np.inf, -np.inf], np.nan).dropna()
    frame = frame.loc[frame["w"] > 0]
    if frame.empty:
        return float("nan")
    err = (frame["p"] - frame["t"]).to_numpy()
    w = frame["w"].to_numpy()
    return float(np.sqrt((w * err**2).sum() / w.sum()))


def evaluate_shares(
    projected: pd.Series,
    gap_now: pd.Series,
    truth_next: pd.Series,
    fit: Share,
    *,
    weights: pd.Series | None = None,
) -> dict[str, float]:
    """RMSE of the fitted S against the two endpoints it has to beat.

    * `s0` -- ignore the season entirely; the projection was already the best guess.
    * `s1` -- trust the season completely; this year's line IS next year's talent.
    * `fitted` -- the estimated share.

    A fitted S that does not beat BOTH endpoints is not earning its place. `s0` is the
    one that matters most: it is the null hypothesis that reading the season adds
    nothing.
    """
    endpoints = {
        "s0": apply_share(
            projected, gap_now, Share(fit.column, 0.0, fit.intercept, fit.n, 0.0, 0.0)
        ),
        "s1": apply_share(
            projected, gap_now, Share(fit.column, 1.0, fit.intercept, fit.n, 0.0, 0.0)
        ),
        "fitted": apply_share(projected, gap_now, fit),
    }
    return {name: rmse(pred, truth_next, weights) for name, pred in endpoints.items()}


@dataclass(frozen=True)
class ReliabilityShare:
    """A persistence share that scales with how much the gap was measured on.

    **Measured, and NOT adopted.** On leave-one-transition-out this form beats the
    constant share by at most 0.2% (hitter `pa` +0.1%, `sv_ip` +0.2%, `k_ip` +0.0%),
    and for `ip`, `w_ip` and `bb_ip` the grid picks `k = 0` outright -- i.e. it asks
    to BE the constant. The volume-tercile wobble that motivated it was noise. Kept
    because it is the instrument that produced that negative result and the one to
    re-run when a fourth transition lands, not because anything ships on it.

    The fitted parameters are nonetheless sensible (hitters `k` of 93-443 PA, right in
    the known rate-stabilization range), so the form measures something real. It just
    does not improve the forecast, and per the teardown doc a gain inside the noise
    floor does not earn a parameter.

    `S(n) = s_max * n / (n + k)`, where `n` is the player's year-Y volume.

    Two parameters, each answering a different question:

    * `k` -- the volume at which HALF of `s_max` is earned. Pure measurement
      reliability: below it a gap is mostly sampling error, above it mostly real.
      `k = 0` collapses to a constant share, so the constant model is nested and
      "did the extra parameter buy anything" is a fair comparison.
    * `s_max` -- what a PERFECTLY measured gap is worth. Deliberately free rather
      than pinned at 1.0: a career year can be precisely observed and still regress,
      because true talent itself reverts. `s_max < 1` is that reversion; `k` is the
      measurement noise. Pinning `s_max = 1` would force every bit of regression to
      be explained as measurement error and inflate `k`.
    """

    column: str
    s_max: float
    k: float
    intercept: float
    n: int
    r2: float

    def share_at(self, sample_size: pd.Series) -> pd.Series:
        """The per-player share implied by each player's own year-Y volume."""
        return self.s_max * sample_size / (sample_size + self.k)


def fit_reliability_share(
    gap_now: pd.Series,
    gap_next: pd.Series,
    sample_size: pd.Series,
    *,
    column: str,
    weights: pd.Series | None = None,
    k_grid: np.ndarray | None = None,
) -> ReliabilityShare:
    """Fit `gap_next = a + s_max * (n/(n+k)) * gap_now` over a grid of `k`.

    `k` enters non-linearly, but `s_max` is linear once `k` is fixed -- so each grid
    point is one weighted least-squares solve on the shrunken regressor and the best
    `k` is chosen by weighted SSE. No optimizer, no convergence to babysit, and the
    whole profile is inspectable.

    The grid spans a wide log range anchored on the data's own scale, so it adapts to
    PA (hundreds) and IP (tens) without a caller-supplied magic number. `k = 0` is
    included so a stat with no reliability structure can return the constant fit
    rather than being forced onto a curve.
    """
    frame = pd.DataFrame({"x": gap_now, "y": gap_next, "n": sample_size})
    frame["w"] = 1.0 if weights is None else weights
    frame = frame.replace([np.inf, -np.inf], np.nan).dropna()
    frame = frame.loc[(frame["w"] > 0) & (frame["n"] > 0)]
    if len(frame) < 4:
        raise ValueError(f"{column}: need at least 4 usable rows to fit, got {len(frame)}")

    x, y, n, w = (frame[c].to_numpy() for c in ("x", "y", "n", "w"))
    if k_grid is None:
        # Cap the grid at a few times the observed spread. Far above it, n/(n+k) is
        # nearly proportional to n, so `k` and `s_max` trade off almost exactly and the
        # search wanders to a meaningless corner (s_max of 28 against a k of 17000 --
        # arithmetically a constant share, dressed up as a curve). Bounding k keeps the
        # curve genuinely curved ACROSS THE RANGE WE OBSERVE, which is the only range
        # the parameter is identified over.
        k_grid = np.concatenate([[0.0], np.geomspace(1.0, 3.0 * float(np.percentile(n, 95)), 150)])

    best: tuple[float, float, float, float] | None = None  # (sse, k, s_max, intercept)
    wsum = w.sum()
    ybar = (w * y).sum() / wsum
    for k in k_grid:
        z = (n / (n + k)) * x
        zbar = (w * z).sum() / wsum
        szz = (w * (z - zbar) ** 2).sum()
        if szz <= 0:
            continue
        s_max = (w * (z - zbar) * (y - ybar)).sum() / szz
        # A perfectly measured gap cannot carry MORE than itself forward. An s_max above
        # 1 is the degenerate corner above, not a discovery, so those grid points are
        # rejected outright rather than clipped -- clipping would keep their (wrong) k.
        if not 0.0 <= s_max <= 1.0:
            continue
        intercept = ybar - s_max * zbar
        sse = float((w * (y - (intercept + s_max * z)) ** 2).sum())
        if best is None or sse < best[0]:
            best = (sse, float(k), float(s_max), float(intercept))
    if best is None:
        raise ValueError(
            f"{column}: no admissible (k, s_max) on the grid -- every candidate implied "
            f"a share outside [0, 1]"
        )

    sse, k_hat, s_max, intercept = best
    ss_tot = (w * (y - ybar) ** 2).sum()
    return ReliabilityShare(
        column=column,
        s_max=s_max,
        k=k_hat,
        intercept=intercept,
        n=len(frame),
        r2=float(1.0 - sse / ss_tot) if ss_tot > 0 else 0.0,
    )


def apply_reliability_share(
    projected: pd.Series,
    gap_now: pd.Series,
    sample_size: pd.Series,
    fit: ReliabilityShare,
) -> pd.Series:
    """`projection + intercept + S(n) * gap`, with S(n) from each player's own volume.

    This is the form that distinguishes a half-season injured star from a full-season
    regular: the same raw gap earns a smaller share when it was measured on less
    playing time. Where the gap is unobserved the projection passes through with the
    drift only, matching `apply_share`.
    """
    share = fit.share_at(sample_size).fillna(0.0)
    return projected + fit.intercept + share * gap_now.fillna(0.0)


def centered_aging(
    out_year: pd.Series,
    base: pd.Series,
    *,
    weights: pd.Series | None = None,
) -> pd.Series:
    """ZiPS's per-player aging view, with its population average removed.

    `out_year - base` between two vintages FROM THE SAME MODEL RUN is pure aging: both
    saw the same seasons, so the only thing that differs is how ZiPS expects that
    specific player to develop or decline. That is precisely the per-player resolution
    a single fitted intercept cannot have.

    **Only the DIFFERENCES are taken, never the level.** The mean is subtracted so this
    term cannot move the population average, which stays owned by the fitted intercept.
    That split is not tidiness, it is a correction for a measured defect: ZiPS's out-year
    holds playing time nearly flat (mean +1.6 PA / -0.9 IP between its 2026 and 2027
    vintages) while realized volume drifts by -78.7 PA / -26.1 IP. Taking ZiPS's level
    would make every hitter roughly 79 PA too optimistic. Taking only its spread keeps
    the part it is good at -- who ages well relative to his peers -- and discards the
    part it does not model.

    A player absent from the out-year vintage yields NaN, which `fold_forecast` treats
    as "no differential information" and falls back to the population drift alone.
    """
    delta = out_year.subtract(base)
    usable = delta.replace([np.inf, -np.inf], np.nan).dropna()
    if usable.empty:
        return delta
    if weights is None:
        mean = float(usable.mean())
    else:
        w = weights.reindex(usable.index).fillna(0.0).clip(lower=0.0)
        mean = float((w * usable).sum() / w.sum()) if w.sum() > 0 else float(usable.mean())
    return delta - mean


def fold_forecast(
    base_projection: pd.Series,
    gap_now: pd.Series,
    fit: Share,
    aging_centered: pd.Series | None = None,
) -> pd.Series:
    """The full forecast: baseline + population drift + per-player aging + kept gap.

        forecast = base + intercept + centered_aging + S * gap

    `base_projection` must be the SAME vintage the gap was measured against, otherwise
    the drift is counted twice. With `aging_centered` omitted (or NaN for a player) this
    degrades exactly to `apply_share`, so a missing out-year file costs per-player
    resolution and nothing else.
    """
    folded = apply_share(base_projection, gap_now, fit)
    if aging_centered is None:
        return folded
    return folded + aging_centered.reindex(folded.index).fillna(0.0)


def fit_counting_share(
    proj_count: pd.Series,
    obs_count: pd.Series,
    next_count: pd.Series,
    *,
    column: str,
    weights: pd.Series | None = None,
) -> Share:
    """The same fit run directly on a COUNTING stat, volume and rate blended.

    Provided for comparison against the rate fit, not for use. A counting stat's gap
    is volume-dominated, so this coefficient largely reports how persistent playing
    time is. When it differs materially from the corresponding rate fit, that
    difference IS the playing-time confound, measured.
    """
    return fit_share(
        gap(obs_count, proj_count),
        gap(next_count, proj_count),
        column=column,
        weights=weights,
    )
