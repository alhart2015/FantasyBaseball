"""Assemble ZiPS-vs-actual year pairs and measure the fit sample.

The one non-negotiable methodological constraint (spec 6.1): the base must be
ZiPS_Y, built knowing only through Y-1, so it has NOT already absorbed year Y --
mirroring production, where ZiPS 2027 has never seen 2026. Using ZiPS_{Y+1} as
the base would fit how much surprise ZiPS already absorbed and drive the
coefficient to zero by construction.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

import numpy as np
import pandas as pd

from fantasy_baseball.keepers.actuals import (
    HITTER_PT,
    PITCHER_PT,
    normalize_hitting,
    normalize_pitching,
)
from fantasy_baseball.keepers.fold import gate_mask, shrink
from fantasy_baseball.keepers.mlb_stats import fetch_mlb_season
from fantasy_baseball.keepers.vintages import load_vintage

# Year Y of each usable (Y, Y+1) pair. 2025 needs a complete 2026 season; 2021 has
# no ZiPS vintage on disk (data/projections starts at 2022).
PAIR_YEARS = (2022, 2023, 2024)


@dataclass(frozen=True)
class YearPair:
    """One (Y, Y+1) observation set, already aligned on mlbam_id."""

    year: int
    base: pd.DataFrame  # ZiPS_Y rates
    residual: pd.DataFrame  # actual_Y rates - ZiPS_Y rates
    target: pd.DataFrame  # actual_{Y+1} rates
    realized_pt: pd.Series  # actual_Y playing time (drives shrink and gate)
    target_pt: pd.Series  # actual_{Y+1} playing time


LAST_COMPLETE_SEASON = 2025


def build_pairs(
    player_type: str,
    cache_dir: Path,
    projections_root: Path,
    years: tuple[int, ...] = PAIR_YEARS,
) -> list[YearPair]:
    """Assemble (Y, Y+1) observation sets.

    Two properties are load-bearing and were both wrong in an earlier draft:

    * The frames carry the PLAYING TIME column alongside the rates. PT is the
      twelfth coefficient, and spec requirement 12 -- the systematic mean of the
      PT residual -- is the single hardest constraint on the estimator. Stripping
      PT here would make that requirement unaddressable.
    * Membership is `zips INTERSECT actual_Y` ONLY. Intersecting year Y+1 as well would
      precondition the sample on having survived, inflating the measured survival
      rate by 7-9 points AND removing non-survivors before any estimator sees
      them -- making spec requirement 5 unmeasurable. Absentees get a NaN target
      and 0.0 target playing time, which is the honest encoding of "did not play".
    """
    if player_type not in {"hitter", "pitcher"}:
        raise ValueError(f"player_type must be 'hitter' or 'pitcher', got {player_type!r}")
    group = "hitting" if player_type == "hitter" else "pitching"
    normalize = normalize_hitting if player_type == "hitter" else normalize_pitching
    pt_col = HITTER_PT if player_type == "hitter" else PITCHER_PT
    for year in years:
        if year + 1 > LAST_COMPLETE_SEASON:
            # fetch_or_cache never invalidates, so a mid-season pull would freeze
            # permanently. Fail loud rather than cache an in-progress season.
            raise ValueError(
                f"pair {year}->{year + 1} needs a complete {year + 1} season; "
                f"last complete is {LAST_COMPLETE_SEASON}"
            )
    # act_next of year Y is act_y of year Y+1, so the pairs overlap: fetch and
    # normalize each season once rather than once per pair that mentions it.
    actuals = {
        y: normalize(fetch_mlb_season(cache_dir, y, group))
        for y in sorted(set(years) | {y + 1 for y in years})
    }
    pairs: list[YearPair] = []
    for year in years:
        zips = load_vintage(year, projections_root, player_type)
        act_y, act_next = actuals[year], actuals[year + 1]
        ids = zips.index.intersection(act_y.index)
        cols = list(zips.columns)  # rates AND the playing-time column
        zips_rows = zips.loc[ids, cols]
        target = act_next.reindex(ids)[cols].copy()
        # A player absent from the year-Y+1 leaderboard has NO Y+1 rate (NaN is the
        # only honest answer) but he does have a well-defined Y+1 MLB playing time
        # of zero. Leaving PT as NaN here would drop every non-survivor from the
        # playing-time fit and evaluation, which is precisely the survivorship
        # deletion spec 6.3 warns against. Rates stay NaN; only PT is filled.
        target[pt_col] = target[pt_col].fillna(0.0)
        pairs.append(
            YearPair(
                year=year,
                base=zips_rows,
                residual=act_y.loc[ids, cols] - zips_rows,
                target=target,
                realized_pt=act_y.loc[ids, pt_col],
                target_pt=target[pt_col],
            )
        )
    return pairs


def survivorship(pairs: list[YearPair], threshold: float) -> pd.DataFrame:
    """Per pair: how many cleared `threshold` in year Y, and how many again in Y+1.

    Fitting on survivors alone measures persistence GIVEN continued play, which
    biases the playing-time coefficient upward. Spec 6.3 requires this measured on
    the actual fit sample, not on the wider MLB population.
    """
    rows = []
    for pair in pairs:
        in_year = pair.realized_pt >= threshold
        survived = in_year & (pair.target_pt >= threshold)
        n_in, n_sur = int(in_year.sum()), int(survived.sum())
        rows.append(
            {
                "year": pair.year,
                "n_matched": len(pair.base),
                "n_in_year": n_in,
                "n_survived": n_sur,
                "survival_rate": (n_sur / n_in) if n_in else float("nan"),
            }
        )
    return pd.DataFrame(rows)


class Estimator(Protocol):
    name: str

    def fit(
        self,
        pairs: list[YearPair],
        column: str,
        n0: float,
        *,
        shrunk: bool = True,
        weighted: bool = True,
    ) -> FittedK:
        """Fit one coefficient for `column`.

        `shrunk` and `weighted` mirror the evaluation switches in `leave_one_out`
        so a fitted estimator can optimize the SAME loss it is scored on (finding
        A.1). The fixed endpoints ignore them.
        """
        ...


class _FixedTransfer:
    """Endpoint estimator: `k` is a constant, not fitted.

    Both endpoints and the fitted estimator return the same `FittedK`, so the
    shipped prediction form is written ONCE. That matters: the whole held-out
    comparison rests on all three applying an identical functional form, and two
    copies of `base + k * w * residual` could silently drift apart.
    """

    name: str
    k: float

    def fit(self, *_args: object, **_kwargs: object) -> FittedK:
        return FittedK({"k": self.k})


class ZeroTransfer(_FixedTransfer):
    """k = 0: ignore the season entirely -- the do-nothing endpoint spec 6.2
    requirement 3 requires the study to beat.

    NOT what main currently ships. Since PR #259, `analysis/keeper_value.py`
    scales a current-season anchor by a ZiPS ratio and regresses it toward the
    ZiPS out-year at `DEFAULT_OUT_YEAR_REGRESSION = 0.6`, so the incumbent already
    carries roughly 40% of the realized-season signal. Beating `k=0` therefore
    says the fold beats ignoring the season, NOT that it beats the shipped
    estimator. Increment 2 owns that comparison -- increment 1 cannot make it
    without importing `fantasy_baseball.analysis`, which spec 9 forbids.
    """

    name = "k=0"
    k = 0.0


class FullTransfer(_FixedTransfer):
    """k = 1: move the full (shrunk) surprise."""

    name = "k=1"
    k = 1.0


def usable(observed: pd.Series, predicted: pd.Series, weight: pd.Series) -> pd.Series:
    """Rows carrying enough information to be fit or scored.

    ONE definition, used by both `weighted_mse` and `ShrunkTransfer.fit`, because
    the fit sample must equal the evaluation sample -- fitting one loss and
    scoring another would make the held-out comparison meaningless. Two copies of
    this expression would let them drift apart silently.
    """
    result: pd.Series = observed.notna() & predicted.notna() & (weight > 0)
    return result


def weighted_mse(pred: pd.Series, actual: pd.Series, weight: pd.Series) -> float:
    """Playing-time-weighted MSE, so a 20-PA player's rate cannot dominate."""
    mask = usable(actual, pred, weight)
    if not mask.any():
        return float("nan")
    err = (pred[mask] - actual[mask]) ** 2
    return float((err * weight[mask]).sum() / weight[mask].sum())


def _eval_weight(pair: YearPair, weighted: bool) -> pd.Series:
    """The metric's weight column, per the pre-registered decision (finding A.1).

    Rate coefficients weight by realized year-Y+1 playing time so a 20-PA rate
    cannot dominate a 600-PA one. The playing-time coefficient is UNWEIGHTED:
    weighting the PT target by target_pt is circular, and it assigns weight 0 to
    every non-survivor -- deleting exactly the players whose lost playing time the
    coefficient exists to learn from.
    """
    if weighted:
        return pair.target_pt
    return pd.Series(1.0, index=pair.target_pt.index)


def leave_one_out(
    estimator: Estimator,
    pairs: list[YearPair],
    column: str,
    n0: float,
    *,
    shrunk: bool = True,
    weighted: bool = True,
) -> pd.DataFrame:
    """Fit on all pairs but one, evaluate on the held-out pair. Spec 6.3.

    `pairs` must ALREADY be gated -- pass them through `gated(pair, threshold)`
    once at the call site. Gating here would redo identical work for every
    column and every estimator, and would leave two places that decide which
    rows the study sees.

    `shrunk=False` is REQUIRED for the playing-time coefficient. The shrink damps
    noisy RATE observations; applying it to the PT residual would damp an injury
    signal in proportion to the playing time the injury suppressed, and would make
    the PT coefficient structurally unable to learn from lost time (spec 5.3).

    `weighted=False` is likewise required for the playing-time coefficient -- see
    `_eval_weight`. Fit and evaluation use the SAME weighting.
    """
    rows = []
    # The shrink weight depends only on (pair, n0, shrunk), none of which vary
    # across the folds, so it is computed once per pair rather than once per fold.
    weights = {p.year: _shrink_weight(p, n0, shrunk) for p in pairs}
    for held in pairs:
        train = [p for p in pairs if p.year != held.year]
        fitted = estimator.fit(train, column, n0, shrunk=shrunk, weighted=weighted)
        pred = fitted.predict(held.base[column], held.residual[column], weights[held.year])
        rows.append(
            {
                "estimator": estimator.name,
                "column": column,
                "held_out_year": held.year,
                "n": len(held.base),
                "error": weighted_mse(pred, held.target[column], _eval_weight(held, weighted)),
                **{f"param_{k}": v for k, v in fitted.params.items()},
            }
        )
    return pd.DataFrame(rows)


def _shrink_weight(pair: YearPair, n0: float, shrunk: bool) -> pd.Series:
    """The fold's shrink weight for one pair, or all-ones when unshrunk."""
    if shrunk:
        return shrink(pair.realized_pt, n0)
    return pd.Series(1.0, index=pair.realized_pt.index)


def gated(pair: YearPair, gate: float) -> YearPair:
    """Restrict a pair to rows clearing the realized-playing-time gate (spec 5.4).

    Delegates the threshold rule to `fold.gate_mask` so the fit sample and the
    serve path cannot drift apart on what "enough playing time" means.
    """
    ids = pair.realized_pt.index[gate_mask(pair.realized_pt, gate)]
    return YearPair(
        year=pair.year,
        base=pair.base.loc[ids],
        residual=pair.residual.loc[ids],
        target=pair.target.loc[ids],
        realized_pt=pair.realized_pt.loc[ids],
        target_pt=pair.target_pt.loc[ids],
    )


def _wls(
    x: np.ndarray, y: np.ndarray, w: np.ndarray, use_intercept: bool
) -> tuple[float, float, float]:
    """Weighted least squares of y on x (plus an intercept), returning
    (slope, intercept, robust slope standard error).

    The standard error is the weighted HC1 sandwich rather than the classical
    formula: year-over-year baseball residuals are heavy-tailed and strongly
    heteroskedastic in playing time, so the classical SE would understate the
    interval. `pinv` rather than `solve` so a degenerate design (a column with no
    variance) yields the minimum-norm answer instead of raising.
    """
    # Slope first, intercept second: column order in a design matrix is arbitrary,
    # and pinning the slope to index 0 keeps it there whether or not an intercept
    # is present, in both `beta` and `cov`.
    design = np.column_stack([x, np.ones_like(x)]) if use_intercept else x[:, None]
    xtwx = design.T @ (design * w[:, None])
    inv = np.linalg.pinv(xtwx)
    beta = inv @ (design.T @ (w * y))
    err = y - design @ beta
    n, p = len(y), design.shape[1]
    meat = design.T @ (design * ((w * err) ** 2)[:, None])
    cov = inv @ meat @ inv * (n / max(n - p, 1))
    slope = float(beta[0])
    intercept = float(beta[1]) if use_intercept else 0.0
    se = float(np.sqrt(max(float(cov[0, 0]), 0.0)))
    return slope, intercept, se


class FittedK:
    """A fitted transfer coefficient. Predicts the SHIPPED form only.

    `predict` deliberately does NOT add the fitted intercept: its production value
    is 0 (see `ShrunkTransfer`), so applying it during held-out evaluation would
    score a form the feature will never run, and would hand the fitted estimator a
    level correction that neither endpoint gets.
    """

    def __init__(self, params: dict[str, float]) -> None:
        self.params = params

    def predict(self, base: pd.Series, residual: pd.Series, weight: pd.Series) -> pd.Series:
        result: pd.Series = base + self.params["k"] * weight * residual.fillna(0.0)
        return result


class ShrunkTransfer:
    """Fit `k` in `pred = base + k * shrink * residual` by weighted least squares.

    The chosen estimator. How it answers spec 6.2's twelve requirements:

    1. **Same functional form in calibration and production.** The shipped form is
       `base + k * w * residual`, and that is exactly what `predict` computes, in
       held-out evaluation and at serve time alike. One term exists only in
       calibration: an additive nuisance intercept `c`, fit but never applied. Its
       production value is **0**, because `ZiPS_Y` is a projection *for year Y*
       while the calibration target is year Y+1, whereas `ZiPS_2027` is already
       aged forward to the year it is being folded into. `c` absorbs exactly that
       one-year pool-level drift so it does not leak into `k`. Note the intercept
       is additive, NOT a free scale term on the base: `a*Z + k*(A - Z)` rewrites
       as `(a - k)*Z + k*A`, which degenerates `k` into the plain OLS slope on
       `actual_Y` and destroys the meaning of the k=0 / k=1 endpoints. Here the
       coefficient on the base stays pinned at `1 - k*w` by construction, so both
       endpoints keep their meaning. Per-player aging is not attempted: no ZiPS
       vintage carries an Age column (spec requirement 1).
    12. **The systematic component of the playing-time residual.** ZiPS hedges
       playing time pool-wide, so the PT residual has a large nonzero mean that is
       not surprise. MEASURED ON THIS FIT SAMPLE it is -91 mean PA, not the +58 the
       spec quotes for a narrower "regulars" population -- ZiPS over-projects PA on
       the ZiPS-matched pool, and the fitted level term is -83.1 PA. The same
       intercept separates that level offset from the cross-sectional signal: `k`
       is then the slope on a player's deviation, not a blend of level and slope.
       This is the requirement that killed the two earlier estimators, and it is
       why the intercept is additive-and-unshipped rather than multiplicative.
    7. **Amplification bounded, not silently shipped.** The raw fit is reported as
       `k_raw` with a robust 95% interval, and the shipped `k` is clamped to
       `bounds` (default [0, 1]); `clamped` flags when the two differ.
    9/11. Metric, weights and gates are pre-registered in the finding document and
       enter here through `shrunk`/`weighted`, which mirror `leave_one_out` so the
       fit optimizes the same loss it is scored on.
    2/3/4/5/6/8/10 are properties of the harness and the finding, not of this
       class: leave-one-pair-out evaluation, the two endpoints, per-pair stability,
       survivorship, the stated conditioning on `n0` and the gate, the train/serve
       gap, and the shrink form.
    """

    name = "fitted-k"

    def __init__(
        self, bounds: tuple[float, float] = (0.0, 1.0), *, use_intercept: bool = True
    ) -> None:
        self.bounds = bounds
        self.use_intercept = use_intercept

    def fit(
        self,
        pairs: list[YearPair],
        column: str,
        n0: float,
        *,
        shrunk: bool = True,
        weighted: bool = True,
    ) -> FittedK:
        xs: list[np.ndarray] = []
        ys: list[np.ndarray] = []
        ws: list[np.ndarray] = []
        for pair in pairs:
            weight = _shrink_weight(pair, n0, shrunk)
            x = weight * pair.residual[column]
            y = pair.target[column] - pair.base[column]
            w = _eval_weight(pair, weighted)
            mask = usable(y, x, w)
            xs.append(x[mask].to_numpy(dtype=float))
            ys.append(y[mask].to_numpy(dtype=float))
            ws.append(w[mask].to_numpy(dtype=float))
        x_all = np.concatenate(xs) if xs else np.empty(0)
        y_all = np.concatenate(ys) if ys else np.empty(0)
        w_all = np.concatenate(ws) if ws else np.empty(0)
        n_fit = len(y_all)
        if n_fit < 3:
            raise ValueError(f"cannot fit {column!r}: only {n_fit} usable rows")
        k_raw, c_fit, se = _wls(x_all, y_all, w_all, self.use_intercept)
        lo, hi = self.bounds
        k = min(max(k_raw, lo), hi)
        return FittedK(
            {
                "k": k,
                "k_raw": k_raw,
                "clamped": float(k != k_raw),
                "c_fit": c_fit,
                "se_k": se,
                "ci_lo": k_raw - 1.96 * se,
                "ci_hi": k_raw + 1.96 * se,
                "n_fit": float(n_fit),
            }
        )
