"""Trajectory by SHAPE: learn from players whose career took a similar turn (#310).

`comps.comp_trajectory` matches on LEVEL -- same age, same current SGP, optionally the
same prior SGP -- and everything inside the band counts equally while everything outside
counts zero. For a star having a down year that is close to unusable. Soto at 21.5 ->
12.9 draws two comps, Johnny Damon and Darin Erstad, because a 27-year-old who was elite
and then fell 40% is a rare and specific box. Pujols misses the ceiling by 0.4 SGP and
contributes nothing.

The players you would actually reason from -- A-Rod, Trout, Cabrera, Harper, Pujols --
are excluded for a subtler reason: at 27 they all STAYED elite, so they are not comps for
a collapse. But ask the question by shape instead of by level and the pool is there. Of
elite (prior >= 16 SGP) age-25-30 seasons, 87 dropped to 45-75% of the year before --
Matt Kemp, Vlad Guerrero, Mookie Betts, Yelich, Machado, Jose Ramirez. They bounce: year
one averages 1.11x the down year, settling near 0.69x of the prior peak.

So this module does two things differently.

**A career is summarised by two anchors, not one level.** How good was he last year
(`peak`) and what is he doing now (`down`)? Both carry signal and their relative weight
is FIT, not chosen -- a weighted least squares of forward SGP on both, per horizon. That
is what converts a comp's future onto the query player's scale, the step level matching
cannot do at all.

**Nothing is excluded on a cliff.** Age contributes on a triangular kernel out to
`AGE_WINDOW` years, and peak level likewise, so a 26-year-old with an identical profile
informs a 27-year-old query instead of being discarded. Weight decays; it never jumps.

The output is a prediction rather than an average over a handful of careers, so
`PathPoint.n` counts FITTING rows and `survivors` describes that sample's attrition, not
a cohort of the query's own comps. `render` labels them accordingly.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .comps import PathPoint, Trajectory, collapse_split_seasons, played

#: Age contributes on a triangular kernel this many years either side. Ages 25-30 age
#: similarly enough to pool; beyond that the shape itself changes.
AGE_WINDOW = 2

#: Peak-level kernel half-width, in SGP. Wide, because `peak` is also a regressor and
#: the linear term carries level -- this only keeps the fit LOCAL enough that one
#: straight line does not have to serve fringe and elite players at once.
PEAK_BAND = 8.0

BOOTSTRAP_DRAWS = 1000

#: Intercept plus the two anchors. The residual degrees of freedom subtract this.
N_PARAMETERS = 3

#: Minimum EFFECTIVE size before a horizon is fit at all. Applied to the Kish size
#: rather than the row count, because kernel weights make those diverge badly: a 41-row
#: fit can carry an effective 15. Comfortably above `N_PARAMETERS`, since a fit with
#: barely more support than parameters interpolates its own sample.
MIN_EFFECTIVE_ROWS = 12.0


@dataclass(frozen=True)
class Anchors:
    """The fitted `forward = intercept + a*down + b*peak` for one horizon."""

    horizon: int
    intercept: float
    on_down: float
    on_peak: float
    n_fit: int
    #: Kish effective sample size, `(sum w)^2 / sum(w^2)`. The raw row count overstates
    #: support when most rows carry a small kernel weight.
    n_effective: float


def _weighted_quantile(values: np.ndarray, weights: np.ndarray, q: float) -> float:
    """The `q`-quantile of `values` under `weights`, by cumulative weight."""
    order = np.argsort(values)
    v, w = values[order], weights[order]
    total = w.sum()
    if total <= 0:
        return float("nan")
    return float(v[np.searchsorted(np.cumsum(w), q * total)])


def _triangular(distance: np.ndarray, width: float) -> np.ndarray:
    """1.0 at zero distance, tapering linearly to 0 at `width`, never negative."""
    clipped: np.ndarray = np.clip(1.0 - np.abs(distance) / width, 0.0, None)
    return clipped


def _weighted_least_squares(x: np.ndarray, y: np.ndarray, w: np.ndarray) -> np.ndarray:
    """Solve for [intercept, a, b] with weights `w`. `x` is (n, 2): down, peak."""
    design = np.column_stack([np.ones(len(x)), x])
    root = np.sqrt(w)[:, None]
    coefficients: np.ndarray
    coefficients, *_ = np.linalg.lstsq(design * root, y * np.sqrt(w), rcond=None)
    return coefficients


def build_history(panel: pd.DataFrame) -> pd.DataFrame:
    """One row per season that has an OBSERVABLE prior, carrying both anchors.

    A season whose predecessor falls before the panel begins is dropped rather than
    given a prior of 0 -- the same censoring the forward path uses, for the same reason:
    "we cannot see it" must never be scored as "he did not play". A player who was
    genuinely out of the league keeps his 0, which is a real observation.

    A split season (a mid-season trade, two rows for one player-year) is collapsed on
    BOTH sides, as `comps.comp_trajectory` does. Collapsing only the lookup would sum his
    future correctly while still entering him into the fit TWICE as two half-seasons,
    each with a half-size `down` against a full-size `peak` and a full-size forward
    value -- attenuating the fitted `on_down`, double-weighting him, and dragging
    `mean_start` low.
    """
    panel = collapse_split_seasons(panel).sort_values(["mlbam_id", "season"])
    first = int(panel["season"].min())
    index = panel.set_index(["mlbam_id", "season"])["sgp"]

    history = panel.copy()
    seasons = history["season"].to_numpy()
    # One vectorized reindex instead of a per-row MultiIndex .get. A missing key means
    # he was out of the league that year (a real 0); a key BEFORE the panel starts is
    # unobservable and stays NaN so `dropna` censors it.
    peak = index.reindex(
        pd.MultiIndex.from_arrays([history["mlbam_id"].to_numpy(), seasons - 1])
    ).to_numpy(dtype=float)
    history["peak"] = np.where(seasons - 1 >= first, np.nan_to_num(peak, nan=0.0), np.nan)
    return history.dropna(subset=["peak"]).rename(columns={"sgp": "down"})


def shape_trajectory(
    panel: pd.DataFrame,
    *,
    kind: str,
    age: int,
    sgp: float,
    peak: float,
    horizons: tuple[int, ...] = (1, 2, 3, 4, 5),
    age_window: int = AGE_WINDOW,
    peak_band: float = PEAK_BAND,
    last_complete_season: int | None = None,
    replacement: float = 0.0,
    seed: int = 0,
    bootstrap_draws: int = BOOTSTRAP_DRAWS,
) -> tuple[Trajectory, tuple[Anchors, ...]]:
    """Forward path for a player at `age`, now producing `sgp`, who produced `peak`.

    Returns the trajectory and the fitted anchors, so the coefficients that produced
    each number can be read rather than trusted.

    `replacement` fits the model on VALUE ABOVE REPLACEMENT instead of raw SGP: the
    response becomes `max(forward - replacement, 0)` while the two anchors stay on the
    SGP scale the caller supplies them in. Flooring the RESPONSE rather than shifting
    the fitted mean is what keeps a departed comp worth 0 instead of minus a floor, and
    it carries through to `spread`, `median` and `mean_if_survived` for free -- shifting
    afterwards moved only the mean and left the other three on a different scale.
    """
    if peak_band <= 0:
        raise ValueError(f"peak_band must be positive, got {peak_band}")
    if age_window < 1:
        raise ValueError(f"age_window must be at least 1, got {age_window}")
    if not horizons:
        raise ValueError("horizons must not be empty")

    horizons = tuple(sorted(horizons))
    last = last_complete_season if last_complete_season is not None else int(panel["season"].max())
    history = build_history(panel)
    index = panel.set_index(["mlbam_id", "season"])["sgp"]
    if index.index.has_duplicates:
        index = index.groupby(level=[0, 1]).sum()

    # Weight once: the kernels describe the QUERY's neighbourhood and do not move with
    # the horizon. `age_window + 1` so a player exactly `age_window` years away still
    # carries weight rather than sitting exactly on zero.
    weights = _triangular(history["age"].to_numpy(dtype=float) - age, age_window + 1) * _triangular(
        history["peak"].to_numpy(dtype=float) - peak, peak_band
    )
    # The NEAREST horizon sets membership, matching `comp_trajectory`: a season too
    # recent to have even one observable forward year enters no fit, so counting it in
    # `n_comps` (which render prints as "fit on N weighted seasons") would describe the
    # fit with rows the fit never saw.
    usable = (weights > 0) & (history["season"] + horizons[0] <= last).to_numpy()
    history, weights = history[usable].reset_index(drop=True), weights[usable]

    rng = np.random.default_rng(seed)
    ids, seasons = history["mlbam_id"].to_numpy(), history["season"].to_numpy()
    anchor_columns = history[["down", "peak"]].to_numpy(dtype=float)
    path, anchors, rows = [], [], []
    for h in horizons:
        # Look up ONLY the observable rows, vectorized. The previous form built a value
        # for every row through a per-row MultiIndex .get and then discarded the ones
        # the mask rejected -- ~3,000 individual gets for a five-horizon query, a
        # growing share of them thrown away.
        keep = np.flatnonzero(seasons + h <= last)
        y = index.reindex(pd.MultiIndex.from_arrays([ids[keep], seasons[keep] + h])).to_numpy(
            dtype=float
        )
        # Missing key = out of the league that year = a real 0, the same convention the
        # comps forward path uses.
        y = np.nan_to_num(y, nan=0.0)
        # Survival off the RAW line: after flooring, a below-replacement season and a
        # career ending are both 0 and no longer tell apart.
        mask = played(y)
        if replacement:
            y = np.maximum(y - replacement, 0.0)
        x, w = anchor_columns[keep], weights[keep]

        # Gate on the EFFECTIVE size, not the row count. Three rows fit a
        # three-parameter model exactly -- residuals identically zero, median collapsed
        # onto the mean, every bootstrap draw refitting a singular design that lstsq
        # resolves silently to a least-norm solution. But the row count overstates
        # support whenever the kernels taper: a 41-row fit carrying an effective 15 was
        # passing a raw-count floor while producing `on_peak` of -1.03, i.e. more
        # production last year predicting LESS next year.
        n_eff = float(w.sum() ** 2 / np.square(w).sum()) if w.sum() > 0 else 0.0
        if n_eff < MIN_EFFECTIVE_ROWS or w.sum() <= 0:
            path.append(_empty_point(h, age))
            anchors.append(Anchors(h, float("nan"), float("nan"), float("nan"), len(y), n_eff))
            continue

        coefficients = _weighted_least_squares(x, y, w)
        query = np.array([1.0, sgp, peak])
        predicted = float(query @ coefficients)

        # Resampling rows and refitting gives the sampling variability of the fitted
        # MEAN, E[forward | down, peak]. It shrinks as sqrt(n) and contains no residual
        # variance, so it is not how far one player can land from the prediction --
        # `spread` below is. Reporting this alone read as though Soto's age-28 season
        # were pinned to within 0.66 SGP.
        draws = np.empty(bootstrap_draws)
        for i in range(bootstrap_draws):
            pick = rng.integers(0, len(y), len(y))
            draws[i] = query @ _weighted_least_squares(x[pick], y[pick], w[pick])
        se = float(draws.std(ddof=1)) if bootstrap_draws > 1 else float("nan")

        residuals = y - np.column_stack([np.ones(len(x)), x]) @ coefficients
        # WEIGHTED, both of them. The fit centres the weighted residual distribution;
        # an unweighted median or variance is dominated by the far-age / far-peak tail
        # the fit itself barely counted, which for an edge-of-window query is most of
        # the row count.
        median = float(predicted + _weighted_quantile(residuals, w, 0.5))
        # These are FITTED residuals from a three-parameter model, so their weighted
        # mean square estimates (1 - p/n_eff) * sigma^2, not sigma^2. Without the
        # correction the spread -- the number `PathPoint.spread` tells the reader to
        # size a decision by -- runs narrow exactly where support is thinnest: ~12% at
        # n_eff 15, ~32% at n_eff 7. Negligible on a healthy fit (0.4% at n_eff 347),
        # which is the point: it self-corrects toward honest at the dangerous end.
        residual_var = float(np.average(residuals**2, weights=w)) * n_eff / (n_eff - N_PARAMETERS)
        survived = mask
        path.append(
            PathPoint(
                horizon=h,
                age=age + h,
                mean=predicted,
                se=se,
                median=median,
                n=len(y),
                survivors=int(survived.sum()),
                # WEIGHTED, for the same reason the median and residual variance are:
                # the raw rate describes the far tail the fit barely counted. Measured
                # gap on real queries, unweighted -> weighted: +0.2% at age 27 / peak
                # 21.5, +4.5% at age 34 / peak 12.
                mean_if_survived=float(np.average(y[survived], weights=w[survived]))
                if survived.any()
                else float("nan"),
                # Predictive, not the SE of the mean: how far ONE player can land from
                # the prediction.
                spread=float(np.sqrt(residual_var + (0.0 if np.isnan(se) else se**2))),
                n_effective=n_eff,
                survival=float(np.average(survived, weights=w)),
            )
        )
        anchors.append(
            Anchors(
                horizon=h,
                intercept=float(coefficients[0]),
                on_down=float(coefficients[1]),
                on_peak=float(coefficients[2]),
                n_fit=len(y),
                n_effective=n_eff,
            )
        )
        rows.append({"horizon": h, "predicted": predicted})

    return (
        Trajectory(
            kind=kind,
            age=age,
            sgp=sgp,
            band=float("nan"),  # no band: weight decays, nothing is excluded on a cliff
            prior_sgp=peak,
            n_comps=len(history),
            mean_start=float(np.average(history["down"], weights=weights))
            if len(history)
            else float("nan"),
            mean_prior=float(np.average(history["peak"], weights=weights))
            if len(history)
            else float("nan"),
            seasons=(int(history["season"].min()), int(history["season"].max()))
            if len(history)
            else None,
            path=tuple(path),
            comps=pd.DataFrame(rows),
            mode="shape",
        ),
        tuple(anchors),
    )


def _empty_point(horizon: int, age: int) -> PathPoint:
    return PathPoint(
        horizon=horizon,
        age=age + horizon,
        mean=float("nan"),
        se=float("nan"),
        median=float("nan"),
        n=0,
        survivors=0,
        mean_if_survived=float("nan"),
    )
