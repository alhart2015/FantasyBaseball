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
(`prior_sgp`) and what is he doing now (`sgp`)? Both carry signal and their relative weight
is FIT, not chosen -- a weighted least squares of forward SGP on both, per horizon. That
is what converts a comp's future onto the query player's scale, the step level matching
cannot do at all.

The two anchors were once called `peak` and `down`, after the case this was built for --
a star having a down year. They are just LAST SEASON and THIS SEASON, and for a breakout
the old names said the opposite of the truth: a rookie at 0.0 -> 13.6 had his 13.6 stored
in `down`. That mattered, because only ONE anchor is kernel-weighted. Under the old names
"peak is banded, down is not" read as though the high side were the controlled one; what
it actually says is that LAST season is controlled and THIS season is unconstrained, so a
player can be matched entirely on his prior year and then have the fitted line evaluated
at a current season nothing in that cohort resembles. `Trajectory.local_support` measures
exactly that gap, and #310 covers closing it.

**Nothing is excluded on a cliff.** Age contributes on a triangular kernel out to
`AGE_WINDOW` years, and prior level likewise, so a 26-year-old with an identical profile
informs a 27-year-old query instead of being discarded. Weight decays; it never jumps.

The output is a prediction rather than an average over a handful of careers, so
`PathPoint.n` counts FITTING rows and `survivors` describes that sample's attrition, not
a cohort of the query's own comps. `render` labels them accordingly.

Scoring a whole board calls this once per player, and most of what one call does is not
about that player at all (#311). `prepare` hoists the panel-level state -- the history
frame and the forward-value lookup -- out of the query, and `shape_trajectory` accepts
the result in place of a panel:

    prepared = prepare(panel, kind="hitter", horizons=(1, 2, 3))
    for player in board:
        shape_trajectory(prepared, kind="hitter", age=..., sgp=..., prior_sgp=...)

ONE STATE PER POOL. A board is mixed hitters and pitchers, and hoisting a single
`prepare` above the whole loop would fit every pitcher on hitter seasons; `kind` is
carried on the state and a disagreeing query is refused rather than answered.

Passing a panel still works and is the CLI's path. The numbers are the same either way;
`tests/test_trajectory/test_shape.py` asserts that rather than leaving it to inspection.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .model import (
    DEFAULT_BAND,
    DEFAULT_HORIZONS,
    DEFAULT_LOOKBACK,
    PathPoint,
    Trajectory,
    collapse_split_seasons,
    played,
)

#: Age contributes on a triangular kernel this many years either side. Ages 25-30 age
#: similarly enough to pool; beyond that the shape itself changes.
#:
#: MEASURED, not chosen -- `scripts/tune_shape_windows.py`, #310. Leave-one-player-out over
#: the whole panel, every grid point scored on the same complete cases so a narrow kernel
#: cannot win by refusing the hard queries. 2 is the argmin on hitters pooled and on the
#: elite-down-year slice, and 1 through 4 are statistically indistinguishable from it under
#: a bootstrap resampled by player. What is NOT indistinguishable is switching the kernel
#: off: a window wide enough to admit the whole panel costs +2.2% RMSE on hitters pooled
#: and +2.6% on elite-down-year, in every bootstrap draw.
#:
#: The one alternative worth recording as rejected is 3. All five CV folds picked it on the
#: elite slice, at all three horizons -- but the bootstrap puts it at 87%, its interval
#: covers zero, and it REVERSES on the pooled set and on fringe. Fold unanimity was a
#: consistent tiny tilt (0.004 SGP), not an effect, and cross-validated tuning came out
#: BEHIND this default. That is the trap the CV exists to catch, so it is written down.
AGE_WINDOW = 2

#: Kernel half-width on the PRIOR season, in SGP. Wide, because `prior_sgp` is also a
#: regressor and the linear term carries level -- this only keeps the fit LOCAL enough
#: that one straight line does not have to serve fringe and elite players at once.
#:
#: A WINDOW, matching `AGE_WINDOW`, not a "band". Both are kernel half-widths that taper
#: weight to zero; `comps.DEFAULT_BAND` and `comp_trajectory`'s `prior_band` are HARD
#: bands where everything inside counts equally and everything outside counts nothing.
#: Naming shape's kernel `prior_band` would have made it read as the CLI's `--prior-band`,
#: which is the hard one and applies only to the comp matchers.
#:
#: Note what has NO window: the current season. It enters the fit as a bare regressor, so
#: this constrains only half the query. See `Trajectory.local_support` and #310.
#:
#: MEASURED alongside `AGE_WINDOW` (#310), and the two pools do not agree.
#:
#: On HITTERS 8.0 is the argmin, and the kernel is buying something real: widening it to
#: 100 SGP -- effectively off, wider than the observed spread -- costs +1.7% RMSE on the
#: elite slice and +2.7% on elite-down-year, the default winning every bootstrap draw. But
#: the optimum is a PLATEAU, not a point: 4 through 16 are indistinguishable. "8 is right"
#: is really "8 is the centre of a wide flat region", which is the useful form of the claim
#: -- it says a future change of a couple of SGP either way needs no defence, and one to 2
#: or 40 does.
#:
#: On PITCHERS the prior kernel buys nothing measurable. Turning it off scores 5.775 on the
#: elite slice against this default's 5.778, and the argmin drifts out to 12-24. That is
#: the same finding as #313's fitted coefficients from the other direction: `on_prior` is
#: flat at ~0.10-0.19 across the whole pitcher level range and never crosses `on_current`,
#: so last season carries little enough signal that localising on it does not help.
#:
#: The constant stays global at 8.0 anyway. Cross-validated selection loses to it on EVERY
#: pitcher slice (-0.0% to -0.4%) and the folds disagree with each other on all four, so
#: the wider pitcher optimum is not stable enough to ship -- and splitting this per pool is
#: #313's call to make, on #313's evidence, not a side effect of tuning a shared window.
PRIOR_WINDOW = 8.0

#: Kernel half-width used ONLY to reweight residuals when reading the band -- never in
#: the fit. Kernelling the current season in the fit too would shrink a query like the
#: 21-year-old above from ~1,100 effective rows to 14, trading a falsely confident
#: estimate for a uselessly thin one; the point estimate is locally unbiased and should
#: be left alone. This changes what the band is read off, not what is predicted.
#:
#: Matched to `PRIOR_WINDOW` so the two anchors are treated alike here.
CURRENT_WINDOW = 8.0

#: Refits behind `PathPoint.se`. Cost is linear in this and it is ~90% of a query once
#: the panel-level work is hoisted out (#311), so a board sweep should LOWER it -- but
#: per call, not here. Across four real hitter queries x four horizons, over twelve seeds:
#:
#:     draws   ms/query   seed-to-seed SD of `se`   shift in `spread` vs 1000
#:      1000      144            0.0084                  --
#:       250       37            0.0184                  0.0006
#:       100       16            0.0260                  0.0007
#:
#: `spread` is the decision-relevant width and it barely moves, because `se` enters it as
#: `sqrt(residual_var + se^2)` and residual variance dominates by an order of magnitude.
#: But `se` is its own printed column at 2 decimals, and 250 draws roughly DOUBLE its
#: seed-to-seed wobble -- a reader comparing how well two horizons are known would be
#: reading resampling noise. So the default stays where the single-query CLI wants it and
#: a sweep passes `bootstrap_draws=250` explicitly, which is the caller that benefits.
BOOTSTRAP_DRAWS = 1000

#: Ceiling on the refits solved per batch. The batch holds three (batch, n) arrays alive
#: at once -- the drawn indices, the bincount over them, and those counts as float -- so
#: a fixed chunk is ~24 bytes per (draw, fitting row) and grows without bound in n. The
#: per-draw loop it replaced peaked at three (n,)-sized gathers, so a fixed 250 would
#: have turned a few hundred KB into ~120 MB on a loose-kernel query over the full panel.
#: `BOOTSTRAP_BYTES` is the real cap and this is only the upper bound on batch size.
BOOTSTRAP_CHUNK = 250

#: Memory the bootstrap's working arrays may occupy, which sets the batch size at
#: `BOOTSTRAP_BYTES // (24 * n)`. Deliberately a budget rather than a chunk count: the
#: batch is a pure vectorization width, so trading it away on a wide fit costs a little
#: speed and nothing else -- the draws, and therefore the answer, are unchanged.
BOOTSTRAP_BYTES = 32 * 1024 * 1024

#: Intercept plus the two anchors. The residual degrees of freedom subtract this.
N_PARAMETERS = 3

#: Minimum EFFECTIVE size before a horizon is fit at all. Applied to the Kish size
#: rather than the row count, because kernel weights make those diverge badly: a 41-row
#: fit can carry an effective 15. Comfortably above `N_PARAMETERS`, since a fit with
#: barely more support than parameters interpolates its own sample.
MIN_EFFECTIVE_ROWS = 12.0


@dataclass(frozen=True)
class Anchors:
    """The fitted `forward = intercept + a*current + b*prior` for one horizon."""

    horizon: int
    intercept: float
    on_current: float
    on_prior: float
    n_fit: int
    #: Kish effective sample size, `(sum w)^2 / sum(w^2)`. The raw row count overstates
    #: support when most rows carry a small kernel weight.
    n_effective: float


def _weighted_quantiles(
    values: np.ndarray, weights: np.ndarray, qs: tuple[float, ...], order: np.ndarray | None = None
) -> list[float]:
    """The `qs`-quantiles of `values` under `weights`, by cumulative weight.

    Takes all the quantiles at once, and optionally a precomputed `order`, because the
    sort key is the same residual array for the median and both band edges -- sorting it
    three times per horizon was the diff's own doing.
    """
    order = np.argsort(values) if order is None else order
    v, w = values[order], weights[order]
    total = w.sum()
    if total <= 0:
        return [float("nan")] * len(qs)
    cumulative = np.cumsum(w)
    return [float(v[np.searchsorted(cumulative, q * total)]) for q in qs]


def _weighted_quantile(values: np.ndarray, weights: np.ndarray, q: float) -> float:
    """One quantile. Kept for callers that want a single number."""
    return _weighted_quantiles(values, weights, (q,))[0]


def _kish(weights: np.ndarray) -> float:
    """Kish effective sample size, `(sum w)^2 / sum(w^2)`.

    The raw row count overstates support whenever kernel weights taper -- a 41-row fit
    can carry an effective 15. Written once because it is compared against
    `MIN_EFFECTIVE_ROWS` in two places, and two spellings of one definition is how the
    two gates drift apart.
    """
    total = weights.sum()
    return float(total**2 / np.square(weights).sum()) if total > 0 else 0.0


def _triangular(distance: np.ndarray, width: float) -> np.ndarray:
    """1.0 at zero distance, tapering linearly to 0 at `width`, never negative."""
    clipped: np.ndarray = np.clip(1.0 - np.abs(distance) / width, 0.0, None)
    return clipped


def _weighted_least_squares(x: np.ndarray, y: np.ndarray, w: np.ndarray) -> np.ndarray:
    """Solve for [intercept, a, b] with weights `w`. `x` is (n, 2): current, prior."""
    design = np.column_stack([np.ones(len(x)), x])
    root = np.sqrt(w)[:, None]
    coefficients: np.ndarray
    coefficients, *_ = np.linalg.lstsq(design * root, y * np.sqrt(w), rcond=None)
    return coefficients


def collapsed_index(panel: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """The panel with split seasons collapsed, and its `(mlbam_id, season) -> sgp` lookup.

    One function because the two must agree. `build_history` collapses to build the
    anchors and `prepare` needs the same lookup for the forward values, and deriving them
    separately spelled the split-season rule out twice -- one via `collapse_split_seasons`
    and one via a `groupby(level=[0, 1]).sum()` over the raw panel. `collapse_split_seasons`
    was made shared precisely so the anchor side and the forward-lookup side cannot
    disagree; if the collapse ever stops being a plain sum (PA-weighting a rate, say), the
    two spellings would diverge and `x` and `y` would describe differently-collapsed
    seasons with nothing failing.
    """
    collapsed = collapse_split_seasons(panel).sort_values(["mlbam_id", "season"])
    return collapsed, collapsed.set_index(["mlbam_id", "season"])["sgp"]


def build_history(panel: pd.DataFrame) -> pd.DataFrame:
    """One row per season that has an OBSERVABLE prior, carrying both anchors.

    A season whose predecessor falls before the panel begins is dropped rather than
    given a prior of 0 -- the same censoring the forward path uses, for the same reason:
    "we cannot see it" must never be scored as "he did not play". A player who was
    genuinely out of the league keeps his 0, which is a real observation.

    A split season (a mid-season trade, two rows for one player-year) is collapsed on
    BOTH sides, as `comps.comp_trajectory` does. Collapsing only the lookup would sum his
    future correctly while still entering him into the fit TWICE as two half-seasons,
    each with a half-size `current` against a full-size `prior` and a full-size forward
    value -- attenuating the fitted `on_current`, double-weighting him, and dragging
    `mean_start` low.
    """
    panel, index = collapsed_index(panel)
    first = int(panel["season"].min())

    history = panel.copy()
    seasons = history["season"].to_numpy()
    # One vectorized reindex instead of a per-row MultiIndex .get. A missing key means
    # he was out of the league that year (a real 0); a key BEFORE the panel starts is
    # unobservable and stays NaN so `dropna` censors it.
    prior = index.reindex(
        pd.MultiIndex.from_arrays([history["mlbam_id"].to_numpy(), seasons - 1])
    ).to_numpy(dtype=float)
    history["prior"] = np.where(seasons - 1 >= first, np.nan_to_num(prior, nan=0.0), np.nan)
    return history.dropna(subset=["prior"]).rename(columns={"sgp": "current"})


#: `eq=False` because the fields are ndarrays. The generated `__eq__` tuple-compares
#: them, so `prepare(p) == prepare(p)` raises "truth value of an array is ambiguous",
#: and `frozen=True` then derives a `__hash__` that raises "unhashable type: ndarray".
#: Both fire on the natural way to use this class -- `player_trajectory.py` already
#: memoizes its panel-side helpers with `@lru_cache`, and an `lru_cache`d scoring helper
#: taking a prepared state would die before scoring a player. Identity semantics are what
#: a cache key wants here anyway: two states off the same panel ARE interchangeable, and
#: comparing 16k-row arrays to discover that is not a comparison anyone wants to pay for.
@dataclass(frozen=True, eq=False)
class Prepared:
    """Panel-level state for shape matching, computed once and reused across queries.

    Everything here depends on the panel and the horizons and NOT on the player being
    asked about, so recomputing it per query is pure waste -- it was ~85% of a board
    sweep (#311). Only the kernel weights and the weighted least squares are genuinely
    per-query, and those are the cheap parts.

    Columns are held as arrays rather than a frame because the per-query step is a
    boolean subset, and slicing a DataFrame 600 times costs more than the fit does.
    """

    #: Which pool the panel was loaded from. Held because `kind` is otherwise a pure
    #: label -- it lands on `Trajectory.kind` for `render` and is never checked against
    #: the panel. That was safe while every caller loaded the panel and named the pool in
    #: one expression, but hoisting the panel out of the loop is exactly what this class
    #: is for, and a board is mixed hitters and pitchers. One `prepare` above that loop
    #: would fit every pitcher on hitter seasons -- different SGP scale, different aging
    #: shape, different survival -- and print it under `kind='pitcher'` with a plausible
    #: `n_comps` and no warning.
    kind: str
    #: Horizons whose forward values are available. A query may ask for any subset.
    horizons: tuple[int, ...]
    #: Last season with an observable outcome; a row is censored past it.
    last: int
    age: np.ndarray
    current: np.ndarray
    prior: np.ndarray
    season: np.ndarray
    #: MLBAM id per history row, aligned to every other array here. `prepare` already
    #: builds this to reindex `forward` and then dropped it, so anything wanting to NAME
    #: a matched row had to rebuild `build_history` alongside and trust that two row
    #: orders agreed. Carrying it makes that class of silent misalignment unreachable.
    mlbam_id: np.ndarray
    #: horizon -> forward SGP for every history row, 0 where he was out of the league.
    #: Rows whose `season + horizon` runs past `last` are unobservable and are masked
    #: off per query rather than being trusted here.
    forward: dict[int, np.ndarray]
    #: offset -> PRIOR SGP for every history row, `offset` seasons back, **NaN** where he
    #: has no season there. `back[0]` is the row's own season and equals `current`.
    #:
    #: THE OPPOSITE CONVENTION TO `forward`, DELIBERATELY, and the asymmetry is the point
    #: rather than an oversight (#358). The two answer different questions:
    #:
    #: * `forward` asks WHAT HAPPENED TO HIM. A player out of the league at 33 was worth
    #:   nothing to a roster slot that year, so 0 is the honest outcome and filling it is
    #:   what keeps a comp set from being all survivors.
    #: * `back` asks WHAT DID HE LOOK LIKE. A year he did not play says nothing about the
    #:   kind of player he was, and scoring it as a 0 makes an injured star and a
    #:   replacement-level journeyman look alike -- which is exactly the match
    #:   `career_comps` exists to stop making. It also cannot be told apart from a season
    #:   that falls before the panel begins, which is unobservable rather than absent.
    #:
    #: So NaN here means "no comparison available at this age", and `career_comps` drops
    #: it from the error and counts the overlap instead. Anything else reading this must
    #: decide what a NaN means for its own question rather than `nan_to_num`-ing it.
    back: dict[int, np.ndarray]


def prepare(
    panel: pd.DataFrame,
    *,
    kind: str,
    horizons: tuple[int, ...] = DEFAULT_HORIZONS,
    lookback: int = DEFAULT_LOOKBACK,
    last_complete_season: int | None = None,
) -> Prepared:
    """Hoist the panel-level half of `shape_trajectory` out of the per-query loop.

    Worth it from roughly the second query onward; below that just pass the panel.

    `kind` names the pool `panel` was loaded from, and every query against the result
    must agree with it -- see `Prepared.kind`.

    `lookback` sizes `back`, the realized-career window `career_comps` matches on. It
    is built HERE rather than in that module because it is panel-level state exactly
    like `forward` -- one vectorized reindex per offset over the whole history, against
    one per query -- and because building it beside `forward` is what makes the two
    conventions (0 versus NaN) sit next to each other where a reader meets both.
    """
    if not horizons:
        raise ValueError("horizons must not be empty")
    if min(horizons) < 1:
        raise ValueError(f"horizons must be at least 1, got {sorted(horizons)}")
    if lookback < 1:
        raise ValueError(f"lookback must be at least 1, got {lookback}")

    last = last_complete_season if last_complete_season is not None else int(panel["season"].max())
    collapsed, index = collapsed_index(panel)
    history = build_history(collapsed)

    ids = history["mlbam_id"].to_numpy()
    seasons = history["season"].to_numpy()
    # One vectorized reindex per horizon over the whole history, instead of one per
    # query. A missing key means he was out of the league that year -- a real 0, the
    # same convention the comps forward path uses.
    forward = {
        h: np.nan_to_num(
            index.reindex(pd.MultiIndex.from_arrays([ids, seasons + h])).to_numpy(dtype=float),
            nan=0.0,
        )
        for h in sorted(set(horizons))
    }
    # The same reindex run backwards, WITHOUT `nan_to_num` -- see `Prepared.back`. The
    # NaN is load-bearing there, so this must not be folded into the loop above.
    back = {
        k: index.reindex(pd.MultiIndex.from_arrays([ids, seasons - k])).to_numpy(dtype=float)
        for k in range(lookback)
    }
    return Prepared(
        kind=kind,
        horizons=tuple(sorted(set(horizons))),
        last=last,
        age=history["age"].to_numpy(dtype=float),
        current=history["current"].to_numpy(dtype=float),
        prior=history["prior"].to_numpy(dtype=float),
        season=seasons,
        mlbam_id=ids,
        forward=forward,
        back=back,
    )


def _bootstrap_predictions(
    x: np.ndarray,
    y: np.ndarray,
    w: np.ndarray,
    query: np.ndarray,
    rng: np.random.Generator,
    draws: int,
) -> np.ndarray:
    """Query predictions from `draws` refits on rows resampled with replacement.

    Solved in BATCHES through the normal equations rather than one `lstsq` per draw. A
    resampled fit differs from the original only in how many times each row appears, so
    every draw's `X'WX` and `X'Wy` is a weighted sum of the same per-row contributions --
    which makes a batch two matrix products and a stack of 3x3 pseudo-inverses instead
    of a thousand full-size SVDs. It was ~46% of a query's cost.

    The resampled indices are drawn in the same order the per-draw loop drew them, so
    the draws themselves are unchanged; only the solver differs, and the two agree far
    inside the resolution `se` is reported at.

    `pinv` rather than `solve`, deliberately. A draw can land on too few distinct rows
    to identify three parameters, and `solve` raises on that while the `lstsq` this
    replaced returned the least-norm solution. Those two are the same answer here --
    `pinv(X'WX) X'Wy` equals `pinv(X'W^.5) W^.5 y` for any rank -- so the rank-deficient
    draw keeps behaving exactly as it did, with no separate branch to go stale. The 3x3
    SVDs cost about 4% of this routine.
    """
    n = len(y)
    design = np.column_stack([np.ones(n), x])
    rooted = design * np.sqrt(w)[:, None]
    response = y * np.sqrt(w)
    # Per-row contributions to the normal equations, so a draw is a weighted SUM of
    # these rather than another pass over its own resampled rows.
    gram = np.einsum("ni,nj->nij", rooted, rooted).reshape(n, -1)
    moment = rooted * response[:, None]

    # Three (batch, n) arrays are alive at the peak, at 8 bytes each. Narrowing the batch
    # on a wide fit costs vectorization width and nothing else -- the draws are drawn in
    # the same order at any batch size, so the result does not depend on this.
    chunk = max(1, min(BOOTSTRAP_CHUNK, BOOTSTRAP_BYTES // (24 * n)))
    out = np.empty(draws)
    for start in range(0, draws, chunk):
        size = min(chunk, draws - start)
        picks = rng.integers(0, n, (size, n))
        # How many times each row was drawn, as one offset bincount rather than `size`
        # separate ones. The offset is applied IN PLACE: `picks` is freshly drawn and
        # discarded here, and a second (size, n) allocation per chunk was costing more
        # than either matrix product it feeds.
        picks += (np.arange(size) * n)[:, None]
        counts = np.bincount(picks.ravel(), minlength=size * n).reshape(size, n).astype(float)
        width = design.shape[1]
        normal = (counts @ gram).reshape(size, width, width)
        out[start : start + size] = (np.linalg.pinv(normal) @ (counts @ moment)[..., None])[
            ..., 0
        ] @ query
    return out


def shape_trajectory(
    panel: pd.DataFrame | Prepared,
    *,
    kind: str,
    age: int,
    sgp: float,
    prior_sgp: float,
    horizons: tuple[int, ...] = DEFAULT_HORIZONS,
    age_window: int = AGE_WINDOW,
    prior_window: float = PRIOR_WINDOW,
    last_complete_season: int | None = None,
    replacement: float = 0.0,
    slot: str | None = None,
    seed: int = 0,
    bootstrap_draws: int = BOOTSTRAP_DRAWS,
) -> tuple[Trajectory, tuple[Anchors, ...]]:
    """Forward path for a player at `age`, now producing `sgp`, who produced `prior_sgp` last season.

    `panel` may be a season panel or a `Prepared` from `prepare`. The second form skips
    the panel-level work and is what a board sweep should pass; the answer is identical.

    Returns the trajectory and the fitted anchors, so the coefficients that produced
    each number can be read rather than trusted.

    `replacement` fits the model on VALUE ABOVE REPLACEMENT instead of raw SGP: the
    response becomes `forward - replacement` while the two anchors stay on the SGP scale
    the caller supplies them in. Shifting the RESPONSE rather than the fitted mean is
    what carries the scale into `spread`, `median` and `mean_if_survived` -- shifting
    afterwards moved only the mean and left the other three on a different scale.

    NOTHING IS CLAMPED, and that is load-bearing rather than incidental (#331). The
    response used to be `max(forward - replacement, 0)` and the prediction, median and
    band edges were each floored at zero on top of it. Every one of those is a NONLINEAR
    transform of an otherwise affine relationship, and the response clamp is the
    dangerous one: it flattens exactly the comps that fall below the floor, which changes
    the fitted SLOPE on `(current, prior)` -- so two players sharing a slot, and therefore
    sharing a floor, came out in a DIFFERENT ORDER on VAR than on SGP. The deviation grew
    with the floor and differed per query (up to 1.63 SGP between two same-floor players
    at a floor of 18), which is why pitchers -- high floors against a narrow spread of
    outcomes -- reshuffled far worse than hitters.

    Unclamped, `y_var = y_sgp - replacement` is affine and the intercept column is exactly
    the vector being subtracted, so the fit moves ONLY its intercept, by `-replacement`.
    Every prediction shifts by the floor and no residual moves at all: `mean`, `median`,
    `p10` and `p90` are the raw fit's minus the floor, `se` and `spread` are identical,
    and same-floor ordering is preserved BY CONSTRUCTION rather than by a test.

    The cost, accepted deliberately (issue #331): a comp who left the league scores a
    structural 0 and now enters the fit at `-replacement` rather than 0, so the model no
    longer prices the option to drop him and start the replacement he was measured
    against. Out of the league is 0 SGP, so his VAR is minus the floor; a reader who wants
    the drop-adjusted number takes `max(var, 0)` at the point of decision. Both matchers
    were changed together -- `comps.comp_trajectory` carries the same reversal -- so the
    two cannot disagree about what a VAR means.
    """
    if prior_window <= 0:
        raise ValueError(f"prior_window must be positive, got {prior_window}")
    if age_window < 1:
        raise ValueError(f"age_window must be at least 1, got {age_window}")
    if not horizons:
        raise ValueError("horizons must not be empty")
    if bootstrap_draws < 2:
        # `std(ddof=1)` on fewer than two draws is NaN plus a RuntimeWarning, which
        # reaches the caller as an SE that is silently missing rather than as a refused
        # argument. `comps._bootstrap_se` does NOT cover this -- it guards the comp
        # count -- so `comp_trajectory` carries the same check, deliberately in both.
        raise ValueError(f"bootstrap_draws must be at least 2, got {bootstrap_draws}")

    # DEDUPED, matching `prepare`. Sorting alone let `horizons=(1, 1, 2)` fit h1 twice,
    # append two identical `PathPoint`s and `Anchors`, run the bootstrap twice, and
    # double-count h1 in `Trajectory.total`.
    horizons = tuple(sorted(set(horizons)))
    if isinstance(panel, Prepared):
        prepared = panel
        if prepared.kind != kind:
            raise ValueError(
                f"prepared state was built from the {prepared.kind} panel and cannot "
                f"answer a {kind} query; prepare one state per pool"
            )
        unavailable = sorted(set(horizons) - set(prepared.horizons))
        if unavailable:
            raise ValueError(
                f"prepared state has no forward values for horizons {unavailable}; "
                f"it was built for {list(prepared.horizons)}"
            )
        # A LOWER cutoff is fine and does not need a rebuild: `prepare` never uses `last`
        # for anything, it just carries it -- `forward` is built for every history row and
        # all censoring happens per query below. Refusing any difference forced an
        # as-of-season sweep to re-run `build_history` and a full reindex per cutoff,
        # which is the exact work `prepare` exists to hoist.
        #
        # A HIGHER one is unsafe and stays refused. `forward` was looked up against the
        # panel, so a season past `prepared.last` came back missing and was recorded as
        # the 0 that means "out of the league" -- raising the cutoff would silently
        # reinterpret "has not happened yet" as "did not play".
        if last_complete_season is not None and last_complete_season > prepared.last:
            raise ValueError(
                f"last_complete_season {last_complete_season} runs past the prepared "
                f"state's {prepared.last}, where a missing season is not yet played "
                f"rather than not played; rebuild the state to reach further"
            )
    else:
        prepared = prepare(
            panel, kind=kind, horizons=horizons, last_complete_season=last_complete_season
        )
    # The query's cutoff wins where it is given -- it can only ever censor MORE than the
    # prepared state does, the guard above having refused the other direction.
    last = last_complete_season if last_complete_season is not None else prepared.last

    # Weight once: the kernels describe the QUERY's neighbourhood and do not move with
    # the horizon. `age_window + 1` so a player exactly `age_window` years away still
    # carries weight rather than sitting exactly on zero.
    weights = _triangular(prepared.age - age, age_window + 1) * _triangular(
        prepared.prior - prior_sgp, prior_window
    )
    # The NEAREST horizon sets membership, matching `comp_trajectory`: a season too
    # recent to have even one observable forward year enters no fit, so counting it in
    # `n_comps` (which render prints as "fit on N weighted seasons") would describe the
    # fit with rows the fit never saw.
    usable = np.flatnonzero((weights > 0) & (prepared.season + horizons[0] <= last))
    seasons = prepared.season[usable]
    current, prior = prepared.current[usable], prepared.prior[usable]
    weights = weights[usable]
    # Weight sitting within a comp band of the QUERY's own current season. The fit has no
    # kernel on the current season, so this is the only thing that says whether the line
    # was evaluated inside its own support or extrapolated past it. Every surviving weight
    # is strictly positive by the `weights > 0` filter above, so `len(usable)` is the only
    # guard needed.
    local_support = (
        float(weights[np.abs(current - sgp) <= DEFAULT_BAND].sum() / weights.sum())
        if len(usable)
        else float("nan")
    )

    rng = np.random.default_rng(seed)
    anchor_columns = np.column_stack([current, prior])
    path, anchors, rows = [], [], []
    for h in horizons:
        # Positions into the PREPARED arrays, so the forward values are a gather rather
        # than a fresh lookup. Observability is still decided here and not in `prepare`:
        # a row too recent to have an outcome at this horizon may well have one at a
        # nearer horizon in the same query.
        keep = np.flatnonzero(seasons + h <= last)
        y = prepared.forward[h][usable[keep]]
        # Survival off the RAW line. Shifted, a career ending reads `-replacement`, which
        # is also what a season spent exactly at the floor's distance below zero reads --
        # and only the raw line has the exact 0 that means "not in the league".
        mask = played(y)
        # SHIFTED, NOT FLOORED. `np.maximum(y - replacement, 0)` here is what reordered
        # same-slot players across the VAR/SGP toggle (#331); see the docstring.
        y = y - replacement
        x, w = anchor_columns[keep], weights[keep]

        # Gate on the EFFECTIVE size, not the row count. Three rows fit a
        # three-parameter model exactly -- residuals identically zero, median collapsed
        # onto the mean, every bootstrap draw refitting a singular design that lstsq
        # resolves silently to a least-norm solution. But the row count overstates
        # support whenever the kernels taper: a 41-row fit carrying an effective 15 was
        # passing a raw-count floor while producing `on_prior` of -1.03, i.e. more
        # production last year predicting LESS next year.
        n_eff = _kish(w)
        if n_eff < MIN_EFFECTIVE_ROWS or w.sum() <= 0:
            path.append(_empty_point(h, age))
            anchors.append(Anchors(h, float("nan"), float("nan"), float("nan"), len(y), n_eff))
            continue

        coefficients = _weighted_least_squares(x, y, w)
        query = np.array([1.0, sgp, prior_sgp])
        predicted = float(query @ coefficients)

        # Resampling rows and refitting gives the sampling variability of the fitted
        # MEAN, E[forward | current, prior]. It shrinks as sqrt(n) and contains no residual
        # variance, so it is not how far one player can land from the prediction --
        # `spread` below is. Reporting this alone read as though Soto's age-28 season
        # were pinned to within 0.66 SGP.
        se = float(_bootstrap_predictions(x, y, w, query, rng, bootstrap_draws).std(ddof=1))

        residuals = y - np.column_stack([np.ones(len(x)), x]) @ coefficients
        # WEIGHTED, both of them. The fit centres the weighted residual distribution;
        # an unweighted median or variance is dominated by the far-age / far-prior tail
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

        # The BAND. Two things distinguish it from `predicted +/- k*spread`.
        #
        # It is EMPIRICAL, because the errors are not Gaussian and not in the same way
        # across pools and horizons -- flatter and right-skewed for pitchers at +3,
        # near-normal and left-skewed for hitters at +1. Quantiles carry that shape.
        #
        # And it is read off residuals reweighted toward the query's OWN current season,
        # not the whole fitting cohort's. Only the prior season is kernel-weighted in the
        # fit, so a player whose current season outruns his prior is matched to a cohort
        # he sits outside: a 21-year-old at 13.6 now / 0.0 prior drew a population whose
        # current seasons average 2.9, and inherited that population's tight scatter.
        # The point estimate survives this -- measured locally it is unbiased, -0.61
        # against a prediction of 13.5 -- but the band did not, reporting a NARROW
        # interval precisely where the model knew least. Reweighting widens it ~20-35% on
        # those queries and leaves a well-supported one alone, which is the whole ask.
        band_weights = w * _triangular(current[keep] - sgp, CURRENT_WINDOW)
        band_effective = _kish(band_weights)
        # Too few comps near his current season to describe a distribution, so fall back
        # to the cohort's own residuals rather than quantile 20 effective rows.
        #
        # This DISENGAGES the fix on the deepest extrapolations -- exactly the queries it
        # exists to protect -- restoring the understated cohort-scatter band. That has to
        # be visible, not inferred: the board tells the reader the band is the trustworthy
        # part, and a reverted row printed identically to a widened one. `local_support`
        # does not cover it either; it measures a different width (`DEFAULT_BAND`, hard)
        # than this gate (`CURRENT_WINDOW`, tapered), so the two disagree on which rows
        # are affected. Recorded per horizon and surfaced on the trajectory.
        fell_back = band_effective < MIN_EFFECTIVE_ROWS
        if fell_back:
            band_weights = w
        # The fitted mean carries its own uncertainty, and it GROWS with leverage -- the
        # query being far from the data is exactly when the line is least pinned down.
        # `spread` picks this up as `se^2`; quantiles have no variance to add it to, so it
        # enters as the same scale factor, sqrt(1 + se^2/sigma^2). Small in practice (~4%
        # here) but it is the term that belongs to extrapolation, so it is not dropped.
        inflate = (
            float(np.sqrt(1.0 + se**2 / residual_var))
            if residual_var > 0 and not np.isnan(se)
            else 1.0
        )
        low, high = _weighted_quantiles(residuals, band_weights, (0.10, 0.90))
        p10 = float(predicted + inflate * low)
        p90 = float(predicted + inflate * high)
        # WIDEN to contain the estimate; never relocate the band to do it. The fit centres
        # the `w`-weighted residuals, so their quantiles straddle zero -- but the band is
        # read off `band_weights`, whose distribution carries its own offset, and where
        # that offset is large enough BOTH quantiles land on one side of `predicted`. The
        # interval then excludes the very number it is drawn around: 135 of 4020
        # horizon-rows on the live board did exactly that (Arenado age-37, mean 0.43,
        # band 0.00..0.40), while the board footer told the reader to trust the band over
        # the estimate.
        #
        # Re-centring on the reweighted median also fixes it, and was tried -- but the
        # offset is real information, not noise, and dropping it cost coverage where the
        # model is weakest: elite pitchers at +3 went from 14% to 22% below p10 against a
        # nominal 10%. Clamping keeps the offset, so calibration is untouched wherever the
        # band already contained the estimate, and can only ever WIDEN elsewhere -- the
        # conservative direction for a keep-or-cut call.
        p10, p90 = min(p10, predicted), max(p90, predicted)
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
                # gap on real queries, unweighted -> weighted: +0.2% at age 27 / prior
                # 21.5, +4.5% at age 34 / prior 12.
                mean_if_survived=float(np.average(y[survived], weights=w[survived]))
                if survived.any()
                else float("nan"),
                # Predictive, not the SE of the mean: how far ONE player can land from
                # the prediction.
                spread=float(np.sqrt(residual_var + (0.0 if np.isnan(se) else se**2))),
                p10=p10,
                p90=p90,
                n_effective=n_eff,
                survival=float(np.average(survived, weights=w)),
                band_fell_back=fell_back,
            )
        )
        anchors.append(
            Anchors(
                horizon=h,
                intercept=float(coefficients[0]),
                on_current=float(coefficients[1]),
                on_prior=float(coefficients[2]),
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
            prior_sgp=prior_sgp,
            n_comps=len(usable),
            local_support=local_support,
            # The trajectory-level flag IS the OR over the path -- comps.py documents
            # it that way and test_shape.py asserts it literally. Derived rather than
            # accumulated, so a later edit inside the fallback branch cannot set one
            # without the other and leave the trajectory claiming a revert that no
            # PathPoint reports (or the reverse).
            band_fell_back=any(p.band_fell_back for p in path),
            mean_start=float(np.average(current, weights=weights)) if len(usable) else float("nan"),
            mean_prior=float(np.average(prior, weights=weights)) if len(usable) else float("nan"),
            seasons=(int(seasons.min()), int(seasons.max())) if len(usable) else None,
            path=tuple(path),
            comps=pd.DataFrame(rows),
            mode="shape",
            floor=replacement,
            slot=slot,
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
