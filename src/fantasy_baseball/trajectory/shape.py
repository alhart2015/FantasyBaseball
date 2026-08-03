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

Scoring a whole board calls this once per player, and most of what one call does is not
about that player at all (#311). `prepare` hoists the panel-level state -- the history
frame and the forward-value lookup -- out of the query, and `shape_trajectory` accepts
the result in place of a panel:

    prepared = prepare(panel, horizons=(1, 2, 3))
    for player in board:
        shape_trajectory(prepared, kind=..., age=..., sgp=..., peak=...)

Passing a panel still works and is the CLI's path. The numbers are the same either way;
`tests/test_trajectory/test_shape.py` asserts that rather than leaving it to inspection.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .comps import (
    DEFAULT_HORIZONS,
    PathPoint,
    Trajectory,
    collapse_split_seasons,
    played,
)

#: Age contributes on a triangular kernel this many years either side. Ages 25-30 age
#: similarly enough to pool; beyond that the shape itself changes.
AGE_WINDOW = 2

#: Peak-level kernel half-width, in SGP. Wide, because `peak` is also a regressor and
#: the linear term carries level -- this only keeps the fit LOCAL enough that one
#: straight line does not have to serve fringe and elite players at once.
PEAK_BAND = 8.0

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
    each with a half-size `down` against a full-size `peak` and a full-size forward
    value -- attenuating the fitted `on_down`, double-weighting him, and dragging
    `mean_start` low.
    """
    panel, index = collapsed_index(panel)
    first = int(panel["season"].min())

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
    down: np.ndarray
    peak: np.ndarray
    season: np.ndarray
    #: horizon -> forward SGP for every history row, 0 where he was out of the league.
    #: Rows whose `season + horizon` runs past `last` are unobservable and are masked
    #: off per query rather than being trusted here.
    forward: dict[int, np.ndarray]


def prepare(
    panel: pd.DataFrame,
    *,
    kind: str,
    horizons: tuple[int, ...] = DEFAULT_HORIZONS,
    last_complete_season: int | None = None,
) -> Prepared:
    """Hoist the panel-level half of `shape_trajectory` out of the per-query loop.

    Worth it from roughly the second query onward; below that just pass the panel.

    `kind` names the pool `panel` was loaded from, and every query against the result
    must agree with it -- see `Prepared.kind`.
    """
    if not horizons:
        raise ValueError("horizons must not be empty")
    if min(horizons) < 1:
        raise ValueError(f"horizons must be at least 1, got {sorted(horizons)}")

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
    return Prepared(
        kind=kind,
        horizons=tuple(sorted(set(horizons))),
        last=last,
        age=history["age"].to_numpy(dtype=float),
        down=history["down"].to_numpy(dtype=float),
        peak=history["peak"].to_numpy(dtype=float),
        season=seasons,
        forward=forward,
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
    peak: float,
    horizons: tuple[int, ...] = DEFAULT_HORIZONS,
    age_window: int = AGE_WINDOW,
    peak_band: float = PEAK_BAND,
    last_complete_season: int | None = None,
    replacement: float = 0.0,
    slot: str | None = None,
    seed: int = 0,
    bootstrap_draws: int = BOOTSTRAP_DRAWS,
) -> tuple[Trajectory, tuple[Anchors, ...]]:
    """Forward path for a player at `age`, now producing `sgp`, who produced `peak`.

    `panel` may be a season panel or a `Prepared` from `prepare`. The second form skips
    the panel-level work and is what a board sweep should pass; the answer is identical.

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
        if last_complete_season is not None and last_complete_season != prepared.last:
            raise ValueError(
                f"last_complete_season {last_complete_season} contradicts the prepared "
                f"state's {prepared.last}; rebuild it rather than censoring twice"
            )
    else:
        prepared = prepare(
            panel, kind=kind, horizons=horizons, last_complete_season=last_complete_season
        )
    last = prepared.last

    # Weight once: the kernels describe the QUERY's neighbourhood and do not move with
    # the horizon. `age_window + 1` so a player exactly `age_window` years away still
    # carries weight rather than sitting exactly on zero.
    weights = _triangular(prepared.age - age, age_window + 1) * _triangular(
        prepared.peak - peak, peak_band
    )
    # The NEAREST horizon sets membership, matching `comp_trajectory`: a season too
    # recent to have even one observable forward year enters no fit, so counting it in
    # `n_comps` (which render prints as "fit on N weighted seasons") would describe the
    # fit with rows the fit never saw.
    usable = np.flatnonzero((weights > 0) & (prepared.season + horizons[0] <= last))
    seasons = prepared.season[usable]
    down, high = prepared.down[usable], prepared.peak[usable]
    weights = weights[usable]

    rng = np.random.default_rng(seed)
    anchor_columns = np.column_stack([down, high])
    path, anchors, rows = [], [], []
    for h in horizons:
        # Positions into the PREPARED arrays, so the forward values are a gather rather
        # than a fresh lookup. Observability is still decided here and not in `prepare`:
        # a row too recent to have an outcome at this horizon may well have one at a
        # nearer horizon in the same query.
        keep = np.flatnonzero(seasons + h <= last)
        y = prepared.forward[h][usable[keep]]
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
        # Flooring the RESPONSE is not enough here. `comp_trajectory` averages values
        # that are already >= 0, so its mean cannot go negative; this is an unconstrained
        # WLS extrapolation, and on a collapsed veteran it printed -2.35 -- a keeper
        # apparently COSTING value, when a below-replacement player costs exactly zero
        # (you drop him and start the replacement). The two matchers disagreed in sign on
        # the same player, and this is the one that runs by default.
        if replacement:
            predicted = max(predicted, 0.0)

        # Resampling rows and refitting gives the sampling variability of the fitted
        # MEAN, E[forward | down, peak]. It shrinks as sqrt(n) and contains no residual
        # variance, so it is not how far one player can land from the prediction --
        # `spread` below is. Reporting this alone read as though Soto's age-28 season
        # were pinned to within 0.66 SGP.
        se = float(_bootstrap_predictions(x, y, w, query, rng, bootstrap_draws).std(ddof=1))

        residuals = y - np.column_stack([np.ones(len(x)), x]) @ coefficients
        # WEIGHTED, both of them. The fit centres the weighted residual distribution;
        # an unweighted median or variance is dominated by the far-age / far-peak tail
        # the fit itself barely counted, which for an edge-of-window query is most of
        # the row count.
        median = float(predicted + _weighted_quantile(residuals, w, 0.5))
        if replacement:
            median = max(median, 0.0)
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
            n_comps=len(usable),
            mean_start=float(np.average(down, weights=weights)) if len(usable) else float("nan"),
            mean_prior=float(np.average(high, weights=weights)) if len(usable) else float("nan"),
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
