"""Match historical comparables on (age, SGP) and average their forward paths.

The estimator is deliberately the plainest thing that answers the question: take
every player-season within `band` SGP of the query at the same age, follow each one
forward, and average. Two choices in it are load-bearing.

**A player who leaves the league scores 0, not "excluded".** He is worth zero to a
roster slot, so the decision-relevant average is unconditional. Measured on the
shipped defaults via `player_trajectory.py --pool hitter --age 25 --sgp 13`, the
age-30 mean is **9.03 among the 110 still playing and 8.07 over all 123** -- scoring
only survivors would overstate it by 12%, and the gap widens with the horizon as
attrition compounds (89% are still playing at age 30, 100% at 26).

**A season that has not been played yet is dropped, not zero-filled, and the test is
applied PER HORIZON.** A 2024 age-25 season has a real age-26 to look at but no age-30,
so it counts toward h1 and is absent from h5. Zero-filling it would score "has not
happened" as "career over" and drag the long horizons toward zero; judging it by the
LONGEST horizon instead -- as this did until the per-horizon fix -- threw away 17% of
usable h1 comps and cut the cohort off five years before the present. This censoring is
why n falls as the horizon grows and is the real limit on how far out the method sees.

What this returns is `E[future SGP | age, current SGP]`. The year-one drop is mostly
regression to the mean -- a 13-SGP season is partly luck -- not aging. Do not read the
path as an aging curve, and do not stack another regression step on top of it.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

DEFAULT_BAND = 2.5

#: Fitting weight that must sit near the query's own current season before its numbers are
#: ranked rather than flagged. Beside `Trajectory.local_support`, which it qualifies, so
#: the board and the single-player CLI cannot disagree about what "unsupported" means.
#:
#: 10% separates the two failure modes cleanly on the real 2026 board -- measured, not
#: chosen. Sal Stewart (16.8 now / 1.5 prior) 1.8%, Kevin McGonigle (13.6 / 0.0) 4.6% and
#: CJ Abrams (20.9 / 14.0) 6.3% are extrapolations; Crow-Armstrong (20.2 / 17.2) 16.1%,
#: Juan Soto (12.9 / 21.5) 33.1% and Bobby Witt Jr. (15.5 / 18.9) 40.1% are supported.
#: The obvious gauge, `sgp - mean_start`, cannot make that split -- it reads "above this
#: cohort's mean" and "outside this cohort" the same way, and flagged 11 of the top 20.
MIN_LOCAL_SUPPORT = 0.10
DEFAULT_HORIZONS = (1, 2, 3, 4, 5)
BOOTSTRAP_DRAWS = 2000


#: A forward season is "he did not play" only when it is EXACTLY zero. SGP is genuinely
#: negative for a below-replacement season -- 7.7% of hitter-seasons and 15.0% of
#: pitcher-seasons in the panel, down to -3.87 -- so testing `> 0` would file every one
#: of those as a career ending, understating survival and inflating the survivor mean.
#: Both estimators import this rather than each spelling the test out.
def played(values: np.ndarray) -> np.ndarray:
    """Mask of forward seasons the player actually played."""
    mask: np.ndarray = values != 0.0
    return mask


@dataclass(frozen=True)
class PathPoint:
    """One horizon of a trajectory."""

    horizon: int
    age: int
    mean: float
    #: Standard error of the MEAN -- how well the central estimate is known. It shrinks
    #: as sqrt(n) and says nothing about how much one player varies. For a keeper
    #: decision read `spread`, not this.
    se: float
    median: float
    n: int
    survivors: int
    mean_if_survived: float
    #: Predictive spread for an INDIVIDUAL player, the decision-relevant uncertainty:
    #: the comp-to-comp SD in comps mode, and sqrt(residual variance + se^2) in shape
    #: mode. Both answer "how far from this number could one player land", which `se`
    #: does not.
    #:
    #: A SINGLE number, so any band read off it assumes the outcomes are normal around
    #: the estimate. Measured out of sample they are not, and not uniformly: hitters at
    #: +1 are close to normal (kurtosis 3.20, skew -0.41) but pitchers at +3 come out
    #: FLATTER than a bell (kurtosis 2.48, skew +0.30), with mass pushed onto both
    #: shoulders -- 41% of them land outside +/-1 spread against a Gaussian 32%. Prefer
    #: `p10`/`p90` for anything a reader will treat as an interval.
    spread: float = float("nan")
    #: 10th and 90th percentile of the outcome, read off the EMPIRICAL distribution --
    #: the weighted residual quantiles in shape mode, the comp values themselves in
    #: comps mode. Not `mean +/- k*spread`, which is a normality assumption the outcomes
    #: do not honour: hitters at +1 come out near-normal (kurtosis 3.20, skew -0.41) but
    #: pitchers at +3 come out FLATTER than a bell (kurtosis 2.48, skew +0.30), so one
    #: width cannot describe both. These carry the shape, and the skew with it.
    #:
    #: MEASURED COVERAGE, out of sample, query player held out. Each tail should hold
    #: 10%; a hot tail is a band that lies about that side.
    #:
    #:     pool       h     n   this band       mean +/- 1.28*spread
    #:     hitter     1   779   11% / 10%       12% / 9%
    #:     hitter     3   729   12% / 10%       12% / 9%
    #:     pitcher    1   503   13% / 11%       13% / 10%
    #:     pitcher    3   469   14% / 13%       11% / 14%
    #:
    #: So: better than the Gaussian reading for hitters and at pitcher +1, and a wash at
    #: pitcher +3, where the two are wrong in opposite directions -- the symmetric band
    #: happens to sit deeper on the downside because it ignores a right-skew the fitting
    #: residuals have and the realized elite-pitcher outcomes do not. Treat the interval
    #: as ROUGHLY 80% and slightly optimistic on the downside for pitchers, not exact.
    p10: float = float("nan")
    p90: float = float("nan")
    #: Support actually behind the number, as a Kish effective size. Equal to `n` in
    #: comps mode, where every comp counts once; strictly below it in shape mode, where
    #: kernel weights taper. Thin-support decisions must read THIS, not `n` -- a shape
    #: fit of 41 rows can carry an effective 15 and be degenerate while `n` looks ample.
    n_effective: float = float("nan")
    #: Fraction of the sample that played, WEIGHTED the way the estimate is. In comps
    #: that is survivors/n, since every comp counts once. In shape it is the
    #: kernel-weighted fraction, because an unweighted rate describes the far-age /
    #: far-prior tail the fit itself barely counted -- the same argument that made the
    #: median and the residual variance weighted. Left as a plain field rather than a
    #: property so the two modes can each say what they mean.
    survival: float = float("nan")


@dataclass(frozen=True)
class Trajectory:
    """A comp-based forward path, plus everything needed to judge whether to trust it."""

    kind: str
    age: int
    sgp: float
    band: float
    #: The prior-season SGP comps were additionally matched on, or None when matching
    #: on the current season alone. Its presence is what distinguishes the two modes.
    prior_sgp: float | None
    #: Comps matched on (age, band) that have at least the NEAREST horizon observable.
    #: Later horizons use fewer -- read `PathPoint.n`, not this, when judging a
    #: particular year.
    n_comps: int
    #: Mean SGP the comps actually started from. Drifts below `sgp` as `band` widens,
    #: because the SGP distribution thins out at the top -- a wide band pulls in more
    #: players from below the query than above it. A large gap means the path is
    #: answering a slightly easier question than the one asked.
    mean_start: float
    #: Mean SGP the comps produced the season BEFORE the one matched on. In
    #: track-record mode it should sit near `prior_sgp`; in current-season mode it is
    #: free, and reading it tells you what track record the plain cohort implicitly had.
    mean_prior: float
    seasons: tuple[int, int] | None
    path: tuple[PathPoint, ...]
    comps: pd.DataFrame = field(repr=False)
    #: Share of the fitting weight sitting within `DEFAULT_BAND` SGP of the query's own
    #: current season. NaN in the comp matchers, where the band IS the matching rule and
    #: this is 1.0 by construction.
    #:
    #: Shape weights `age` and `prior_sgp` on kernels but takes `sgp` -- this season -- as
    #: a bare regressor with no locality, so a query can be matched to a cohort it sits
    #: entirely outside and then priced by extrapolating that cohort's line. A 21-year-old
    #: at (13.6 now, 0.0 prior) draws the prior~0 population, whose own current seasons average
    #: 2.9, and the line fitted on those fringe seasons is evaluated 4.7x beyond their
    #: centre -- reporting 12.4 VAR over three years against a survivor mean near 1.0.
    #:
    #: The band comes out NARROW in exactly that case, because the residuals it is read
    #: off belong to the tightly-clustered fringe cohort rather than to anyone like the
    #: query. Confident and wrong together, which is the combination worth refusing.
    #:
    #: `mean_start` shows the same thing but cannot separate "above this cohort's mean"
    #: from "outside this cohort" -- on a real board it flagged 11 of the top 20,
    #: including players with perfectly ordinary support. This measures the support
    #: directly. Read it before ranking on `total`; #310 covers fixing the estimator.
    local_support: float = float("nan")
    #: True when at least one horizon could not find enough comps near the query's own
    #: current season to read a band from, and fell back to the whole cohort's residual
    #: scatter -- the understated interval the reweighting exists to replace. It happens
    #: on the DEEPEST extrapolations, the ones most in need of a wide band, so a caller
    #: telling a reader to trust the band must say when it silently reverted.
    band_fell_back: bool = False
    #: Which matcher produced this -- "current", "track" (comps.comp_trajectory) or
    #: "shape" (shape.shape_trajectory). In "shape" the numbers are a fitted prediction
    #: rather than an average over comps, so `PathPoint.n` counts FITTING rows and
    #: `band` does not apply.
    mode: str = "current"
    #: The replacement floor already netted out of every number here, and the slot it
    #: came from. Carried ON the trajectory rather than passed alongside it, because a
    #: caller who has to remember the scale is a caller who will eventually forget:
    #: every round of review on this feature found another consumer left on the raw
    #: scale -- the median, the survivor mean, the comps frame, the column headers, the
    #: total. A reader of this object can no longer be wrong about what its numbers mean.
    floor: float = 0.0
    slot: str | None = None

    @property
    def extrapolated(self) -> bool:
        """True when the fitted line was evaluated outside its own support.

        The THRESHOLD lives here, beside the field it qualifies, so every consumer shares
        one definition of "unsupported". It did not: the board owned the number and
        flagged rows with `(!)`, while `player_trajectory.py` -- the tool a human actually
        uses to make a single keep-or-cut call -- printed the same extrapolated fit with
        no warning at all. 10% separates the two failure modes on the real board; see
        `MIN_LOCAL_SUPPORT`. NaN (the comp matchers, where the band IS the matching rule)
        is not extrapolation.
        """
        return not np.isnan(self.local_support) and self.local_support < MIN_LOCAL_SUPPORT

    @property
    def scale(self) -> str:
        """ "var" once a floor has been netted out, "sgp" otherwise."""
        return "var" if self.slot is not None else "sgp"

    @property
    def observable(self) -> tuple[PathPoint, ...]:
        """Path points backed by at least one comp. A horizon further out than the panel
        can see yields n=0, and those carry no estimate."""
        return tuple(p for p in self.path if p.n > 0)

    @property
    def total(self) -> float:
        """Summed expected SGP over the OBSERVABLE horizons. The keeper-relevant scalar.

        Each year is the mean of ITS OWN observable cohort, so this sums across slightly
        different comp sets rather than following one set of careers. That is the right
        expectation year by year; it is not a distribution of realized multi-year totals,
        and a spread on it has to be built from the subset of comps observable at every
        horizon.

        Horizons with no comps at all are SKIPPED rather than summed as NaN -- one
        unreachable year would otherwise poison the headline number for every reachable
        one. Compare `len(observable)` against `len(path)` before reading this as a
        total over the horizon that was asked for.
        """
        return float(sum(p.mean for p in self.observable))


def collapse_split_seasons(panel: pd.DataFrame) -> pd.DataFrame:
    """One row per (player, season), summing a season split across two rows.

    A mid-season trade can produce two rows for one player-year. They must be collapsed
    BEFORE anything reads the panel, on BOTH the forward-lookup side and the
    cohort/fitting side. Collapsing only the lookup sums his future correctly while
    still entering him twice as two half-seasons -- inflating n, double-weighting him,
    and dragging `mean_start` low because each half sits below his real total.

    Shared by `comp_trajectory` and `shape.build_history` so the two estimators cannot
    drift apart on it; fixing one and not the other is exactly how this recurred.
    """
    if not panel.set_index(["mlbam_id", "season"]).index.has_duplicates:
        return panel
    return (
        panel.groupby(["mlbam_id", "season"], as_index=False)
        .agg(sgp=("sgp", "sum"), age=("age", "first"))
        .reset_index(drop=True)
    )


def _bootstrap_se(values: np.ndarray, rng: np.random.Generator, draws: int) -> float:
    if len(values) < 2:
        return float("nan")
    idx = rng.integers(0, len(values), size=(draws, len(values)))
    return float(values[idx].mean(axis=1).std(ddof=1))


def comp_trajectory(
    panel: pd.DataFrame,
    *,
    kind: str,
    age: int,
    sgp: float,
    band: float = DEFAULT_BAND,
    prior_sgp: float | None = None,
    prior_band: float | None = None,
    horizons: tuple[int, ...] = DEFAULT_HORIZONS,
    last_complete_season: int | None = None,
    replacement: float = 0.0,
    slot: str | None = None,
    seed: int = 0,
    bootstrap_draws: int = BOOTSTRAP_DRAWS,
) -> Trajectory:
    """Forward SGP path for a player at `age` producing `sgp`.

    `panel` is a scored, era-normalized panel of COMPLETE seasons (see
    `panel.load_scored_panel` and `era.era_normalize`); passing in-progress seasons
    would average a two-thirds year in as if it were a full one.

    `replacement` scores every horizon as VALUE ABOVE REPLACEMENT rather than raw SGP,
    flooring each comp at zero BEFORE aggregating: `max(sgp - replacement, 0)`. That has
    to happen here rather than as a shift of the finished mean, because a comp who left
    the league scores a structural 0 and is worth 0 to a roster slot -- not minus a
    floor. Subtracting afterwards charges the floor a second time against attrition;
    measured at age 30 it turned a +0.46 five-year value into -6.01.

    `last_complete_season` defaults to the panel's maximum and defines observability,
    which is applied PER HORIZON: a 2024 age-25 season has a real age-26 to look at and
    counts toward h1, while its h5 has not happened and is left out of h5 only. Judging
    every horizon by the longest one discarded 17% of usable h1 comps and, worse, cut
    the whole cohort off five years before the present -- no age-25 season after 2020
    contributed to any horizon, including the year-one estimate a decision leans on
    hardest.

    **`prior_sgp` switches on track-record matching (#305).** Left None, comps are
    matched on the current season alone and a 4 -> 13 breakout draws the same cohort as
    a steady 13 -> 13. Supplied, a comp must ALSO have produced within `prior_band` of
    it in his own preceding season, so the two queries get different comps and the
    breakout is priced with the give-back its comps actually had.

    A comp who was not in the league the year before scores a prior of **0**, the same
    convention the forward path uses -- not playing is an observation, and for a young
    player it is the normal one. But a season whose predecessor falls before the panel
    begins is UNOBSERVABLE and the comp is dropped, mirroring the forward censoring:
    "we cannot see it" must never be scored as "he did not play".
    """
    if band <= 0:
        raise ValueError(f"band must be positive, got {band}")
    if prior_band is not None and prior_band <= 0:
        raise ValueError(f"prior_band must be positive, got {prior_band}")
    if prior_band is not None and prior_sgp is None:
        raise ValueError("prior_band has no effect without prior_sgp")
    if not horizons:
        raise ValueError("horizons must not be empty")
    if bootstrap_draws < 2:
        # `std(ddof=1)` over fewer than two draws is NaN plus a RuntimeWarning, and the
        # caller sees an SE that is silently missing rather than a refused argument.
        # `_bootstrap_se` guards the COMP count, not the draw count, so it does not
        # cover this -- `shape_trajectory` refuses the same argument and the two modes
        # must not disagree about it (test_mode_parity).
        raise ValueError(f"bootstrap_draws must be at least 2, got {bootstrap_draws}")

    horizons = tuple(sorted(horizons))
    last = last_complete_season if last_complete_season is not None else int(panel["season"].max())
    panel = collapse_split_seasons(panel)
    by_player_season = panel.set_index(["mlbam_id", "season"])["sgp"]

    matched = panel[(panel["age"] == age) & (panel["sgp"] - sgp).abs().le(band)]
    # The NEAREST horizon sets membership: a comp too recent to have even one observable
    # forward season tells us nothing at all.
    matched = matched[matched["season"] + horizons[0] <= last]

    first = int(panel["season"].min())
    rows = []
    for row in matched.itertuples(index=False):
        # NaN and 0.0 mean different things and must not be conflated. NaN is "that
        # season has not been played yet" and drops out of the horizon; 0.0 is "he was
        # not in the league that year", which is a real observation worth exactly zero
        # to a roster slot.
        forward = {
            f"h{h}": float(by_player_season.get((row.mlbam_id, row.season + h), 0.0))
            if row.season + h <= last
            else float("nan")
            for h in horizons
        }
        # Same distinction looking BACKWARD: absent is 0, before the panel is NaN.
        back = (
            float(by_player_season.get((row.mlbam_id, row.season - 1), 0.0))
            if row.season - 1 >= first
            else float("nan")
        )
        rows.append(
            {"mlbam_id": row.mlbam_id, "season": row.season, "sgp0": row.sgp, "sgp_prior": back}
            | forward
        )
    comps = pd.DataFrame(rows)

    if prior_sgp is not None and not comps.empty:
        width = prior_band if prior_band is not None else band
        comps = comps[(comps["sgp_prior"] - prior_sgp).abs().le(width)].reset_index(drop=True)

    # Keep the pre-floor forwards: survival must stay readable off the RAW line, since
    # after flooring a below-replacement season and a career ending are both 0.
    raw_forward = {f"h{h}": comps[f"h{h}"].copy() for h in horizons} if not comps.empty else {}
    if replacement and not comps.empty:
        # The FRAME is floored too, not just the aggregates. It is what `--show-comps`
        # prints, and flooring only the aggregates left it listing raw SGP directly
        # beneath a VAR table -- anyone checking the arithmetic got a different mean
        # than the row above.
        for h in horizons:
            column = comps[f"h{h}"]
            comps[f"h{h}"] = column.where(column.isna(), (column - replacement).clip(lower=0.0))

    rng = np.random.default_rng(seed)
    path = []
    for h in horizons:
        values = comps[f"h{h}"].dropna().to_numpy(dtype=float) if not comps.empty else np.array([])
        # Survival off the RAW line: after flooring, a below-replacement season and a
        # career ending are both 0 and no longer tell apart, so the mask is taken from
        # the pre-floor values kept alongside.
        raw = raw_forward[f"h{h}"].dropna().to_numpy(dtype=float) if not comps.empty else values
        survived = values[played(raw)]
        path.append(
            PathPoint(
                horizon=h,
                age=age + h,
                mean=float(values.mean()) if len(values) else float("nan"),
                se=_bootstrap_se(values, rng, bootstrap_draws),
                median=float(np.median(values)) if len(values) else float("nan"),
                n=len(values),
                survivors=len(survived),
                mean_if_survived=float(survived.mean()) if len(survived) else float("nan"),
                spread=float(values.std(ddof=1)) if len(values) > 1 else float("nan"),
                # Straight off the comps, which ARE the empirical distribution here --
                # no residuals and no normality assumed. Same reason as shape mode: the
                # outcome spread changes shape by pool and horizon, and one width cannot
                # carry that. A departed comp's structural 0 is in `values`, so the low
                # end reflects attrition rather than hiding it.
                p10=float(np.percentile(values, 10)) if len(values) else float("nan"),
                p90=float(np.percentile(values, 90)) if len(values) else float("nan"),
                # Every comp counts exactly once, so the effective size IS the count and
                # the weighted survival rate IS survivors/n.
                n_effective=float(len(values)),
                survival=float(len(survived) / len(values)) if len(values) else float("nan"),
            )
        )

    return Trajectory(
        kind=kind,
        age=age,
        sgp=sgp,
        band=band,
        prior_sgp=prior_sgp,
        n_comps=len(comps),
        mean_start=float(comps["sgp0"].mean()) if not comps.empty else float("nan"),
        mean_prior=float(comps["sgp_prior"].mean()) if not comps.empty else float("nan"),
        seasons=(int(comps["season"].min()), int(comps["season"].max()))
        if not comps.empty
        else None,
        path=tuple(path),
        comps=comps,
        mode="track" if prior_sgp is not None else "current",
        floor=replacement,
        slot=slot,
    )
