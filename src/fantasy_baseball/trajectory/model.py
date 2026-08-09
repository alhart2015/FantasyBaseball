"""Primitives both trajectory estimators share.

Extracted from `comps.py` so the comp matchers can be retired (#325) without taking
`shape` with them -- `shape.py` imported six names from that module, and one of them,
`DEFAULT_BAND`, is not a comp band at all: it is the width `local_support` measures
the fitting weight inside. The design doc had it down as comps-only.

Nothing here is a matcher. `PathPoint` and `Trajectory` are the shapes an estimator
returns, `played` and `collapse_split_seasons` are panel semantics both must agree on,
and the rest are widths and defaults.
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
    #: True when THIS horizon's band could not find enough weight near the query's own
    #: current season and fell back to the whole cohort's residual scatter -- see
    #: `Trajectory.band_fell_back`, which is this OR-ed across the path.
    #:
    #: Per horizon rather than per trajectory because one sweep at the longest horizon
    #: serves every shorter range (#321): the per-year points are identical whichever
    #: tuple was fitted, so a board for 2027 alone is read off the same fit as 2027-2031.
    #: A latched flag would carry a +5 fallback onto a +1 view and mark a well-supported
    #: year unreliable.
    band_fell_back: bool = False


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
    #:
    #: The trajectory-level OR of `PathPoint.band_fell_back`. Read the per-point flag when
    #: showing a SUBSET of the path -- a consumer summing only the first year off a
    #: five-year fit would otherwise inherit a fallback that happened at +5.
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
    #: Per-horizon `h{n}` boolean columns, row-aligned to `comps`: True where the comp
    #: was out of the league that year. Taken BEFORE the floor is netted out, because
    #: after it a career ending reads `-floor` and is no longer recoverable from the
    #: value -- at the OF floor that is -9.96, four hundredths from a real -10.00
    #: season. `played` keys on the raw exact 0.0 for the same reason.
    #:
    #: NaN is left False: "has not happened yet" already renders as `--`, and folding
    #: it in here would paint an unobservable year as a career ending.
    #:
    #: Empty for `mode="shape"`, whose `comps` frame is per-HORIZON predictions rather
    #: than per-comp seasons and which refuses `--show-comps` outright.
    departed: pd.DataFrame = field(default_factory=pd.DataFrame, repr=False)

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
