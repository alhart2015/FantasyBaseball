"""Conformal calibration of the trajectory band, so its tails hold what they claim.

`scripts/calibrate_band_coverage.py` measured the shipped band against 62k held-out
query-horizons and found it miscalibrated in BOTH directions, monotonically in local
support: too narrow below 30% support (24% of pitchers under a nominal-10% p10) and too
wide above it (3-4%). The multi-year total was honest only by accident -- `sweep.totals`
sums the yearly bands, which assumes the years move together and over-widens by ~1.7x,
roughly cancelling the per-year narrowness. Fixing either half alone breaks the other.

This module makes coverage a CONSTRUCTION rather than a property to re-measure after
every estimator change. It is split conformal prediction, Mondrian (group-conditional) on
`(pool, target, support bucket, side)`:

    m = Quantile_{1-alpha} of  (predicted - actual) / (predicted - p10)     [lower]
    m = Quantile_{1-alpha} of  (actual - predicted) / (p90 - predicted)     [upper]

Scaling each half-width by its own `m` puts exactly `alpha` in that tail BY DEFINITION of
the quantile. No Gaussian assumption, no assumption about the residual shape, and no
assumption about how the years correlate -- because of the design rule below.

CONFORMALIZE EVERY QUANTITY THAT IS DISPLAYED, SEPARATELY, AND EACH FROM THE RAW BAND. A
single year gets a single-year multiplier; the 1..k sum gets its own, derived against the
raw summed band. That is what removes the correlation assumption -- nobody ever asks how
year 2 and year 3 covary, because the sum is calibrated as its own target -- and it is
also what keeps the corrections independent. `TARGETS` enumerates every span the board can
render, so there is no uncovered path needing a caveat.

THE CONSUMER SUMS RAW BANDS AND CORRECTS ONCE. Correcting the years and then summing THOSE
is the bug this design exists to prevent: a span multiplier fitted against already-
corrected inputs is valid only in composition with the yearly one, an ordering contract
nothing in the type system carries. Measured, applying such a table to a raw sum reads
13-17% below p10 in-sample against a nominal 10% -- a miscalibration wearing the shape of
a correction. `apply` takes the target precisely so the caller names which one it wants.

The finite-sample level (`_conformal_level`) is `ceil((n+1)(1-alpha))/n`, not `1-alpha`.
That is the difference between "approximately calibrated on this sample" and the standard
conformal guarantee of AT LEAST `1-alpha` coverage on the next exchangeable draw. It
matters most exactly where the cells are thinnest, which is the low-support end this
exists to fix.

EXCHANGEABILITY IS THE ASSUMPTION THAT REMAINS, and it is the one to check rather than
state: the guarantee holds if a 2027 query is exchangeable with the calibration seasons.
`scripts/build_band_calibration.py --validate` checks it by ROLLING ORIGIN -- fit on every
outcome observable before season Y, measure on the queries resolving in Y, which is the
production refresh exactly. Measured that way, at `CALIBRATION_WINDOW_YEARS` and excluding
spans containing the 60-game 2020 season: y1 holds 10.0%/9.7%, s3 holds 10.0%/9.3%, s5
holds 10.8%/9.2%, against a nominal 10%/10%.

Do not read a single early-vs-late split instead; it extrapolates nine years and reads
~2 points hotter than the tool will ever be in use. What that split IS good for is
choosing the bucket edges, since a multiplier that cannot survive nine years cannot
survive one -- see `SUPPORT_EDGES`.

2020 is reported separately rather than corrected for. A 60-game season scaled to 162
(`panel.py`) carries ~2.7x the sampling variance of a full one, so spans touching it are
genuinely more dispersed than anything the band was fitted on -- year-1 coverage there is
16.4% against 10%. Excluding those spans from the FIT was measured and moves the
multipliers by a median 0.005, so it buys nothing; pricing a pandemic into every future
band permanently would be the larger error.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[3]

#: Nominal mass in EACH tail. The band is p10..p90, so 10% below and 10% above.
ALPHA = 0.10

#: Upper edges of the local-support buckets the multipliers are conditioned on. A single
#: pooled multiplier would not do: the miscalibration reverses sign across this edge --
#: the band needs WIDENING below it and TIGHTENING above -- so one number would split the
#: difference and leave both ends wrong.
#:
#: ONE EDGE, AT 30%, CHOSEN BY MEASURED STABILITY rather than by matching the finer
#: buckets `calibrate_band_coverage.py` reports on. Multipliers were fitted on 2001-2015
#: and again on 2016-2024 and compared, since a multiplier that does not survive a
#: nine-year gap will not survive the gap to 2027 either:
#:
#:     buckets                          median drift   max drift
#:     <5% | 5-10% | 10-30% | >30%          0.040        0.677
#:     <10% | 10-30% | >30%                 0.042        0.245
#:     <30% | >30%                          0.039        0.096
#:
#: The median is flat across all three, so the finer splits buy no accuracy -- they only
#: add variance, and at the thin end that variance INVERTS the correction: the <5% cell
#: wanted x0.66 on 2001-2015 and x1.34 on 2016-2024, and applying the former to the latter
#: took hitter s3 coverage from 13.3% to 30.0% against a nominal 10%. A correction that
#: makes a band worse than no correction is the one outcome this module must not ship.
#:
#: 30% is also where the coverage cliff actually sits (see
#: `docs/trajectory-band-calibration-2026-09-04.md`): below it every bucket runs hot by a
#: similar amount, above it the band is uniformly too wide. The finer edges were
#: describing noise inside one real population.
SUPPORT_EDGES = (0.30,)
BUCKET_LABELS = ("<30%", ">30%")

#: Horizons the board can render. `--horizon` is clamped to this so every displayed span
#: has a calibrated multiplier and no query falls through to an uncorrected band.
MAX_HORIZON = 5

#: Every displayed quantity. `y{h}` is the year-{h} column, `s{k}` the 1..k total that
#: headlines the board. `s1` and `y1` are the same number and get the same multiplier --
#: computed separately anyway, because a shared cell would silently couple the two if the
#: board ever displayed a span not starting at 1.
TARGETS = tuple(f"y{h}" for h in range(1, MAX_HORIZON + 1)) + tuple(
    f"s{k}" for k in range(1, MAX_HORIZON + 1)
)

#: Outcome seasons a cell is fitted on, counted back from the newest outcome in the frame.
#: The required multiplier drifts slowly, so an all-history fit lags it and leaves the
#: upper tail systematically cold. Measured by rolling origin, 2020 spans excluded
#: (below p10 / above p90, nominal 10%/10% on each):
#:
#:     fit window     y1            s3            s5
#:     all history    10.5 / 9.8    10.2 / 8.8    10.8 / 8.7
#:     last 12y       10.3 / 9.9    10.1 / 9.0    10.8 / 8.7
#:     last 8y        10.0 / 9.7    10.0 / 9.3    10.8 / 9.2
#:
#: Eight years is best or tied on every target and halves the worst deviation, 1.3 points
#: to 0.8. Shorter was not tried below eight because the cells thin out and `MIN_CELL_ROWS`
#: would start pooling the buckets away -- which costs the one distinction that matters.
#: The build prints `fallbacks`, so that failure is visible if a future panel triggers it.
CALIBRATION_WINDOW_YEARS = 8

#: Below this many calibration rows a cell falls back to its pool-and-target row pooled
#: across buckets, and below that to 1.0. A conformal quantile off 30 rows is itself noise
#: and would ship as a confident correction. Deterministic and recorded in the artifact's
#: `fallbacks`, so a thin cell is visible rather than inferred.
MIN_CELL_ROWS = 150

#: Where the fitted table ships. Under `data/trajectory/` beside the panels it was fitted
#: on, so the two are backed up, copied and deleted together -- a calibration that
#: outlives its panel is the failure `BandCalibration.load` refuses.
CALIBRATION_PATH = Path("data") / "trajectory" / "band_calibration.json"

#: A band side pinned to zero width by `shape`'s containment clamp, which fires when the
#: reweighted residual quantiles both land on one side of `predicted`. Measured at 35 of
#: 90,555 held-out rows (0.04%). No ratio can be formed against a zero denominator, so
#: these are dropped from the fit; the correction is applied UPSTREAM of that clamp in
#: `shape`, so in production the scaling happens while both sides still have width.
MIN_HALF_WIDTH = 1e-9

#: Narrowest a correction may make a half-width. A conformal quantile is signed, so an
#: over-wide cell can in principle fit a multiplier at or below zero, and `apply` would
#: then emit p10 at or above the point estimate -- an inverted band that prints as an
#: ordinary one. Small enough to bind on nothing the fit has ever produced (the shipped
#: table bottoms out at 0.67) and large enough that the band stays an interval.
MIN_MULTIPLIER = 1e-3


def panel_vintage_of(panel_dir: Path | None = None, *, missing_ok: bool = False) -> str | None:
    """The panel filenames, joined -- the identity a calibration is fitted against.

    FILENAMES, not a timestamp, matching `push_trajectory_board.py`'s vintage line: the
    panel is a build artifact whose name carries its season range, and an mtime changes
    on a copy that did not rebuild anything.

    `missing_ok` RETURNS None WHERE THERE IS NO PANEL AT ALL, and that is the deployed
    case, not an exotic one. `data/trajectory/*` is gitignored except the two fitted JSON
    artifacts, so Render has the calibration and NOT the CSVs it was fitted on --
    `panel_path` raises FileNotFoundError there. `load_shipped` sits on the board render
    path, so raising took every trajectory page to a 500; the vintage guard exists to
    catch a calibration paired with the WRONG panel, and with no panel present there is
    no pairing to be wrong about. The build scripts leave it False: they are about to
    read the panel anyway, and a missing one there must still be an error.
    """
    from .panel import panel_path

    try:
        return "+".join(panel_path(kind, panel_dir).name for kind in ("hitter", "pitcher"))
    except FileNotFoundError:
        if missing_ok:
            return None
        raise


def bucket_of(support: float) -> str:
    """The support bucket a query falls in. NaN (the comp matchers) reads as best-supported.

    NaN means no line was fitted, so there is no extrapolation to correct for -- treating
    it as the thinnest bucket would widen a band that was never narrow.
    """
    if math.isnan(support):
        return BUCKET_LABELS[-1]
    # `strict=True`: `BUCKET_LABELS` has exactly one more entry than `SUPPORT_EDGES` (the
    # open top bucket), so pairing the edges with all but the last label is total. Adding
    # an edge without its label would otherwise be SILENT -- the extra bucket would fold
    # into the one above it and every query in it would take the wrong multiplier.
    for edge, label in zip(SUPPORT_EDGES, BUCKET_LABELS[:-1], strict=True):
        if support < edge:
            return label
    return BUCKET_LABELS[-1]


def _conformal_level(n: int, alpha: float = ALPHA) -> float:
    """The finite-sample quantile level giving at least `1 - alpha` coverage.

    `ceil((n+1)(1-alpha))/n`, the standard split-conformal correction. At n=200 it asks
    for the 0.905 quantile rather than the 0.900 -- small, and precisely the margin that
    turns a sample statistic into a guarantee on the next draw. Capped at 1.0, which is
    what a cell too small to express the level collapses to (the widest observed score).
    """
    return min(1.0, math.ceil((n + 1) * (1.0 - alpha)) / n)


def conformal_multipliers(
    predicted: np.ndarray, p10: np.ndarray, p90: np.ndarray, actual: np.ndarray
) -> tuple[float, float, int]:
    """`(lower, upper, n)` multipliers putting `ALPHA` in each tail of this cell.

    The score is the realized error as a MULTIPLE of the half-width the band offered, so
    a cell mixing wide and narrow bands is still pooled correctly -- what is being
    calibrated is the band's own scale, not an absolute number of SGP.

    Taken over ALL rows rather than only the rows that fell outside. A quantile
    conditioned on having already missed answers a different question, and there is no
    guarantee attached to it.

    FLOORED AT `MIN_MULTIPLIER`. The quantile is a signed score, so a cell whose band is
    grossly too wide can hand back a non-positive multiplier -- and `apply` would then
    put p10 ON or ABOVE the point estimate, an INVERTED band that renders as a plausible
    "[12.4, 9.1]" rather than as an error. The shipped table's smallest is 0.67, so this
    binds on nothing today; it is the guard that keeps a future refit from shipping one.
    """
    lo_w = predicted - p10
    hi_w = p90 - predicted
    usable = (lo_w > MIN_HALF_WIDTH) & (hi_w > MIN_HALF_WIDTH)
    lo_w, hi_w = lo_w[usable], hi_w[usable]
    err = actual[usable] - predicted[usable]
    n = int(usable.sum())
    if n == 0:
        return 1.0, 1.0, 0
    level = _conformal_level(n)
    return (
        max(MIN_MULTIPLIER, float(np.quantile(-err / lo_w, level))),
        max(MIN_MULTIPLIER, float(np.quantile(err / hi_w, level))),
        n,
    )


#: Levels the signed-score curve is stored at. Dense in the tails, where a keeper question
#: actually lives -- "what are the odds he clears a top-10 bar" is a question about the
#: 80th percentile and up, and a uniform grid would resolve that region worst.
CURVE_LEVELS = (
    0.005,
    0.01,
    0.02,
    0.03,
    0.05,
    0.075,
    0.10,
    0.15,
    0.20,
    0.25,
    0.30,
    0.35,
    0.40,
    0.45,
    0.50,
    0.55,
    0.60,
    0.65,
    0.70,
    0.75,
    0.80,
    0.85,
    0.90,
    0.925,
    0.95,
    0.97,
    0.98,
    0.99,
    0.995,
)


def signed_scores(
    predicted: np.ndarray, p10: np.ndarray, p90: np.ndarray, actual: np.ndarray
) -> np.ndarray:
    """The realized error in units of the half-width on ITS OWN side of `predicted`.

    One monotone quantity covering both tails: positive outcomes are measured against
    `p90 - predicted` and negative ones against `predicted - p10`, so an asymmetric band
    is normalised correctly on each side and the score is still increasing in `actual`.
    That monotonicity is what lets `exceedance` invert it.

    It reduces exactly to `conformal_multipliers`: the score's `1 - ALPHA` quantile IS the
    upper multiplier, and its `ALPHA` quantile is minus the lower one. The band edges are
    two points on this curve rather than a separate calculation.
    """
    lo_w = predicted - p10
    hi_w = p90 - predicted
    usable = (lo_w > MIN_HALF_WIDTH) & (hi_w > MIN_HALF_WIDTH)
    err = actual[usable] - predicted[usable]
    scores: np.ndarray = np.where(err >= 0, err / hi_w[usable], err / lo_w[usable])
    return scores


@dataclass(frozen=True)
class BandCalibration:
    """Multipliers keyed `pool -> target -> bucket -> (lower, upper)`, plus provenance.

    `panel_vintage` is checked on load against the panel actually in use. A calibration
    fitted on one panel and applied to another is silently wrong in the direction that
    looks fine -- the bands still print, at a scale nothing measured -- so the mismatch
    is an error rather than a warning.
    """

    panel_vintage: str
    alpha: float
    #: Outcome seasons the fit was windowed to. Recorded because it is a CHOICE with a
    #: measured basis (see `CALIBRATION_WINDOW_YEARS`), so a table fitted under a different
    #: one is a different object and the artifact should say which.
    window_years: int
    multipliers: dict[str, dict[str, dict[str, tuple[float, float]]]]
    #: Signed-score quantiles at `CURVE_LEVELS`, keyed like `multipliers`. The band edges
    #: are two points on this; the curve is what answers "how likely is he to clear X".
    #: Empty on a table built before curves existed, which `exceedance` reports as None
    #: rather than guessing a shape the outcomes are documented not to have.
    curves: dict[str, dict[str, dict[str, tuple[float, ...]]]]
    #: Cells that fell back for want of rows, as "pool/target/bucket" -> the n it had.
    #: Recorded so a thin correction is auditable rather than indistinguishable from a
    #: well-populated one.
    fallbacks: dict[str, int]

    def scale(self, *, pool: str, target: str, support: float) -> tuple[float, float]:
        """The `(lower, upper)` multipliers for one query, or `(1.0, 1.0)` if uncalibrated.

        An unknown pool or target returns the identity rather than raising: the caller is
        rendering a band either way, and a KeyError deep in a sweep would take out a whole
        board over one row. `build_band_calibration.py` is what guarantees the table is
        complete; this is the read path.
        """
        cell = self.multipliers.get(pool, {}).get(target, {}).get(bucket_of(support))
        if cell is None:
            return 1.0, 1.0
        return float(cell[0]), float(cell[1])

    def apply(
        self, predicted: float, p10: float, p90: float, *, pool: str, target: str, support: float
    ) -> tuple[float, float]:
        """One band, corrected. Scales each half-width about `predicted`.

        About `predicted` and not about the band's own centre, because `predicted` is the
        number the multipliers were derived against -- the score's denominator is the
        offset from it. Re-centring here would leave the correction measuring one thing
        and correcting another.
        """
        lo, hi = self.scale(pool=pool, target=target, support=support)
        return predicted - lo * (predicted - p10), predicted + hi * (p90 - predicted)

    def apply_year(
        self, predicted: float, p10: float, p90: float, *, pool: str, horizon: int, support: float
    ) -> tuple[float, float]:
        """One YEAR's band, corrected. The per-year sibling of `apply`.

        Separate entry point rather than a `target=` at every call site, because the two
        are easy to swap and the failure is silent: a year band scaled by `s3` is scaled
        by a multiplier fitted on a sum three times its width. Naming the horizon makes
        the wrong one hard to write.

        A horizon past `MAX_HORIZON` returns the band unchanged -- `scale` already falls
        through to the identity on an unknown target, and this states the reason.
        """
        return self.apply(predicted, p10, p90, pool=pool, target=f"y{horizon}", support=support)

    def exceedance(
        self,
        threshold: float,
        predicted: float,
        raw_p10: float,
        raw_p90: float,
        *,
        pool: str,
        target: str,
        support: float,
    ) -> float | None:
        """P(outcome > `threshold`), read off the calibrated score curve.

        THE BAND MUST BE THE RAW ONE, hence the parameter names. `build_table` fits this
        curve on `signed_scores` of the UNCORRECTED held-out band, so the half-widths here
        are its denominators; passing an already-corrected band divides by a scaled width
        and shifts the answer. Measured on the live board, Caminero reads 23% bust on the
        raw band and 16% on the corrected one -- both plausible, one wrong.

        NO DISTRIBUTIONAL ASSUMPTION. The threshold is converted to a signed score in the
        same units the curve is stored in, and the answer is the share of held-out
        outcomes that scored higher. Fitting a normal through p10/p90 would be the easy
        alternative and is documented as wrong for this data -- `PathPoint.p10` measures
        hitters near-normal at +1 but pitchers FLATTER than a bell at +3, with 41% of them
        outside +/-1 spread against a Gaussian 32%.

        Interpolated between stored levels, and clamped at the ends: past the outermost
        level the honest answer is "under 0.5%", not an extrapolated tail. Returns None
        when the cell is unknown or the band has a degenerate side, so a caller renders a
        blank rather than a fabricated probability.
        """
        curve = self.curves.get(pool, {}).get(target, {}).get(bucket_of(support))
        if not curve:
            return None
        lo_w, hi_w = predicted - raw_p10, raw_p90 - predicted
        if lo_w <= MIN_HALF_WIDTH or hi_w <= MIN_HALF_WIDTH:
            return None
        offset = threshold - predicted
        score = offset / hi_w if offset >= 0 else offset / lo_w
        # `np.interp` clamps outside the grid, which is the behaviour wanted here: it
        # saturates at the outermost measured level instead of inventing a tail.
        level = float(np.interp(score, np.asarray(curve), np.asarray(CURVE_LEVELS)))
        return 1.0 - level

    def to_json(self) -> str:
        return json.dumps(
            {
                "panel_vintage": self.panel_vintage,
                "alpha": self.alpha,
                "window_years": self.window_years,
                "multipliers": self.multipliers,
                "curves": self.curves,
                "curve_levels": list(CURVE_LEVELS),
                "fallbacks": self.fallbacks,
            },
            indent=2,
            sort_keys=True,
        )

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(self.to_json(), encoding="utf-8")

    @classmethod
    def load(cls, path: Path, *, panel_vintage: str | None = None) -> BandCalibration:
        """Read the artifact, refusing a vintage that does not match the panel in use."""
        raw = json.loads(path.read_text(encoding="utf-8"))
        # BACK TO TUPLES. JSON has no tuple, so a round-tripped table came back holding
        # lists and compared unequal to the one that wrote it -- the pair is a fixed
        # (lower, upper), not a sequence, and a loaded table must be indistinguishable
        # from a built one or the difference surfaces somewhere far from here.
        multipliers = {
            pool: {
                target: {label: (float(v[0]), float(v[1])) for label, v in cells.items()}
                for target, cells in per_pool.items()
            }
            for pool, per_pool in raw["multipliers"].items()
        }
        curves = {
            pool: {
                target: {label: tuple(float(x) for x in v) for label, v in cells.items()}
                for target, cells in per_pool.items()
            }
            for pool, per_pool in raw.get("curves", {}).items()
        }
        stored = tuple(raw.get("curve_levels", ()))
        if curves and stored != CURVE_LEVELS:
            # The curve is meaningless without the levels it was sampled at, and a grid
            # change is exactly the edit that would not fail on its own.
            raise ValueError(
                f"band calibration stores {len(stored)} curve levels but this build "
                f"expects {len(CURVE_LEVELS)}. Regenerate with "
                f"`python scripts/build_band_calibration.py`."
            )
        table = cls(
            panel_vintage=raw["panel_vintage"],
            alpha=raw["alpha"],
            window_years=raw["window_years"],
            multipliers=multipliers,
            curves=curves,
            fallbacks=raw.get("fallbacks", {}),
        )
        if panel_vintage is not None and table.panel_vintage != panel_vintage:
            raise ValueError(
                f"band calibration was fitted on panel {table.panel_vintage!r} but the "
                f"panel in use is {panel_vintage!r}. Regenerate with "
                f"`python scripts/build_band_calibration.py`, or the bands print at a "
                f"scale nothing measured."
            )
        return table


def span_frame(frame: pd.DataFrame, target: str) -> pd.DataFrame:
    """The held-out rows for one target: a single year (`y{h}`) or a 1..k sum (`s{k}`).

    A span drops any query missing one of its horizons rather than summing short -- a
    two-year sum and a three-year sum are different quantities, and pooling their scores
    would calibrate neither.
    """
    kind, k = target[0], int(target[1:])
    if kind == "y":
        return frame[frame["horizon"] == k]
    wanted = set(range(1, k + 1))
    span = frame[frame["horizon"].isin(wanted)]
    grouped = span.groupby(["mlbam_id", "season"], sort=False)
    span = span[grouped["horizon"].transform("nunique") == len(wanted)]
    if span.empty:
        return span
    return (
        span.groupby(["mlbam_id", "season"], sort=False)
        .agg(
            support=("support", "first"),
            predicted=("predicted", "sum"),
            p10=("p10", "sum"),
            p90=("p90", "sum"),
            actual=("actual", "sum"),
        )
        .reset_index()
    )


#: A table that corrects nothing. `scale` falls through to the identity on an unknown
#: cell, so an empty one leaves every band exactly as `shape` produced it.
#:
#: EXISTS SO "NO CALIBRATION" IS SAYABLE. `totals(calibration=None)` means "load the
#: shipped table", which is the right default for production and leaves a caller who
#: genuinely wants the raw band with nothing to pass. That caller is a test grounding the
#: sweep against `shape_trajectory` itself, and it should not have to hand-build a
#: `BandCalibration` to say so.
IDENTITY = BandCalibration(
    panel_vintage="identity",
    alpha=ALPHA,
    window_years=0,
    multipliers={},
    curves={},
    fallbacks={},
)


def span_target(horizons: tuple[int, ...]) -> str | None:
    """The `s{k}` target a displayed range maps to, or None if nothing calibrated covers it.

    A range must be 1..k contiguous, because that is what was fitted -- `TARGETS` has no
    entry for 2..4, and silently pricing it with `s4` would apply a multiplier derived
    from a strictly wider sum. None means the caller leaves the band alone, which is the
    only honest option: an uncalibrated band is better than one corrected by the wrong
    number, and `trajectory_board.py` clamps `--horizon` so the board cannot reach here.
    """
    if not horizons:
        return None
    ordered = tuple(sorted(horizons))
    k = ordered[-1]
    if ordered != tuple(range(1, k + 1)) or k > MAX_HORIZON:
        return None
    return f"s{k}"


@lru_cache(maxsize=4)
def load_shipped(panel_dir: Path | None = None) -> BandCalibration | None:
    """The artifact under `data/trajectory/`, or None when it has not been built.

    None rather than a raise: a fresh clone that has built the panel but not yet the
    calibration should still render a board, and the band it renders is the pre-#331
    band -- worse, but not wrong in a new way. A MISMATCHED artifact is different and
    still raises, because that one is silently wrong.

    Cached, since `totals` is called per board render and this reads and parses JSON.
    """
    path = PROJECT_ROOT / CALIBRATION_PATH
    if not path.exists():
        return None
    return BandCalibration.load(path, panel_vintage=panel_vintage_of(panel_dir, missing_ok=True))


def newest_outcome(frame: pd.DataFrame) -> int:
    """The most recent season any row in `frame` has an outcome for."""
    return int((frame["season"] + frame["horizon"]).max())


def fit_rows(frame: pd.DataFrame, target: str, *, window_years: int, newest: int) -> pd.DataFrame:
    """The rows one target is fitted on: spanned, then windowed to recent outcomes.

    ONE DEFINITION, called by both the fit and every report that checks it. These were
    written twice -- `build_table` windowed and the build script's in-sample report did
    not -- and the report then measured a windowed table against 24 years of rows, came
    out 1-3 points off nominal, and flagged the discrepancy as a defect in the estimator.
    The plumbing check has to evaluate exactly what was fitted or it is checking nothing.

    WINDOWED ON THE TARGET'S LAST OUTCOME YEAR (`season + k`), not on the query season. A
    1..5 total is not observable until five years after its query, so windowing on the
    query would keep spans whose outcomes are far newer than a year-1 row's and fit the
    two on quietly different periods.
    """
    k = int(target[1:])
    return span_frame(frame[frame["season"] + k > newest - window_years], target)


def build_table(
    frame: pd.DataFrame,
    *,
    panel_vintage: str,
    alpha: float = ALPHA,
    window_years: int = CALIBRATION_WINDOW_YEARS,
) -> BandCalibration:
    """Fit multipliers from held-out predictions. Every target against the RAW band.

    `frame` is `calibrate_band_coverage.score_pool` output: one row per (query, horizon)
    carrying `predicted`, `p10`, `p90`, `actual`, `support` and `pool`.

    ONE STAGE, and that is the correctness property, not a simplification. Each target is
    calibrated from the raw band independently, so applying it needs nothing else to have
    happened first: a year column takes `y{h}` against the raw year band, a total takes
    `s{k}` against the raw SUM of year bands. Neither reads the other's output.

    The alternative was tried and is wrong. Fitting `s{k}` against ALREADY-CORRECTED
    yearly bands -- on the reasoning that this is what `totals` would be summing -- makes
    the span multiplier valid only in composition with the yearly one, an ordering
    contract no type carries and nothing enforces. Applied to a raw sum it reads 13-17%
    below p10 in-sample against a nominal 10%, which is a MISCALIBRATION THAT LOOKS LIKE
    A CORRECTION. The consumer must therefore sum raw bands and correct once; `apply`
    takes the target that says which.
    """
    multipliers: dict[str, dict[str, dict[str, tuple[float, float]]]] = {}
    curves: dict[str, dict[str, dict[str, tuple[float, ...]]]] = {}
    fallbacks: dict[str, int] = {}

    def arrays(rows: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """The four columns both fits read. Extracted once per frame rather than four
        times -- `conformal_multipliers` and `signed_scores` want the same vectors."""
        return (
            rows["predicted"].to_numpy(),
            rows["p10"].to_numpy(),
            rows["p90"].to_numpy(),
            rows["actual"].to_numpy(),
        )

    def fit(pool_rows: pd.DataFrame, pool: str, target: str) -> None:
        pooled_arrays = arrays(pool_rows)
        pooled = conformal_multipliers(*pooled_arrays)
        pooled_scores = signed_scores(*pooled_arrays)
        cells: dict[str, tuple[float, float]] = {}
        curve_cells: dict[str, tuple[float, ...]] = {}
        buckets = pool_rows["support"].map(bucket_of)
        for label in BUCKET_LABELS:
            sub_arrays = arrays(pool_rows[buckets == label])
            lo, hi, n = conformal_multipliers(*sub_arrays)
            scores = signed_scores(*sub_arrays)
            if n < MIN_CELL_ROWS:
                fallbacks[f"{pool}/{target}/{label}"] = n
                lo, hi = pooled[0], pooled[1]
                scores = pooled_scores
            cells[label] = (lo, hi)
            # The curve and the two edges come off the SAME rows, so a band edge and the
            # probability of clearing it can never disagree.
            curve_cells[label] = (
                tuple(float(x) for x in np.quantile(scores, CURVE_LEVELS)) if len(scores) else ()
            )
        multipliers.setdefault(pool, {})[target] = cells
        curves.setdefault(pool, {})[target] = curve_cells

    # ONCE, over the WHOLE frame: the window is a property of the panel, not of a pool or
    # a target, and recomputing it inside the loop was 20 full-frame passes per build.
    newest = newest_outcome(frame)
    for pool, pool_rows in frame.groupby("pool", sort=False):
        for target in TARGETS:
            rows = fit_rows(pool_rows, target, window_years=window_years, newest=newest)
            if not rows.empty:
                fit(rows, str(pool), target)

    return BandCalibration(
        panel_vintage=panel_vintage,
        alpha=alpha,
        window_years=window_years,
        multipliers=multipliers,
        curves=curves,
        fallbacks=fallbacks,
    )
