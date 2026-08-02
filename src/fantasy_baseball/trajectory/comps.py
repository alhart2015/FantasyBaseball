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
DEFAULT_HORIZONS = (1, 2, 3, 4, 5)
BOOTSTRAP_DRAWS = 2000


@dataclass(frozen=True)
class PathPoint:
    """One horizon of a trajectory."""

    horizon: int
    age: int
    mean: float
    se: float
    median: float
    n: int
    survivors: int
    mean_if_survived: float

    @property
    def survival(self) -> float:
        return self.survivors / self.n if self.n else float("nan")


@dataclass(frozen=True)
class Trajectory:
    """A comp-based forward path, plus everything needed to judge whether to trust it."""

    kind: str
    age: int
    sgp: float
    band: float
    #: Comps matched on (age, band) that have at least the NEAREST horizon observable.
    #: Later horizons use fewer -- read `PathPoint.n`, not this, when judging a
    #: particular year.
    n_comps: int
    #: Mean SGP the comps actually started from. Drifts below `sgp` as `band` widens,
    #: because the SGP distribution thins out at the top -- a wide band pulls in more
    #: players from below the query than above it. A large gap means the path is
    #: answering a slightly easier question than the one asked.
    mean_start: float
    seasons: tuple[int, int] | None
    path: tuple[PathPoint, ...]
    comps: pd.DataFrame = field(repr=False)

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
    horizons: tuple[int, ...] = DEFAULT_HORIZONS,
    last_complete_season: int | None = None,
    seed: int = 0,
    bootstrap_draws: int = BOOTSTRAP_DRAWS,
) -> Trajectory:
    """Forward SGP path for a player at `age` producing `sgp`.

    `panel` is a scored, era-normalized panel of COMPLETE seasons (see
    `panel.load_scored_panel` and `era.era_normalize`); passing in-progress seasons
    would average a two-thirds year in as if it were a full one.

    `last_complete_season` defaults to the panel's maximum and defines observability,
    which is applied PER HORIZON: a 2024 age-25 season has a real age-26 to look at and
    counts toward h1, while its h5 has not happened and is left out of h5 only. Judging
    every horizon by the longest one discarded 17% of usable h1 comps and, worse, cut
    the whole cohort off five years before the present -- no age-25 season after 2020
    contributed to any horizon, including the year-one estimate a decision leans on
    hardest.
    """
    if band <= 0:
        raise ValueError(f"band must be positive, got {band}")
    if not horizons:
        raise ValueError("horizons must not be empty")

    horizons = tuple(sorted(horizons))
    last = last_complete_season if last_complete_season is not None else int(panel["season"].max())
    # A player with two rows for one season (a mid-season trade split, say) must be
    # collapsed BEFORE anything reads him, on BOTH sides. Doing it only on the forward
    # lookup summed his future correctly while still entering him into the cohort twice
    # as two half-seasons -- inflating n, double-weighting him in the mean, and dragging
    # `mean_start` toward the band's lower edge, since each half sits below his real
    # total.
    if panel.set_index(["mlbam_id", "season"]).index.has_duplicates:
        panel = (
            panel.groupby(["mlbam_id", "season"], as_index=False)
            .agg(sgp=("sgp", "sum"), age=("age", "first"))
            .reset_index(drop=True)
        )
    by_player_season = panel.set_index(["mlbam_id", "season"])["sgp"]

    matched = panel[(panel["age"] == age) & (panel["sgp"] - sgp).abs().le(band)]
    # The NEAREST horizon sets membership: a comp too recent to have even one observable
    # forward season tells us nothing at all.
    matched = matched[matched["season"] + horizons[0] <= last]

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
        rows.append({"mlbam_id": row.mlbam_id, "season": row.season, "sgp0": row.sgp} | forward)
    comps = pd.DataFrame(rows)

    rng = np.random.default_rng(seed)
    path = []
    for h in horizons:
        values = comps[f"h{h}"].dropna().to_numpy(dtype=float) if not comps.empty else np.array([])
        survived = values[values != 0.0]
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
            )
        )

    return Trajectory(
        kind=kind,
        age=age,
        sgp=sgp,
        band=band,
        n_comps=len(comps),
        mean_start=float(comps["sgp0"].mean()) if not comps.empty else float("nan"),
        seasons=(int(comps["season"].min()), int(comps["season"].max()))
        if not comps.empty
        else None,
        path=tuple(path),
        comps=comps,
    )
