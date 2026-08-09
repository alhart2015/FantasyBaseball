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

import numpy as np
import pandas as pd

from .model import (
    DEFAULT_BAND,
    DEFAULT_HORIZONS,
    PathPoint,
    Trajectory,
    collapse_split_seasons,
    played,
)

BOOTSTRAP_DRAWS = 2000


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
    shifting each comp BEFORE aggregating: `sgp - replacement`. It happens here rather
    than as a shift of the finished mean so that the median, the survivor mean, the band
    and the `--show-comps` frame all land on the same scale as the mean -- shifting
    afterwards left every one of them on the raw scale.

    NOT floored at zero, deliberately (#331). It was `max(sgp - replacement, 0)`, which
    kept a departed comp at 0 rather than minus a floor. The reversal was forced by the
    OTHER matcher, where the identical clamp sat on a regression response and reordered
    players sharing a slot; `shape.shape_trajectory` carries the full argument and the
    cost accepted with it. It is applied here so the two cannot disagree about what a
    VAR is -- see `test_mode_parity.test_both_modes_shift_var_by_the_floor_unclamped`.

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

    # Survival off the RAW line, so take the mask BEFORE the shift: shifted, a career
    # ending reads `-replacement` rather than the exact 0 `played` keys on. Only the mask
    # is kept, not a copy of the values -- `shape_trajectory` does the same thing the same
    # way, and NaN positions are untouched by a subtraction so it stays aligned.
    survived_at = (
        {f"h{h}": played(comps[f"h{h}"].dropna().to_numpy(dtype=float)) for h in horizons}
        if not comps.empty
        else {}
    )
    # Row-aligned to the frame rather than to the dropna'd arrays `survived_at` uses,
    # because this one is consumed per PRINTED CELL and has to line up with what the
    # reader sees. Same raw-0.0 test, same reason for taking it before the shift.
    departed = pd.DataFrame(
        {f"h{h}": comps[f"h{h}"] == 0.0 for h in horizons} if not comps.empty else {}
    )
    if not comps.empty:
        # The FRAME is shifted too, not just the aggregates. It is what `--show-comps`
        # prints, and shifting only the aggregates left it listing raw SGP directly
        # beneath a VAR table -- anyone checking the arithmetic got a different mean
        # than the row above. Subtraction propagates NaN, so an unobservable horizon
        # stays unobservable rather than becoming a value.
        for h in horizons:
            comps[f"h{h}"] = comps[f"h{h}"] - replacement

    rng = np.random.default_rng(seed)
    path = []
    for h in horizons:
        values = comps[f"h{h}"].dropna().to_numpy(dtype=float) if not comps.empty else np.array([])
        survived = values[survived_at[f"h{h}"]] if not comps.empty else values
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
                # carry that. A departed comp is in `values` -- at `-replacement` on the
                # VAR scale -- so the low end reflects attrition rather than hiding it.
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
        departed=departed,
        mode="track" if prior_sgp is not None else "current",
        floor=replacement,
        slot=slot,
    )
