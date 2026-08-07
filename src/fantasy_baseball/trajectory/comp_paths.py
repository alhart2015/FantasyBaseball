"""Historical paths closest to a predicted one.

Given a shape-predicted five-year path, find the player-seasons whose REALIZED path
minimizes RMSE against it. This asks a different question from the comp matchers in
`comps.py`: those pick a cohort that looks similar at the STARTING point and average
what it did, while this picks whole forward paths that match the prediction -- which is
the thing a chart of that prediction actually draws.

THE RESULT IS SELECTED ON THE OUTCOME. These are the paths that happened to land closest
out of ~1,200. That makes them a fair illustration of what this shape looked like when it
played out, and it makes them NOT evidence for the prediction. A consumer that draws them
without the p10-p90 band beside them is making the forecast look more certain than it is.

Everything needed is already on `Prepared`: `forward[h]` is realized SGP for every history
row at +h, with `age`, `season` and `mlbam_id` alongside. The match is a broadcast subtract
and a sort -- about 0.2 ms for one query against a live hitter panel.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

from fantasy_baseball.trajectory.shape import Prepared


@dataclass(frozen=True)
class CompPath:
    """One historical season whose forward path is close to the prediction."""

    mlbam_id: int
    season: int
    #: Root mean squared error against the predicted path, over every horizon.
    rmse: float
    #: The REALIZED forward path, one value per horizon, ascending.
    path: tuple[float, ...]


def closest_paths(
    prepared: Prepared,
    predicted: Sequence[float],
    age: int,
    n: int,
) -> list[CompPath]:
    """The `n` realized paths closest to `predicted`, best first.

    Ids, never names: naming needs the people cache, and keeping it out of here is what
    lets this be tested against a hand-built `Prepared` with no data files at all.
    """
    horizons = tuple(sorted(prepared.horizons))
    target = np.asarray(predicted, dtype=float)
    if target.size != len(horizons):
        raise ValueError(
            f"predicted has {target.size} values but prepared carries "
            f"{len(horizons)} horizons {horizons}"
        )

    # EXACT age, and every forward year realized. `forward` stores a real 0.0 for "out of
    # the league", which is indistinguishable from "has not happened yet" -- so the
    # censoring has to come from the season, not the value. Horizons ascend, so clearing
    # the longest clears them all.
    candidates = np.flatnonzero(
        (prepared.age == float(age)) & (prepared.season + horizons[-1] <= prepared.last)
    )
    if candidates.size == 0:
        return []

    paths = np.column_stack([prepared.forward[h][candidates] for h in horizons])
    rmse = np.sqrt(((paths - target) ** 2).mean(axis=1))

    # Sorted on the full key, not just rmse: two identical paths would otherwise swap
    # between reads on nothing but row order.
    order = sorted(
        range(candidates.size),
        key=lambda i: (
            float(rmse[i]),
            int(prepared.mlbam_id[candidates[i]]),
            int(prepared.season[candidates[i]]),
        ),
    )
    return [
        CompPath(
            mlbam_id=int(prepared.mlbam_id[candidates[i]]),
            season=int(prepared.season[candidates[i]]),
            rmse=float(rmse[i]),
            path=tuple(float(v) for v in paths[i]),
        )
        for i in order[:n]
    ]
