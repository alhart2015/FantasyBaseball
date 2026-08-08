"""Historical paths closest to a predicted one.

Given a shape-predicted five-year path, find the player-seasons whose REALIZED path
minimizes RMSE against it. This asks a different question from the comp matchers in
`comps.py`: those pick a cohort that looks similar at the STARTING point and average
what it did, while this picks whole forward paths that match the prediction -- which is
the thing a chart of that prediction actually draws.

THE RESULT IS SELECTED ON THE OUTCOME. These are the paths that happened to land closest
out of ~1,200. That makes them a fair illustration of what this shape looked like when it
played out, and it makes them NOT evidence for the prediction. Any surface presenting
them AS the forecast's range must draw the p10-p90 band beside them, or it makes the
forecast look more certain than it is -- which is why the main chart on /trajectory
carries the band and the comps are thin and faint on it. The per-comp career cards below
it (#346) are a different question -- what this player's whole arc looked like, and where
the match sits in it -- and carry no band because they make no claim about the spread.

Everything needed is already on `Prepared`: `forward[h]` is realized SGP for every history
row at +h, with `age`, `season` and `mlbam_id` alongside. The match is a broadcast subtract
and a sort -- about 0.2 ms for one query against a live hitter panel.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

from fantasy_baseball.trajectory.shape import Prepared

#: Comps per player: what `push_trajectory_board.py` STORES and what the chart's `n`
#: control may ASK FOR. One number, because they are one requirement -- the blob is
#: built hours earlier by another process, so the control can only ever slice what was
#: stored, and a ceiling above the stored count asks for comps that do not exist.
#:
#: Ten, not the five the chart draws by default: every legal `n` has to be servable.
#: Costs ~370 bytes a player over five. It lives HERE, beside `closest_paths`, because
#: this is the module both sides go through; it was defined twice, once per side, and
#: kept honest by a test asserting the two literals were equal.
MAX_COMPS = 10


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
    #
    # `lexsort` rather than `sorted` with a key: only the top n (at most `MAX_COMPS`)
    # survive, but the whole candidate pool has to be ordered to find them either way,
    # and a Python-level key function pays an interpreter round trip and three scalar
    # unboxes per candidate to do it. Measured over the 1,169 calls one push makes:
    # 626 ms with the lambda, 157 ms here. `lexsort` reads its keys LAST-IS-PRIMARY, so
    # this tuple is the same (rmse, mlbam_id, season) priority spelled backwards.
    order = np.lexsort((prepared.season[candidates], prepared.mlbam_id[candidates], rmse))
    return [
        CompPath(
            mlbam_id=int(prepared.mlbam_id[candidates[i]]),
            season=int(prepared.season[candidates[i]]),
            rmse=float(rmse[i]),
            path=tuple(float(v) for v in paths[i]),
        )
        for i in order[:n]
    ]
