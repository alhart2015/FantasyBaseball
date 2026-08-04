"""One sweep of the whole pool, on both scales, at the longest horizon offered (#321).

`scripts/trajectory_board.py` owned this and nothing under `src/` can import from
`scripts/`, so the web board would have needed its own copy -- two definitions of "what
is this player worth over three years", free to drift. It lives here now and the CLI is a
caller.

WHY A SINGLE SWEEP SERVES EVERY TIMEFRAME. With the start year locked at base+1,
`horizons[0]` is always 1, so the comp mask in `shape_trajectory` is identical whatever
the tuple, and each horizon is fitted independently. The per-year `PathPoint` for a given
horizon therefore comes out bit-identical whether the tuple was `(1,)` or `(1..5)` --
measured across 6,600 values on the live panel, max absolute difference 0.0, and asserted
in `test_a_shorter_range_is_a_prefix_of_the_longest_sweep`. So a shorter end year is a
prefix sum over the cached points, not a refit.

WHY TWO FITS PER PLAYER. `shape_trajectory(replacement=floor)` fits on
`y = max(forward - replacement, 0)`: the floor is baked into the response and clamped at
zero, so raw SGP is NOT `VAR + floor`. For anyone below replacement that formula reports
his SGP as exactly the floor. Honest columns for both scales mean fitting each player
twice, which is why this is an offline job and not a request-time one.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from fantasy_baseball.trajectory.board import BoardRow
from fantasy_baseball.trajectory.shape import prepare, shape_trajectory

#: Bootstrap refits per query. The board reports no SE column, and `se` enters the band
#: only through `spread`, which moves by 0.0006 between 250 draws and 1000 -- so the
#: sweep buys a 4x speedup for precision it does not print. The single-player CLI keeps
#: the higher default, where the SE IS a printed column.
SWEEP_DRAWS = 250

#: Rank movement worth drawing an arrow for. Below this the two rankings are saying the
#: same thing with noise on top. Lives beside `add_ranks`, which produces the two ranks
#: being compared, so the CLI and the web board cannot disagree about which players get
#: flagged as hold-rather-than-start -- the decision this board exists to make.
RANK_MOVE = 5

#: The two scales a board row can be read on. "var" nets out the position-aware waiver
#: floor and clamps at zero; "sgp" is the raw projection.
SCALES = ("var", "sgp")


@dataclass(frozen=True)
class YearPoint:
    """One projected season, on one scale.

    Only the fields a board needs. The full `PathPoint` carries the survivor mean, the
    predictive spread and the SE as well; none of them are shown, and a cached payload
    that stores what it does not render invites a later reader to start trusting them.
    """

    horizon: int
    age: int
    mean: float
    p10: float
    p90: float
    n_effective: float
    #: THIS year's band fell back to the whole cohort's residual scatter. Per horizon so a
    #: one-year view cannot inherit a fallback that happened four years further out.
    band_fell_back: bool


@dataclass(frozen=True)
class SweptPlayer:
    """One player, fitted once per scale, at every horizon the sweep reached."""

    #: The only unique key on a board row. Names are NOT unique even within a pool -- the
    #: live board carries two hitters called Max Muncy -- so anything joining, charting or
    #: linking a row must key on this. #324 needs it to name comps; #284 is the roster
    #: side of the same problem.
    mlbam_id: int
    name: str
    pool: str
    age: int
    slot: str
    floor: float
    #: This season's SGP, paced to a full year. The query the whole fit is built on.
    now: float
    prior: float
    #: Share of the fitting weight near this player's own current season, and whether it
    #: is under `MIN_LOCAL_SUPPORT`. Both are range-independent: they are read off the
    #: comp mask, which `horizons[0] == 1` pins for every range.
    support: float
    extrapolated: bool
    var: tuple[YearPoint, ...]
    #: Empty when the sweep was asked for the VAR scale alone -- the CLI's case, where a
    #: second fit would double a 17s run to serve a column it does not print.
    sgp: tuple[YearPoint, ...]

    def points(self, scale: str) -> tuple[YearPoint, ...]:
        if scale not in SCALES:
            raise ValueError(f"scale must be one of {SCALES}, got {scale!r}")
        return self.var if scale == "var" else self.sgp


def _points(traj: Any) -> tuple[YearPoint, ...]:
    """Observable path points only -- a horizon past what the panel can see carries no
    estimate, and summing it as NaN would poison every reachable year with it."""
    return tuple(
        YearPoint(
            horizon=p.horizon,
            age=p.age,
            mean=p.mean,
            p10=p.p10,
            p90=p.p90,
            n_effective=p.n_effective,
            band_fell_back=p.band_fell_back,
        )
        for p in traj.observable
    )


def sweep_pool(
    rows: Iterable[BoardRow],
    panel: pd.DataFrame,
    kind: str,
    horizons: tuple[int, ...],
    *,
    scales: tuple[str, ...] = SCALES,
    draws: int = SWEEP_DRAWS,
) -> list[SweptPlayer]:
    """Fit every row against one prepared state, once per requested scale.

    `panel` must be the COMPLETE seasons only: an in-progress year averaged in as though
    it were a full one is a systematically low comp. One prepared state per pool; a
    hitter-fitted state cannot price a pitcher and will refuse to try.

    `scales` defaults to both because the cached web board needs both, but each one is a
    separate fit and the cost is linear in them -- ask for `("var",)` when that is all
    you render.
    """
    unknown = set(scales) - set(SCALES)
    if unknown or "var" not in scales:
        raise ValueError(
            f"scales must be a non-empty subset of {SCALES} including 'var', got {scales!r}"
        )
    prepared = prepare(panel, kind=kind, horizons=horizons)
    swept: list[SweptPlayer] = []
    for row in rows:
        fits = {}
        for scale in scales:
            traj, _ = shape_trajectory(
                prepared,
                kind=kind,
                age=row.age,
                sgp=row.sgp,
                prior_sgp=row.prior_sgp,
                horizons=horizons,
                # The raw pass takes no floor AND no slot: `Trajectory.scale` reads the
                # slot, so carrying it would label an unfloored path "var".
                replacement=row.floor if scale == "var" else 0.0,
                slot=row.slot if scale == "var" else None,
                bootstrap_draws=draws,
            )
            fits[scale] = traj
        if not fits["var"].observable:
            continue
        raw = fits.get("sgp")
        swept.append(
            SweptPlayer(
                mlbam_id=row.mlbam_id,
                name=row.name,
                pool=row.pool,
                age=row.age,
                slot=row.slot,
                floor=row.floor,
                now=row.sgp,
                prior=row.prior_sgp,
                support=fits["var"].local_support,
                extrapolated=fits["var"].extrapolated,
                var=_points(fits["var"]),
                sgp=_points(raw) if raw is not None else (),
            )
        )
    return swept


def totals(
    players: Iterable[SweptPlayer], horizons: tuple[int, ...], scale: str = "var"
) -> list[dict]:
    """Collapse the per-year points to one row per player over `horizons`.

    The prefix sum the cached sweep exists to make cheap. Bands are summed the way
    `Trajectory.total` sums them, which assumes the years move together and overstates
    the width if they do not -- stated rather than hidden, and it is the conservative
    direction for a keep-or-cut call.
    """
    wanted = set(horizons)
    scored = []
    for player in players:
        points = [p for p in player.points(scale) if p.horizon in wanted]
        if not points:
            continue
        first = [p for p in points if p.horizon == 1]
        scored.append(
            {
                "id": player.mlbam_id,
                "name": player.name,
                "pool": player.pool,
                "age": player.age,
                "slot": player.slot,
                # THIS season on the scale being read, so the leftmost number and the
                # projections it sits beside mean the same thing. VAR is SGP minus the
                # slot's replacement level -- that is the definition -- and it is NOT
                # clamped at zero here: a player already below his waiver floor is
                # exactly what a keeper reader needs to see, and clamping would render
                # him identical to a replacement-level one. (The PROJECTED var does
                # clamp, inside `shape_trajectory`, so a row can read a negative Now
                # against a 0.0 forecast -- that asymmetry is real and intended.)
                "now": player.now if scale == "sgp" else player.now - player.floor,
                "floor": player.floor,
                "prior": player.prior,
                "total": sum(p.mean for p in points),
                "p10": sum(p.p10 for p in points),
                "p90": sum(p.p90 for p in points),
                "years": len(points),
                "n_eff": min(p.n_effective for p in points),
                "support": player.support,
                "extrapolated": player.extrapolated,
                # Scoped to the range on screen, not latched across the whole fit.
                "band_fell_back": any(p.band_fell_back for p in points),
                # NEXT season alone, for the second ranking. A one-year board and a
                # multi-year board answer different questions -- who helps now versus who
                # is worth holding -- and the gap between a player's two ranks is the
                # keeper decision in one number.
                "next": first[0].mean if first else float("nan"),
                "by_year": [{"horizon": p.horizon, "age": p.age, "mean": p.mean} for p in points],
            }
        )
    return scored


#: Payload schema version. Bumped when the shape changes incompatibly; a reader that
#: finds a version it does not know refuses the blob rather than mis-indexing the compact
#: point arrays into confidently wrong numbers.
PAYLOAD_VERSION = 1

#: Decimals kept on a cached point. Not chosen for display -- the board prints one -- but
#: for the RANKING, which is a sum of these and is what the page sorts on. The pool is
#: dense: 127 of 562 adjacent hitters sit within 0.001 VAR of each other, and some tie
#: exactly. Rounding therefore reorders rows. Measured, hitters at 1..3:
#:
#:     round to 2:  267 of 563 rows change rank, worst move 66
#:     round to 3:   30 of 563 rows change rank, worst move  2
#:     round to 4:    8 of 563 rows change rank, worst move  1
#:     round to 6:    0 of 563 rows change rank, worst move  0
#:
#: So 6, where the cached board and a direct sweep agree on every position. The cost is
#: ~3 characters per number against 3 decimals, on a blob prod already carries 6x of.
_PRECISION = 6


def _pack(point: YearPoint) -> list[float]:
    """A point as a positional array. `age` is dropped -- it is `player.age + horizon`,
    and a stored copy is a second source for a derived fact."""
    return [
        point.horizon,
        round(point.mean, _PRECISION),
        round(point.p10, _PRECISION),
        round(point.p90, _PRECISION),
        round(point.n_effective, 1),
        int(point.band_fell_back),
    ]


def _unpack(packed: list[float], age: int) -> YearPoint:
    horizon = int(packed[0])
    return YearPoint(
        horizon=horizon,
        age=age + horizon,
        mean=float(packed[1]),
        p10=float(packed[2]),
        p90=float(packed[3]),
        n_effective=float(packed[4]),
        band_fell_back=bool(packed[5]),
    )


def to_payload(players: Iterable[SweptPlayer], **meta: Any) -> dict:
    """Serialize a sweep for the KV. `meta` carries the vintage the reader must show."""
    return {
        "version": PAYLOAD_VERSION,
        **meta,
        "players": [
            {
                "id": p.mlbam_id,
                "name": p.name,
                "pool": p.pool,
                "age": p.age,
                "slot": p.slot,
                "floor": round(p.floor, _PRECISION),
                "now": round(p.now, _PRECISION),
                "prior": round(p.prior, _PRECISION),
                "support": round(p.support, 4),
                "extrapolated": int(p.extrapolated),
                "var": [_pack(y) for y in p.var],
                "sgp": [_pack(y) for y in p.sgp],
            }
            for p in players
        ],
    }


def from_payload(payload: dict) -> list[SweptPlayer]:
    """Rebuild the sweep from a cached payload.

    Deliberately reconstructs `SweptPlayer` rather than letting the web layer read the
    dicts directly, so the page and the CLI collapse a range through the same `totals()`
    and cannot disagree about what a three-year VAR is.
    """
    version = payload.get("version")
    if version != PAYLOAD_VERSION:
        raise ValueError(
            f"trajectory board payload is version {version!r}, this build reads "
            f"{PAYLOAD_VERSION}; re-run scripts/push_trajectory_board.py"
        )
    players = []
    for row in payload["players"]:
        age = int(row["age"])
        players.append(
            SweptPlayer(
                mlbam_id=int(row["id"]),
                name=row["name"],
                pool=row["pool"],
                age=age,
                slot=row["slot"],
                floor=float(row["floor"]),
                now=float(row["now"]),
                prior=float(row["prior"]),
                support=float(row["support"]),
                extrapolated=bool(row["extrapolated"]),
                var=tuple(_unpack(y, age) for y in row["var"]),
                sgp=tuple(_unpack(y, age) for y in row["sgp"]),
            )
        )
    return players


def add_ranks(scored: list[dict]) -> None:
    """Stamp each row with BOTH rankings, over the whole scored pool.

    Ranks are computed once over everyone and then carried, so a per-team view shows a
    player's LEAGUE rank rather than his rank among his own teammates -- the latter would
    make every team's best player look like a 1.
    """
    for key, rank_field in (("total", "rank_total"), ("next", "rank_next")):

        def sort_key(row: dict, k: str = key) -> tuple[float, str]:
            # A NaN sorts LAST rather than poisoning the comparison: `next` is NaN
            # whenever horizon 1 is out of range, and one unrankable row must not decide
            # where every other one lands.
            value = float(row[k])
            return (float("inf") if np.isnan(value) else -value, row["name"])

        for i, row in enumerate(sorted(scored, key=sort_key), start=1):
            row[rank_field] = i
