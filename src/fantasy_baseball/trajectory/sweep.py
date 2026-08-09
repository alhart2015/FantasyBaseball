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

WHY ONE FIT PER PLAYER. There used to be two, one per scale, because
`shape_trajectory(replacement=floor)` clamped its response at zero and a clamped fit is
not a shifted one. Removing that clamp (#331) made `y_var = y_sgp - floor` affine, so the
VAR fit moves only its intercept and every VAR number is the raw number minus the slot's
floor -- exactly, to floating-point noise, `mean` / `p10` / `p90` alike. The second fit
was therefore computing a shift, at the price of doubling the sweep. It is gone, and
`SweptPlayer.points("var")` derives the scale instead.

That is also what makes the VAR/SGP toggle trustworthy. Two players in the same slot
share a floor, so subtracting it cannot reorder them -- the property #331 was opened
against, now true by construction rather than by assertion. The toggle still reorders
ACROSS slots, which is the whole point of it: a catcher's floor is not a UTIL's.
"""

from __future__ import annotations

import math
from collections.abc import Iterable
from dataclasses import dataclass, replace
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

#: The two scales a board row can be read on. "sgp" is the raw projection; "var" is that
#: minus the slot's position-aware waiver floor, and is NOT clamped -- a player projected
#: under his floor reads negative, which is the information a keeper call needs (#331).
SCALES = ("var", "sgp")


def var_offset(floor: float, scale: str) -> float:
    """What to subtract from a raw SGP number to read it on `scale`.

    THE definition of the VAR scale, in ONE place. It was spelled twice -- once for the
    fitted path and again inline in `totals` for the Now column -- and two spellings of
    this rule is the drift #331 is about: the board printed an unclamped Now beside a
    clamped forecast, which was only possible because the two columns did not get their
    scale from the same site. `SweptPlayer.offset` delegates here rather than
    re-deriving it, and the trajectory-chart view (`build_player_view`, #324) reaches
    it through that method rather than spelling the rule a third time.
    """
    if scale not in SCALES:
        raise ValueError(f"scale must be one of {SCALES}, got {scale!r}")
    return 0.0 if scale == "sgp" else floor


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
    """One player, fitted ONCE on raw SGP, at every horizon the sweep reached."""

    #: HALF the unique key on a board row, which is `(mlbam_id, pool)`. Names are NOT
    #: unique even within a pool -- the live board carries two hitters called Max Muncy --
    #: and the id alone is not unique either, because a two-way player is produced ONCE
    #: PER POOL (see `test_a_two_way_player_keeps_one_line_per_pool`). Anything joining,
    #: charting or linking a row must key on the PAIR: this comment used to call the id
    #: "the only unique key", and that belief is what keyed the chart extras on the bare
    #: id, so Ohtani's hitter row carried his pitching career line and pitcher comps.
    #: `chart_key` is where that pair is now spelled, once, for writer and reader alike.
    #: #324 needs the id to name comps; #284 is the roster side of the same problem.
    mlbam_id: int
    name: str
    pool: str
    age: int
    slot: str
    floor: float
    #: This season's SGP on its full-season line -- realized while the season is over,
    #: season-to-date plus a rest-of-season projection while it is not (`ros_anchor`).
    #: The query the whole fit is built on.
    now: float
    prior: float
    #: Share of the fitting weight near this player's own current season, and whether it
    #: is under `MIN_LOCAL_SUPPORT`. Both are range-independent: they are read off the
    #: comp mask, which `horizons[0] == 1` pins for every range.
    support: float
    extrapolated: bool
    #: The RAW SGP fit, and the only path stored or run. See the module docstring.
    sgp: tuple[YearPoint, ...]

    def offset(self, scale: str) -> float:
        """What to subtract from a raw SGP number to read it on `scale`. Delegates to
        `var_offset`, the module-level home for this rule -- see its docstring."""
        return var_offset(self.floor, scale)

    def points(self, scale: str) -> tuple[YearPoint, ...]:
        """The fitted path on `scale`. "var" is derived, never stored."""
        floor = self.offset(scale)
        if floor == 0.0:
            return self.sgp
        # Only the LEVELS move. `n_effective` and `band_fell_back` describe the fit, which
        # is the same fit; the widths (`p90 - p10`) are unchanged because both edges shift
        # by the same floor.
        return tuple(
            replace(p, mean=p.mean - floor, p10=p.p10 - floor, p90=p.p90 - floor) for p in self.sgp
        )


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
    draws: int = SWEEP_DRAWS,
) -> list[SweptPlayer]:
    """Fit every row against one prepared state, once, on the raw SGP scale.

    `panel` must be the COMPLETE seasons only: an in-progress year averaged in as though
    it were a full one is a systematically low comp. One prepared state per pool; a
    hitter-fitted state cannot price a pitcher and will refuse to try.

    Both scales come out of this one fit. There is no `scales` argument any more because
    there is nothing left to skip: the CLI once passed `("var",)` to halve a 17s run, and
    the run is now that half either way.
    """
    prepared = prepare(panel, kind=kind, horizons=horizons)
    swept: list[SweptPlayer] = []
    for row in rows:
        # NO floor and NO slot. The floor is applied by `points("var")`, and `slot` would
        # label this path "var" on `Trajectory.scale` while it carries raw SGP.
        traj, _ = shape_trajectory(
            prepared,
            kind=kind,
            age=row.age,
            sgp=row.sgp,
            prior_sgp=row.prior_sgp,
            horizons=horizons,
            bootstrap_draws=draws,
        )
        if not traj.observable:
            continue
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
                support=traj.local_support,
                extrapolated=traj.extrapolated,
                sgp=_points(traj),
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
                # THIS season, netted through the SAME `offset` the fitted path above
                # used, so the leftmost number and the projections beside it cannot end
                # up on different scales. NOTE that `prior` below is deliberately NOT
                # netted, which is a real inconsistency on the CLI's VAR board -- see the
                # comment there.
                "now": player.now - player.offset(scale),
                "floor": player.floor,
                # RAW, on both scales, unlike `now`. That is an inconsistency rather than
                # a decision -- `scripts/trajectory_board.py` prints the two side by side
                # under a VAR header, so on the VAR board `now` is netted and `prior`
                # beside it is not. Left alone here because netting it changes a rendered
                # number and this year's floor is a questionable thing to charge against
                # last year's production. Tracked as #333 -- named here so the deferral
                # is checkable rather than a promise that a reader has to take on faith.
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


def _pack(point: YearPoint) -> dict[str, float]:
    """A point as named fields. `age` is dropped -- it is `player.age + horizon`, and a
    stored copy is a second source for a derived fact.

    These were positional arrays, which cost ~340 KB less and bought a whole schema
    register to go with it: a positional point that changes shape does not fail, it
    indexes to the wrong field and renders confident nonsense, so the blob carried a
    `PAYLOAD_VERSION` and the reader refused anything that did not match exactly. That
    refusal is strict in BOTH directions -- an old build rejects a new blob just as a
    new build rejects an old one -- so every bump had a mandatory window where the
    deployed page was down, and it could not be closed by ordering the deploy and the
    push, only shortened.

    Named keys delete the failure mode rather than detecting it: a missing or renamed
    field raises on its own name. The register went with it. The cost is the size, on a
    blob with no ceiling enforced anywhere in this repo -- ~420 KB against ~760 KB, both
    far under the plan limit. Clarity now, packing later if it is ever measured to
    matter.
    """
    return {
        "horizon": point.horizon,
        "mean": round(point.mean, _PRECISION),
        "p10": round(point.p10, _PRECISION),
        "p90": round(point.p90, _PRECISION),
        "n_effective": round(point.n_effective, 1),
        "band_fell_back": int(point.band_fell_back),
    }


def _unpack(packed: dict[str, float], age: int) -> YearPoint:
    if not isinstance(packed, dict):
        # The blob deployed when this landed is positional, and `packed[0]` on a list
        # would read as a TypeError about integer indices -- true, and useless. Say
        # what to do instead. Self-limiting: once prod is re-pushed it never fires,
        # and unlike a version register nothing has to be remembered to keep it honest.
        #
        # ONE guard, because there is one reader. The message used to be a module
        # constant shared with `web.trajectory_view.build_player_view`, which unpacked
        # `row["sgp"]` itself and so needed its own copy of this check; that reader now
        # goes through `player_from_row` and inherits this one (#324).
        raise ValueError(
            "trajectory board payload stores points as positional arrays, which this "
            "build no longer reads; re-run scripts/push_trajectory_board.py"
        )
    horizon = int(packed["horizon"])
    return YearPoint(
        horizon=horizon,
        age=age + horizon,
        mean=float(packed["mean"]),
        p10=float(packed["p10"]),
        p90=float(packed["p90"]),
        n_effective=float(packed["n_effective"]),
        band_fell_back=bool(packed["band_fell_back"]),
    )


def to_payload(players: Iterable[SweptPlayer], **meta: Any) -> dict:
    """Serialize a sweep for the KV. `meta` carries the vintage the reader must show.

    WHAT THE CHART NEEDS IS NOT IN HERE. Career history and comps used to be merged into
    every row of this payload, which took it from 762 KB to 1,861 KB while only
    `build_player_view` read them -- so the league board and the By-team view, the two
    default views, each carried ~1.1 MB they never render. They live in their own blob
    now (`to_chart_payload`, #344) and this payload is exactly what all three views use.
    """
    return {
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
                # RAW only; VAR is derived on read. See the module docstring.
                "sgp": [_pack(y) for y in p.sgp],
            }
            for p in players
        ],
    }


def chart_key(mlbam_id: int, pool: str) -> str:
    """The chart blob's key for ONE board row, spelled once for writer and reader.

    `(mlbam_id, pool)` is the unique key on a board row -- see `SweptPlayer.mlbam_id` --
    and JSON object keys must be strings, so the pair is joined here rather than at
    either end. A bare id is the defect this collapses two ways: a two-way player is
    swept ONCE PER POOL, so keying on the id alone let the pitcher pass overwrite the
    hitter's entry and Ohtani's hitter row rendered his pitching career and pitcher
    comps.
    """
    return f"{mlbam_id}:{pool}"


def to_chart_payload(
    extras: dict[tuple[int, str], dict],
    *,
    generated_at: str,
    careers: dict[str, list] | None = None,
) -> dict:
    """Serialize the per-player chart extras -- career history and comps -- for the KV.

    A SECOND blob beside the board, not a section of it (#344). Only the player view
    reads it; the league board and the By-team view never do, which is the whole point
    of the split.

    `generated_at` is the BOARD'S stamp, passed in rather than taken here, so the two
    blobs a single push writes carry one identical vintage. The player view compares
    them and refuses to draw extras that do not match the board it is rendering -- two
    keys can be refreshed independently, and a stale career line under a fresh
    projection is wrong in a way that renders perfectly.

    These are the fields the SWEEP does not produce: they need the panel and the people
    cache rather than the fit, which is why they are assembled by the push script and
    passed in here rather than hung off `SweptPlayer`.

    `careers` is the per-comp arc map (#346), keyed the same way `players` is. It is
    optional and always WRITTEN, empty or not: a key whose presence depends on the
    writer's vintage makes every reader carry a second branch for the same state.
    `careers or {}` is safe where CLAUDE.md warns against `x or default` -- the value is
    a dict, not a number, and an empty one means exactly what `None` does here.
    """
    return {
        "generated_at": generated_at,
        "players": {chart_key(mlbam_id, pool): data for (mlbam_id, pool), data in extras.items()},
        # COMP CAREERS, deduped across the whole board rather than nested under each
        # comp. The same arc is a comp for many players -- nesting writes it once per
        # (player, comp) pair, roughly 5x the entries for the same information. Keyed
        # by `chart_key` like `players`, so a two-way comp keeps one career per pool.
        "careers": dict(careers or {}),
    }


def player_from_row(row: dict) -> SweptPlayer:
    """ONE cached payload row as a `SweptPlayer`.

    Split out of `from_payload` for the consumers that want a single player rather than
    the pool -- the trajectory chart (#324) resolves one name and has no use for the
    other 1,168 fits. Before this existed that view unpacked `row["sgp"]` itself, which
    made it a second reader of the point schema: it re-spelled `age + horizon`, re-spelled
    the floor subtraction `SweptPlayer.points` already owns, and carried its own copy of
    `_unpack`'s positional-blob guard to keep a legacy blob from 500ing it.
    """
    age = int(row["age"])
    return SweptPlayer(
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
        sgp=tuple(_unpack(y, age) for y in row["sgp"]),
    )


def from_payload(payload: dict) -> list[SweptPlayer]:
    """Rebuild the sweep from a cached payload.

    Deliberately reconstructs `SweptPlayer` rather than letting the web layer read the
    dicts directly, so the page and the CLI collapse a range through the same `totals()`
    and cannot disagree about what a three-year VAR is.

    A payload this build cannot read raises out of `_unpack`, on the field that is
    actually wrong, rather than being screened by a schema register up front -- see
    `_pack` for why that register existed and why named fields replaced it.
    """
    return [player_from_row(row) for row in payload["players"]]


def rank_move(row: dict) -> int:
    """The hold-vs-start arrow for a ranked row, or 0 where it would be meaningless.

    The THRESHOLD is only half the rule, and shipping only the threshold is what let the
    CLI and the web board disagree about which players get flagged -- the decision this
    board exists to make. Both halves live here so a caller cannot take one without the
    other.

    Withheld when there is **no next-year estimate**. `next` is NaN when horizon 1 is
    unobservable, and `add_ranks` sorts NaN last so one unrankable row cannot decide where
    the others land. That pairs a last-place `rank_next` with a real `rank_total` and
    renders as the strongest HOLD signal on the board, produced by an absence rather than
    a strength.

    There was a second guard, for rows reading 0.0 on BOTH rankings. It existed because
    VAR clamped at zero, which collapsed every below-replacement player onto an identical
    all-zero row -- 469 of them on a live-shaped 1,169-row pool, of which 432 drew an
    arrow off nothing but the offset between the two zero-blocks. #331 removed the clamp,
    so those rows now carry distinct negative values and rank honestly, and an exact 0.0
    on both is no longer a systematic block but a coincidence at the payload's sixth
    decimal. The guard went with the clamp that made it necessary.
    """
    nxt = row["next"]
    if math.isnan(nxt):
        return 0
    move = row["rank_next"] - row["rank_total"]
    return move if abs(move) >= RANK_MOVE else 0


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
