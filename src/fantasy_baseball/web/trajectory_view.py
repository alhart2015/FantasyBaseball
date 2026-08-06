"""View model for the trajectory board page (#321).

Everything between the cached sweep and the template: pick a scale, pick a timeframe,
collapse the per-year points to it, rank and filter. No Flask, so it is testable
directly.

The page is a PURE READER of `cache:trajectory_board`, written offline by
`scripts/push_trajectory_board.py`. It cannot compute the board itself -- the fit needs
`data/trajectory/` and `data/cache/keeper_skills`, both gitignored and so absent on
Render. The board therefore does NOT move with a dashboard refresh, which is why
`Board.meta` carries the vintage and the template prints it.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from fantasy_baseball.trajectory.comps import MIN_LOCAL_SUPPORT
from fantasy_baseball.trajectory.sweep import (
    SCALES,
    add_ranks,
    from_payload,
    rank_move,
    require_supported_version,
    totals,
)
from fantasy_baseball.utils.name_utils import normalize_name

#: Default rows on the league-wide board. Deliberately not the CLI's 25: a web table
#: scrolls where a terminal dump does not.
DEFAULT_TOP = 50


@dataclass(frozen=True)
class Board:
    """One rendered board: the rows, the controls that produced them, and the vintage."""

    rows: list[dict]
    #: Rows in THIS view at this timeframe, before the top-N slice -- so the page can
    #: say "50 of 606" rather than implying 50.
    scored: int
    #: Rows scored league-wide, which is what the rank column counts against. Equal to
    #: `scored` unless a pool filter is on; separate because `add_ranks` deliberately
    #: ranks the whole pool, so a pitcher view legitimately shows ranks past its own
    #: count and the page must not print the two as if they were one number.
    ranked: int
    base_season: int
    end_year: int
    end_years: list[int]
    pool: str
    #: Rows shown, or "all". A magic large number instead would silently truncate: the
    #: pool is 1,169 and the obvious ceiling to reach for is 1,000.
    top: int | str
    #: Which fit the whole table is showing: "var" or "sgp".
    scale: str
    #: True when a live roster read returned SOMETHING. An empty result counts as a
    #: failure: `live_rosters` returns [] without raising on a missing blob, so an empty
    #: set cannot be told apart from an unreachable Upstash, and the page must not claim
    #: "you own none of these" on the strength of it.
    has_rosters: bool = False
    #: Season labels for the per-year columns, e.g. [2027, 2028, 2029]. Empty for a
    #: single-year board, where a breakout column would just repeat the total.
    year_columns: list[int] = field(default_factory=list)
    meta: dict = field(default_factory=dict)

    @property
    def span(self) -> str:
        """The timeframe as SEASONS -- "2027-29", not "3-year". Not what a keeper thinks
        in, and the CLI header made the same choice."""
        start = self.base_season + 1
        return f"{start}" if self.end_year == start else f"{start}-{str(self.end_year)[-2:]}"


def _clamp(value: Any, low: int, high: int, default: int) -> int:
    """A query param as an int inside [low, high]. A junk or out-of-range value falls
    back to the default rather than 500ing -- these arrive from a URL a user can edit."""
    try:
        return max(low, min(high, int(value)))
    except (TypeError, ValueError):
        return default


#: Derived state for ONE payload vintage, as a single immutable triple:
#: (vintage, parsed players, {(horizons, scale): ranked rows}).
#:
#: The cached blob is an immutable offline artifact -- it changes only when
#: `push_trajectory_board.py` runs -- so deriving it once per vintage is safe, and
#: `generated_at` (plus the shape fields, since two payloads can share a timestamp) is
#: the invalidation key. At most `max_horizon` x 2 entries per generation, so the map is
#: bounded without an eviction policy.
#:
#: WHY ONE TUPLE RATHER THAN THREE MODULE GLOBALS. The ranked map is bound to the
#: generation that owns it. A thread that read the state before a push lands writes its
#: rows into THAT generation's map, which by then is orphaned -- it cannot reach into the
#: new one. With three independent globals it could: the vintage swap and the row write
#: were separate steps, so a thread still holding the pre-push players could store them
#: under the post-push vintage and pin the old board for the life of the process, while
#: the page printed the new push's timestamp beside it.
#:
#: Still lock-free. Rebinding `_STATE` is a single reference assignment, and every reader
#: takes ONE snapshot into a local. Two requests racing a cold cache both derive and one
#: rebind wins; the values are equal within a generation, so that race costs a wasted
#: derivation. Rows are never mutated after `add_ranks` -- `build_board` copies each into
#: a new dict -- which is what makes sharing them across requests safe.
_STATE: tuple[tuple, list, dict[tuple, list[dict]]] | None = None


def clear_board_cache() -> None:
    """Drop the derived-state cache. For tests, and for anything needing a cold read."""
    global _STATE
    _STATE = None


def _derive(payload: dict, horizons: tuple[int, ...], scale: str) -> list[dict]:
    """Parse and rank, with no caching. The uncacheable path."""
    rows = totals(from_payload(payload), horizons, scale)
    add_ranks(rows)
    return rows


def _ranked_rows(payload: dict, horizons: tuple[int, ...], scale: str) -> list[dict]:
    """Ranked rows for this payload/timeframe/scale, derived once per vintage."""
    global _STATE
    if not payload.get("generated_at"):
        # No vintage, no cache. Every real payload carries one -- push_trajectory_board
        # always stamps it -- so this is a hand-built payload, and inventing a key for it
        # would let two unrelated fixtures share derived rows.
        return _derive(payload, horizons, scale)

    vintage = (
        str(payload["generated_at"]),
        payload.get("base_season"),
        payload.get("max_horizon"),
        len(payload.get("players", ())),
    )
    state = _STATE  # ONE snapshot; everything below reads from it, not from the global.
    if state is None or state[0] != vintage:
        state = (vintage, from_payload(payload), {})
        _STATE = state

    _, players, ranked = state
    key = (horizons, scale)
    if key not in ranked:
        rows = totals(players, horizons, scale)
        add_ranks(rows)
        ranked[key] = rows
    return ranked[key]


def _year_cells(by_year: list[dict], horizons: tuple[int, ...]) -> list[float | None]:
    """One cell per rendered year column, keyed on horizon rather than list position."""
    if len(horizons) < 2:
        return []
    means = {c["horizon"]: c["mean"] for c in by_year}
    return [means.get(h) for h in horizons]


def build_board(
    payload: dict,
    *,
    end: Any = None,
    pool: str = "both",
    top: Any = None,
    scale: str = "var",
    mine: set[tuple[str, str]] | None = None,
) -> Board:
    """Collapse the cached sweep to one timeframe and rank it.

    START IS LOCKED at `base_season + 1`; only the end year moves. That is what makes the
    dropdown free -- `horizons[0]` stays 1, so every range reads the same cached points
    (see `trajectory.sweep`) -- and it is also what keeps `rank_next` meaningful, since
    `next` is only populated when horizon 1 is in range.
    """
    require_supported_version(payload)
    base = int(payload["base_season"])
    max_horizon = int(payload["max_horizon"])
    end_years = [base + h for h in range(1, max_horizon + 1)]

    end_year = _clamp(end, end_years[0], end_years[-1], end_years[0])
    show_all = str(top).lower() == "all"
    top_n = None if show_all else _clamp(top, 1, 5000, DEFAULT_TOP)
    if pool not in ("both", "hitter", "pitcher"):
        pool = "both"
    if scale not in SCALES:
        scale = "var"
    horizons = tuple(range(1, end_year - base + 1))

    # ONE SCALE ON SCREEN. The two scales differ by the slot's floor (#331 made that
    # exact), and a board mixing them forces the reader to track which number belongs to
    # which -- the rank, the band and the per-year cells then have to agree about a scale
    # that is only implicit. The toggle makes it explicit and the ambiguity structurally
    # impossible.
    #
    # RANKED OVER THE WHOLE POOL, then filtered. A pitcher-only view shows LEAGUE ranks,
    # so its top row can read #7 -- correct, and the same rule #322/#323 depend on, where
    # ranking within a subset would make every team's best player a #1.
    ranked_rows = _ranked_rows(payload, horizons, scale)

    # (normalized name, pool) is the only key a roster blob can be joined on -- they
    # carry no mlbam_id (#284). It is NOT unique: the live board has two hitters called
    # Max Muncy. So count the rows each key matches and mark a multi-hit as unsure rather
    # than silently putting a player the reader does not own on his keeper shortlist.
    owned = mine or set()
    key_counts: dict[tuple[str, str], int] = {}
    for row in ranked_rows:
        k = (normalize_name(row["name"]), row["pool"])
        key_counts[k] = key_counts.get(k, 0) + 1

    rows = []
    for row in ranked_rows:
        if pool != "both" and row["pool"] != pool:
            continue
        move = rank_move(row)
        key = (normalize_name(row["name"]), row["pool"])
        is_mine = key in owned
        rows.append(
            {
                **row,
                "mine": is_mine,
                "mine_ambiguous": is_mine and key_counts[key] > 1,
                # The MOVE between the two ranks is the keeper signal in one number: a
                # player far better over the range than next year is who you hold rather
                # than who you start. See `_rank_move` for when it is withheld.
                "rank_move": move,
                # Per-year cells ALIGNED TO THE YEAR COLUMNS by horizon, one entry each,
                # None where this player has no point for that year. Each horizon is a
                # separate fit, so `by_year` carries no guarantee of being a prefix;
                # rendering it positionally and padding the tail would put a year-2
                # figure under the year-1 header for any path with a hole. Defensive --
                # no such gap has been reproduced from the model -- but the alignment is
                # free here and the assumption is not worth carrying in the template.
                "year_cells": _year_cells(row["by_year"], horizons),
            }
        )

    scored = len(rows)
    ranked = len(ranked_rows)
    # `add_ranks` already ordered by total descending with a name tie-break, so ranking
    # IS the sort order -- no second sort key, and nothing to get wrong about NaNs.
    rows.sort(key=lambda r: r["rank_total"])

    return Board(
        rows=rows if top_n is None else rows[:top_n],
        scored=scored,
        ranked=ranked,
        base_season=base,
        end_year=end_year,
        end_years=end_years,
        pool=pool,
        top="all" if top_n is None else top_n,
        scale=scale,
        # An EMPTY read is not a successful one. `live_rosters` returns [] without
        # raising when the roster blobs are absent -- its module docstring calls that out
        # as reading like "you own nobody" rather than an error -- so treating `set()` as
        # success suppressed the warning AND highlighted nothing, which is the one
        # combination this field exists to prevent.
        has_rosters=bool(mine),
        # A per-year breakout only earns its columns once the range spans more than one.
        year_columns=[base + h for h in horizons] if len(horizons) > 1 else [],
        meta={
            "generated_at": payload.get("generated_at"),
            "panel_vintage": payload.get("panel_vintage"),
            "season_elapsed": payload.get("season_elapsed"),
            "min_sgp": payload.get("min_sgp"),
            "floors": payload.get("floors", {}),
            # Who is NOT on the board. A shortened board reads as "these are the best
            # players" when it is "these are the ones the model can price", and the
            # larger exclusion is the silent one: a player with no current-season line
            # was never a candidate, so he is absent with no row and no flag.
            "excluded": payload.get("excluded", {}),
            # The RULE behind the (!) flag, not just the verdict. It is a tuned number
            # with a measured table behind it and an open issue (#310) to change the
            # estimator it guards, so the page must not restate it as prose: the CLI
            # renders it from the constant and a hardcoded template string would say
            # "under 10%" about rows now flagged at something else.
            "min_local_support": MIN_LOCAL_SUPPORT,
        },
    )
