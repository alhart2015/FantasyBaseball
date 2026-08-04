"""View model for the trajectory board page (#321).

Everything between the cached sweep and the template: pick a timeframe, collapse the
per-year points to it, rank, filter, sort. No Flask, so it is testable directly.

The page is a PURE READER of `cache:trajectory_board`, written offline by
`scripts/push_trajectory_board.py`. It cannot compute the board itself -- the fit needs
`data/trajectory/` and `data/cache/keeper_skills`, both gitignored and so absent on
Render. The board therefore does NOT move with a dashboard refresh, which is why
`Board.meta` carries the vintage and the template prints it.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from fantasy_baseball.trajectory.sweep import SCALES, add_ranks, from_payload, totals

#: Default rows on the league-wide board. Deliberately not the CLI's 25: a web table
#: scrolls where a terminal dump does not.
DEFAULT_TOP = 50

#: Rank movement worth drawing an arrow for. Below this the two rankings are saying the
#: same thing with noise on top.
RANK_MOVE = 5


@dataclass(frozen=True)
class Board:
    """One rendered board: the rows, the controls that produced them, and the vintage."""

    rows: list[dict]
    #: Every row scored at this timeframe, before the top-N slice -- the denominator the
    #: rank column is against, so the page can say "50 of 1169" rather than implying 50.
    scored: int
    base_season: int
    end_year: int
    end_years: list[int]
    pool: str
    #: Rows shown, or "all". A magic large number instead would silently truncate: the
    #: pool is 1,169 and the obvious ceiling to reach for is 1,000.
    top: int | str
    sort: str
    hide_unsupported: bool
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


def build_board(
    payload: dict,
    *,
    end: Any = None,
    pool: str = "both",
    top: Any = None,
    sort: str = "var",
    hide_unsupported: bool = False,
) -> Board:
    """Collapse the cached sweep to one timeframe and rank it.

    START IS LOCKED at `base_season + 1`; only the end year moves. That is what makes the
    dropdown free -- `horizons[0]` stays 1, so every range reads the same cached points
    (see `trajectory.sweep`) -- and it is also what keeps `rank_next` meaningful, since
    `next` is only populated when horizon 1 is in range.
    """
    players = from_payload(payload)
    base = int(payload["base_season"])
    max_horizon = int(payload["max_horizon"])
    end_years = [base + h for h in range(1, max_horizon + 1)]

    end_year = _clamp(end, end_years[0], end_years[-1], end_years[0])
    show_all = str(top).lower() == "all"
    top_n = None if show_all else _clamp(top, 1, 5000, DEFAULT_TOP)
    if pool not in ("both", "hitter", "pitcher"):
        pool = "both"
    if sort not in SCALES:
        sort = "var"
    horizons = tuple(range(1, end_year - base + 1))

    # RANKED OVER THE WHOLE POOL, then filtered. A pitcher-only view shows LEAGUE ranks,
    # so its top row can read #7 -- correct, and the same rule #322/#323 depend on, where
    # ranking within a subset would make every team's best player a #1.
    var_rows = totals(players, horizons, "var")
    add_ranks(var_rows)
    # Keyed on (id, pool), the same pairing as the `name::player_type` rule everywhere
    # else, and BOTH halves are load-bearing on the live board. A name key collapses the
    # two hitters called Max Muncy onto one row; a bare-id key collapses Ohtani's hitter
    # line onto his pitcher line, since a two-way player is scored in both pools under
    # one MLBAM id.
    sgp_by_key = {(r["id"], r["pool"]): r for r in totals(players, horizons, "sgp")}

    rows = []
    for row in var_rows:
        if pool != "both" and row["pool"] != pool:
            continue
        if hide_unsupported and row["extrapolated"]:
            continue
        raw = sgp_by_key.get((row["id"], row["pool"]))
        move = row["rank_next"] - row["rank_total"]
        rows.append(
            {
                **row,
                # The raw scale is a SEPARATE fit, not `total + floor` -- for a
                # below-replacement player VAR is 0 and that sum would report his SGP as
                # exactly the floor. Absent when the sweep only ran the VAR scale.
                "sgp_total": raw["total"] if raw else None,
                "sgp_by_year": raw["by_year"] if raw else [],
                # The MOVE between the two ranks is the keeper signal in one number: a
                # player far better over the range than next year is who you hold rather
                # than who you start.
                "rank_move": move if abs(move) >= RANK_MOVE else 0,
            }
        )

    scored = len(rows)
    key = "sgp_total" if sort == "sgp" else "total"
    # Rows missing the raw fit sort last rather than crashing the comparison.
    rows.sort(key=lambda r: (r[key] is None, -(r[key] or 0.0), r["name"]))

    return Board(
        rows=rows if top_n is None else rows[:top_n],
        scored=scored,
        base_season=base,
        end_year=end_year,
        end_years=end_years,
        pool=pool,
        top="all" if top_n is None else top_n,
        sort=sort,
        hide_unsupported=hide_unsupported,
        # A per-year breakout only earns its columns once the range spans more than one.
        year_columns=[base + h for h in horizons] if len(horizons) > 1 else [],
        meta={
            "generated_at": payload.get("generated_at"),
            "panel_vintage": payload.get("panel_vintage"),
            "season_elapsed": payload.get("season_elapsed"),
            "min_sgp": payload.get("min_sgp"),
            "floors": payload.get("floors", {}),
        },
    )
