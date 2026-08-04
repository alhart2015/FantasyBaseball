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

from fantasy_baseball.trajectory.sweep import SCALES, add_ranks, from_payload, totals
from fantasy_baseball.utils.name_utils import normalize_name

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
    #: Which fit the whole table is showing: "var" or "sgp".
    scale: str
    #: True when a live roster read succeeded, so the page can tell "none of these
    #: are yours" apart from "we could not reach Upstash".
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
    players = from_payload(payload)
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

    # ONE SCALE ON SCREEN. VAR and SGP are separate fits, so a board mixing them forces
    # the reader to track which number belongs to which -- and the rank, the band and the
    # per-year cells then have to agree about a scale that is only implicit. The toggle
    # makes it explicit and the ambiguity structurally impossible.
    #
    # RANKED OVER THE WHOLE POOL, then filtered. A pitcher-only view shows LEAGUE ranks,
    # so its top row can read #7 -- correct, and the same rule #322/#323 depend on, where
    # ranking within a subset would make every team's best player a #1.
    ranked = totals(players, horizons, scale)
    add_ranks(ranked)

    # (normalized name, pool) is the only key a roster blob can be joined on -- they
    # carry no mlbam_id (#284). It is NOT unique: the live board has two hitters called
    # Max Muncy. So count the rows each key matches and mark a multi-hit as unsure rather
    # than silently putting a player the reader does not own on his keeper shortlist.
    owned = mine or set()
    key_counts: dict[tuple[str, str], int] = {}
    for row in ranked:
        k = (normalize_name(row["name"]), row["pool"])
        key_counts[k] = key_counts.get(k, 0) + 1

    rows = []
    for row in ranked:
        if pool != "both" and row["pool"] != pool:
            continue
        move = row["rank_next"] - row["rank_total"]
        key = (normalize_name(row["name"]), row["pool"])
        is_mine = key in owned
        rows.append(
            {
                **row,
                "mine": is_mine,
                "mine_ambiguous": is_mine and key_counts[key] > 1,
                # The MOVE between the two ranks is the keeper signal in one number: a
                # player far better over the range than next year is who you hold rather
                # than who you start.
                "rank_move": move if abs(move) >= RANK_MOVE else 0,
            }
        )

    scored = len(rows)
    # `add_ranks` already ordered by total descending with a name tie-break, so ranking
    # IS the sort order -- no second sort key, and nothing to get wrong about NaNs.
    rows.sort(key=lambda r: r["rank_total"])

    return Board(
        rows=rows if top_n is None else rows[:top_n],
        scored=scored,
        base_season=base,
        end_year=end_year,
        end_years=end_years,
        pool=pool,
        top="all" if top_n is None else top_n,
        scale=scale,
        has_rosters=mine is not None,
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
        },
    )
