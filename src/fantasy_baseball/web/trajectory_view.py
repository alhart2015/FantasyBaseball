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

from collections.abc import Collection, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

from fantasy_baseball.data.rosters import RosterSpot
from fantasy_baseball.trajectory.comps import MIN_LOCAL_SUPPORT
from fantasy_baseball.trajectory.roster_join import index_rosters
from fantasy_baseball.trajectory.sweep import (
    SCALES,
    add_ranks,
    from_payload,
    rank_move,
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
    #: The selected team, or "all". Not a filter over `teams` -- an unknown value
    #: is clamped to "all" the way `pool` and `scale` are, because this arrives
    #: from a user-editable query string and survives a team rename.
    team: str = "all"
    #: Dropdown order: my team first, then alphabetical. Empty when no roster
    #: data arrived, which is what hides the control entirely.
    teams: tuple[str, ...] = ()
    #: Rostered players on the SELECTED team with no scored row. Empty for "all".
    #: Rendering this is what keeps an absent player from reading as a bad one.
    unscored: list[str] = field(default_factory=list)
    #: True when MY OWN team joined at least one scored row on this board. Strictly
    #: stronger than "a roster read returned something", and deliberately so -- the
    #: reasoning is at the assignment site. A read can succeed, populate `teams` and
    #: leave this False, so it is not a verdict on the read and nothing rendered from
    #: it may name a cause. What it gates is whether any row on screen is marked
    #: `mine`, which is the only claim the banner is entitled to make.
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


#: The pool filter's choices. `SCALES` next door is a shared constant for exactly
#: this reason; this tuple was hand-spelled in both builders, which is the drift
#: `_clamp_choice`'s own docstring was written to stop.
POOLS = ("both", "hitter", "pitcher")

#: Players shown per team block. Matches the CLI's `--per-team` default, and is
#: deliberately NOT the league board's `DEFAULT_TOP`: one binding for both would mean
#: setting 5 collapses the league board, or 50 puts fifty players in every block.
DEFAULT_PER_TEAM = 5


@dataclass(frozen=True)
class TeamBlock:
    """One team's slice of the board."""

    team: str
    #: The best `per_team`, league-ranked, strongest first.
    rows: list[dict]
    #: This team's rows AFTER the pool filter and BEFORE the slice, so a block reads
    #: "5 of 24" -- and "5 of 14" under ?pool=hitter. Counting pre-pool would print a
    #: total the visible rows cannot add up to.
    scored: int
    unscored: list[str]
    is_mine: bool

    @property
    def total(self) -> float:
        """The BEST-N total, never the roster total -- `rows` is already the slice.

        Derived rather than stored because it was being summed one line after `rows`
        was assigned from the same list. 7e74b7b1 removed the roster version from the
        CLI after measuring it: 93.5% of scored players carry a negative VAR and tails
        run -62 to -196 against a best-5 signal of 15 to 73, so as a sort key the
        roster total orders the page by depth of junk.
        """
        return float(sum(r["total"] for r in self.rows))


@dataclass(frozen=True)
class TeamsBoard:
    """Every team's block, ordered by strength."""

    blocks: list[TeamBlock]
    #: League-wide row count, so a block's #37 is readable as a league position.
    ranked: int
    base_season: int
    end_year: int
    end_years: list[int]
    pool: str
    scale: str
    per_team: int
    year_columns: list[int]
    meta: dict = field(default_factory=dict)

    @property
    def mine_missing(self) -> bool:
        """No block is the reader's own -- a rename, a config mismatch, or a failed
        config read. Drives a one-line banner, so unhighlighted blocks are explained
        rather than read as "you own none of these". Derived, like `span`."""
        return not any(b.is_mine for b in self.blocks)

    @property
    def span(self) -> str:
        """Same label the league board prints, from the same rule."""
        start = self.base_season + 1
        return f"{start}" if self.end_year == start else f"{start}-{str(self.end_year)[-2:]}"


def _sweep_setup(
    payload: dict, end: Any, pool: str, scale: str
) -> tuple[int, int, list[int], str, str, tuple[int, ...], list[dict]]:
    """Everything both board builders need before they diverge.

    `build_board` and `build_teams_board` parsed the payload, clamped `end`/`pool`/
    `scale` and derived `horizons` with identical code, so a change to any of it had
    to be made twice with nothing forcing the second edit. What they do NOT share is
    what comes after -- one flattens, the other groups -- which is why this returns
    the shared prefix rather than trying to be a common body.
    """
    base = int(payload["base_season"])
    max_horizon = int(payload["max_horizon"])
    end_years = [base + h for h in range(1, max_horizon + 1)]
    end_year = _clamp(end, end_years[0], end_years[-1], end_years[0])
    pool = _clamp_choice(pool, POOLS, "both")
    scale = _clamp_choice(scale, SCALES, "var")
    horizons = tuple(range(1, end_year - base + 1))
    return base, end_year, end_years, pool, scale, horizons, _ranked_rows(payload, horizons, scale)


def _board_meta(payload: dict) -> dict:
    """The vintage and provenance block, identical for both views.

    `min_local_support` is in here because its ABSENCE was not something a template
    could route around: without the threshold the (!) flag has no rule to name, so
    the teams view dropped the flags entirely and every block total silently summed
    unmarked extrapolated rows.
    """
    return {
        "generated_at": payload.get("generated_at"),
        "panel_vintage": payload.get("panel_vintage"),
        "season_elapsed": payload.get("season_elapsed"),
        "min_sgp": payload.get("min_sgp"),
        "floors": payload.get("floors", {}),
        "excluded": payload.get("excluded", {}),
        "min_local_support": MIN_LOCAL_SUPPORT,
    }


def _annotate(
    row: dict,
    *,
    key: tuple[str, str],
    owners: frozenset[str],
    my_team: str | None,
    index: Any,
    horizons: tuple[int, ...],
    always_attributed: bool,
) -> dict:
    """A board row plus the four per-row annotations both views render.

    A COPY, never a mutation: these rows come from `_ranked_rows`, a cache shared
    across requests. `always_attributed` is the one genuine difference -- on the
    teams view every row is shown under some team, so an unresolvable key is always
    a guess worth flagging; on the league board at `team=all` most rows are
    attributed to nobody and flagging them all would be noise.

    `year_cells` is aligned to the year columns BY HORIZON, one entry each, None
    where a player has no point for that year. Each horizon is a separate fit, so
    `by_year` carries no guarantee of being a prefix; rendering it positionally and
    padding the tail would put a year-2 figure under the year-1 header for any path
    with a hole. Defensive -- no such gap has been reproduced from the model -- but
    the alignment is free here and the assumption is not worth carrying in a
    template, let alone in two of them.

    `rank_move` is the keeper signal in one number: a player far better over the
    range than next year is who you hold rather than who you start. Only the league
    board renders it; it is computed here anyway so the two views' rows stay one
    shape, which is what lets a future column move between them without a second
    derivation appearing.
    """
    is_mine = my_team is not None and my_team in owners
    return {
        **row,
        "mine": is_mine,
        "owner_ambiguous": (always_attributed or is_mine) and key in index.ambiguous,
        "rank_move": rank_move(row),
        "year_cells": _year_cells(row["by_year"], horizons),
    }


def _clamp_choice(value: Any, allowed: Collection[str], default: str) -> str:
    """A query param that must name one of `allowed`. The enum twin of `_clamp`, and
    for the same reason its docstring gives: these arrive from a URL a user can edit.

    Written once rather than inline per filter -- there were three hand-copied
    `if x not in (...): x = default` blocks, which is how one of them eventually gets
    `!=` where its neighbours have `not in`."""
    return value if value in allowed else default


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
    spots: Sequence[RosterSpot] | None = None,
    my_team: str | None = None,
    team: str = "all",
) -> Board:
    """Collapse the cached sweep to one timeframe and rank it.

    START IS LOCKED at `base_season + 1`; only the end year moves. That is what makes the
    dropdown free -- `horizons[0]` stays 1, so every range reads the same cached points
    (see `trajectory.sweep`) -- and it is also what keeps `rank_next` meaningful, since
    `next` is only populated when horizon 1 is in range.
    """
    base, end_year, end_years, pool, scale, horizons, ranked_rows = _sweep_setup(
        payload, end, pool, scale
    )
    show_all = str(top).lower() == "all"
    top_n = None if show_all else _clamp(top, 1, 5000, DEFAULT_TOP)

    # ONE SCALE ON SCREEN. The two scales differ by the slot's floor (#331 made that
    # exact), and a board mixing them forces the reader to track which number belongs to
    # which -- the rank, the band and the per-year cells then have to agree about a scale
    # that is only implicit. The toggle makes it explicit and the ambiguity structurally
    # impossible.
    #
    # RANKED OVER THE WHOLE POOL, then filtered. A pitcher-only view shows LEAGUE ranks,
    # so its top row can read #7 -- correct, and the same rule #322/#323 depend on, where
    # ranking within a subset would make every team's best player a #1.

    # ONE index serves both the highlight and the filter. The route used to build
    # the ownership set itself, which meant two places decided what a roster spot
    # meant to a board row.
    index = index_rosters(ranked_rows, spots or [], my_team)
    # Same clamp `pool` and `scale` get, through the same helper: this arrives from a
    # query string a reader can edit, and a bookmark outlives a team rename.
    team = _clamp_choice(team, ("all", *index.teams), "all")

    rows = []
    for row in ranked_rows:
        if pool != "both" and row["pool"] != pool:
            continue
        # The key ONCE per row. `team_for` and `is_ambiguous` each normalize the
        # name themselves, so going through both cost two more NFKD passes per row
        # on top of the one `index_rosters` already did -- three per row, for one
        # lookup pair. The dicts they wrap are public; use them.
        key = (normalize_name(row["name"]), row["pool"])
        # MEMBERSHIP, not the winning spot's team. Two owners of one key both see
        # the row (flagged); deriving this from a single winner silently took the
        # loser's own player off his page -- see `RosterIndex.owners_of`.
        owners = index.owners_of.get(key, frozenset())
        if team != "all" and team not in owners:
            continue
        # `always_attributed` is the whole difference between the two views: under a
        # team filter every row shown is being claimed for that team, so an
        # unresolvable key is always a guess; at `team=all` most rows are claimed for
        # nobody and flagging them all would be noise.
        rows.append(
            {
                **_annotate(
                    row,
                    key=key,
                    owners=owners,
                    my_team=my_team,
                    index=index,
                    horizons=horizons,
                    always_attributed=team != "all",
                ),
            }
        )

    scored = len(rows)
    ranked = len(ranked_rows)
    # THIS SORT CREATES THE ORDER -- the rows do not arrive in it. `add_ranks` stamps
    # ranks off a temporary sorted view and leaves its input in `totals()` order, which
    # is every hitter and then every pitcher. Ranking by `rank_total` rather than by
    # `total` is what makes it a re-use of the existing order rather than a second
    # ranking: one key, already computed, and nothing to get wrong about NaNs.
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
        team=team,
        teams=index.teams,
        # Filtered to the pool on screen: this list sits under a table that may be
        # showing hitters only, and naming a pitcher there reads as a hole in it.
        unscored=index.unscored_for(team, pool) if team != "all" else [],
        # MY roster joined something -- deliberately NOT "the read returned
        # data". This gates the not-highlighted banner, and a successful read
        # where my own roster joined nothing must still show it.
        has_rosters=my_team in index.matched_teams if my_team else False,
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


#: The two views `/trajectory` renders. A junk or absent value is the league board.
VIEWS = ("board", "teams")


def filter_state(view: str, board: Any, args: Mapping[str, str]) -> dict:
    """The page's full filter state, as ONE dict, derived in ONE place.

    The two views carry different models -- `Board` has `top`/`team`, `TeamsBoard` has
    `per_team`, and neither has the other's -- so whichever field the rendering view
    does not own has to come from the query string instead. That asymmetry is real.
    What was not real was spelling it out twice: the route built a seven-key dict per
    branch by hand, and one of those fourteen values was written as a literal `"all"`
    where its neighbours were pass-throughs. The effect was that /trajectory?team=X ->
    "By team" -> "League" came back unfiltered, and the test that should have caught it
    ran on only one of the two views.

    A field a view owns is read off the model, so it is already clamped. A field it does
    not own passes through raw: the OTHER builder owns that clamp, and clamping in two
    places is how two spellings drift apart.
    """
    owned = view == "teams"
    return {
        "view": view,
        "end_year": board.end_year if board else 0,
        "pool": board.pool if board else "both",
        "scale": board.scale if board else "var",
        # Owned by `build_teams_board` on the teams view; by the query string otherwise.
        "per": board.per_team if (owned and board) else args.get("per", DEFAULT_PER_TEAM),
        # Owned by `build_board` on the league view; by the query string otherwise.
        "top": args.get("top", DEFAULT_TOP) if owned else (board.top if board else DEFAULT_TOP),
        "team": args.get("team", "all") if owned else (board.team if board else "all"),
    }


def select_view(value: Any) -> str:
    """Which view a query string is asking for, clamped to one that exists.

    Public because the ROUTE has to branch before it can build a view model, so it
    cannot learn this from the returned object the way it learns `pool` or `scale`.
    Exported deliberately rather than having the route import `_clamp_choice`:
    reaching across a module boundary for a private helper is how that helper stops
    being free to change.
    """
    return _clamp_choice(value, VIEWS, "board")


def build_teams_board(
    payload: dict,
    *,
    end: Any = None,
    pool: str = "both",
    scale: str = "var",
    spots: Sequence[RosterSpot] | None = None,
    my_team: str | None = None,
    per_team: Any = None,
) -> TeamsBoard:
    """The same cached sweep, grouped by team instead of flattened.

    Shares `_ranked_rows` with `build_board`, so switching views costs a grouping
    pass and no refit. A separate function rather than a mode flag on `Board`
    because `rows`, `scored`, `top`, `team` and `unscored` would all change meaning
    between the two, which is how a reader ends up asking which fields are live.
    """
    base, end_year, end_years, pool, scale, horizons, ranked_rows = _sweep_setup(
        payload, end, pool, scale
    )
    n = _clamp(per_team, 1, 50, DEFAULT_PER_TEAM)
    index = index_rosters(ranked_rows, spots or [], my_team)

    # BLOCKS COME FROM THE ROSTERS, not the rows. A team whose players were all
    # filtered out has no rows at all, and deriving the block list from `ranked_rows`
    # would drop the team AND its unpriced list -- leaving nothing on screen to say
    # it exists. #323 names this as the failure mode the CLI already guards.
    grouped: dict[str, list[dict]] = {team: [] for team in index.teams}

    # The RAW row goes in the bucket; annotation waits until after the slice below.
    # Enriching first built a dict and a year-cell list for every owned row and then
    # threw most of them away -- at the default N, roughly half.
    for row in ranked_rows:
        if pool != "both" and row["pool"] != pool:
            continue
        owners = index.owners_of.get((normalize_name(row["name"]), row["pool"]), frozenset())
        for team in owners:
            if team in grouped:
                grouped[team].append(row)

    blocks = []
    for team, rows in grouped.items():
        # THIS SORT CREATES THE ORDER, and it is load-bearing three times over. Rows
        # arrive in `totals()` order -- every hitter, then every pitcher -- because
        # `add_ranks` ranks off a temporary sorted view and leaves its input alone. Drop
        # it and the `[:n]` below slices the wrong players, `total` sums the wrong ones,
        # and the block ordering this whole view exists to compare is wrong with it.
        rows.sort(key=lambda r: r["rank_total"])
        blocks.append(
            TeamBlock(
                team=team,
                rows=[
                    _annotate(
                        row,
                        key=(normalize_name(row["name"]), row["pool"]),
                        owners=index.owners_of.get(
                            (normalize_name(row["name"]), row["pool"]), frozenset()
                        ),
                        my_team=my_team,
                        index=index,
                        horizons=horizons,
                        always_attributed=True,
                    )
                    for row in rows[:n]
                ],
                scored=len(rows),
                unscored=index.unscored_for(team, pool),
                is_mine=my_team is not None and team == my_team,
            )
        )
    # Name is the tie-break, not decoration: two teams with nothing scored both total
    # 0.0, and leaving that to dict order makes the page reorder between reads.
    blocks.sort(key=lambda b: (-b.total, b.team))

    return TeamsBoard(
        blocks=blocks,
        ranked=len(ranked_rows),
        base_season=base,
        end_year=end_year,
        end_years=end_years,
        pool=pool,
        scale=scale,
        per_team=n,
        year_columns=[base + h for h in horizons] if len(horizons) > 1 else [],
        meta=_board_meta(payload),
    )
