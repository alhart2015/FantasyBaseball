"""View model for the trajectory board page (#321).

Everything between the cached sweep and the template: pick a scale, pick a timeframe,
collapse the per-year points to it, rank and filter. No Flask, so it is testable
directly.

The page is a PURE READER of `cache:trajectory_board`, written offline by
`scripts/push_trajectory_board.py`. It cannot compute the board itself -- the fit needs
`data/trajectory/` and `data/cache/keeper_skills`, both gitignored and so absent on
Render. The board therefore does NOT move with a dashboard refresh, which is why
`Board.meta` carries the vintage and the template prints it.

TWO KEYS, AND ONLY ONE VIEW READS BOTH (#344). Career history and comps live in
`cache:trajectory_chart_data`, written by the same script in the same run. Only
`build_player_view` takes it; `build_board` and `build_teams_board` neither receive nor
need it, which is what keeps ~1.1 MB off the two default views. Because the two blobs
can be refreshed independently, the chart is drawn only when its `generated_at` matches
the board's -- see `_chart_extras`.
"""

from __future__ import annotations

import re
from collections.abc import Collection, Mapping, Sequence
from dataclasses import dataclass, field, replace
from typing import Any, NoReturn

from fantasy_baseball.data.rosters import RosterSpot
from fantasy_baseball.trajectory.comp_paths import MAX_COMPS
from fantasy_baseball.trajectory.comps import MIN_LOCAL_SUPPORT
from fantasy_baseball.trajectory.roster_join import index_rosters
from fantasy_baseball.trajectory.sweep import (
    SCALES,
    add_ranks,
    chart_key,
    from_payload,
    player_from_row,
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
    base = _board_base_season(payload)
    horizons_all = _board_horizons(payload)
    end_years = [base + h for h in horizons_all]
    end_year = _clamp(end, end_years[0], end_years[-1], end_years[0])
    pool = _clamp_choice(pool, POOLS, "both")
    scale = _clamp_choice(scale, SCALES, "var")
    horizons = tuple(range(1, end_year - base + 1))
    return base, end_year, end_years, pool, scale, horizons, _ranked_rows(payload, horizons, scale)


def _board_meta(payload: dict) -> dict:
    """The vintage and provenance block, identical for all three views -- and now built
    HERE for all three. `build_board` used to spell the same seven fields inline, so a
    field added here had to be remembered into there as well.

    `min_local_support` is in here because its ABSENCE was not something a template
    could route around: without the threshold the (!) flag has no rule to name, so
    the teams view dropped the flags entirely and every block total silently summed
    unmarked extrapolated rows. That is the cost of the copy, already paid once.

    `excluded` is who is NOT on the board. A shortened board reads as "these are the
    best players" when it is "these are the ones the model can price", and the larger
    exclusion is the silent one: a player with no current-season line was never a
    candidate, so he is absent with no row and no flag.

    `min_local_support` travels as the RULE behind the (!) flag, not just the verdict.
    It is a tuned number with a measured table behind it and an open issue (#310) to
    change the estimator it guards, so the page must not restate it as prose: the CLI
    renders it from the constant and a hardcoded template string would say "under 10%"
    about rows now flagged at something else.
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
        meta=_board_meta(payload),
    )


#: The three views `/trajectory` renders. A junk or absent value is the league board.
VIEWS = ("board", "teams", "player")


def filter_state(view: str, board: Any, args: Mapping[str, str]) -> dict:
    """The page's full filter state, as ONE dict, derived in ONE place.

    The three views carry different models -- `Board` has `top`/`team`, `TeamsBoard` has
    `per_team`, `PlayerView` has `name`/`n` -- and none has the others'. So whichever
    field the rendering view does not own has to come from the query string instead.
    That asymmetry is real. What was not real was spelling it out twice: the route built
    a seven-key dict per branch by hand, and one of those fourteen values was written as
    a literal `"all"` where its neighbours were pass-throughs. The effect was that
    /trajectory?team=X -> "By team" -> "League" came back unfiltered, and the test that
    should have caught it ran on only one of the two views.

    A field a view owns is read off the model, so it is already clamped. A field it does
    not own passes through raw: the OTHER builder owns that clamp, and clamping in two
    places is how two spellings drift apart.

    THE KEYS ARE THE QUERY-STRING NAMES, exactly. That is what lets `board_url` be
    `url_for('trajectory', **cur)` and the search form a loop over `cur`, instead of
    three hand-copied lists of the same nine names -- the fourth copy of which (the
    guard test) had already drifted two filters behind by the time this was written.
    """
    owned_teams = view == "teams"
    owned_player = view == "player"
    # `top` and `team` belong to `Board` and to nothing else, so every OTHER view has
    # to pass them through. Named once rather than spelled twice: a fourth view added
    # to `VIEWS` has to be added to this one expression, not remembered into two.
    owned_by_board = not (owned_teams or owned_player)
    return {
        "view": view,
        # `PlayerView` carries neither -- a per-player fit has no "end year" to move
        # (the chart always shows the full history) and no pool filter (the search is
        # a single resolved name, not a ranked list). Pass through from the query
        # string on the player view, exactly like `top`/`team`: a round trip through
        # another view must not drop them, which is the same bug the literal `"all"`
        # this docstring memorializes already caused once for `team`.
        "end": (args.get("end", 0) if owned_player else (board.end_year if board else 0)),
        "pool": (args.get("pool", "both") if owned_player else (board.pool if board else "both")),
        "scale": board.scale if board else "var",
        # Owned by `build_teams_board` on the teams view; by the query string otherwise.
        "per": board.per_team if (owned_teams and board) else args.get("per", DEFAULT_PER_TEAM),
        # Owned by `build_board` on the league view; by the query string otherwise.
        "top": (board.top if board else DEFAULT_TOP)
        if owned_by_board
        else args.get("top", DEFAULT_TOP),
        "team": (board.team if board else "all") if owned_by_board else args.get("team", "all"),
        # Owned by the player view; a pass-through elsewhere so a round trip through
        # another view does not drop the searched name.
        "player": board.name if (owned_player and board) else args.get("player", ""),
        "n": board.n if (owned_player and board) else args.get("n", DEFAULT_COMPS),
        # The player view's NAME-NARROWING keys, owned by it and passed through
        # elsewhere exactly like `player` and `n`. Deliberately not `pool`: that is the
        # board's hitter/pitcher filter, two lines up, and one key cannot mean both a
        # board filter and a tie-break between two rows sharing a name.
        "pid": board.pid if (owned_player and board) else args.get("pid", ""),
        "ppool": board.ppool if (owned_player and board) else args.get("ppool", ""),
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
        base_season=base,
        end_year=end_year,
        end_years=end_years,
        pool=pool,
        scale=scale,
        per_team=n,
        year_columns=[base + h for h in horizons] if len(horizons) > 1 else [],
        meta=_board_meta(payload),
    )


#: Comps drawn by default. The payload stores `MAX_COMPS` (10); this is what the chart
#: shows until asked otherwise. Deliberately NOT the same constant: this one is a
#: display preference and that one is a storage ceiling, and binding them would mean
#: changing what is drawn to change what is kept.
DEFAULT_COMPS = 5


@dataclass(frozen=True)
class PlayerView:
    """One player's chart: what happened, what is predicted, and what it looked like
    when this shape played out before."""

    name: str
    age: int
    slot: str
    #: The replacement level EVERY series on the VAR axis is netted against -- his own.
    #: Not each comp's own slot floor: the chart asks what a trajectory would be worth in
    #: THIS player's slot, and per-comp floors would put non-comparable lines on one axis.
    floor: float
    scale: str
    n: int
    #: [[age, value], ...] realized, ascending. Empty when no chart data was paired with
    #: this board -- see `chart_vintage_mismatch` for the two ways that happens.
    history: list[list[float]]
    #: [{age, mean, p10, p90}, ...] one per projected year.
    projection: list[dict]
    #: [{name, season, rmse, path: [{age, value}, ...], career: [[age, value], ...]},
    #: ...] closest first. `path` is his forward path over the projected ages, drawn on
    #: the main chart; `career` is his WHOLE arc, drawn on his own card, and is empty
    #: when the blob carries no entry for him. ONE list, because a card titles itself and
    #: draws itself from the same entry -- a second parallel list would have to stay the
    #: same length and order forever, enforced by nothing but construction.
    #:
    #: The age at which he matched the subject is `PlayerView.age`, identical on every
    #: card: `closest_paths` selects on `prepared.age == float(age)`, an exact match.
    comps: list[dict]
    #: Populated when the name was ambiguous OR when it matched nothing exactly and the
    #: substring fallback found something (#350). The caller renders these instead of a
    #: chart, as LINKS carrying `pid`/`ppool`. Guessing puts one man's career under
    #: another's name; offering an unselectable list makes the name a permanent dead end.
    candidates: list[dict]
    #: The narrowing that RESOLVED this view, echoed back so `filter_state` can put it on
    #: every control link. Empty when the name resolved on its own, and empty on the
    #: candidate page -- a link there gets its narrowing from the candidate it names.
    #: Strings because they are query-string values and go straight back out as such.
    pid: str
    ppool: str
    found: bool
    extrapolated: bool
    #: True when `candidates` came from the substring fallback rather than from an
    #: exact-name collision (#350). THE TWO READ DIFFERENTLY: "more than one player is
    #: named Max Muncy, pick one" is a statement about the board, and it is false of a
    #: list produced by typing `bat`. Without this the page would assert a collision
    #: that did not happen.
    suggested: bool
    base_season: int
    end_years: list[int]
    #: The PACED base season as [age, value], floor-netted exactly like `history`.
    #:
    #: NOT part of `history`, which means "realized complete seasons" -- several
    #: template branches key on `not board.history` to report a missing or mismatched
    #: chart blob, and folding a board-sourced point into a chart-blob-sourced list
    #: would make that check stop meaning what it says. It is DRAWN as the same line.
    #:
    #: The VALUE is straight-line prorated realized stats, never a projection blend:
    #: `board_inputs` -> `_paced` -> `prorate_partial` divides realized SGP by the
    #: elapsed fraction. No ROS projection reaches this model (#346), which
    #: tests/test_trajectory/test_no_ros_dependency.py keeps true.
    #:
    #: `None` when the player was not found, or when `age` is ALREADY a realized row in
    #: `history` -- see the suppression rule in `build_player_view`.
    #:
    #: Defaulted, and placed here rather than beside `history` where it reads more
    #: naturally: `meta` below is the first field carrying a default, so a defaulted
    #: field any earlier puts one ahead of `projection`/`candidates`/`found` and raises
    #: `TypeError: non-default argument follows default argument` at import.
    paced: list[float] | None = None
    #: What to call the paced point, finished server-side. Same rule as `axis_label`:
    #: ship the string, not the ingredients, so the chart and the table cannot disagree
    #: about whether the season is over.
    paced_label: str = ""
    meta: dict = field(default_factory=dict)
    #: The chart data that arrived is stamped for a DIFFERENT board than this one, so
    #: `history` and `comps` were dropped rather than drawn. A distinct state from
    #: "no chart data at all", and the page must say so distinctly: a missing blob means
    #: the board predates the feature (or the push never wrote it), a mismatched one
    #: means the two keys are out of step and one re-push fixes it. Saying "predates"
    #: for a mismatch sends a reader looking in the wrong place.
    chart_vintage_mismatch: bool = False

    @property
    def axis_label(self) -> str:
        """What the value axis is measuring, said in words, for EVERY surface.

        Derived rather than stored, like `Board.span`, so it cannot disagree with the
        `floor` it names. It was spelled twice and in two languages -- a Jinja `{% set %}`
        for the table header and a JS template literal for the chart's y-axis -- and the
        two had to agree about a subtraction, which is precisely the duplicated-rule
        class `var_offset` was extracted to stop. The chart now reads this through the
        JSON island (`data.axis_label`) rather than re-deriving it from `floor`.

        `floor` here is the APPLIED offset, so on the SGP scale it is 0.0 and the label
        must not claim a subtraction that did not happen.
        """
        if self.scale != "var":
            return self.scale.upper()
        return f"VAR (SGP - {self.floor:.2f} slot floor)"


def _narrow(hits: list[dict], field: str, wanted: Any) -> list[dict]:
    """`hits` restricted to rows whose `field` reads `wanted`, when that helps.

    A blank or absent `wanted` is not a filter. Neither is one that matches NOTHING: the
    player page's search form is a GET that re-submits every key in `filter_state`, so a
    new name always arrives carrying the PREVIOUS player's narrowing. Obeying it strictly
    would report "no player named X" for a player who is on the board -- swapping one
    dead end for another. Compared as strings because these arrive from a query string.
    """
    if wanted in (None, ""):
        return hits
    kept = [p for p in hits if str(p[field]) == str(wanted)]
    return kept or hits


FIND_MIN_CHARS = 2
FIND_RESULT_CAP = 25


def _raise_stale_board(exc: KeyError, *, where: str = "row") -> NoReturn:
    """The one sentence a stale board gets, wherever it is discovered.

    `where` names WHAT is missing the key. Saying "row is missing 'max_horizon'" sends
    the reader looking for a bad player row over a key that lives on the payload -- and
    "row is missing 'players'" sends them looking for a row in a payload that has none.

    A bare `KeyError('now')` reaches the reader as a red banner containing literally
    `'now'`: it names the field and nothing else, with no hint that the payload is the
    problem or that a re-push fixes it. Spelled ONCE because there are now two callers
    that can hit it -- resolving a row, and scanning every row to suggest names -- and
    the second was added ABOVE the first, which silently bypassed the guard the first
    one had.
    """
    # An argless KeyError would IndexError inside the error handler, which replaces a
    # bad message with a worse traceback.
    field = exc.args[0] if exc.args else "an unnamed field"
    # Joined rather than interpolated: `where=""` names the payload itself, and
    # "payload {where} is" left a double space there -- invisible in HTML, and visible
    # in every log line and in the tests that pin the sentence.
    subject = " ".join(part for part in ("trajectory board payload", where) if part)
    raise ValueError(
        f"{subject} is missing {field!r}, which this "
        "build requires; re-run scripts/push_trajectory_board.py"
    ) from exc


def _whole_number(raw: Any) -> int | None:
    """`raw` as an int if it IS one, else None. Never truncates.

    `int(3.7)` is 3, so a guard built on `int()` accepted a fractional horizon and
    scored one season fewer than the payload claimed -- while its own message said it
    rejects anything that is not a whole number. Booleans are excluded even though
    `bool` is an `int` subclass: `max_horizon=True` is a corrupt payload, not one.
    """
    if isinstance(raw, bool):
        return None
    if isinstance(raw, int):
        return raw
    if isinstance(raw, float):
        return int(raw) if raw.is_integer() else None
    if isinstance(raw, str):
        try:
            return int(raw.strip())
        except ValueError:
            return None
    return None


def _required_int(payload: dict, field: str) -> int:
    """A payload scalar the page cannot proceed without, validated in ONE place.

    `max_horizon` and `base_season` are read together by every board builder and were
    validated separately -- one through a guard, the other through a bare
    `int(payload[...])` on the next line. That asymmetry is the whole defect: a missing
    key raised a KeyError that renders as a banner naming only the field, and a null
    raised a TypeError, which the route's (ValueError, KeyError) handler does not catch
    at all, so it reached the reader as an unhandled 500.

    Not `.get(field, <default>)`: a default silently ranks a single horizon or dates the
    board to the wrong year, which is a wrong answer with no error.
    """
    try:
        raw = payload[field]
    except KeyError as exc:
        # NOT a bare KeyError. It renders as a banner reading literally 'max_horizon',
        # which is the same unactionable failure `_raise_stale_board` exists to remove
        # for 'now' -- and it was reintroduced here, for a different key, in the batch
        # that added that helper.
        _raise_stale_board(exc, where="")
    value = _whole_number(raw)
    if value is None:
        raise ValueError(
            f"trajectory board payload has {field}={raw!r}, which is not a whole "
            "number; re-run scripts/push_trajectory_board.py"
        )
    return value


def _board_base_season(payload: dict) -> int:
    """The season the board's horizons count forward from. REQUIRED."""
    return _required_int(payload, "base_season")


def _board_horizons(payload: dict) -> tuple[int, ...]:
    """Every horizon this board carries. `max_horizon` is REQUIRED.

    Not `.get("max_horizon", 1)`: a default silently ranks a single horizon, which
    drops every player with no horizon-1 point out of `totals()` and makes them
    unsearchable while the rest of the page still lists them -- a wrong answer with no
    error. Every other reader in this module treats the field as required.
    """
    horizon = _required_int(payload, "max_horizon")
    if horizon < 1:
        raise ValueError(
            f"trajectory board payload has max_horizon={horizon!r}, which scores no "
            "seasons; re-run scripts/push_trajectory_board.py"
        )
    return tuple(range(1, horizon + 1))


_WHITESPACE = re.compile(r"\s+")


def normalized_query(query: str) -> str:
    """A search string reduced to what matching actually compares.

    `normalize_name` lowercases and strips accents, and its docstring claims it
    "removes extra whitespace" -- it does not. `find_players` trusted that claim, so a
    copy-pasted double space was a dead end: exactly the failure this feature exists to
    remove. Collapsed here rather than in `name_utils` because that function is a join
    key for keeper matching and the draft board, and changing what it returns would
    silently re-key every one of those callers.

    THE ONE OWNER OF THE LENGTH RULE. The route used to length-check the raw string
    while the matcher re-checked the normalized one, so a two-character query that
    normalized to one was accepted and then returned nothing -- "no match" for input
    the API had just called long enough, which is the absent-vs-no-match conflation
    #350 exists to remove.
    """
    return _WHITESPACE.sub(" ", normalize_name(query or "")).strip()


def find_players_counted(
    payload: dict, query: str, *, cap: int = FIND_RESULT_CAP
) -> tuple[list[dict], int]:
    """`find_players`, plus how many matched BEFORE the cap.

    The cap was silent on both surfaces, so 25-of-300 rendered identically to
    25-of-25. A reader whose player fell past the cut sees him missing and concludes he
    is not on the board -- the conclusion the feature exists to prevent.
    """
    hits = _find(payload, query, cap=None)
    return hits[:cap], len(hits)


def find_players(payload: dict, query: str, *, cap: int = FIND_RESULT_CAP) -> list[dict]:
    """Board rows whose name contains `query`, best offer first.

    THE SHARED MATCHER. `/api/trajectory/find` and `build_player_view`'s fallback both
    call it, so a suggestion can never be a name the resolver then fails on -- which is
    the failure that would make this feature worse than the dead end it replaces.
    Matching goes through `normalize_name`, the same function resolution uses.

    Searches the BOARD, not `ros_projections`. The two populations differ by a lot: the
    board drops everyone with no current-season line and everyone pacing under MIN_SGP.
    Suggesting a name from the wider set would offer a row that then renders "no player
    named X on this board".

    Substring, not prefix -- `Witt` has to find `Bobby Witt Jr.`, and that is the
    complaint. Ranked exact, then prefix, then substring, and inside a tier by the
    board's own `rank_total` so the better player is offered first. Ranks come from
    `_ranked_rows`, which holds a parsed copy per vintage, so this is a scan and no I/O.

    Ranked on the DEFAULT timeframe and scale (every horizon, VAR) rather than on
    whatever the caller has on screen. A suggestion list is a name picker: re-deriving
    it per scale would cost a second ranked copy per vintage to reorder rows the user is
    choosing between by name.
    """
    return _find(payload, query, cap=cap)


def _find(payload: dict, query: str, *, cap: int | None) -> list[dict]:
    """The scan. `cap=None` returns every match, for the truncation count."""
    needle = normalized_query(query)
    if len(needle) < FIND_MIN_CHARS:
        return []

    horizons = _board_horizons(payload)
    try:
        rows = _ranked_rows(payload, horizons, "var")
    except KeyError as exc:
        # This scans EVERY row, so a stale board is discovered here rather than on the
        # one resolved row -- and it must say the same thing when it is.
        _raise_stale_board(exc)

    scored: list[tuple[int, int, dict]] = []
    for row in rows:
        name = normalized_query(row["name"])
        if needle == name:
            tier = 0
        elif name.startswith(needle):
            tier = 1
        elif needle in name:
            tier = 2
        else:
            continue
        scored.append((tier, int(row["rank_total"]), row))

    scored.sort(key=lambda item: (item[0], item[1]))
    return [
        {
            "name": row["name"],
            # REQUIRED, not decoration. `board_url(..., pid=, ppool=)` needs both to
            # build a link that lands on the player instead of the candidate page, and
            # a two-way player's two rows differ only by `pool`.
            "id": row["id"],
            "pool": row["pool"],
            "age": row["age"],
            "slot": row["slot"],
        }
        for _, _, row in (scored if cap is None else scored[:cap])
    ]


def _netted(pairs: Any, floor: float) -> list[list[float]]:
    """Stored `[[age, sgp], ...]` as floor-netted points, ascending by age.

    ONE rule for both stored arcs -- the subject's career and a comp's -- because a comp
    card draws them on one axis and the two must mean the same thing. They were spelled
    separately and only one of them sorted.

    Sorted rather than trusted in stored order: the push script's groupby happens to emit
    ascending, but nothing in the wire format enforces it, and an unsorted blob would
    zigzag the line with every other assertion still green.
    """
    return sorted(
        ([int(a), float(v) - floor] for a, v in pairs),
        key=lambda pt: pt[0],
    )


def _chart_extras(
    payload: dict, chart: Any, mlbam_id: int, pool: str
) -> tuple[dict, Mapping, bool]:
    """One player's stored `{history, comps}`, the comp-career map, and whether the
    pair was REFUSED.

    The vintage guard (#344). `cache:trajectory_chart_data` is a second blob, written
    beside the board but stored separately, so the two can end up out of step -- a
    board refreshed at noon beside extras from Tuesday. Drawn together, that is a stale
    career line under a fresh projection: silent, and both halves look plausible. So the
    extras are used ONLY when their stamp equals the board's, and are otherwise treated
    as absent.

    An absent stamp on EITHER side refuses too, which is what the `in (None, "")` clause
    buys over a plain `!=`: a hand-built or pre-vintage payload gives `None` on both
    sides, and `None == None` would read as a match and pair two blobs on no evidence at
    all. Empty strings pair the same way and are refused for the same reason.

    Returns `({}, {}, True)` for a refusal so the caller can tell it from
    `({}, {}, False)`, which is "no chart data arrived" -- different causes, different
    fixes, and the page names them separately. `None` is the ONLY input that reads as
    "nothing arrived": anything else was stored by somebody, so an unreadable shape is a
    writer out of step with this reader (a refusal), not a key that was never pushed.

    The SECOND element is `careers` (#346): comp arcs, deduped across the whole board
    and keyed by `chart_key`. Absent on every blob written before that feature, and not
    a mapping if some future writer stores something else there -- both are `{}`, never
    an exception, for the same #332 reason the refusals above are refusals.

    Typed `Mapping`, not `dict`: it is handed straight back off the cached blob and only
    ever read through `.get`. Copying it into a dict to satisfy the narrower annotation
    would duplicate ~300 KB on every player-page request for nothing.
    """
    if chart is None:
        return {}, {}, False
    if not isinstance(chart, Mapping):
        # STORED, but not a mapping -- a JSON array under this key, which is what the
        # board itself would serialize as if it were written here by mistake. Refused
        # rather than raised: the extras are auxiliary, and #332 is the standing
        # reminder that refusing a page over data it could largely render is how
        # /trajectory goes down. Reaching this at all is why the route reads the chart
        # key with `read_cache` and not `read_cache_dict` -- the latter narrows a stored
        # list to None, which is indistinguishable from a key that was never written and
        # would print "this board predates the feature" at a blob that is merely stale.
        return {}, {}, True
    board_at, chart_at = payload.get("generated_at"), chart.get("generated_at")
    if board_at in (None, "") or chart_at in (None, "") or str(board_at) != str(chart_at):
        return {}, {}, True
    players = chart.get("players")
    if not isinstance(players, Mapping):
        # The same refusal one level in: the BOARD's `players` is a list, so this is the
        # board written to the chart key by mistake, stamped identically because the one
        # push produced both. `players.get` on a list is an AttributeError, which the
        # route does not catch and which 500s the page.
        return {}, {}, True
    careers = chart.get("careers")
    return (
        players.get(chart_key(mlbam_id, pool), {}),
        careers if isinstance(careers, Mapping) else {},
        False,
    )


def build_player_view(
    payload: dict,
    *,
    player: str,
    scale: str = "var",
    n: Any = None,
    pid: Any = None,
    ppool: Any = None,
    chart: Any = None,
) -> PlayerView:
    """One player's career, projection and comps, on one scale.

    `chart` is the `cache:trajectory_chart_data` blob -- career history and comps, which
    live outside the board because this is the only view that reads them (#344). It is
    OPTIONAL: a missing key renders the projection alone, which is the shape prod held
    before the split ever ran. Typed as whatever the KV handed back rather than as a
    dict, because that is what it is -- `_chart_extras` decides whether the thing is
    readable and whether it pairs with this board, and both answers are refusals rather
    than exceptions.

    Resolved BY NAME, never by an id from the query string. CLAUDE.md names a
    hand-carried id as a defect class that has twice landed on a real row belonging to
    someone else, and `player_trajectory.py` already refuses a `--mlbam-id` that
    disagrees with its `--player`.

    `pid` and `ppool` NARROW that name's hits; they never select on their own. An id
    naming a row this name does not match is discarded, so the rule the CLI enforces --
    the id must agree with the name -- holds here too, and the resolved row is always one
    the searched name produced.

    THEY ARE NOT `pool`. That key is the board's hitter/pitcher filter, which this view
    only passes through; overloading it would couple a board filter to name resolution.
    """
    scale = _clamp_choice(scale, SCALES, "var")
    # The ceiling is the push script's STORED count, from the one place it is defined:
    # the blob is built hours earlier, so `n` can only slice what is already in it.
    want = _clamp(n, 1, MAX_COMPS, DEFAULT_COMPS)
    base = _board_base_season(payload)
    horizons_all = _board_horizons(payload)
    end_years = [base + h for h in horizons_all]

    # THROUGH THE SAME FUNCTION THE MATCHER USES. Resolving with bare `normalize_name`
    # while `_find` compared the whitespace-collapsed form made the two disagree about
    # what a name is: a double-spaced name missed here, then matched as a tier-0 exact
    # hit below, and the page printed "no player is named exactly X" above a list
    # containing exactly X.
    target = normalized_query(player)
    hits = [p for p in payload.get("players", []) if normalized_query(p["name"]) == target]
    hits = _narrow(hits, "id", pid)
    hits = _narrow(hits, "pool", ppool)

    empty = PlayerView(
        name=player or "",
        age=0,
        slot="",
        floor=0.0,
        scale=scale,
        n=want,
        history=[],
        projection=[],
        comps=[],
        candidates=[],
        pid="",
        ppool="",
        found=False,
        extrapolated=False,
        suggested=False,
        base_season=base,
        end_years=end_years,
        meta=_board_meta(payload),
    )
    if not hits:
        # THE DEAD END, replaced. An exact miss falls back to the same substring matcher
        # the suggestion box uses, so `bat` lands on a candidate list rather than on "No
        # player named "bat" on this board." Works with JS off, which is why the issue
        # recommends landing it first.
        #
        # A name that substring-matches NOTHING still returns `empty`, candidates and
        # all: that sentence has to stay distinguishable from a typo, because a player
        # genuinely absent from the board -- no current line, or pacing under MIN_SGP --
        # is a real answer and not a search failure.
        suggestions = find_players(payload, player or "")
        if not suggestions:
            return empty
        # `find_players` already returns exactly the keys the template reads. Rebuilding
        # them here was a no-op that would silently drop any field added to it later.
        return replace(empty, suggested=True, candidates=suggestions)
    if len(hits) > 1:
        return replace(
            empty,
            candidates=[
                {
                    "id": p["id"],
                    "name": p["name"],
                    "age": p["age"],
                    "slot": p["slot"],
                    # THE DISCRIMINATOR, not decoration. Ohtani's two rows share an id
                    # and an age and differ only by slot and pool, so a list without it
                    # offers two lines identical in every field it prints.
                    "pool": p["pool"],
                }
                for p in sorted(hits, key=lambda p: (p["id"], p["pool"]))
            ],
        )

    row = hits[0]
    # THROUGH THE PARSER, not a second hand-unpack of `row["sgp"]` beside it.
    # `player_from_row` exists for exactly this caller: it applies the same point
    # schema the board reads, so `age + horizon`, the floor subtraction and the
    # legacy-positional-blob refusal are each spelled ONCE, in `sweep`. Hand-unpacking
    # here re-spelled all three, and the guard had to be re-derived to keep a legacy
    # blob raising a `ValueError` the route degrades on rather than a `TypeError` 500.
    try:
        sp = player_from_row(row)
    except KeyError as exc:
        # `player_from_row` reads six fields this view never uses -- `id`, `pool`,
        # `now`, `prior`, `support`, `extrapolated` -- so a stale blob missing any of
        # them raises `KeyError('now')`, the route catches it, and `str(exc)` renders a
        # red banner reading literally `'now'`. That names the field and nothing else:
        # no hint that the payload is the problem or that a re-push fixes it.
        #
        # Re-raised HERE rather than loosened in `player_from_row`: the strictness is
        # right -- a missing field must fail on its own name, which is exactly what
        # `_pack`'s docstring says named keys bought. What was missing is the same
        # actionable sentence `_unpack` already gives the positional-blob case.
        _raise_stale_board(exc)
    # THE offset, from the one place it is defined -- see `var_offset`'s docstring.
    # Not `float(row["floor"]) if scale == "var" else 0.0` inline: that is a third
    # spelling of the same rule `SweptPlayer.offset` already carries, and it disagreed
    # with it silently (any non-"var" scale read as SGP, where `var_offset` raises).
    floor = sp.offset(scale)
    # KEYED ON THE RESOLVED ROW's `(mlbam_id, pool)`, never the searched name or the
    # bare id: a two-way player has one entry per pool, and the id alone would hand the
    # hitter row the pitcher's career.
    extras, careers, mismatch = _chart_extras(payload, chart, sp.mlbam_id, sp.pool)
    history = _netted(extras.get("history", []), floor)
    # THE SUPPRESSION RULE. `_live_seasons` (build_pt_panel.py) flags a season partial
    # iff `year >= today.year`, so a panel rebuilt in January un-flags the season that
    # just ended: it enters `complete`, lands in `history`, and `base_season` still
    # names it. Appending `now` beside it draws two points at one age, one of them
    # labelled a pace, on a finished year.
    #
    # Decided from `history` rather than from the stored `base_season_partial` flag
    # because this works on EVERY blob, including ones written before that flag existed.
    # The flag labels the point; this decides whether there is one.
    realized_ages = {pt[0] for pt in history}
    paced = None if sp.age in realized_ages else [sp.age, float(sp.now) - floor]
    return replace(
        empty,
        name=sp.name,
        age=sp.age,
        slot=sp.slot,
        # The RESOLVED row's own key, whatever narrowing (if any) got here. Echoed back
        # so every control link carries a fully-resolved identity and a scale toggle
        # cannot drop a two-way player back onto the candidate list.
        pid=str(sp.mlbam_id),
        ppool=sp.pool,
        # The APPLIED offset, not the raw slot floor: under scale="sgp" nothing was
        # netted, and this field's own docstring promises "what every series was
        # netted against". A template reading this to label the chart must not print
        # "netted against 6.0" over a line that was left alone.
        floor=floor,
        found=True,
        extrapolated=sp.extrapolated,
        chart_vintage_mismatch=mismatch,
        history=history,
        paced=paced,
        # Read off the payload, not out of `meta`: the league and teams boards share
        # `_board_meta` and neither renders this, so routing it through there put a key
        # on two views that have no use for it. Default True -- every blob written
        # before the flag existed was written mid-season.
        paced_label=(
            f"{base} pace" if bool(payload.get("base_season_partial", True)) else str(base)
        ),
        # `points(scale)` applies the offset; `YearPoint.age` is already `age + horizon`.
        projection=[
            {"age": p.age, "mean": p.mean, "p10": p.p10, "p90": p.p90} for p in sp.points(scale)
        ],
        comps=[
            {
                "name": c["name"],
                "season": c["season"],
                "rmse": c["rmse"],
                # Truncated to the PROJECTED horizons, not the stored path length. A
                # produced blob never needs this slice to do anything: comp paths come
                # off the same `horizons` tuple as `row["sgp"]`, and a player whose
                # observable path is SHORTER than those horizons gets no comps at all
                # (`push_trajectory_board.player_comps`) rather than a long path beside
                # a short projection. So this is defence against a hand-built or future
                # blob, not a rule the current pipeline exercises -- cheap and correct
                # to keep regardless.
                "path": [
                    {"age": sp.age + h, "value": float(v) - floor}
                    for h, v in enumerate(c["path"][: len(sp.sgp)], start=1)
                ],
                # HIS WHOLE ARC, on the same entry as his forward path rather than in a
                # second list beside it. A parallel list would have to stay the same
                # length and order forever, enforced by nothing but construction, and
                # the card that titles itself from one and draws from the other is a
                # single off-by-one from putting one man's name over another's career.
                #
                # `c.get("id")` -- an older push wrote comps with no id at all, which is
                # a missing join key rather than an error. `chart_key` is never called
                # with None: the lookup short-circuits first.
                #
                # Netted against the SUBJECT's floor, like `path` above and for the same
                # reason: the card asks what this arc would be worth in his slot, and
                # per-comp floors would put non-comparable lines on one axis.
                "career": _netted(careers.get(chart_key(c["id"], sp.pool), []), floor)
                if c.get("id") is not None
                else [],
            }
            # Sliced ONCE, here. This is the only comp list there is.
            for c in extras.get("comps", [])[:want]
        ],
    )
