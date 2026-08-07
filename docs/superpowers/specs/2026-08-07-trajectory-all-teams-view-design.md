# Trajectory board: all-teams view -- Design

Date: 2026-08-07
Status: approved design (brainstorming complete)
Author: session (Hart)
Implements: #323, with two deviations recorded below
Builds on: the team dropdown (#336, merged as d3cd0dcb)

## Problem

The trajectory board answers "of everyone I could hold, who is worth the most over the
years I would hold them". The team dropdown (#336) narrowed that to **one** roster at a
time. Neither answers the scouting question: *what is everyone else holding?*

That question is where a trade or keeper conversation starts, and today it can only be
answered from the CLI (`scripts/trajectory_board.py --by-team`), which needs a terminal
and a ~50-second sweep.

Flipping the dropdown through ten teams is not a substitute: comparison across rosters is
the point, and a control that shows one at a time makes the reader hold nine boards in
their head.

## Goals

- A second view on `/trajectory`: **the best N players on every team**, one block per
  team, as one scrolling page.
- **N is its own control**, defaulting to 5, independent of the league board's 50.
- **Blocks ordered by strength**, so the most dangerous roster reads first.
- **Ranks stay league-wide.** A block's top row reads `#2`, not `#1`.
- Teams whose players all filtered out **still render**, with their unpriced list.
- The end-year and scale selections are **shared** with the league board.

## Non-goals

- **No tab per opponent.** #323 rejected this and it stays rejected: eleven tabs is more
  chrome than content, and the CLI already reads well as a scroll.
- **No IL/DTD status.** #323 asked for it here specifically, on the grounds that an
  injured player is who you ask about in a trade. Hart declined during brainstorming,
  consistent with the same call on the league board: the trajectory model is explicitly
  multi-year, and whether someone is on the IL today is a question the season dashboard
  already answers. `RosterIndex.status_of` remains available for the CLI.
- **No team-level headline over the whole roster.** See the deviation below.
- **No new compute.** A filter and a grouping over the cached sweep. No refit, no new
  cache key, no change to `push_trajectory_board.py`.
- **No client-side view toggling.** See "Route and template".

## Deviations from #323

**1. Block headers and the block sort key use the BEST-N total, not the roster total.**

#323 specifies `({n} scored, {total} total {span} VAR)` per block, sorted by that total.
It was written before `7e74b7b1`, which removed exactly that number from the CLI after
measuring it on live rosters at the 2027-29 range:

- 1,093 of 1,169 scored players (93.5%) carry a **negative** VAR.
- Each roster's tail runs **-62 to -196** against a best-5 signal of **15 to 73**.
- Boston Estrellas ranked **last in the league** on the whole-roster total while its best
  five were **4th**. Seven of ten positions moved.

Using the roster total as the block sort key would order the page by depth of junk. The
best-N total is the honest comparator and is what the CLI now prints.

**2. My own team is included in the scroll, sorted by strength like everyone else.**

#323 says "every other team". Including mine makes the comparison answerable -- where do
my five sit against nine others -- which is the view's whole purpose. It is **not**
promoted to the top: sorting it first would destroy the ordering that justifies including
it. It is marked (`is_mine`) for highlighting only.

## Chosen approach

A second view function beside `build_board`, sharing the rank cache.

```python
def build_teams_board(
    payload: dict, *, end=None, pool="both", scale="var",
    spots: Sequence[RosterSpot] | None = None,
    my_team: str | None = None,
    per_team: Any = None,
) -> TeamsBoard
```

`per_team` arrives from a query string, so it is `Any` and is clamped inside:
`_clamp(per_team, 1, 50, DEFAULT_PER_TEAM)` with `DEFAULT_PER_TEAM = 5`, matching the
CLI's `--per-team`. `None` means "not supplied" and yields the default.

It reuses `_ranked_rows(payload, horizons, scale)` -- the expensive part, the sweep parse
plus `add_ranks` -- and `index_rosters`, then groups instead of flattening.

### Why not extend `Board`

Rejected: `Board` would carry `rows` **or** `blocks` depending on a mode flag, and
`scored`, `ranked`, `top`, `team` and `unscored` all change meaning between the two modes.
That is the shape that makes a later reader ask which fields are live.
`trajectory_view.py` is already ~300 lines.

### Why not group in the template

Rejected: per-team sorting, best-N slicing and block ordering would become Jinja logic,
testable only through rendered HTML. Two defects this session were caught by unit-level
mutation checks that a template test could not have expressed.

### Data shape

```python
@dataclass(frozen=True)
class TeamBlock:
    team: str
    rows: list[dict]        # best `per_team`, league-ranked, total desc
    scored: int             # this team's rows AFTER the pool filter, before the
                            # per_team slice -- so a block reads "5 of 24", and
                            # under ?pool=hitter it reads "5 of 14"
    total: float            # sum over `rows` -- the best-N total, NOT the roster
    unscored: list[str]
    is_mine: bool

@dataclass(frozen=True)
class TeamsBoard:
    blocks: list[TeamBlock]
    ranked: int             # league-wide, so a block's #37 is readable
    base_season: int
    end_year: int
    end_years: list[int]
    pool: str
    scale: str
    per_team: int
    year_columns: list[int]
    #: True when `my_team` names no block. The counterpart of the league board's
    #: `has_rosters`: it drives a one-line banner saying the page could not find the
    #: reader's own roster, so ten unhighlighted blocks are explained rather than
    #: read as "you own none of these".
    mine_missing: bool
    meta: dict
```

`scored` is the team's count **after the pool filter and before the `per_team` slice**,
and `len(rows)` is what is shown, so a block says "5 of 24" -- and "5 of 14" under
`?pool=hitter`. Counting pre-pool would print a total the visible rows cannot add up to,
which is the same mismatch `unscored` had before it learned about `pool`.

### Membership, not a winner

A row is appended to **every** block whose team is in `index.owners_of[key]`. A name two
teams roster appears in both blocks, flagged -- the rule the dropdown already follows, and
the one whose absence caused the regression fixed in `06bf2646`.

`owner_ambiguous` is `True` for every ambiguous row here, because every row in this view
is being attributed to a team on screen. Same condition the filtered board uses.

### Blocks come from the rosters, not the rows

`index.teams` is roster-derived. This is #323's first named failure mode: a team whose
players were all filtered out -- by `--min-sgp`, by support, or by the join failing
wholesale -- has no scored rows, so building the block list from rows would drop the team
**and its unpriced list**, leaving nothing on screen to say it existed. Such a team sorts
last on a total of 0.0 and still renders.

### Route and template

`?view=teams` on `/trajectory`, clamped with `_clamp_choice(view, ("board", "teams"),
"board")`. The route branches once and renders `trajectory.html` or
`trajectory_teams.html`.

URL-driven, **not** client-side pills. `standings.html` toggles hidden divs with JS, but
`trajectory.html` documents the opposite rule -- every control is a URL built by
`board_url`, because encoding filter state in two mechanisms once caused filters to
silently reset -- and carries a `<noscript>` fallback. Following standings here would
break that rule inside the same page and put both views in one document. (Migrating
standings the other way is #338.)

`board_url` gains `view` and `per`.

| control | shared between views? |
|---|---|
| `end`, `scale`, `pool` | **shared** -- one range and one scale across the page, per #323 |
| `top` (50) vs `per` (5) | **independent** -- a shared binding means 5 collapses the league board, or 50 puts fifty players in every block |

`board_url` carries **both** `top` and `per` on **every** link, whichever view is
rendering. They are independent in what they control, not in when they are transmitted:
emitting `per` only on the teams view would silently reset it to 5 whenever the reader
flipped to the league board and back -- precisely the hidden-input failure the macro's
own comment records.

## Requirements

1. `?view=teams` renders one block per team in `index.teams`; absent or `?view=board`
   renders today's league board unchanged.
2. Blocks are ordered by best-N total descending, **tie-broken by team name
   ascending**, with my own block in that ordering. A tie is reachable -- two teams
   with no scored rows both total 0.0 -- and leaving it to dict order would make the
   page reorder between reads, which is the same arbitrary-ordering defect
   `index_rosters` was just fixed for (`06bf2646`).
3. Within a block, rows are ordered by total descending and carry **league** ranks.
4. `per_team` slices an already-ranked list; it never re-ranks.
5. A team with zero scored rows renders, sorts last, and shows its unpriced list.
6. `unscored` per block honours the `pool` filter.
7. A row whose key two teams roster appears in both blocks, flagged `owner_ambiguous`.
8. `end`, `scale` and `pool` survive a view switch; `top` and `per` do not bleed into each
   other.
9. `mine_missing` is True exactly when no block is marked `is_mine` -- including when
   `my_team` is None because the config read failed -- and drives a one-line banner. The
   page never renders ten unhighlighted blocks with no explanation.
10. A block renders the same per-year columns as the league board -- present when the
   range spans more than one year, absent when it does not -- driven by the same
   `year_columns`, so the two views cannot disagree about what a year column means.

## Edge cases and failure modes

| case | behaviour |
|---|---|
| No roster data (read failed, or returned `[]`) | `index.teams` is empty, so the view pills are not rendered -- same gate as the dropdown -- and `?view=teams` clamps back to `board`. A stale bookmark degrades to the league board rather than rendering an empty page. |
| `?view=` junk | Clamped to `board` by `_clamp_choice`, like `pool`/`scale`/`team`. |
| `?per=` junk or out of range | `_clamp(per, 1, 50, 5)`. The ceiling is 50 because a block is a summary; the dropdown already shows one roster in full. |
| A team rosters nobody the board scored | Renders with `scored=0`, `total=0.0`, empty `rows`, and its full unpriced list. Requirement 5. |
| Every team has zero scored rows | Every block renders empty. The page is honest rather than blank; nothing special-cased. |
| A name two teams roster | Appears in both blocks with `(?)`. Neither block's `scored` double-counts the other's. |
| My team is not in `index.teams` (rename, or config mismatch) | No block is marked `is_mine`. Blocks still render and sort normally, **and the page says so** -- see `mine_missing` below. Silently rendering ten unhighlighted blocks would read as "none of these are yours", which is a claim the page cannot support. |
| `pool` filter excludes a team's every row | Same as zero scored rows: block renders with its pool-filtered unpriced list. |

## Testing expectations

**Unit, `build_teams_board`.** Blocks ordered by best-N total with mine *in* that ordering
and not above it; **two teams tied on total order by name, and the order is identical when
`spots` is passed reversed** -- the assertion that makes the tie-break a rule rather than
a coincidence of dict order; a zero-scored team renders and sorts last; a block's top row
is not `#1` (league ranks survive); `per_team` slices without re-ranking (compare rank
sequences at two values of N); a colliding key appears in both blocks flagged; `unscored`
follows the pool filter; **`scored` counts the team's rows after the pool filter and
before the slice, so it changes under `?pool=hitter` while `len(rows)` may not**;
`mine_missing` is True when `my_team` matches no block and False when it matches one.

**Route.** `?view=teams` renders blocks; absent and `?view=board` render the league board;
`top` and `per` do not affect each other across a view switch; `?view=teams` with no
rosters falls back without 500ing; the view pills are absent when there is no roster data.

**Anti-vacuity.** The fixture needs two teams with **different** rosters, one team with
**zero** scored rows, and one unpriced player. Without all three, the ordering assertion,
the render-empty-team guard, and the unscored assertion cannot fail. The ordering test
must also place my team somewhere other than first on strength, or "mine is not promoted"
passes trivially. Every assertion that pins a rule gets a mutation check -- four vacuous
tests were shipped and caught this session, two of them by mutation.

## Phasing

One PR, three commits:

1. `TeamBlock` / `TeamsBoard` / `build_teams_board` plus unit tests. No caller yet, so it
   lands green and is independently revertible.
2. `board_url` gains `view` and `per` (threading only -- **no pills yet**). League board
   unchanged, proven by its existing tests.
3. The route branch, `trajectory_teams.html`, **and** the view pills, plus route tests.

Commit 1 is standalone. Commits 2 and 3 are the feature.

**The pills ship in the same commit as the route branch that answers them.** Splitting
them would put a visible control on the page one commit before `?view=teams` does
anything -- it would clamp back to `board`, so the pill would render, be clickable, and
appear broken. That is the same defect shape as `c96cd79b`, where a commit boundary drawn
between a change and its consumer left `/trajectory` returning 500 at that revision; the
lesson is not "do not change signatures" but "do not land a caller and its callee in
different commits", and a control is a caller.

No data migration and no push: the cached payload is untouched.
