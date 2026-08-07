# Trajectory board: team dropdown -- Design

Date: 2026-08-07
Status: approved design (brainstorming complete)
Author: session (Hart)
Supersedes: the tab-based approach in #322

## Problem

The trajectory board (`/trajectory`, #321) ranks every scorable player in the
league on multi-year VAR. It answers "who is worth the most over the years I
would hold them" across the whole pool, but there is no way to ask it about one
roster. The keep-or-cut call is made per team -- mine, or a trade partner's --
and today that question can only be answered from the CLI
(`scripts/trajectory_board.py --by-team`), which needs a terminal and a 50-second
sweep.

Issue #322 proposed a second **tab** showing my roster in full. Hart chose a
different shape during brainstorming: a **team dropdown on the existing board**,
defaulting to "All teams" (today's behaviour). This is strictly more useful --
it answers for every team rather than just mine -- and it adds one control
instead of a second view with its own layout, header, and empty states.

Because Top defaults to 50 and no roster carries more than ~26 scored players,
selecting a team shows that roster in full without touching the Top control.

## Goals

- A **team dropdown** on the trajectory board. Default "All teams" = current
  behaviour, byte for byte.
- Selecting a team narrows the board to that team's scored players.
- **My team sorts first among the teams**, directly below the default
  "All teams" entry; the rest alphabetical.
- **Ranks stay league-wide.** A selected team's best player reads `#37`, not
  `#1`.
- Roster players the model could not price are **named on screen**, not silently
  absent.
- The `(normalized_name, player_type)` roster join gets **one implementation**
  shared by the CLI and the web, rather than a second copy.

## Non-goals

- **No second tab, no second template.** Superseding #322's tab shape.
- **No IL/DTD status column.** `RosterSpot.status` will be in hand once spots
  reach the view, and #322 argued for showing it. Hart declined during
  brainstorming: the board answers a multi-year keeper question and a day-to-day
  tag is this-week information that does not change it.
- **No team-level headline total.** #322 raised it as an open question. Rejected:
  the analogous CLI headline was just fixed (`7e74b7b1`) because summing a whole
  roster of unclamped VAR is dominated by fringe negatives -- 93.5% of scored
  players carry a negative VAR, and roster tails run -62 to -196 against a best-5
  signal of 15 to 73. A whole-roster total on screen would reintroduce exactly
  the number that fix removed. If a headline is ever wanted it must be a fixed-N
  sum, and that is a separate decision.
- **No "keepers only" filter.** #322's other open question; not asked for.
- **No new compute.** This is a filter over the cached sweep plus a roster join.
  No refit, no new cache key, no change to `push_trajectory_board.py`.
- **No fix for the fragile join itself.** Roster blobs carry no `mlbam_id`
  (#284), so `(normalized_name, player_type)` is the only join available and it
  is not unique. This spec makes the ambiguity *visible and shared*; it does not
  resolve it. #284 remains the real fix.

## Chosen approach

Three pieces:

1. **`src/fantasy_baseball/trajectory/roster_join.py`** -- new shared module
   holding the roster-to-board join.
2. **`src/fantasy_baseball/web/trajectory_view.py`** -- `build_board` takes
   roster spots instead of a pre-computed ownership set, and gains a `team`
   filter.
3. **`src/fantasy_baseball/web/templates/season/trajectory.html`** + the
   `/trajectory` route -- the control and the unscored line.

`scripts/trajectory_board.py:assign_teams` is **deleted**; `by_team` calls the
shared module.

### Why the join moves to `src/`

Nothing under `src/` can import from `scripts/`. Today the only implementation of
this join is `scripts/trajectory_board.py:assign_teams`, so a web team filter
would need its own copy -- two implementations of a join already known to be
fragile (#284), free to drift on exactly the ambiguity handling that matters.

This is the same move `trajectory/sweep.py` records in its module docstring for
the same reason: *"`scripts/trajectory_board.py` owned this and nothing under
`src/` can import from `scripts/`, so the web board would have needed its own
copy -- two definitions of 'what is this player worth over three years', free to
drift."*

Rejected alternatives:

- **Build the team map inline in `trajectory_view.py`.** Smallest diff, but it is
  the second-copy outcome above.
- **Stamp `team` into the pushed payload.** The web would need no roster read at
  all, but the payload is an offline artifact pushed by hand while rosters change
  on every trade and waiver claim, so team data would be stale until the next
  sweep. The route's existing comment rules this out: *"LIVE rosters, not the
  local mirror: which players are mine is exactly the state that goes stale
  silently, and a trade since the last sync would mark the wrong rows."*

### `index_rosters` returns a lookup and mutates nothing

```python
def index_rosters(
    rows: Sequence[dict], spots: Sequence[RosterSpot], my_team: str | None
) -> RosterIndex
```

`RosterIndex` is a frozen dataclass:

| field | type | meaning |
|---|---|---|
| `team_of` | `dict[tuple[str, str], str]` | `(normalized_name, pool)` -> owning team |
| `ambiguous` | `set[tuple[str, str]]` | keys matching more than one board row |
| `unscored` | `dict[str, list[str]]` | roster players with no scored row, per team |
| `teams` | `tuple[str, ...]` | league teams, my team first then alphabetical; plain alphabetical when `my_team` is None or matches no team in `spots` |

**The no-mutation property is load-bearing, not stylistic.** `assign_teams`
stamps `row["team"]` in place. That is safe for rows the CLI just built, but the
web's rows come from `_ranked_rows`, whose cache comment states: *"Rows are never
mutated after `add_ranks` -- `build_board` copies each into a new dict -- which
is what makes sharing them across requests safe."* A shared mutating helper would
be correct in one caller and a cross-request data race in the other. Returning a
lookup makes that hazard unreachable rather than merely avoided.

`by_team` keeps stamping its own rows; it just reads the values off the index.

### View changes

`build_board`'s `mine: set[tuple[str, str]] | None` parameter is **replaced** by
`spots: Sequence[RosterSpot] | None` plus `my_team: str | None`, and a new
`team: str = "all"`.

`build_board` calls `index_rosters` on the ranked rows and derives both the
ownership highlighting and the team filter from that one index. This removes the
route's knowledge of how to turn roster spots into board facts -- today the route
builds the `mine` set with its own comprehension and its own notion of which
spots count.

The team filter runs **in the existing row loop, beside the `pool` filter** --
deliberately not in the `_ranked_rows` cache key, which is `(horizons, scale)`.
Adding team would multiply cached entries by ~11 for a filter that is one
comparison per row.

`Board` gains three fields: `team: str`, `teams: tuple[str, ...]`, and
`unscored: list[str]` (for the selected team; empty when "all").

**`has_rosters` keeps its current meaning and does not follow `spots`.** Today it
is `bool(mine)` -- "my team joined at least one scored row" -- and it gates the
banner "Your players are not highlighted". Deriving it from `spots` instead would
change the predicate to "any roster data arrived", so a successful read where my
own roster joined nothing would suppress the banner while nothing was in fact
highlighted. That is the state the field's docstring exists to prevent: *"An
empty result counts as a failure ... an empty set cannot be told apart from an
unreachable Upstash."*

These are now two distinct facts and get two expressions:

- `has_rosters` = my team joined >= 1 row -> gates the highlight banner
  (unchanged behaviour).
- non-empty `spots` -> gates whether the dropdown renders at all.

`scored` naturally becomes the selected team's row count while `ranked` stays
league-wide. The template's existing sentence -- *"ranked against all 1,169 -- the
# column is a league-wide position, so it runs past this count"* -- already
renders that correctly with no change.

### Route and template

The route drops its `mine` comprehension and passes `spots` and `my_team`
through.

`board_url` gains a `team` parameter. It is the single place that knows the full
filter state, and its comment records why: *"The selects previously lived in a
`<form>` and needed a hidden input per filter to carry the ones they did not own
-- so the state was encoded twice, in two mechanisms, and adding a filter meant
remembering both. Missing the hidden input silently reset that filter whenever a
dropdown changed."*

The control is a `<select onchange="location = this.value">` matching Through and
Top, with "All teams" first (value `all`), then `board.teams` in order. It is
also added to the `<noscript>` pill fallback: Through and Top both have one, and
leaving a single control JS-only is the inconsistency that bites later.

When a team is selected and it has unscored players, beneath the table:

> not scored: Jacob Misiorowski, Roman Anthony -- no 2026 line or pacing under
> 0.0 SGP, so the model could not price them.

Rendered only when a team is selected and the list is non-empty. Dropping it
would turn "the model could not price these" into "these are not worth listing",
which on a keeper board is the most misleading place for that confusion. #322
states this as a hard requirement.

**It renders OUTSIDE the `{% if not board.rows %}` / `{% else %}` pair, after
it** -- not inside the table branch. The template today renders a bare "Nothing
scored at this timeframe." when there are no rows (trajectory.html:108), so a
line placed "beneath the table" would vanish for a team with zero scored
players, which is precisely the team whose unscored list explains the empty
page.

## Requirements

1. Default state (`?team` absent or `all`) produces a board identical to today's.
2. `?team=<name>` narrows rows to players whose roster spot names that team.
3. `rank_total` and `rank_next` are unchanged by the team filter.
4. The dropdown lists "All teams" first, then my team, then the rest
   alphabetically.
5. `team` composes with `pool`, `scale`, `end`, and `top` -- any combination is
   reachable and each control preserves the others.
6. Ambiguous rows (a `(name, pool)` key matching >1 board row) appear under the
   team, flagged `(?)`. The row field is **`owner_ambiguous`**, replacing
   `mine_ambiguous`: true when the key is ambiguous **and** the row is being
   attributed to a team on screen. In the "All teams" view that means `mine`
   only, which is exactly today's behaviour; in a team view it means any row
   shown, because attributing an opponent's player is as wrong as attributing
   mine. The template condition moves with the field name -- **and so does the
   tooltip copy**, which today reads "*You* roster a player with this name..."
   (trajectory.html:137). Under a team filter that string would tell the reader
   he rosters an opponent's player. It becomes owner-relative, naming the team
   the row is attributed to, and keeps the #284 explanation.
7. When a team is selected, its unscored roster players are named beneath the
   table.
8. `by_team` in the CLI produces the same team assignment as the web. This is
   **structural, not tested**: it holds because `assign_teams` is deleted and
   `index_rosters` is the only implementation left. The deletion is the
   evidence; a test asserting two callers of one function agree would assert
   nothing. Stated so a future reader does not mistake the missing test for an
   oversight.

## Edge cases and failure modes

| case | behaviour |
|---|---|
| Roster read fails or returns `[]` | `has_rosters` False; **the dropdown is not rendered at all** rather than shown and broken. The existing banner ("Your players are not highlighted -- the live roster read failed") already explains. A `team=` in the URL falls back to `all`, since it cannot be honoured. |
| `?team=` names a team that does not exist (typo, traded-away franchise, stale bookmark) | Falls back to `all`. Same clamp `pool`, `scale`, and `end` already apply; `test_a_junk_or_out_of_range_end_year_falls_back_instead_of_500ing` pins that these params are user-editable. |
| Selected team has zero scored rows | The existing "Nothing scored at this timeframe." branch renders. The unscored line still shows, so the page explains itself rather than looking broken. |
| Two board rows share a `(name, pool)` key and one is rostered | Both render under that team with `(?)`. Hiding the one you own is worse than showing one you do not, clearly marked. Consistent with today's `mine_ambiguous`. |
| A rostered player is scored but his roster spot does not join (accent, suffix, nickname) | Both halves of the same miss happen at once: the board row appears **only** under "All teams", and the roster spot appears in that team's `unscored` line under its roster spelling. So one human shows up twice, spelled two ways, in two places. Accepted and deliberately visible -- this is #284's fragility surfaced by requirement 7 rather than hidden. NOT a different set from `unscored`: a spot that found no row IS an unscored spot, by construction. |
| Roster read succeeds but my own team joins nothing | `has_rosters` False (banner shown, nothing highlighted); dropdown still renders, because `spots` is non-empty and other teams are selectable. The two conditions are deliberately separate -- see "View changes". |
| `my_team` is None (config read failed) or names no team in `spots` | `teams` is plain alphabetical with nothing promoted; `has_rosters` False. The dropdown still works for every other team. |
| Team names contain spaces, apostrophes, `!` (`Jon's Underdogs`, `Hello Peanuts!`) | Carried as URL-encoded query values by `url_for`; compared as exact strings against `RosterSpot.team`. No normalization -- the roster blob is the only source of both sides. |

## Testing expectations

**`index_rosters` (unit).** My team sorts first; ambiguity detected; unscored
grouped per team; and **the input rows are not mutated** -- that last assertion
is the entire reason for the return-a-lookup shape, so it is pinned rather than
left to a comment.

**`build_board` (unit).** The team filter narrows rows; a selected team's top row
is **not** `#1`, proving league ranks survive; an unknown team falls back to
`all`; `team` and `pool` compose; `unscored` is populated only when a team is
selected.

**Route/template.** The dropdown renders with my team first; it is absent when
**`spots` is empty** -- not when `has_rosters` is False, which is now a different
condition. Both sides of that split get a test: read-failed (empty `spots`, no
dropdown) and read-succeeded-but-my-roster-joined-nothing (`has_rosters` False,
dropdown still rendered, banner still shown). A `?team=` with no rosters does not
500.

**CLI parity.** The three `by_team` tests added in `7e74b7b1` must stay green
through the `assign_teams` deletion -- but they are **not sufficient**, and the
spec previously claimed they were. All three call `by_team(scored, spots, {}, ...)`
with `missing` hardcoded empty (test_trajectory_board_cli.py:80, 101, 119), and
`missing` is precisely what `assign_teams` returns and what the CLI's
`not scored:` line consumes. The one output being deleted and re-homed therefore
has **zero** coverage today.

So commit 1 additionally requires a new test: `by_team` prints
`not scored: <names>` for a roster spot with no matching board row, driven
through the real `index_rosters` rather than a hand-passed `missing` dict.
Without it the CLI could silently stop reporting unscored players with a green
suite.

**Anti-vacuity.** The fixture must carry two teams with *different* rosters, one
unscored player, and an ambiguous name **on an opponent's roster, not mine**.
Without all three the filter, flag, and unscored assertions cannot fail -- and
the ambiguous name specifically must not sit on my team, or the fixture only
exercises today's `mine`-only path and the `owner_ambiguous` extension (with its
wrong-tooltip failure) ships untested. Two vacuous tests were shipped and caught by
mutation earlier in this session (an alignment test where every cell was already
column width, and a same-slot ordering test whose fixture had uniform year
counts); the assertions that matter here get the same mutation check.

## Phasing

One PR, three commits:

1. `roster_join.py` plus its unit tests; `assign_teams` deleted and `by_team`
   switched over. CLI behaviour unchanged -- proven by the existing `by_team`
   tests **plus** the new `not scored:` test described under Testing, which is
   the gate for this commit. The existing three do not touch `missing` and
   cannot catch its loss.
2. `build_board` takes `spots`/`my_team`/`team`; `Board` gains its three fields;
   view tests.
3. Route, `board_url`, dropdown, unscored line, noscript fallback; route tests.

Commit 1 stands alone and is independently revertible. Commits 2 and 3 are the
feature.

No data migration, no push required -- the cached payload is untouched.
