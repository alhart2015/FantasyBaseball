# Trajectory All-Teams View Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Source spec:** `docs/superpowers/specs/2026-08-07-trajectory-all-teams-view-design.md`

**Goal:** Add a second view to `/trajectory` showing the best N players on every team, one block per team, ordered by strength.

**Architecture:** A second view function, `build_teams_board`, sits beside `build_board` in `trajectory_view.py` and shares the same `_ranked_rows` cache and `index_rosters` join — it groups where `build_board` flattens. The view is selected by `?view=teams`, not by client-side toggling, because `trajectory.html` documents an all-controls-are-URLs rule and carries a `<noscript>` fallback. The control bar becomes a shared Jinja partial taking explicit arguments, since `TeamsBoard` deliberately does not carry `Board`'s `top`/`team`/`teams`.

**Tech Stack:** Python 3.11, Flask + Jinja2, pytest. No new dependencies.

## Global Constraints

- **ASCII only** in source, templates, log messages, and anything reaching `print()`. Windows box, cp1252 stdout.
- **Roster identity is `(normalized_name, player_type)`** — roster blobs carry no `mlbam_id` (#284). Never key on a bare name.
- **Ownership is MEMBERSHIP**, never a winner: a row belongs to every team in `index.owners_of[key]`. Deriving it from a single winning spot caused the regression fixed in `06bf2646`.
- **Ranks are league-wide.** Never re-rank inside a group or a filter.
- **Never mutate rows returned by `_ranked_rows`** — they are shared across requests; copy into a new dict.
- **Never `x or default` for numeric defaults** — `0`, `0.0`, `""` are falsy.
- Tests are the guardrail: no assertion may be loosened, skipped, or deleted to make something pass. A test that cannot fail is a defect.
- Verification for every task: `pytest tests/test_web/ tests/test_trajectory/ -q`, `ruff check .`, `ruff format --check .`, `mypy`, `vulture`. Pre-existing `resend` `ModuleNotFoundError` failures in `test_send_daily_summary.py` / `test_summary` are unrelated — do not "fix" them.

---

### Task 1: `build_teams_board` and its dataclasses

**Files:**
- Modify: `src/fantasy_baseball/web/trajectory_view.py`
- Test: `tests/test_web/test_trajectory_view.py`

**Interfaces:**
- Consumes: `_ranked_rows(payload, horizons, scale)`, `_clamp`, `_clamp_choice`, `_year_cells`, `rank_move`, `SCALES`, and `index_rosters` -> `RosterIndex` with `.teams`, `.owners_of`, `.ambiguous`, `.unscored_for(team, pool)`, `.matched_teams`.
- Produces: `DEFAULT_PER_TEAM = 5`, `TeamBlock`, `TeamsBoard`, and `build_teams_board(payload, *, end, pool, scale, spots, my_team, per_team) -> TeamsBoard`. Tasks 2 and 3 depend on these names.

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_web/test_trajectory_view.py`. The module already provides the `payload` fixture (six players) and a module-level `_spot(name, team, pool="hitter", status="")` helper — use them, do not rebuild either.

```python
def _teams_fixture():
    """Rosters for the `payload` fixture, shaped so every assertion below can fail.

    THREE things are load-bearing and none may be dropped:
      * two teams with DIFFERENT rosters, or the ordering assertion is trivial;
      * "Mine" is deliberately the WEAKER of the two, so "my block is not promoted
        to the top" cannot pass by accident;
      * two teams with NO scored rows ("Empty FC", "Aardvark FC"), which is both
        the render-an-empty-team case and the only way to reach a total tie.
    """
    return [
        _spot("Big Bat", "Rivals"),
        _spot("Big Arm", "Rivals", pool="pitcher"),
        _spot("Small Bat", "Mine"),
        _spot("Small Arm", "Mine", pool="pitcher"),
        _spot("Never Scored", "Mine"),
        _spot("Ghost", "Empty FC"),
        _spot("Phantom", "Aardvark FC"),
    ]


def test_blocks_are_ordered_by_strength_and_mine_is_not_promoted(payload: dict) -> None:
    """The comparison IS the view. Sorting my own block to the top would destroy
    the ordering that justifies including it at all."""
    board = build_teams_board(payload, spots=_teams_fixture(), my_team="Mine")

    names = [b.team for b in board.blocks]
    assert names[0] == "Rivals", "the strongest roster reads first"
    assert names.index("Mine") == 1, "mine sits where its strength puts it"
    assert [b.total for b in board.blocks] == sorted(
        (b.total for b in board.blocks), reverse=True
    )
    assert next(b for b in board.blocks if b.is_mine).team == "Mine"
    assert not board.mine_missing


def test_teams_tied_on_total_are_ordered_by_name_not_by_roster_order(payload: dict) -> None:
    """Two teams with nothing scored both total 0.0. Left to dict order the page
    would reorder between reads -- the arbitrary-ordering defect `index_rosters`
    was fixed for in 06bf2646."""
    spots = _teams_fixture()
    forward = build_teams_board(payload, spots=spots, my_team="Mine")
    reverse = build_teams_board(payload, spots=list(reversed(spots)), my_team="Mine")

    empties = [b.team for b in forward.blocks if b.total == 0.0]
    assert empties == ["Aardvark FC", "Empty FC"], "ties break on name, ascending"
    assert [b.team for b in reverse.blocks] == [b.team for b in forward.blocks]


def test_a_team_with_nothing_scored_still_renders_with_its_unpriced_list(payload: dict) -> None:
    """#323's first named failure mode. Building the block list from the ROWS would
    drop this team and its unpriced list, leaving nothing on screen to say it exists."""
    board = build_teams_board(payload, spots=_teams_fixture(), my_team="Mine")

    ghost = next(b for b in board.blocks if b.team == "Empty FC")
    assert ghost.rows == []
    assert ghost.scored == 0
    assert ghost.total == 0.0
    assert ghost.unscored == ["Ghost"]
    assert board.blocks[-1].total == 0.0, "and it sorts last"


def test_a_blocks_rows_carry_league_ranks(payload: dict) -> None:
    """Within-team ranks would make every team's best player read #1."""
    league = {r["name"]: r["rank_total"] for r in build_board(payload, top="all").rows}
    board = build_teams_board(payload, spots=_teams_fixture(), my_team="Mine")

    for block in board.blocks:
        for row in block.rows:
            assert row["rank_total"] == league[row["name"]]
    assert any(b.rows and b.rows[0]["rank_total"] != 1 for b in board.blocks)


def test_per_team_slices_without_re_ranking(payload: dict) -> None:
    """N is a slice of an already-ranked list, so the first row must not move."""
    one = build_teams_board(payload, spots=_teams_fixture(), my_team="Mine", per_team=1)
    two = build_teams_board(payload, spots=_teams_fixture(), my_team="Mine", per_team=2)

    rivals_one = next(b for b in one.blocks if b.team == "Rivals")
    rivals_two = next(b for b in two.blocks if b.team == "Rivals")
    assert len(rivals_one.rows) == 1
    assert len(rivals_two.rows) == 2
    assert rivals_one.rows[0]["name"] == rivals_two.rows[0]["name"]
    assert rivals_one.scored == rivals_two.scored, "scored is the team's set, not the slice"


def test_scored_follows_the_pool_filter_and_unscored_does_too(payload: dict) -> None:
    """A block that says "5 of 24" under a hitters-only table must mean 24 hitters,
    or the visible rows cannot add up to the number printed beside them."""
    both = build_teams_board(payload, spots=_teams_fixture(), my_team="Mine")
    hitters = build_teams_board(
        payload, spots=_teams_fixture(), my_team="Mine", pool="hitter"
    )

    assert next(b for b in both.blocks if b.team == "Rivals").scored == 2
    assert next(b for b in hitters.blocks if b.team == "Rivals").scored == 1
    assert next(b for b in both.blocks if b.team == "Mine").unscored == ["Never Scored"]
    assert next(b for b in hitters.blocks if b.team == "Mine").unscored == ["Never Scored"]


def test_a_name_two_teams_roster_appears_in_both_blocks_flagged(payload: dict) -> None:
    """Membership, not a winner. Attributing the row to one team would take the
    other owner's player off his own block with nothing on screen to say so."""
    spots = [*_teams_fixture(), _spot("Big Bat", "Mine")]
    board = build_teams_board(payload, spots=spots, my_team="Mine")

    holders = [b.team for b in board.blocks if any(r["name"] == "Big Bat" for r in b.rows)]
    assert sorted(holders) == ["Mine", "Rivals"]
    for block in board.blocks:
        for row in block.rows:
            if row["name"] == "Big Bat":
                assert row["owner_ambiguous"], "the board cannot tell which Big Bat"


def test_mine_missing_when_my_team_names_no_block(payload: dict) -> None:
    """Ten unhighlighted blocks read as "you own none of these" -- a claim the page
    cannot support when the truth is that it never found the reader's roster."""
    board = build_teams_board(payload, spots=_teams_fixture(), my_team="Renamed FC")
    assert board.mine_missing
    assert not any(b.is_mine for b in board.blocks)

    assert not build_teams_board(payload, spots=_teams_fixture(), my_team="Mine").mine_missing
    assert build_teams_board(payload, spots=_teams_fixture(), my_team=None).mine_missing


def test_per_team_and_end_year_clamp_junk_from_the_query_string(payload: dict) -> None:
    """These arrive from a URL a reader can edit."""
    board = build_teams_board(payload, spots=_teams_fixture(), my_team="Mine", per_team="junk")
    assert board.per_team == 5
    assert build_teams_board(payload, spots=_teams_fixture(), per_team=0).per_team == 1
    assert build_teams_board(payload, spots=_teams_fixture(), per_team=999).per_team == 50
    assert build_teams_board(payload, spots=_teams_fixture(), end="nonsense").end_year == 2027
```

Add `build_teams_board` to the module's import line at the top of the test file:

```python
from fantasy_baseball.web.trajectory_view import DEFAULT_TOP, build_board, build_teams_board
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `pytest tests/test_web/test_trajectory_view.py -q -k "block or team or mine_missing or per_team"`
Expected: FAIL at collection — `ImportError: cannot import name 'build_teams_board'`.

- [ ] **Step 3: Add the dataclasses**

In `trajectory_view.py`, after the `Board` class, add:

```python
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
    #: Sum over `rows` -- the BEST-N total, never the roster total. 7e74b7b1 removed
    #: the roster version from the CLI after measuring it: 93.5% of scored players
    #: carry a negative VAR and tails run -62 to -196 against a best-5 signal of 15 to
    #: 73, so as a sort key it orders the page by depth of junk.
    total: float
    unscored: list[str]
    is_mine: bool


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
    #: True when `my_team` names no block -- a rename, a config mismatch, or a failed
    #: config read. Drives a one-line banner, so unhighlighted blocks are explained
    #: rather than read as "you own none of these".
    mine_missing: bool
    meta: dict = field(default_factory=dict)

    @property
    def span(self) -> str:
        """Same label the league board prints, from the same rule."""
        start = self.base_season + 1
        return f"{start}" if self.end_year == start else f"{start}-{str(self.end_year)[-2:]}"
```

- [ ] **Step 4: Add `build_teams_board`**

Below `build_board`:

```python
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
    base = int(payload["base_season"])
    max_horizon = int(payload["max_horizon"])
    end_years = [base + h for h in range(1, max_horizon + 1)]

    end_year = _clamp(end, end_years[0], end_years[-1], end_years[0])
    pool = _clamp_choice(pool, ("both", "hitter", "pitcher"), "both")
    scale = _clamp_choice(scale, SCALES, "var")
    n = _clamp(per_team, 1, 50, DEFAULT_PER_TEAM)
    horizons = tuple(range(1, end_year - base + 1))

    ranked_rows = _ranked_rows(payload, horizons, scale)
    index = index_rosters(ranked_rows, spots or [], my_team)

    # BLOCKS COME FROM THE ROSTERS, not the rows. A team whose players were all
    # filtered out has no rows at all, and deriving the block list from `ranked_rows`
    # would drop the team AND its unpriced list -- leaving nothing on screen to say
    # it exists. #323 names this as the failure mode the CLI already guards.
    grouped: dict[str, list[dict]] = {team: [] for team in index.teams}

    for row in ranked_rows:
        if pool != "both" and row["pool"] != pool:
            continue
        key = (normalize_name(row["name"]), row["pool"])
        owners = index.owners_of.get(key, frozenset())
        if not owners:
            continue
        cell = {
            **row,
            "mine": my_team is not None and my_team in owners,
            # Every row here is attributed to a team on screen, so an unresolvable
            # key is always a guess worth flagging -- no `team != "all"` condition
            # to apply, unlike the league board.
            "owner_ambiguous": key in index.ambiguous,
            "rank_move": rank_move(row),
            "year_cells": _year_cells(row["by_year"], horizons),
        }
        for team in owners:
            if team in grouped:
                grouped[team].append(cell)

    blocks = []
    for team, rows in grouped.items():
        # `add_ranks` ordered by total descending, so ranking IS the sort order.
        rows.sort(key=lambda r: r["rank_total"])
        shown = rows[:n]
        blocks.append(
            TeamBlock(
                team=team,
                rows=shown,
                scored=len(rows),
                total=sum(r["total"] for r in shown),
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
        mine_missing=not any(b.is_mine for b in blocks),
        meta={
            "generated_at": payload.get("generated_at"),
            "panel_vintage": payload.get("panel_vintage"),
            "season_elapsed": payload.get("season_elapsed"),
            "min_sgp": payload.get("min_sgp"),
            "floors": payload.get("floors", {}),
            "excluded": payload.get("excluded", {}),
        },
    )
```

- [ ] **Step 5: Run the tests to verify they pass**

Run: `pytest tests/test_web/test_trajectory_view.py -q`
Expected: PASS, all of them.

- [ ] **Step 6: Mutation-check the three rules that matter**

Each of these must FAIL the named test, then be reverted:

1. Change `blocks.sort(key=lambda b: (-b.total, b.team))` to `blocks.sort(key=lambda b: -b.total)` — `test_teams_tied_on_total_are_ordered_by_name_not_by_roster_order` must fail.
2. Change `grouped = {team: [] for team in index.teams}` to `grouped = {}` plus `grouped.setdefault(team, []).append(cell)` in the loop — `test_a_team_with_nothing_scored_still_renders_with_its_unpriced_list` must fail.
3. Change `scored=len(rows)` to `scored=len(shown)` — `test_per_team_slices_without_re_ranking` must fail.

Report all three transcripts. A guard that cannot fail is not a guard; four vacuous tests were shipped in this codebase in one session.

- [ ] **Step 7: Run the checks and commit**

Run: `pytest tests/test_web/ tests/test_trajectory/ -q && ruff check . && ruff format --check . && mypy && vulture`

`vulture` may report `TeamsBoard.span` and `TeamBlock` fields as unused until Task 3 renders them; that is expected and resolves there.

```bash
git add src/fantasy_baseball/web/trajectory_view.py tests/test_web/test_trajectory_view.py
git commit -m "feat(trajectory): build_teams_board, the same sweep grouped by team"
```

---

### Task 2: Extract the control bar into a shared partial

**Files:**
- Create: `src/fantasy_baseball/web/templates/season/_trajectory_controls.html`
- Modify: `src/fantasy_baseball/web/templates/season/trajectory.html`
- Modify: `src/fantasy_baseball/web/season_routes.py` (the `/trajectory` route's `render_template` call)
- Test: `tests/test_web/test_season_routes.py`

**Interfaces:**
- Consumes: nothing from Task 1.
- Produces: `_trajectory_controls.html` exposing two macros — `board_url(...)` and `controls(...)` — both taking every value explicitly. Task 3's template imports the same file.

**Why explicit arguments and not `board.*`:** the current bar reads `board.end_years`, `board.top`, `board.team`, `board.teams`. `TeamsBoard` carries none of `top`, `team`, `teams` — by design, since they mean nothing in a grouped view. A partial reading `board.*` would therefore break the moment Task 3 renders it.

**This task adds NO pills and changes NO rendered output.** It is a pure extraction, proven by the league board's existing tests passing untouched.

- [ ] **Step 1: Write the failing test**

Append to `tests/test_web/test_season_routes.py`:

```python
def test_trajectory_controls_carry_every_filter_on_every_link(client):
    """`board_url` is the single place that knows full filter state. A filter it
    does not emit gets silently reset when any other control is used -- the
    hidden-input failure the macro's own comment records.

    `view` and `per` are threaded now so the link shape is right before Task 3
    gives them anything to select.
    """
    with (
        patch(
            "fantasy_baseball.web.season_routes.read_cache_dict",
            return_value=_trajectory_payload(),
        ),
        patch(
            "fantasy_baseball.data.rosters.live_rosters",
            return_value=_trajectory_spots(),
        ),
    ):
        resp = client.get("/trajectory?end=2028&pool=hitter&scale=sgp&top=25&per=3")
    assert resp.status_code == 200
    body = resp.data.decode()

    # Every control link must carry all six, or using one resets the others.
    import re

    links = re.findall(r'href="(/trajectory\?[^"]*)"', body)
    assert links, "the control bar rendered no links"
    for link in links:
        for param in ("end=", "pool=", "scale=", "top=", "per=", "view="):
            assert param in link, f"{param} missing from {link}"
```

- [ ] **Step 2: Run it to verify it fails**

Run: `pytest tests/test_web/test_season_routes.py::test_trajectory_controls_carry_every_filter_on_every_link -q`
Expected: FAIL — `per=` and `view=` are not in any link yet.

- [ ] **Step 3: Create the partial**

Create `src/fantasy_baseball/web/templates/season/_trajectory_controls.html`:

```jinja
{#- The trajectory page's control bar, shared by the league board and the all-teams
    view.

    EVERY MACRO TAKES ITS VALUES EXPLICITLY rather than reading `board.*`. The two
    views carry different objects: `Board` has `top`/`team`/`teams`, `TeamsBoard` has
    `per_team` and none of those, because they mean nothing in a grouped view. A
    partial reaching into `board` would work for one caller and break for the other.

    `board_url` remains the single place that knows the full filter state. The selects
    once lived in a <form> with a hidden input per filter, so the state was encoded
    twice and missing an input silently reset that filter whenever a dropdown changed.
    Every parameter is therefore emitted on every link, whichever view is rendering --
    `per` included on the league board, `top` included on the teams view. -#}

{% macro board_url(cur, end=None, pool=None, top=None, scale=None, team=None,
                   view=None, per=None) -%}
{{ url_for('trajectory',
           end=end if end is not none else cur.end_year,
           pool=pool if pool is not none else cur.pool,
           top=top if top is not none else cur.top,
           scale=scale if scale is not none else cur.scale,
           team=team if team is not none else cur.team,
           view=view if view is not none else cur.view,
           per=per if per is not none else cur.per) }}
{%- endmacro %}

{% macro controls(cur, end_years, teams) -%}
<div class="trajectory-controls">
  <label>Through
    <select onchange="location = this.value">
      {% for y in end_years %}
      <option value="{{ board_url(cur, end=y) }}" {% if y == cur.end_year %}selected{% endif %}>{{ y }}</option>
      {% endfor %}
    </select>
  </label>
  {% if cur.view == 'board' %}
  <label>Top
    <select onchange="location = this.value">
      {% for n in [25, 50, 100, 250, 'all'] %}
      <option value="{{ board_url(cur, top=n) }}" {% if n == cur.top %}selected{% endif %}>{{ n }}</option>
      {% endfor %}
    </select>
  </label>
  {% if teams %}
  <label>Team
    <select onchange="location = this.value">
      <option value="{{ board_url(cur, team='all') }}" {% if cur.team == 'all' %}selected{% endif %}>All teams</option>
      {% for t in teams %}
      <option value="{{ board_url(cur, team=t) }}" {% if t == cur.team %}selected{% endif %}>{{ t }}</option>
      {% endfor %}
    </select>
  </label>
  {% endif %}
  {% else %}
  <label>Per team
    <select onchange="location = this.value">
      {% for n in [3, 5, 10, 25] %}
      <option value="{{ board_url(cur, per=n) }}" {% if n == cur.per %}selected{% endif %}>{{ n }}</option>
      {% endfor %}
    </select>
  </label>
  {% endif %}
  <span class="pill-group">
    {% for value, label in [('both', 'Both'), ('hitter', 'Hitters'), ('pitcher', 'Pitchers')] %}
    <a class="pill {% if cur.pool == value %}active{% endif %}" href="{{ board_url(cur, pool=value) }}">{{ label }}</a>
    {% endfor %}
  </span>
  <span class="pill-group">
    {% for value, label in [('var', 'VAR'), ('sgp', 'SGP')] %}
    <a class="pill {% if cur.scale == value %}active{% endif %}" href="{{ board_url(cur, scale=value) }}">{{ label }}</a>
    {% endfor %}
  </span>
  <noscript>
    <span class="pill-group">
      {% for y in end_years %}
      <a class="pill {% if y == cur.end_year %}active{% endif %}" href="{{ board_url(cur, end=y) }}">{{ y }}</a>
      {% endfor %}
    </span>
    {% if cur.view == 'board' %}
    <span class="pill-group">
      {% for n in [25, 50, 100, 250, 'all'] %}
      <a class="pill {% if n == cur.top %}active{% endif %}" href="{{ board_url(cur, top=n) }}">{{ n }}</a>
      {% endfor %}
    </span>
    {% if teams %}
    <span class="pill-group">
      <a class="pill {% if cur.team == 'all' %}active{% endif %}" href="{{ board_url(cur, team='all') }}">All teams</a>
      {% for t in teams %}
      <a class="pill {% if t == cur.team %}active{% endif %}" href="{{ board_url(cur, team=t) }}">{{ t }}</a>
      {% endfor %}
    </span>
    {% endif %}
    {% else %}
    <span class="pill-group">
      {% for n in [3, 5, 10, 25] %}
      <a class="pill {% if n == cur.per %}active{% endif %}" href="{{ board_url(cur, per=n) }}">{{ n }}</a>
      {% endfor %}
    </span>
    {% endif %}
  </noscript>
</div>
{%- endmacro %}
```

- [ ] **Step 4: Have the route pass a `cur` state object**

In `season_routes.py`, the `/trajectory` route builds a small dict carrying the full filter state and passes it to the template. Replace the `render_template` call's arguments so it reads:

```python
        return render_template(
            "season/trajectory.html",
            meta=read_meta(),
            active_page="trajectory",
            board=board,
            error=error,
            # The full filter state in ONE object, so the control macros never have to
            # reach into a view model that may not carry a given field.
            cur={
                "end_year": board.end_year if board else 0,
                "pool": board.pool if board else "both",
                "top": board.top if board else DEFAULT_TOP,
                "scale": board.scale if board else "var",
                "team": board.team if board else "all",
                "view": "board",
                # Raw here on purpose: the Per-team select only renders on the teams
                # view, and `build_teams_board` owns the clamp. Clamping in two places
                # is how the two spellings drift.
                "per": request.args.get("per", DEFAULT_PER_TEAM),
            },
        )
```

Add the imports the route now needs, beside the existing `build_board` import inside the route function:

```python
        from fantasy_baseball.web.trajectory_view import (
            DEFAULT_PER_TEAM,
            DEFAULT_TOP,
            build_board,
        )
```

- [ ] **Step 5: Point `trajectory.html` at the partial**

At the top of `trajectory.html`, replace the local `board_url` macro definition (lines 5-12) with an import, and replace the whole `<div class="trajectory-controls">...</div>` block (lines 47-102) with a call:

```jinja
{% import "season/_trajectory_controls.html" as ctl %}
```

```jinja
{{ ctl.controls(cur, board.end_years, board.teams) }}
```

Then replace every remaining `board_url(...)` call in the file with `ctl.board_url(cur, ...)`. Find them with `grep -n "board_url" src/fantasy_baseball/web/templates/season/trajectory.html` — after removing the controls block there should be none left, but check rather than assume.

- [ ] **Step 6: Run the tests**

Run: `pytest tests/test_web/ -q`
Expected: PASS. The league board's existing route and view tests are the proof this extraction changed no rendered behaviour — if any of them fail, the extraction is wrong, not the test.

- [ ] **Step 7: Run the checks and commit**

Run: `ruff check . && ruff format --check . && mypy && vulture`

```bash
git add src/fantasy_baseball/web/templates/season/_trajectory_controls.html \
        src/fantasy_baseball/web/templates/season/trajectory.html \
        src/fantasy_baseball/web/season_routes.py \
        tests/test_web/test_season_routes.py
git commit -m "refactor(trajectory): control bar becomes a shared partial taking explicit state"
```

---

### Task 3: The route branch, the teams template, and the view pills

**Files:**
- Create: `src/fantasy_baseball/web/templates/season/trajectory_teams.html`
- Modify: `src/fantasy_baseball/web/templates/season/_trajectory_controls.html` (add the view pills)
- Modify: `src/fantasy_baseball/web/season_routes.py`
- Test: `tests/test_web/test_season_routes.py`

**Interfaces:**
- Consumes: `build_teams_board`, `TeamsBoard`, `TeamBlock`, `DEFAULT_PER_TEAM` from Task 1; `ctl.controls` / `ctl.board_url` from Task 2.
- Produces: `VIEWS = ("board", "teams")` and `select_view(value: Any) -> str` in
  `trajectory_view.py` (defined in Step 3), plus the `trajectory_teams.html` template.
  `select_view` is public because the route must branch BEFORE it can build a view
  model, so unlike `pool` or `scale` it cannot read the clamped value off the result.

**The pills ship HERE, with the route branch that answers them.** Landing them in Task 2 would put a clickable control on the page one commit before `?view=teams` does anything — it would clamp straight back to `board`, so the pill would render and appear broken. That is `c96cd79b`'s defect shape: a caller and its callee in different commits.

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_web/test_season_routes.py`:

```python
def test_trajectory_teams_view_renders_one_block_per_team(client):
    with (
        patch(
            "fantasy_baseball.web.season_routes.read_cache_dict",
            return_value=_trajectory_payload(),
        ),
        patch(
            "fantasy_baseball.data.rosters.live_rosters",
            return_value=_trajectory_spots(),
        ),
    ):
        resp = client.get("/trajectory?view=teams")
    assert resp.status_code == 200
    body = resp.data.decode()

    # NOT a bare `"Hart of the Order" in body` -- that string is also the site header,
    # so it is present whether or not a single block rendered. Assert on markup only a
    # block emits. (The same trap made a dropdown-ordering test vacuous on #336.)
    assert body.count("from the best 5") == 2, "one header per team block"
    assert "Aardvarks" in body
    assert "Testy McTestface" in body, "a block's rows render"


def test_trajectory_defaults_to_the_league_board(client):
    with (
        patch(
            "fantasy_baseball.web.season_routes.read_cache_dict",
            return_value=_trajectory_payload(),
        ),
        patch(
            "fantasy_baseball.data.rosters.live_rosters",
            return_value=_trajectory_spots(),
        ),
    ):
        plain = client.get("/trajectory")
        explicit = client.get("/trajectory?view=board")
        junk = client.get("/trajectory?view=nonsense")
    for resp in (plain, explicit, junk):
        assert resp.status_code == 200
        assert "All teams" in resp.data.decode(), "the league board's team dropdown"


def test_trajectory_teams_view_falls_back_when_no_rosters_arrived(client):
    """A stale bookmark must degrade to the league board, not render an empty page."""
    with (
        patch(
            "fantasy_baseball.web.season_routes.read_cache_dict",
            return_value=_trajectory_payload(),
        ),
        patch("fantasy_baseball.data.rosters.live_rosters", return_value=[]),
    ):
        resp = client.get("/trajectory?view=teams")
    assert resp.status_code == 200
    assert "Testy McTestface" in resp.data.decode(), "the league board rendered instead"
```

- [ ] **Step 2: Run them to verify they fail**

Run: `pytest tests/test_web/test_season_routes.py -q -k "teams_view or defaults_to_the_league"`
Expected: `test_trajectory_teams_view_renders_one_block_per_team` FAILS (no blocks render — `?view=teams` is ignored). The other two pass already and must keep passing.

- [ ] **Step 3: Branch the route**

In `season_routes.py`, replace the single `build_board(...)` call with a branch on `view`. The `view` value is clamped against the teams actually available, so no-roster degrades:

```python
            view = select_view(request.args.get("view"))
            try:
                if view == "teams":
                    teams_board = build_teams_board(
                        payload,
                        end=request.args.get("end"),
                        pool=request.args.get("pool", "both"),
                        scale=request.args.get("scale", "var"),
                        spots=spots,
                        my_team=my_team,
                        per_team=request.args.get("per"),
                    )
                    # No roster data means no blocks to show. Fall back rather than
                    # render an empty page -- a bookmark outlives an Upstash outage.
                    # Reset to "board" rather than inventing a third view value: the
                    # only values that exist anywhere are the two `select_view` allows.
                    if not teams_board.blocks:
                        teams_board, view = None, "board"
                if view == "board":
                    board = build_board(
                        payload,
                        end=request.args.get("end"),
                        pool=request.args.get("pool", "both"),
                        top=request.args.get("top"),
                        scale=request.args.get("scale", "var"),
                        spots=spots,
                        my_team=my_team,
                        team=request.args.get("team", "all"),
                    )
            except (ValueError, KeyError) as exc:
                error = str(exc)
```

Then select the template and build `cur` from whichever object exists:

```python
        if view == "teams" and teams_board is not None:
            cur = {
                "end_year": teams_board.end_year,
                "pool": teams_board.pool,
                "top": request.args.get("top", DEFAULT_TOP),
                "scale": teams_board.scale,
                "team": "all",
                "view": "teams",
                "per": teams_board.per_team,
            }
            return render_template(
                "season/trajectory_teams.html",
                meta=read_meta(),
                active_page="trajectory",
                board=teams_board,
                error=error,
                cur=cur,
            )
```

Initialise `teams_board = None` beside `board, error = None, None`, and add
`build_teams_board` and `select_view` to the route's import block.

**`select_view` is a new PUBLIC helper in `trajectory_view.py`**, added in this step:

```python
#: The two views `/trajectory` renders. A junk or absent value is the league board.
VIEWS = ("board", "teams")


def select_view(value: Any) -> str:
    """Which view a query string is asking for, clamped to one that exists.

    Public because the ROUTE has to branch before it can build a view model, so it
    cannot learn this from the returned object the way it learns `pool` or `scale`.
    Exported deliberately rather than having the route import `_clamp_choice`:
    reaching across a module boundary for a private helper is how that helper stops
    being free to change.
    """
    return _clamp_choice(value, VIEWS, "board")
```

- [ ] **Step 4: Add the view pills to the shared partial**

In `_trajectory_controls.html`, inside `controls(...)`, immediately after the opening `<div class="trajectory-controls">`, add:

```jinja
  {% if teams %}
  <span class="pill-group">
    <a class="pill {% if cur.view == 'board' %}active{% endif %}" href="{{ board_url(cur, view='board') }}">League</a>
    <a class="pill {% if cur.view == 'teams' %}active{% endif %}" href="{{ board_url(cur, view='teams') }}">By team</a>
  </span>
  {% endif %}
```

Gated on `teams` for the same reason the team dropdown is: with no roster data there are no blocks, so offering the view would hand the reader a control that clamps straight back.

- [ ] **Step 5: Create the teams template**

Create `src/fantasy_baseball/web/templates/season/trajectory_teams.html`:

```jinja
{% extends "season/base.html" %}
{% block title %}Trajectory by Team -- Season Dashboard{% endblock %}
{% block content %}
{% import "season/_trajectory_controls.html" as ctl %}

<div class="page-trajectory">
<div class="page-header">
    <h2>Trajectory Board -- by team</h2>
</div>

{% if error %}
  <p class="warning">{{ error }}</p>
{% else %}

<p class="muted">
  The best {{ board.per_team }} on every roster over {{ board.span }}, strongest team first.
  Ranks are LEAGUE ranks -- a block's top row reads its position among all
  {{ board.ranked }} scored players, not its position on that roster.
</p>

{{ ctl.controls(cur, board.end_years, board.blocks | map(attribute='team') | list) }}

{% if board.mine_missing %}
<p class="muted"><strong>None of these blocks is marked as yours</strong> -- the page could
not match your team name to any roster it read, so no block is highlighted. That is a
statement about the lookup, not about what you own.</p>
{% endif %}

{% for block in board.blocks %}
<div class="team-block{% if block.is_mine %} user-team{% endif %}">
  <h3>{{ block.team }}
    <span class="muted">({{ block.rows | length }} of {{ block.scored }} scored,
    {{ '%.1f'|format(block.total) }} {{ board.span }} {{ board.scale|upper }} from the best
    {{ board.per_team }})</span>
  </h3>
  {% if block.rows %}
  <table class="data-table trajectory-table">
    <thead>
      <tr>
        <th title="League-wide rank over the whole scored pool">#</th>
        <th>Player</th>
        <th>Age</th>
        <th>Slot</th>
        <th class="sorted">{{ board.span }} {{ board.scale|upper }}</th>
        {% for y in board.year_columns %}
        <th title="{{ board.scale|upper }} in {{ y }} alone">'{{ (y|string)[2:] }}</th>
        {% endfor %}
      </tr>
    </thead>
    <tbody>
      {% for r in block.rows %}
      <tr>
        <td>{{ r.rank_total }}</td>
        <td>{{ r.name }}
          {%- if r.owner_ambiguous %} <span class="flag flag-extrap" title="This name is not unique -- either the board carries more than one player under it, or more than one team rosters one. Roster blobs have no MLBAM id (#284), so the name is the only join available and it cannot tell them apart. Who owns this row is a guess; check before acting on it.">(?)</span>
          {%- endif %}
        </td>
        <td>{{ r.age }}</td>
        <td>{{ r.slot }}</td>
        <td>{{ '%.1f'|format(r.total) }}</td>
        {% for cell in r.year_cells %}
        <td>{% if cell is not none %}{{ '%.1f'|format(cell) }}{% else %}--{% endif %}</td>
        {% endfor %}
      </tr>
      {% endfor %}
    </tbody>
  </table>
  {% else %}
  <p class="muted">Nothing on this roster was scored at this timeframe.</p>
  {% endif %}
  {% if block.unscored %}
  <p class="muted">
    not scored: {{ block.unscored|join(', ') }} --
    no {{ board.base_season }} line (injured, released, or not yet up) or pacing under
    {{ '%.1f'|format(board.meta.min_sgp) }} SGP, so the model could not price them.
  </p>
  {% endif %}
</div>
{% endfor %}

{% endif %}
</div>
{% endblock %}
```

- [ ] **Step 6: Run the tests**

Run: `pytest tests/test_web/ -q`
Expected: PASS.

- [ ] **Step 7: Mutation-check the fallback**

Change `if not teams_board.blocks:` to `if False:` and re-run
`test_trajectory_teams_view_falls_back_when_no_rosters_arrived`. It must FAIL. Revert.

- [ ] **Step 8: Run the full checks and commit**

Run: `pytest -n auto -q && ruff check . && ruff format --check . && mypy && vulture`
Expected: the only failures are the pre-existing `resend` ones (1 failed, 5 errors) — confirm they reproduce on `main` before accepting them.

```bash
git add src/fantasy_baseball/web/templates/season/trajectory_teams.html \
        src/fantasy_baseball/web/templates/season/_trajectory_controls.html \
        src/fantasy_baseball/web/season_routes.py \
        tests/test_web/test_season_routes.py
git commit -m "feat(trajectory): all-teams view, one block per roster"
```

---

## Self-Review

**Spec coverage.** Requirements 1-10: R1 Task 3 Steps 3/5; R2 Task 1 Step 4 (`blocks.sort`) + Step 1 test; R3 Task 1 Step 4 (`rows.sort` on `rank_total`) + `test_a_blocks_rows_carry_league_ranks`; R4 `test_per_team_slices_without_re_ranking`; R5 `grouped` seeded from `index.teams` + `test_a_team_with_nothing_scored_still_renders...`; R6 `unscored_for(team, pool)` + `test_scored_follows_the_pool_filter...`; R7 membership loop + `test_a_name_two_teams_roster_appears_in_both_blocks_flagged`; R8 Task 2's every-param-on-every-link test; R9 `mine_missing` + its test + Task 3 Step 5 banner; R10 `year_columns` shared and rendered in Task 3 Step 5.

Every edge-case row has a home: no roster data (Task 3 Step 3 fallback + its test), junk `view` (`_clamp_choice`), junk `per` (`_clamp`, tested), zero-scored team (R5), all teams empty (the fallback catches it — `blocks` is non-empty but every block is, and the page renders honestly), colliding name (R7), `my_team` absent (`mine_missing`), pool excluding a team's rows (same path as zero-scored).

**Placeholder scan.** No TBD/TODO; every code step carries literal code; no "similar to Task N".

**Type consistency.** `build_teams_board`, `TeamsBoard`, `TeamBlock`, `DEFAULT_PER_TEAM`, `ctl.controls`, `ctl.board_url` and the `cur` dict's seven keys are spelled identically in every task that touches them. `cur` is a plain dict rather than a dataclass because Jinja reads both the same way and the route builds it from two different view models.

**One gap found while writing:** the spec said "both extend the shared control bar" without saying what makes it shareable. `TeamsBoard` deliberately lacks `top`/`team`/`teams`, so a partial reading `board.*` breaks on the second caller. Task 2 therefore passes a `cur` state dict explicitly, and that is why the extraction is its own commit rather than folded into Task 3.
