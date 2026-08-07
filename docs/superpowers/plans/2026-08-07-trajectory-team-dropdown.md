# Trajectory Board Team Dropdown Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Source spec:** `docs/superpowers/specs/2026-08-07-trajectory-team-dropdown-design.md`

**Goal:** Add a team dropdown to the trajectory board so it can be narrowed to one league roster, defaulting to today's all-teams view.

**Architecture:** The `(normalized_name, player_type)` roster join moves out of `scripts/trajectory_board.py` into a new shared `src/fantasy_baseball/trajectory/roster_join.py` that returns a lookup and mutates nothing. `build_board` consumes raw `RosterSpot`s instead of a pre-computed ownership set and gains a `team` filter applied beside the existing `pool` filter — after the rank cache, never in its key. The CLI and the web then share one implementation of a join that is known to be non-unique (#284).

**Tech Stack:** Python 3.11, Flask + Jinja2, pytest. No new dependencies.

## Global Constraints

- **ASCII only** in source, log messages, format strings and anything reaching `print()`. This dev box is Windows and stdout defaults to cp1252.
- **Player identity is `(normalized_name, player_type)`** for roster joins — roster blobs carry no `mlbam_id` (#284). Never key on a bare name.
- **Never use `x or default` for numeric defaults.** Use `d["k"] if d.get("k") is not None else default`.
- **Do not mutate rows returned by `_ranked_rows`.** They are shared across requests; `build_board` copies each into a new dict.
- **Ranks are league-wide** and must not be recomputed inside any filter.
- Verification for every task: `pytest -v` (relevant subset acceptable, state which), `ruff check .`, `ruff format --check .`, `vulture`, and `mypy` (both `src/fantasy_baseball/trajectory/` and `src/fantasy_baseball/web/trajectory_view.py` are under `[tool.mypy].files`).
- Tests are the guardrail: do not loosen an assertion to make it pass.

**Task-to-commit mapping:** the spec's Phasing calls for three commits. Tasks 1 and 2 together are its commit 1 (the module lands green before anything is rewired onto it, so a reviewer can accept the module and reject the rewiring). Task 3 is commit 2, Task 4 is commit 3.

---

### Task 1: The shared roster join

**Files:**
- Create: `src/fantasy_baseball/trajectory/roster_join.py`
- Test: `tests/test_trajectory/test_roster_join.py`

**Interfaces:**
- Consumes: `fantasy_baseball.data.rosters.RosterSpot` (fields: `name`, `normalized`, `player_type`, `team`, `yahoo_id`, `status`); `fantasy_baseball.utils.name_utils.normalize_name`.
- Produces: `RosterIndex` (frozen dataclass with `team_of`, `status_of`, `ambiguous`, `unscored`, `teams`, and methods `team_for(name, pool)`, `status_for(name, pool)`, `is_ambiguous(name, pool)`) and `index_rosters(rows, spots, my_team) -> RosterIndex`. Tasks 2, 3 and 4 all depend on these exact names.

- [ ] **Step 1: Write the failing tests**

Create `tests/test_trajectory/test_roster_join.py`:

```python
from __future__ import annotations

from fantasy_baseball.data.rosters import RosterSpot
from fantasy_baseball.trajectory.roster_join import index_rosters


def _spot(name: str, team: str, pool: str = "hitter", status: str = "") -> RosterSpot:
    return RosterSpot(
        name=name,
        normalized=name.lower(),
        player_type=pool,
        team=team,
        yahoo_id="0",
        status=status,
    )


def _row(name: str, pool: str = "hitter") -> dict:
    return {"name": name, "pool": pool}


def test_my_team_sorts_first_and_the_rest_alphabetically() -> None:
    """The dropdown order. Mine first because it is the roster I act on."""
    spots = [_spot("A", "Zebras"), _spot("B", "Mine"), _spot("C", "Aardvarks")]
    index = index_rosters([], spots, "Mine")
    assert index.teams == ("Mine", "Aardvarks", "Zebras")


def test_teams_are_plain_alphabetical_when_my_team_is_unknown() -> None:
    """`my_team` is None when the config read fails, and can name a team that is
    not in the blob after a rename. Neither may promote a phantom entry."""
    spots = [_spot("A", "Zebras"), _spot("C", "Aardvarks")]
    assert index_rosters([], spots, None).teams == ("Aardvarks", "Zebras")
    assert index_rosters([], spots, "Nobody").teams == ("Aardvarks", "Zebras")


def test_a_name_matching_two_board_rows_is_ambiguous() -> None:
    """The live board carries two hitters called Max Muncy. Roster blobs have no
    mlbam_id (#284), so the join cannot tell them apart and must say so."""
    rows = [_row("Max Muncy"), _row("Max Muncy"), _row("Elly De La Cruz")]
    index = index_rosters(rows, [_spot("Max Muncy", "Mine")], "Mine")
    assert index.is_ambiguous("Max Muncy", "hitter")
    assert not index.is_ambiguous("Elly De La Cruz", "hitter")


def test_the_same_name_in_two_pools_is_not_ambiguous() -> None:
    """A two-way player is two assets in this league, so (name, pool) is the key
    and a hitter does not collide with a pitcher."""
    rows = [_row("Shohei Ohtani", "hitter"), _row("Shohei Ohtani", "pitcher")]
    index = index_rosters(rows, [_spot("Shohei Ohtani", "Mine")], "Mine")
    assert not index.is_ambiguous("Shohei Ohtani", "hitter")


def test_roster_players_with_no_scored_row_are_grouped_by_team() -> None:
    """The list the UI must render -- "the model could not price these", not
    "these are not worth listing"."""
    rows = [_row("Scored Guy")]
    spots = [
        _spot("Scored Guy", "Mine"),
        _spot("Rookie", "Mine"),
        _spot("Their Rookie", "Theirs"),
    ]
    index = index_rosters(rows, spots, "Mine")
    assert index.unscored == {"Mine": ["Rookie"], "Theirs": ["Their Rookie"]}


def test_team_and_status_are_looked_up_by_normalized_name_and_pool() -> None:
    spots = [_spot("Yoan Moncada", "Mine", status="IL10")]
    index = index_rosters([_row("Yoan Moncada")], spots, "Mine")
    assert index.team_for("Yoan Moncada", "hitter") == "Mine"
    assert index.status_for("Yoan Moncada", "hitter") == "IL10"
    assert index.team_for("Yoan Moncada", "pitcher") is None
    assert index.status_for("Nobody", "hitter") == ""


def test_the_input_rows_are_not_mutated() -> None:
    """The entire reason this returns a lookup instead of stamping rows.

    The web's rows come from `_ranked_rows`, a cross-request cache whose comment
    states they are never mutated after `add_ranks`. A helper that stamped them
    would be correct in the CLI and a data race on the web.
    """
    rows = [_row("Scored Guy")]
    before = [dict(r) for r in rows]
    index_rosters(rows, [_spot("Scored Guy", "Mine")], "Mine")
    assert rows == before
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `pytest tests/test_trajectory/test_roster_join.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'fantasy_baseball.trajectory.roster_join'`

- [ ] **Step 3: Write the implementation**

Create `src/fantasy_baseball/trajectory/roster_join.py`:

```python
"""The roster-to-board join, shared by the CLI and the web board.

`(normalized_name, player_type)` is the only key available: roster blobs carry
no `mlbam_id` (#284), so this join is NOT unique -- the live board has two
hitters called Max Muncy. That is surfaced through `ambiguous` rather than
resolved here.

WHY THIS LIVES UNDER `src/`. Nothing under `src/` can import from `scripts/`,
and `scripts/trajectory_board.py` owned the only implementation. A web team
filter would have needed its own copy -- two spellings of a fragile join, free
to drift on exactly the ambiguity handling that matters. Same reason
`trajectory/sweep.py` was extracted, recorded in its module docstring.

WHY IT RETURNS A LOOKUP AND MUTATES NOTHING. The predecessor stamped
`row["team"]` in place, which is safe for rows the CLI just built. The web's
rows come from `trajectory_view._ranked_rows`, a cache shared across requests
whose comment states rows are "never mutated after `add_ranks`". One helper
that mutated would be correct in one caller and a cross-request data race in
the other, so the hazard is made unreachable instead of merely avoided.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field

from fantasy_baseball.data.rosters import RosterSpot
from fantasy_baseball.utils.name_utils import normalize_name


@dataclass(frozen=True)
class RosterIndex:
    """Everything a board needs to know about who owns which row."""

    #: (normalized name, pool) -> owning team.
    team_of: dict[tuple[str, str], str]
    #: Same key -> "" / "IL10" / "DTD". The CLI renders this; the web does not.
    status_of: dict[tuple[str, str], str]
    #: Keys matching MORE THAN ONE board row. A consumer attributing such a row
    #: to a team is guessing, and must say so on screen.
    ambiguous: set[tuple[str, str]]
    #: Roster players with no scored row at all, per team. Rendering this is what
    #: keeps "the model could not price him" from reading as "he ranked last".
    unscored: dict[str, list[str]]
    #: Dropdown order: my team first, then the rest alphabetically. Plain
    #: alphabetical when `my_team` is None or names no team on any roster.
    teams: tuple[str, ...] = field(default=())

    def team_for(self, name: str, pool: str) -> str | None:
        return self.team_of.get((normalize_name(name), pool))

    def status_for(self, name: str, pool: str) -> str:
        return self.status_of.get((normalize_name(name), pool), "")

    def is_ambiguous(self, name: str, pool: str) -> bool:
        return (normalize_name(name), pool) in self.ambiguous


def index_rosters(
    rows: Sequence[dict],
    spots: Sequence[RosterSpot],
    my_team: str | None,
) -> RosterIndex:
    """Join `spots` against `rows` without touching either.

    `rows` need only carry "name" and "pool".
    """
    by_key: dict[tuple[str, str], RosterSpot] = {
        (s.normalized, s.player_type): s for s in spots
    }

    counts: dict[tuple[str, str], int] = {}
    for row in rows:
        key = (normalize_name(row["name"]), row["pool"])
        counts[key] = counts.get(key, 0) + 1

    unscored: dict[str, list[str]] = {}
    for spot in spots:
        if (spot.normalized, spot.player_type) not in counts:
            unscored.setdefault(spot.team, []).append(spot.name)

    rostered = {s.team for s in spots}
    # `my_team in rostered` rather than `is not None`: a renamed or mistyped team
    # must not be promoted into a dropdown it cannot filter to.
    promoted = (my_team,) if my_team in rostered else ()
    teams = promoted + tuple(sorted(rostered - set(promoted)))

    return RosterIndex(
        team_of={k: s.team for k, s in by_key.items()},
        status_of={k: s.status for k, s in by_key.items()},
        ambiguous={k for k, c in counts.items() if c > 1},
        unscored=unscored,
        teams=teams,
    )
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `pytest tests/test_trajectory/test_roster_join.py -v`
Expected: PASS (7 tests)

- [ ] **Step 5: Verify the no-mutation test actually bites**

Temporarily add `row["team"] = "X"` inside the `for row in rows` loop in `index_rosters`, re-run `pytest tests/test_trajectory/test_roster_join.py::test_the_input_rows_are_not_mutated -v`, confirm it FAILS, then revert. A guard that cannot fail is not a guard — two vacuous tests were already caught this way in this codebase.

- [ ] **Step 6: Run the checks**

Run: `ruff check . && ruff format --check . && mypy && vulture`
Expected: all clean. `vulture` may report `RosterIndex.status_for` / `is_ambiguous` as unused until Task 2 wires them; that is expected and resolves in the next task.

- [ ] **Step 7: Commit**

```bash
git add src/fantasy_baseball/trajectory/roster_join.py tests/test_trajectory/test_roster_join.py
git commit -m "feat(trajectory): shared roster join that returns a lookup, not a mutation"
```

---

### Task 2: Move the CLI onto the shared join

**Files:**
- Modify: `scripts/trajectory_board.py` — delete `assign_teams` (lines 67-88), rewire `main`'s roster block and `by_team`'s parameter
- Test: `tests/test_scripts/test_trajectory_board_cli.py` (add one test)

**Interfaces:**
- Consumes: `index_rosters`, `RosterIndex` from Task 1.
- Produces: `by_team(scored, spots, unscored, my_team, per_team, base, horizons, only)` — the third parameter is renamed from `missing` to `unscored` and is still a `dict[str, list[str]]`, so the three existing tests keep passing `{}` positionally.

- [ ] **Step 1: Write the failing test**

Append to `tests/test_scripts/test_trajectory_board_cli.py`:

```python
def test_a_rostered_player_the_model_cannot_price_is_named(capsys) -> None:
    """The output `assign_teams` used to produce, driven end to end.

    All three tests above hand `by_team` an empty `missing` dict, so the map
    `assign_teams` returned -- the one thing being deleted and re-homed -- had
    ZERO coverage. Without this test the CLI could stop reporting unscored
    players entirely and the suite would stay green.
    """
    from fantasy_baseball.trajectory.roster_join import index_rosters

    module = _script()
    scored = [_row("Scored Guy", 12.0, "T")]
    spots = [_spot("Scored Guy", "T"), _spot("Bench Rookie", "T")]
    index = index_rosters(scored, spots, "T")

    module.by_team(scored, spots, index.unscored, "OTHER", 5, 2026, (1, 2, 3), None)
    out = capsys.readouterr().out

    assert "not scored: Bench Rookie" in out
    assert "Scored Guy" in out, "the scored player still renders"
```

- [ ] **Step 2: Run it to verify it fails**

Run: `pytest tests/test_scripts/test_trajectory_board_cli.py::test_a_rostered_player_the_model_cannot_price_is_named -v`
Expected: FAIL — `ModuleNotFoundError` on `roster_join` is already resolved by Task 1, so this fails on the assertion or on `by_team`'s signature, depending on ordering. Either failure is the expected red.

- [ ] **Step 3: Delete `assign_teams` and rewire `main`**

In `scripts/trajectory_board.py`, delete the whole `assign_teams` function (lines 67-88) and its now-unused `normalize_name` import if nothing else in the file uses it (check with `grep -n "normalize_name" scripts/trajectory_board.py` first — leave the import if there are other callers).

Add to the imports:

```python
from fantasy_baseball.trajectory.roster_join import index_rosters
```

Replace the roster block in `main`:

```python
        try:
            spots = live_rosters(config.team_name)
            print(f"\n  {len(spots)} roster spots read from Upstash")
            missing = assign_teams(scored, spots)
        except Exception as exc:
            if show_teams:
                raise
            print(f"\n  NOTE: rosters unavailable ({type(exc).__name__}); CSV has no team column.")
            spots, missing = [], {}
        if show_teams:
            by_team(
                scored, spots, missing, config.team_name, args.per_team, season, horizons, args.team
            )
```

with:

```python
        try:
            spots = live_rosters(config.team_name)
            print(f"\n  {len(spots)} roster spots read from Upstash")
            # The index is READ-ONLY, so the stamping the CLI needs happens here
            # rather than inside it -- these rows are the CLI's own and mutating
            # them is safe, which is not true of the web's cached rows.
            index = index_rosters(scored, spots, config.team_name)
            for row in scored:
                row["team"] = index.team_for(row["name"], row["pool"])
                row["status"] = index.status_for(row["name"], row["pool"])
            unscored = index.unscored
        except Exception as exc:
            if show_teams:
                raise
            print(f"\n  NOTE: rosters unavailable ({type(exc).__name__}); CSV has no team column.")
            spots, unscored = [], {}
        if show_teams:
            by_team(
                scored, spots, unscored, config.team_name, args.per_team, season, horizons, args.team
            )
```

- [ ] **Step 4: Rename `by_team`'s parameter**

In `by_team`'s signature change `missing: dict[str, list[str]]` to `unscored: dict[str, list[str]]`, and in `block()` change:

```python
        if team in missing:
            print(f"  not scored: {', '.join(sorted(missing[team]))}")
```

to:

```python
        if team in unscored:
            print(f"  not scored: {', '.join(sorted(unscored[team]))}")
```

Update the docstring reference in `block()` that says `missing` is keyed on the raw name to say `unscored`.

- [ ] **Step 5: Run the CLI tests**

Run: `pytest tests/test_scripts/test_trajectory_board_cli.py -v`
Expected: PASS (4 tests — the three from `7e74b7b1` plus the new one). The three existing ones passing unchanged IS the parity check that the extraction did not alter CLI behaviour.

- [ ] **Step 6: Confirm the CLI still runs end to end**

Run: `RENDER=true python scripts/trajectory_board.py --by-team --top 3 2>&1 | head -20`
Expected: per-team blocks render with `total ... VAR from the best 5` headers, `[IL10]`-style suffixes still present on injured players, and `not scored:` lines where a roster carries unpriced players. This is the check that `status_of` did its job — deleting `assign_teams` without it would silently drop the suffix.

- [ ] **Step 7: Run the checks and commit**

Run: `pytest tests/test_scripts/ tests/test_trajectory/ -q && ruff check . && ruff format --check . && mypy && vulture`

```bash
git add scripts/trajectory_board.py tests/test_scripts/test_trajectory_board_cli.py
git commit -m "refactor(trajectory): CLI reads the shared roster join; assign_teams deleted"
```

---

### Task 3: `build_board` takes roster spots and a team filter

**Files:**
- Modify: `src/fantasy_baseball/web/trajectory_view.py` — `Board` dataclass, `build_board` signature and body
- Test: `tests/test_web/test_trajectory_view.py`

**Interfaces:**
- Consumes: `index_rosters`, `RosterIndex` from Task 1.
- Produces: `build_board(payload, *, end=None, pool="both", top=None, scale="var", spots=None, my_team=None, team="all") -> Board`, where `Board` additionally carries `team: str`, `teams: tuple[str, ...]`, `unscored: list[str]`. The `mine` parameter is REMOVED. Row dicts carry `owner_ambiguous` in place of `mine_ambiguous`. Task 4 depends on all of these names.

- [ ] **Step 1: Write the failing tests**

Add to `tests/test_web/test_trajectory_view.py`. Note the fixture requirements from the spec: two teams with different rosters, an unscored player, and the ambiguous name **on an opponent**, not on my team.

```python
def _spots_fixture():
    """Rosters for the `payload` fixture's six players.

    "Big Bat" is deliberately duplicated onto an OPPONENT: the ambiguity flag
    used to render only for my own rows, and a fixture that put the collision on
    my roster would exercise the old path and leave the new one untested.
    """
    from fantasy_baseball.data.rosters import RosterSpot

    def spot(name, team, pool="hitter", status=""):
        return RosterSpot(
            name=name,
            normalized=name.lower(),
            player_type=pool,
            team=team,
            yahoo_id="0",
            status=status,
        )

    return [
        spot("Big Bat", "Theirs"),
        spot("Small Bat", "Mine"),
        spot("Under Water", "Mine"),
        spot("Big Arm", "Mine", pool="pitcher"),
        spot("Never Scored", "Mine"),
    ]


def test_selecting_a_team_narrows_the_board_to_that_roster(payload: dict) -> None:
    board = build_board(payload, spots=_spots_fixture(), my_team="Mine", team="Mine")
    assert {r["name"] for r in board.rows} == {"Small Bat", "Under Water", "Big Arm"}
    assert board.team == "Mine"


def test_a_teams_best_player_keeps_his_league_rank(payload: dict) -> None:
    """Ranking within the subset would make every team's best player a #1 and
    destroy the only comparison the board exists for."""
    everyone = build_board(payload, spots=_spots_fixture(), my_team="Mine")
    league = {r["name"]: r["rank_total"] for r in everyone.rows}
    mine = build_board(payload, spots=_spots_fixture(), my_team="Mine", team="Mine")

    assert [r["rank_total"] for r in mine.rows] == [league[r["name"]] for r in mine.rows]
    assert mine.rows[0]["rank_total"] != 1, "expected a league rank, not a within-team one"


def test_an_unknown_team_falls_back_to_the_whole_board(payload: dict) -> None:
    """The query string is user-editable and survives a team rename."""
    everyone = build_board(payload, spots=_spots_fixture(), my_team="Mine")
    junk = build_board(payload, spots=_spots_fixture(), my_team="Mine", team="Nobody FC")
    assert junk.team == "all"
    assert len(junk.rows) == len(everyone.rows)


def test_the_team_filter_composes_with_the_pool_filter(payload: dict) -> None:
    board = build_board(
        payload, spots=_spots_fixture(), my_team="Mine", team="Mine", pool="pitcher"
    )
    assert {r["name"] for r in board.rows} == {"Big Arm"}


def test_unscored_is_populated_only_for_a_selected_team(payload: dict) -> None:
    assert build_board(payload, spots=_spots_fixture(), my_team="Mine").unscored == []
    mine = build_board(payload, spots=_spots_fixture(), my_team="Mine", team="Mine")
    assert mine.unscored == ["Never Scored"]


def test_my_team_leads_the_dropdown(payload: dict) -> None:
    board = build_board(payload, spots=_spots_fixture(), my_team="Mine")
    assert board.teams == ("Mine", "Theirs")


def test_an_opponents_ambiguous_row_is_flagged_too(payload: dict) -> None:
    """`mine_ambiguous` only ever fired for my rows. Attributing an opponent's
    player on a guess is exactly as wrong."""
    spots = _spots_fixture()
    board = build_board(payload, spots=spots, my_team="Mine", team="Theirs")
    big_bats = [r for r in board.rows if r["name"] == "Big Bat"]
    assert big_bats, "fixture must place an ambiguous name on the opponent"
    assert all(r["owner_ambiguous"] for r in big_bats)


def test_has_rosters_still_tracks_my_own_roster_not_the_read(payload: dict) -> None:
    """Two different facts. `has_rosters` gates the not-highlighted banner and
    must stay false when MY roster joined nothing, even though the read
    succeeded and the dropdown is perfectly usable."""
    from fantasy_baseball.data.rosters import RosterSpot

    others_only = [
        RosterSpot(
            name="Big Bat",
            normalized="big bat",
            player_type="hitter",
            team="Theirs",
            yahoo_id="0",
            status="",
        )
    ]
    board = build_board(payload, spots=others_only, my_team="Mine")
    assert board.has_rosters is False
    assert board.teams == ("Theirs",), "the dropdown still works for other teams"
```

Note: `payload`'s fixture has two "Big Bat"-named rows only if the ambiguity is real. The fixture has ONE "Big Bat". Add a second board row with the same name in the module-level `payload` fixture so the collision exists:

In the `payload` fixture's `hitters` list, add:

```python
        # A SECOND player sharing a normalized name with "Big Bat". The live
        # board carries two Max Muncys; without a collision here the ambiguity
        # flag is unreachable and re-breaking it would pass unnoticed.
        BoardRow(11, "Big Bat", "hitter", 27, 19.0, 18.0, "OF", 4.0),
```

- [ ] **Step 2: Run them to verify they fail**

Run: `pytest tests/test_web/test_trajectory_view.py -v -k "team or ambiguous or has_rosters"`
Expected: FAIL — `build_board() got an unexpected keyword argument 'spots'`

- [ ] **Step 3: Add the `Board` fields**

In `trajectory_view.py`, add to the `Board` dataclass after `scale`:

```python
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
```

- [ ] **Step 4: Rewrite `build_board`'s signature and roster handling**

Replace the `mine` parameter:

```python
    mine: set[tuple[str, str]] | None = None,
```

with:

```python
    spots: Sequence[RosterSpot] | None = None,
    my_team: str | None = None,
    team: str = "all",
```

Add the imports at the top of the file:

```python
from collections.abc import Sequence

from fantasy_baseball.data.rosters import RosterSpot
from fantasy_baseball.trajectory.roster_join import index_rosters
```

Replace the ownership block — this:

```python
    owned = mine or set()
    key_counts: dict[tuple[str, str], int] = {}
    for row in ranked_rows:
        k = (normalize_name(row["name"]), row["pool"])
        key_counts[k] = key_counts.get(k, 0) + 1
```

with:

```python
    # ONE index serves both the highlight and the filter. The route used to build
    # the ownership set itself, which meant two places decided what a roster spot
    # meant to a board row.
    index = index_rosters(ranked_rows, spots or [], my_team)
    if team != "all" and team not in index.teams:
        # Same clamp `pool` and `scale` get: this arrives from a query string a
        # reader can edit and a bookmark outlives a team rename.
        team = "all"
```

Replace the row loop's per-row ownership lines — this:

```python
        move = rank_move(row)
        key = (normalize_name(row["name"]), row["pool"])
        is_mine = key in owned
        rows.append(
            {
                **row,
                "mine": is_mine,
                "mine_ambiguous": is_mine and key_counts[key] > 1,
```

with:

```python
        owner = index.team_for(row["name"], row["pool"])
        if team != "all" and owner != team:
            continue
        move = rank_move(row)
        is_mine = my_team is not None and owner == my_team
        rows.append(
            {
                **row,
                "mine": is_mine,
                # Flagged whenever the row is being ATTRIBUTED to a team on
                # screen and the join cannot tell which player it is. In the
                # all-teams view that is `mine` alone, exactly as before; under a
                # team filter it is any row shown, because putting an opponent's
                # player on a guess is as wrong as putting mine.
                "owner_ambiguous": (is_mine or team != "all")
                and index.is_ambiguous(row["name"], row["pool"]),
```

- [ ] **Step 5: Populate the new `Board` fields**

In the `return Board(...)` call, change `has_rosters=bool(mine)` to:

```python
        # MY roster joined something -- deliberately NOT "the read returned
        # data". This gates the not-highlighted banner, and a successful read
        # where my own roster joined nothing must still show it.
        has_rosters=any(
            index.team_for(r["name"], r["pool"]) == my_team for r in ranked_rows
        )
        if my_team
        else False,
```

and add:

```python
        team=team,
        teams=index.teams,
        unscored=sorted(index.unscored.get(team, [])) if team != "all" else [],
```

- [ ] **Step 6: Run the tests**

Run: `pytest tests/test_web/test_trajectory_view.py -v`
Expected: PASS. Existing tests that passed `mine=` must be updated to pass `spots=`/`my_team=` — search with `grep -n "mine=" tests/test_web/test_trajectory_view.py` and convert each.

- [ ] **Step 7: Verify the ambiguity test bites**

Temporarily change `(is_mine or team != "all")` to `is_mine`, re-run
`pytest tests/test_web/test_trajectory_view.py::test_an_opponents_ambiguous_row_is_flagged_too -v`, confirm FAIL, revert.

- [ ] **Step 8: Run the checks and commit**

Run: `pytest tests/test_web/ tests/test_trajectory/ -q && ruff check . && ruff format --check . && mypy && vulture`

```bash
git add src/fantasy_baseball/web/trajectory_view.py tests/test_web/test_trajectory_view.py
git commit -m "feat(trajectory): build_board takes roster spots and a team filter"
```

---

### Task 4: The dropdown, the route, and the unscored line

**Files:**
- Modify: `src/fantasy_baseball/web/season_routes.py` (the `/trajectory` route)
- Modify: `src/fantasy_baseball/web/templates/season/trajectory.html`
- Test: `tests/test_web/test_season_routes.py`

**Interfaces:**
- Consumes: `build_board(..., spots=, my_team=, team=)` and `Board.team` / `Board.teams` / `Board.unscored` from Task 3; row field `owner_ambiguous`.
- Produces: no new Python interfaces; `?team=` becomes part of the board's URL state.

- [ ] **Step 1: Write the failing route tests**

Add to `tests/test_web/test_season_routes.py`:

```python
def test_trajectory_page_offers_a_team_dropdown(client):
    """Rendered from live rosters, with my own team promoted to the top."""
    from fantasy_baseball.data.rosters import RosterSpot

    spots = [
        RosterSpot("Testy McTestface", "testy mctestface", "hitter", "Zebras", "1", ""),
        RosterSpot("Someone Else", "someone else", "hitter", "Hart of the Order", "2", ""),
    ]
    with patch("fantasy_baseball.data.rosters.live_rosters", return_value=spots):
        resp = client.get("/trajectory")
    assert resp.status_code == 200
    assert b"All teams" in resp.data
    assert resp.data.index(b"Hart of the Order") < resp.data.index(b"Zebras")


def test_trajectory_page_hides_the_dropdown_when_no_rosters_arrived(client):
    """Empty `spots` -- an unreachable Upstash cannot be told from an empty
    league, so the control is not rendered at all rather than rendered empty."""
    with patch("fantasy_baseball.data.rosters.live_rosters", return_value=[]):
        resp = client.get("/trajectory")
    assert resp.status_code == 200
    assert b"All teams" not in resp.data


def test_trajectory_page_survives_a_team_param_with_no_rosters(client):
    """A stale bookmark must not 500."""
    with patch("fantasy_baseball.data.rosters.live_rosters", return_value=[]):
        resp = client.get("/trajectory?team=Nobody+FC")
    assert resp.status_code == 200
```

- [ ] **Step 2: Run them to verify they fail**

Run: `pytest tests/test_web/test_season_routes.py -v -k "dropdown or team_param"`
Expected: FAIL — "All teams" is not in the rendered page.

- [ ] **Step 3: Simplify the route**

In `season_routes.py`, replace the `mine` comprehension:

```python
            mine = None
            try:
                from fantasy_baseball.data.rosters import live_rosters

                my_team = _load_config().team_name
                mine = {
                    (spot.normalized, spot.player_type)
                    for spot in live_rosters(my_team)
                    if spot.team == my_team
                }
            except Exception:
                logger.warning("trajectory: live roster read failed; rendering unmarked")
```

with:

```python
            spots: list = []
            my_team = None
            try:
                from fantasy_baseball.data.rosters import live_rosters

                my_team = _load_config().team_name
                # Every team's spots, not just mine: the dropdown filters to any
                # roster, and `build_board` derives ownership from the same read.
                spots = list(live_rosters(my_team))
            except Exception:
                logger.warning("trajectory: live roster read failed; rendering unmarked")
```

and change the `build_board` call's `mine=mine,` to:

```python
                    spots=spots,
                    my_team=my_team,
                    team=request.args.get("team", "all"),
```

- [ ] **Step 4: Add `team` to `board_url` and render the control**

In `trajectory.html`, change the macro:

```jinja
{% macro board_url(end=None, pool=None, top=None, scale=None) -%}
{{ url_for('trajectory',
           end=end if end is not none else board.end_year,
           pool=pool if pool is not none else board.pool,
           top=top if top is not none else board.top,
           scale=scale if scale is not none else board.scale) }}
{%- endmacro %}
```

to:

```jinja
{% macro board_url(end=None, pool=None, top=None, scale=None, team=None) -%}
{{ url_for('trajectory',
           end=end if end is not none else board.end_year,
           pool=pool if pool is not none else board.pool,
           top=top if top is not none else board.top,
           scale=scale if scale is not none else board.scale,
           team=team if team is not none else board.team) }}
{%- endmacro %}
```

Add the control inside `<div class="trajectory-controls">`, after the Top select:

```jinja
  {% if board.teams %}
  <label>Team
    <select onchange="location = this.value">
      <option value="{{ board_url(team='all') }}" {% if board.team == 'all' %}selected{% endif %}>All teams</option>
      {% for t in board.teams %}
      <option value="{{ board_url(team=t) }}" {% if t == board.team %}selected{% endif %}>{{ t }}</option>
      {% endfor %}
    </select>
  </label>
  {% endif %}
```

And inside the existing `<noscript>` block, after the Top pill group:

```jinja
      {% if board.teams %}
      <span class="pill-group">
        <a class="pill {% if board.team == 'all' %}active{% endif %}" href="{{ board_url(team='all') }}">All teams</a>
        {% for t in board.teams %}
        <a class="pill {% if t == board.team %}active{% endif %}" href="{{ board_url(team=t) }}">{{ t }}</a>
        {% endfor %}
      </span>
      {% endif %}
```

- [ ] **Step 5: Rename the flag and fix its tooltip**

Find the `mine_ambiguous` span (around line 137) and replace it with:

```jinja
        {%- if r.owner_ambiguous %} <span class="flag flag-extrap" title="The board has more than one player with this name, and roster blobs carry no MLBAM id (#284), so the name is the only join available and it cannot tell them apart. This row is attributed on a guess -- check which player it is before acting on it.">(?)</span>
```

The old copy said "**You** roster a player with this name", which is wrong the moment the flag renders on an opponent's row.

- [ ] **Step 6: Render the unscored line OUTSIDE the empty-rows branch**

After the `{% endif %}` that closes the `{% if not board.rows %}` / `{% else %}` table pair, add:

```jinja
{% if board.team != 'all' and board.unscored %}
<p class="muted">
  not scored: {{ board.unscored|join(', ') }} --
  no {{ board.base_season }} line or pacing under
  {{ '%.1f'|format(board.meta.min_sgp) }} SGP, so the model could not price them.
</p>
{% endif %}
```

It must sit outside the branch: a team with no scored rows renders the bare "Nothing scored at this timeframe." message, and that is precisely the team whose unscored list explains the empty page.

- [ ] **Step 7: Run the tests**

Run: `pytest tests/test_web/ -v`
Expected: PASS.

- [ ] **Step 8: Look at the real page**

Run: `RENDER=true python scripts/run_season_dashboard.py --no-sync` and open `/trajectory`. Confirm: the Team control appears, "All teams" is selected by default and the board is unchanged from today; selecting your team narrows it to your roster with league ranks intact; the other filters survive the change; and an unscored line appears if your roster has unpriced players.

- [ ] **Step 9: Run the full checks and commit**

Run: `pytest -n auto -q && ruff check . && ruff format --check . && mypy && vulture`
Expected: the only failures are the pre-existing `resend` `ModuleNotFoundError` ones (1 failed, 5 errors) — confirm they reproduce on `main` before accepting them.

```bash
git add src/fantasy_baseball/web/season_routes.py \
        src/fantasy_baseball/web/templates/season/trajectory.html \
        tests/test_web/test_season_routes.py
git commit -m "feat(trajectory): team dropdown on the board, with the unscored roster line"
```

---

## Self-Review

**Spec coverage.** Requirements 1-8: R1 Task 3 Step 4 (`team="all"` default, no filtering branch taken) and Task 4 Step 3; R2 Task 3 Step 4; R3 Task 3 Step 1 (`test_a_teams_best_player_keeps_his_league_rank`); R4 Task 3 Step 5 + Task 4 Step 4; R5 Task 3 Step 1 (`..._composes_with_the_pool_filter`) + Task 4 Step 4 (`board_url` carries all five); R6 Task 3 Steps 4/7 + Task 4 Step 5; R7 Task 4 Step 6; R8 structural, satisfied by Task 2 deleting `assign_teams`. Every edge-case row has a home: empty spots (Task 4 Step 2), unknown team (Task 3 Step 4), zero scored rows (Task 4 Step 6 placement), ambiguity (Task 3 Step 4), join miss (`unscored` by construction, Task 1 Step 3), my-roster-joins-nothing (Task 3 Step 1), `my_team` None (Task 1 Step 1), punctuation in team names (`url_for` encodes; no normalization anywhere in Task 1's team handling).

**Placeholder scan.** No TBD/TODO; every code step carries the literal code; no "similar to Task N".

**Type consistency.** `index_rosters(rows, spots, my_team)` and the `team_for` / `status_for` / `is_ambiguous` method names are used identically in Tasks 2, 3 and 4. `unscored` is a `dict[str, list[str]]` on `RosterIndex` and a `list[str]` on `Board` — deliberate and different, flagged here because the shared name invites confusion.

**One gap found and fixed during writing:** deleting `assign_teams` would also have deleted `row["status"]`, which the CLI's `[IL10]` suffix depends on. `RosterIndex.status_of` was added to both the spec and Task 1, and Task 2 Step 6 checks the suffix on real output.
