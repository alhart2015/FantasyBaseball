# Trajectory Player Chart Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Source spec:** `docs/superpowers/specs/2026-08-07-trajectory-player-chart-design.md`

**Goal:** Search a player on `/trajectory` and see his career to date, his projected path with an uncertainty band, and his closest historical comps on one chart.

**Architecture:** A new `trajectory/comp_paths.py` finds the historical player-seasons whose *realized* five-year path best matches a shape-predicted one. `push_trajectory_board.py` runs it offline, where the panel lives, and bakes the result plus each player's career history into the same blob the board already ships. The dashboard stays a pure reader; the chart is Chart.js through the loader `season_trends.js` already owns.

**Tech Stack:** Python 3.11, NumPy, Flask + Jinja2, Chart.js 4.4.4 (CDN, see #342), pytest.

## Global Constraints

- **ASCII only** in source, templates, log messages, and anything reaching `print()`. Windows box, cp1252 stdout.
- **Never key on a bare name.** Player identity is `name::player_type`; roster joins use `(normalized_name, player_type)` because roster blobs carry no `mlbam_id` (#284).
- **Identifiers are looked up, never recalled.** A bare id literal in a query is a defect even when the output looks right. Resolve the name to the id in code.
- **Never `x or default` for numeric defaults** — `0`, `0.0`, `""` are falsy and this repo has been bitten in sort keys.
- **Rows from `_ranked_rows` are shared across requests and must not be mutated** — copy into a new dict.
- **New payload keys are read with `.get(..., [])`.** A blob pushed before this feature must render what it can, not 500. That is #332's outage applied deliberately.
- **The comps are selected ON THE OUTCOME.** They are not evidence for the projection. The p10-p90 band stays visually dominant, the block is labelled "closest realized paths", and each comp shows its RMSE.
- Tests are the guardrail: no assertion may be loosened, skipped or deleted to make something pass. **A test that cannot fail is a defect** — seven were caught in this codebase recently, three by mutation alone.
- Verification each task: `pytest tests/test_trajectory/ tests/test_web/ tests/test_scripts/ -q`, `ruff check .`, `ruff format --check .`, `mypy`, `vulture`. Pre-existing `resend` `ModuleNotFoundError`s in `test_send_daily_summary.py` / `test_summary` are unrelated — do not "fix" them.

---

### Task 1: `Prepared.mlbam_id` and the comp finder

**Files:**
- Modify: `src/fantasy_baseball/trajectory/shape.py` (the `Prepared` dataclass, and `prepare`'s return)
- Create: `src/fantasy_baseball/trajectory/comp_paths.py`
- Test: `tests/test_trajectory/test_comp_paths.py`

**Interfaces:**
- Consumes: `Prepared` (fields `kind`, `horizons`, `last`, `age`, `current`, `prior`, `season`, `forward`), from `fantasy_baseball.trajectory.shape`.
- Produces: `Prepared.mlbam_id: np.ndarray`; `CompPath` (frozen dataclass: `mlbam_id: int`, `season: int`, `rmse: float`, `path: tuple[float, ...]`); `closest_paths(prepared, predicted, age, n) -> list[CompPath]`. Tasks 2 and 3 depend on these names.

- [ ] **Step 1: Write the failing tests**

Create `tests/test_trajectory/test_comp_paths.py`:

```python
from __future__ import annotations

import numpy as np

from fantasy_baseball.trajectory.comp_paths import closest_paths
from fantasy_baseball.trajectory.shape import Prepared

HORIZONS = (1, 2, 3, 4, 5)


def _prepared(rows: list[tuple[int, int, int, list[float]]], last: int = 2020) -> Prepared:
    """Build a Prepared straight from (mlbam_id, season, age, forward path) rows.

    Hand-built rather than swept from a panel: `closest_paths` reads seven arrays and
    nothing else, so a fixture that goes through `prepare()` would test `prepare()`.
    """
    return Prepared(
        kind="hitter",
        horizons=HORIZONS,
        last=last,
        age=np.array([r[2] for r in rows], dtype=float),
        current=np.zeros(len(rows)),
        prior=np.zeros(len(rows)),
        season=np.array([r[1] for r in rows]),
        mlbam_id=np.array([r[0] for r in rows]),
        forward={h: np.array([r[3][h - 1] for r in rows], dtype=float) for h in HORIZONS},
    )


def test_comps_come_back_ordered_by_rmse() -> None:
    """Closest first -- that ordering is the whole product."""
    target = [10.0, 10.0, 10.0, 10.0, 10.0]
    prepared = _prepared([
        (1, 2010, 25, [12.0] * 5),   # off by 2.0
        (2, 2011, 25, [10.5] * 5),   # off by 0.5  <- closest
        (3, 2012, 25, [7.0] * 5),    # off by 3.0
    ])
    got = closest_paths(prepared, target, age=25, n=3)
    assert [c.mlbam_id for c in got] == [2, 1, 3]
    assert got[0].rmse < got[1].rmse < got[2].rmse
    assert got[0].path == (10.5, 10.5, 10.5, 10.5, 10.5)


def test_a_near_age_row_is_not_a_candidate() -> None:
    """EXACT age, so the comp's +1..+5 lands on the query's projected ages. A
    26-year-old's next five years are a different five years on the x-axis."""
    target = [10.0] * 5
    prepared = _prepared([
        (1, 2010, 26, [10.0] * 5),   # perfect match, WRONG age
        (2, 2011, 25, [14.0] * 5),   # bad match, right age
    ])
    got = closest_paths(prepared, target, age=25, n=5)
    assert [c.mlbam_id for c in got] == [2]


def test_a_short_path_cannot_win_by_having_less_to_match() -> None:
    """The recency rule. A row whose season+5 runs past `last` has unrealized years,
    and scoring it on the two that exist would beat a five-year match for free."""
    target = [10.0] * 5
    prepared = _prepared(
        [
            (1, 2019, 25, [10.0, 10.0, 0.0, 0.0, 0.0]),  # only +1 and +2 realized
            (2, 2010, 25, [11.0] * 5),                    # all five realized
        ],
        last=2021,
    )
    got = closest_paths(prepared, target, age=25, n=5)
    assert [c.mlbam_id for c in got] == [2], "the 2019 row has no realized +3..+5"


def test_ties_break_deterministically_regardless_of_row_order() -> None:
    """Two identical paths must not swap between reads -- the arbitrary-ordering
    defect `index_rosters` was fixed for in 06bf2646, one module over."""
    target = [10.0] * 5
    rows = [
        (7, 2011, 25, [12.0] * 5),
        (3, 2010, 25, [12.0] * 5),
    ]
    forward = [c.mlbam_id for c in closest_paths(_prepared(rows), target, age=25, n=2)]
    reverse = [c.mlbam_id for c in closest_paths(_prepared(rows[::-1]), target, age=25, n=2)]
    assert forward == [3, 7], "tie breaks on mlbam_id ascending"
    assert forward == reverse


def test_no_candidates_returns_empty_rather_than_raising() -> None:
    prepared = _prepared([(1, 2010, 30, [10.0] * 5)])
    assert closest_paths(prepared, [10.0] * 5, age=25, n=5) == []


def test_n_larger_than_the_candidate_pool_returns_what_exists() -> None:
    prepared = _prepared([(1, 2010, 25, [10.0] * 5)])
    assert len(closest_paths(prepared, [10.0] * 5, age=25, n=10)) == 1
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `pytest tests/test_trajectory/test_comp_paths.py -q`
Expected: FAIL at collection — `ModuleNotFoundError: No module named 'fantasy_baseball.trajectory.comp_paths'`.

- [ ] **Step 3: Carry `mlbam_id` on `Prepared`**

In `src/fantasy_baseball/trajectory/shape.py`, add the field to the dataclass immediately after `season`:

```python
    season: np.ndarray
    #: MLBAM id per history row, aligned to every other array here. `prepare` already
    #: builds this to reindex `forward` and then dropped it, so anything wanting to NAME
    #: a matched row had to rebuild `build_history` alongside and trust that two row
    #: orders agreed. Carrying it makes that class of silent misalignment unreachable.
    mlbam_id: np.ndarray
```

and pass it in `prepare`'s return, where `ids` is already in scope:

```python
        season=seasons,
        mlbam_id=ids,
```

- [ ] **Step 4: Write `comp_paths.py`**

```python
"""Historical paths closest to a predicted one.

Given a shape-predicted five-year path, find the player-seasons whose REALIZED path
minimizes RMSE against it. This asks a different question from the comp matchers in
`comps.py`: those pick a cohort that looks similar at the STARTING point and average
what it did, while this picks whole forward paths that match the prediction -- which is
the thing a chart of that prediction actually draws.

THE RESULT IS SELECTED ON THE OUTCOME. These are the paths that happened to land closest
out of ~1,200. That makes them a fair illustration of what this shape looked like when it
played out, and it makes them NOT evidence for the prediction. A consumer that draws them
without the p10-p90 band beside them is making the forecast look more certain than it is.

Everything needed is already on `Prepared`: `forward[h]` is realized SGP for every history
row at +h, with `age`, `season` and `mlbam_id` alongside. The match is a broadcast subtract
and a sort -- about 0.2 ms for one query against a live hitter panel.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

from fantasy_baseball.trajectory.shape import Prepared


@dataclass(frozen=True)
class CompPath:
    """One historical season whose forward path is close to the prediction."""

    mlbam_id: int
    season: int
    #: Root mean squared error against the predicted path, over every horizon.
    rmse: float
    #: The REALIZED forward path, one value per horizon, ascending.
    path: tuple[float, ...]


def closest_paths(
    prepared: Prepared,
    predicted: Sequence[float],
    age: int,
    n: int,
) -> list[CompPath]:
    """The `n` realized paths closest to `predicted`, best first.

    Ids, never names: naming needs the people cache, and keeping it out of here is what
    lets this be tested against a hand-built `Prepared` with no data files at all.
    """
    horizons = tuple(sorted(prepared.horizons))
    target = np.asarray(predicted, dtype=float)
    if target.size != len(horizons):
        raise ValueError(
            f"predicted has {target.size} values but prepared carries "
            f"{len(horizons)} horizons {horizons}"
        )

    # EXACT age, and every forward year realized. `forward` stores a real 0.0 for "out of
    # the league", which is indistinguishable from "has not happened yet" -- so the
    # censoring has to come from the season, not the value. Horizons ascend, so clearing
    # the longest clears them all.
    candidates = np.flatnonzero(
        (prepared.age == float(age)) & (prepared.season + horizons[-1] <= prepared.last)
    )
    if candidates.size == 0:
        return []

    paths = np.column_stack([prepared.forward[h][candidates] for h in horizons])
    rmse = np.sqrt(((paths - target) ** 2).mean(axis=1))

    # Sorted on the full key, not just rmse: two identical paths would otherwise swap
    # between reads on nothing but row order.
    order = sorted(
        range(candidates.size),
        key=lambda i: (
            float(rmse[i]),
            int(prepared.mlbam_id[candidates[i]]),
            int(prepared.season[candidates[i]]),
        ),
    )
    return [
        CompPath(
            mlbam_id=int(prepared.mlbam_id[candidates[i]]),
            season=int(prepared.season[candidates[i]]),
            rmse=float(rmse[i]),
            path=tuple(float(v) for v in paths[i]),
        )
        for i in order[:n]
    ]
```

- [ ] **Step 5: Run the tests to verify they pass**

Run: `pytest tests/test_trajectory/test_comp_paths.py -q`
Expected: PASS (6 tests).

**There is deliberately no test that "matching is on SGP, not VAR."** That property is
structural, not behavioural: `closest_paths` takes no floor, receives no slot, and reads
`Prepared.forward`, which holds raw SGP. There is no code path that could net anything, so
any such test would pass against every possible implementation. The guarantee is recorded
in the module docstring and enforced by the signature -- adding a floor parameter is what a
reviewer should object to. A test that cannot fail is worse than none: it reads as coverage.

- [ ] **Step 6: Mutation-check the three rules that matter**

Each must fail the named test, then be reverted. Report all three transcripts.

1. Drop the age filter — change the mask to `(prepared.season + horizons[-1] <= prepared.last)` alone. `test_a_near_age_row_is_not_a_candidate` must fail.
2. Drop the censoring — change the mask to `(prepared.age == float(age))` alone. `test_a_short_path_cannot_win_by_having_less_to_match` must fail.
3. Sort on rmse alone — `key=lambda i: float(rmse[i])`. `test_ties_break_deterministically_regardless_of_row_order` must fail.

A guard that cannot fail is not a guard.

- [ ] **Step 7: Run the checks and commit**

Run: `pytest tests/test_trajectory/ -q && ruff check . && ruff format --check . && mypy && vulture`

`vulture` may report `CompPath` fields as unused until Task 2 consumes them; that is expected and resolves there.

```bash
git add src/fantasy_baseball/trajectory/comp_paths.py \
        src/fantasy_baseball/trajectory/shape.py \
        tests/test_trajectory/test_comp_paths.py
git commit -m "feat(trajectory): find the realized paths closest to a predicted one"
```

---

### Task 2: Bake `history` and `comps` into the payload

**Files:**
- Modify: `scripts/push_trajectory_board.py` (`build_payload`)
- Modify: `src/fantasy_baseball/trajectory/sweep.py` (`to_payload`, to carry the two new per-player keys)
- Test: `tests/test_scripts/test_push_trajectory_board.py`

**Interfaces:**
- Consumes: `closest_paths`, `CompPath` from Task 1; `prepare` from `trajectory.shape`; `player_names` from `trajectory.board`.
- Produces: two new keys on each payload player — `history: list[[int, float]]` and `comps: list[dict]` with keys `name`, `season`, `rmse`, `path`. `MAX_COMPS = 10` in `push_trajectory_board.py`. Tasks 3 and 4 read these.

- [ ] **Step 1: Write the failing test**

Append to `tests/test_scripts/test_push_trajectory_board.py`:

```python
def test_the_payload_carries_career_history_and_comps(monkeypatch) -> None:
    """Both are computed where the panel lives and travel in the same blob.

    The dashboard has no panel and never will -- `data/trajectory/` is gitignored and
    absent on Render -- so anything the chart needs has to be baked here or it does not
    exist at request time.
    """
    module = _script()
    payload, scored = module.build_payload(max_horizon=5, panel_dir=PROJECT_ROOT / "data" / "trajectory")

    assert scored > 0
    player = payload["players"][0]

    assert player["history"], "every scored player has at least his current season"
    ages = [row[0] for row in player["history"]]
    assert ages == sorted(ages), "history ascends by age so a line can be drawn from it"
    assert all(len(row) == 2 for row in player["history"])

    assert len(player["comps"]) <= module.MAX_COMPS
    if player["comps"]:
        first = player["comps"][0]
        assert set(first) == {"name", "season", "rmse", "path"}
        assert len(first["path"]) == 5
        rmses = [c["rmse"] for c in player["comps"]]
        assert rmses == sorted(rmses), "closest first"
```

Note: this test runs the real sweep and is slow (~60s). Mark it `@pytest.mark.slow` if that marker exists in `pyproject.toml`; otherwise leave it unmarked and accept the cost — it is the only test that proves the two halves meet.

- [ ] **Step 2: Run it to verify it fails**

Run: `pytest tests/test_scripts/test_push_trajectory_board.py -q -k history_and_comps`
Expected: FAIL — `KeyError: 'history'`.

- [ ] **Step 3: Let `to_payload` carry the new keys**

In `src/fantasy_baseball/trajectory/sweep.py`, `to_payload` builds each player dict. Add the two keys, defaulting to empty so the function stays usable by callers that have neither (the CLI and every existing test):

```python
def to_payload(
    players: Iterable[SweptPlayer],
    *,
    extras: dict[int, dict] | None = None,
    **meta: Any,
) -> dict:
    """Serialize a sweep for the KV. `meta` carries the vintage the reader must show.

    `extras` is per-player data the SWEEP does not produce -- career history and comps,
    which need the panel and the people cache rather than the fit. Keyed by mlbam_id and
    merged in here so `SweptPlayer` does not grow fields that only one consumer wants.
    """
    per_player = extras or {}
    return {
        **meta,
        "players": [
            {
                "id": p.mlbam_id,
                ...
                "sgp": [_pack(y) for y in p.sgp],
                **per_player.get(p.mlbam_id, {}),
            }
            for p in players
        ],
    }
```

Keep every existing key exactly as it is; only the `extras` parameter and the `**per_player.get(...)` spread are new.

- [ ] **Step 4: Compute both in `build_payload`**

In `scripts/push_trajectory_board.py`, add the constant near `MIN_SGP`:

```python
#: Comps stored per player. TEN, not the five the chart shows by default: `N` is a
#: display control and this blob is built hours earlier by another process, so the
#: control can only ever slice what was stored. Ten is the control's ceiling, so every
#: legal N is servable. Costs ~370 bytes a player over five.
MAX_COMPS = 10
```

Inside the per-kind loop, after `produced = sweep_pool(rows, complete, kind, horizons)`, build the extras:

```python
        # `sweep_pool` builds its own prepared state and does not return it. Preparing a
        # second time costs one vectorized reindex per pool -- cheap next to the sweep,
        # and far cheaper than widening `sweep_pool`'s signature, which the CLI and its
        # tests also call.
        from fantasy_baseball.trajectory.comp_paths import closest_paths
        from fantasy_baseball.trajectory.shape import prepare

        prepared = prepare(complete, kind=kind, horizons=horizons)
        by_id = {int(i): g for i, g in complete.groupby("mlbam_id")}

        for player in produced:
            seasons = by_id.get(player.mlbam_id)
            history = (
                [[int(a), round(float(s), 4)]
                 for a, s in zip(seasons["age"], seasons["sgp"], strict=True)]
                if seasons is not None
                else []
            )
            comps = closest_paths(
                prepared,
                [point.mean for point in player.sgp],
                age=player.age,
                n=MAX_COMPS,
            )
            extras[player.mlbam_id] = {
                "history": history,
                "comps": [
                    {
                        # Named HERE, not in `closest_paths`: naming needs the people
                        # cache, and keeping it out of that module is what lets it be
                        # tested with no data files. An unknown id renders as its id
                        # rather than vanishing -- a comp is still a comp.
                        "name": names.get(c.mlbam_id, str(c.mlbam_id)),
                        "season": c.season,
                        "rmse": round(c.rmse, 3),
                        "path": [round(v, 3) for v in c.path],
                    }
                    for c in comps
                ],
            }
```

Initialise `extras: dict[int, dict] = {}` beside `swept = []`, and pass it through: `to_payload(swept, extras=extras, base_season=season, ...)`.

- [ ] **Step 5: Run the test**

Run: `pytest tests/test_scripts/test_push_trajectory_board.py -q`
Expected: PASS.

- [ ] **Step 6: Measure the real payload**

Run: `python scripts/push_trajectory_board.py --dry-run`
Expected: it prints a payload size near **1.8 MB** (was 762 KB). Record the actual figure in your report — the spec predicted ~1.84 MB and a large miss means one of the two extras is bigger than modelled.

- [ ] **Step 7: Run the checks and commit**

Run: `pytest tests/test_trajectory/ tests/test_scripts/ -q && ruff check . && ruff format --check . && mypy && vulture`

```bash
git add scripts/push_trajectory_board.py src/fantasy_baseball/trajectory/sweep.py \
        tests/test_scripts/test_push_trajectory_board.py
git commit -m "feat(trajectory): bake career history and comps into the pushed board"
```

---

### Task 3: `build_player_view`

**Files:**
- Modify: `src/fantasy_baseball/web/trajectory_view.py`
- Test: `tests/test_web/test_trajectory_view.py`

**Interfaces:**
- Consumes: `_sweep_setup`, `_clamp`, `_clamp_choice`, `SCALES`, `POOLS`, `_ranked_rows` from the same module.
- Produces: `DEFAULT_COMPS = 5`; `PlayerView` (frozen dataclass: `name`, `age`, `slot`, `floor`, `scale`, `n`, `history`, `projection`, `comps`, `candidates`, `found`, `extrapolated`, `base_season`, `end_years`, `meta`); `build_player_view(payload, *, player, scale="var", n=None) -> PlayerView`. Task 4 renders it.

- [ ] **Step 1: Write the failing tests**

Add to `tests/test_web/test_trajectory_view.py`. The module already has the `payload` fixture (six players) and `_spot`; use them.

```python
def _payload_with_extras(payload: dict) -> dict:
    """The `payload` fixture plus the two keys the push script bakes.

    Built here rather than in the fixture so every OTHER test keeps exercising the
    `.get(..., [])` fallback path for a blob that predates this feature.
    """
    out = dict(payload)
    out["generated_at"] = f"hand-{next(_HAND_SEQ)}"
    out["players"] = [
        {
            **p,
            "history": [[25, 14.0], [26, 16.0], [27, 20.0]],
            "comps": [
                {"name": f"Comp {i}", "season": 2010 + i, "rmse": 0.5 * i,
                 "path": [10.0 - i, 9.0 - i, 8.0 - i, 7.0 - i, 6.0 - i]}
                for i in range(1, 8)
            ],
        }
        for p in payload["players"]
    ]
    return out


def test_a_player_is_found_by_name_with_his_career_and_projection(payload: dict) -> None:
    view = build_player_view(_payload_with_extras(payload), player="Big Bat")
    assert view.found
    assert view.name == "Big Bat"
    assert [pt[0] for pt in view.history] == [25, 26, 27], "career ascends by age"
    assert len(view.projection) == 3, "the fixture sweeps three horizons"
    assert all(p["p10"] <= p["mean"] <= p["p90"] for p in view.projection)


def test_an_unknown_name_is_not_found_and_lists_nobody(payload: dict) -> None:
    view = build_player_view(_payload_with_extras(payload), player="Nobody At All")
    assert not view.found
    assert view.candidates == []


def test_an_ambiguous_name_lists_candidates_and_renders_no_chart(payload: dict) -> None:
    """Two players can share a normalized name -- the live board carries two hitters
    called Max Muncy. Guessing puts one man's career under another's name."""
    twin = _payload_with_extras(payload)
    first = twin["players"][0]
    twin["players"] = [*twin["players"], {**first, "id": first["id"] + 10_000}]
    twin["generated_at"] = f"hand-{next(_HAND_SEQ)}"

    view = build_player_view(twin, player=first["name"])
    assert not view.found, "an ambiguous name renders no chart"
    assert len(view.candidates) == 2
    assert {c["id"] for c in view.candidates} == {first["id"], first["id"] + 10_000}


def test_comps_are_sliced_to_n_and_n_is_clamped(payload: dict) -> None:
    full = _payload_with_extras(payload)
    assert len(build_player_view(full, player="Big Bat").comps) == 5, "default"
    assert len(build_player_view(full, player="Big Bat", n=3).comps) == 3
    assert len(build_player_view(full, player="Big Bat", n=999).comps) == 7, "what exists"
    assert len(build_player_view(full, player="Big Bat", n="junk").comps) == 5
    assert len(build_player_view(full, player="Big Bat", n=0).comps) == 1


def test_a_payload_without_the_new_keys_still_renders_the_projection(payload: dict) -> None:
    """A blob pushed before this feature. #332 took /trajectory down by refusing one it
    could largely read; this renders what it has and lets the page say what is missing."""
    view = build_player_view(payload, player="Big Bat")
    assert view.found
    assert view.projection, "the fit is in every payload"
    assert view.history == []
    assert view.comps == []


def test_the_var_axis_nets_every_series_against_the_QUERY_players_floor(payload: dict) -> None:
    """Career, projection and comps alike. Netting each comp against its own slot would
    put lines on one axis that are not comparable -- the mixed-scale defect of #331."""
    full = _payload_with_extras(payload)
    var = build_player_view(full, player="Under Water", scale="var")
    sgp = build_player_view(full, player="Under Water", scale="sgp")
    floor = var.floor
    assert floor > 0, "fixture must net against a real floor"

    assert var.history[0][1] == pytest.approx(sgp.history[0][1] - floor)
    assert var.projection[0]["mean"] == pytest.approx(sgp.projection[0]["mean"] - floor)
    assert var.comps[0]["path"][0]["value"] == pytest.approx(
        sgp.comps[0]["path"][0]["value"] - floor
    )
    assert [c["name"] for c in var.comps] == [c["name"] for c in sgp.comps], "same comps"
```

- [ ] **Step 2: Run them to verify they fail**

Run: `pytest tests/test_web/test_trajectory_view.py -q -k "player_view or ambiguous or comps_are_sliced or var_axis or without_the_new_keys or unknown_name"`
Expected: FAIL at collection — `ImportError: cannot import name 'build_player_view'`. Add it to the module's import line in the test file.

- [ ] **Step 3: Add the dataclass and the builder**

In `trajectory_view.py`:

```python
#: Comps drawn by default. The payload stores `MAX_COMPS` (10); this is what the chart
#: shows until asked otherwise.
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
    #: [[age, value], ...] realized, ascending. Empty for a payload predating the feature.
    history: list[list[float]]
    #: [{age, mean, p10, p90}, ...] one per projected year.
    projection: list[dict]
    #: [{name, season, rmse, path: [{age, value}, ...]}, ...] closest first.
    comps: list[dict]
    #: Populated ONLY when the name was ambiguous; the caller renders these instead of a
    #: chart. Guessing puts one man's career under another's name.
    candidates: list[dict]
    found: bool
    extrapolated: bool
    base_season: int
    end_years: list[int]
    meta: dict = field(default_factory=dict)


def build_player_view(
    payload: dict,
    *,
    player: str,
    scale: str = "var",
    n: Any = None,
) -> PlayerView:
    """One player's career, projection and comps, on one scale.

    Resolved BY NAME, never by an id from the query string. CLAUDE.md names a
    hand-carried id as a defect class that has twice landed on a real row belonging to
    someone else, and `player_trajectory.py` already refuses a `--mlbam-id` that
    disagrees with its `--player`.
    """
    scale = _clamp_choice(scale, SCALES, "var")
    want = _clamp(n, 1, MAX_COMPS_SHOWN, DEFAULT_COMPS)
    base = int(payload["base_season"])
    max_horizon = int(payload["max_horizon"])
    end_years = [base + h for h in range(1, max_horizon + 1)]

    target = normalize_name(player or "")
    hits = [p for p in payload.get("players", []) if normalize_name(p["name"]) == target]

    empty = PlayerView(
        name=player or "", age=0, slot="", floor=0.0, scale=scale, n=want,
        history=[], projection=[], comps=[], candidates=[], found=False,
        extrapolated=False, base_season=base, end_years=end_years,
        meta=_board_meta(payload),
    )
    if not hits:
        return empty
    if len(hits) > 1:
        return replace(
            empty,
            candidates=[
                {"id": p["id"], "name": p["name"], "age": p["age"], "slot": p["slot"]}
                for p in sorted(hits, key=lambda p: p["id"])
            ],
        )

    row = hits[0]
    floor = float(row["floor"]) if scale == "var" else 0.0
    return replace(
        empty,
        name=row["name"],
        age=int(row["age"]),
        slot=row["slot"],
        floor=float(row["floor"]),
        found=True,
        extrapolated=bool(row.get("extrapolated")),
        history=[[int(a), float(v) - floor] for a, v in row.get("history", [])],
        projection=[
            {
                "age": int(row["age"]) + int(pt["horizon"]),
                "mean": float(pt["mean"]) - floor,
                "p10": float(pt["p10"]) - floor,
                "p90": float(pt["p90"]) - floor,
            }
            for pt in row["sgp"]
        ],
        comps=[
            {
                "name": c["name"],
                "season": c["season"],
                "rmse": c["rmse"],
                # Truncated to the PROJECTED horizons, not the stored path length.
                # The payload always stores five, but a board swept to three years draws
                # three -- a comp running two ages past the projection would be the only
                # line on the chart with no dashed line beside it to compare against.
                "path": [
                    {"age": int(row["age"]) + h, "value": float(v) - floor}
                    for h, v in list(enumerate(c["path"], start=1))[: len(row["sgp"])]
                ],
            }
            for c in row.get("comps", [])[:want]
        ],
    )
```

Add `MAX_COMPS_SHOWN = 10` beside `DEFAULT_COMPS` (the clamp ceiling; it mirrors the push script's `MAX_COMPS` and the two are asserted equal in Task 4's route test), and `from dataclasses import replace` to the imports if absent.

- [ ] **Step 4: Run the tests**

Run: `pytest tests/test_web/test_trajectory_view.py -q`
Expected: PASS.

- [ ] **Step 5: Mutation-check the two rules that carry money**

1. Net comps against a fixed 0.0 instead of `floor` — `test_the_var_axis_nets_every_series_against_the_QUERY_players_floor` must fail.
2. Return the first hit instead of listing candidates when `len(hits) > 1` — `test_an_ambiguous_name_lists_candidates_and_renders_no_chart` must fail.

Report both transcripts.

- [ ] **Step 6: Run the checks and commit**

Run: `pytest tests/test_web/ tests/test_trajectory/ -q && ruff check . && ruff format --check . && mypy && vulture`

```bash
git add src/fantasy_baseball/web/trajectory_view.py tests/test_web/test_trajectory_view.py
git commit -m "feat(trajectory): build_player_view -- one player's career, projection and comps"
```

---

### Task 4: The route, the template, the chart, and the view pill

**Files:**
- Modify: `src/fantasy_baseball/web/trajectory_view.py` (`VIEWS`, `filter_state`)
- Modify: `src/fantasy_baseball/web/season_routes.py`
- Modify: `src/fantasy_baseball/web/templates/season/_trajectory_controls.html`
- Create: `src/fantasy_baseball/web/templates/season/trajectory_player.html`
- Create: `src/fantasy_baseball/web/static/trajectory_chart.js`
- Modify: `src/fantasy_baseball/web/static/season_trends.js` (export the Chart.js loader)
- Test: `tests/test_web/test_season_routes.py`

**Interfaces:**
- Consumes: `build_player_view`, `PlayerView`, `DEFAULT_COMPS`, `MAX_COMPS_SHOWN`, `select_view`, `filter_state` from Task 3 and the existing module.
- Produces: no new Python interfaces; `?view=player&player=<name>&n=<int>` becomes part of the board's URL state.

**The pills ship in this commit, with the route branch that answers them** — a control landing a commit before its handler renders, is clickable, and clamps straight back. That is `c96cd79b`'s defect, which left `/trajectory` returning 500 at one revision.

- [ ] **Step 1: Write the failing route tests**

Append to `tests/test_web/test_season_routes.py`:

```python
def _trajectory_payload_with_extras():
    """The route fixture plus the keys the push script bakes."""
    payload = _trajectory_payload()
    payload["players"] = [
        {
            **p,
            "history": [[25, 14.0], [26, 16.0]],
            "comps": [
                {"name": "Andre Ethier", "season": 2007, "rmse": 1.25,
                 "path": [12.7, 14.4, 12.1, 9.2, 12.2]},
                {"name": "Bryan Reynolds", "season": 2020, "rmse": 1.31,
                 "path": [11.0, 12.0, 11.5, 10.0, 9.5]},
            ],
        }
        for p in payload["players"]
    ]
    return payload


def test_trajectory_player_view_renders_a_chart_for_a_resolved_name(client):
    with patch(
        "fantasy_baseball.web.season_routes.read_cache_dict",
        return_value=_trajectory_payload_with_extras(),
    ):
        resp = client.get("/trajectory?view=player&player=Testy+McTestface")
    assert resp.status_code == 200
    body = resp.data.decode()
    assert "trajectory-chart" in body, "the canvas the chart draws into"
    assert "Andre Ethier" in body, "comps are named"
    assert "closest realized paths" in body, "labelled as illustration, not evidence"
    assert "1.25" in body, "each comp shows its RMSE"


def test_trajectory_player_view_states_the_five_year_comp_rule(client):
    """A reader who notices no comp is recent must find the rule, not infer a bug."""
    with patch(
        "fantasy_baseball.web.season_routes.read_cache_dict",
        return_value=_trajectory_payload_with_extras(),
    ):
        resp = client.get("/trajectory?view=player&player=Testy+McTestface")
    assert "five realized seasons" in resp.data.decode()


def test_trajectory_player_view_with_no_name_renders_the_search_box(client):
    with patch(
        "fantasy_baseball.web.season_routes.read_cache_dict",
        return_value=_trajectory_payload_with_extras(),
    ):
        resp = client.get("/trajectory?view=player")
    assert resp.status_code == 200
    assert 'name="player"' in resp.data.decode(), "the search input"


def test_trajectory_player_view_unknown_name_does_not_500(client):
    with patch(
        "fantasy_baseball.web.season_routes.read_cache_dict",
        return_value=_trajectory_payload_with_extras(),
    ):
        resp = client.get("/trajectory?view=player&player=Nobody+At+All")
    assert resp.status_code == 200
    assert "No player named" in resp.data.decode()


def test_the_three_trajectory_views_coexist(client):
    """Each renders its own thing, and the other two are still reachable."""
    with (
        patch("fantasy_baseball.web.season_routes.read_cache_dict",
              return_value=_trajectory_payload_with_extras()),
        patch("fantasy_baseball.data.rosters.live_rosters", return_value=_trajectory_spots()),
    ):
        board = client.get("/trajectory")
        teams = client.get("/trajectory?view=teams")
        player = client.get("/trajectory?view=player&player=Testy+McTestface")
    assert all(r.status_code == 200 for r in (board, teams, player))
    assert "All teams" in board.data.decode()
    assert "team-block" in teams.data.decode()
    assert "trajectory-chart" in player.data.decode()


def test_the_stored_and_displayed_comp_ceilings_agree(client):
    """The view clamps N to a ceiling; the push script stores that many. If they drift,
    the control asks for comps the blob never carried."""
    import importlib.util

    from fantasy_baseball.web.trajectory_view import MAX_COMPS_SHOWN

    spec = importlib.util.spec_from_file_location(
        "push_trajectory_board", PROJECT_ROOT / "scripts" / "push_trajectory_board.py"
    )
    push = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(push)
    assert push.MAX_COMPS == MAX_COMPS_SHOWN
```

`PROJECT_ROOT` already exists in that file; reuse it rather than re-deriving.

- [ ] **Step 2: Run them to verify they fail**

Run: `pytest tests/test_web/test_season_routes.py -q -k "player_view or three_trajectory or comp_ceilings"`
Expected: FAIL — `?view=player` currently clamps to `board`, so no canvas renders.

- [ ] **Step 3: Widen `VIEWS` and `filter_state`**

In `trajectory_view.py`:

```python
VIEWS = ("board", "teams", "player")
```

and in `filter_state`, add the two fields the player view owns, keeping the same
owned-vs-query rule the docstring already states:

Replace the whole body -- the existing `owned` boolean reads as "the one other view"
and stops being meaningful at three. This is the function in full:

```python
    owned_teams = view == "teams"
    owned_player = view == "player"
    return {
        "view": view,
        "end_year": board.end_year if board else 0,
        "pool": board.pool if board else "both",
        "scale": board.scale if board else "var",
        # Owned by `build_teams_board` on the teams view; by the query string otherwise.
        "per": board.per_team if (owned_teams and board) else args.get("per", DEFAULT_PER_TEAM),
        # Owned by `build_board` on the league view; by the query string otherwise.
        "top": (
            args.get("top", DEFAULT_TOP)
            if (owned_teams or owned_player)
            else (board.top if board else DEFAULT_TOP)
        ),
        "team": (
            args.get("team", "all")
            if (owned_teams or owned_player)
            else (board.team if board else "all")
        ),
        # Owned by the player view; a pass-through elsewhere so a round trip through
        # another view does not drop the searched name.
        "player": board.name if (owned_player and board) else args.get("player", ""),
        "n": board.n if (owned_player and board) else args.get("n", DEFAULT_COMPS),
    }
```

Note `top` and `team` now pass through on BOTH non-board views. Getting that wrong is
what dropped the team filter on a view round trip once already.

- [ ] **Step 4: Branch the route**

In `season_routes.py`, add a third branch before the teams branch:

```python
            if view == "player":
                player_view = build_player_view(
                    payload,
                    player=request.args.get("player", ""),
                    scale=request.args.get("scale", "var"),
                    n=request.args.get("n"),
                )
```

and render it:

```python
        if view == "player" and player_view is not None:
            return render_template(
                "season/trajectory_player.html",
                meta=read_meta(),
                active_page="trajectory",
                board=player_view,
                error=error,
                cur=filter_state("player", player_view, request.args),
            )
```

Initialise `player_view = None` beside `teams_board = None`, and add `build_player_view`
to the route's import block. The player view needs no roster read, so it renders whether
or not `live_rosters` succeeded.

- [ ] **Step 5: Add the third pill**

In `_trajectory_controls.html`, the existing view pill group becomes three. The player pill
is NOT gated on `teams` — it needs no roster data:

```jinja
  <span class="pill-group">
    <a class="pill {% if cur.view == 'board' %}active{% endif %}" href="{{ board_url(cur, view='board') }}">League</a>
    {% if teams %}
    <a class="pill {% if cur.view == 'teams' %}active{% endif %}" href="{{ board_url(cur, view='teams') }}">By team</a>
    {% endif %}
    <a class="pill {% if cur.view == 'player' %}active{% endif %}" href="{{ board_url(cur, view='player') }}">Player</a>
  </span>
```

and `board_url` gains `player` and `n`, threaded like every other filter:

```jinja
           player=player if player is not none else cur.player,
           n=n if n is not none else cur.n) }}
```

with `player=None, n=None` added to the macro signature.

- [ ] **Step 6: Create the template**

Create `src/fantasy_baseball/web/templates/season/trajectory_player.html`:

```jinja
{% extends "season/base.html" %}
{% block title %}Player Trajectory -- Season Dashboard{% endblock %}
{% block content %}
{% import "season/_trajectory_controls.html" as ctl %}

<div class="page-trajectory">
<div class="page-header"><h2>Player Trajectory</h2></div>

{% if error %}<p class="warning">{{ error }}</p>{% else %}

{{ ctl.controls(cur, board.end_years, []) }}

<form method="get" class="trajectory-search">
  <input type="hidden" name="view" value="player">
  <input type="hidden" name="scale" value="{{ board.scale }}">
  <input type="hidden" name="n" value="{{ board.n }}">
  <label>Player <input name="player" value="{{ board.name }}" placeholder="Juan Soto"></label>
  <button type="submit">Search</button>
</form>

{% if board.candidates %}
<p class="warning">More than one player is named "{{ board.name }}" on this board. Roster
blobs carry no MLBAM id (#284), so the name cannot tell them apart -- pick one:</p>
<ul>
  {% for c in board.candidates %}
  <li>{{ c.name }} -- age {{ c.age }}, {{ c.slot }} (id {{ c.id }})</li>
  {% endfor %}
</ul>

{% elif not board.found %}
  {% if board.name %}<p>No player named "{{ board.name }}" on this board.</p>
  {% else %}<p class="muted">Search a player to see his trajectory.</p>{% endif %}

{% else %}
<p class="muted">
  {{ board.name }} -- age {{ board.age }}, {{ board.slot }}.
  Solid is what happened; dashed is projected, with the shaded band its p10-p90 range.
  {% if board.extrapolated %}<strong>(!)</strong> This fit was evaluated outside its own
  support -- read the band, not the point estimate.{% endif %}
</p>

<div class="chart-wrapper"><canvas id="trajectory-chart"></canvas></div>
<script type="application/json" id="trajectory-chart-data">{{ {
  "name": board.name, "scale": board.scale, "history": board.history,
  "projection": board.projection, "comps": board.comps} | tojson }}</script>

<h3>Closest realized paths <span class="muted">({{ board.comps | length }})</span></h3>
<p class="muted">
  These are the historical paths that landed closest to the projection -- chosen ON the
  outcome, so they illustrate what this shape looked like when it played out and are
  <strong>not evidence for the forecast</strong>. The honest uncertainty is the shaded
  band, which is wider than they are by construction. A comp needs five realized seasons
  to be scored on the same horizons, so none is recent.
</p>
<table class="data-table">
  <thead><tr><th>Player</th><th>Season</th><th>RMSE</th>
    {% for pt in board.projection %}<th>{{ pt.age }}</th>{% endfor %}</tr></thead>
  <tbody>
    {% for c in board.comps %}
    <tr><td>{{ c.name }}</td><td>{{ c.season }}</td><td>{{ '%.2f'|format(c.rmse) }}</td>
      {% for pt in c.path %}<td>{{ '%.1f'|format(pt.value) }}</td>{% endfor %}</tr>
    {% endfor %}
  </tbody>
</table>

<h3>The numbers</h3>
<table class="data-table">
  <thead><tr><th>Age</th><th>{{ board.scale|upper }}</th><th>p10..p90</th></tr></thead>
  <tbody>
    {% for a, v in board.history %}
    <tr><td>{{ a }}</td><td>{{ '%.1f'|format(v) }}</td><td class="muted">actual</td></tr>
    {% endfor %}
    {% for pt in board.projection %}
    <tr><td>{{ pt.age }}</td><td>{{ '%.1f'|format(pt.mean) }}</td>
      <td>{{ '%.1f'|format(pt.p10) }}..{{ '%.1f'|format(pt.p90) }}</td></tr>
    {% endfor %}
  </tbody>
</table>
{% endif %}
{% endif %}
</div>
<script src="{{ url_for('static', filename='trajectory_chart.js') }}"></script>
{% endblock %}
```

- [ ] **Step 7: Export the Chart.js loader and write the chart**

In `season_trends.js`, the `ensureChartJs()` function is inside an IIFE. Expose it at the
end of that IIFE so a second page can reuse the loader rather than re-implementing its
race guard:

```js
  window.ensureChartJs = ensureChartJs;
```

Create `src/fantasy_baseball/web/static/trajectory_chart.js`:

```js
// One player's trajectory: career solid, projection dashed, p10-p90 as a filled band,
// comps thin and faint across the projected ages only.
//
// The band must stay visually dominant over the comps. They were selected ON the outcome
// -- the closest paths out of ~1,200 -- so drawn as equals they hug the projection and
// make the forecast look far more certain than it is.
(function () {
  const node = document.getElementById("trajectory-chart-data");
  const canvas = document.getElementById("trajectory-chart");
  if (!node || !canvas) return;
  const data = JSON.parse(node.textContent);

  const at = (pts, key) => pts.map((p) => ({ x: p.age, y: p[key] }));

  const datasets = [
    // Comps first so later datasets paint over them.
    ...data.comps.map((c) => ({
      label: `${c.name} (${c.season})`,
      data: c.path.map((p) => ({ x: p.age, y: p.value })),
      borderColor: "rgba(120,120,120,0.35)",
      borderWidth: 1,
      pointRadius: 0,
      order: 3,
    })),
    {
      label: "p10-p90",
      data: at(data.projection, "p90"),
      borderColor: "transparent",
      backgroundColor: "rgba(78,121,167,0.18)",
      fill: "+1",
      pointRadius: 0,
      order: 2,
    },
    { label: "_p10", data: at(data.projection, "p10"), borderColor: "transparent",
      pointRadius: 0, fill: false, order: 2 },
    {
      label: "projected",
      data: at(data.projection, "mean"),
      borderColor: "#4e79a7",
      borderDash: [6, 4],
      borderWidth: 2,
      pointRadius: 2,
      order: 1,
    },
    {
      label: data.name,
      data: data.history.map(([age, v]) => ({ x: age, y: v })),
      borderColor: "#4e79a7",
      borderWidth: 2.5,
      pointRadius: 2,
      order: 0,
    },
  ];

  window.ensureChartJs().then(() => {
    new Chart(canvas.getContext("2d"), {
      type: "line",
      data: { datasets },
      options: {
        parsing: false,
        scales: {
          x: { type: "linear", title: { display: true, text: "age" } },
          y: { title: { display: true, text: data.scale.toUpperCase() } },
        },
        plugins: {
          legend: { labels: { filter: (item) => item.text !== "_p10" } },
        },
      },
    });
  });
})();
```

- [ ] **Step 8: Run the tests**

Run: `pytest tests/test_web/ -q`
Expected: PASS.

- [ ] **Step 9: Mutation-check the honesty requirement**

Delete the "not evidence for the forecast" paragraph from the template and confirm
`test_trajectory_player_view_renders_a_chart_for_a_resolved_name` fails on
`"closest realized paths"`. Restore. Then delete the five-realized-seasons sentence and
confirm `test_trajectory_player_view_states_the_five_year_comp_rule` fails. Restore.

These are the assertions most likely to be quietly true for the wrong reason — check that
each string appears in exactly one place in the rendered page before trusting it.

- [ ] **Step 10: Run the full checks and commit**

Run: `pytest -n auto -q && ruff check . && ruff format --check . && mypy && vulture`
Expected: only the pre-existing `resend` failures (1 failed, 5 errors) — confirm they
reproduce on `main` before accepting them.

```bash
git add src/fantasy_baseball/web/ tests/test_web/test_season_routes.py
git commit -m "feat(trajectory): player search, career chart, and closest realized paths"
```

---

## Self-Review

**Spec coverage.** R1 Task 4 Steps 3-6; R2 Task 3 Step 3 (`candidates`/`found`) + Task 4's
unknown/ambiguous tests; R3 Task 4 Step 7 (`borderDash`, `fill`, faint comps); R4 Task 3
Step 3 (comp `path` ages start at `age + 1`); R5 Task 3 Step 3 (`_clamp(n, 1,
MAX_COMPS_SHOWN, DEFAULT_COMPS)`) + its test; R6 Task 3's floor test; R7 Task 4's RMSE and
five-year-rule tests; R8 Task 4 Step 6 (pace note is in `board.meta`, `(!)` from
`extrapolated`); R9 Task 3's no-extras test; R10 Task 1's shuffle test.

Every edge-case row has a home: unknown and ambiguous names (Task 3/4), missing `history`
(Task 3), fewer than `N` comps and no comps at all (Task 3's slice returns what exists,
template renders a count), rookie one-point career (a one-point dataset draws a dot), junk
`?n=` (clamped), Chart.js failing to load (the two tables render server-side and are
outside the `<script>`).

**Placeholder scan.** No TBD/TODO; every code step carries literal code; no "similar to
Task N".

**Type consistency.** `closest_paths`, `CompPath`, `MAX_COMPS`, `MAX_COMPS_SHOWN`,
`DEFAULT_COMPS`, `build_player_view`, `PlayerView` are spelled identically across tasks.
`MAX_COMPS` (push script) and `MAX_COMPS_SHOWN` (view) are deliberately two names for one
number in two processes — Task 4 Step 1 asserts they are equal, which is the only thing
that stops them drifting.

**One gap found while writing:** the spec says the pace note carries through, but
`build_payload` puts `season_elapsed` in `meta`, not per player, and the paced figure
itself is only printed by the CLI. Task 4 Step 6 renders `board.meta.season_elapsed` in the
header; the per-player "8.7 so far, pacing to 12.5" split is NOT available in the payload
and is out of scope here. Recorded rather than silently dropped.
