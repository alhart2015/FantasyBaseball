# Stash Board v3 -- Retire Cost, Rank by P(helps) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Retire the IL-slot `cost` from the stash board and rank candidates by P(helps) (probability the swap helps), showing expected Value alongside.

**Architecture:** Slim `StashScore` to `stash_value` + `band` + `recommended_drop` (drop `gain`/`cost`). Sort by a new `_rank_key` = (P(helps), value). Replace the cost/drop helper with `_weakest_owned_drop` (the owned stash at the bottom of the same ranking). Update the dashboard template and the test suite to the new contract.

**Tech Stack:** Python 3, dataclasses, pytest, Jinja2 (Flask season dashboard), ruff, mypy, vulture.

**Spec:** `docs/superpowers/specs/2026-05-26-stash-retire-cost-rank-by-phelps-design.md`

---

## File Structure

- **Modify** `src/fantasy_baseball/lineup/stash_value.py` -- data model (`StashScore`), ranking (`_rank_key`, sort), drop selection (`_cost_and_drop` -> `_weakest_owned_drop`), `score_stash_candidates` pass-2, docstrings. In `[tool.mypy].files`.
- **Modify** `tests/test_lineup/test_stash_value.py` -- rewrite cost/gain assertions to the new contract; add `_rank_key` and `_weakest_owned_drop` unit tests.
- **Modify** `src/fantasy_baseball/web/templates/season/stash.html` -- Value + P(helps) columns; drop Cost and the now-duplicate Gain column.

`web/refresh_pipeline.py` and `web/season_routes.py` are untouched: they pass `StashResult.to_dict()` through, and `asdict` drops the removed fields automatically.

---

## Task 1: Retire cost + rank by P(helps) in `stash_value.py` and its tests

**Files:**
- Modify: `src/fantasy_baseball/lineup/stash_value.py`
- Test: `tests/test_lineup/test_stash_value.py`

This is one atomic refactor (removing dataclass fields breaks the producer and the tests simultaneously), so tests and code change together and land in a single commit. Steps are ordered red -> green.

- [ ] **Step 1: Rewrite the test imports and the contract tests (red)**

In `tests/test_lineup/test_stash_value.py`, change the import block (lines 5-15) from:

```python
from fantasy_baseball.lineup.il_return_planner import _activate
from fantasy_baseball.lineup.stash_value import (
    StashResult,
    StashScore,
    _cost_and_drop,
    _marginal_value,
    _open_il_slots,
    _owned_il_stashes,
    _solve_active,
    score_stash_candidates,
)
```

to:

```python
from fantasy_baseball.lineup.il_return_planner import _activate
from fantasy_baseball.lineup.stash_value import (
    StashResult,
    StashScore,
    _marginal_value,
    _open_il_slots,
    _owned_il_stashes,
    _rank_key,
    _solve_active,
    _weakest_owned_drop,
    score_stash_candidates,
)
```

Replace `test_stash_score_to_dict_shape` (lines 280-296) with:

```python
def test_stash_score_to_dict_shape():
    s = StashScore(
        name="Blake Snell",
        player_type="pitcher",
        status="IL15",
        owned=False,
        stash_value=4.2,
        band={"mean": 4.2, "sd": 1.1, "p_positive": 0.91, "verdict": "real"},
        recommended_drop=None,
    )
    d = s.to_dict()
    assert d["name"] == "Blake Snell"
    assert d["stash_value"] == 4.2
    assert d["band"]["p_positive"] == 0.91
    assert d["recommended_drop"] is None
    # cost and gain were retired in v3.
    assert "cost" not in d
    assert "gain" not in d
```

Replace `test_open_slot_stash_value_equals_gain_and_no_drop` (lines 317-333) with:

```python
def test_open_slot_positive_value_and_no_drop(stash_fixture):
    roster, standings, sds, slots, team = stash_fixture  # empty IL
    elite_fa = _make_elite_low_ip_pitcher()
    result = score_stash_candidates(
        roster=roster,
        free_agents=[elite_fa],
        projected_standings=standings,
        roster_slots=slots,
        team_name=team,
        team_sds=sds,
        fraction_remaining=0.5,
    )
    top = result.candidates[0]
    assert top.name == elite_fa.name
    assert top.stash_value > 0.0
    assert top.stash_value == top.band["mean"]  # Value is the band mean
    assert top.recommended_drop is None  # open slot -> nothing to drop
```

Replace `test_il_full_upgrade_recommends_dropping_weakest_stash` (lines 336-352) with:

```python
def test_il_full_upgrade_recommends_dropping_weakest_stash(stash_fixture_il_full):
    # roster has 2 owned IL stashes; a better FA exists and the IL is full.
    roster, standings, sds, slots, team, _weak_stash_name = stash_fixture_il_full
    better_fa = _make_elite_low_ip_pitcher()
    result = score_stash_candidates(
        roster=roster,
        free_agents=[better_fa],
        projected_standings=standings,
        roster_slots=slots,
        team_name=team,
        team_sds=sds,
        fraction_remaining=0.5,
    )
    fa_row = next(c for c in result.candidates if c.name == better_fa.name)
    owned_rows = [c for c in result.candidates if c.owned]
    assert owned_rows, "fixture should have owned IL stashes"
    # The drop is the owned stash at the BOTTOM of the board ranking
    # (lowest P(helps), value breaking ties) -- consistent with how the
    # board sorts. Derived from the result so it does not hinge on fixture
    # tuning of which arm is weaker.
    expected_drop = min(
        owned_rows, key=lambda c: (c.band["p_positive"], c.stash_value)
    ).name
    assert fa_row.recommended_drop == expected_drop
    assert fa_row.stash_value == fa_row.band["mean"]
```

Replace `test_owned_candidates_have_zero_cost` (lines 355-376) with:

```python
def test_owned_candidates_have_no_recommended_drop(stash_fixture_il_full):
    """An owned player already holds his IL slot, so there is nothing to drop
    to keep him and no cost to charge: his Value is just the band mean and his
    recommended_drop is None. (v3: cost retired entirely.)"""
    roster, standings, sds, slots, team, _weak = stash_fixture_il_full
    result = score_stash_candidates(
        roster=roster,
        free_agents=[],
        projected_standings=standings,
        roster_slots=slots,
        team_name=team,
        team_sds=sds,
        fraction_remaining=0.5,
    )
    owned = [c for c in result.candidates if c.owned]
    assert owned, "expected owned IL candidates in the fixture"
    for c in owned:
        assert c.stash_value == c.band["mean"]
        assert c.recommended_drop is None
```

In `test_owned_and_fa_player_get_equal_gain` (lines 406-454), change the final two assertions (lines 453-454) from:

```python
    assert fa_row.gain > 0.3  # elite closer into a contested SV cat is worth real points
    assert owned_row.gain == pytest.approx(fa_row.gain, abs=0.05)
```

to:

```python
    assert fa_row.stash_value > 0.3  # elite closer into a contested SV cat is worth real points
    assert owned_row.stash_value == pytest.approx(fa_row.stash_value, abs=0.05)
    # Equal Value AND equal P(helps): same band whether owned or FA.
    assert owned_row.band["p_positive"] == pytest.approx(fa_row.band["p_positive"], abs=0.02)
```

In `test_injured_fa_hitter_rate_upgrade_beats_volume` (lines 514-533), change the two assertions (lines 532-533) from:

```python
    assert row.gain > 0.0
    assert row.band["sd"] > 0.0  # the lineup actually changed -- a real swap
```

to:

```python
    assert row.stash_value > 0.0
    assert row.band["sd"] > 0.0  # the lineup actually changed -- a real swap
```

In `test_injured_fa_closer_rate_upgrade` (lines 536-554), change the final assertion (line 554) from:

```python
    assert row.gain > 0.0
```

to:

```python
    assert row.stash_value > 0.0
```

In `test_uncontested_fa_hitter_upgrade_is_zero` (lines 557-574), change the final assertion (line 574) from:

```python
    assert row.gain == 0.0
```

to:

```python
    assert row.stash_value == 0.0
```

Replace `test_fa_cost_floored_at_zero` (lines 577-590) with two tests for the new helpers:

```python
def test_weakest_owned_drop_picks_lowest_p_helps():
    """The FA drop target is the owned stash at the BOTTOM of the board
    ranking -- lowest P(helps), not lowest Value. Here the lower-P stash has
    the HIGHER value, so picking it proves the drop tracks P(helps)."""
    low_p = _arm("Low P", ip=40.0, k=20.0, slot="IL", status="IL15")
    high_p = _arm("High P", ip=40.0, k=20.0, slot="IL", status="IL15")
    bands = {
        "Low P": {"mean": 3.0, "sd": 9.0, "p_positive": 0.55, "verdict": "lean"},
        "High P": {"mean": 1.0, "sd": 0.5, "p_positive": 0.95, "verdict": "real"},
    }
    drop = _weakest_owned_drop(
        [low_p, high_p],
        bands,
        roster=[low_p, high_p],
        roster_slots={"IL": 1, "P": 9, "BN": 1},  # 2 stashes, 1 slot -> IL full
    )
    assert drop == "Low P"


def test_weakest_owned_drop_none_when_slot_open():
    """An open IL slot means the FA just slots in -- no drop suggested."""
    stash = _arm("Owned Stash", ip=40.0, k=20.0, slot="IL", status="IL15")
    bands = {"Owned Stash": {"mean": 1.0, "sd": 0.5, "p_positive": 0.9, "verdict": "real"}}
    drop = _weakest_owned_drop(
        [stash],
        bands,
        roster=[stash],
        roster_slots={"IL": 2, "P": 9, "BN": 1},  # 1 stash, 2 slots -> slot open
    )
    assert drop is None


def test_rank_key_orders_by_p_helps_then_value():
    """Board sorts by P(helps) first: a lower-Value, higher-P(helps) candidate
    ranks ABOVE a higher-Value, lower-P(helps) one (NOT the same order as
    sorting by Value -- see the v3 design doc)."""
    big_shaky = StashScore(
        name="Big Shaky",
        player_type="pitcher",
        status="IL15",
        owned=False,
        stash_value=2.0,
        band={"mean": 2.0, "sd": 4.0, "p_positive": 0.69, "verdict": "lean"},
        recommended_drop=None,
    )
    small_sure = StashScore(
        name="Small Sure",
        player_type="pitcher",
        status="IL15",
        owned=False,
        stash_value=1.0,
        band={"mean": 1.0, "sd": 0.5, "p_positive": 0.98, "verdict": "real"},
        recommended_drop=None,
    )
    ranked = sorted([big_shaky, small_sure], key=_rank_key, reverse=True)
    assert [s.name for s in ranked] == ["Small Sure", "Big Shaky"]


def test_rank_key_breaks_ties_by_value():
    """Equal P(helps) -> higher Value ranks first (deterministic tie-break)."""
    low_val = StashScore(
        name="Low Val",
        player_type="pitcher",
        status="IL15",
        owned=False,
        stash_value=1.0,
        band={"mean": 1.0, "sd": 0.5, "p_positive": 0.90, "verdict": "real"},
        recommended_drop=None,
    )
    high_val = StashScore(
        name="High Val",
        player_type="pitcher",
        status="IL15",
        owned=False,
        stash_value=2.0,
        band={"mean": 2.0, "sd": 1.0, "p_positive": 0.90, "verdict": "real"},
        recommended_drop=None,
    )
    ranked = sorted([low_val, high_val], key=_rank_key, reverse=True)
    assert [s.name for s in ranked] == ["High Val", "Low Val"]
```

- [ ] **Step 2: Run the test module to confirm it is red**

Run: `pytest tests/test_lineup/test_stash_value.py -q`
Expected: collection/import error -- `cannot import name '_rank_key'` (and `_weakest_owned_drop`) -- so the module fails to load. That is the red state; proceed to implement.

- [ ] **Step 3: Slim the `StashScore` dataclass**

In `src/fantasy_baseball/lineup/stash_value.py`, replace the `StashScore` dataclass (lines 46-61) with:

```python
@dataclass
class StashScore:
    """One injured player's stash evaluation.

    ``stash_value`` is the expected roto-point gain (band mean, floored at ~0);
    P(helps) is ``band["p_positive"]``. The board is ranked by P(helps) (see
    :func:`_rank_key`). There is no slot cost -- scarcity is priced by the
    cutline, and ``recommended_drop`` names the owned stash a free agent bumps
    when the IL is full."""

    name: str
    player_type: str
    status: str  # IL10 / IL15 / IL60 / ...
    owned: bool  # already on the user's roster
    stash_value: float  # expected roto-point gain (band mean, floored at ~0)
    band: dict[str, Any]  # {mean, sd, p_positive, verdict}; P(helps) = p_positive
    recommended_drop: str | None  # owned stash to drop (None if free slot / owned)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
```

- [ ] **Step 4: Update the module docstring**

Replace lines 16-18 (currently the "Cost is the IL-slot allocation cost..." paragraph) with:

```python
The board is ranked by P(helps) -- the probability the candidate's best swap
improves the user's roto total -- with the expected Value shown alongside. There
is no slot "cost": scarcity is priced by the cutline (the top IL-capacity
candidates earn a slot), and ``recommended_drop`` names the owned stash a free
agent would bump when the IL is full.
```

- [ ] **Step 5: Add `_rank_key`**

Insert this function immediately before `_open_il_slots` (currently line 337):

```python
def _rank_key(score: StashScore) -> tuple[float, float]:
    """Board sort key: P(helps) first, then expected Value as a deterministic
    tie-break. Higher is better -- callers sort with ``reverse=True``.

    P(helps) (``band["p_positive"]``) is the probability the candidate's best
    swap improves the user's roto total. Ranking by it is risk-averse: a
    smaller-but-likelier upgrade outranks a larger-but-shakier one, and it is
    NOT the same order as ranking by Value (see the v3 design doc). Value breaks
    ties so equal-probability rows order deterministically."""
    return (score.band["p_positive"], score.stash_value)
```

- [ ] **Step 6: Replace `_cost_and_drop` with `_weakest_owned_drop`**

Replace the whole `_cost_and_drop` function (lines 349-377) with:

```python
def _weakest_owned_drop(
    owned_il: list[Player],
    bands: Mapping[str, Mapping[str, Any]],
    roster: list[Player],
    roster_slots: dict[str, int],
) -> str | None:
    """Owned stash a free agent would bump, or None when an IL slot is open or
    there are no owned stashes.

    With a free IL slot the FA just slots in -- no drop. When the IL is full we
    name the owned stash at the BOTTOM of the board ranking (min P(helps), Value
    breaking ties -- the same order as :func:`_rank_key`), so the suggestion is
    consistent with how the board sorts. Slot scarcity is priced by the cutline,
    not by a per-row cost (see the v3 design doc: a per-FA cost double-counted
    the cutline)."""
    if _open_il_slots(roster, roster_slots) > 0 or not owned_il:
        return None
    return min(
        owned_il,
        key=lambda p: (bands[p.name]["p_positive"], bands[p.name]["mean"]),
    ).name
```

- [ ] **Step 7: Rewrite `score_stash_candidates` pass-2 and the sort**

Replace the docstring + body of `score_stash_candidates` from the docstring through the `return` (currently lines 391-466) with:

```python
    """Rank injured players (owned IL + injured FAs) by P(helps).

    Each candidate's Value is its marginal active value (band mean, floored at
    ~0); P(helps) is ``band["p_positive"]``. The board sorts by P(helps) with
    Value breaking ties (:func:`_rank_key`); the top ``IL``-capacity candidates
    earn a slot. For a free agent, ``recommended_drop`` names the owned stash he
    would bump when the IL is full (the bottom of the same ranking). Slot
    scarcity is priced by the cutline, not by a per-row cost.
    """
    il_capacity = roster_slots.get("IL", 0)
    owned_il = _owned_il_stashes(roster)
    injured_fas = [fa for fa in free_agents if fa.is_on_il()]

    # No injured players -> nothing to rank; skip the optimizer baseline entirely.
    if not owned_il and not injured_fas:
        return StashResult(
            open_il_slots=_open_il_slots(roster, roster_slots),
            cutline_rank=il_capacity,
        )

    # before_active is identical for every candidate: the optimized lineup over
    # the counted (non-IL-slot) bodies, with NO candidate activated.
    before_active = _solve_active(
        _counted_pool(roster), roster_slots, projected_standings, team_name, team_sds
    )

    # Value + band for every candidate (owned + FA).
    candidates_in: list[tuple[Player, bool]] = [(p, True) for p in owned_il] + [
        (p, False) for p in injured_fas
    ]
    bands: dict[str, dict[str, Any]] = {}
    for player, _owned in candidates_in:
        bands[player.name] = _marginal_band(
            player,
            before_active=before_active,
            projected_standings=projected_standings,
            team_name=team_name,
            team_sds=team_sds,
            fraction_remaining=fraction_remaining,
        )

    # One drop target for every FA when the IL is full: the bottom-ranked owned
    # stash. None for owned players (they already hold a slot).
    fa_drop = _weakest_owned_drop(owned_il, bands, roster, roster_slots)

    scores: list[StashScore] = []
    for player, owned in candidates_in:
        band = bands[player.name]
        scores.append(
            StashScore(
                name=player.name,
                player_type=player.player_type.value,
                status=player.status,
                owned=owned,
                stash_value=round(band["mean"], 2),
                band=band,
                recommended_drop=None if owned else fa_drop,
            )
        )

    scores.sort(key=_rank_key, reverse=True)
    return StashResult(
        open_il_slots=_open_il_slots(roster, roster_slots),
        cutline_rank=il_capacity,
        candidates=scores[:max_candidates],
    )
```

- [ ] **Step 8: Run the test module to confirm it is green**

Run: `pytest tests/test_lineup/test_stash_value.py -q`
Expected: all tests pass. If `test_il_full_upgrade_recommends_dropping_weakest_stash` fails, it is a real signal (the drop no longer equals the bottom-ranked owned row) -- do NOT loosen it; investigate `_weakest_owned_drop` vs `_rank_key` for an ordering mismatch.

- [ ] **Step 9: Commit**

```bash
git add src/fantasy_baseball/lineup/stash_value.py tests/test_lineup/test_stash_value.py
git commit -m "feat(stash): retire IL-slot cost; rank board by P(helps)

Drop StashScore.gain/.cost; Value = band mean. Sort by _rank_key =
(P(helps), value). Replace _cost_and_drop with _weakest_owned_drop (the
owned stash at the bottom of the same ranking). Slot scarcity is now
priced only by the cutline. Tests rewritten to the new contract + sort
key pinned.

Co-Authored-By: Claude Opus 4.7 <noreply@anthropic.com>"
```

---

## Task 2: Update the dashboard template

**Files:**
- Modify: `src/fantasy_baseball/web/templates/season/stash.html`

- [ ] **Step 1: Replace the header row**

Replace the `<thead>` row block (lines 24-28) from:

```html
      <th>#</th><th>Player</th><th>Status</th><th>Owned</th>
      <th>Stash value</th><th>Gain</th><th>Cost</th>
      <th>P(helps)</th><th>Drop to add</th>
```

to:

```html
      <th>#</th><th>Player</th><th>Status</th><th>Owned</th>
      <th>Value</th><th>P(helps)</th><th>Drop to add</th>
```

- [ ] **Step 2: Replace the body cells**

Replace the per-row `<td>` block (lines 33-41) from:

```html
      <td>{{ loop.index }}</td>
      <td>{{ c.name }}</td>
      <td>{{ c.status }}</td>
      <td>{{ 'yes' if c.owned else '-' }}</td>
      <td>{{ '%.2f'|format(c.stash_value) }}</td>
      <td>{{ '%.2f'|format(c.gain) }}</td>
      <td>{{ '%.2f'|format(c.cost) }}</td>
      <td>{{ '%.0f'|format(c.band.p_positive * 100) }}%</td>
      <td>{{ c.recommended_drop or '-' }}</td>
```

to:

```html
      <td>{{ loop.index }}</td>
      <td>{{ c.name }}</td>
      <td>{{ c.status }}</td>
      <td>{{ 'yes' if c.owned else '-' }}</td>
      <td>{{ '%.2f'|format(c.stash_value) }}</td>
      <td>{{ '%.0f'|format(c.band.p_positive * 100) }}%</td>
      <td>{{ c.recommended_drop or '-' }}</td>
```

- [ ] **Step 3: Confirm no stale field references remain in the template**

Run: `grep -nE "c\.(gain|cost)" src/fantasy_baseball/web/templates/season/stash.html`
Expected: no matches (exit non-zero / empty). If anything prints, remove it.

- [ ] **Step 4: Commit**

```bash
git add src/fantasy_baseball/web/templates/season/stash.html
git commit -m "style(stash): show Value + P(helps); drop Cost and Gain columns

Co-Authored-By: Claude Opus 4.7 <noreply@anthropic.com>"
```

---

## Task 3: Repo-wide reference sweep + full verification

**Files:** none (verification only)

- [ ] **Step 1: Confirm nothing else reads the removed fields/helper**

Run:
```bash
grep -rnE "_cost_and_drop|\.gain\b|\.cost\b" src/fantasy_baseball/lineup/stash_value.py src/fantasy_baseball/web tests/test_lineup/test_stash_value.py
grep -rn "stash" src/fantasy_baseball/web/templates src/fantasy_baseball/web/static/*.js 2>/dev/null
```
Expected: no surviving `_cost_and_drop`, `.gain`, or `.cost` references tied to stash; the only stash hits in web are class names / labels, not removed fields. If a real consumer turns up, update it before continuing (per CLAUDE.md rule 10, check templates, static JS, and the cached-payload readers).

- [ ] **Step 2: Run the lineup test directory**

Run: `pytest tests/test_lineup -q`
Expected: all pass (catches any cross-test fallout beyond the stash module).

- [ ] **Step 3: Lint**

Run: `ruff check .`
Expected: zero violations. (Watch for unused-import `F401` if any helper import was left behind.)

- [ ] **Step 4: Format check**

Run: `ruff format --check .`
Expected: no drift. If it reports changes, run `ruff format .` and re-stage.

- [ ] **Step 5: Dead-code check**

Run: `vulture`
Expected: no NEW findings from this change. `_marginal_value` (test-only wrapper) is pre-existing -- acceptable. If a removed symbol is still referenced, vulture/ruff will have surfaced it earlier.

- [ ] **Step 6: Type check**

Run: `mypy`
Expected: passes. `stash_value.py` is in `[tool.mypy].files`, so this is required.

- [ ] **Step 7: Final commit if formatting changed**

Only if Step 4 modified files:
```bash
git add -A
git commit -m "chore(stash): ruff format

Co-Authored-By: Claude Opus 4.7 <noreply@anthropic.com>"
```

---

## Self-Review

**Spec coverage:**
- Retire `cost` -> Task 1 Steps 3, 6, 7 (field removed, helper replaced, pass-2 no longer subtracts cost).
- Remove `gain` field, Value = band mean -> Task 1 Step 3 + Step 7 (`stash_value=round(band["mean"], 2)`).
- Sort by P(helps), tie-break Value -> Task 1 Step 5 (`_rank_key`) + Step 7 (`scores.sort(key=_rank_key, reverse=True)`); pinned by `test_rank_key_orders_by_p_helps_then_value` and `test_rank_key_breaks_ties_by_value`.
- `recommended_drop` = bottom of the same ranking -> Task 1 Step 6 (`_weakest_owned_drop`), tested by `test_weakest_owned_drop_picks_lowest_p_helps` / `_none_when_slot_open`.
- Cutline unchanged -> Task 1 Step 7 keeps `cutline_rank=il_capacity`.
- UI Value + P(helps), drop Cost/Gain -> Task 2.
- Docstrings -> Task 1 Steps 3, 4, 7.
- Tests rewritten to new contract + equal-scale guard kept -> Task 1 Step 1 (incl. `test_owned_and_fa_player_get_equal_gain` updated to `stash_value` and P(helps)).
- mypy coverage -> Task 3 Step 6.

**Placeholder scan:** none -- every code/test/template block is concrete.

**Type consistency:** `_rank_key(StashScore) -> tuple[float, float]`; `_weakest_owned_drop(list[Player], Mapping[str, Mapping[str, Any]], list[Player], dict[str, int]) -> str | None`; `StashScore` fields used in tests (`stash_value`, `band`, `recommended_drop`, `owned`, `name`) match the dataclass. `bands` is `dict[str, dict[str, Any]]` and is accepted where `Mapping[str, Mapping[str, Any]]` is expected. `Mapping` is already imported in `stash_value.py` (line 23).
