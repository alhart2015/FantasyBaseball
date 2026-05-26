# IL Stash Board Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a leverage-aware "who to stash" ranking that scores injured players (owned IL + injured free agents) by their marginal active value (deltaRoto), nets the IL-slot allocation cost, and surfaces a ranked Stash Board on the season dashboard.

**Architecture:** A new pure-scoring module `lineup/stash_value.py`, built as a sibling of `lineup/il_return_planner.py`. It reuses the optimizer (`optimize_hitter_lineup`/`optimize_pitcher_lineup`) and the double-count-safe band (`compute_delta_roto_band(before_active, after_active, ...)`) to compute each candidate's marginal active value ("Gain"), then derives Cost from IL-slot availability (0 when a slot is open; the weakest displaced stash's Gain when full). The refresh pipeline computes it after the roster audit (where FAs + standings + sds are already in scope), caches it under `CacheKey.STASH`, and a new `/stash` route renders it.

**Tech Stack:** Python 3.12, pandas, Flask + Jinja2, pytest. KV cache via `web/season_data.py` (`write_cache`/`read_cache`). Existing primitives: `lineup/delta_roto.py`, `lineup/optimizer.py`, `scoring.py`, `models/standings.py`, `models/positions.py`.

---

## Key design facts (verified against the code)

- **Gain = marginal active value.** For a candidate, `before_active` = the team's optimized active lineup WITHOUT him; `after_active` = the optimized lineup WITH him activated (IL signals cleared). `compute_delta_roto_band(before_active, after_active, field_stats, team_name, fraction_remaining, projected_standings=..., team_sds=...)` returns the band. If the candidate can't crack the lineup, `after_active == before_active` and the band mean is ~0 -> **downside floored at zero** ("no harm, no foul"). This mirrors `il_return_planner._make_plan` exactly and is double-count-safe (the band applies the active-set delta to the standings row that already includes baseline players).
- **Cost = IL-slot allocation.** Open IL slot -> 0. IL slots full -> the candidate displaces the lowest-Gain *owned IL stash* (IL-for-IL, the user's rule); Cost = that stash's Gain, `recommended_drop` = that stash. A non-IL-eligible candidate (e.g. DTD) cannot take an IL slot: Cost = 0 if an active/bench body is open, else the lowest-Gain active/bench body's marginal value.
- **StashValue = Gain - Cost.** Ranked descending. The top `IL capacity` candidates are "hold/grab"; below the cutline is "not worth a slot."
- **Slot accounting is slot-based, not status-based** (reuse `roster_capacity`, `_counts_against_cap`, `IL_SLOTS`). A BN+IL-status player occupies a *body*, not an IL slot.
- **`compute_delta_roto` is one-for-one and raises if `drop_name` is absent — do NOT use it for the add-only Gain.** Use `compute_delta_roto_band` with explicit before/after lists.

## File structure

- Create: `src/fantasy_baseball/lineup/stash_value.py` — scoring module (data models + `score_stash_candidates`).
- Create: `tests/test_lineup/test_stash_value.py` — unit tests for the module.
- Modify: `src/fantasy_baseball/data/cache_keys.py` — add `STASH = "stash"`.
- Modify: `src/fantasy_baseball/web/refresh_pipeline.py` — compute + cache the stash payload in `_audit_roster` (inputs already in scope there).
- Modify: `src/fantasy_baseball/web/season_routes.py` — add `/stash` route.
- Create: `src/fantasy_baseball/web/templates/season/stash.html` — Stash Board page.
- Modify: `tests/test_web/test_season_routes.py` — route render test.
- Modify: `tests/test_web/test_refresh_pipeline.py` — assert `CacheKey.STASH` written.
- Modify: `pyproject.toml` — add `stash_value.py` to `[tool.mypy].files`.

---

### Task 1: Add the STASH cache key

**Files:**
- Modify: `src/fantasy_baseball/data/cache_keys.py`
- Test: `tests/test_data/test_cache_keys.py` (create if absent)

- [ ] **Step 1: Write the failing test**

```python
# tests/test_data/test_cache_keys.py
from fantasy_baseball.data.cache_keys import CacheKey, redis_key


def test_stash_key_exists_and_namespaced():
    assert CacheKey.STASH == "stash"
    assert redis_key(CacheKey.STASH) == "cache:stash"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_data/test_cache_keys.py -v`
Expected: FAIL with `AttributeError: STASH` (or ImportError if file is new).

- [ ] **Step 3: Add the enum member**

In `src/fantasy_baseball/data/cache_keys.py`, add to the `CacheKey` StrEnum (after `STREAK_SCORES = "streak_scores"`):

```python
    STASH = "stash"
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_data/test_cache_keys.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/fantasy_baseball/data/cache_keys.py tests/test_data/test_cache_keys.py
git commit -m "feat(cache): add STASH cache key for the stash board"
```

---

### Task 2: Stash data models

**Files:**
- Create: `src/fantasy_baseball/lineup/stash_value.py`
- Test: `tests/test_lineup/test_stash_value.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_lineup/test_stash_value.py
from fantasy_baseball.lineup.stash_value import StashScore, StashResult


def test_stash_score_to_dict_shape():
    s = StashScore(
        name="Blake Snell",
        player_type="pitcher",
        status="IL15",
        owned=False,
        gain=4.2,
        cost=0.0,
        stash_value=4.2,
        band={"mean": 4.2, "sd": 1.1, "p_positive": 0.91, "verdict": "real"},
        recommended_drop=None,
    )
    d = s.to_dict()
    assert d["name"] == "Blake Snell"
    assert d["stash_value"] == 4.2
    assert d["band"]["verdict"] == "real"
    assert d["recommended_drop"] is None


def test_stash_result_to_dict_shape():
    r = StashResult(open_il_slots=1, cutline_rank=2, candidates=[], warning=None)
    d = r.to_dict()
    assert d["open_il_slots"] == 1
    assert d["cutline_rank"] == 2
    assert d["candidates"] == []
    assert d["warning"] is None
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_lineup/test_stash_value.py -v`
Expected: FAIL with `ModuleNotFoundError: fantasy_baseball.lineup.stash_value`.

- [ ] **Step 3: Create the module with data models**

```python
# src/fantasy_baseball/lineup/stash_value.py
"""Stash board -- rank injured players (owned IL + injured FAs) by their
leverage-aware marginal active value, and allocate the scarce IL slots.

Sibling of ``il_return_planner``: reuses the optimizer and the
double-count-safe deltaRoto band. A candidate's Gain is the band mean of
activating him into the optimized lineup (floored at ~0 when he can't crack
it -- "no harm, no foul"). Cost is the IL-slot allocation cost: 0 when a slot
is open, else the Gain of the weakest owned IL stash he displaces.
"""

from __future__ import annotations

import dataclasses
from collections.abc import Mapping
from dataclasses import asdict, dataclass, field
from typing import Any

from fantasy_baseball.utils.constants import Category

__all__ = ["StashScore", "StashResult", "score_stash_candidates"]


@dataclass
class StashScore:
    """One injured player's stash evaluation."""

    name: str
    player_type: str
    status: str  # IL10 / IL15 / IL60 / DTD / ...
    owned: bool  # already on the user's roster
    gain: float  # marginal active value (deltaRoto band mean), floored at ~0
    cost: float  # deltaRoto sacrificed to roster him (0 if open IL slot)
    stash_value: float  # gain - cost
    band: dict[str, Any]  # {mean, sd, p_positive, verdict}
    recommended_drop: str | None  # who to drop to make room (None if free slot)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class StashResult:
    """Ranked stash board."""

    open_il_slots: int
    cutline_rank: int  # = IL capacity; top-N are "hold/grab"
    candidates: list[StashScore] = field(default_factory=list)
    warning: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "open_il_slots": self.open_il_slots,
            "cutline_rank": self.cutline_rank,
            "candidates": [c.to_dict() for c in self.candidates],
            "warning": self.warning,
        }
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_lineup/test_stash_value.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/fantasy_baseball/lineup/stash_value.py tests/test_lineup/test_stash_value.py
git commit -m "feat(stash): add StashScore/StashResult data models"
```

---

### Task 3: Marginal active value (Gain) helper

This is the core. It mirrors `il_return_planner._solve_lineup` + the band call in `_make_plan`. `before_active` is computed once by the caller and passed in; this helper optimizes the lineup WITH the candidate activated and bands the difference.

**Files:**
- Modify: `src/fantasy_baseball/lineup/stash_value.py`
- Test: `tests/test_lineup/test_stash_value.py`

- [ ] **Step 1: Write the failing test**

Use a synthetic roster where one elite low-volume pitcher should crack the lineup (positive Gain) and one weak pitcher should not (Gain ~0). Reuse the fixture style in `tests/test_lineup/test_il_return_planner.py` (read it for the exact `Player`/`PitcherStats` construction and a ready `ProjectedStandings`/`team_sds`).

```python
# tests/test_lineup/test_stash_value.py  (add)
from fantasy_baseball.lineup.stash_value import _marginal_value, _activate, _solve_active
from fantasy_baseball.models.player import Player, PlayerType


def _band_mean(roster, candidate, standings, team_name, slots, sds):
    before = _solve_active(roster, slots, standings, team_name, sds)
    return _marginal_value(
        candidate,
        before_active=before,
        roster=roster,
        roster_slots=slots,
        projected_standings=standings,
        team_name=team_name,
        team_sds=sds,
        fraction_remaining=0.5,
    )


def test_elite_low_volume_arm_has_positive_gain(stash_fixture):
    # stash_fixture: (roster, standings, team_sds, roster_slots, team_name)
    roster, standings, sds, slots, team = stash_fixture
    elite = _make_elite_low_ip_pitcher()  # ~90 IP, sub-3 ERA, high K/9
    gain = _band_mean(roster, elite, standings, team, slots, sds)
    assert gain > 0.0


def test_scrub_arm_gain_is_floored_at_zero(stash_fixture):
    roster, standings, sds, slots, team = stash_fixture
    scrub = _make_replacement_level_pitcher()  # worse than every active arm
    gain = _band_mean(roster, scrub, standings, team, slots, sds)
    assert gain == 0.0  # cannot crack the lineup -> before == after -> mean 0
```

Add a `stash_fixture` pytest fixture and the two `_make_*` helpers at the top of the test module, copying the roster/standings construction from `tests/test_lineup/test_il_return_planner.py`. (Read that file first; reuse its `Player`/stats builders verbatim so the projection fields line up.)

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_lineup/test_stash_value.py -k gain -v`
Expected: FAIL with `ImportError: cannot import name '_marginal_value'`.

- [ ] **Step 3: Implement `_activate`, `_solve_active`, `_marginal_value`**

Add to `stash_value.py` (imports at top of file):

```python
from fantasy_baseball.lineup.delta_roto import compute_delta_roto_band
from fantasy_baseball.lineup.optimizer import (
    optimize_hitter_lineup,
    optimize_pitcher_lineup,
)
from fantasy_baseball.models.player import Player, PlayerType
from fantasy_baseball.models.positions import IL_SLOTS
from fantasy_baseball.models.standings import ProjectedStandings
```

```python
def _activate(p: Player) -> Player:
    """Copy with IL signals cleared so the optimizer treats the player as
    active-eligible (valuing his production WHEN healthy). Identical to
    ``il_return_planner._activate``."""
    return dataclasses.replace(p, status="", selected_position=None)


def _solve_active(
    pool: list[Player],
    roster_slots: dict[str, int],
    projected_standings: ProjectedStandings,
    team_name: str,
    team_sds: Mapping[str, Mapping[Category, float]] | None,
) -> list[Player]:
    """Optimized active lineup (hitters + pitcher starters) over ``pool``.

    ``fraction_remaining=None`` skips per-starter band computation, matching
    ``il_return_planner._solve_lineup``."""
    hitters = [p for p in pool if p.player_type != PlayerType.PITCHER]
    pitchers = [p for p in pool if p.player_type == PlayerType.PITCHER]
    h_assign = optimize_hitter_lineup(
        hitters=hitters,
        full_roster=pool,
        projected_standings=projected_standings,
        team_name=team_name,
        roster_slots=roster_slots,
        team_sds=team_sds,
        fraction_remaining=None,
    )
    p_starters, _bench = optimize_pitcher_lineup(
        pitchers=pitchers,
        full_roster=pool,
        projected_standings=projected_standings,
        team_name=team_name,
        slots=roster_slots.get("P", 9),
        team_sds=team_sds,
        fraction_remaining=None,
    )
    return [a.player for a in h_assign] + [s.player for s in p_starters]


def _counted_pool(roster: list[Player], exclude_name: str | None = None) -> list[Player]:
    """Active + bench bodies (everything not in a true IL slot), optionally
    excluding one player by name. Owned IL-slotted players are left out -- we
    add the candidate back explicitly when valuing him."""
    out: list[Player] = []
    for p in roster:
        if p.selected_position in IL_SLOTS:
            continue
        if exclude_name is not None and p.name == exclude_name:
            continue
        out.append(p)
    return out


def _marginal_value(
    candidate: Player,
    *,
    before_active: list[Player],
    roster: list[Player],
    roster_slots: dict[str, int],
    projected_standings: ProjectedStandings,
    team_name: str,
    team_sds: Mapping[str, Mapping[Category, float]] | None,
    fraction_remaining: float,
) -> float:
    """Gain = band mean of activating ``candidate`` into the optimized lineup.

    Floored at ~0: if the candidate can't crack the lineup, ``after_active``
    equals ``before_active`` and the band mean is ~0. Double-count-safe via the
    same before/after-active mechanism as ``il_return_planner``."""
    band = _marginal_band(
        candidate,
        before_active=before_active,
        roster=roster,
        roster_slots=roster_slots,
        projected_standings=projected_standings,
        team_name=team_name,
        team_sds=team_sds,
        fraction_remaining=fraction_remaining,
    )
    return band["mean"]


def _marginal_band(
    candidate: Player,
    *,
    before_active: list[Player],
    roster: list[Player],
    roster_slots: dict[str, int],
    projected_standings: ProjectedStandings,
    team_name: str,
    team_sds: Mapping[str, Mapping[Category, float]] | None,
    fraction_remaining: float,
) -> dict[str, Any]:
    """Return the deltaRoto band dict for activating ``candidate``."""
    pool_with = _counted_pool(roster, exclude_name=candidate.name) + [_activate(candidate)]
    after_active = _solve_active(
        pool_with, roster_slots, projected_standings, team_name, team_sds
    )
    band = compute_delta_roto_band(
        before_active,
        after_active,
        projected_standings.field_stats(team_name),
        team_name,
        fraction_remaining,
        projected_standings=projected_standings,
        team_sds=team_sds,
    )
    return band.to_dict()
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_lineup/test_stash_value.py -k gain -v`
Expected: PASS (both: elite arm gain > 0; scrub gain == 0).

- [ ] **Step 5: Commit**

```bash
git add src/fantasy_baseball/lineup/stash_value.py tests/test_lineup/test_stash_value.py
git commit -m "feat(stash): marginal active value (Gain) via deltaRoto band, floored at zero"
```

---

### Task 4: IL-slot accounting and Cost

**Files:**
- Modify: `src/fantasy_baseball/lineup/stash_value.py`
- Test: `tests/test_lineup/test_stash_value.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_lineup/test_stash_value.py  (add)
from fantasy_baseball.lineup.stash_value import _open_il_slots, _owned_il_stashes


def test_open_il_slots_counts_true_il_slots_only(stash_fixture):
    roster, *_ = stash_fixture
    # With an empty IL, both IL slots are open.
    assert _open_il_slots(roster, {"IL": 2}) == 2


def test_owned_il_stashes_uses_is_on_il(monkeypatched_il_roster):
    roster = monkeypatched_il_roster  # one player with status="IL15"
    names = {p.name for p in _owned_il_stashes(roster)}
    assert "Injured Owned Arm" in names
```

(Add a `monkeypatched_il_roster` fixture: a roster where one pitcher has `status="IL15"`.)

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_lineup/test_stash_value.py -k "il_slots or owned_il" -v`
Expected: FAIL with `ImportError`.

- [ ] **Step 3: Implement slot accounting + Cost**

```python
def _open_il_slots(roster: list[Player], roster_slots: dict[str, int]) -> int:
    """IL capacity minus players currently in true IL slots."""
    capacity = roster_slots.get("IL", 0)
    occupied = sum(1 for p in roster if p.selected_position in IL_SLOTS)
    return max(0, capacity - occupied)


def _owned_il_stashes(roster: list[Player]) -> list[Player]:
    """Owned players on the IL (slot or status)."""
    return [p for p in roster if p.is_on_il()]


def _cost_and_drop(
    candidate: Player,
    *,
    gain_by_name: dict[str, float],
    roster: list[Player],
    roster_slots: dict[str, int],
) -> tuple[float, str | None]:
    """Cost to roster ``candidate`` and the recommended drop.

    - IL-eligible + open IL slot -> (0, None).
    - IL-eligible + IL full -> displace the lowest-Gain owned IL stash
      (IL-for-IL, the user's rule). Cost = that stash's Gain.
    - Not IL-eligible (e.g. DTD) -> cannot use an IL slot; if an active/bench
      body is open, (0, None), else displace the lowest-Gain active/bench body.
    """
    il_eligible = candidate.is_on_il()
    if il_eligible and _open_il_slots(roster, roster_slots) > 0:
        return 0.0, None

    if il_eligible:
        # Displace the weakest owned IL stash (exclude the candidate itself).
        pool = [p for p in _owned_il_stashes(roster) if p.name != candidate.name]
    else:
        from fantasy_baseball.lineup.il_return_planner import roster_capacity

        counted = _counted_pool(roster, exclude_name=candidate.name)
        if len(counted) < roster_capacity(roster_slots):
            return 0.0, None
        pool = counted

    if not pool:
        return 0.0, None
    drop = min(pool, key=lambda p: gain_by_name.get(p.name, 0.0))
    return gain_by_name.get(drop.name, 0.0), drop.name
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_lineup/test_stash_value.py -k "il_slots or owned_il" -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/fantasy_baseball/lineup/stash_value.py tests/test_lineup/test_stash_value.py
git commit -m "feat(stash): IL-slot accounting and allocation cost (IL-for-IL drop)"
```

---

### Task 5: `score_stash_candidates` orchestration

**Files:**
- Modify: `src/fantasy_baseball/lineup/stash_value.py`
- Test: `tests/test_lineup/test_stash_value.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_lineup/test_stash_value.py  (add)
from fantasy_baseball.lineup.stash_value import score_stash_candidates


def test_open_slot_stash_value_equals_gain_and_no_drop(stash_fixture):
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
    assert top.cost == 0.0
    assert top.stash_value == top.gain > 0.0
    assert top.recommended_drop is None


def test_il_full_upgrade_recommends_dropping_weakest_stash(stash_fixture_il_full):
    # roster has 2 IL stashes: a strong one and a weak one; a better FA exists.
    roster, standings, sds, slots, team, weak_stash_name = stash_fixture_il_full
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
    assert fa_row.recommended_drop == weak_stash_name
    assert fa_row.cost > 0.0
    assert fa_row.stash_value == fa_row.gain - fa_row.cost
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_lineup/test_stash_value.py -k "open_slot or il_full" -v`
Expected: FAIL with `ImportError: cannot import name 'score_stash_candidates'`.

- [ ] **Step 3: Implement the orchestrator**

```python
def score_stash_candidates(
    roster: list[Player],
    free_agents: list[Player],
    projected_standings: ProjectedStandings,
    roster_slots: dict[str, int],
    team_name: str,
    *,
    team_sds: Mapping[str, Mapping[Category, float]] | None,
    fraction_remaining: float,
    max_candidates: int = 25,
) -> StashResult:
    """Rank injured players (owned IL + injured FAs) by stash value.

    Gain = marginal active value (band mean, floored at ~0). Cost = IL-slot
    allocation cost. stash_value = gain - cost, ranked descending. The top
    ``IL`` -capacity candidates are worth a slot.
    """
    il_capacity = roster_slots.get("IL", 0)
    owned_il = _owned_il_stashes(roster)
    injured_fas = [fa for fa in free_agents if fa.is_on_il()]

    # before_active is identical for every candidate: the optimized lineup over
    # the counted (non-IL-slot) bodies, with NO candidate activated.
    before_active = _solve_active(
        _counted_pool(roster), roster_slots, projected_standings, team_name, team_sds
    )

    # Pass 1: Gain + band for every candidate (owned + FA) and every owned IL
    # stash (needed as Cost drop targets).
    bands: dict[str, dict[str, Any]] = {}
    candidates_in: list[tuple[Player, bool]] = [(p, True) for p in owned_il] + [
        (p, False) for p in injured_fas
    ]
    for player, _owned in candidates_in:
        bands[player.name] = _marginal_band(
            player,
            before_active=before_active,
            roster=roster,
            roster_slots=roster_slots,
            projected_standings=projected_standings,
            team_name=team_name,
            team_sds=team_sds,
            fraction_remaining=fraction_remaining,
        )
    gain_by_name = {name: b["mean"] for name, b in bands.items()}

    # Pass 2: Cost + stash value.
    scores: list[StashScore] = []
    for player, owned in candidates_in:
        band = bands[player.name]
        gain = band["mean"]
        cost, drop = _cost_and_drop(
            player,
            gain_by_name=gain_by_name,
            roster=roster,
            roster_slots=roster_slots,
        )
        scores.append(
            StashScore(
                name=player.name,
                player_type=player.player_type.value,
                status=player.status,
                owned=owned,
                gain=round(gain, 2),
                cost=round(cost, 2),
                stash_value=round(gain - cost, 2),
                band=band,
                recommended_drop=drop,
            )
        )

    scores.sort(key=lambda s: s.stash_value, reverse=True)
    return StashResult(
        open_il_slots=_open_il_slots(roster, roster_slots),
        cutline_rank=il_capacity,
        candidates=scores[:max_candidates],
    )
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_lineup/test_stash_value.py -v`
Expected: PASS (all tasks 2-5 tests).

- [ ] **Step 5: Commit**

```bash
git add src/fantasy_baseball/lineup/stash_value.py tests/test_lineup/test_stash_value.py
git commit -m "feat(stash): score_stash_candidates orchestration (rank + allocate IL slots)"
```

---

### Task 6: Wire into the refresh pipeline and cache

**Files:**
- Modify: `src/fantasy_baseball/web/refresh_pipeline.py` (inside `_audit_roster`, ~lines 955-982, after `write_cache(CacheKey.ROSTER_AUDIT, ...)`)
- Test: `tests/test_web/test_refresh_pipeline.py`

- [ ] **Step 1: Write the failing test**

Find `test_all_expected_cache_files_written` in `tests/test_web/test_refresh_pipeline.py` (~line 45). Add `CacheKey.STASH` to its expected-keys assertion. If the test enumerates keys in a set/list, add `"stash"` (or `CacheKey.STASH`) there.

```python
# tests/test_web/test_refresh_pipeline.py  (edit the expected set)
expected_keys = {
    # ... existing keys ...
    CacheKey.STASH,
}
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_web/test_refresh_pipeline.py::test_all_expected_cache_files_written -v`
Expected: FAIL — `cache:stash` not written.

- [ ] **Step 3: Compute + cache the stash payload**

In `refresh_pipeline.py`, at the end of `_audit_roster` (immediately after the `write_cache(CacheKey.ROSTER_AUDIT, ...)` line), add:

```python
        from fantasy_baseball.lineup.stash_value import score_stash_candidates

        stash_result = score_stash_candidates(
            self.roster_players,
            self.fa_players,
            self.projected_standings,
            self.config.roster_slots,
            self.config.team_name,
            team_sds=self.team_sds,
            fraction_remaining=self.fraction_remaining,
        )
        write_cache(CacheKey.STASH, stash_result.to_dict())
        self._progress(
            f"Stash board: {len(stash_result.candidates)} injured candidate(s)"
        )
```

(`CacheKey` and `write_cache` are already imported in this module — confirm at the top; if not, add `from fantasy_baseball.web.season_data import write_cache` / `from fantasy_baseball.data.cache_keys import CacheKey`.)

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_web/test_refresh_pipeline.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/fantasy_baseball/web/refresh_pipeline.py tests/test_web/test_refresh_pipeline.py
git commit -m "feat(stash): compute and cache the stash board during refresh"
```

---

### Task 7: `/stash` route + template

**Files:**
- Modify: `src/fantasy_baseball/web/season_routes.py` (mirror the `/roster-audit` route, ~lines 530-541)
- Create: `src/fantasy_baseball/web/templates/season/stash.html`
- Test: `tests/test_web/test_season_routes.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_web/test_season_routes.py  (add)
def test_stash_route_renders_ranked_board(client, kv_isolation):
    from fantasy_baseball.web import season_data
    from fantasy_baseball.data.cache_keys import CacheKey

    payload = {
        "open_il_slots": 1,
        "cutline_rank": 2,
        "candidates": [
            {
                "name": "Blake Snell",
                "player_type": "pitcher",
                "status": "IL15",
                "owned": False,
                "gain": 4.2,
                "cost": 0.0,
                "stash_value": 4.2,
                "band": {"mean": 4.2, "sd": 1.1, "p_positive": 0.91, "verdict": "real"},
                "recommended_drop": None,
            }
        ],
        "warning": None,
    }
    season_data.write_cache(CacheKey.STASH, payload)
    season_data.write_cache(CacheKey.META, {"last_refresh": "9:00 AM"})

    resp = client.get("/stash")
    assert resp.status_code == 200
    html = resp.data.decode()
    assert "Blake Snell" in html
    assert "Grab &amp; Stash" in html or "Grab & Stash" in html
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_web/test_season_routes.py::test_stash_route_renders_ranked_board -v`
Expected: FAIL with 404 (no `/stash` route).

- [ ] **Step 3: Add the route + template**

In `season_routes.py` (mirror `/roster-audit`):

```python
@app.route("/stash")
def stash():
    meta = read_meta()
    payload = read_cache_dict(CacheKey.STASH) or {
        "open_il_slots": 0,
        "cutline_rank": 0,
        "candidates": [],
        "warning": None,
    }
    return render_template(
        "season/stash.html",
        meta=meta,
        active_page="stash",
        stash=payload,
    )
```

(Confirm `read_cache_dict` is imported in `season_routes.py`; it lives in `season_data.py`. If `/roster-audit` uses `read_cache_list`, add `read_cache_dict` to the existing import line.)

Create `src/fantasy_baseball/web/templates/season/stash.html` extending the dashboard base (copy the `{% extends %}`/header block from `season/roster_audit.html`). ASCII-only labels:

```html
{% extends "season/base.html" %}
{% block content %}
<h2>Grab & Stash</h2>
<p class="muted">
  Who deserves your {{ stash.cutline_rank }} IL slot(s).
  Open IL slots: {{ stash.open_il_slots }}.
</p>
{% if not stash.candidates %}
  <p>No injured candidates right now.</p>
{% else %}
<table class="stash-table">
  <thead>
    <tr>
      <th>#</th><th>Player</th><th>Status</th><th>Owned</th>
      <th>Stash value</th><th>Gain</th><th>Cost</th>
      <th>P(helps)</th><th>Drop to add</th>
    </tr>
  </thead>
  <tbody>
    {% for c in stash.candidates %}
    <tr class="{{ 'above-cutline' if loop.index <= stash.cutline_rank else 'below-cutline' }}">
      <td>{{ loop.index }}</td>
      <td>{{ c.name }}</td>
      <td>{{ c.status }}</td>
      <td>{{ 'yes' if c.owned else '-' }}</td>
      <td>{{ '%.2f'|format(c.stash_value) }}</td>
      <td>{{ '%.2f'|format(c.gain) }}</td>
      <td>{{ '%.2f'|format(c.cost) }}</td>
      <td>{{ '%.0f'|format(c.band.p_positive * 100) }}%</td>
      <td>{{ c.recommended_drop or '-' }}</td>
    </tr>
    {% endfor %}
  </tbody>
</table>
{% endif %}
{% endblock %}
```

(If `season/base.html` is not the base template name, open `season/roster_audit.html` and copy its exact `{% extends %}`/block names.)

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_web/test_season_routes.py::test_stash_route_renders_ranked_board -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/fantasy_baseball/web/season_routes.py src/fantasy_baseball/web/templates/season/stash.html tests/test_web/test_season_routes.py
git commit -m "feat(stash): /stash route and Stash Board template"
```

---

### Task 8: Owned-IL hold/drop hint + nav link

**Files:**
- Modify: `src/fantasy_baseball/web/templates/season/stash.html`
- Modify: the dashboard nav partial (find it: `grep -rl "roster-audit" src/fantasy_baseball/web/templates/`)
- Test: `tests/test_web/test_season_routes.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_web/test_season_routes.py  (add)
def test_stash_below_cutline_owned_flagged_droppable(client, kv_isolation):
    from fantasy_baseball.web import season_data
    from fantasy_baseball.data.cache_keys import CacheKey

    payload = {
        "open_il_slots": 0,
        "cutline_rank": 1,
        "candidates": [
            {"name": "Better FA", "player_type": "pitcher", "status": "IL15",
             "owned": False, "gain": 5.0, "cost": 1.0, "stash_value": 4.0,
             "band": {"mean": 4.0, "sd": 1.0, "p_positive": 0.9, "verdict": "real"},
             "recommended_drop": "Weak Owned Stash"},
            {"name": "Weak Owned Stash", "player_type": "pitcher", "status": "IL60",
             "owned": True, "gain": 1.0, "cost": 0.0, "stash_value": 1.0,
             "band": {"mean": 1.0, "sd": 0.8, "p_positive": 0.6, "verdict": "lean"},
             "recommended_drop": None},
        ],
        "warning": None,
    }
    season_data.write_cache(CacheKey.STASH, payload)
    season_data.write_cache(CacheKey.META, {"last_refresh": "9:00 AM"})
    html = client.get("/stash").data.decode()
    assert "below-cutline" in html  # the weak owned stash is below the cutline
    assert "Weak Owned Stash" in html
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_web/test_season_routes.py::test_stash_below_cutline_owned_flagged_droppable -v`
Expected: FAIL if the `below-cutline` class or row is missing (it should already pass from Task 7's template if the class is present; if so, this is a regression guard — confirm it passes and move on).

- [ ] **Step 3: Add nav link**

In the nav partial returned by the grep above, add a link next to the roster-audit link:

```html
<a href="/stash" class="{{ 'active' if active_page == 'stash' }}">Grab &amp; Stash</a>
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_web/test_season_routes.py -k stash -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/fantasy_baseball/web/templates/ tests/test_web/test_season_routes.py
git commit -m "feat(stash): nav link and below-cutline drop hint"
```

---

### Task 9: Final verification (lint, types, full suite)

**Files:**
- Modify: `pyproject.toml` (`[tool.mypy].files`)

- [ ] **Step 1: Add the module to mypy coverage**

In `pyproject.toml`, under `[tool.mypy]` `files = [...]`, add:

```toml
    "src/fantasy_baseball/lineup/stash_value.py",
```

- [ ] **Step 2: Run mypy**

Run: `mypy`
Expected: no errors in `stash_value.py` (fix any). Resolve `Mapping`/`Category` typing as needed.

- [ ] **Step 3: Run lint + format**

Run: `ruff check . && ruff format --check .`
Expected: zero violations, no formatting drift (run `ruff format .` if needed).

- [ ] **Step 4: Run the targeted + full suite**

Run: `pytest tests/test_lineup/test_stash_value.py tests/test_web/test_season_routes.py tests/test_web/test_refresh_pipeline.py -v`
Then: `pytest -q` (full suite; or `pytest -n auto`).
Expected: all PASS. Confirm no `vulture` regressions: `vulture` (pre-existing findings acceptable; none new from `stash_value.py`).

- [ ] **Step 5: Commit**

```bash
git add pyproject.toml
git commit -m "chore(stash): add stash_value.py to mypy files; verify suite green"
```

---

## Self-review

**Spec coverage:**
- Grab ranking (injured FAs) -> Task 5 (`injured_fas`) + Task 7 (board). Covered.
- Hold-vs-drop for owned IL -> Task 5 (`owned_il` candidates, ranked together) + Task 8 (below-cutline hint). Covered.
- Leverage-aware deltaRoto value -> Task 3 (`compute_delta_roto_band`). Covered.
- Downside floored at zero -> Task 3 (scrub-arm test asserts `gain == 0.0`). Covered.
- IL-slot allocation cost / IL-for-IL drop -> Task 4 (`_cost_and_drop`) + Task 5 (il-full test). Covered.
- DTD never consumes a free IL slot -> Task 4 (`il_eligible` branch). Add an explicit test in Task 4 if time: a DTD candidate with full active+bench yields a non-IL drop.
- Double-count safety -> Task 3 uses the il_return_planner before/after-active mechanism; verified pattern.
- Surfaces (board + owned column + nav) -> Tasks 7-8. Covered.
- Not a standings-impact widget -> the route renders a ranking only. Covered.

**Placeholder scan:** No TODO/TBD; every code step has concrete code. The two `_make_*` test helpers and fixtures in Task 3 are specified to be copied from `tests/test_lineup/test_il_return_planner.py` — the executor MUST read that file first (noted in the step) to reuse its exact `Player`/stats builders. This is the one spot requiring the executor to lift existing fixture code rather than invent it.

**Type consistency:** `score_stash_candidates` signature matches the spec. `StashScore`/`StashResult` fields are consistent across Tasks 2, 5, 7, 8. `_marginal_value`/`_marginal_band`/`_solve_active`/`_activate`/`_counted_pool`/`_open_il_slots`/`_owned_il_stashes`/`_cost_and_drop` names are used consistently. `compute_delta_roto_band` argument order matches the verified signature.

## Risks / notes for the executor

1. **Band reconciliation is the #1 risk.** Task 3's `_marginal_band` must mirror `il_return_planner._make_plan` exactly (before_active = optimized lineup without candidate; after_active = optimized lineup with candidate activated). Read `il_return_planner.py:225-339` before implementing. If the elite-arm Gain test comes back ~0 unexpectedly, the candidate isn't being added to the optimizer pool — check `_activate` clears both `status` and `selected_position`.
2. **Fixtures:** Tasks 3-5 reuse the roster/standings/team_sds construction from `tests/test_lineup/test_il_return_planner.py`. Read it first; do not invent projection fields.
3. **Optimizer return fields:** `optimize_hitter_lineup` returns assignments with `.player`; `optimize_pitcher_lineup` returns `(starters, bench)` where starters have `.player`. Confirmed against `il_return_planner._solve_lineup`.
4. **`read_cache_dict` import** in `season_routes.py` (Task 7) — confirm/add it to the existing `season_data` import.
5. **Refresh attribute names** (Task 6) are `self.roster_players`, `self.fa_players`, `self.projected_standings`, `self.team_sds`, `self.fraction_remaining`, `self.config.roster_slots`, `self.config.team_name` — verified in `_audit_roster`.
