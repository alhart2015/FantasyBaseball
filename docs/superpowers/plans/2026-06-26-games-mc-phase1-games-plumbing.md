# Games-based MC -- Phase 1 (games-data plumbing) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`).

**Goal:** Thread games (`g` for hitters and pitchers, `gs` for pitchers) from the FanGraphs projection CSVs through the blend into `HitterStats`/`PitcherStats`, so later phases can read ROS games. PA/IP unchanged. NO model logic -- plumbing + audit only.

**Architecture:** Add the fields to the two dataclasses; add the CSV->normalized column mappings; add the columns to the blended counting-stat lists. Audit that the new fields cannot leak into SGP, that serialization round-trips, and that a missing/zero `g` cannot be mistaken for "plays zero games."

**Tech Stack:** Python, pandas, pytest. Files: `models/player.py`, `data/fangraphs.py`, `data/projections.py`.

## Global Constraints

- ASCII-only in source/strings (Windows cp1252).
- Numeric defaults via `is not None`, never `x or default`.
- Spec: `docs/superpowers/specs/2026-06-26-games-based-availability-mc-design.md`, Component 1.
- The new fields MUST NOT enter SGP (`calculate_player_sgp` reads only explicitly named fields -- confirm by test, do not assume).
- `g`/`gs` are ROS-scaled games for the in-season path (sourced from a fresh refresh's `rest_of_season`); a literal `g=0` from stale persisted JSON must NEVER be trusted as "plays zero games" by later consumers (the per-game-value rule derives from PA/IP when `g` is 0/absent). Phase 1 only lands the field + records the hazard; the derivation gate is implemented in Phases 2-3.

---

### Task 1: Add `g`/`gs` fields to the stat dataclasses

**Files:** Modify `src/fantasy_baseball/models/player.py` (`HitterStats` ~17-59, `PitcherStats` ~62-109); Test `tests/test_models/test_player.py` (or the existing player-stats test file -- locate it first).

**Interfaces:**
- Produces: `HitterStats.g: float = 0`; `PitcherStats.g: float = 0`, `PitcherStats.gs: float = 0`. `from_dict` (already `{k: float(d.get(k,0) or 0) for k in stat_fields}`) picks them up automatically; `to_dict` (already `fields()`-driven) emits them.

- [ ] **Step 1: Write the failing test** (find the existing player test file first; if none, create `tests/test_models/test_player.py`):

```python
from fantasy_baseball.models.player import HitterStats, PitcherStats


def test_hitter_stats_carries_games():
    h = HitterStats.from_dict({"r": 80, "hr": 25, "rbi": 80, "sb": 10, "h": 150, "ab": 550, "g": 150})
    assert h.g == 150
    assert h.to_dict()["g"] == 150


def test_pitcher_stats_carries_games_and_starts():
    p = PitcherStats.from_dict({"w": 10, "k": 180, "ip": 190, "er": 70, "bb": 50, "h_allowed": 160, "g": 32, "gs": 32})
    assert p.g == 32
    assert p.gs == 32
    rt = PitcherStats.from_dict(p.to_dict())
    assert rt.g == 32 and rt.gs == 32  # round-trip stable


def test_missing_games_defaults_zero_not_error():
    # Backward-compat: dicts lacking g/gs must not raise; they default 0.
    h = HitterStats.from_dict({"r": 80, "hr": 25, "rbi": 80, "sb": 10, "h": 150, "ab": 550})
    p = PitcherStats.from_dict({"w": 10, "k": 180, "ip": 190, "er": 70, "bb": 50, "h_allowed": 160})
    assert h.g == 0 and p.g == 0 and p.gs == 0
```

- [ ] **Step 2: Run, confirm FAIL** (`AttributeError: ... has no attribute 'g'`):
`pytest tests/test_models/test_player.py -k games -v`

- [ ] **Step 3: Add the fields.** In `HitterStats`, add `g: float = 0` among the counting fields (e.g. after `ab`). In `PitcherStats`, add `g: float = 0` and `gs: float = 0` (e.g. after `ip`). Do NOT touch the `sgp` field handling. Read both dataclasses first to place the fields consistently with existing style.

- [ ] **Step 4: Run, confirm PASS:** `pytest tests/test_models/test_player.py -k games -v`
- [ ] **Step 5: Full model + projections tests, confirm no regression:** `pytest tests/test_models/ tests/test_data/ -q`.
- [ ] **Step 6: mypy (models/ is under `[tool.mypy].files`):** `mypy src/fantasy_baseball/models/player.py` -- expected clean (trivial `float` field). (Full `mypy` may error on the pre-existing deleted `category_odds.py`; that is unrelated.)
- [ ] **Step 7: Commit:**
```bash
git add src/fantasy_baseball/models/player.py tests/test_models/test_player.py
git commit -m "feat(models): add g to HitterStats; g/gs to PitcherStats (games plumbing)"
```

---

### Task 2: Thread the CSV `G`/`GS` columns through the blend

**Files:** Modify `src/fantasy_baseball/data/fangraphs.py` (column maps ~7-41); `src/fantasy_baseball/data/projections.py` (counting-col lists ~22-23). Tests: column-map assertions in `tests/test_data/test_fangraphs.py` (where the existing `GS->gs` map/parse tests live); counting-col assertions in `tests/test_data/test_projections.py`.

**Interfaces:**
- Consumes: Task 1's fields.
- Produces: CSV `G`->`g` (both hitter and pitcher maps); `GS`->`gs` (already present in pitcher map); blended frames carry `g` (hitters + pitchers) and `gs` (pitchers).

Current state (verified): `HITTING_COLUMN_MAP` has NO `G`; `PITCHING_COLUMN_MAP` has `"GS": "gs"` but NO `"G"`. `HITTING_COUNTING_COLS = [...,"pa"]` (no `g`); `PITCHING_COUNTING_COLS = [...,"gs"]` (no `g`).

- [ ] **Step 1: Write the failing test** in the projections test file:

```python
def test_blend_threads_games_columns():
    import pandas as pd
    from fantasy_baseball.data.fangraphs import HITTING_COLUMN_MAP, PITCHING_COLUMN_MAP

    # G must normalize to g for both; GS->gs already present.
    assert HITTING_COLUMN_MAP.get("G") == "g"
    assert PITCHING_COLUMN_MAP.get("G") == "g"
    assert PITCHING_COLUMN_MAP.get("GS") == "gs"

    from fantasy_baseball.data.projections import (
        HITTING_COUNTING_COLS, PITCHING_COUNTING_COLS,
    )
    assert "g" in HITTING_COUNTING_COLS
    assert "g" in PITCHING_COUNTING_COLS and "gs" in PITCHING_COUNTING_COLS
```

(If the projections test suite has an end-to-end blend fixture with real/synthetic CSVs, ALSO assert the blended hitter/pitcher frame contains a `g` column with the expected weighted value -- follow the existing blend-test pattern in that file.)

- [ ] **Step 2: Run, confirm FAIL.**
- [ ] **Step 3: Implement.** In `fangraphs.py`: add `"G": "g"` to `HITTING_COLUMN_MAP` and add `"G": "g"` to `PITCHING_COLUMN_MAP` (leave the existing `"GS": "gs"`). In `projections.py`: append `"g"` to `HITTING_COUNTING_COLS` and append `"g"` to `PITCHING_COUNTING_COLS` (leave `"gs"`). Do NOT add `g`/`gs` to `REQUIRED_*_COLS` (a system that omits `G` must still load -- the blend should drop it from the `G`-average for that system, not error; confirm the blend tolerates a missing column for a stat in the COUNTING list -- if it does not, guard so a missing `G` is skipped, matching the spec's "drop it from the G blend rather than zeroing").
- [ ] **Step 4: Run, confirm PASS.**
- [ ] **Step 5: Run the full data suite, confirm no regression:** `pytest tests/test_data/ -q`. If any blend test asserts an exact column set, update it to include `g` (justify in the commit -- this is the intended new column, not a loosened assertion).
- [ ] **Step 6: mypy (both files under `[tool.mypy].files`):** `mypy src/fantasy_baseball/data/fangraphs.py src/fantasy_baseball/data/projections.py` -- expected clean.
- [ ] **Step 7: Commit:**
```bash
git add src/fantasy_baseball/data/fangraphs.py src/fantasy_baseball/data/projections.py tests/test_data/test_fangraphs.py tests/test_data/test_projections.py
git commit -m "feat(data): thread CSV G/GS into the blend as g/gs (games plumbing)"
```

---

### Task 3: Audit -- SGP isolation, round-trip, backward-compat hazard

**Files:** Test only -- `tests/test_sgp/` (SGP isolation) and the player test file (round-trip already in Task 1). No source changes expected unless the audit finds a leak.

**Interfaces:** Consumes Tasks 1-2.

- [ ] **Step 1: SGP-isolation test.** Assert that adding `g`/`gs` does NOT change SGP. Locate `calculate_player_sgp` (`sgp/player_value.py`); construct a hitter/pitcher with and without `g`/`gs` set and assert identical SGP:

```python
# tests/test_sgp/test_sgp_ignores_games.py
from fantasy_baseball.models.player import HitterStats, PitcherStats
from fantasy_baseball.sgp.player_value import calculate_player_sgp


def test_sgp_unaffected_by_games_fields():
    base_h = {"r": 80, "hr": 25, "rbi": 80, "sb": 10, "h": 150, "ab": 550}
    assert calculate_player_sgp(HitterStats.from_dict(base_h)) == \
           calculate_player_sgp(HitterStats.from_dict({**base_h, "g": 150}))
    base_p = {"w": 10, "k": 180, "ip": 190, "er": 70, "bb": 50, "h_allowed": 160}
    assert calculate_player_sgp(PitcherStats.from_dict(base_p)) == \
           calculate_player_sgp(PitcherStats.from_dict({**base_p, "g": 32, "gs": 32}))
```

- [ ] **Step 2: Run it.** Expected PASS (SGP reads only explicitly named stat fields). If it FAILS, SGP is `fields()`-driven somewhere -- STOP and report; the spec's "by construction" claim would be wrong and needs a guard.
- [ ] **Step 3: Record the backward-compat hazard.** Add a short note to the Phase-1 commit body (and confirm `TODO.md`/spec already capture it): persisted JSON written before this change lacks `G`, so `from_dict`'s `or 0` yields `g=0`; downstream per-game-value consumers (Phase 3) MUST derive games from PA/IP when `g==0`/absent rather than trust the literal zero. No code change in Phase 1 -- the consumers don't exist yet.
- [ ] **Step 4: Commit** (if any test files added):
```bash
git add tests/test_sgp/test_sgp_ignores_games.py
git commit -m "test(sgp): pin that g/gs do not leak into SGP (games-plumbing audit)"
```

---

## Self-Review

**Spec coverage:** Implements spec Component 1 + Phase 1 exactly -- `g` on `HitterStats`, `g`/`gs` on `PitcherStats`, threaded from the CSVs (`G`/`GS` already in the exports) through the blend; the three audit items (SGP isolation by test, forward round-trip, backward-compat `g=0` hazard recorded). No model logic, no consumers -- those are Phases 2-6.

**Placeholder scan:** Concrete file locations and exact map/list edits given (verified against source: `fangraphs.py:7-41`, `projections.py:22-23`, `player.py:17-109`). Test files: locate the real paths first (the plan names likely paths; confirm before writing).

**Type consistency:** `g`/`gs` are `float` like the other counting fields; `from_dict`/`to_dict` are `fields()`-driven so they pick up automatically. `gs` reuses the already-present `PITCHING_COLUMN_MAP["GS"]` and `PITCHING_COUNTING_COLS` entry -- only the `PitcherStats.gs` field was missing.
