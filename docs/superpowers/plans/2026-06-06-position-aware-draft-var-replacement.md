# Position-aware draft VAR replacement — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the flat, demand-based draft *hitter* replacement floor with an empirical per-position waiver floor (`REPLACEMENT_BY_POSITION`), so VAR stops over-crediting cheap categories (MI speed) and under-crediting scarce ones (catcher).

**Architecture:** Add one new function in `sgp/replacement.py` that starts from the existing demand-based levels (reused unchanged for the pitcher "P" floor) and overrides each hitter position + UTIL with the scalar SGP of its waiver stat line. `calculate_var` is untouched — it just subtracts a better floor. Wire the new function into the two VAR consumers: the static board (`draft/board.py`) and the live recommender (`draft/recommender.py`). Measure the before/after on real data with a throwaway diagnostic first.

**Tech Stack:** Python, pandas, pytest. SGP math in `fantasy_baseball.sgp`. Spec: `docs/superpowers/specs/2026-06-06-position-aware-draft-var-replacement-design.md`.

---

## File Structure

- **Modify** `src/fantasy_baseball/sgp/replacement.py` — add `position_aware_replacement_levels` + a private `_empirical_floor_sgp` helper. (`calculate_replacement_levels` is **not** modified — it is reused for the P floor and its tests stay green.)
- **Modify** `src/fantasy_baseball/draft/board.py` — line 75 calls the new function (denoms + repl_rates already in scope).
- **Modify** `src/fantasy_baseball/draft/recommender.py` — `get_recommendations` computes denoms + repl_rates live and calls the new function; update imports.
- **Test** `tests/test_sgp/test_replacement.py` — new `TestPositionAwareReplacementLevels` class.
- **Create (throwaway)** `scripts/diag_draft_replacement.py` — before/after evidence; deleted in Task 5.

---

## Task 1: `position_aware_replacement_levels` + unit tests

**Files:**
- Modify: `src/fantasy_baseball/sgp/replacement.py`
- Test: `tests/test_sgp/test_replacement.py`

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_sgp/test_replacement.py`. Add these imports at the top of the file (extend the existing `from fantasy_baseball.sgp.replacement import (...)` block to include `position_aware_replacement_levels`, and add the others):

```python
from fantasy_baseball.models.player import PlayerType
from fantasy_baseball.sgp.denominators import get_sgp_denominators
from fantasy_baseball.sgp.player_value import calculate_player_sgp
from fantasy_baseball.sgp.var import calculate_var
from fantasy_baseball.utils.constants import REPLACEMENT_BY_POSITION
```

Then append the new test class:

```python
class TestPositionAwareReplacementLevels:
    _HITTER_STARTERS = {
        "C": 1, "1B": 1, "2B": 1, "3B": 1, "SS": 1, "OF": 1, "UTIL": 1, "P": 5,
    }

    def test_pitcher_floor_matches_demand_based(self):
        """We only touch hitters -- the 'P' floor must be byte-identical to
        the existing demand-based calculation on the same pool."""
        pool = _make_player_pool()
        starters = {"P": 50}
        base = calculate_replacement_levels(pool, starters)
        pa = position_aware_replacement_levels(pool, starters)
        assert pa["P"] == pytest.approx(base["P"])

    def test_hitter_floor_equals_constant_sgp(self):
        """Each hitter floor equals the SGP of that position's waiver line,
        computed on the same scale (asserted against the directly-computed
        scalar, never a magic number)."""
        pool = _make_player_pool()
        denoms = get_sgp_denominators()
        levels = position_aware_replacement_levels(
            pool, self._HITTER_STARTERS, denoms, {"avg": 0.250}
        )
        for pos in ("C", "1B", "2B", "3B", "SS", "OF"):
            line = REPLACEMENT_BY_POSITION[pos]
            expected = calculate_player_sgp(
                pd.Series(
                    {
                        "player_type": PlayerType.HITTER,
                        "r": line["r"],
                        "hr": line["hr"],
                        "rbi": line["rbi"],
                        "sb": line["sb"],
                        "ab": line["ab"],
                        "avg": line["h"] / line["ab"],
                    }
                ),
                denoms=denoms,
                replacement_avg=0.250,
            )
            assert levels[pos] == pytest.approx(expected)

    def test_catcher_is_scarcest_outfield_is_deepest(self):
        """The whole point: floors are NOT flat. Catcher (nothing free on
        waivers) is the lowest hitter floor; OF (deepest pool) the highest.
        Pins the spread so a future flat regression fails loudly."""
        pool = _make_player_pool()
        levels = position_aware_replacement_levels(pool, self._HITTER_STARTERS)
        assert levels["C"] < levels["1B"]
        assert levels["C"] < levels["OF"]
        assert levels["OF"] > levels["3B"]

    def test_util_floor_is_max_hitter_floor(self):
        """A UTIL slot is streamed with the best free hitter -> the highest
        hitter floor."""
        pool = _make_player_pool()
        levels = position_aware_replacement_levels(pool, self._HITTER_STARTERS)
        hitter_floors = [levels[p] for p in ("C", "1B", "2B", "3B", "SS", "OF")]
        assert levels["UTIL"] == pytest.approx(max(hitter_floors))

    def test_corner_outranks_mi_at_equal_sgp(self):
        """Downstream effect on VAR: with empirical floors, two players with
        equal total_sgp -- one valued at SS (deep speed, high floor), one at
        3B (scarcer, lower floor) -- the corner gets the higher VAR. This is
        the behavior the fix exists to produce."""
        pool = _make_player_pool()
        levels = position_aware_replacement_levels(pool, self._HITTER_STARTERS)
        ss_player = pd.Series({"total_sgp": 12.0, "positions": ["SS"]})
        tb_player = pd.Series({"total_sgp": 12.0, "positions": ["3B"]})
        assert calculate_var(tb_player, levels) > calculate_var(ss_player, levels)

    def test_tiny_pool_does_not_raise_and_keeps_pitcher_floor(self):
        """Empirical hitter overrides do not depend on pool contents, so a
        pool with no catchers still yields a 'C' floor and a 'P' floor."""
        pool = pd.DataFrame(
            [{"name": "P_0", "positions": ["SP"], "total_sgp": 5.0, "player_type": "pitcher"}]
        )
        levels = position_aware_replacement_levels(pool, {"C": 1, "P": 1})
        assert "P" in levels
        assert "C" in levels
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `pytest tests/test_sgp/test_replacement.py::TestPositionAwareReplacementLevels -v`
Expected: FAIL — `ImportError: cannot import name 'position_aware_replacement_levels'`.

- [ ] **Step 3: Implement the function**

In `src/fantasy_baseball/sgp/replacement.py`, extend the imports at the top:

```python
from fantasy_baseball.models.player import PlayerType
from fantasy_baseball.sgp.denominators import get_sgp_denominators
from fantasy_baseball.sgp.player_value import (
    REPLACEMENT_AVG,
    REPLACEMENT_ERA,
    REPLACEMENT_WHIP,
    calculate_player_sgp,
)
from fantasy_baseball.utils.constants import (
    REPLACEMENT_BY_POSITION,
    STARTERS_PER_POSITION,
    Category,
)
```

(The existing file already imports `REPLACEMENT_AVG, REPLACEMENT_ERA, REPLACEMENT_WHIP` from `player_value` and `STARTERS_PER_POSITION` from `constants` — merge, don't duplicate. `calculate_player_sgp`, `get_sgp_denominators`, `REPLACEMENT_BY_POSITION`, `PlayerType`, and `Category` are the new symbols.)

Append at the end of the file:

```python
# Hitter positions for which REPLACEMENT_BY_POSITION carries a waiver line.
_EMPIRICAL_HITTER_POSITIONS = ("C", "1B", "2B", "3B", "SS", "OF")


def _empirical_floor_sgp(
    position: str,
    denoms: dict[Category, float],
    replacement_avg: float,
) -> float:
    """SGP of a position's empirical waiver line (REPLACEMENT_BY_POSITION).

    Built and scored through the same ``calculate_player_sgp`` path as real
    players so the floor and player values land on one scale.
    """
    line = REPLACEMENT_BY_POSITION[position]
    row = pd.Series(
        {
            "player_type": PlayerType.HITTER,
            "r": line["r"],
            "hr": line["hr"],
            "rbi": line["rbi"],
            "sb": line["sb"],
            "ab": line["ab"],
            "avg": line["h"] / line["ab"] if line["ab"] else 0.0,
        }
    )
    return calculate_player_sgp(row, denoms=denoms, replacement_avg=replacement_avg)


def position_aware_replacement_levels(
    player_pool: pd.DataFrame,
    starters_per_position: dict[str, int] | None = None,
    denoms: dict[Category, float] | None = None,
    repl_rates: dict[str, float] | None = None,
) -> dict[str, float]:
    """Replacement levels with empirical waiver floors for hitter positions.

    Starts from the demand-based ``calculate_replacement_levels`` (so the
    pitcher "P" floor and all demand math are reused unchanged), then
    overrides each hitter position -- and UTIL -- with the scalar SGP of its
    empirical waiver line. ``calculate_var`` consumes the result unchanged.
    """
    levels = calculate_replacement_levels(player_pool, starters_per_position)

    if denoms is None:
        denoms = get_sgp_denominators()
    replacement_avg = repl_rates["avg"] if repl_rates and "avg" in repl_rates else REPLACEMENT_AVG

    empirical: dict[str, float] = {}
    for pos in _EMPIRICAL_HITTER_POSITIONS:
        if pos in levels and pos in REPLACEMENT_BY_POSITION:
            empirical[pos] = _empirical_floor_sgp(pos, denoms, replacement_avg)

    levels.update(empirical)
    if "UTIL" in levels and empirical:
        levels["UTIL"] = max(empirical.values())

    return levels
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `pytest tests/test_sgp/test_replacement.py -v`
Expected: PASS — the new class passes and every pre-existing `TestReplacementLevels` / `TestReplacementRates` / `TestFindReplacementPlayers` test still passes (the demand-based function is unmodified).

- [ ] **Step 5: Commit**

```bash
git add src/fantasy_baseball/sgp/replacement.py tests/test_sgp/test_replacement.py
git commit -m "feat(sgp): position-aware empirical replacement floors for hitters

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

## Task 2: Before/after diagnostic on real data

**Files:**
- Create: `scripts/diag_draft_replacement.py`

This runs against the *un-wired* board (still demand-based) and computes the empirical floors with the new function, so we see the per-position deltas and the cross-type tilt before changing the production VAR path.

- [ ] **Step 1: Write the diagnostic script**

Create `scripts/diag_draft_replacement.py`:

```python
"""Throwaway diagnostic: demand-based vs empirical draft replacement floors.

Read-only. Builds the real board (still demand-based at this point), then
prints, per position, the demand-based floor vs the empirical waiver floor and
the resulting VAR shift for a few representative players. Delete before merge.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from fantasy_baseball.config import load_config
from fantasy_baseball.data.db import get_connection
from fantasy_baseball.draft.board import build_draft_board
from fantasy_baseball.sgp.denominators import get_sgp_denominators
from fantasy_baseball.sgp.replacement import (
    calculate_replacement_levels,
    calculate_replacement_rates,
    position_aware_replacement_levels,
)
from fantasy_baseball.sgp.var import calculate_var
from fantasy_baseball.utils.constants import compute_starters_per_position

SAMPLE_NAMES = [
    "Bobby Witt Jr.",   # speed SS
    "Jose Altuve",      # MI
    "Rafael Devers",    # power corner
    "Salvador Perez",   # catcher
    "Aaron Judge",      # power OF
    "Zack Wheeler",     # SP (cross-type tilt)
    "Emmanuel Clase",   # RP
]


def main() -> None:
    config = load_config(Path("config/league.yaml"))
    conn = get_connection()
    try:
        board = build_draft_board(
            conn=conn,
            sgp_overrides=config.sgp_overrides or None,
            roster_slots=config.roster_slots or None,
            num_teams=config.num_teams,
        )
    finally:
        conn.close()

    starters = compute_starters_per_position(config.roster_slots or None, config.num_teams)
    denoms = get_sgp_denominators(config.sgp_overrides or None)
    repl_rates = calculate_replacement_rates(board, starters)
    demand = calculate_replacement_levels(board, starters)
    emp = position_aware_replacement_levels(board, starters, denoms, repl_rates)

    print(f"{'POS':6s} {'demand':>8s} {'empirical':>10s} {'delta':>8s}")
    for pos in sorted(set(demand) | set(emp)):
        d = demand.get(pos)
        e = emp.get(pos)
        if d is None or e is None:
            print(f"{pos:6s} {str(d):>8s} {str(e):>10s}")
            continue
        print(f"{pos:6s} {d:8.2f} {e:10.2f} {e - d:8.2f}")

    print(f"\n{'PLAYER':22s} {'pos':5s} {'VAR_before':>11s} {'VAR_after':>10s} {'delta':>8s}")
    for name in SAMPLE_NAMES:
        match = board[board["name"] == name]
        if match.empty:
            print(f"{name:22s} (not on board)")
            continue
        row = match.iloc[0]
        before = calculate_var(row, demand)
        after = calculate_var(row, emp)
        pos = str(row.get("best_position", ""))
        print(f"{name:22s} {pos:5s} {before:11.2f} {after:10.2f} {after - before:8.2f}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Run it and read the output**

Run: `python scripts/diag_draft_replacement.py`
Expected: a per-position table where **C** has the smallest empirical floor (often *below* demand-based -> positive VAR for catchers) and **2B/SS/OF** have the largest (negative VAR delta for MI/OF), plus a player table showing speed MIs losing VAR and catchers gaining. Pitchers ("P") show `delta = 0.00` (unchanged) — confirming the cross-type tilt comes entirely from the hitter side. If any sample name is "not on board", that's fine — it's illustrative.

**Checkpoint:** Eyeball the cross-type tilt. If hitter floors move only modestly (a few tenths of an SGP) and the C-low / OF-high ordering holds, proceed. If the tilt is wild (e.g. every hitter drops several SGP, collapsing hitters below pitchers wholesale), stop and report — that would argue for pulling pitcher floors into this increment rather than deferring. (No commit — this script is deleted in Task 5.)

---

## Task 3: Wire the static board

**Files:**
- Modify: `src/fantasy_baseball/draft/board.py:11-15` (imports) and `:75` (call site)

- [ ] **Step 1: Update the import**

In `src/fantasy_baseball/draft/board.py`, change the replacement import block (currently lines 11-14):

```python
from fantasy_baseball.sgp.replacement import (
    calculate_replacement_rates,
    position_aware_replacement_levels,
)
```

(Drop `calculate_replacement_levels` — the board no longer calls it directly; keep `calculate_replacement_rates`, still used at line 62.)

- [ ] **Step 2: Update the call site**

In `build_draft_board`, change line 75 from:

```python
    replacement_levels = calculate_replacement_levels(pool, starters)
```

to:

```python
    replacement_levels = position_aware_replacement_levels(pool, starters, denoms, repl_rates)
```

(`denoms` is defined at line 54, `repl_rates` at line 62 — both already in scope.)

- [ ] **Step 3: Run the board + sgp tests**

Run: `pytest tests/test_draft tests/test_sgp -v`
Expected: PASS. If a board test asserts a specific demand-based hitter VAR or floor, that is an **intended** behavior change — update the expected value to the new position-aware result and note the justification in the commit body (per the repo's "don't silently change tests" rule). Tests that assert `calculate_replacement_levels` directly are untouched and must stay green; if one breaks, you changed the wrong thing.

- [ ] **Step 4: Commit**

```bash
git add src/fantasy_baseball/draft/board.py
git commit -m "feat(draft): board VAR uses position-aware empirical floors

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

## Task 4: Wire the live recommender

**Files:**
- Modify: `src/fantasy_baseball/draft/recommender.py:17` (import) and `:146-147` (call site)

- [ ] **Step 1: Update the imports**

In `src/fantasy_baseball/draft/recommender.py`, replace line 17:

```python
from fantasy_baseball.sgp.replacement import calculate_replacement_levels
```

with:

```python
from fantasy_baseball.sgp.denominators import get_sgp_denominators
from fantasy_baseball.sgp.replacement import (
    calculate_replacement_rates,
    position_aware_replacement_levels,
)
```

- [ ] **Step 2: Update the call site**

In `get_recommendations`, replace lines 146-147:

```python
    starters = compute_starters_per_position(roster_slots, num_teams)
    repl_levels = calculate_replacement_levels(available, starters)
```

with:

```python
    starters = compute_starters_per_position(roster_slots, num_teams)
    denoms = get_sgp_denominators()
    repl_rates = calculate_replacement_rates(available, starters)
    repl_levels = position_aware_replacement_levels(available, starters, denoms, repl_rates)
```

This matches the board: the live floors use the same denominators and pool-derived rates as the players they are subtracted from. (Default denominators match current `league.yaml`; the `sgp_overrides` threading gap is the separate TODO line 17 and is not widened here.)

- [ ] **Step 3: Run the recommender tests**

Run: `pytest tests/test_draft -v`
Expected: PASS. Same rule as Task 3 — a recommender test asserting a specific demand-based hitter VAR is an intended change; update the expected value with a noted justification. VONA tests must be unaffected (VONA does not consume replacement levels).

- [ ] **Step 4: Commit**

```bash
git add src/fantasy_baseball/draft/recommender.py
git commit -m "feat(draft): live recommender VAR uses position-aware floors

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

## Task 5: Full verification + cleanup

**Files:**
- Delete: `scripts/diag_draft_replacement.py`

- [ ] **Step 1: Delete the throwaway diagnostic**

```bash
git rm -f scripts/diag_draft_replacement.py 2>/dev/null || rm -f scripts/diag_draft_replacement.py
```

(It was never committed, so `git rm` may report nothing — just ensure the file is gone so `vulture`/`ruff` do not see it.)

- [ ] **Step 2: Run the full checklist (repo CLAUDE.md requirement)**

```bash
pytest -v
ruff check .
ruff format --check .
vulture
mypy
```

Expected: all green. `replacement.py`, `board.py`, and `var.py` are in the mypy set, so type errors block. Common fixes if mypy complains: ensure `_empirical_floor_sgp` returns `float` (it does — `calculate_player_sgp` returns `float`) and that `denoms` is `dict[Category, float] | None`. `vulture` must show no NEW findings — the diagnostic is deleted and `calculate_replacement_levels` is still referenced (internally by the new function + its own tests), so it is not dead.

- [ ] **Step 3: Final commit (only if the checklist forced any fixups)**

```bash
git add -A
git commit -m "chore(draft): cleanup + verification for position-aware VAR floors

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

If Steps 1-2 produced no changes (diagnostic uncommitted, checklist already green), skip this commit.

---

## Notes for the implementer

- **Do not modify `calculate_replacement_levels` or `calculate_var`.** The whole design hinges on reusing them: the demand-based function supplies the pitcher floor, and `calculate_var` just subtracts whatever floor it is handed.
- **Scope is hitters only.** Pitchers intentionally keep the demand-based unified-"P" floor. If you find yourself editing pitcher floors, stop — that is the separate TODO line 61.
- **`calculate_player_sgp` accepts a `pd.Series`** (its dict-branch reads `.get(...)`). The helper wraps the line in `pd.Series(...)` for this reason; do not pass a bare `dict` (mypy expects `HitterStats | PitcherStats | pd.Series`).
- If a pre-existing draft test encodes the old flat-floor VAR, updating its expected value **with a one-line justification in the commit** is correct here — the requirement changed. Do not loosen assertions to vague ranges; compute the new exact expected value.
```
