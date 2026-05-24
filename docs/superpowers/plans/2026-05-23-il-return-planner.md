# IL Return Planner Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add an "IL Returns" planner that, given the IL players a manager checks as returning, computes the optimal legal roster under the body-count cap and returns the top-5 drop/start/bench transaction plans ranked by deltaRoto (with std dev + P(helps)), surfaced on the `/roster-audit` page via an on-demand JSON route.

**Architecture:** A new pure-logic module `lineup/il_return_planner.py` builds the post-return body pool, computes the slot-based body-count overflow (forced drops), enumerates drop-sets, re-solves each with the existing `optimize_hitter_lineup`/`optimize_pitcher_lineup`, and ranks them with `compute_delta_roto_band`. The baseline for every plan's band is the *pre-drop ideal* active lineup (which already includes the returning players), so they cancel and the activation gain the standings already price is never double-counted; each plan's deltaRoto is purely the cost of its forced drop. A Flask JSON route reconstructs cached inputs (the same ones the trade-builder route uses) and calls the planner; the `/roster-audit` template gains an interactive section.

**Tech Stack:** Python 3, dataclasses, scipy (via existing optimizer), Flask (season app), Jinja2 templates, vanilla JS fetch. Tests: pytest.

**Spec:** `docs/superpowers/specs/2026-05-23-il-return-planner-design.md`. This plan resolves that spec's open risk #1 (band/deltaRoto reconciliation) by using `compute_delta_roto_band` on the active-lineup change with a pre-drop-ideal baseline, rather than diffing whole-roster `project_team_stats` scores.

---

## File Structure

- **Create** `src/fantasy_baseball/lineup/il_return_planner.py` — all planner logic + the `Move`, `MovePlan`, `IlReturnPlanResult` dataclasses and the `plan_il_returns` entry point. One responsibility: turn (roster, activating IL players, slots) into ranked transaction plans.
- **Create** `tests/test_lineup/test_il_return_planner.py` — unit tests (fixtures copied from `test_roster_audit.py`).
- **Modify** `src/fantasy_baseball/web/season_routes.py` — add the `/api/il-return-plan` GET route inside `register_routes(app)`.
- **Modify** `tests/test_web/test_season_routes.py` — one route test.
- **Modify** `src/fantasy_baseball/web/templates/season/roster_audit.html` — add the "IL Returns" section + fetch/render JS.

Key facts the code depends on (verified against the tree):
- `roster_slots` (config) keys are strings like `"C"`, `"P"`, `"BN"`, `"IL"`; `Position.parse("IL")` is in `IL_SLOTS = {IL, IL+, DL, DL+}`, `Position.parse("BN")` is not.
- `Player.is_on_il()` is True for status in `IL_STATUSES` **or** slot in `IL_SLOTS` (so it catches both Hader-in-IL-slot and Webb-on-BN-with-IL-status). The body-count cap, however, must key on slot only (`selected_position in IL_SLOTS`), because Yahoo counts a BN+IL-status player against the roster size.
- `optimize_hitter_lineup(hitters, full_roster, projected_standings, team_name, roster_slots=None, team_sds=None, fraction_remaining=None) -> list[HitterAssignment]`. Does **not** filter IL from `hitters` — caller pre-filters. `HitterAssignment` has `.slot: Position`, `.name`, `.player`.
- `optimize_pitcher_lineup(pitchers, full_roster, projected_standings, team_name, slots=9, team_sds=None, fraction_remaining=None) -> tuple[list[PitcherStarter], list[Player]]` i.e. `(starters, bench)`. `PitcherStarter` has `.name`, `.player` (no `.slot`).
- Passing `fraction_remaining=None` to the optimizers makes them skip per-starter band computation (faster) without changing the lineup chosen. The planner computes the plan-level band separately, so always pass `None` to the optimizers.
- `compute_delta_roto_band(before_players, after_players, field_stats, team_name, fraction_remaining, *, projected_standings, team_sds) -> DeltaRotoBand`. `_swap_sets` dedups by player **name**; players in both lists cancel. `DeltaRotoBand.to_dict()` -> `{"mean","sd","p_positive","verdict"}`, verdict in `{"real","coin-flip","downgrade"}`. `.mean` is the EV deltaRoto.
- `projected_standings.field_stats(team_name)` returns `{other_team: CategoryStats}` (param is positional `exclude`).
- Fixture helpers in `tests/test_lineup/test_roster_audit.py`: `_hitter(name, positions, **stats)`, `_pitcher(name, positions, **stats)`, `_projected(rows)`, `_minimal_standings()`. IL status is set via `player.status = "IL15"` or the `Player(..., selected_position=Position.parse("IL"), status="IL15")` constructor.

---

## Task 1: Module skeleton, dataclasses, and capacity/overflow helpers

**Files:**
- Create: `src/fantasy_baseball/lineup/il_return_planner.py`
- Test: `tests/test_lineup/test_il_return_planner.py`

- [ ] **Step 1: Write the failing test**

Create `tests/test_lineup/test_il_return_planner.py`:

```python
from fantasy_baseball.lineup.il_return_planner import (
    IlReturnPlanResult,
    Move,
    MovePlan,
    _counts_against_cap,
    roster_capacity,
)
from fantasy_baseball.models.player import PitcherStats, Player, PlayerType
from fantasy_baseball.models.positions import Position

ROSTER_SLOTS = {
    "C": 1, "1B": 1, "2B": 1, "3B": 1, "SS": 1,
    "OF": 3, "UTIL": 1, "P": 3, "BN": 1, "IL": 2,
}


def _pitcher(name, slot=None, status=""):
    return Player(
        name=name,
        player_type=PlayerType.PITCHER,
        positions=[Position.P],
        rest_of_season=PitcherStats(
            ip=60.0, w=3.0, k=60.0, sv=0.0, er=20.0, bb=20.0,
            h_allowed=50.0, era=3.00, whip=1.17,
        ),
        selected_position=Position.parse(slot) if slot else None,
        status=status,
    )


class TestCapacity:
    def test_capacity_excludes_il_slots_only(self):
        # 9 hitter slots (C,1B,2B,3B,SS,OF*3,UTIL) + 3 P + 1 BN = 13; IL excluded.
        assert roster_capacity(ROSTER_SLOTS) == 13

    def test_bn_il_status_counts_against_cap(self):
        webb = _pitcher("Webb", slot="BN", status="IL10")
        assert _counts_against_cap(webb) is True

    def test_true_il_slot_does_not_count(self):
        hader = _pitcher("Hader", slot="IL", status="IL15")
        assert _counts_against_cap(hader) is False

    def test_active_slot_counts(self):
        active = _pitcher("Active", slot="P")
        assert _counts_against_cap(active) is True


class TestDataclasses:
    def test_move_to_dict(self):
        m = Move(name="Webb", player_type="pitcher", from_slot="BN", to_slot="P")
        assert m.to_dict() == {
            "name": "Webb", "player_type": "pitcher", "from_slot": "BN", "to_slot": "P",
        }

    def test_move_plan_to_dict_rounds_delta(self):
        plan = MovePlan(
            drops=["Scrub"],
            moves=[Move("Webb", "pitcher", "BN", "P")],
            delta_roto=-0.123,
            band={"mean": -0.12, "sd": 0.4, "p_positive": 0.4, "verdict": "coin-flip"},
        )
        d = plan.to_dict()
        assert d["drops"] == ["Scrub"]
        assert d["delta_roto"] == -0.12
        assert d["moves"][0]["name"] == "Webb"
        assert d["band"]["verdict"] == "coin-flip"

    def test_result_to_dict(self):
        res = IlReturnPlanResult(activating=["Hader"], capacity=13, overflow=1, plans=[])
        d = res.to_dict()
        assert d == {
            "activating": ["Hader"], "capacity": 13, "overflow": 1,
            "plans": [], "warning": None,
        }
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_lineup/test_il_return_planner.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'fantasy_baseball.lineup.il_return_planner'`.

- [ ] **Step 3: Write minimal implementation**

Create `src/fantasy_baseball/lineup/il_return_planner.py`:

```python
"""IL return planner -- compute the optimal legal roster + transaction plan
when injured-list players are reactivated.

When IL players come off the IL they temporarily push the roster over the
active+bench body-count cap, forcing a drop plus an active/bench reshuffle.
Given the IL players a manager wants to activate, this module computes the
forced drops and returns the top transaction plans ranked by deltaRoto.
"""

from __future__ import annotations

import dataclasses
import logging
from collections.abc import Mapping
from dataclasses import asdict, dataclass, field
from itertools import combinations
from typing import Any

from fantasy_baseball.lineup.delta_roto import compute_delta_roto_band
from fantasy_baseball.lineup.optimizer import (
    optimize_hitter_lineup,
    optimize_pitcher_lineup,
)
from fantasy_baseball.models.player import Player, PlayerType
from fantasy_baseball.models.positions import IL_SLOTS, Position
from fantasy_baseball.models.standings import ProjectedStandings
from fantasy_baseball.sgp.denominators import get_sgp_denominators
from fantasy_baseball.sgp.player_value import calculate_player_sgp
from fantasy_baseball.utils.constants import Category

logger = logging.getLogger(__name__)


@dataclass
class Move:
    """A single roster transaction for one player."""

    name: str
    player_type: str
    from_slot: str
    to_slot: str  # active slot label, "BN", or "DROP"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class MovePlan:
    """One complete plan: the forced drop(s) plus the resulting move list."""

    drops: list[str]
    moves: list[Move]
    delta_roto: float
    band: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return {
            "drops": list(self.drops),
            "moves": [m.to_dict() for m in self.moves],
            "delta_roto": round(self.delta_roto, 2),
            "band": self.band,
        }


@dataclass
class IlReturnPlanResult:
    """All plans for activating a chosen set of IL players."""

    activating: list[str]
    capacity: int
    overflow: int
    plans: list[MovePlan] = field(default_factory=list)
    warning: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "activating": list(self.activating),
            "capacity": self.capacity,
            "overflow": self.overflow,
            "plans": [p.to_dict() for p in self.plans],
            "warning": self.warning,
        }


def roster_capacity(roster_slots: dict[str, int]) -> int:
    """Active + bench slot count -- every slot except IL slots.

    IL slots are exempt from Yahoo's active-roster size limit, so they do
    not count toward the body-count cap that forces a drop.
    """
    total = 0
    for key, count in roster_slots.items():
        pos = key if isinstance(key, Position) else Position.parse(key)
        if pos in IL_SLOTS:
            continue
        total += count
    return total


def _counts_against_cap(p: Player) -> bool:
    """True if this body counts against the active+bench cap.

    Slot-based, NOT status-based: a BN+IL-status player (Yahoo lets you
    stash an IL guy on the bench) still counts; only a true IL-slot body
    is exempt. This is why activating an IL-slot player is what forces a
    drop.
    """
    return p.selected_position not in IL_SLOTS
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_lineup/test_il_return_planner.py -v`
Expected: PASS (8 tests).

- [ ] **Step 5: Commit**

```bash
git add src/fantasy_baseball/lineup/il_return_planner.py tests/test_lineup/test_il_return_planner.py
git commit -m "feat(lineup): IL return planner dataclasses + capacity helpers"
```

---

## Task 2: Pool construction and IL clearing

**Files:**
- Modify: `src/fantasy_baseball/lineup/il_return_planner.py`
- Test: `tests/test_lineup/test_il_return_planner.py`

- [ ] **Step 1: Write the failing test**

Append to `tests/test_lineup/test_il_return_planner.py`:

```python
from fantasy_baseball.lineup.il_return_planner import _activate, _build_pool


class TestActivate:
    def test_activate_clears_il_signals(self):
        hader = _pitcher("Hader", slot="IL", status="IL15")
        cleared = _activate(hader)
        assert cleared.status == ""
        assert cleared.selected_position is None
        assert cleared.name == "Hader"
        # Original is untouched (dataclasses.replace returns a copy).
        assert hader.status == "IL15"


class TestBuildPool:
    def test_pool_includes_counted_bodies_plus_returning_il_slot_players(self):
        active = _pitcher("Active", slot="P")
        webb = _pitcher("Webb", slot="BN", status="IL10")     # counts (BN)
        hader = _pitcher("Hader", slot="IL", status="IL15")   # exempt (IL slot)
        parked = _pitcher("Parked", slot="IL", status="IL60")  # not activated
        roster = [active, webb, hader, parked]

        pool = _build_pool(roster, activating_il=[webb, hader])
        names = {p.name for p in pool}
        # Active + Webb (already counted) + Hader (added from IL). Parked excluded.
        assert names == {"Active", "Webb", "Hader"}
        # Activated players have IL signals cleared.
        webb_p = next(p for p in pool if p.name == "Webb")
        hader_p = next(p for p in pool if p.name == "Hader")
        assert webb_p.status == "" and webb_p.selected_position is None
        assert hader_p.status == "" and hader_p.selected_position is None
        # Non-activated active player is unchanged.
        active_p = next(p for p in pool if p.name == "Active")
        assert active_p.selected_position == Position.P
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_lineup/test_il_return_planner.py::TestBuildPool -v`
Expected: FAIL with `ImportError: cannot import name '_activate'`.

- [ ] **Step 3: Write minimal implementation**

Append to `src/fantasy_baseball/lineup/il_return_planner.py`:

```python
def _activate(p: Player) -> Player:
    """Return a copy with IL signals cleared so the optimizer treats the
    player as active-eligible. ``is_on_il()`` and the optimizers would
    otherwise keep excluding a returning IL player."""
    return dataclasses.replace(p, status="", selected_position=None)


def _build_pool(roster: list[Player], activating_il: list[Player]) -> list[Player]:
    """The set of players competing for active/bench slots after activation.

    = current counted bodies (active + healthy bench + any BN+IL-status
    players) UNION the activating players that were in true IL slots.
    Activating players get their IL signals cleared. Unchecked IL players
    stay parked and are excluded.
    """
    activating_names = {p.name for p in activating_il}
    counted = [p for p in roster if _counts_against_cap(p)]
    counted_names = {p.name for p in counted}
    extra = [p for p in activating_il if p.name not in counted_names]
    pool: list[Player] = []
    for p in counted + extra:
        pool.append(_activate(p) if p.name in activating_names else p)
    return pool
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_lineup/test_il_return_planner.py -v`
Expected: PASS (all tests including the new `TestActivate`, `TestBuildPool`).

- [ ] **Step 5: Commit**

```bash
git add src/fantasy_baseball/lineup/il_return_planner.py tests/test_lineup/test_il_return_planner.py
git commit -m "feat(lineup): IL planner pool construction + IL clearing"
```

---

## Task 3: Lineup solver + active/bench extraction

**Files:**
- Modify: `src/fantasy_baseball/lineup/il_return_planner.py`
- Test: `tests/test_lineup/test_il_return_planner.py`

- [ ] **Step 1: Write the failing test**

Append to `tests/test_lineup/test_il_return_planner.py`. Add a hitter helper at the top of the file (next to `_pitcher`), then the test:

```python
from fantasy_baseball.models.player import HitterStats

# add near _pitcher:
def _hitter(name, positions, slot=None, **stats):
    return Player(
        name=name,
        player_type=PlayerType.HITTER,
        positions=[Position.parse(p) for p in positions],
        rest_of_season=HitterStats(
            pa=int(stats.get("ab", 500) * 1.15),
            ab=stats.get("ab", 500),
            h=stats.get("h", 130),
            r=stats.get("r", 70),
            hr=stats.get("hr", 20),
            rbi=stats.get("rbi", 70),
            sb=stats.get("sb", 5),
            avg=stats.get("avg", 0.260),
        ),
        selected_position=Position.parse(slot) if slot else None,
    )


def _good_pitcher(name, **stats):
    return Player(
        name=name,
        player_type=PlayerType.PITCHER,
        positions=[Position.P],
        rest_of_season=PitcherStats(
            ip=stats.get("ip", 180.0), w=stats.get("w", 14.0), k=stats.get("k", 200.0),
            sv=stats.get("sv", 0.0), er=stats.get("er", 56.0), bb=stats.get("bb", 35.0),
            h_allowed=stats.get("h_allowed", 150.0), era=stats.get("era", 2.80),
            whip=stats.get("whip", 1.05),
        ),
    )
```

```python
from fantasy_baseball.lineup.il_return_planner import _solve_lineup
from fantasy_baseball.models.standings import ProjectedStandings

TEAM_NAME = "Test Team"


def _standings():
    base = {
        "R": 800, "HR": 200, "RBI": 800, "SB": 100, "AVG": 0.260,
        "W": 70, "K": 1200, "SV": 50, "ERA": 3.50, "WHIP": 1.20,
        "AB": 5000, "H": 1300, "IP": 1400, "ER": 560, "BB": 420, "H_ALLOWED": 1300,
    }
    return ProjectedStandings.from_json(
        {
            "effective_date": "2026-04-01",
            "teams": [
                {"name": TEAM_NAME, "stats": dict(base)},
                {"name": "Opponent", "stats": {**base, "SV": 30, "ERA": 3.80}},
            ],
        }
    )


class TestSolveLineup:
    def test_solver_returns_active_and_bench(self):
        # 1 OF hitter + 4 pitchers for a 1-OF / 1-P / 1-BN setup.
        hitters = [_hitter("OF1", ["OF"])]
        pitchers = [
            _good_pitcher("Ace", k=220),
            _good_pitcher("Mid", k=150, era=3.5, whip=1.2),
            _good_pitcher("Low", k=120, era=4.0, whip=1.3),
        ]
        slots = {"OF": 1, "P": 1, "BN": 1, "IL": 0}
        h_assign, ps, pb = _solve_lineup(
            hitters + pitchers, slots, _standings(), TEAM_NAME, None, None
        )
        assert len(h_assign) == 1
        assert h_assign[0].name == "OF1"
        assert len(ps) == 1  # one P slot
        assert len(pb) == 2  # two pitchers benched
        # The single active pitcher is one of the three.
        assert ps[0].name in {"Ace", "Mid", "Low"}
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_lineup/test_il_return_planner.py::TestSolveLineup -v`
Expected: FAIL with `ImportError: cannot import name '_solve_lineup'`.

- [ ] **Step 3: Write minimal implementation**

Append to `src/fantasy_baseball/lineup/il_return_planner.py`:

```python
def _solve_lineup(
    pool: list[Player],
    roster_slots: dict[str, int],
    projected_standings: ProjectedStandings,
    team_name: str,
    team_sds: Mapping[str, Mapping[Category, float]] | None,
    fraction_remaining: float | None,
):
    """Run both optimizers over ``pool``; return
    ``(hitter_assignments, pitcher_starters, pitcher_bench)``.

    Pass ``fraction_remaining=None`` through to the optimizers so they skip
    per-starter band computation (the planner computes a plan-level band
    separately).
    """
    hitters = [p for p in pool if p.player_type != PlayerType.PITCHER]
    pitchers = [p for p in pool if p.player_type == PlayerType.PITCHER]
    hitter_assignments = optimize_hitter_lineup(
        hitters=hitters,
        full_roster=pool,
        projected_standings=projected_standings,
        team_name=team_name,
        roster_slots=roster_slots,
        team_sds=team_sds,
        fraction_remaining=None,
    )
    pitcher_starters, pitcher_bench = optimize_pitcher_lineup(
        pitchers=pitchers,
        full_roster=pool,
        projected_standings=projected_standings,
        team_name=team_name,
        slots=roster_slots.get("P", 9),
        team_sds=team_sds,
        fraction_remaining=None,
    )
    return hitter_assignments, pitcher_starters, pitcher_bench
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_lineup/test_il_return_planner.py::TestSolveLineup -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/fantasy_baseball/lineup/il_return_planner.py tests/test_lineup/test_il_return_planner.py
git commit -m "feat(lineup): IL planner lineup solver wrapper"
```

---

## Task 4: Move-list builder

**Files:**
- Modify: `src/fantasy_baseball/lineup/il_return_planner.py`
- Test: `tests/test_lineup/test_il_return_planner.py`

- [ ] **Step 1: Write the failing test**

Append to `tests/test_lineup/test_il_return_planner.py`:

```python
from fantasy_baseball.lineup.il_return_planner import _build_moves


class TestBuildMoves:
    def test_moves_capture_activation_bench_and_drop(self):
        active = _pitcher("Active", slot="P")
        webb = _pitcher("Webb", slot="BN", status="IL10")
        hader = _pitcher("Hader", slot="IL", status="IL15")
        scrub = _pitcher("Scrub", slot="P")
        roster = [active, webb, hader, scrub]
        pool = _build_pool(roster, [webb, hader])  # Active, Webb, Hader (cleared); Scrub counts too
        # Note: Scrub is on a P slot so it counts; add it to the pool explicitly
        # to mirror the real pool (all counted bodies). _build_pool already
        # includes it because _counts_against_cap(Scrub) is True.
        assert {p.name for p in pool} == {"Active", "Webb", "Hader", "Scrub"}

        # Pretend the solver put Active + Webb active in P and benched Hader,
        # and we dropped Scrub.
        from fantasy_baseball.lineup.optimizer import HitterAssignment, PitcherStarter

        active_player = next(p for p in pool if p.name == "Active")
        webb_player = next(p for p in pool if p.name == "Webb")
        pitcher_starters = [
            PitcherStarter(name="Active", player=active_player, roto_delta=0.0),
            PitcherStarter(name="Webb", player=webb_player, roto_delta=0.0),
        ]
        moves = _build_moves(
            roster=roster,
            pool=pool,
            hitter_assignments=[],
            pitcher_starters=pitcher_starters,
            dropped_names={"Scrub"},
        )
        by_name = {m.name: m for m in moves}
        # Webb activates from BN -> P
        assert by_name["Webb"].from_slot == "BN"
        assert by_name["Webb"].to_slot == "P"
        # Hader was IL, not active, not dropped -> goes to BN
        assert by_name["Hader"].from_slot == "IL"
        assert by_name["Hader"].to_slot == "BN"
        # Scrub dropped
        assert by_name["Scrub"].to_slot == "DROP"
        assert by_name["Scrub"].from_slot == "P"
        # Active stays in P -> no move emitted
        assert "Active" not in by_name
        # player_type populated
        assert by_name["Webb"].player_type == "pitcher"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_lineup/test_il_return_planner.py::TestBuildMoves -v`
Expected: FAIL with `ImportError: cannot import name '_build_moves'`.

- [ ] **Step 3: Write minimal implementation**

Append to `src/fantasy_baseball/lineup/il_return_planner.py`:

```python
def _slot_value(p: Player) -> str:
    """The player's current slot label, defaulting to BN when unset."""
    return p.selected_position.value if p.selected_position is not None else "BN"


def _build_moves(
    roster: list[Player],
    pool: list[Player],
    hitter_assignments,
    pitcher_starters,
    dropped_names: set[str],
) -> list[Move]:
    """Build the transaction list for one plan.

    ``from_slot`` is the player's CURRENT slot on ``roster`` (so a returning
    IL player reads as ``IL`` and Webb as ``BN``); ``to_slot`` is the
    assigned active slot, ``BN``, or ``DROP``. Only players whose slot
    changes get a move. Sorted by name for deterministic output.
    """
    orig_slot = {p.name: _slot_value(p) for p in roster}
    type_by_name = {p.name: p.player_type.value for p in pool}

    active_slot: dict[str, str] = {a.name: a.slot.value for a in hitter_assignments}
    for s in pitcher_starters:
        active_slot[s.name] = "P"

    moves: list[Move] = []
    for p in pool:
        name = p.name
        frm = orig_slot.get(name, "BN")
        if name in dropped_names:
            to = "DROP"
        elif name in active_slot:
            to = active_slot[name]
        else:
            to = "BN"
        if frm != to:
            moves.append(
                Move(
                    name=name,
                    player_type=type_by_name.get(name, ""),
                    from_slot=frm,
                    to_slot=to,
                )
            )
    moves.sort(key=lambda m: m.name)
    return moves
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_lineup/test_il_return_planner.py::TestBuildMoves -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/fantasy_baseball/lineup/il_return_planner.py tests/test_lineup/test_il_return_planner.py
git commit -m "feat(lineup): IL planner move-list builder"
```

---

## Task 5: `plan_il_returns` orchestration (enumerate, score, rank)

**Files:**
- Modify: `src/fantasy_baseball/lineup/il_return_planner.py`
- Test: `tests/test_lineup/test_il_return_planner.py`

- [ ] **Step 1: Write the failing test**

Append to `tests/test_lineup/test_il_return_planner.py`. This is the headline Webb/Hader scenario plus edge cases:

```python
from fantasy_baseball.lineup.il_return_planner import plan_il_returns

SMALL_SLOTS = {
    "C": 1, "1B": 1, "2B": 1, "3B": 1, "SS": 1,
    "OF": 3, "UTIL": 1, "P": 3, "BN": 1, "IL": 2,
}  # 9 hitter + 3 P + 1 BN = capacity 13


def _full_hitters():
    # Nine solid hitters that fill the nine hitter slots.
    specs = [
        ("C1", ["C"]), ("1B1", ["1B"]), ("2B1", ["2B"]), ("3B1", ["3B"]),
        ("SS1", ["SS"]), ("OFa", ["OF"]), ("OFb", ["OF"]), ("OFc", ["OF"]),
        ("UT", ["1B"]),
    ]
    return [_hitter(n, pos, r=75, hr=22, rbi=75, sb=8, avg=0.275, ab=520, h=143)
            for n, pos in specs]


def _webb_hader_roster():
    hitters = _full_hitters()                       # 9 counted
    sp1 = _good_pitcher("SP1", k=160, era=3.4, whip=1.15)
    sp2 = _good_pitcher("SP2", k=155, era=3.5, whip=1.18)
    scrub = _pitcher("Scrub", slot="P")             # weak; ip=60,k=60,era=3.0 default
    sp1.selected_position = Position.P
    sp2.selected_position = Position.P
    webb = _good_pitcher("Webb", k=210, era=2.7, whip=1.02)
    webb.selected_position = Position.BN
    webb.status = "IL10"                            # BN + IL status -> counts
    hader = _good_pitcher("Hader", k=110, sv=35, era=2.4, whip=0.95)
    hader.selected_position = Position.parse("IL")
    hader.status = "IL15"                           # true IL slot -> exempt
    # counted bodies: 9 hitters + SP1 + SP2 + Scrub + Webb = 13 (== capacity)
    return hitters + [sp1, sp2, scrub, webb, hader]


class TestPlanIlReturns:
    def test_webb_hader_forces_one_drop_and_benches_a_pitcher(self):
        roster = _webb_hader_roster()
        webb = next(p for p in roster if p.name == "Webb")
        hader = next(p for p in roster if p.name == "Hader")

        result = plan_il_returns(
            roster, [webb, hader], SMALL_SLOTS,
            projected_standings=_standings(), team_name=TEAM_NAME,
            fraction_remaining=1.0, team_sds=None,
        )

        assert result.overflow == 1, "activating Hader (IL slot) forces exactly one drop"
        assert result.capacity == 13
        assert result.plans, "expected at least one plan"
        assert len(result.plans) <= 5

        top = result.plans[0]
        assert len(top.drops) == 1
        # The clearly-worst arm (Scrub) is the cheapest drop -> top plan.
        assert top.drops == ["Scrub"]

        by_name = {m.name: m for m in top.moves}
        # Both elite returnees end up active in P.
        assert by_name["Webb"].to_slot == "P"
        assert by_name["Hader"].to_slot == "P"
        # Exactly one pitcher gets benched (P -> BN).
        benched = [m for m in top.moves if m.to_slot == "BN" and m.player_type == "pitcher"]
        assert len(benched) == 1
        # Scrub is dropped.
        assert by_name["Scrub"].to_slot == "DROP"

    def test_plans_sorted_by_delta_roto_desc(self):
        roster = _webb_hader_roster()
        webb = next(p for p in roster if p.name == "Webb")
        hader = next(p for p in roster if p.name == "Hader")
        result = plan_il_returns(
            roster, [webb, hader], SMALL_SLOTS,
            projected_standings=_standings(), team_name=TEAM_NAME,
            fraction_remaining=1.0, team_sds=None,
        )
        deltas = [p.delta_roto for p in result.plans]
        assert deltas == sorted(deltas, reverse=True)
        for p in result.plans:
            assert set(p.band.keys()) == {"mean", "sd", "p_positive", "verdict"}

    def test_no_activation_returns_empty(self):
        roster = _webb_hader_roster()
        result = plan_il_returns(
            roster, [], SMALL_SLOTS,
            projected_standings=_standings(), team_name=TEAM_NAME,
            fraction_remaining=1.0, team_sds=None,
        )
        assert result.plans == []
        assert result.activating == []

    def test_open_bench_means_no_drop(self):
        # Drop Scrub from the roster so there's an open body slot: now
        # activating Hader fills it with no forced drop.
        roster = [p for p in _webb_hader_roster() if p.name != "Scrub"]
        webb = next(p for p in roster if p.name == "Webb")
        hader = next(p for p in roster if p.name == "Hader")
        result = plan_il_returns(
            roster, [webb, hader], SMALL_SLOTS,
            projected_standings=_standings(), team_name=TEAM_NAME,
            fraction_remaining=1.0, team_sds=None,
        )
        assert result.overflow == 0
        assert len(result.plans) == 1
        assert result.plans[0].drops == []
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_lineup/test_il_return_planner.py::TestPlanIlReturns -v`
Expected: FAIL with `ImportError: cannot import name 'plan_il_returns'`.

- [ ] **Step 3: Write minimal implementation**

Append to `src/fantasy_baseball/lineup/il_return_planner.py`:

```python
def _sgp(p: Player, denoms) -> float:
    if p.rest_of_season is None:
        return 0.0
    return calculate_player_sgp(p.rest_of_season, denoms)


def _make_plan(
    roster: list[Player],
    pool: list[Player],
    dropset: tuple[Player, ...],
    base_h,
    base_ps,
    base_pb,
    before_active: list[Player],
    roster_slots: dict[str, int],
    projected_standings: ProjectedStandings,
    team_name: str,
    team_sds,
    fraction_remaining: float,
    bn_slots: int,
) -> MovePlan | None:
    """Solve one drop-set into a MovePlan, or None if infeasible.

    Only re-solves the side(s) the drop touches; the untouched side reuses
    the pre-drop baseline (the lineup there is identical). Feasibility:
    the benched survivors must fit in the BN slots.
    """
    drop_names = {p.name for p in dropset}
    survivors = [p for p in pool if p.name not in drop_names]
    dropped_hitter = any(p.player_type != PlayerType.PITCHER for p in dropset)
    dropped_pitcher = any(p.player_type == PlayerType.PITCHER for p in dropset)

    # Re-solve only the side(s) the drop touches; reuse the pre-drop baseline
    # for the untouched side (its lineup is unchanged). An empty dropset
    # (overflow <= 0) touches neither and reuses both baselines.
    if dropped_hitter:
        h_assign = optimize_hitter_lineup(
            hitters=[p for p in survivors if p.player_type != PlayerType.PITCHER],
            full_roster=survivors,
            projected_standings=projected_standings,
            team_name=team_name,
            roster_slots=roster_slots,
            team_sds=team_sds,
            fraction_remaining=None,
        )
    else:
        h_assign = base_h

    if dropped_pitcher:
        ps, pb = optimize_pitcher_lineup(
            pitchers=[p for p in survivors if p.player_type == PlayerType.PITCHER],
            full_roster=survivors,
            projected_standings=projected_standings,
            team_name=team_name,
            slots=roster_slots.get("P", 9),
            team_sds=team_sds,
            fraction_remaining=None,
        )
    else:
        ps, pb = base_ps, base_pb

    active_names = {a.name for a in h_assign} | {s.name for s in ps}
    benched = [p for p in survivors if p.name not in active_names]
    if len(benched) > bn_slots:
        return None  # infeasible: can't bench everyone left over

    after_active = [a.player for a in h_assign] + [s.player for s in ps]
    try:
        band = compute_delta_roto_band(
            before_active,
            after_active,
            projected_standings.field_stats(team_name),
            team_name,
            fraction_remaining,
            projected_standings=projected_standings,
            team_sds=team_sds,
        )
    except KeyError as exc:
        logger.warning("IL plan band failed for drop %s: %s", sorted(drop_names), exc)
        return None

    moves = _build_moves(roster, pool, h_assign, ps, drop_names)
    return MovePlan(
        drops=sorted(drop_names),
        moves=moves,
        delta_roto=band.mean,
        band=band.to_dict(),
    )


def plan_il_returns(
    roster: list[Player],
    activating_il: list[Player],
    roster_slots: dict[str, int],
    *,
    projected_standings: ProjectedStandings,
    team_name: str,
    fraction_remaining: float,
    team_sds: Mapping[str, Mapping[Category, float]] | None = None,
    max_plans: int = 5,
) -> IlReturnPlanResult:
    """Plan the roster moves to reactivate ``activating_il`` players.

    Returns up to ``max_plans`` plans ranked by deltaRoto descending. Each
    plan's deltaRoto is the cost of its forced drop relative to the pre-drop
    ideal lineup (which already includes the returning players, so the
    activation gain the standings already price is not double-counted).
    """
    capacity = roster_capacity(roster_slots)
    activating_names = [p.name for p in activating_il]

    if not activating_il:
        return IlReturnPlanResult(activating=[], capacity=capacity, overflow=0, plans=[])

    pool = _build_pool(roster, activating_il)
    overflow = len(pool) - capacity
    denoms = get_sgp_denominators()
    bn_slots = roster_slots.get("BN", 0)

    # Pre-drop ideal lineup -> the band baseline (returning players present here).
    base_h, base_ps, base_pb = _solve_lineup(
        pool, roster_slots, projected_standings, team_name, team_sds, fraction_remaining
    )
    before_active = [a.player for a in base_h] + [s.player for s in base_ps]

    if overflow <= 0:
        plan = _make_plan(
            roster, pool, (), base_h, base_ps, base_pb, before_active,
            roster_slots, projected_standings, team_name, team_sds,
            fraction_remaining, bn_slots,
        )
        plans = [plan] if plan is not None else []
        return IlReturnPlanResult(
            activating=activating_names, capacity=capacity, overflow=0, plans=plans
        )

    # Forced drops: enumerate drop-sets of size `overflow`. For overflow >= 3
    # (rare) restrict to the bottom-12 bodies by SGP to bound the combinatorics.
    droppable = pool
    if overflow >= 3:
        droppable = sorted(pool, key=lambda p: _sgp(p, denoms))[:12]

    scored: list[MovePlan] = []
    for dropset in combinations(droppable, overflow):
        plan = _make_plan(
            roster, pool, dropset, base_h, base_ps, base_pb, before_active,
            roster_slots, projected_standings, team_name, team_sds,
            fraction_remaining, bn_slots,
        )
        if plan is not None:
            scored.append(plan)

    if not scored:
        return IlReturnPlanResult(
            activating=activating_names,
            capacity=capacity,
            overflow=overflow,
            plans=[],
            warning=f"No legal roster after dropping {overflow} player(s).",
        )

    # Rank by deltaRoto; tie-break by dropping the lower-SGP body.
    name_to_player = {p.name: p for p in pool}

    def _dropped_sgp(plan: MovePlan) -> float:
        return sum(_sgp(name_to_player[n], denoms) for n in plan.drops if n in name_to_player)

    scored.sort(key=lambda p: (p.delta_roto, -_dropped_sgp(p)), reverse=True)
    return IlReturnPlanResult(
        activating=activating_names,
        capacity=capacity,
        overflow=overflow,
        plans=scored[:max_plans],
    )
```

Note: for the common `overflow == 1` case exactly one side is re-solved (the side matching the dropped player's type) and the other reuses the baseline. `base_pb` (baseline pitcher bench) is consumed only when no pitcher is dropped.

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_lineup/test_il_return_planner.py -v`
Expected: PASS (all classes). If `test_webb_hader_*` fails on the exact drop identity, print `[(p.drops, p.delta_roto) for p in result.plans]` and confirm Scrub is the deltaRoto-max drop; the scenario is built so Scrub (ip=60, k=60, era=3.0) is strictly the weakest arm and dropping the elite/mid arms scores strictly lower. Do NOT weaken the assertion — if Scrub is not the top drop, the bug is in the planner (likely the before/after band sets), fix the code.

- [ ] **Step 5: Commit**

```bash
git add src/fantasy_baseball/lineup/il_return_planner.py tests/test_lineup/test_il_return_planner.py
git commit -m "feat(lineup): plan_il_returns orchestration with deltaRoto ranking"
```

---

## Task 6: Flask JSON route `/api/il-return-plan`

**Files:**
- Modify: `src/fantasy_baseball/web/season_routes.py`
- Test: `tests/test_web/test_season_routes.py`

- [ ] **Step 1: Read the existing route-test harness**

Run: `pytest tests/test_web/test_season_routes.py -v --collect-only` and open `tests/test_web/test_season_routes.py`. Identify (a) the Flask test-client fixture, (b) how a test seeds `CacheKey.ROSTER` and `CacheKey.PROJECTIONS` (look for `write_cache` usage), and (c) how auth is bypassed (the `_global_auth_gate` before_request). Mirror that exact setup in the new test below. Also confirm the helper names `_load_config`, `_projected_from_cache`, `_team_sds_from_cache` exist in `season_routes.py` (they are used by the trade-builder route around line 831-952).

- [ ] **Step 2: Write the failing test**

Append a test to `tests/test_web/test_season_routes.py` mirroring the harness found in Step 1. The body (adapt the client/seed fixtures to match the file's existing pattern):

```python
def test_il_return_plan_route_returns_plans(client, seed_cache):
    # seed_cache: mirror however other tests write CacheKey.ROSTER (list of
    # Player dicts) and CacheKey.PROJECTIONS ({"projected_standings", "team_sds",
    # "fraction_remaining"}). Build a roster with one IL-slot pitcher so the
    # route has something to activate. Reuse this file's existing roster/standings
    # builders if present; otherwise construct minimal Player.to_dict() payloads.
    resp = client.get("/api/il-return-plan?activate=" )  # no ids -> activate all IL
    assert resp.status_code in (200, 404)
    if resp.status_code == 200:
        data = resp.get_json()
        assert set(data.keys()) >= {"activating", "capacity", "overflow", "plans"}
        assert isinstance(data["plans"], list)


def test_il_return_plan_route_404_without_data(client):
    # With no cache seeded, the route reports missing data rather than 500.
    resp = client.get("/api/il-return-plan?activate=abc")
    assert resp.status_code == 404
```

If the file already has a fixture that seeds a full roster + projections (most route tests do), prefer asserting the concrete shape: `data["capacity"] >= 1` and, when an IL pitcher is present, `data["overflow"] >= 0` and each plan has `drops`, `moves`, `delta_roto`, `band` keys.

- [ ] **Step 3: Run test to verify it fails**

Run: `pytest tests/test_web/test_season_routes.py -k il_return_plan -v`
Expected: FAIL with 404 on the first test (route not registered) — actually a Flask 404 page, so `resp.get_json()` is None; the assertion `status_code in (200,404)` may pass spuriously. To force a real RED, first assert the route exists: add `assert b"plans" in resp.data or resp.status_code == 404` only AFTER implementing; for the RED step assert `resp.status_code == 200` against a seeded fixture so it fails because the route is missing (returns 404). Confirm the failure is "route not found", not a harness error.

- [ ] **Step 4: Write minimal implementation**

In `src/fantasy_baseball/web/season_routes.py`, inside `register_routes(app)` (next to the other `@app.route` closures, e.g. near the `/roster-audit` route), add:

```python
    @app.route("/api/il-return-plan")
    def api_il_return_plan():
        from fantasy_baseball.lineup.il_return_planner import plan_il_returns
        from fantasy_baseball.models.player import Player

        activate_param = request.args.get("activate", "")
        activate_ids = {a for a in activate_param.split(",") if a}

        roster_raw = read_cache_list(CacheKey.ROSTER)
        if not roster_raw:
            return jsonify({"error": "No roster data. Run a refresh first."}), 404
        proj_cache = read_cache_dict(CacheKey.PROJECTIONS) or {}
        ps_raw = proj_cache.get("projected_standings")
        if not ps_raw:
            return jsonify({"error": "No projected standings. Run a refresh first."}), 404

        config = _load_config()
        roster = [Player.from_dict(p) for p in roster_raw]
        projected = _projected_from_cache(ps_raw)
        team_sds = _team_sds_from_cache(proj_cache.get("team_sds"))
        fr = proj_cache.get("fraction_remaining")
        fr = 1.0 if fr is None else float(fr)

        il_players = [p for p in roster if p.is_on_il()]
        if activate_ids:
            activating = [
                p for p in il_players if (p.yahoo_id or p.name) in activate_ids
            ]
        else:
            activating = il_players

        result = plan_il_returns(
            roster,
            activating,
            config.roster_slots,
            projected_standings=projected,
            team_name=config.team_name,
            fraction_remaining=fr,
            team_sds=team_sds,
        )
        return jsonify(result.to_dict())
```

Confirm `_load_config`, `_projected_from_cache`, `_team_sds_from_cache`, `read_cache_list`, `read_cache_dict`, `CacheKey`, `request`, `jsonify` are all already imported/defined in the module (they are, per the trade-builder route). If `config.team_name` / `config.roster_slots` attribute names differ, match what the trade route uses (`config.roster_slots`, `config.team_name`).

- [ ] **Step 5: Run test to verify it passes**

Run: `pytest tests/test_web/test_season_routes.py -k il_return_plan -v`
Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add src/fantasy_baseball/web/season_routes.py tests/test_web/test_season_routes.py
git commit -m "feat(web): /api/il-return-plan route for IL return planner"
```

---

## Task 7: "IL Returns" section on the roster-audit page

**Files:**
- Modify: `src/fantasy_baseball/web/templates/season/roster_audit.html`

- [ ] **Step 1: Read the template**

Open `src/fantasy_baseball/web/templates/season/roster_audit.html`. Confirm it `{% extends "season/base.html" %}`, the `{% block content %}` boundaries, and that IL players appear in `audit` as entries with `slot == "IL"` and a `player_id` field. Find the end of the audit `<table>` inside the content block — the new section goes after it, before `{% endblock %}`.

- [ ] **Step 2: Add the section markup**

Inside the `content` block, after the closing `</table>` (and before `{% include "season/_stat_cell_tooltip.html" %}` if that include is last), insert:

```html
<section class="il-returns" id="il-returns" style="margin-top:2rem">
  <h2>IL Returns</h2>
  {% set il_players = audit | selectattr('slot', 'equalto', 'IL') | list %}
  {% if not il_players %}
    <p style="color:var(--text-secondary)">No IL players on your roster.</p>
  {% else %}
    <p style="color:var(--text-secondary)">
      Check the players returning this week to see the best drop/start/bench plans.
    </p>
    <div class="il-checkboxes" style="display:flex;gap:1rem;flex-wrap:wrap;margin-bottom:1rem">
      {% for e in il_players %}
      <label><input type="checkbox" class="il-activate"
                    value="{{ e.player_id }}" data-name="{{ e.player }}">
        {{ e.player }}</label>
      {% endfor %}
    </div>
    <div id="il-plan-results"><p style="color:var(--text-secondary)">
      Select returning players to see move plans.</p></div>
  {% endif %}
</section>
```

- [ ] **Step 3: Add the fetch/render JS**

At the end of the `content` block (after the section), add a `<script>` block. It replicates the `band_cell` verdict->class mapping in JS (the Jinja macro can't run on dynamically fetched data). ASCII only (`->`, not an arrow glyph):

```html
<script>
(function () {
  var boxes = document.querySelectorAll('.il-activate');
  if (!boxes.length) return;
  boxes.forEach(function (cb) { cb.addEventListener('change', refresh); });

  function gapClass(v) {
    if (v === 'real') return 'gap-positive';
    if (v === 'downgrade') return 'gap-negative';
    return 'gap-marginal';
  }
  function refresh() {
    var ids = Array.prototype.slice.call(document.querySelectorAll('.il-activate:checked'))
      .map(function (c) { return c.value; });
    var out = document.getElementById('il-plan-results');
    if (!ids.length) {
      out.innerHTML = '<p style="color:var(--text-secondary)">Select returning players to see move plans.</p>';
      return;
    }
    out.innerHTML = '<p>Computing...</p>';
    fetch('/api/il-return-plan?activate=' + encodeURIComponent(ids.join(',')))
      .then(function (r) { return r.json(); })
      .then(render)
      .catch(function (e) { out.innerHTML = '<p>Error: ' + e + '</p>'; });
  }
  function render(res) {
    var out = document.getElementById('il-plan-results');
    if (res.error) { out.innerHTML = '<p>' + res.error + '</p>'; return; }
    if (res.warning) { out.innerHTML = '<p>' + res.warning + '</p>'; return; }
    if (!res.plans || !res.plans.length) {
      out.innerHTML = '<p>No legal plans found.</p>'; return;
    }
    var html = '<p style="color:var(--text-secondary)">Roster cap ' + res.capacity +
      '; must drop ' + res.overflow + '.</p>';
    res.plans.forEach(function (p) {
      var b = p.band;
      var sign = b.mean >= 0 ? '+' : '';
      html += '<div class="il-plan" style="border:1px solid var(--border);border-radius:6px;padding:0.75rem;margin-bottom:0.75rem">';
      html += '<div><span class="gap-badge ' + gapClass(b.verdict) + '">' +
        sign + b.mean.toFixed(1) + '</span> ' +
        'Std dev ' + b.sd.toFixed(2) + ' &middot; P(helps) ' +
        Math.round(b.p_positive * 100) + '%</div>';
      html += '<ul style="margin:0.5rem 0 0;padding-left:1.25rem">';
      p.moves.forEach(function (m) {
        html += '<li>' + m.name + ': ' + m.from_slot + ' -&gt; ' + m.to_slot + '</li>';
      });
      html += '</ul></div>';
    });
    out.innerHTML = html;
  }
})();
</script>
```

- [ ] **Step 4: Verify the page renders**

Run the season app and load `/roster-audit` (requires seeded cache from a refresh, or reuse an existing local cache). Confirm: the "IL Returns" section appears; checking an IL player triggers a fetch and renders plan cards with the deltaRoto badge / Std dev / P(helps) line and a `Name: FROM -> TO` move list. If no local cache is available, instead verify via the route test from Task 6 plus a Jinja render smoke check (`pytest tests/test_web/test_season_routes.py -k roster_audit`).

- [ ] **Step 5: Commit**

```bash
git add src/fantasy_baseball/web/templates/season/roster_audit.html
git commit -m "feat(web): IL Returns section on roster-audit page"
```

---

## Task 8: Full verification

**Files:** none (verification only)

- [ ] **Step 1: Run the targeted test suites**

Run:
```bash
pytest tests/test_lineup/test_il_return_planner.py tests/test_web/test_season_routes.py tests/test_lineup/test_roster_audit.py -v
```
Expected: all PASS. (Including roster-audit confirms no regression in the shared optimizer/band code.)

- [ ] **Step 2: Lint**

Run: `ruff check .`
Expected: zero violations. Fix any (unused imports are the usual offender — e.g. if the mixed-drop guard left an import unused).

- [ ] **Step 3: Format check**

Run: `ruff format --check .`
Expected: no drift. If it reports the new files, run `ruff format .` and re-commit.

- [ ] **Step 4: Dead-code check**

Run: `vulture src/fantasy_baseball/lineup/il_return_planner.py`
Expected: no NEW findings. The module's public surface is `plan_il_returns` + the dataclasses; helpers prefixed `_` are all referenced. Note any pre-existing unrelated findings without fixing them.

- [ ] **Step 5: Type check (if covered)**

Check `pyproject.toml` `[tool.mypy].files`. If `src/fantasy_baseball/lineup/` (or the new module) is in scope, run `mypy src/fantasy_baseball/lineup/il_return_planner.py` and fix any errors. If the lineup package is not yet under mypy, state that it is out of mypy scope and skip.

- [ ] **Step 6: Run the broader lineup + web suite once**

Run: `pytest tests/test_lineup/ tests/test_web/ -q`
Expected: all PASS. Paste a concise summary (counts) as evidence.

- [ ] **Step 7: Final commit (if any fixes were made)**

```bash
git add -A
git commit -m "chore(lineup): lint/format/type fixes for IL return planner"
```

---

## Self-Review notes (for the implementer)

- **Spec coverage:** Task 1 = capacity/overflow + dataclasses; Task 2 = pool + IL clearing (slot-based cap, the Webb-counts/Hader-doesn't distinction); Tasks 3-5 = displacement-safe scoring via the pre-drop-ideal band baseline + ranking + move list; Task 6 = on-demand JSON route (checkbox-driven); Task 7 = top-5 plans with deltaRoto/std-dev/P(helps) styled like the audit. Edge cases (overflow<=0, infeasible drop -> warning, activating a hitter, no-IL) are covered by Task 5 tests and the `_make_plan` feasibility guard.
- **No double-count:** every plan's band uses `before_active` = the pre-drop ideal lineup, which already contains the returning players; they cancel in `_swap_sets`, so deltaRoto reflects only the forced drop. This realizes the spec's "displacement-aware" decision.
- **Determinism:** moves sorted by name; plans tie-broken by dropping lower SGP. The Webb/Hader fixture makes `Scrub` the unique weakest arm so `drops == ["Scrub"]` is stable.
- **Type consistency:** `Move`/`MovePlan`/`IlReturnPlanResult` field names and `to_dict` keys are identical across module, route (`result.to_dict()`), and JS (`p.band.mean/sd/p_positive/verdict`, `m.from_slot/to_slot/name`).
