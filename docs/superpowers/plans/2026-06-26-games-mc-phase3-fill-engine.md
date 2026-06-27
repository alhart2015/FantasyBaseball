# Games-based MC -- Phase 3 (pure bench-injury-fill allocation engine, hitters) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans. Steps use checkbox (`- [ ]`).

**Goal:** Deliver the PURE bench-injury-fill ALLOCATION algorithm as a standalone, fully unit-testable function. Given an already-built `EffectiveRoster` (Phase 2) plus, per body, an already-sampled per-game counting line and a per-active-body `frac_missed` (all supplied by the CALLER -- Phase 4 will feed these from the sampler), the function returns ONLY the FILL contributions (bench + replacement) to add on top of each active body's own realized counting. No sampler surgery in this phase.

**Architecture:** New module `src/fantasy_baseball/mc_fill.py`. Phase 3 does NOT touch `_apply_variance_batch` or `simulate_remaining_season_batch` -- all plumbing (extracting `frac_missed`/`scales`, suppressing the built-in `repl_contrib`, ROS-direct wiring) is deferred to Phase 4. The function is pure: it takes concrete numbers in, returns concrete fill numbers out, so every spec mechanism is testable on hand-built inputs without running the batch.

**Tech Stack:** Python, pytest. New file: `src/fantasy_baseball/mc_fill.py`; test `tests/test_mc_fill.py`. Reuses `mc_roster.{ActiveBody, BenchBody, EffectiveRoster, PA_PER_GAME}`, `simulation._replacement_line`, `utils.constants.HITTING_COUNTING`, `models.positions.Position`.

## Global Constraints

- ASCII-only in source/strings. Numeric defaults via `is not None`, never `x or default` (the falsy-zero footgun -- a real `0.0` per-game stat must survive).
- All imports at top of module.
- Spec: `docs/superpowers/specs/2026-06-26-games-based-availability-mc-design.md` -- the "Missed playing-time accounting (single authority)" section and Component 3 (value rule, one-body capacity, replacement-last, rate-stat handling, `g_ros_full` vs `g_ros_adj`). Binding.
- **`g_ros_adj` (= `factor * g_ros_full`, the displacement-reduced baseline) is the games-missed multiplier AND the active body's fill cap. NEVER use `g_ros_full` as the multiplier** -- that over-seats a displaced (shared) slot. `g_ros_full` is ONLY the per-game-VALUE denominator, which Phase 2 already baked into `BenchBody.per_game_value`; this function does not recompute per-game value.
- Reuse the shared `PA_PER_GAME` from `mc_roster` (already `4.3`) for the replacement per-game conversion -- the replacement line has NO games field of its own, so divide its per-stat totals by `PA_PER_GAME` (one constant, not a second).
- New module `src/fantasy_baseball/mc_fill.py` is ALREADY listed under `[tool.mypy].files` in `pyproject.toml` (verified line 88 carries `mc_roster.py`; add the `mc_fill.py` line). Run mypy on it.
- Hitters ONLY this phase. Pitcher bench-fill is deferred (Phase 5). Counting columns: `HITTING_COUNTING = ["r", "hr", "rbi", "sb", "h", "ab"]` (verified `utils/constants.py:127`). Rate-stat components `h`, `ab` flow through identically as counting columns -- no separate rate handling in the fill.
- **Test assertions MUST be MECHANISM-ONLY.** Do NOT assert absolute magnitudes that depend on the `frac_missed` distribution (it changes in Phase 4). Allowed: nonzero-vs-zero fill, ordering picks the better body, capacity never exceeded, residual routes to replacement, the conservation bound (a displaced body's fill <= `g_ros_adj`-derived games). Pin the `g_ros_adj` (not `g_ros_full`) multiplier with a conservation assertion.

---

### Task 1: Input/output dataclasses + the pure `allocate_bench_fill` function

**Files:** Create `src/fantasy_baseball/mc_fill.py`; test `tests/test_mc_fill.py`. Modify `pyproject.toml` ([tool.mypy].files).

**Interfaces (the chosen signature -- pure, per-team, per-iteration):**

```python
# Per-active-body sampled inputs for ONE iteration (the caller assembles these).
@dataclass(frozen=True)
class ActiveSample:
    body: ActiveBody                     # carries factor + g_ros_adj (the cap/multiplier)
    frac_missed: float                   # stochastic shortfall fraction, max(0, 1 - scale)

# Per-bench-body sampled per-game counting line for ONE iteration.
@dataclass(frozen=True)
class BenchSample:
    body: BenchBody                      # carries g_ros_full, per_game_value, eligible_positions
    per_game_counts: dict[str, float]    # sampled counting stats PER GAME (HITTING_COUNTING keys)

@dataclass(frozen=True)
class FillResult:
    # Total FILL counting contributions (bench + replacement) to ADD on top of the
    # active bodies' OWN realized counting. Keyed by HITTING_COUNTING column.
    fill_counts: dict[str, float]

def allocate_bench_fill(
    actives: list[ActiveSample],
    benches: list[BenchSample],
    replacement_for: Callable[[ActiveBody], dict[str, float]],
) -> FillResult: ...
```

- `replacement_for` is injected (the caller passes a closure over `simulation._replacement_line(player.to_flat_dict_full_season(), is_hitter=True)`), so this module stays pure and does not import the sampler. It returns the per-stat replacement TOTAL line (no games field); the fill converts to per-game by dividing by `PA_PER_GAME`.
- The active body's OWN realized counting is NOT included here -- the caller (Phase 4) adds it. This function returns only the fill on top.

**Algorithm (per the spec Component 3, hitters):**
1. For each active sample: `games_missed = frac_missed * body.g_ros_adj` (the reduced baseline -- the cap; NEVER `g_ros_full`). Skip if `games_missed <= 0`.
2. Process active shortfalls LARGEST-first (by `games_missed`). For each, among bench samples that (a) share a real eligible position with the active body and (b) have remaining games, pick the highest `per_game_value`; tie-break by player-id ascending. Assign `min(shortfall_games, bench_remaining_games)`; that bench body contributes `assigned_games * bench.per_game_counts[col]` for every counting col. Decrement both the shortfall and the bench body's remaining-games pool. One-body capacity: a bench body's TOTAL assigned games across all shortfalls <= its `g_ros_full`. Loop until the shortfall is covered or no eligible bench body has remaining games.
3. Residual shortfall games (no eligible bench left) -> replacement: `replacement_line[col] / PA_PER_GAME * residual_games` per counting col.
4. Sum every contribution into `fill_counts`.

Position eligibility: `bench.body.eligible_positions & active_eligible_positions` non-empty, where active eligible positions = `body.player`'s real positions. Reuse the precomputed `BenchBody.eligible_positions` (a `frozenset[Position]`, from Phase 2's `_real_positions`); for the active body, compute its real positions once via the same `scoring._real_positions` (verified `scoring.py:223`, returns `frozenset(p.positions) - _GENERIC_SLOTS`). Keep it a Python loop (<=12 active, <=2 bench).

- [ ] **Step 1: Write the failing tests** in `tests/test_mc_fill.py`. Build `ActiveBody`/`BenchBody` directly (frozen dataclasses from `mc_roster`) -- no need to run `build_effective_roster`. Use a tiny `Player` factory for positions. ALL assertions mechanism-only.

```python
import pytest

from fantasy_baseball.mc_roster import ActiveBody, BenchBody
from fantasy_baseball.mc_fill import (
    ActiveSample,
    BenchSample,
    allocate_bench_fill,
)
from fantasy_baseball.models.player import Player, PlayerType
from fantasy_baseball.models.positions import Position
from fantasy_baseball.utils.constants import HITTING_COUNTING


def _player(name, pid, pos=Position.OF):
    return Player(name=name, player_type=PlayerType.HITTER, positions=[pos],
                  selected_position=pos, yahoo_id=pid)


def _active(name, pid, g_ros_adj, factor=1.0, pos=Position.OF):
    return ActiveBody(player=_player(name, pid, pos), factor=factor, g_ros_adj=g_ros_adj)


def _bench(name, pid, g_ros_full, per_game_value, pos=Position.OF):
    return BenchBody(player=_player(name, pid, pos), g_ros_full=g_ros_full,
                     per_game_value=per_game_value, eligible_positions=frozenset({pos}))


def _line(**kw):
    return {c: float(kw.get(c, 0.0)) for c in HITTING_COUNTING}


def _bench_sample(b, per_game):
    return BenchSample(body=b, per_game_counts=_line(**per_game))


def _no_replacement(_active_body):
    return _line()  # zero replacement -> isolates bench-fill mechanism


def _flat_replacement(val):
    return lambda _b: _line(**{c: val for c in HITTING_COUNTING})


def test_eligible_bench_gets_nonzero_fill_on_low_availability():
    a = _active("Star", "1", g_ros_adj=80.0)
    b = _bench("Depth", "2", g_ros_full=60.0, per_game_value=2.0)
    res = allocate_bench_fill(
        [ActiveSample(a, frac_missed=0.5)],
        [_bench_sample(b, {"r": 0.5, "h": 1.0, "ab": 4.0})],
        _no_replacement,
    )
    assert res.fill_counts["r"] > 0  # eligible bench fills an injured starter


def test_full_availability_yields_zero_fill():
    a = _active("Star", "1", g_ros_adj=80.0)
    b = _bench("Depth", "2", g_ros_full=60.0, per_game_value=2.0)
    res = allocate_bench_fill(
        [ActiveSample(a, frac_missed=0.0)],
        [_bench_sample(b, {"r": 0.5})],
        _no_replacement,
    )
    assert all(v == 0.0 for v in res.fill_counts.values())  # no injury -> no fill


def test_position_mismatch_routes_to_replacement_not_bench():
    a = _active("OFstar", "1", g_ros_adj=80.0, pos=Position.OF)
    b = _bench("Catcher", "2", g_ros_full=60.0, per_game_value=2.0, pos=Position.C)
    res = allocate_bench_fill(
        [ActiveSample(a, frac_missed=0.5)],
        [_bench_sample(b, {"r": 99.0})],            # huge bench rate, but wrong position
        _flat_replacement(0.1),
    )
    # bench (C) cannot fill an OF shortfall -> all fill is replacement, not the
    # catcher's 99-per-game line.
    assert res.fill_counts["r"] < 1.0


def test_fill_never_exceeds_bench_g_ros_full_capacity():
    # Two OF starters both injured; one bench body eligible for both. Its total
    # contributed games cannot exceed its g_ros_full -> bounded total fill.
    a1 = _active("S1", "1", g_ros_adj=100.0, pos=Position.OF)
    a2 = _active("S2", "2", g_ros_adj=100.0, pos=Position.OF)
    cap = 10.0
    b = _bench("Depth", "3", g_ros_full=cap, per_game_value=2.0, pos=Position.OF)
    res = allocate_bench_fill(
        [ActiveSample(a1, frac_missed=1.0), ActiveSample(a2, frac_missed=1.0)],
        [_bench_sample(b, {"r": 1.0})],
        _no_replacement,                            # zero replacement -> only bench contributes
    )
    # bench gives 1 r/game, capped at cap games -> bench-only fill r <= cap.
    assert res.fill_counts["r"] <= cap + 1e-9


def test_per_game_value_ordering_picks_better_body():
    a = _active("Star", "1", g_ros_adj=20.0, pos=Position.OF)
    good = _bench("Good", "2", g_ros_full=100.0, per_game_value=5.0, pos=Position.OF)
    bad = _bench("Bad", "3", g_ros_full=100.0, per_game_value=1.0, pos=Position.OF)
    res = allocate_bench_fill(
        [ActiveSample(a, frac_missed=1.0)],
        [_bench_sample(good, {"r": 10.0}), _bench_sample(bad, {"r": 0.0})],
        _no_replacement,
    )
    # both have ample capacity for the 20-game shortfall, so the higher per-game
    # body covers it all -> nonzero r from "good", proving it was chosen first.
    assert res.fill_counts["r"] > 0


def test_residual_goes_to_replacement_when_bench_exhausted():
    a = _active("Star", "1", g_ros_adj=100.0, pos=Position.OF)
    b = _bench("Depth", "2", g_ros_full=5.0, per_game_value=2.0, pos=Position.OF)
    res = allocate_bench_fill(
        [ActiveSample(a, frac_missed=1.0)],          # 100 games missed, bench covers 5
        [_bench_sample(b, {"r": 0.0})],              # bench gives 0 r -> all r must be replacement
        _flat_replacement(0.5),
    )
    assert res.fill_counts["r"] > 0                  # residual 95 games -> replacement r


def test_displaced_body_fill_bounded_by_g_ros_adj_not_g_ros_full():
    # CONSERVATION: a displaced body (factor 0.5, g_ros_full 80 -> g_ros_adj 40)
    # with frac_missed=1.0 misses at most g_ros_adj=40 games, NOT g_ros_full=80.
    # Pin the multiplier: bench gives exactly 1 r/game with ample capacity, so
    # bench-only fill r == games_missed == frac_missed * g_ros_adj == 40, never 80.
    a = _active("Displaced", "1", g_ros_adj=40.0, factor=0.5, pos=Position.OF)
    b = _bench("Depth", "2", g_ros_full=200.0, per_game_value=2.0, pos=Position.OF)
    res = allocate_bench_fill(
        [ActiveSample(a, frac_missed=1.0)],
        [_bench_sample(b, {"r": 1.0})],
        _no_replacement,
    )
    assert abs(res.fill_counts["r"] - 40.0) < 1e-6   # g_ros_adj (40), NOT g_ros_full (80)


def test_tie_break_by_player_id_ascending():
    # Two equal-per-game-value eligible bodies, only enough shortfall for one game.
    # Deterministic: id "2" (ascending) is chosen, contributing its distinctive rate.
    a = _active("Star", "1", g_ros_adj=1.0, pos=Position.OF)   # 1 game missed at frac 1.0
    b_lo = _bench("LowId", "2", g_ros_full=100.0, per_game_value=3.0, pos=Position.OF)
    b_hi = _bench("HighId", "9", g_ros_full=100.0, per_game_value=3.0, pos=Position.OF)
    res = allocate_bench_fill(
        [ActiveSample(a, frac_missed=1.0)],
        [_bench_sample(b_hi, {"r": 0.0}), _bench_sample(b_lo, {"r": 7.0})],
        _no_replacement,
    )
    assert abs(res.fill_counts["r"] - 7.0) < 1e-6    # id "2" (LowId) chosen, gives 7
```

(Before finalizing: confirm `Player` accepts `yahoo_id` as the id field and that `scoring._real_positions` returns the OF/C positions for these constructed players -- read both first and pin the eligibility tests to actual behavior. If `_real_positions` strips a position you used, switch the test to one it keeps, e.g. `Position.OF`/`Position.C`.)

- [ ] **Step 2: Run, confirm FAIL** (`ModuleNotFoundError: fantasy_baseball.mc_fill`).
- [ ] **Step 3: Implement `mc_fill.py`** (all imports at top; ASCII-only; `is not None` for numeric defaults):

```python
"""Pure per-iteration bench-injury-fill allocation (hitters).

Given an EffectiveRoster's active bodies (each with its stochastic frac_missed)
and bench bodies (each with a sampled per-game counting line), allocate each
active body's missed games to eligible bench bodies (highest per-game value
first, one-body capacity), then replacement-level for any residual. Returns ONLY
the FILL contributions to add on top of the active bodies' own realized counting
(the caller adds that). PURE: no sampler import, no globals -- Phase 4 feeds the
sampled inputs. Pitcher fill is deferred (Phase 5)."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

from fantasy_baseball.mc_roster import ActiveBody, BenchBody, PA_PER_GAME
from fantasy_baseball.scoring import _real_positions
from fantasy_baseball.utils.constants import HITTING_COUNTING


@dataclass(frozen=True)
class ActiveSample:
    body: ActiveBody
    frac_missed: float  # max(0, 1 - scale); stochastic shortfall fraction this iter


@dataclass(frozen=True)
class BenchSample:
    body: BenchBody
    per_game_counts: dict[str, float]  # sampled HITTING_COUNTING stats PER GAME


@dataclass(frozen=True)
class FillResult:
    fill_counts: dict[str, float]  # bench + replacement fill to add on top of own


def allocate_bench_fill(
    actives: list[ActiveSample],
    benches: list[BenchSample],
    replacement_for: Callable[[ActiveBody], dict[str, float]],
) -> FillResult:
    """Allocate missed games to bench (value-ordered, capped) then replacement.

    games_missed = frac_missed * g_ros_adj (the reduced baseline -- the cap;
    NEVER g_ros_full). Largest shortfalls first; per shortfall pick the highest
    per_game_value position-eligible bench body with remaining capacity, assign
    min(shortfall, remaining), decrement both; residual -> replacement per-game
    (replacement total / PA_PER_GAME). One bench body's total assigned games
    <= its g_ros_full. Tie-break: higher per_game_value, then player-id ascending.
    """
    fill: dict[str, float] = {col: 0.0 for col in HITTING_COUNTING}

    # Remaining capacity per bench body (mutated as we allocate).
    remaining = {id(bs): bs.body.g_ros_full for bs in benches}

    # Shortfalls, largest games_missed first.
    shortfalls = [
        (a.frac_missed * a.body.g_ros_adj, a)
        for a in actives
        if a.frac_missed * a.body.g_ros_adj > 0.0
    ]
    shortfalls.sort(key=lambda t: t[0], reverse=True)

    for games_missed, a in shortfalls:
        need = games_missed
        active_pos = _real_positions(a.body.player)

        while need > 0.0:
            # Eligible bench bodies with remaining capacity.
            eligible = [
                bs
                for bs in benches
                if remaining[id(bs)] > 0.0 and (bs.body.eligible_positions & active_pos)
            ]
            if not eligible:
                break
            # Highest per-game value, then player-id ascending (deterministic).
            eligible.sort(
                key=lambda bs: (-bs.body.per_game_value, _pid(bs.body)),
            )
            bs = eligible[0]
            assign = min(need, remaining[id(bs)])
            for col in HITTING_COUNTING:
                pg = bs.per_game_counts.get(col, 0.0)
                fill[col] += assign * (pg if pg is not None else 0.0)
            remaining[id(bs)] -= assign
            need -= assign

        if need > 0.0:
            repl = replacement_for(a.body)
            for col in HITTING_COUNTING:
                total = repl.get(col, 0.0)
                per_game = (total / PA_PER_GAME) if total is not None else 0.0
                fill[col] += per_game * need

    return FillResult(fill_counts=fill)


def _pid(b: BenchBody) -> str:
    """Player-id for the deterministic tie-break (ascending). Falls back to the
    name::player_type id when yahoo_id is absent (never bare name)."""
    yid = b.player.yahoo_id
    return str(yid) if yid is not None else f"{b.player.name}::{b.player.player_type}"
```

(`Player.id` does NOT exist -- the fallback uses the canonical `name::player_type` form, never a bare name. `Player.yahoo_id: str | None` exists (player.py:179). Before finalizing: confirm `replacement_for`'s returned dict keys match `HITTING_COUNTING` exactly -- `_replacement_line` returns a `REPLACEMENT_BY_POSITION` entry; verify those entries carry `r/hr/rbi/sb/h/ab` keys.)

- [ ] **Step 4: Run, confirm PASS:** `pytest tests/test_mc_fill.py -v`.
- [ ] **Step 5: ruff + mypy.** `ruff check src/fantasy_baseball/mc_fill.py tests/test_mc_fill.py`; `ruff format --check .`; add `"src/fantasy_baseball/mc_fill.py"` under `[tool.mypy].files` in `pyproject.toml` (next to the `mc_roster.py` line, verified line 88); `mypy src/fantasy_baseball/mc_fill.py` -- expected clean. `vulture` -- confirm no NEW findings (the dataclasses + the public function are referenced by tests; `_pid` is used internally).
- [ ] **Step 6: Commit:**
```bash
git add src/fantasy_baseball/mc_fill.py tests/test_mc_fill.py pyproject.toml
git commit -m "feat(mc): pure bench-injury-fill allocation engine (hitters, Phase 3)"
```

---

## Self-Review

**Spec coverage:** Implements Component 3's allocation only (the pure piece): `games_missed = frac_missed * g_ros_adj` (reduced baseline as cap/multiplier, never `g_ros_full`); largest-first processing; per-game-value ordering with player-id tie-break; one-body capacity via the per-bench remaining-games pool; replacement-last via `_replacement_line / PA_PER_GAME`; rate components `h`/`ab` flow as ordinary counting cols; returns ONLY fill (caller adds the body's own realized counting). Sampler surgery (`_apply_variance_batch` `frac_missed` extraction + `repl_contrib` suppression, ROS-direct) is explicitly deferred to Phase 4 -- the function takes already-sampled inputs so it is testable on concrete numbers now.

**Input shape matches Phase 4's source (contract pinned for Phase 4):** `ActiveSample.frac_missed` is exactly `np.maximum(0.0, 1.0 - scales)` from `_apply_variance_batch:701`; `ActiveBody.g_ros_adj` comes from Phase 2; `replacement_for` wraps `simulation._replacement_line` (verified `simulation.py:435`). `BenchSample.per_game_counts[col]` is DEFINED as `bench_body_ROS_total_counts[col] / g_ros_full` -- the per-game rate (total counting production over its full ROS games). The batch emits per-player TOTALS (`base*scales`), not per-game, so Phase 4 must do this division; whether the "total" is the clean base projection or the per-iteration sampled draw is Phase 4's call (the spec's variance note bears on it -- using the sampled draw adds fill variance; using base makes fill deterministic). Phase 3 takes `per_game_counts` as a given input, so this is a Phase-4 contract note, not a Phase-3 dependency.

**Eligibility edge (stated, inherits ERoto):** `_real_positions` strips generic slots (`UTIL`/`IF`/`DH`), so a hitter with NO real position (pure DH-only -- `positions == [DH]`/`[UTIL]`) yields an empty eligible set and routes its entire shortfall to replacement, never to a bench bat (and a pure-DH bench bat can fill no one). This is the same eligibility ERoto's displacement uses; pure-DH-only hitters are rare and `_replacement_line` has a generic fallback, so it is acceptable. A UTIL-aware eligibility refinement is deferred.

**Mechanism-only assertions:** No test pins an absolute magnitude that depends on the `frac_missed` distribution. The one numeric pin (`== 40.0`) is the CONSERVATION assertion that fails the wrong `g_ros_full` multiplier (would give 80) -- a mechanism check, not a distribution claim. Ordering/capacity/residual/tie-break are all qualitative or capacity-bounded.

**Constants:** Single shared `PA_PER_GAME` reused from `mc_roster` (no second constant). `is not None` guards every numeric default. ASCII-only. New module added to mypy coverage.

**Open verification (TDD pin-to-real):** Confirm `Player.yahoo_id`/`id` and that `REPLACEMENT_BY_POSITION` entries carry the `HITTING_COUNTING` keys, and that `_real_positions` keeps the `OF`/`C` positions used in tests -- all flagged inline as "read first, pin to actual behavior."
