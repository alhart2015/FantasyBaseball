# Games-based MC -- Phase 2 (setup: classification + IL displacement + LeagueContext plumbing) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans. Steps use checkbox (`- [ ]`).

**Goal:** Build the MC's deterministic SETUP layer: (1) make the ERoto pass-1 `eos_baseline` reusable so the MC can build the same `LeagueContext` ERoto uses, and (2) a pure, tested helper that turns a `list[Player]` + `LeagueContext` into the **effective active set** (active-slot + IL bodies, each with its displacement factor and `g_ros_adj`) and the **healthy-bench hitter fill pool** (each with `g_ros_full`, per-game value, eligible positions). NOTHING consumes this yet -- the per-iteration fill (Phase 3) and the MC wiring (Phase 4) come later. This phase is setup + plumbing, fully unit-testable.

**Architecture:** Reuse ERoto's exact functions (`_classify_roster`, `_compute_displacement_factors`) so the MC's IL handling agrees with ERoto by construction. Thread one shared `build_eos_baseline` object (no divergent recompute). New module `mc_roster.py` holds the pure setup helper + its dataclasses.

**Tech Stack:** Python, pytest. Files: `models/standings.py`, `web/refresh_pipeline.py`, new `mc_roster.py`. Reuses `scoring._classify_roster`/`_compute_displacement_factors`/`LeagueContext`/`_real_positions`, `sgp/player_value.calculate_player_sgp`, `models/standings.build_eos_baseline`.

## Global Constraints

- ASCII-only in source/strings. Numeric defaults via `is not None`, never `x or default`.
- Spec: `docs/superpowers/specs/2026-06-26-games-based-availability-mc-design.md` (Component 2, accounting section, Phase 2).
- `g_ros_full` (= `rest_of_season.g`, per-game-VALUE denominator) and `g_ros_adj` (= `factor * g_ros_full`, games-missed multiplier / fill cap) are DISTINCT and load-bearing -- name them apart.
- Displacement factors come back keyed by NAME (`_compute_displacement_factors` returns `dict[str, float]`); RE-KEY onto `Player` objects so all downstream keying is by `yahoo_id`/identity, and GUARD duplicate names within the active+IL set (the name-keyed source's inherent collision limit -- a deeper id-keying fix in `scoring.py` is out of scope).
- `LeagueContext` is REQUIRED for the pitcher pool model; the helper takes it as a non-optional arg.
- mypy: `models/standings.py`, `web/refresh_pipeline.py`, `sgp/`, `models/` are under `[tool.mypy].files`; run mypy on touched files (full mypy errors on the pre-existing deleted `category_odds.py` -- unrelated, ignore).
- The single shared PA-per-game / IP-per-appearance constant (for deriving `g_ros_full` when `rest_of_season.g` is 0/absent) is defined ONCE here and reused in Phase 3; do not introduce a second.

---

### Task 1: Make `eos_baseline` reusable (LeagueContext baseline plumbing)

**Files:** Modify `src/fantasy_baseball/models/standings.py` (`ProjectedStandings.from_rosters` ~398-499); `src/fantasy_baseball/web/refresh_pipeline.py` (`_build_projected_standings` ~861-940, and `__init__` ~394-446). Tests: `tests/test_models/test_standings.py`.

**Interfaces:**
- Produces: `from_rosters(..., baseline_stats: dict[str, CategoryStats] | None = None)` -- when provided, used as the pass-1 baseline INSTEAD of recomputing; when `None` (default, all existing callers), recomputes exactly as today. `RefreshRun.eos_baseline: dict[str, CategoryStats] | None` populated in `_build_projected_standings`.

- [ ] **Step 1: Write the failing test** in `tests/test_models/test_standings.py` (it already constructs rosters + standings for `from_rosters`; follow that fixture pattern):

```python
def test_from_rosters_accepts_precomputed_baseline_identical():
    """Passing the same baseline build_eos_baseline would produce yields
    byte-identical ProjectedStandings to letting from_rosters recompute it."""
    from fantasy_baseball.models.standings import ProjectedStandings, build_eos_baseline
    rosters, actual = _two_team_rosters_and_standings()  # reuse the file's existing helper/fixture
    eff_date = actual.effective_date
    ytd_by_team = {e.team_name: e.ytd_components() for e in actual.entries}
    baseline = build_eos_baseline(rosters, ytd_by_team)

    without = ProjectedStandings.from_rosters(rosters, effective_date=eff_date, actual_standings=actual, fraction_remaining=0.5)
    with_ = ProjectedStandings.from_rosters(rosters, effective_date=eff_date, actual_standings=actual, fraction_remaining=0.5, baseline_stats=baseline)
    a = {e.team_name: e.stats.to_dict() for e in without.entries}
    b = {e.team_name: e.stats.to_dict() for e in with_.entries}
    assert a == b
```

(If `test_standings.py` lacks a reusable rosters+standings fixture, build a minimal two-team one inline with `Player`/`HitterStats` + a `Standings` carrying `ytd_components`. Read the file first.)

- [ ] **Step 2: Run, confirm FAIL** (`from_rosters() got an unexpected keyword argument 'baseline_stats'`).
- [ ] **Step 3: Refactor `from_rosters`.** Add the keyword-only param and guard the recompute:

```python
    @classmethod
    def from_rosters(
        cls,
        team_rosters: Mapping[str, Any],
        effective_date: date,
        *,
        actual_standings: Standings | None = None,
        fraction_remaining: float = 1.0,
        baseline_stats: dict[str, CategoryStats] | None = None,
    ) -> ProjectedStandings:
```

At line 469, replace the unconditional build with:
```python
        if baseline_stats is None:
            baseline_stats = build_eos_baseline(team_rosters, ytd_by_team)
```
(Everything else in `from_rosters` is unchanged -- the Pass-2 loop already reads `baseline_stats`.)

- [ ] **Step 4: Run, confirm PASS.** Also run the full standings suite: `pytest tests/test_models/test_standings.py -q`.
- [ ] **Step 5: Wire the pipeline.** In `refresh_pipeline.py` `__init__`, add `self.eos_baseline: dict[str, CategoryStats] | None = None` alongside the other `self.*` projected-standings attrs (~411-423). The instance-attr annotation is NOT runtime-evaluated, so add `CategoryStats` to the existing `TYPE_CHECKING` import block (~64-68, where `ProjectedStandings` etc. are already TYPE_CHECKING-imported) so mypy resolves it -- no runtime import needed. In `_build_projected_standings`, AFTER `ytd_standings` is built (~line 930) and BEFORE the `from_rosters` call (932), insert:

```python
        from fantasy_baseball.models.standings import build_eos_baseline

        ytd_by_team = {e.team_name: e.ytd_components() for e in ytd_standings.entries}
        self.eos_baseline = build_eos_baseline(all_team_rosters, ytd_by_team)
```

and pass it through:
```python
        self.projected_standings = ProjectedStandings.from_rosters(
            all_team_rosters,
            effective_date=self.effective_date,
            actual_standings=ytd_standings,
            fraction_remaining=self.fraction_remaining,
            baseline_stats=self.eos_baseline,
        )
```
This computes `build_eos_baseline` ONCE and shares the identical object with both `self.eos_baseline` (for the MC, Phase 4) and the standings build -- no divergent recompute.

- [ ] **Step 6: Run the refresh-pipeline test suite** (`pytest tests/test_web/test_refresh_pipeline.py -q`) to confirm the wiring doesn't regress; if a fixture exercises `_build_projected_standings`, add an assertion that `run.eos_baseline` is populated and non-empty.
- [ ] **Step 7: mypy** on `models/standings.py` + `web/refresh_pipeline.py`; ruff check/format.
- [ ] **Step 8: Commit:**
```bash
git add src/fantasy_baseball/models/standings.py src/fantasy_baseball/web/refresh_pipeline.py tests/test_models/test_standings.py
git commit -m "feat(standings): reusable eos_baseline via optional from_rosters param + store on RefreshRun"
```

---

### Task 2: `build_effective_roster` setup helper

**Files:** Create `src/fantasy_baseball/mc_roster.py`; Test `tests/test_mc_roster.py`.

**Interfaces:**
- Consumes: `scoring._classify_roster(list[Player]) -> (active, il, bench)`; `scoring._compute_displacement_factors(active, il, *, league_context) -> dict[str, float]` (name-keyed); `scoring.LeagueContext`; `scoring._real_positions(Player) -> frozenset[Position]`; `sgp.player_value.calculate_player_sgp(rest_of_season)`.
- Produces:
  - `@dataclass(frozen=True) ActiveBody(player: Player, factor: float, g_ros_adj: float)`
  - `@dataclass(frozen=True) BenchBody(player: Player, g_ros_full: float, per_game_value: float, eligible_positions: frozenset[Position])`
  - `@dataclass(frozen=True) EffectiveRoster(active: list[ActiveBody], bench: list[BenchBody])`
  - `build_effective_roster(roster: list[Player], league_context: LeagueContext) -> EffectiveRoster`
  - module constant `PA_PER_GAME: float = 4.3` (the shared per-game constant; Phase 3 reuses it).

Scope notes: `active` = active-slot bodies + IL bodies (both player types; factors from `_compute_displacement_factors`). `bench` = healthy-bench HITTERS only (the fill pool). Healthy-bench PITCHERS are dropped (pitcher bench-fill is deferred to Phase 5; pitcher displacement still happens via the active/IL path). IL bodies have `factor` from the dict (often 1.0 / pool sf); a body absent from the factor dict gets `factor=1.0`.

- [ ] **Step 1: Write the failing tests** in `tests/test_mc_roster.py`. Use the `LeagueContext` fixture pattern from `tests/test_scoring.py:1068` (build a `baseline_other_team_stats` of `CategoryStats` + a `team_sds` map).

```python
import numpy as np
from fantasy_baseball.models.player import HitterStats, PitcherStats, Player, PlayerType
from fantasy_baseball.models.positions import Position
from fantasy_baseball.scoring import LeagueContext
from fantasy_baseball.models.standings import CategoryStats
from fantasy_baseball.utils.constants import ALL_CATEGORIES
from fantasy_baseball.mc_roster import build_effective_roster, PA_PER_GAME


def _h(name, slot, pid, r=80, g=150, pa=600):
    return Player(name=name, player_type=PlayerType.HITTER, positions=[Position.OF],
                  selected_position=slot, yahoo_id=pid,
                  rest_of_season=HitterStats.from_dict({"r": r, "hr": 20, "rbi": 70, "sb": 5, "h": 150, "ab": 550, "pa": pa, "g": g}))


def _ctx(team="Me", others=("Opp",)):
    base = {t: CategoryStats() for t in others}
    sds = {t: {c: 5.0 for c in ALL_CATEGORIES} for t in (team, *others)}
    return LeagueContext(baseline_other_team_stats=base, team_sds=sds, team_name=team)


def test_active_and_bench_classification():
    roster = [
        _h("Starter", Position.OF, "1"),
        _h("BenchBat", Position.BN, "2"),     # healthy bench hitter -> fill pool
    ]
    eff = build_effective_roster(roster, _ctx())
    assert [b.player.name for b in eff.active] == ["Starter"]
    assert [b.player.name for b in eff.bench] == ["BenchBat"]


def test_bench_body_value_and_games():
    roster = [_h("Starter", Position.OF, "1"), _h("BenchBat", Position.BN, "2", g=120)]
    eff = build_effective_roster(roster, _ctx())
    bench = eff.bench[0]
    assert bench.g_ros_full == 120                      # rest_of_season.g
    assert bench.per_game_value > 0                     # sgp / g_ros_full, guarded
    assert Position.OF in bench.eligible_positions


def test_missing_g_derives_from_pa():
    roster = [_h("Starter", Position.OF, "1"), _h("BenchBat", Position.BN, "2", g=0, pa=516)]
    eff = build_effective_roster(roster, _ctx())
    assert abs(eff.bench[0].g_ros_full - 516 / PA_PER_GAME) < 1e-6   # not zeroed


def test_il_hitter_in_active_set_with_partial_factor_and_g_ros_adj():
    # IL hitter activates and displaces an active match by its expected ROS PT.
    # IL pa=300 vs active pa=600 -> factor = (600-300)/600 = 0.5 (a PARTIAL factor,
    # so g_ros_adj = 0.5 * g is non-trivial -- NOT a vacuous 0 == 0).
    roster = [
        _h("Star", Position.OF, "1", r=100, pa=600),
        _h("Weak", Position.OF, "2", r=40, pa=600),    # active body, the displaced target
        _h("ILbat", Position.IL, "3", r=90, pa=300, g=80),  # IL -> activates, displaces by its 300 PA
    ]
    eff = build_effective_roster(roster, _ctx())
    names = {b.player.name for b in eff.active}
    assert "ILbat" in names                       # IL body in the active set
    # exactly one active body should be displaced to a PARTIAL (~0.5) factor.
    displaced = [b for b in eff.active if b.factor < 0.999]
    assert len(displaced) == 1, "one active body should be displaced by the IL return"
    b = displaced[0]
    assert 0.0 < b.factor < 1.0, f"expected a partial factor, got {b.factor}"
    assert abs(b.g_ros_adj - b.factor * b.player.rest_of_season.g) < 1e-6
    # undisplaced bodies (and IL body) keep factor 1.0 -> g_ros_adj == g
    star = next(x for x in eff.active if x.player.name == "Star")
    assert star.factor == 1.0 and abs(star.g_ros_adj - star.player.rest_of_season.g) < 1e-6


def test_duplicate_name_in_active_set_guarded():
    # Same-name collision in active+il must not silently mis-scale: the helper
    # raises (or logs+falls back to 1.0) rather than apply an ambiguous factor.
    import pytest
    roster = [_h("Same", Position.OF, "1"), _h("Same", Position.OF, "2")]
    # No IL -> no displacement, factors empty -> safe; force a collision under displacement:
    roster.append(_h("Same", Position.IL, "3"))
    with pytest.raises(ValueError):
        build_effective_roster(roster, _ctx())
```

(Adjust the displacement expectation to what `_compute_displacement_factors` actually returns for this tiny roster -- read the function first and pin the test to real behavior; if the picker leaves all factors 1.0 for a degenerate 2-hitter case, construct a roster where displacement genuinely fires, e.g. enough active OF that one is clearly the worst match.)

- [ ] **Step 2: Run, confirm FAIL** (ModuleNotFoundError).
- [ ] **Step 3: Implement `mc_roster.py`** (all imports at top):

```python
"""MC setup: classification + IL displacement -> effective active set + bench fill pool.

Reuses ERoto's _classify_roster + _compute_displacement_factors so the MC's IL
handling agrees with ERoto by construction. Pure/deterministic -- runs once per
team at MC setup on the ROS means. Consumed by the per-iteration fill engine
(Phase 3) and the MC integration (Phase 4); nothing consumes it yet.
"""

from __future__ import annotations

from dataclasses import dataclass

from fantasy_baseball.models.player import Player, PlayerType
from fantasy_baseball.models.positions import Position
from fantasy_baseball.scoring import (
    LeagueContext,
    _classify_roster,
    _compute_displacement_factors,
    _real_positions,
)
from fantasy_baseball.sgp.player_value import calculate_player_sgp

PA_PER_GAME: float = 4.3  # shared per-game constant (Phase 3 reuses; do not duplicate)


@dataclass(frozen=True)
class ActiveBody:
    player: Player
    factor: float       # displacement factor (1.0 if undisplaced)
    g_ros_adj: float    # factor * g_ros_full -- the games-missed multiplier / fill cap


@dataclass(frozen=True)
class BenchBody:
    player: Player
    g_ros_full: float                       # ROS games (per-game-value denominator)
    per_game_value: float                   # ROS SGP per ROS game
    eligible_positions: frozenset[Position]


@dataclass(frozen=True)
class EffectiveRoster:
    active: list[ActiveBody]   # active-slot + IL bodies, with factors
    bench: list[BenchBody]     # healthy-bench HITTER fill pool


def _g_ros_full(p: Player) -> float:
    """ROS games, with a PA-derived fallback when the projection lacks g.

    Never trusts a literal g==0 as 'plays zero games' (the falsy-zero footgun):
    derives from ROS PA via PA_PER_GAME. Pitchers fall back to their own g (now
    plumbed) or, absent that, are left at 0 -- pitcher bench-fill is deferred.
    """
    ros = p.rest_of_season
    if ros is None:
        return 0.0
    g = float(getattr(ros, "g", 0) or 0)
    if g > 0:
        return g
    if p.player_type == PlayerType.HITTER:
        pa = float(getattr(ros, "pa", 0) or 0)
        return pa / PA_PER_GAME if pa > 0 else 0.0
    return 0.0


def build_effective_roster(roster: list[Player], league_context: LeagueContext) -> EffectiveRoster:
    active, il, bench = _classify_roster([p for p in roster if isinstance(p, Player)])

    factors_by_name = _compute_displacement_factors(active, il, league_context=league_context)

    # Re-key factors onto Player objects; guard same-name collisions in active+il.
    bodies = [*il, *active]
    seen: set[str] = set()
    for b in bodies:
        if b.name in seen and b.name in factors_by_name:
            raise ValueError(
                f"Ambiguous displacement factor: duplicate name {b.name!r} in the "
                "active+IL set; cannot re-key a name-scoped factor by identity."
            )
        seen.add(b.name)

    active_bodies: list[ActiveBody] = []
    for b in bodies:
        factor = factors_by_name.get(b.name, 1.0)
        active_bodies.append(ActiveBody(player=b, factor=factor, g_ros_adj=factor * _g_ros_full(b)))

    bench_bodies: list[BenchBody] = []
    for b in bench:
        if b.player_type != PlayerType.HITTER:
            continue  # pitcher bench-fill deferred (Phase 5); healthy bench pitchers excluded
        gf = _g_ros_full(b)
        sgp = calculate_player_sgp(b.rest_of_season) if b.rest_of_season is not None else 0.0
        per_game = (sgp / gf) if gf > 0 else 0.0
        bench_bodies.append(
            BenchBody(player=b, g_ros_full=gf, per_game_value=per_game, eligible_positions=_real_positions(b))
        )

    return EffectiveRoster(active=active_bodies, bench=bench_bodies)
```

(Before finalizing: READ `_compute_displacement_factors` and `_real_positions` to confirm signatures/returns, and confirm `calculate_player_sgp` accepts a `HitterStats`/`PitcherStats` instance. Pin the IL/collision tests to the picker's ACTUAL behavior on the constructed rosters.)

- [ ] **Step 4: Run, confirm PASS:** `pytest tests/test_mc_roster.py -v`.
- [ ] **Step 5: ruff check/format on `mc_roster.py` + test; add to mypy coverage.** Add `"src/fantasy_baseball/mc_roster.py"` to `[tool.mypy].files` in `pyproject.toml` (repo convention is near-exhaustive coverage of `src/fantasy_baseball/`), then `mypy src/fantasy_baseball/mc_roster.py` -- expected clean.
- [ ] **Step 6: Commit:**
```bash
git add src/fantasy_baseball/mc_roster.py tests/test_mc_roster.py
git commit -m "feat(mc): build_effective_roster setup helper (classification + IL displacement reuse)"
```

---

## Self-Review

**Spec coverage:** Implements spec Component 2 / Phase 2 -- reuse `_classify_roster` + `_compute_displacement_factors` (IL agrees with ERoto by construction); re-key factors onto Players by identity with a collision guard; effective active set carries `factor` + `g_ros_adj`; healthy-bench hitter pool carries `g_ros_full`, per-game value (`g_ros_full` denominator), eligible positions; mandatory `g_ros` presence/derivation gate (PA fallback); LeagueContext baseline threaded from `_build_projected_standings` as ONE shared object (no divergent recompute). Pitcher bench-fill deferred (Phase 5). Nothing consumed yet -- Phases 3-4.

**Placeholder scan:** Concrete code + tests given; the two "read the real function/fixture first and pin to actual behavior" notes are genuine TDD instructions (the displacement picker's exact factor on a tiny roster must be observed, not guessed), not placeholders.

**Type consistency:** `from_rosters` gains `baseline_stats: dict[str, CategoryStats] | None = None`; `self.eos_baseline` same type. `build_effective_roster -> EffectiveRoster(active: list[ActiveBody], bench: list[BenchBody])`; `g_ros_adj = factor * g_ros_full` consistent with the spec's two-quantity split; `per_game_value = sgp / g_ros_full` (full, not adj) per the spec value rule.
