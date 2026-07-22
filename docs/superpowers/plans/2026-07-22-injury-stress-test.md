# Injury Stress-Test Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Source spec:** `docs/superpowers/specs/2026-07-22-injury-stress-test-design.md`

**Goal:** A standalone, re-runnable script that quantifies how robust the user's rest-of-season (ROS) lead is to injuries: a headline attribution of what availability risk costs the win%, P(everyone stays healthy / 1 / 2+ lose significant time), and ranked single + pair "lose-a-player" counterfactuals priced as value-over-replacement.

**Architecture:** One additive engine change (an `availability_variance_off` flag on the ROS Monte Carlo) plus a new library module `analysis/injury_stress.py` (pure logic: input assembly from stored Upstash, health-probability sampler, replacement-level substitution, stress orchestration, ASCII renderer) driven by a thin CLI `scripts/injury_stress_test.py`. Every number rides the existing `simulation.run_ros_monte_carlo` so it reconciles with the dashboard.

**Tech Stack:** Python 3, numpy, existing `fantasy_baseball` modules (`simulation`, `mc_roster`, `scoring`, `models.player/standings`, `data.kv_store/redis_store`, `utils.playing_time/constants`). Tests: pytest.

## Global Constraints

- **ASCII-only** in all source, log messages, format strings, and the rendered report (this is a Windows/cp1252 box; non-ASCII crashes `print`). Use `-`, `->`, `sigma`, straight quotes. Player names pulled from data may be non-ASCII, so the CLI entry point must `sys.stdout.reconfigure(encoding="utf-8", errors="replace")`.
- **Player IDs are `name::player_type`**; never key on bare names where collisions matter. Within one team's roster, names are effectively unique for our counterfactual targeting; tie-break/guard is not required for v1 but do not assume global name uniqueness.
- **No `x or default` for numeric defaults** (0/0.0 are falsy). Use `v if v is not None else default`.
- **Read live season state from remote Upstash**, not local SQLite. Use `build_explicit_upstash_kv()` (no `RENDER` mutation), matching `scripts/compare_eroto_mc_means.py`.
- **Do not modify a failing test to make it pass.** Fix the code.
- **Every run uses `seed=42`, `n_iterations=1000`** by default so baseline / availability-off / counterfactual runs share random draws (common random numbers) and the win% deltas are low-noise.
- **End-of-effort verification** (Task 9): `pytest -v`, `ruff check .`, `ruff format --check .`, `vulture`, and `mypy` if any touched file is under `[tool.mypy].files`.

---

## File Structure

- **Modify** `src/fantasy_baseball/simulation.py` — add `pin_role` to `_sv_role_mu`; add `availability_variance_off` to `_apply_variance_batch`, `_sample_hitter_bodies`, `_simulate_team_hitters_ros_direct`, `_simulate_team_pitchers_ros_direct`, `simulate_remaining_season_batch`, `run_ros_monte_carlo`.
- **Create** `src/fantasy_baseball/analysis/injury_stress.py` — all stress-test logic (dataclasses, loader, sampler, substitution, orchestration, renderer).
- **Create** `scripts/injury_stress_test.py` — CLI entry point.
- **Create** `tests/test_analysis/test_injury_stress.py` — unit + integration tests for the new module.
- **Modify** `tests/test_mc_integration.py` OR **create** tests in `tests/test_simulation.py` — the `availability_variance_off` engine tests (put them where the existing `_apply_variance_batch` tests live: `tests/test_mc_integration.py`).

---

## Task 1: `availability_variance_off` flag in the batch sampler

Add the flag at the single choke point (`_apply_variance_batch`) and the SV pin (`_sv_role_mu`). The flag keeps all RNG draws (so common random numbers hold vs. the baseline) but pins the playing-time scale to `eff_mean` and the SV role multiplier to its mean (1.0).

**Files:**
- Modify: `src/fantasy_baseball/simulation.py` (`_sv_role_mu` ~628-645; `_apply_variance_batch` ~763-895)
- Test: `tests/test_mc_integration.py`

**Interfaces:**
- Produces: `_apply_variance_batch(..., *, availability_variance_off: bool = False)`; `_sv_role_mu(..., pin_role: bool = False)`.

- [ ] **Step 1: Write the failing test**

Add to `tests/test_mc_integration.py` (it already imports `_apply_variance_batch`, `VarianceBatch`, `np`, and defines `_players()`):

```python
def test_availability_variance_off_pins_scales_to_eff_mean():
    import numpy as np
    from fantasy_baseball.simulation import _apply_variance_batch

    rng = np.random.default_rng(777)
    vb = _apply_variance_batch(
        _players(), "hitter", rng, 0.4, 200, availability_variance_off=True
    )
    # frac_missed = 1 - scale; with availability off, scale is pinned to eff_mean
    # per player, so every iteration is identical -> zero spread down each column.
    assert np.allclose(vb.frac_missed.std(axis=0), 0.0)
    assert np.allclose(vb.scales.std(axis=0), 0.0)


def test_availability_variance_off_default_is_byte_identical():
    import numpy as np
    from fantasy_baseball.simulation import _apply_variance_batch

    a = _apply_variance_batch(_players(), "hitter", np.random.default_rng(5), 0.4, 8)
    b = _apply_variance_batch(
        _players(), "hitter", np.random.default_rng(5), 0.4, 8,
        availability_variance_off=False,
    )
    for col in a.counts:
        assert np.array_equal(a.counts[col], b.counts[col])
```

- [ ] **Step 2: Run to verify it fails**

Run: `pytest tests/test_mc_integration.py::test_availability_variance_off_pins_scales_to_eff_mean -v`
Expected: FAIL with `TypeError: _apply_variance_batch() got an unexpected keyword argument 'availability_variance_off'`.

- [ ] **Step 3: Add `pin_role` to `_sv_role_mu`**

Replace the body of `_sv_role_mu` (keep the docstring; add the param and the pin). Current last two lines are:
```python
    x = closer_mixture.role_multiplier_draw(sv_curve, rng, fraction_remaining, n_iter=n_iter)
    return np.asarray(base_sv * eff_mean * x, dtype=float)
```
Change the signature to add `pin_role: bool = False` (keyword-only, after `n_iter`) and replace those two lines with:
```python
    # Draw unconditionally so the rng stream (and thus common-random-numbers vs the
    # baseline) is unchanged; when pinning, use the mixture's expected multiplier
    # (E[X'] == 1 by construction, see closer_mixture) instead of the sampled role.
    x = closer_mixture.role_multiplier_draw(sv_curve, rng, fraction_remaining, n_iter=n_iter)
    role = 1.0 if pin_role else x
    return np.asarray(base_sv * eff_mean * role, dtype=float)
```

- [ ] **Step 4: Add `availability_variance_off` to `_apply_variance_batch`**

Add `availability_variance_off: bool = False` to the keyword-only block of the signature (after `sv_curve`). Then:

Change the scale line (currently `scales = np.maximum(0.0, eff_mean[None, :] + z_pt * eff_sd[None, :])`, ~line 849) to:
```python
        # Availability-off: pin every draw to eff_mean (zero playing-time spread)
        # while STILL consuming `us`/`z_pt` above, so the rng stream stays aligned
        # with the baseline run (common random numbers) -- only the transform differs.
        pt_spread = 0.0 if availability_variance_off else z_pt * eff_sd[None, :]
        scales = np.maximum(0.0, eff_mean[None, :] + pt_spread)
```

Change the SV mean line (currently the `_sv_role_mu(...)` call ~line 868-870) to pass the pin:
```python
        mu_mat[:, :, idx_map["sv"]] = _sv_role_mu(
            base["sv"], sv_curve_arr, eff_mean, rng, fraction_remaining,
            n_iter=n_iter, pin_role=availability_variance_off,
        )
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `pytest tests/test_mc_integration.py -k availability_variance_off -v`
Expected: PASS (both).

- [ ] **Step 6: Commit**

```bash
git add src/fantasy_baseball/simulation.py tests/test_mc_integration.py
git commit -m "feat(sim): availability_variance_off flag in _apply_variance_batch"
```

---

## Task 2: Thread `availability_variance_off` through to `run_ros_monte_carlo`

The flag must reach every `_apply_variance_batch` call in the ROS MC: the two top-k calls and the two ROS-direct helpers (via `_sample_hitter_bodies`).

**Files:**
- Modify: `src/fantasy_baseball/simulation.py` (`_sample_hitter_bodies` ~898; `_simulate_team_hitters_ros_direct` ~926; `_simulate_team_pitchers_ros_direct` ~1052; `simulate_remaining_season_batch` ~1120; `run_ros_monte_carlo` ~1437)
- Test: `tests/test_mc_integration.py`

**Interfaces:**
- Consumes: Task 1's `_apply_variance_batch(..., availability_variance_off=...)`.
- Produces: `run_ros_monte_carlo(..., availability_variance_off: bool = False)` returning the same result shape; `team_results[team]["first_pct"]` is the win%.

- [ ] **Step 1: Write the failing test**

```python
def test_run_ros_mc_availability_off_threads_through():
    import numpy as np
    from fantasy_baseball.simulation import run_ros_monte_carlo
    rosters = _mixed_rosters()
    actuals = {t: {} for t in rosters}
    eff = {t: _eff_roster(players, team=t) for t, players in rosters.items()}
    kw = dict(
        team_rosters=rosters, actual_standings=actuals, fraction_remaining=0.4,
        h_slots=13, p_slots=9, user_team_name="Me", n_iterations=60, seed=42,
        effective_rosters=eff,
    )
    base = run_ros_monte_carlo(**kw)
    off = run_ros_monte_carlo(**kw, availability_variance_off=True)
    # Availability-off narrows each team's spread (removes the playing-time and SV
    # role variance), so the user's p90-p10 band must not widen.
    b = base["team_results"]["Me"]
    o = off["team_results"]["Me"]
    assert (o["p90"] - o["p10"]) <= (b["p90"] - b["p10"]) + 1e-9
    assert np.isfinite(o["first_pct"])
```
(Reuse the existing `_mixed_rosters` / `_eff_roster` helpers already in this file. If `_mixed_rosters` is defined below this test, move the new test after those definitions.)

- [ ] **Step 2: Run to verify it fails**

Run: `pytest tests/test_mc_integration.py::test_run_ros_mc_availability_off_threads_through -v`
Expected: FAIL with `TypeError: ... unexpected keyword argument 'availability_variance_off'`.

- [ ] **Step 3: Thread the kwarg (keyword-only) through each function**

In each signature add `*, availability_variance_off: bool = False` (or extend the existing keyword-only block), and forward it at each call site:

`_sample_hitter_bodies(bodies, rng, fraction_remaining, n_iter, *, availability_variance_off=False)` -> pass `availability_variance_off=availability_variance_off` into its `_apply_variance_batch(...)` call (~914).

`_simulate_team_hitters_ros_direct(effective_roster, fraction_remaining, rng, n_iter, *, availability_variance_off=False)` -> forward it into BOTH `_sample_hitter_bodies(active_h_bodies, ...)` (~983) and `_sample_hitter_bodies(bench_h_bodies, ...)` (~998).

`_simulate_team_pitchers_ros_direct(effective_roster, fraction_remaining, rng, n_iter, *, availability_variance_off=False)` -> forward into its `_apply_variance_batch(...)` (~1098).

`simulate_remaining_season_batch(..., effective_rosters=None, *, availability_variance_off=False)` -> forward into all four inner calls: `_simulate_team_hitters_ros_direct(eff, fraction_remaining, rng, n_iter, availability_variance_off=availability_variance_off)` (~1181), the top-k hitter `_apply_variance_batch(hitters, ...)` (~1183-1185), `_simulate_team_pitchers_ros_direct(eff, ...)` (~1212), and the top-k pitcher `_apply_variance_batch(pitchers, ...)` (~1214-1216).

`run_ros_monte_carlo(..., effective_rosters=None, *, availability_variance_off: bool = False)` -> pass into the `simulate_remaining_season_batch(...)` call (~1504-1513): add `availability_variance_off=availability_variance_off`.

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_mc_integration.py -v`
Expected: PASS (new test + all existing MC-integration tests still green — the default path is unchanged).

- [ ] **Step 5: Commit**

```bash
git add src/fantasy_baseball/simulation.py tests/test_mc_integration.py
git commit -m "feat(sim): thread availability_variance_off through run_ros_monte_carlo"
```

---

## Task 3: `injury_stress` module skeleton + health-probability sampler

Create the module with its constants and the pure health sampler. The sampler mirrors the MC's per-type playing-time moments and measures shortfall below each player's own expected level (`eff_mean`) -- the spec's Section 2 definition.

**Files:**
- Create: `src/fantasy_baseball/analysis/injury_stress.py`
- Test: `tests/test_analysis/test_injury_stress.py` (create; ensure `tests/test_analysis/__init__.py` exists -- it already does, as `tests/test_analysis/` holds other tests)

**Interfaces:**
- Produces:
  - Constants `SIGNIFICANT_TIME_THRESHOLD = 0.20`, `PAIR_TOP_K = 8`, `HEALTH_SAMPLES = 20000`, `DEFAULT_N_ITER = 1000`, `SEED = 42`.
  - `@dataclass(frozen=True) HealthProbs(p_all_healthy: float, p_one: float, p_two_plus: float, per_player: dict[str, float], threshold: float)`.
  - `health_probabilities(active_players: list[Player], fraction_remaining: float, *, threshold: float = SIGNIFICANT_TIME_THRESHOLD, n_samples: int = HEALTH_SAMPLES, seed: int = SEED) -> HealthProbs`.

- [ ] **Step 1: Write the failing test**

Create `tests/test_analysis/test_injury_stress.py`:
```python
import numpy as np
from fantasy_baseball.analysis.injury_stress import HealthProbs, health_probabilities
from fantasy_baseball.models.player import HitterStats, Player, PlayerType
from fantasy_baseball.models.positions import Position


def _hitter(name, *, pa, ab, g):
    return Player(
        name=name, player_type=PlayerType.HITTER, positions=[Position.OF],
        rest_of_season=HitterStats.from_dict({"r": 80, "hr": 20, "rbi": 70, "sb": 5,
                                              "h": 150, "ab": ab, "pa": pa, "g": g}),
        full_season_projection=HitterStats.from_dict({"r": 80, "hr": 20, "rbi": 70,
                                              "sb": 5, "h": 150, "ab": ab, "pa": pa, "g": g}),
    )


def test_health_probabilities_sum_to_one_and_ordered():
    players = [_hitter("A", pa=600, ab=550, g=150), _hitter("B", pa=600, ab=550, g=150)]
    hp = health_probabilities(players, 0.5, n_samples=5000, seed=42)
    assert isinstance(hp, HealthProbs)
    assert abs(hp.p_all_healthy + hp.p_one + hp.p_two_plus - 1.0) < 1e-9
    assert 0.0 <= hp.p_two_plus <= hp.p_one  # two-or-more is rarer than exactly-one here
    assert set(hp.per_player) == {"A", "B"}


def test_health_haircut_alone_is_not_significant():
    # A player realizing EXACTLY his expected level (eff_mean) must NOT count as
    # losing significant time -- guards the haircut-vs-injury bug. With threshold
    # 0 no one is ever significant regardless of the systematic mean haircut... so
    # instead assert per-player significance stays well below 1.0 for a healthy
    # full-timer (the haircut does not by itself trip the eff_mean-relative bar).
    players = [_hitter("A", pa=600, ab=550, g=150)]
    hp = health_probabilities(players, 0.5, threshold=0.20, n_samples=20000, seed=1)
    assert hp.per_player["A"] < 0.5
```

- [ ] **Step 2: Run to verify it fails**

Run: `pytest tests/test_analysis/test_injury_stress.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'fantasy_baseball.analysis.injury_stress'`.

- [ ] **Step 3: Write the module + sampler**

Create `src/fantasy_baseball/analysis/injury_stress.py`:
```python
"""Injury stress-test: how robust is the user's ROS lead to lost playing time?

Rides the existing ROS Monte Carlo (simulation.run_ros_monte_carlo) so every
number reconciles with the season dashboard. See
docs/superpowers/specs/2026-07-22-injury-stress-test-design.md.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from fantasy_baseball.models.player import Player, PlayerType
from fantasy_baseball.simulation import _full_season_pt_volume
from fantasy_baseball.utils.constants import QUANTILE_LEVELS
from fantasy_baseball.utils.playing_time import (
    playing_time_moments,
    playing_time_params,
    playing_time_shape,
)

SIGNIFICANT_TIME_THRESHOLD: float = 0.20
PAIR_TOP_K: int = 8
HEALTH_SAMPLES: int = 20000
DEFAULT_N_ITER: int = 1000
SEED: int = 42


@dataclass(frozen=True)
class HealthProbs:
    p_all_healthy: float
    p_one: float
    p_two_plus: float
    per_player: dict[str, float]
    threshold: float


def health_probabilities(
    active_players: list[Player],
    fraction_remaining: float,
    *,
    threshold: float = SIGNIFICANT_TIME_THRESHOLD,
    n_samples: int = HEALTH_SAMPLES,
    seed: int = SEED,
) -> HealthProbs:
    """P(0 / exactly-1 / 2-or-more active players lose significant time).

    Per player, sample realized playing-time scale with the SAME moments the MC
    uses (mean horizon 1.0 for hitters -> eff_mean == mean_scale; 0.0 for pitchers
    -> eff_mean == 1.0; sd horizon == fraction_remaining), then count a
    "significant" loss when realized scale <= eff_mean * (1 - threshold), i.e. at
    least `threshold` below the player's OWN expected remaining playing time. This
    isolates the injury/availability tail from the systematic mean haircut. Draws
    are independent across players (injuries are ~independent).
    """
    rng = np.random.default_rng(seed)
    n = len(active_players)
    if n == 0:
        return HealthProbs(1.0, 0.0, 0.0, {}, threshold)
    significant = np.zeros((n_samples, n), dtype=bool)
    for j, p in enumerate(active_players):
        is_hitter = p.player_type == PlayerType.HITTER
        vol = _full_season_pt_volume(p, is_hitter=is_hitter)
        mean_scale, cv_pt = playing_time_params(p.player_type, vol)
        fr_mean = 1.0 if is_hitter else 0.0
        eff_mean, _ = playing_time_moments(mean_scale, cv_pt, fr_mean)
        _, eff_sd = playing_time_moments(mean_scale, cv_pt, fraction_remaining)
        ladder = np.asarray(playing_time_shape(p.player_type, vol), dtype=float)
        u = rng.random(n_samples)
        z = np.interp(u, QUANTILE_LEVELS, ladder)
        scale = np.maximum(0.0, eff_mean + z * eff_sd)
        significant[:, j] = scale <= eff_mean * (1.0 - threshold)
    counts = significant.sum(axis=1)
    per_player = {p.name: float(significant[:, j].mean()) for j, p in enumerate(active_players)}
    return HealthProbs(
        p_all_healthy=float((counts == 0).mean()),
        p_one=float((counts == 1).mean()),
        p_two_plus=float((counts >= 2).mean()),
        per_player=per_player,
        threshold=threshold,
    )
```
Confirm `_full_season_pt_volume` is importable from `simulation` (it is used at `simulation.py:913, 1095`). If it is prefixed differently, grep `def _full_season_pt_volume` and import the actual name.

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_analysis/test_injury_stress.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/fantasy_baseball/analysis/injury_stress.py tests/test_analysis/test_injury_stress.py
git commit -m "feat(injury-stress): health-probability sampler"
```

---

## Task 4: Replacement-level substitution + counterfactual win%

Model "lose player X" by cloning the user's `Player` list and swapping X's `rest_of_season` line for a position-matched replacement-level line scaled to X's ROS volume, then rebuilding `effective_rosters` and reading the drop in `first_pct`. YTD (in `actual_standings`) stays banked -- correct "lose him from here on" semantics.

**Files:**
- Modify: `src/fantasy_baseball/analysis/injury_stress.py`
- Test: `tests/test_analysis/test_injury_stress.py`

**Interfaces:**
- Produces:
  - `@dataclass(frozen=True) McInputs(team_rosters, actual_standings, fraction_remaining, h_slots, p_slots, eos_baseline, team_sds, denoms, user_team_name, projected_margin)` (types below).
  - `substitute_replacement(user_players: list[Player], target_names: list[str]) -> list[Player]`.
  - `win_pct(inputs: McInputs, user_players: list[Player], *, availability_variance_off: bool = False, n_iter: int = DEFAULT_N_ITER, seed: int = SEED) -> float`.

- [ ] **Step 1: Write the failing test**

Add to `tests/test_analysis/test_injury_stress.py`. Build a minimal 2-team league inline. `eos_baseline` is a `{team: CategoryStats()}` stand-in (matches the `LeagueContext` type used by the existing `_ctx` helper at `tests/test_mc_integration.py:300-303`; the values don't matter for a no-IL roster), so no `TeamYtdComponents`/`build_eos_baseline` is needed in the fixture:
```python
import dataclasses
import math

from fantasy_baseball.analysis.injury_stress import (
    McInputs, substitute_replacement, win_pct,
)
from fantasy_baseball.models.player import PitcherStats
from fantasy_baseball.models.positions import Position
from fantasy_baseball.models.standings import CategoryStats
from fantasy_baseball.scoring import build_team_sds
from fantasy_baseball.sgp.denominators import get_sgp_denominators
from fantasy_baseball.utils.constants import HITTING_COUNTING, PITCHING_COUNTING


def _mk_hitter(name, pid, *, r=90, hr=30, rbi=95, sb=12, h=165, ab=560, pa=620, g=155):
    line = {"r": r, "hr": hr, "rbi": rbi, "sb": sb, "h": h, "ab": ab, "pa": pa, "g": g}
    return Player(
        name=name, player_type=PlayerType.HITTER, positions=[Position.OF],
        selected_position=Position.OF, yahoo_id=pid,
        rest_of_season=HitterStats.from_dict(line),
        full_season_projection=HitterStats.from_dict(line),
    )


def _mk_pitcher(name, pid, *, w=12, k=190, sv=0, ip=170, er=60, bb=45, ha=140, g=30):
    line = {"w": w, "k": k, "sv": sv, "ip": ip, "er": er, "bb": bb, "h_allowed": ha, "g": g}
    return Player(
        name=name, player_type=PlayerType.PITCHER, positions=[Position.SP],
        selected_position=Position.SP, yahoo_id=pid,
        rest_of_season=PitcherStats.from_dict(line),
        full_season_projection=PitcherStats.from_dict(line),
    )


def _synth_inputs():
    """Minimal 2-team league good enough to drive run_ros_monte_carlo."""
    star = _mk_hitter("Star", "1")            # high value
    weak = _mk_hitter("Weak", "2", r=40, hr=3, rbi=35, sb=2, h=95, ab=430, pa=470, g=120)
    ace = _mk_pitcher("Ace", "3")             # high-value pitcher
    me = ([star, weak]
          + [_mk_hitter(f"H{i}", str(10 + i)) for i in range(11)]
          + [ace] + [_mk_pitcher(f"P{i}", str(30 + i)) for i in range(8)])
    opp = ([_mk_hitter(f"O{i}", str(50 + i)) for i in range(13)]
           + [_mk_pitcher(f"Q{i}", str(70 + i)) for i in range(9)])
    team_rosters = {"Me": me, "Opp": opp}
    actual_standings = {t: {} for t in team_rosters}       # preseason-like (no YTD)
    fr = 1.0
    eos = {t: CategoryStats() for t in team_rosters}        # LeagueContext baseline stand-in
    sds = build_team_sds(team_rosters, math.sqrt(fr))
    denoms = get_sgp_denominators(None)
    return McInputs(team_rosters=team_rosters, actual_standings=actual_standings,
                    fraction_remaining=fr, h_slots=13, p_slots=9, eos_baseline=eos,
                    team_sds=sds, denoms=denoms, user_team_name="Me", projected_margin=0.0)


def test_substitute_swaps_ros_to_scaled_replacement():
    inp = _synth_inputs()
    me = inp.team_rosters["Me"]
    sub = substitute_replacement(me, ["Star"])
    orig = {p.name: p for p in me}
    subd = {p.name: p for p in sub}
    assert subd["Weak"].rest_of_season is orig["Weak"].rest_of_season  # untouched
    # Star's ROS counting stats dropped to replacement level (all strictly lower).
    for col in HITTING_COUNTING:
        assert getattr(subd["Star"].rest_of_season, col) < getattr(orig["Star"].rest_of_season, col)
    assert subd["Star"].positions == orig["Star"].positions  # slot preserved


def test_substitute_works_for_a_pitcher():
    inp = _synth_inputs()
    me = inp.team_rosters["Me"]
    subd = {p.name: p for p in substitute_replacement(me, ["Ace"])}
    orig = {p.name: p for p in me}
    for col in PITCHING_COUNTING:
        # Ace's counting stats collapse to replacement level (K/W/IP strictly lower);
        # ER/BB/H may not be monotone, so only assert the "good" counting cats drop.
        if col in ("w", "k", "ip"):
            assert getattr(subd["Ace"].rest_of_season, col) < getattr(orig["Ace"].rest_of_season, col)


def test_counterfactual_star_costs_more_than_weak():
    inp = _synth_inputs()
    me = inp.team_rosters["Me"]
    base = win_pct(inp, me, n_iter=300)
    lose_star = win_pct(inp, substitute_replacement(me, ["Star"]), n_iter=300)
    lose_weak = win_pct(inp, substitute_replacement(me, ["Weak"]), n_iter=300)
    assert base - lose_star >= base - lose_weak    # star hurts at least as much
    assert base - lose_star > 0.0                  # losing the star has a real cost


def test_counterfactual_pitcher_has_cost_and_no_raw_hole():
    inp = _synth_inputs()
    me = inp.team_rosters["Me"]
    base = win_pct(inp, me, n_iter=300)
    lose_ace = win_pct(inp, substitute_replacement(me, ["Ace"]), n_iter=300)
    assert base - lose_ace > 0.0    # a replacement arm still pitches -> real, finite cost


def test_win_pct_is_deterministic():
    # Same inputs + same seed -> identical first_pct (locks the reconciliation /
    # common-random-numbers contract).
    inp = _synth_inputs()
    me = inp.team_rosters["Me"]
    assert win_pct(inp, me, n_iter=200) == win_pct(inp, me, n_iter=200)
```

- [ ] **Step 2: Run to verify it fails**

Run: `pytest tests/test_analysis/test_injury_stress.py -k "substitute or counterfactual" -v`
Expected: FAIL with `ImportError: cannot import name 'McInputs'`.

- [ ] **Step 3: Implement `McInputs`, `substitute_replacement`, `win_pct`**

Add to `injury_stress.py`:
```python
import dataclasses

from fantasy_baseball.mc_roster import build_effective_rosters
from fantasy_baseball.simulation import _replacement_line, run_ros_monte_carlo
from fantasy_baseball.utils.constants import (
    AB_PER_PA, HITTING_COUNTING, PITCHING_COUNTING,
)


@dataclass(frozen=True)
class McInputs:
    team_rosters: dict[str, list[Player]]
    actual_standings: dict[str, dict[str, float]]
    fraction_remaining: float
    h_slots: int
    p_slots: int
    eos_baseline: dict
    team_sds: dict
    denoms: dict
    user_team_name: str
    projected_margin: float


def _replacement_ros(player: Player):
    """Replacement-level ROS stats object at `player`'s slot, scaled to his ROS
    playing-time volume (AB for hitters, IP for pitchers). Returns a NEW stats
    object; `player.rest_of_season` is not mutated."""
    is_hitter = player.player_type == PlayerType.HITTER
    ros = player.rest_of_season
    repl = _replacement_line(player.to_flat_dict(), is_hitter)
    if is_hitter:
        x_ab = float(ros.ab) if ros is not None and ros.ab else 0.0
        factor = (x_ab / repl["ab"]) if repl.get("ab") else 0.0
        s = {c: repl[c] * factor for c in HITTING_COUNTING}
        avg = (s["h"] / s["ab"]) if s["ab"] else 0.0
        return dataclasses.replace(
            ros, r=s["r"], hr=s["hr"], rbi=s["rbi"], sb=s["sb"], h=s["h"],
            ab=s["ab"], pa=(s["ab"] / AB_PER_PA), avg=avg, sgp=None,
        )
    x_ip = float(ros.ip) if ros is not None and ros.ip else 0.0
    factor = (x_ip / repl["ip"]) if repl.get("ip") else 0.0
    s = {c: repl[c] * factor for c in PITCHING_COUNTING}
    era = (s["er"] * 9.0 / s["ip"]) if s["ip"] else 0.0
    whip = ((s["bb"] + s["h_allowed"]) / s["ip"]) if s["ip"] else 0.0
    return dataclasses.replace(
        ros, w=s["w"], k=s["k"], sv=s["sv"], ip=s["ip"], er=s["er"], bb=s["bb"],
        h_allowed=s["h_allowed"], era=era, whip=whip, sgp=None,
    )


def substitute_replacement(user_players: list[Player], target_names: list[str]) -> list[Player]:
    """Clone `user_players`, replacing each named player's ROS line with a
    position-matched replacement-level line (see `_replacement_ros`). Non-targets
    are shared unchanged (same object)."""
    targets = set(target_names)
    out: list[Player] = []
    for p in user_players:
        if p.name in targets:
            out.append(dataclasses.replace(p, rest_of_season=_replacement_ros(p)))
        else:
            out.append(p)
    return out


def win_pct(
    inputs: McInputs,
    user_players: list[Player],
    *,
    availability_variance_off: bool = False,
    n_iter: int = DEFAULT_N_ITER,
    seed: int = SEED,
) -> float:
    """User's P(finish 1st) for a given user roster. Rebuilds effective_rosters
    (fixed eos_baseline/team_sds/fraction_remaining) so the substitution takes
    effect in the ROS-direct path, then runs the ROS MC."""
    team_rosters = {**inputs.team_rosters, inputs.user_team_name: user_players}
    eff = build_effective_rosters(
        team_rosters, inputs.eos_baseline, inputs.team_sds,
        inputs.fraction_remaining, denoms=inputs.denoms,
    )
    mc = run_ros_monte_carlo(
        team_rosters=team_rosters, actual_standings=inputs.actual_standings,
        fraction_remaining=inputs.fraction_remaining, h_slots=inputs.h_slots,
        p_slots=inputs.p_slots, user_team_name=inputs.user_team_name,
        n_iterations=n_iter, seed=seed, effective_rosters=eff,
        availability_variance_off=availability_variance_off,
    )
    return float(mc["team_results"][inputs.user_team_name]["first_pct"])
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_analysis/test_injury_stress.py -v`
Expected: PASS (all).

- [ ] **Step 5: Commit**

```bash
git add src/fantasy_baseball/analysis/injury_stress.py tests/test_analysis/test_injury_stress.py
git commit -m "feat(injury-stress): replacement-level substitution + counterfactual win%"
```

---

## Task 5: Stress-test orchestration

Assemble the headline, health probabilities, ranked single counterfactuals, and top-K pairs into a `StressResult`.

**Files:**
- Modify: `src/fantasy_baseball/analysis/injury_stress.py`
- Test: `tests/test_analysis/test_injury_stress.py`

**Interfaces:**
- Produces:
  - `@dataclass(frozen=True) PlayerExposure(name: str, player_type: str, win_pct_cost: float)`.
  - `@dataclass(frozen=True) PairExposure(name_a: str, name_b: str, joint_cost: float, super_additive: float)`.
  - `@dataclass(frozen=True) StressResult(baseline_win_pct, availability_off_win_pct, projected_margin, health: HealthProbs, singles: list[PlayerExposure], pairs: list[PairExposure], threshold: float, n_iter: int, seed: int)`.
  - `run_stress_test(inputs: McInputs, *, threshold=SIGNIFICANT_TIME_THRESHOLD, pair_top_k=PAIR_TOP_K, n_iter=DEFAULT_N_ITER, seed=SEED) -> StressResult`.

- [ ] **Step 1: Write the failing test**

```python
def test_run_stress_test_ranks_and_flags():
    from fantasy_baseball.analysis.injury_stress import run_stress_test
    inp = _synth_inputs()
    res = run_stress_test(inp, n_iter=300, pair_top_k=4)
    names = [e.name for e in res.singles]
    assert names[0] == "Star"                       # highest exposure ranked first
    assert res.singles == sorted(res.singles, key=lambda e: e.win_pct_cost, reverse=True)
    assert 0.0 <= res.health.p_all_healthy <= 1.0
    # pairs are top-K choose 2 and ranked by joint cost
    assert len(res.pairs) == 6                       # C(4, 2)
    assert res.pairs == sorted(res.pairs, key=lambda e: e.joint_cost, reverse=True)
```

- [ ] **Step 2: Run to verify it fails**

Run: `pytest tests/test_analysis/test_injury_stress.py::test_run_stress_test_ranks_and_flags -v`
Expected: FAIL with `ImportError: cannot import name 'run_stress_test'`.

- [ ] **Step 3: Implement the dataclasses + orchestration**

Add to `injury_stress.py`:
```python
import itertools

from fantasy_baseball.scoring import _classify_roster


@dataclass(frozen=True)
class PlayerExposure:
    name: str
    player_type: str
    win_pct_cost: float


@dataclass(frozen=True)
class PairExposure:
    name_a: str
    name_b: str
    joint_cost: float
    super_additive: float


@dataclass(frozen=True)
class StressResult:
    baseline_win_pct: float
    availability_off_win_pct: float
    projected_margin: float
    health: HealthProbs
    singles: list[PlayerExposure]
    pairs: list[PairExposure]
    threshold: float
    n_iter: int
    seed: int


def run_stress_test(
    inputs: McInputs,
    *,
    threshold: float = SIGNIFICANT_TIME_THRESHOLD,
    pair_top_k: int = PAIR_TOP_K,
    n_iter: int = DEFAULT_N_ITER,
    seed: int = SEED,
) -> StressResult:
    me = inputs.team_rosters[inputs.user_team_name]
    base = win_pct(inputs, me, n_iter=n_iter, seed=seed)
    avail_off = win_pct(inputs, me, availability_variance_off=True, n_iter=n_iter, seed=seed)

    active, _il, _bench = _classify_roster(me)
    health = health_probabilities(active, inputs.fraction_remaining, threshold=threshold, seed=seed)

    singles: list[PlayerExposure] = []
    for p in active:
        wp = win_pct(inputs, substitute_replacement(me, [p.name]), n_iter=n_iter, seed=seed)
        singles.append(PlayerExposure(p.name, p.player_type.value, base - wp))
    singles.sort(key=lambda e: e.win_pct_cost, reverse=True)

    cost_by_name = {e.name: e.win_pct_cost for e in singles}
    top = singles[:pair_top_k]
    pairs: list[PairExposure] = []
    for a, b in itertools.combinations(top, 2):
        wp = win_pct(inputs, substitute_replacement(me, [a.name, b.name]), n_iter=n_iter, seed=seed)
        joint = base - wp
        pairs.append(PairExposure(a.name, b.name, joint,
                                  joint - (cost_by_name[a.name] + cost_by_name[b.name])))
    pairs.sort(key=lambda e: e.joint_cost, reverse=True)

    return StressResult(
        baseline_win_pct=base, availability_off_win_pct=avail_off,
        projected_margin=inputs.projected_margin, health=health, singles=singles,
        pairs=pairs, threshold=threshold, n_iter=n_iter, seed=seed,
    )
```
Confirm `_classify_roster` returns `(active, il, bench)` of `Player` objects (it does -- `mc_roster.build_effective_roster:89` uses exactly this). `player_type.value` is the string ("hitter"/"pitcher").

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_analysis/test_injury_stress.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/fantasy_baseball/analysis/injury_stress.py tests/test_analysis/test_injury_stress.py
git commit -m "feat(injury-stress): stress-test orchestration (headline + singles + pairs)"
```

---

## Task 6: ASCII report renderer

Render a `StressResult` to a plain-ASCII report with the five sections, including the explicit generic-injury-risk note (no silent caps).

**Files:**
- Modify: `src/fantasy_baseball/analysis/injury_stress.py`
- Test: `tests/test_analysis/test_injury_stress.py`

**Interfaces:**
- Produces: `render_report(result: StressResult) -> str`.

- [ ] **Step 1: Write the failing test**

```python
def test_render_report_is_ascii_and_has_sections():
    from fantasy_baseball.analysis.injury_stress import render_report, run_stress_test
    res = run_stress_test(_synth_inputs(), n_iter=200, pair_top_k=4)
    text = render_report(res)
    text.encode("ascii")  # raises if any non-ASCII slipped in
    for marker in ["WHAT INJURY RISK COSTS", "STAYS HEALTHY", "MOST EXPOSED",
                   "LOSING TWO", "generic"]:
        assert marker.lower() in text.lower()
    assert "Star" in text
```

- [ ] **Step 2: Run to verify it fails**

Run: `pytest tests/test_analysis/test_injury_stress.py::test_render_report_is_ascii_and_has_sections -v`
Expected: FAIL with `ImportError: cannot import name 'render_report'`.

- [ ] **Step 3: Implement the renderer**

Add to `injury_stress.py`:
```python
def _pct(x: float) -> str:
    return f"{x:5.1f}%"


def render_report(result: StressResult) -> str:
    r = result
    lines: list[str] = []
    lines.append("=" * 72)
    lines.append("INJURY STRESS-TEST")
    lines.append("=" * 72)

    lines.append("")
    lines.append("1. WHAT INJURY RISK COSTS YOU")
    lines.append("-" * 72)
    margin = r.projected_margin
    side = "ahead of" if margin >= 0 else "behind"
    lines.append(f"  Deterministic projected roto margin : {margin:+.1f} pts ({side} the leader)")
    lines.append(f"  Win% if availability lands as expected: {_pct(r.availability_off_win_pct)}")
    lines.append(f"  Win% (real, with injury risk)         : {_pct(r.baseline_win_pct)}")
    lines.append(f"  -> Injury/availability risk costs you : "
                 f"{r.availability_off_win_pct - r.baseline_win_pct:+.1f} win pts")

    lines.append("")
    lines.append("2. HOW LIKELY IS EVERYONE STAYS HEALTHY?")
    lines.append("-" * 72)
    thr = int(round(r.threshold * 100))
    lines.append(f"  (a player 'loses significant time' = >= {thr}% below expected playing time)")
    lines.append(f"  P(no active player loses significant time): {_pct(r.health.p_all_healthy * 100)}")
    lines.append(f"  P(exactly one does)                       : {_pct(r.health.p_one * 100)}")
    lines.append(f"  P(two or more)                            : {_pct(r.health.p_two_plus * 100)}")

    lines.append("")
    lines.append("3. WHO ARE YOU MOST EXPOSED TO? (lose one, replaced)")
    lines.append("-" * 72)
    lines.append(f"  {'Player':<24}{'Type':<9}{'win% cost':>10}")
    for e in r.singles:
        lines.append(f"  {e.name[:23]:<24}{e.player_type:<9}{e.win_pct_cost:>9.1f}")

    lines.append("")
    lines.append("4. LOSING TWO (top exposures, ranked by joint win% cost)")
    lines.append("-" * 72)
    lines.append(f"  {'Pair':<40}{'joint':>8}{'vs sum':>9}")
    for p in r.pairs:
        pair = f"{p.name_a[:18]} + {p.name_b[:18]}"
        tag = "  (worse than additive)" if p.super_additive > 0.5 else ""
        lines.append(f"  {pair:<40}{p.joint_cost:>8.1f}{p.super_additive:>+9.1f}{tag}")

    lines.append("")
    lines.append("5. NOTE")
    lines.append("-" * 72)
    lines.append("  Section 2 uses a GENERIC (volume/role) injury model -- every player in a")
    lines.append("  PA/IP band shares the same downside. Per-player injury history is not yet")
    lines.append("  modeled (deferred; see the design's Future work).")
    lines.append(f"  MC: n_iter={r.n_iter}, seed={r.seed} (common random numbers across scenarios).")
    return "\n".join(lines)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_analysis/test_injury_stress.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/fantasy_baseball/analysis/injury_stress.py tests/test_analysis/test_injury_stress.py
git commit -m "feat(injury-stress): ASCII report renderer"
```

---

## Task 7: Live Upstash input loader

Assemble `McInputs` from stored Upstash blobs -- no Yahoo pipeline. Follows `scripts/compare_eroto_mc_means.py` for reading the cache envelope, and recomputes `eos_baseline`/`team_sds` with the vintage `fraction_remaining` (so the baseline matches the dashboard's stored MC).

**Files:**
- Modify: `src/fantasy_baseball/analysis/injury_stress.py`
- Test: manual/live (Task 9); the pure helper `build_actual_standings` gets a unit test here.

**Interfaces:**
- Produces:
  - `build_actual_standings(standings) -> dict[str, dict[str, float]]` (pure; testable).
  - `projected_margin_from_eos(eos_baseline, user_team_name) -> float` (pure).
  - `load_mc_inputs_from_upstash(config_path: Path | None = None) -> McInputs`.

- [ ] **Step 1: Write the failing test (pure helpers)**

```python
def test_projected_margin_from_eos_signs_correctly():
    from fantasy_baseball.analysis.injury_stress import projected_margin_from_eos
    inp = _synth_inputs()
    m = projected_margin_from_eos(inp.eos_baseline, "Me")
    assert isinstance(m, float)   # 2-team synthetic: sign follows Me-vs-Opp roto totals
```

- [ ] **Step 2: Run to verify it fails**

Run: `pytest tests/test_analysis/test_injury_stress.py::test_projected_margin_from_eos_signs_correctly -v`
Expected: FAIL with `ImportError: cannot import name 'projected_margin_from_eos'`.

- [ ] **Step 3: Implement the loader + helpers**

Idioms confirmed from `scripts/compare_eroto_mc_means.py` and the source: `CacheKey`/`redis_key` live in `fantasy_baseball.data.cache_keys`; the cache envelope is unwrapped by taking `o["_data"]`; `score_roto_dict` (`scoring.py:1519`) accepts `CategoryStats` values directly (signature `Mapping[str, Mapping[str,float] | CategoryStats]`), so NO CategoryStats->dict conversion is needed; `redis_store.get_latest_standings(client)` returns a `Standings` object with `.entries` (each has `.team_name`, `.stats` (CategoryStats), `.extras`, `.ytd_components()`).

Add:
```python
import json
import math
from pathlib import Path

from fantasy_baseball.models.standings import build_eos_baseline
from fantasy_baseball.scoring import build_team_sds, score_roto_dict
from fantasy_baseball.utils.constants import AB_PER_PA, OpportunityStat


def build_actual_standings(standings) -> dict[str, dict[str, float]]:
    """{team: {R..WHIP, IP, AB}} -- mirrors refresh_pipeline.py:1451-1460 exactly
    (AB from PA * AB_PER_PA), so it matches what the dashboard MC consumed."""
    out: dict[str, dict[str, float]] = {}
    for e in standings.entries:
        row = e.stats.to_dict()
        ip = e.extras.get(OpportunityStat.IP)
        pa = e.extras.get(OpportunityStat.PA)
        if ip is not None:
            row["IP"] = float(ip)
        if pa is not None:
            row["AB"] = float(pa) * AB_PER_PA
        out[e.team_name] = row
    return out


def projected_margin_from_eos(eos_baseline: dict, user_team_name: str) -> float:
    """Signed deterministic projected roto margin: user total minus the best other
    team's total. score_roto_dict accepts the {team: CategoryStats} baseline directly."""
    roto = score_roto_dict(eos_baseline)
    user = roto[user_team_name]["total"]
    others = [v["total"] for t, v in roto.items() if t != user_team_name]
    return float(user - max(others)) if others else float(user)


def load_mc_inputs_from_upstash(config_path: Path | None = None) -> McInputs:
    """Assemble the ROS-MC inputs from STORED (last-refresh vintage) Upstash blobs;
    no Yahoo call. eos_baseline/team_sds are recomputed on the stored
    fraction_remaining so the baseline matches the dashboard's stored MC. (Minor AVG
    drift is possible vs the pipeline's un-persisted ownership-attributed team-AB
    overlay; it cancels in the scenario deltas, which are the deliverable.)"""
    from fantasy_baseball.config import load_config
    from fantasy_baseball.data.cache_keys import CacheKey, redis_key
    from fantasy_baseball.data.kv_store import build_explicit_upstash_kv
    from fantasy_baseball.data.redis_store import get_latest_standings
    from fantasy_baseball.models.player import Player
    from fantasy_baseball.models.positions import BENCH_SLOTS
    from fantasy_baseball.sgp.denominators import get_sgp_denominators

    root = Path(__file__).resolve().parents[3]
    cfg = load_config(config_path or (root / "config" / "league.yaml"))
    kv = build_explicit_upstash_kv()

    def cache(key: CacheKey):  # unwrap the {"_meta","_data"} envelope (cache:* blobs)
        raw = kv.get(redis_key(key))
        if raw is None:
            raise RuntimeError(f"Upstash missing {key}; run a refresh first.")
        o = json.loads(raw) if isinstance(raw, str) else raw
        return o["_data"] if isinstance(o, dict) and "_data" in o else o

    user_blob = cache(CacheKey.ROSTER)
    opp_blob = cache(CacheKey.OPP_ROSTERS)
    proj_blob = cache(CacheKey.PROJECTIONS)

    user_players = [Player.from_dict(p) for p in user_blob]
    opp_players = {t: [Player.from_dict(p) for p in r] for t, r in opp_blob.items()}
    team_rosters = {cfg.team_name: user_players, **opp_players}

    standings = get_latest_standings(kv)   # Standings object (not a cache:* envelope)
    actual_standings = build_actual_standings(standings)

    fr = float(proj_blob["fraction_remaining"])   # vintage -> matches dashboard MC
    denoms = get_sgp_denominators(cfg.sgp_overrides)

    ytd_by_team = {e.team_name: e.ytd_components() for e in standings.entries}
    eos_baseline = build_eos_baseline(team_rosters, ytd_by_team)
    team_sds = build_team_sds(team_rosters, math.sqrt(fr))

    non_hitter = {str(s) for s in BENCH_SLOTS} | {"P"}
    h_slots = sum(v for k, v in cfg.roster_slots.items() if k not in non_hitter)
    p_slots = cfg.roster_slots.get("P", 9)

    return McInputs(
        team_rosters=team_rosters, actual_standings=actual_standings,
        fraction_remaining=fr, h_slots=h_slots, p_slots=p_slots,
        eos_baseline=eos_baseline, team_sds=team_sds, denoms=denoms,
        user_team_name=cfg.team_name,
        projected_margin=projected_margin_from_eos(eos_baseline, cfg.team_name),
    )
```
Confirm `CacheKey` has `ROSTER`, `OPP_ROSTERS`, `PROJECTIONS` members (compare_eroto_mc_means.py uses exactly these). If `get_latest_standings` needs a different arg name, check its signature in `redis_store.py:348`.

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_analysis/test_injury_stress.py -v`
Expected: PASS (the pure-helper test; the live loader is exercised in Task 9).

- [ ] **Step 5: Commit**

```bash
git add src/fantasy_baseball/analysis/injury_stress.py tests/test_analysis/test_injury_stress.py
git commit -m "feat(injury-stress): live Upstash MC-input loader"
```

---

## Task 8: CLI script

Thin entry point: parse args, load inputs, run the stress test, print (and optionally write) the ASCII report.

**Files:**
- Create: `scripts/injury_stress_test.py`

**Interfaces:**
- Consumes: `injury_stress.load_mc_inputs_from_upstash`, `run_stress_test`, `render_report`.

- [ ] **Step 1: Write the script**

```python
"""Injury stress-test CLI: how robust is my ROS lead to injuries?

Reads live season state from remote Upstash and prints a ranked report:
headline attribution, P(everyone healthy), single + pair lose-a-player
counterfactuals. See docs/superpowers/specs/2026-07-22-injury-stress-test-design.md.

Usage: python scripts/injury_stress_test.py [--threshold 0.20] [--pair-top-k 8]
                                            [--n-iter 1000] [--out report.md]
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from fantasy_baseball.analysis.injury_stress import (  # noqa: E402
    PAIR_TOP_K, SIGNIFICANT_TIME_THRESHOLD, load_mc_inputs_from_upstash,
    render_report, run_stress_test,
)


def main() -> int:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")  # names may be non-ASCII
    ap = argparse.ArgumentParser(description="Injury stress-test for the ROS lead.")
    ap.add_argument("--threshold", type=float, default=SIGNIFICANT_TIME_THRESHOLD)
    ap.add_argument("--pair-top-k", type=int, default=PAIR_TOP_K)
    ap.add_argument("--n-iter", type=int, default=1000)
    ap.add_argument("--out", type=str, default=None, help="write the report to this path")
    args = ap.parse_args()

    print("Loading live season state from Upstash ...", file=sys.stderr)
    inputs = load_mc_inputs_from_upstash()
    print(f"Running stress test ({args.n_iter} iters) ...", file=sys.stderr)
    result = run_stress_test(
        inputs, threshold=args.threshold, pair_top_k=args.pair_top_k, n_iter=args.n_iter
    )
    report = render_report(result)
    print(report)
    if args.out:
        Path(args.out).write_text(report + "\n", encoding="utf-8")
        print(f"\nWrote {args.out}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
```

- [ ] **Step 2: Verify it imports cleanly (no live call)**

Run: `python -c "import ast; ast.parse(open('scripts/injury_stress_test.py').read())"`
Expected: no output (parses).

- [ ] **Step 3: Commit**

```bash
git add scripts/injury_stress_test.py
git commit -m "feat(injury-stress): CLI script"
```

---

## Task 9: Benchmark, live run, and end-of-effort verification

Measure one MC run, gate the runtime, do a real end-to-end run against Upstash, and run all project checks.

**Files:** none (verification only), unless a fix is needed.

- [ ] **Step 1: Benchmark one 1k-iteration run against live Upstash**

Run:
```bash
python - <<'PY'
import sys, time
from pathlib import Path
sys.path.insert(0, str(Path("src").resolve()))
from fantasy_baseball.analysis.injury_stress import load_mc_inputs_from_upstash, win_pct
inp = load_mc_inputs_from_upstash()
me = inp.team_rosters[inp.user_team_name]
t = time.perf_counter(); wp = win_pct(inp, me, n_iter=1000); dt = time.perf_counter() - t
n_active = len(__import__("fantasy_baseball.scoring", fromlist=["_classify_roster"])._classify_roster(me)[0])
runs = 2 + n_active + (min(n_active, 8) * (min(n_active, 8) - 1) // 2)
print(f"one run: {dt:.2f}s  baseline win%: {wp}  est total runs: {runs}  est wall: {runs*dt:.0f}s")
PY
```
Expected: prints a per-run time and an estimated total. **Runtime gate:** if `runs * dt > 180s` (~3 min), STOP and report the measurement to the user for a scope decision (per the spec's Runtime section) rather than proceeding to the pre-rank optimization (which is out of v1 scope).

- [ ] **Step 2: Full end-to-end run**

Run: `python scripts/injury_stress_test.py --out injury_stress_report.md`
Expected: a complete ASCII report to stdout with all five sections and a written `injury_stress_report.md`. Read it and sanity-check: baseline win% is in a plausible range, the singles list is ranked with the biggest bats on top, availability-off win% >= baseline win%.

- [ ] **Step 3: Run the full test suite**

Run: `pytest tests/test_analysis/test_injury_stress.py tests/test_mc_integration.py tests/test_simulation.py -v`
Then the whole suite: `pytest -n auto`
Expected: all pass. If anything in the existing MC suite changed, the flag threading regressed the default path -- fix the code, not the test.

- [ ] **Step 4: Lint, format, dead-code, types**

Run each and fix every finding:
```bash
ruff check .
ruff format --check .
vulture            # no NEW findings from injury_stress.py / the simulation edits
```
Check `pyproject.toml` `[tool.mypy].files`: if `src/fantasy_baseball/simulation.py` or `src/fantasy_baseball/analysis/` is listed, run `mypy` and fix. (`_replacement_line`, `_full_season_pt_volume`, `_classify_roster`, `_sv_role_mu` are private imports -- if mypy or ruff flags the private import, add a targeted `# noqa`/type-ignore consistent with how the repo already imports these privates in `mc_roster.py`/`mc_fill.py`.)

- [ ] **Step 5: Commit any fixes**

```bash
git add -A
git commit -m "chore(injury-stress): verification fixes (lint/format/types)"
```

- [ ] **Step 6: Report results**

In the final message, show the exact commands run and their output (test counts, ruff/vulture/mypy results, and the benchmark timing + gate decision). Do not claim completion without this evidence.

---

## Notes for the implementer

- **Reconciliation:** the baseline uses the stored (last-refresh vintage) rosters/standings + the vintage `fraction_remaining`, and recomputes `eos_baseline`/`team_sds` on that same `fr`. It will match the dashboard's stored MC closely; a small AVG drift is possible because the pipeline's ownership-attributed team-AB overlay is not persisted. This cancels in the scenario deltas (all scenarios share `eos_baseline`), which are the deliverable.
- **Common random numbers:** every `win_pct` call uses `seed=42` and preserves team count/order, so the RNG draws align and the win% deltas are low-noise. Do not vary the seed between baseline and counterfactuals.
- **Pitcher vs hitter replacement asymmetry (intended):** hitters keep their real bench-fill on top of the substituted line; pitchers get only the generic replacement arm (the engine has no pitcher bench-fill). This matches how the sim already treats a lost pitcher; the Section 5 note and the design document it.
- **Out of v1 scope:** analytic deltaRoto pre-rank (only if the Task 9 benchmark trips the gate), and per-player injury propensity (design Future work).
