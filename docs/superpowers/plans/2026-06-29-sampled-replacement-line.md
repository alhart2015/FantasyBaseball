# Sampled Replacement-Fill Line Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Sample the terminal replacement-fill term in the ROS-direct hitter Monte Carlo with the same mean-neutral per-game-rate model the bench bodies use, so the hitter counting-category SD under-dispersion (R/RBI ~0.70-0.75 of analytic) is corrected while team-total means stay neutral.

**Architecture:** Two code tasks on the existing `mc-sampled-bench-fill` branch. Task 1 is a behavior-preserving contract flip: move the deterministic full-season -> per-game replacement conversion out of the pure `allocate_bench_fill` and into the caller (`_repl_for`), so the allocator's `replacement_for` callback now returns a PER-GAME line. Byte-identical -- no number moves. Task 2 adds the actual sampling: a batched `_apply_variance_batch` draw over each active's synthetic replacement body (appended after the bench draw), decomposed to a mean-neutral per-game rate, replacing the deterministic `_repl_for`. Tasks 3-4 are verification and the acceptance gate.

**Tech Stack:** Python, numpy, scipy (NegBin copula), pytest. Reuses `_apply_variance_batch`, `_replacement_line`, `allocate_bench_fill`, `PA_PER_GAME`, `AB_PER_PA`.

## Global Constraints

- ASCII-only in all source, comments, strings, and docs (Windows cp1252 stdout; no Unicode minus/dashes/arrows/smart-quotes).
- Player IDs are `name::player_type`; never key on bare names.
- No `x or default` for numeric defaults; use `is not None` / explicit `> eps`. The replacement EPS guard is `eps = 1e-9`.
- mypy covers `simulation.py` and `mc_fill.py` (in `[tool.mypy].files`). Keep them clean. NOTE: a FULL `mypy` run fails on a pre-existing stale `category_odds.py` entry (a deleted file still listed in `pyproject.toml`) -- unrelated to this work; verify by running mypy on the two touched files directly.
- Don't loosen/skip/delete a failing test to make it pass. Re-pin a golden ONLY when the change legitimately shifts it (rng-stream shift, intended variance increase), and record the reason. A test asserting a true invariant that breaks is a real regression -- STOP and fix the code.
- The mean-neutral denominator MUST be the UNCAPPED `repl_implied_games * scale`. A denominator capped at `repl_implied_games` reintroduces an upward mean bias -- forbidden.
- `repl_implied_games = repl_ab / PA_PER_GAME` (mean-consistency with the old deterministic path). The curve volume is `repl_pa_volume = repl_ab / AB_PER_PA` (consistent with `_full_season_pt_volume`). `repl_ab` is the replacement line's `ab` field (AT-BATS).
- The sampled replacement line gets NO displacement `factor` (replacement is the healthy-scrub floor; the deterministic path applies no factor).
- `_replacement_line` returns SHARED read-only module constants -- never mutate a flat-dict on this path.
- Spec: `docs/superpowers/specs/2026-06-29-sampled-replacement-line-design.md`. This plan implements it.

---

## File Structure

- `src/fantasy_baseball/mc_fill.py` -- `allocate_bench_fill`: `replacement_for` becomes a per-game contract; residual branch multiplies by `need` (conversion deleted); docstring rewritten. (Task 1)
- `src/fantasy_baseball/simulation.py` -- `_simulate_team_hitters_ros_direct`: `_repl_for` first inline-converts (Task 1, byte-identical), then is replaced by the sampled draw + mean-neutral per-game decomposition (Task 2). Import `PA_PER_GAME`. (Tasks 1, 2)
- `tests/test_mc_fill.py` -- per-game `replacement_for` contract: update factories + retarget the conversion test. (Task 1)
- `tests/test_mc_integration.py` -- new replacement-sampling de-bias test; re-pin shifted goldens + the SD ceiling. (Task 2)
- `docs/superpowers/sampled-replacement-line-sd-evidence-2026-06-29.md` -- acceptance evidence. (Task 4)

---

## Task 1: Behavior-preserving `replacement_for` per-game contract flip

Move the full-season -> per-game conversion out of the pure allocator into the caller. NOTHING changes numerically: this is a pure refactor that the byte-identical de-bias and anchor tests must confirm.

**Files:**
- Modify: `src/fantasy_baseball/mc_fill.py` (the `allocate_bench_fill` residual branch ~94-106 + docstring ~44-55)
- Modify: `src/fantasy_baseball/simulation.py` (`_repl_for` ~927-928; the `mc_roster` import ~17)
- Test: `tests/test_mc_fill.py` (factories ~51-58; `test_replacement_per_game_not_overscaled` ~96-106; `test_position_mismatch_routes_to_replacement_not_bench` ~83-93; `test_residual_goes_to_replacement_when_bench_exhausted` ~172-180)

**Interfaces:**
- Consumes: `allocate_bench_fill(actives, benches, replacement_for)` where `replacement_for: Callable[[ActiveBody], dict[str, float]]`.
- Produces: the SAME callable signature, but `replacement_for` now returns a PER-GAME counting line (per-game stat values), and the allocator multiplies it by `need` directly (no internal `repl_ab/PA_PER_GAME` conversion). Task 2 consumes this contract.

- [ ] **Step 1: Retarget the allocator-contract test in `tests/test_mc_fill.py`**

The current `test_replacement_per_game_not_overscaled` asserts the allocator's now-deleted conversion. Replace it with a test of the NEW contract (the allocator multiplies a per-game line by `need`, no re-conversion). Also update `_realistic_replacement` to return a PER-GAME line (the per-game equivalent of the old full-season `r=43/ab=516`, i.e. `43/(516/4.3) = 0.358` r/game), and adjust the two tests that consume it.

Replace the `_realistic_replacement` factory (currently returns a full-season line) with a per-game line:

```python
def _realistic_replacement(_b):
    # A real per-game replacement line (the per-game form of a ~43-R / ~516-AB
    # full-season scrub: 43 / (516 / 4.3) = 0.358 r/game). Under the per-game
    # replacement_for contract the allocator multiplies this by games_missed
    # directly -- no implied-games conversion.
    return _line(r=0.358, hr=0.10, rbi=0.375, sb=0.033, h=1.0, ab=4.3)
```

Replace `test_replacement_per_game_not_overscaled` with a DISCRIMINATING test of
the new contract. CRITICAL: a *valid* per-game line has `ab == PA_PER_GAME` (4.3)
by construction (per-game AB = full-season AB / implied-games), and on such a line
the OLD allocator's `repl_ab/PA_PER_GAME` conversion is the IDENTITY -- so a valid
per-game line CANNOT distinguish the old vs new allocator. To get a fail-first,
discriminating test, probe with a line whose `ab != PA_PER_GAME` and assert the
allocator multiplies the line by `need` WITHOUT re-dividing by `ab`:

```python
def test_allocator_does_not_convert_per_game_replacement_line():
    # New contract: replacement_for returns a PER-GAME line; the allocator
    # multiplies by games_missed directly and must IGNORE the line's `ab` field
    # (no re-division by ab/PA_PER_GAME). Probe with ab=8.6 (!= PA_PER_GAME=4.3)
    # so old and new diverge: OLD converts -> 1.0/(8.6/4.3)*10 = 5.0; NEW
    # multiplies -> 1.0*10 = 10.0.
    a = _active("OFstar", "1", g_ros_adj=20.0, pos=Position.OF)
    res = allocate_bench_fill(
        [ActiveSample(a, frac_missed=0.5)],  # 0.5 * 20 = 10 games missed
        [],
        lambda _b: _line(r=1.0, ab=8.6),
    )
    assert abs(res.fill_counts["r"] - 10.0) < 1e-9
```

Update `test_position_mismatch_routes_to_replacement_not_bench` (the catcher bench can't fill an OF shortfall, so the OF routes to replacement): with the per-game `_realistic_replacement` (0.358 r/game) and 40 games missed, the fill is `0.358 * 40 ~= 14.3`. Change its assertion to:

```python
    # bench (C) cannot fill an OF shortfall -> fill is per-game replacement
    # (0.358 r/game * 40 games ~= 14.3), NOT the catcher's 99/game.
    assert 0.0 < res.fill_counts["r"] < 100.0
```

(The `< 100.0` bound already holds; the comment is the substantive change. Keep the assertion.)

`test_residual_goes_to_replacement_when_bench_exhausted` uses `_flat_replacement(0.5)` and only asserts `res.fill_counts["r"] > 0`. Under the per-game contract `_flat_replacement(0.5)` returns `{col: 0.5}` interpreted as per-game, and the allocator does `0.5 * need > 0`. The assertion still holds; no change needed, but update its comment:

```python
    [_bench_sample(b, {"r": 0.0})],  # bench gives 0 r -> all r from per-game replacement
    _flat_replacement(0.5),  # 0.5 r/game (per-game contract)
```

- [ ] **Step 2: Run the discriminating test to verify it FAILS-first (allocator still converts)**

Run: `pytest tests/test_mc_fill.py::test_allocator_does_not_convert_per_game_replacement_line -v`
Expected: FAIL -- the OLD allocator divides by `repl_ab/PA_PER_GAME = 8.6/4.3 = 2.0`, so it returns `fill r = 1.0/2.0 * 10 = 5.0`, not the asserted `10.0`. This is the ONLY retargeted test that discriminates old vs new (the `_realistic_replacement`-based tests use a valid per-game line with `ab=4.3`, on which the conversion is identity, so they pass under both and serve as regression guards, not the red gate).

- [ ] **Step 3: Flip the allocator residual branch in `mc_fill.py`**

Replace the residual branch (currently `~94-106`):

```python
        if need > 0.0:
            # The replacement line is a FULL-SEASON counting bundle with NO games
            # field, so convert to per-game by dividing each stat by the line's
            # IMPLIED games (ab / PA_PER_GAME -- the shared per-game heuristic),
            # NOT by PA_PER_GAME directly (that would treat a ~65-R full-season
            # total as a per-game rate, ~30x too high).
            repl = replacement_for(a.body)
            repl_ab = repl.get("ab", 0.0) or 0.0
            repl_games = (repl_ab / PA_PER_GAME) if repl_ab > 0 else 0.0
            for col in HITTING_COUNTING:
                total = repl.get(col, 0.0) or 0.0
                per_game = (total / repl_games) if repl_games > 0 else 0.0
                fill[col] += per_game * need
```

with:

```python
        if need > 0.0:
            # replacement_for returns a PER-GAME line (the caller owns the
            # implied-games conversion / sampling); multiply by the residual games.
            repl_pg = replacement_for(a.body)
            for col in HITTING_COUNTING:
                fill[col] += repl_pg.get(col, 0.0) * need
```

Then remove the now-unused `PA_PER_GAME` import from `mc_fill.py` if it is no longer referenced (check: `from fantasy_baseball.mc_roster import PA_PER_GAME, ActiveBody, BenchBody` -> drop `PA_PER_GAME` if grep shows no other use in the file), and update the `allocate_bench_fill` docstring (`~44-55`): replace the sentence describing the `repl_ab / PA_PER_GAME` conversion with the per-game contract:

```python
    """Allocate missed games to bench (value-ordered, capped) then replacement.

    games_missed = frac_missed * g_ros_adj (the reduced baseline -- the cap;
    NEVER g_ros_full). Largest shortfalls first; per shortfall pick the highest
    per_game_value position-eligible bench body with remaining capacity, assign
    min(shortfall, remaining), decrement both; residual -> the caller-supplied
    PER-GAME replacement line times the residual games (replacement_for returns a
    per-game counting line; the allocator does NOT convert from full-season). One
    bench body's total assigned games <= its per-iteration ``capacity``
    (g_ros_full * sampled scale; CAN exceed g_ros_full when sampled more available
    than projected). Tie-break: higher per_game_value, then player-id ascending.
    """
```

- [ ] **Step 4: Convert `_repl_for` in `simulation.py` to return a per-game line (byte-identical to the old allocator conversion)**

Add `PA_PER_GAME` to the mc_roster import (`~17`):

```python
from fantasy_baseball.mc_roster import PA_PER_GAME, ActiveBody, BenchBody, EffectiveRoster
```

Replace `_repl_for` (`~927-928`):

```python
    def _repl_for(ab: ActiveBody) -> dict[str, float]:
        return _replacement_line(ab.player.to_flat_dict(), is_hitter=True)
```

with the inline per-game conversion (the SAME math the allocator used to do, moved here -- byte-identical result):

```python
    def _repl_for(ab: ActiveBody) -> dict[str, float]:
        # Per-game replacement line: full-season counting bundle / implied games
        # (repl_ab / PA_PER_GAME). The allocator now multiplies this by the
        # residual games. Byte-identical to the conversion the allocator used to
        # do. (Task 2 replaces this with the SAMPLED per-game rate.)
        repl = _replacement_line(ab.player.to_flat_dict(), is_hitter=True)
        repl_ab = repl.get("ab", 0.0)
        repl_games = (repl_ab / PA_PER_GAME) if repl_ab > 0 else 0.0
        if repl_games <= 0.0:
            return {col: 0.0 for col in HITTING_COUNTING}
        return {col: repl.get(col, 0.0) / repl_games for col in HITTING_COUNTING}
```

- [ ] **Step 5: Run the allocator tests + the byte-identical guard**

Run: `pytest tests/test_mc_fill.py tests/test_mc_integration.py::test_variance_batch_default_matches_legacy_columns -v`
Expected: all PASS. The `test_mc_fill.py` suite passes the new per-game contract; the legacy byte-anchor is untouched.

- [ ] **Step 6: Run the full MC integration + simulation suites to confirm BYTE-IDENTICAL behavior**

Run: `pytest tests/test_mc_integration.py tests/test_simulation.py -v`
Expected: all PASS with NO golden re-pinning. Task 1 moves WHERE the conversion happens, not the numbers -- if any team-total golden or the de-bias SD value shifts, the conversion was changed (a bug), not just moved. STOP and fix.

- [ ] **Step 7: Lint/type the touched files**

Run: `ruff check src/fantasy_baseball/mc_fill.py src/fantasy_baseball/simulation.py tests/test_mc_fill.py && ruff format --check src/fantasy_baseball/mc_fill.py src/fantasy_baseball/simulation.py && mypy src/fantasy_baseball/mc_fill.py src/fantasy_baseball/simulation.py`
Expected: clean (run `ruff format` to fix drift; full `mypy` fails only on the pre-existing stale `category_odds.py`).

- [ ] **Step 8: Commit**

```bash
git add src/fantasy_baseball/mc_fill.py src/fantasy_baseball/simulation.py tests/test_mc_fill.py
git commit -m "refactor(mc): per-game replacement_for contract; conversion moves to caller (replacement-line task 1)

Behavior-preserving: allocate_bench_fill's replacement_for now returns a per-game
line and the allocator multiplies by games_missed (the repl_ab/PA_PER_GAME
conversion moves into _repl_for). Byte-identical -- de-bias SD + anchors unchanged.
Sets up Task 2 to swap _repl_for for a sampled per-game rate.

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

## Task 2: Sampled replacement-body draw + mean-neutral per-game rate

Replace the deterministic `_repl_for` with a per-iteration SAMPLED per-game rate, drawn from a batched `_apply_variance_batch` over each active's synthetic replacement body. This injects the missing variance into the ~77% of fill that is replacement.

**Files:**
- Modify: `src/fantasy_baseball/simulation.py` (`_simulate_team_hitters_ros_direct`: add the replacement draw after the bench draw `~925`; precompute the per-game rate arrays; replace `_repl_for` `~927`; update the helper docstring `~890-895`)
- Test: `tests/test_mc_integration.py` (new de-bias test; re-pin shifted goldens + the SD ceiling `~695`)

**Interfaces:**
- Consumes: the per-game `replacement_for` contract from Task 1; `_apply_variance_batch(players, player_type, rng, fraction_remaining, n_iter, *, pt_mean_fraction, suppress_repl, pt_volumes) -> VarianceBatch` (with `.counts`, `.scales`); `_replacement_line`, `PA_PER_GAME`, `AB_PER_PA`, `HITTING_COUNTING`, `PlayerType`.
- Produces: a `_simulate_team_hitters_ros_direct` whose replacement fill is stochastic and mean-neutral.

- [ ] **Step 1: Add the BEFORE de-bias baseline test (placeholder constant) to `tests/test_mc_integration.py`**

This mirrors the bench task's fail-first pattern. Add a test that builds a deep roster (actives that shed games + an EMPTY bench, so ALL fill is replacement), runs `_simulate_team_hitters_ros_direct`, and asserts the team-total R SD STRICTLY exceeds a pinned deterministic baseline. Use the existing `_bench_deep_roster` / `_eff_roster` helpers and seeding pattern from the bench de-bias test (`test_sampled_bench_fill_widens_team_total_sd_vs_deterministic_baseline`). The baseline constants start at 0.0 and are captured in Step 2.

```python
# Captured in Step 2 from the Task-1 (deterministic-replacement) code on
# _empty_bench_shedding_roster() at seed=7, fr=0.2, n_iter=4000.
_DET_REPL_R_SD = 0.0
_DET_REPL_RBI_SD = 0.0
_DET_REPL_R_MEAN = 0.0


def test_sampled_replacement_fill_widens_team_total_sd_vs_deterministic_baseline():
    # Actives shed games, bench is EMPTY -> 100% of fill is the replacement line.
    # Sampling it must STRICTLY widen team-total R/RBI SD vs the deterministic
    # baseline pinned from the pre-change (Task 1) code.
    eff = _empty_bench_shedding_roster()  # defined in Step 1b
    assert len(eff.bench) == 0 and len(eff.active) == 3  # fixture: all fill -> replacement
    out = _simulate_team_hitters_ros_direct(eff, 0.2, np.random.default_rng(7), 4000)
    assert np.all(np.isfinite(out["R"])) and np.all(np.isfinite(out["RBI"]))
    assert out["R"].std() > _DET_REPL_R_SD
    assert out["RBI"].std() > _DET_REPL_RBI_SD


def test_sampled_replacement_fill_is_mean_neutral():
    # Mean-neutrality (gate #3's load-bearing property, at unit scale): sampling
    # the replacement line must NOT drift the team-total R mean UP vs the
    # deterministic baseline -- the only upward mechanism is a capped denominator,
    # which is forbidden. A small downward drift (eps-guard mass) is allowed.
    eff = _empty_bench_shedding_roster()
    out = _simulate_team_hitters_ros_direct(eff, 0.2, np.random.default_rng(7), 4000)
    assert out["R"].mean() <= _DET_REPL_R_MEAN * 1.01  # no upward drift > 1%
    assert out["R"].mean() >= _DET_REPL_R_MEAN * 0.90  # not grossly under-filled
```

- [ ] **Step 1b: Add the `_empty_bench_shedding_roster` fixture**

Use the existing `_hitter` + `_eff_roster` helpers. All three hitters are OF-SLOTTED (`Position.OF`), so `build_effective_roster` seats them ALL as active and `eff.bench` is empty -- every missed game routes to the replacement line (the sole fill-variance source). Do NOT mutate `eff.bench` (`EffectiveRoster` is `@dataclass(frozen=True)` -- assignment raises `FrozenInstanceError`); the empty bench comes from having no BN-slotted players. `_hitter` defaults `g=150` (so `rest_of_season.g > 0` -> `g_ros_adj > 0`), and at the low `fraction_remaining=0.2` the active playing-time draw sheds a large fraction of games (`frac_missed = max(0, 1-scale)`), guaranteeing `need > 0`. (The replacement bodies' PT band is set by `repl_ab/AB_PER_PA` regardless of the actives' projections, so the fixture needs no `full_season_projection`.)

```python
def _empty_bench_shedding_roster():
    # All OF-slotted -> build_effective_roster seats them ALL active, bench empty,
    # so 100% of injury-fill routes to the (now sampled) replacement line.
    return _eff_roster(
        [
            _hitter("S1", Position.OF, "1"),
            _hitter("S2", Position.OF, "2"),
            _hitter("S3", Position.OF, "3"),
        ]
    )
```

- [ ] **Step 2: Capture the deterministic baseline (run against the CURRENT Task-1 code) and paste it in**

Run this snippet (Task-1 code is still the deterministic replacement; this captures its team-total SD and MEAN). Use the SAME seed/fr/n_iter as the tests (seed=7, fr=0.2, 4000):

```bash
python -c "import numpy as np; from tests.test_mc_integration import _empty_bench_shedding_roster; from fantasy_baseball.simulation import _simulate_team_hitters_ros_direct; eff=_empty_bench_shedding_roster(); o=_simulate_team_hitters_ros_direct(eff, 0.2, np.random.default_rng(7), 4000); print(repr(o['R'].std()), repr(o['RBI'].std()), repr(o['R'].mean()))"
```

Paste the three printed values into `_DET_REPL_R_SD` / `_DET_REPL_RBI_SD` / `_DET_REPL_R_MEAN`. (The mean baseline anchors `test_sampled_replacement_fill_is_mean_neutral`.)

- [ ] **Step 3: Run the de-bias test to verify it FAILS-first**

Run: `pytest tests/test_mc_integration.py::test_sampled_replacement_fill_widens_team_total_sd_vs_deterministic_baseline -v`
Expected: FAIL -- with the deterministic Task-1 code the SD equals the pinned baseline, so `std() > baseline` is False. This proves the test detects the added variance.

- [ ] **Step 4: Add the sampled replacement draw + per-game rate precompute in `_simulate_team_hitters_ros_direct`**

After the bench draw (`bench_vb = _sample_hitter_bodies(...)`, `~925`) and BEFORE `_repl_for`, insert:

```python
    # Sampled replacement fill (finding #5): each active's position-routed
    # replacement line is sampled like a bench body and the residual `need` is
    # filled at that stochastic, MEAN-NEUTRAL per-game rate (the de-bias for the
    # ~77% of injury-fill that cascades to replacement). Drawn AFTER the bench
    # draw (rng-stream shift is intended; empty-active teams already returned
    # above). suppress_repl=True (no replacement-of-a-replacement); the line's
    # `scale` is divided back out for mean-neutrality (NOT a capacity cap -- the
    # pool is terminal). See the sampled-replacement-line spec (2026-06-29).
    repl_lines = [
        _replacement_line(b.player.to_flat_dict(), is_hitter=True) for b in active_h_bodies
    ]
    repl_ab = np.array([rl.get("ab", 0.0) for rl in repl_lines])
    repl_implied_games = repl_ab / PA_PER_GAME  # mean denominator (old-path consistent)
    repl_pa_volume = repl_ab / AB_PER_PA  # curve volume (consistent with _full_season_pt_volume)
    repl_vb = _apply_variance_batch(
        repl_lines,
        PlayerType.HITTER,
        rng,
        fraction_remaining,
        n_iter,
        pt_mean_fraction=1.0,
        suppress_repl=True,
        pt_volumes=repl_pa_volume,
    )
    # Mean-neutral per-game rate per (iter, active): realized / (implied_games *
    # scale) (UNCAPPED -> E = repl_base / implied_games). eps guards the division
    # and a repl_ab == 0 body (rate 0).
    repl_games_mat = repl_implied_games[None, :] * repl_vb.scales  # (n_iter, n_active)
    repl_ok = repl_games_mat > eps
    repl_safe = np.where(repl_ok, repl_games_mat, 1.0)
    repl_per_game = {
        col: np.where(repl_ok, repl_vb.counts[col] / repl_safe, 0.0) for col in HITTING_COUNTING
    }
    active_index = {id(b): i for i, b in enumerate(active_h_bodies)}
```

Note `eps` is defined a few lines below in the current code (`eps = 1e-9`); MOVE the `eps = 1e-9` line up to just before this block so it is in scope here (it is also used by the bench loop below -- one definition serves both).

- [ ] **Step 5: Replace `_repl_for` with the sampled, index-aligned per-game lookup**

Replace the Task-1 deterministic `_repl_for` with:

```python
    def _repl_for_at(it: int) -> Callable[[ActiveBody], dict[str, float]]:
        # Per-iteration sampled per-game replacement line, index-aligned to the
        # active bodies (id()-keyed map of bodies that outlive this loop).
        def _repl_for(ab: ActiveBody) -> dict[str, float]:
            j = active_index[id(ab)]
            return {col: float(repl_per_game[col][it, j]) for col in HITTING_COUNTING}

        return _repl_for
```

and in the fill loop change the `allocate_bench_fill` call (`~958`) to use the per-iteration callable:

```python
        fill = allocate_bench_fill(actives, bench_samples, _repl_for_at(it)).fill_counts
```

REQUIRED IMPORT: `Callable` is NOT currently imported in `simulation.py` (verified -- no `Callable` reference, and no `from __future__ import annotations`, so the return annotation is evaluated at def-time and would raise `NameError`). Add at the top of the module's imports:

```python
from collections.abc import Callable
```

- [ ] **Step 6: Update the `_simulate_team_hitters_ros_direct` docstring**

In the docstring bullet list (`~890-895`), add a bullet after the bench-sampling bullet:

```python
    - The terminal replacement fill is ALSO sampled per iteration: each active's
      position-routed ``_replacement_line`` is drawn via ``_apply_variance_batch``
      (appended after the bench draw) and the residual ``need`` is filled at the
      mean-neutral per-game rate ``realized / (repl_ab/PA_PER_GAME * scale)``. This
      is the finding-#5 de-bias for the dominant (replacement) fill term. The pool
      is terminal (no capacity cap); the rate's ``1/scale`` is bounded by the eps
      guard and the full-time curve band. See the sampled-replacement-line spec.
```

- [ ] **Step 7: Run the de-bias + mean-neutrality tests to verify GREEN**

Run: `pytest tests/test_mc_integration.py::test_sampled_replacement_fill_widens_team_total_sd_vs_deterministic_baseline tests/test_mc_integration.py::test_sampled_replacement_fill_is_mean_neutral -v`
Expected: BOTH PASS -- the sampled replacement SD strictly exceeds the pinned baseline AND the team-total R mean shows no upward drift > 1%. If the SD test does NOT pass, the decomposition is not adding variance; if the mean test fails on the UPWARD bound, the denominator is capped (the forbidden bug) -- fix the code, do NOT loosen either assertion. (The mean-neutrality test is a guard, not a red gate -- it also passes under the Task-1 deterministic code, since its baseline was captured there.)

- [ ] **Step 8: Run the full MC integration + simulation suites; adjudicate goldens**

Run: `pytest tests/test_mc_integration.py tests/test_simulation.py -v`
Adjudicate per the spec's "RNG stream and test impact":
- MEAN-INVARIANT guardrails `test_repl_not_double_counted_on_new_path`, `test_displacement_factor_scales_hitter_mean` MUST still pass byte-identical (healthy iters fire no fill; active draw is first). If one breaks -> real regression, STOP and fix.
- SD-CEILING `test_ros_direct_uses_full_season_volume_for_cv_pt`: the `:695` `helper_sd < ...` ceiling rises (variance now injected into the replacement fill there). Re-derive ONLY that bound with the new fill variance and re-pin it WITH the recorded analysis. The `:690-691` healthy-iter exact-matches stay byte-identical -- do NOT touch them.
- DO NOT re-pin the bench de-bias baselines. `test_sampled_bench_fill_widens_team_total_sd_vs_deterministic_baseline` asserts an INEQUALITY (`out["R"].std() > _DET_R_SD`); the deterministic `_DET_R_SD`/`_DET_RBI_SD` constants (`~test_mc_integration.py:380-381`) are FIXED historical baselines. The appended replacement draw shifts that test's rng stream so its `out["R"].std()` VALUE changes, but it only rises further above the fixed baseline -- the inequality still holds and the test passes UNCHANGED. Do NOT re-capture `_DET_R_SD`/`_DET_RBI_SD`.
- Only EQUALITY-pinned goldens (if any) shift from the rng-stream move; re-capture those with the reason recorded.
Expected: all PASS after the deliberate `:695`-only re-pin; no mean-invariant assertion loosened; the bench de-bias baselines untouched.

- [ ] **Step 9: Lint/type the touched files**

Run: `ruff check src/fantasy_baseball/simulation.py tests/test_mc_integration.py && ruff format --check src/fantasy_baseball/simulation.py tests/test_mc_integration.py && mypy src/fantasy_baseball/simulation.py src/fantasy_baseball/mc_fill.py`
Expected: clean (full `mypy` fails only on the pre-existing stale `category_odds.py`).

- [ ] **Step 10: Commit**

```bash
git add src/fantasy_baseball/simulation.py tests/test_mc_integration.py
git commit -m "feat(sim): sampled replacement-fill line (mean-neutral per-game rate) (replacement-line task 2)

Each active's position-routed replacement line is drawn via _apply_variance_batch
(appended after the bench draw) and the residual need is filled at the mean-neutral
per-game rate realized/(repl_ab/PA_PER_GAME * scale), curve-banded at
repl_ab/AB_PER_PA. De-bias for the ~77% replacement fill term. Empty-active teams
unchanged; mean-invariant guardrails byte-identical; SD-ceiling :695 re-pinned with
analysis; de-bias test fails-first then passes.

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

## Task 3: Verification sweep

**Files:** none (verification only; a cleanup commit only if a check finds something).

- [ ] **Step 1: Run the suites touching the changed code**

Run: `pytest tests/test_mc_fill.py tests/test_mc_integration.py tests/test_simulation.py tests/test_mc_roster.py tests/test_mc_selection.py tests/test_negbin -v`
Expected: all PASS. (The full `pytest -n auto` may be run too, but the only pre-existing failure is `tests/test_draft/test_parity_golden.py::test_recs_match_golden`, which fails on `main` and imports none of the changed code.)

- [ ] **Step 2: Lint, format, dead-code, type**

Run: `ruff check . && ruff format --check . && vulture && mypy src/fantasy_baseball/simulation.py src/fantasy_baseball/mc_fill.py`
Expected: ruff/format/vulture clean; mypy clean on the two touched files (full `mypy` fails only on the pre-existing stale `category_odds.py` -- state this).

- [ ] **Step 3: Commit any cleanup (skip if Steps 1-2 produced no changes)**

```bash
git add -A
git commit -m "chore(mc): verification cleanup for sampled replacement-line (task 3)

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

## Task 4: Acceptance-gate diagnostic + evidence (re-run Task 5)

Re-run the SD-calibration diagnostic before/after and adjudicate the PASS/PARTIAL/FAIL ladder. REQUIRES a Yahoo-authed local refresh -- the user runs the auth-bound step; the implementer drives the rest. This mirrors the prior `sampled-bench-fill-sd-evidence-2026-06-29.md` run.

**Files:**
- Create: `docs/superpowers/sampled-replacement-line-sd-evidence-2026-06-29.md`

- [ ] **Step 1: Add the temporary diagnostics (evidence-only; reverted after)**

Behind `FB_SELECTION_ATTRIBUTION`, add (a) the replacement-fill-share counter by RE-ADDING the prior bench-task seam (it was instrument-then-reverted, so `FillResult` currently has only `fill_counts` -- re-add `replacement_games` + a module accumulator summed in the fill loop) AND (b) the per-cat MEAN of the new_engine team totals alongside the existing median in `compute_sd_calibration` (`np.mean(batch[t][cat])` beside the `np.std`). Both are instrument-then-revert, exactly like the gate-#4 share counter in the prior run. Keep ASCII.

- [ ] **Step 2: Run the diagnostic BEFORE (deterministic = bench-only) and AFTER (this change)**

The BEFORE baseline is the branch state immediately before Task 2 (i.e. the Task-1 commit, deterministic replacement). The AFTER is the Task-2 HEAD. Use the local `run_full_refresh()` driver pattern the prior evidence run used (a throwaway `scripts/_task*_diag.py` calling `run_full_refresh()` directly against LOCAL SQLite -- NOT `run_season_dashboard.py`, which only launches Flask, and NOT `refresh_remote.py`, which writes REMOTE Upstash). PowerShell:

```powershell
$env:FB_SELECTION_ATTRIBUTION = '1'
python scripts/_taskN_diag.py   # local-only; emits phase0_attribution.txt
```

- **Before:** record the SHA, `git checkout <task-1-sha>`, run, capture SD ratios + per-cat new_engine MEANS + replacement share.
- **After:** `git checkout mc-sampled-bench-fill`, run, capture the same three.
Use the same seed / n_iter / fraction_remaining for both (apples-to-apples).

- [ ] **Step 3: Adjudicate the acceptance ladder (from the spec)**

Apply the spec's strict precedence ladder: gate#2 (pooled in [0.8,1.25]) / gate#3 (new_engine MEAN drift: no upward > +1%, no downward > ~5% unexplained) breach is an independent hard fail checked FIRST; then OVER-CORRECTION (R or RBI >= 1.20) > FAIL (R or RBI rose < +0.08) > PARTIAL (a ratio in [baseline+0.08, 0.85)) > PASS (both in [0.85, 1.20)). Record R/RBI before/after, the mean-drift deltas, and the replacement share. If OVER-CORRECTION: apply the rate-cap damper (then proper-NegBin) and re-run, re-checking gate #3 downward drift. If PARTIAL: SURFACE to the user. If FAIL: STOP, re-design. Do NOT loosen the +0.08 floor.

- [ ] **Step 4: Write the evidence doc**

Create `docs/superpowers/sampled-replacement-line-sd-evidence-2026-06-29.md` (ASCII): run conditions (seed, n_iter, fraction), before/after SD-ratio table, mean-drift table (means, not medians), replacement-fill share, and the PASS/PARTIAL/FAIL verdict against the ladder.

- [ ] **Step 5: Revert the temporary diagnostics; commit the evidence doc**

```bash
git add docs/superpowers/sampled-replacement-line-sd-evidence-2026-06-29.md
git commit -m "docs(mc): sampled replacement-line SD acceptance evidence (task 4)

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

Confirm `git status` is clean (diagnostics reverted; throwaway scripts removed) on branch `mc-sampled-bench-fill`.
