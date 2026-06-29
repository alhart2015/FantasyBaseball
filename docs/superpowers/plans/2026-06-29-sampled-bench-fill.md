# Sampled Bench Injury-Fill Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** De-bias the ROS-direct hitter MC's under-dispersed counting categories by sampling the bench injury-fill body's per-game rate and own availability (mean-neutral) instead of penciling it in with a fixed, zero-variance line.

**Architecture:** The active hitter draw in `_simulate_team_hitters_ros_direct` already produces a per-iteration `frac_missed` shortfall that is backfilled from the bench. Today the bench fill line is deterministic. This change samples the bench pool with the SAME `_apply_variance_batch` path the actives use, decomposes each bench body's sampled line into a mean-neutral per-game rate (`realized / (g_ros_full * scale)`) plus an availability-driven capacity (`g_ros_full * scale`), and feeds those per-iteration into the existing `allocate_bench_fill` allocator (which gains a `capacity` input). Hitters only; pitchers untouched.

**Tech Stack:** Python 3.11+, numpy, pytest. Files under `src/fantasy_baseball/` and `tests/`.

**Spec:** `docs/superpowers/specs/2026-06-29-sampled-bench-fill-design.md` (converged).

## Global Constraints

- **ASCII-only** in all source, comments, log/format strings (Windows cp1252 stdout). Use `-`, `--`, straight quotes, `sigma`.
- **Player IDs are `name::player_type` or yahoo_id** — never key on bare names.
- **No `x or default` for numeric defaults** — use explicit `is not None` / threshold checks (the falsy-zero footgun). The EPS guard here is an explicit `> EPS` threshold, not `or`.
- **mypy** covers `src/fantasy_baseball/simulation.py` and `src/fantasy_baseball/mc_fill.py` (both in `[tool.mypy].files`). Per-iteration values indexed out of numpy arrays are `np.float64`; wrap `capacity` and each `per_game_rate[col]` in `float(...)` to satisfy the `dict[str, float]` / `capacity: float` annotations.
- **Don't silently fix tests.** Two test changes here are deliberate contract changes (the retargeted capacity test, and any team-total golden the rng shift moves) — state the justification in the commit message. Empty-bench guardrail tests and the legacy `_apply_variance_batch` snapshot MUST stay byte-identical; if they break, that is a bug, not a golden to update.
- **Commit frequently** — one commit per task.

---

### Task 1: Expose `scales` on `VarianceBatch`

The rate decomposition needs the unclamped playing-time `scale` (the games that generated `realized`). `VarianceBatch` currently exposes only `frac_missed = max(0, 1 - scales)`, which clamps away `scale > 1`. Add `scales`.

**Files:**
- Modify: `src/fantasy_baseball/simulation.py` (the `VarianceBatch` dataclass ~85-96, the `n_players == 0` early return ~759-763, the populated return ~820)
- Test: `tests/test_mc_integration.py`

**Interfaces:**
- Produces: `VarianceBatch.scales: np.ndarray` of shape `(n_iter, n_players)`, equal to the `scales` array computed in `_apply_variance_batch` at ~line 789 (`np.maximum(0.0, eff_mean + z_pt*eff_sd)`); consumed by Task 3.

- [ ] **Step 1: Write the failing tests**

In `tests/test_mc_integration.py`, after `test_frac_missed_exposed_and_in_unit_range`:

```python
def test_scales_exposed_and_consistent_with_frac_missed():
    """scales is exposed, shape (n_iter, n_players), and frac_missed == max(0, 1-scales)."""
    rng = np.random.default_rng(7)
    result = _apply_variance_batch(_players(), "hitter", rng, 0.4, 6)
    assert result.scales.shape == (6, 2)
    np.testing.assert_array_equal(result.frac_missed, np.maximum(0.0, 1.0 - result.scales))


def test_scales_empty_player_list_is_shape_correct():
    """The n_players==0 early return still builds a shape-correct scales array."""
    result = _apply_variance_batch([], "hitter", np.random.default_rng(1), 0.4, 4)
    assert result.scales.shape == (4, 0)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_mc_integration.py::test_scales_exposed_and_consistent_with_frac_missed tests/test_mc_integration.py::test_scales_empty_player_list_is_shape_correct -v`
Expected: FAIL with `AttributeError: 'VarianceBatch' object has no attribute 'scales'`.

- [ ] **Step 3: Add the field and populate both constructor sites**

In `src/fantasy_baseball/simulation.py`, add the field to `VarianceBatch` (after `frac_missed`):

```python
    counts: dict[str, np.ndarray]  # {col: (n_iter, n_players)}
    frac_missed: np.ndarray  # (n_iter, n_players) = max(0, 1 - scales)
    scales: np.ndarray  # (n_iter, n_players) -- the unclamped playing-time scale that drove mu (Task 3 consumes)
```

In the `n_players == 0` early return (~759-763), add the shape-correct empty array:

```python
        return VarianceBatch(
            counts={col: np.zeros((n_iter, 0)) for col in counting_cols},
            frac_missed=np.zeros((n_iter, 0)),
            scales=np.zeros((n_iter, 0)),
        )
```

In the populated return (~820), pass the existing `scales` array (computed at ~789):

```python
    return VarianceBatch(counts=out, frac_missed=frac_missed, scales=scales)
```

- [ ] **Step 4: Run the new tests AND the rng-stability snapshot**

Run: `pytest tests/test_mc_integration.py::test_scales_exposed_and_consistent_with_frac_missed tests/test_mc_integration.py::test_scales_empty_player_list_is_shape_correct tests/test_mc_integration.py::test_variance_batch_default_matches_legacy_columns -v`
Expected: all PASS. (The legacy snapshot proves adding `scales` did not perturb the rng stream — `scales` was already computed, we only return it.)

- [ ] **Step 5: Commit**

```bash
git add src/fantasy_baseball/simulation.py tests/test_mc_integration.py
git commit -m "feat(sim): expose scales on VarianceBatch (sampled-bench-fill task 1)"
```

---

### Task 2: Add `BenchSample.capacity` and feed it through `allocate_bench_fill`

The allocator must cap each bench body at a per-iteration `capacity` instead of the static `g_ros_full`. Add the field and the new unit tests; retarget the named cap test.

**Files:**
- Modify: `src/fantasy_baseball/mc_fill.py` (`BenchSample` ~27-30, `allocate_bench_fill` docstring ~43-52, `remaining` init ~55-56)
- Modify: `src/fantasy_baseball/simulation.py` (`_simulate_team_hitters_ros_direct` bench construction ~908 — behavior-preserving `capacity=bb.g_ros_full` stopgap so the no-default field does not break the helper before Task 3 replaces the block)
- Test: `tests/test_mc_fill.py` (`_bench_sample` factory ~39-40, retarget ~105, new tests)

**Interfaces:**
- Produces: `BenchSample(body, per_game_counts, capacity)` — `capacity: float` (no default), the games this body may cover this iteration. `allocate_bench_fill` initializes each body's remaining capacity from `bs.capacity`. Consumed by Task 3.

- [ ] **Step 1: Update the test factory and write the failing tests**

In `tests/test_mc_fill.py`, change `_bench_sample` to thread `capacity` (default to `g_ros_full` so existing tests keep their old semantics):

```python
def _bench_sample(b, per_game, capacity=None):
    return BenchSample(
        body=b,
        per_game_counts=_line(**per_game),
        capacity=b.g_ros_full if capacity is None else capacity,
    )
```

Retarget the named cap test (was `test_fill_never_exceeds_bench_g_ros_full_capacity` ~105) to assert against `bs.capacity`:

```python
def test_fill_never_exceeds_bench_capacity():
    # Two OF starters both injured; one bench body eligible for both. Its total
    # contributed games cannot exceed its per-iteration CAPACITY (no longer the
    # static g_ros_full -- capacity = g_ros_full*scale, which CAN exceed g_ros_full).
    a1 = _active("S1", "1", g_ros_adj=100.0, pos=Position.OF)
    a2 = _active("S2", "2", g_ros_adj=100.0, pos=Position.OF)
    cap = 10.0
    b = _bench("Depth", "3", g_ros_full=cap, per_game_value=2.0, pos=Position.OF)
    res = allocate_bench_fill(
        [ActiveSample(a1, frac_missed=1.0), ActiveSample(a2, frac_missed=1.0)],
        [_bench_sample(b, {"r": 1.0}, capacity=cap)],
        _no_replacement,
    )
    assert res.fill_counts["r"] <= cap + 1e-9
```

Add new cascade tests:

```python
def test_capacity_below_g_ros_full_limits_fill_and_cascades():
    # Starter misses 50 games. Best bench bat has capacity 10 (sampled low
    # availability); the residual cascades to the second bench bat.
    a = _active("Star", "1", g_ros_adj=100.0, pos=Position.OF)
    b1 = _bench("D1", "2", g_ros_full=60.0, per_game_value=3.0, pos=Position.OF)
    b2 = _bench("D2", "3", g_ros_full=60.0, per_game_value=1.0, pos=Position.OF)
    res = allocate_bench_fill(
        [ActiveSample(a, frac_missed=0.5)],  # 0.5 * 100 = 50 games missed
        [_bench_sample(b1, {"r": 1.0}, capacity=10.0),
         _bench_sample(b2, {"r": 0.5}, capacity=60.0)],
        _no_replacement,
    )
    # b1: 10 games * 1.0 = 10 r; b2: remaining 40 games * 0.5 = 20 r -> 30 r.
    assert abs(res.fill_counts["r"] - 30.0) < 1e-9


def test_zero_capacity_body_skipped_and_cascades():
    # Best bench bat sampled fully unavailable (capacity 0) -> contributes nothing
    # despite the highest rate; the next eligible body covers the shortfall.
    a = _active("Star", "1", g_ros_adj=100.0, pos=Position.OF)
    b1 = _bench("D1", "2", g_ros_full=60.0, per_game_value=3.0, pos=Position.OF)
    b2 = _bench("D2", "3", g_ros_full=60.0, per_game_value=1.0, pos=Position.OF)
    res = allocate_bench_fill(
        [ActiveSample(a, frac_missed=0.5)],  # 50 games missed
        [_bench_sample(b1, {"r": 9.0}, capacity=0.0),
         _bench_sample(b2, {"r": 0.5}, capacity=60.0)],
        _no_replacement,
    )
    # b1 skipped (cap 0); b2 covers all 50 -> 25 r.
    assert abs(res.fill_counts["r"] - 25.0) < 1e-9
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_mc_fill.py -k "capacity or cascade" -v`
Expected: FAIL — `BenchSample.__init__() got an unexpected keyword argument 'capacity'`.

- [ ] **Step 3: Add the field, use it, fix the docstring/comment**

In `src/fantasy_baseball/mc_fill.py`, add the field to `BenchSample`:

```python
@dataclass(frozen=True)
class BenchSample:
    body: BenchBody
    per_game_counts: dict[str, float]  # sampled HITTING_COUNTING stats PER GAME
    capacity: float  # games this body may cover THIS iteration (g_ros_full * sampled scale)
```

In `allocate_bench_fill`, change the `remaining` init (~56) to read `bs.capacity`:

```python
    # Remaining capacity per bench body (mutated as we allocate). Per-iteration
    # capacity = g_ros_full * sampled availability (the caller builds it); it can
    # exceed g_ros_full when the body is sampled MORE available than projected.
    remaining = {id(bs): bs.capacity for bs in benches}
```

Fix the now-false docstring line in `allocate_bench_fill` (~50-52). Replace:

```
    / PA_PER_GAME). One bench body's total assigned games
    <= its g_ros_full. Tie-break: higher per_game_value, then player-id ascending.
```

with:

```
    / PA_PER_GAME). One bench body's total assigned games <= its per-iteration
    ``capacity`` (g_ros_full * sampled scale; CAN exceed g_ros_full when the body
    is sampled more available than projected). Tie-break: higher per_game_value,
    then player-id ascending.
```

Because `capacity` has NO default, every existing `BenchSample(...)` construction must now pass it. There is one in production code: the still-deterministic bench block in `_simulate_team_hitters_ros_direct` (`simulation.py:908`). Task 3 rewrites that block, but it does not run until the next task, so patch it NOW with a behavior-preserving stopgap so this task's commit keeps the whole suite green (`capacity=bb.g_ros_full` reproduces the old `remaining = g_ros_full` cap exactly — `allocate_bench_fill` behavior is unchanged):

```python
        bench_samples.append(
            BenchSample(body=bb, per_game_counts=per_game, capacity=bb.g_ros_full)
        )
```

- [ ] **Step 4: Run the mc_fill suite AND the helper's integration tests (must all stay green)**

Run: `pytest tests/test_mc_fill.py tests/test_mc_integration.py -v`
Expected: all PASS. `test_mc_fill.py`: existing tests unchanged (`_bench_sample` defaults capacity to `g_ros_full`); new cascade tests pass; retargeted cap test passes. `test_mc_integration.py`: still green because the helper stopgap passes `capacity=bb.g_ros_full`, so `allocate_bench_fill` behaves exactly as before. If any `test_mc_integration.py` test errors with `BenchSample.__init__() missing 1 required positional argument: 'capacity'`, the Step-3 helper stopgap was not applied.

- [ ] **Step 5: Commit**

```bash
git add src/fantasy_baseball/mc_fill.py src/fantasy_baseball/simulation.py tests/test_mc_fill.py
git commit -m "feat(mc): BenchSample.capacity drives allocate_bench_fill cap (sampled-bench-fill task 2)

Retarget test_fill_never_exceeds_bench_g_ros_full_capacity -> _capacity:
the <= g_ros_full invariant is intentionally replaced by the per-iteration
capacity cap (capacity = g_ros_full*scale can exceed g_ros_full by design).
Helper passes capacity=bb.g_ros_full as a behavior-preserving stopgap until
Task 3 replaces the bench block (keeps every commit green)."
```

---

### Task 3: Sample the bench pool in `_simulate_team_hitters_ros_direct`

Replace the once-built deterministic bench line with a per-iteration sampled rate + capacity.

**Files:**
- Modify: `src/fantasy_baseball/simulation.py` (`_simulate_team_hitters_ros_direct` docstring ~860-862 and body ~898-923)
- Test: `tests/test_mc_integration.py`

**Interfaces:**
- Consumes: `VarianceBatch.scales` (Task 1), `BenchSample(body, per_game_counts, capacity)` (Task 2), existing `_apply_variance_batch`, `allocate_bench_fill`, `_full_season_pt_volume`, `_replacement_line`, `HITTING_COUNTING`.

> **Why a pinned deterministic baseline (not bench-vs-no-bench):** removing the
> bench changes the WHOLE fill term (a strong bench bat's high fixed rate vs the
> low replacement rate), so `with_bench.std() > no_bench.std()` is ALREADY true on
> the current deterministic code and can never fail-first. The valid red step
> compares the SAMPLED fill against the pre-change DETERMINISTIC fill on the SAME
> bench-deep fixture: pin the current deterministic team-total SD, then assert the
> sampled SD strictly exceeds it. The deterministic helper is reproducible, so the
> pin is exact; pre-change `std == pinned` (so `> pinned` is False -> FAIL),
> post-change the added rate noise makes `std > pinned` -> PASS.

- [ ] **Step 1: Add the import, the bench-deep fixture, and the tests (baseline placeholders)**

Add `BenchBody` to the existing `from fantasy_baseball.mc_roster import ...` block in `tests/test_mc_integration.py` (it already imports `EffectiveRoster`, `build_effective_roster`). Then, after `test_injured_active_body_gets_bench_fill`, add:

```python
# Captured from the CURRENT (deterministic-fill) helper on _bench_deep_roster() at
# seed 11, fraction_remaining=0.2, n_iter=4000, in Step 2 (paste the printed
# values). The deterministic fill is reproducible so these are exact; the SAMPLED
# fill adds independent rate noise -> strictly larger SD. If the default path ever
# changes, re-capture under git stash (cf. _LEGACY_DEFAULT_COUNTS).
_DET_R_SD = 0.0    # <- paste in Step 2
_DET_RBI_SD = 0.0  # <- paste in Step 2


def _bench_deep_roster():
    # BenchBat is slotted to the BENCH (Position.BN) so build_effective_roster seats
    # it in eff.bench, NOT the active set (an OF-slotted bat seats as ACTIVE and
    # eff.bench would be empty -- the whole feature path would go untested).
    # positions=[OF] (set by _hitter) keeps it eligible to fill the OF starter.
    return [
        _hitter("Starter", Position.OF, "1"),
        _hitter("BenchBat", Position.BN, "2", r=120, hr=40, rbi=110, sb=30),
    ]


def test_sampled_bench_fill_widens_team_total_sd_vs_deterministic_baseline():
    """De-bias: the sampled bench fill carries its own rate noise, so the bench-deep
    team's R/RBI total SD STRICTLY EXCEEDS the pinned pre-change deterministic-fill
    SD (same fixture, same seed; only the fill sampling changed)."""
    eff = _eff_roster(_bench_deep_roster())
    assert len(eff.bench) == 1  # the fixture really seats a bench fill body
    out = _simulate_team_hitters_ros_direct(eff, 0.2, np.random.default_rng(11), 4000)
    assert np.all(np.isfinite(out["R"])) and np.all(np.isfinite(out["RBI"]))
    assert out["R"].std() > _DET_R_SD
    assert out["RBI"].std() > _DET_RBI_SD


def test_sampled_bench_fill_is_deterministic_under_seed():
    """Same seed -> identical team totals (regression guard; passes pre and post)."""
    eff = _eff_roster(_bench_deep_roster())
    a = _simulate_team_hitters_ros_direct(eff, 0.2, np.random.default_rng(5), 128)
    b = _simulate_team_hitters_ros_direct(eff, 0.2, np.random.default_rng(5), 128)
    for cat in ("R", "HR", "RBI", "SB", "ros_h", "ros_ab"):
        np.testing.assert_array_equal(a[cat], b[cat])


def test_zero_volume_bench_body_is_finite_and_skipped():
    """A g_ros_full==0 bench body hits the EPS guard (games_played = 0*scale = 0 ->
    capacity 0), contributing nothing with NO division-by-zero. Hand-built so the
    zero-volume body is guaranteed present regardless of classification."""
    starter = _hitter("Starter", Position.OF, "1")
    zero_bench = BenchBody(
        player=_hitter("ZeroVol", Position.BN, "3", r=120, hr=40, rbi=110, sb=30),
        g_ros_full=0.0,
        per_game_value=0.0,
        eligible_positions=frozenset({Position.OF}),
    )
    eff = EffectiveRoster(active=[_active_body(starter)], bench=[zero_bench])
    out = _simulate_team_hitters_ros_direct(eff, 0.2, np.random.default_rng(4), 256)
    for cat in ("R", "HR", "RBI", "SB", "ros_h", "ros_ab"):
        assert np.all(np.isfinite(out[cat]))
```

- [ ] **Step 2: Capture the deterministic baseline on the CURRENT code, paste it in**

The helper is still deterministic at this point (Task 2's stopgap passes `capacity=bb.g_ros_full`). Run:

```bash
python -c "import numpy as np, sys; sys.path.insert(0,'tests'); \
from test_mc_integration import _eff_roster, _bench_deep_roster; \
from fantasy_baseball.simulation import _simulate_team_hitters_ros_direct as h; \
o=h(_eff_roster(_bench_deep_roster()),0.2,np.random.default_rng(11),4000); \
print('R', repr(o['R'].std()), 'RBI', repr(o['RBI'].std()))"
```

Paste the printed `R` and `RBI` values verbatim into `_DET_R_SD` and `_DET_RBI_SD`.

- [ ] **Step 3: Run tests to verify the de-bias test fails-first**

Run: `pytest tests/test_mc_integration.py::test_sampled_bench_fill_widens_team_total_sd_vs_deterministic_baseline tests/test_mc_integration.py::test_sampled_bench_fill_is_deterministic_under_seed tests/test_mc_integration.py::test_zero_volume_bench_body_is_finite_and_skipped -v`
Expected: `..._widens_team_total_sd_...` FAILS (current deterministic `std == _DET_*_SD`, so `> _DET_*_SD` is False). The determinism and zero-volume tests PASS already (regression/guard tests — the deterministic helper is reproducible and the zero-volume body is skipped by `allocate_bench_fill`'s `remaining > 0` filter today). That is expected; the de-bias test is the red one.

- [ ] **Step 4: Replace the deterministic bench block with sampled per-iteration fill**

In `src/fantasy_baseball/simulation.py`, in `_simulate_team_hitters_ros_direct`, replace the block from the `# Bench per-game counts: ...` comment (now the Task-2 stopgap construction) through the end of the per-iteration `for it in range(n_iter):` loop (~898-923) with:

```python
    # Bench bodies are sampled with their OWN variance (rate + availability), the
    # same way actives are -- the de-bias for the hitter-cat under-dispersion. The
    # draw is APPENDED after the active draw above (actives already consumed rng),
    # so the active stream is unchanged and an EMPTY bench pool no-ops the rng
    # (_apply_variance_batch returns before any draw when n_players == 0). See the
    # sampled-bench-fill spec (2026-06-29).
    bench_flats = [bb.player.to_flat_dict() for bb in bench_h_bodies]
    bench_pt_volumes = np.array(
        [_full_season_pt_volume(bb.player, is_hitter=True) for bb in bench_h_bodies]
    )
    bench_vb = _apply_variance_batch(
        bench_flats,
        PlayerType.HITTER,
        rng,
        fraction_remaining,
        n_iter,
        pt_mean_fraction=1.0,
        suppress_repl=True,
        pt_volumes=bench_pt_volumes,
    )

    def _repl_for(ab: ActiveBody) -> dict[str, float]:
        return _replacement_line(ab.player.to_flat_dict(), is_hitter=True)

    # Per-iteration fill allocation (the sanctioned small Python loop: <=12 active,
    # <=2 bench). Sampling (active + bench) is vectorized above; only the cheap
    # per-iteration BenchSample build and the allocation loop here.
    eps = 1e-9
    fill_totals: dict[str, np.ndarray] = {col: np.zeros(n_iter) for col in HITTING_COUNTING}
    for it in range(n_iter):
        actives = [
            ActiveSample(body=body, frac_missed=float(frac_missed[it, idx]))
            for idx, body in enumerate(active_h_bodies)
        ]
        bench_samples: list[BenchSample] = []
        for b_idx, bb in enumerate(bench_h_bodies):
            # games that generated this iter's realized line = g_ros_full * scale
            # (UNCAPPED). Dividing realized by it recovers a MEAN-NEUTRAL per-game
            # rate (E = base/g_ros_full); capacity is that same games count (how
            # many of the starter's missed games this body can cover). eps guards
            # the division and also zeroes a g_ros_full == 0 body.
            games_played = bb.g_ros_full * float(bench_vb.scales[it, b_idx])
            if games_played > eps:
                per_game = {
                    col: float(bench_vb.counts[col][it, b_idx]) / games_played
                    for col in HITTING_COUNTING
                }
                capacity = float(games_played)
            else:
                per_game = {col: 0.0 for col in HITTING_COUNTING}
                capacity = 0.0
            bench_samples.append(
                BenchSample(body=bb, per_game_counts=per_game, capacity=capacity)
            )
        fill = allocate_bench_fill(actives, bench_samples, _repl_for).fill_counts
        for col in HITTING_COUNTING:
            fill_totals[col][it] = fill[col]
```

Then update the helper's docstring (~860-862): replace the bullet

```
    - Bench per-game lines are the clean DETERMINISTIC base ROS projection
      (``base_ros_total / g_ros_full``), iteration-independent -- built once.
```

with

```
    - Bench bodies are SAMPLED per iteration with their own variance: a
      ``_apply_variance_batch`` draw (appended after the active draw) yields each
      bench body's realized line and ``scale``; the per-game fill rate is
      ``realized / (g_ros_full * scale)`` (mean-neutral) and its fill capacity is
      ``g_ros_full * scale`` (its own sampled availability). An empty bench pool
      no-ops the rng. See the sampled-bench-fill spec (2026-06-29).
```

- [ ] **Step 5: Run the new tests (de-bias now green) + the empty-bench guardrails (must stay byte-identical)**

Run: `pytest tests/test_mc_integration.py::test_sampled_bench_fill_widens_team_total_sd_vs_deterministic_baseline tests/test_mc_integration.py::test_sampled_bench_fill_is_deterministic_under_seed tests/test_mc_integration.py::test_zero_volume_bench_body_is_finite_and_skipped tests/test_mc_integration.py::test_repl_not_double_counted_on_new_path tests/test_mc_integration.py::test_displacement_factor_scales_hitter_mean tests/test_mc_integration.py::test_ros_direct_uses_full_season_volume_for_cv_pt -v`
Expected: all PASS. The de-bias test now goes green (sampled SD > pinned deterministic baseline) — if it does NOT clear the baseline, the rate decomposition is wrong (most likely the denominator was capped at `g_ros_full` instead of `g_ros_full*scale`); fix it, do not loosen the assertion. The three guardrail tests use EMPTY-bench `EffectiveRoster`s, so the bench draw no-ops the rng and they stay byte-identical. **If any guardrail test fails, STOP — the empty-bench path is consuming rng (a bug), not a golden to re-pin.**

- [ ] **Step 6: Run the full MC test suite and adjudicate any pinned-value shifts**

Run: `pytest tests/test_mc_integration.py tests/test_simulation.py -v`
Expected: PASS. Most `_simulate_team_hitters_ros_direct` tests assert inequalities / active-only equalities ("Mechanism only -- no magnitude pinned"), so they should hold. For ANY failing test that pins a magnitude on a NON-empty-bench team total (e.g. a `run_ros_monte_carlo` / `simulate_remaining_season_batch` golden), confirm the failure is the intended rng-stream shift from the appended bench draw (a different but still-valid simulated value), then update the pinned literal. Do NOT touch empty-bench guardrails or the `test_variance_batch_default_matches_legacy_columns` snapshot — those must not move.

- [ ] **Step 7: Commit**

```bash
git add src/fantasy_baseball/simulation.py tests/test_mc_integration.py
git commit -m "feat(sim): sampled bench injury-fill (rate + availability) (sampled-bench-fill task 3)

Bench fill body is now drawn per-iteration with its own variance and its
availability caps its fill capacity; mean-neutral rate = realized/(g*scale).
Any updated team-total golden reflects the intended rng-stream shift, not a
regression (empty-bench guardrails + legacy snapshot stay byte-identical)."
```

---

### Task 4: Full verification sweep

**Files:** none (verification only).

- [ ] **Step 1: Full test suite**

Run: `pytest -n auto`
Expected: all PASS. If a non-MC test fails, investigate (the change is scoped to the ROS-direct hitter path; an unrelated failure may be pre-existing — confirm against `main`).

- [ ] **Step 2: Lint + format + dead code**

Run: `ruff check . && ruff format --check . && vulture`
Expected: zero ruff violations, no format drift, no NEW vulture findings. If `safe_float` (or another import) is now unused in `simulation.py` after removing the deterministic block, remove the dead import; if a pre-existing vulture finding appears unrelated, note it.

- [ ] **Step 3: Types**

Run: `mypy`
Expected: PASS for `simulation.py` and `mc_fill.py`. If a `np.float64` vs `float` error appears, confirm the `float(...)` wraps from Task 3 are in place.

- [ ] **Step 4: Commit any cleanup**

```bash
git add -A
git commit -m "chore(mc): lint/type cleanup for sampled bench fill (sampled-bench-fill task 4)"
```

(Skip the commit if Steps 1-3 produced no changes.)

---

### Task 5: Acceptance-gate diagnostic + evidence

Re-run the SD-calibration diagnostic that measured the original bias and record the evidence the spec's acceptance gate requires.

**Files:**
- Create: `docs/superpowers/sampled-bench-fill-sd-evidence-2026-06-29.md` (evidence doc)

- [ ] **Step 1: Add a replacement-fill-share counter (temporary diagnostic)**

The emitted diagnostic does not report the replacement-fill share (gate #4), so add it. Behind the existing `FB_SELECTION_ATTRIBUTION` env check, accumulate, inside the `for it in range(n_iter)` fill loop of `_simulate_team_hitters_ros_direct`, the replacement games vs total filled games. The cleanest seam: have `allocate_bench_fill` also return the replacement-game count (or, lower-touch, log per-team `sum(frac_missed*g_ros_adj) - bench_assigned` once). Record the team-weighted `replacement_games / total_filled_games`. Keep it ASCII; this counter is evidence-only and may be reverted after the run (note it in the evidence doc if reverted).

- [ ] **Step 2: Run the diagnostic on BOTH the pre-change and post-change code**

The "before" baseline must be numeric for the mean-drift check (gate #3), and the SD-calibration table the diagnostic emits does NOT include category means — so capture means from the cache instead. Run a local refresh twice, recording the SD table (written to the attribution file by `refresh_pipeline.py` under `FB_SELECTION_ATTRIBUTION`, NOT stdout — read the file) AND the hitter category team-total means computed from `cache:standings_breakdown` (per team: `team_ytd[col] + sum(player.contribution_stats[col])`, the same computation used to produce the projected standings).

PowerShell (this box is PowerShell-primary; set the env var the PowerShell way, and use `--no-sync` so the Upstash sync does not clobber the local cache):

```powershell
$env:FB_SELECTION_ATTRIBUTION = '1'
python scripts/run_season_dashboard.py --no-sync   # triggers the local refresh + diagnostic
```

- **Before (pre-change):** record the current branch SHA, then `git checkout <sha-before-task-3>` (the commit before this branch's bench-fill change — i.e. the Task 2 stopgap state, which is byte-identical-deterministic), run the refresh, and capture the SD ratios + hitter cat means + replacement share. (The Phase 6 evidence doc already lists the SD ratios R 0.72 / RBI 0.70 / pooled 0.92 — cross-check against them; the means come from the breakdown.)
- **After (post-change):** `git checkout mc-sampled-bench-fill`, run the refresh, capture the same three.

- [ ] **Step 3: Check the acceptance gate (from the spec)**

Confirm ALL of:
1. **R and RBI** each rose by at least `+0.08` over their ~0.70-0.72 baseline, with neither exceeding `1.20`. Target `>= 0.85`; record the actual values. If either lands in `[baseline+0.08, 0.85)`, note it as a partial success and flag whether sampling the replacement line (the deferred follow-up) is warranted.
2. **Pooled ratio** stays in `[0.8, 1.25]`.
3. **Hitter category team-total means**: no UPWARD drift `> +1%` vs the pre-change run (a hard fail — the mean-neutral decomposition must hold); a downward drift up to `~5%` is expected (the intended replacement re-damping). Record the deltas.
4. **Replacement-fill share** pre vs post — record it; flag a material rise.

- [ ] **Step 4: Write the evidence doc**

Create `docs/superpowers/sampled-bench-fill-sd-evidence-2026-06-29.md` with: the run conditions (seed, n_iter, frac), the before/after SD-ratio table, the mean-drift table, the replacement-fill share, and a PASS/PARTIAL/FAIL verdict against the four gate criteria above. Keep it ASCII-only.

- [ ] **Step 5: Commit (and revert the temporary counter if it was not kept)**

```bash
git add docs/superpowers/sampled-bench-fill-sd-evidence-2026-06-29.md
git commit -m "docs(mc): sampled-bench-fill SD-calibration acceptance evidence (sampled-bench-fill task 5)"
```

If the Step-1 replacement-share counter was instrumentation-only, revert it in a follow-up commit (or fold a kept, tidy version into the fill path with a one-line comment).

---

## Notes for the implementer

- **The mean-neutral invariant is the headline correctness property.** `per_game_rate = realized / (g_ros_full * scale)` recovers `base_ros / g_ros_full` in expectation (the old deterministic rate), now with NegBin noise. If Task 5's mean-drift check shows an UPWARD shift, the decomposition regressed (most likely a denominator capped at `g_ros_full` instead of the uncapped `g_ros_full * scale`) — fix it before merge.
- **EPS guard**, not `or`: a `g_ros_full == 0` or fully-unavailable bench body has `games_played <= eps`, yielding `capacity = 0.0` so the allocator skips it and cascades. Never write `games_played or default`.
- **Empty-bench no-op** is load-bearing for the byte-identical guardrails: call `_apply_variance_batch` unconditionally with a possibly-empty `bench_flats`; its `n_players == 0` early return consumes no rng.
