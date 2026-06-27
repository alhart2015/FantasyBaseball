# Games-based MC -- Phase 5 (pitchers: ROS-direct active-slot + IL displacement, no fill) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development. Steps use checkbox (`- [ ]`).

**Goal:** Bring PITCHERS onto the ROS-direct engine, mirroring the hitter path MINUS the bench injury-fill: sample the `EffectiveRoster`'s active PITCHER bodies (active-slot + IL, already pool-displaced by Phase 2's `_compute_displacement_factors`) directly, apply each body's displacement `factor`, sum, and blend `team_total = YTD + ROS`. Healthy bench pitchers are EXCLUDED (already dropped from `EffectiveRoster`). This fixes the pitcher over-credit (top-k seating deep bullpens) that, left alone, tilts 5x5 standings toward pitching-deep teams. NO bench injury-fill, NO closer-role SV modeling (both deferred).

**Why no fill (the key design point):** `pt_mean_fraction=1.0` already applies the FULL playing-time haircut (`eff_mean = mean_scale`), bringing the pitcher mean to `projection * mean_scale` ~= ERoto's haircut mean. ERoto does NOT fill pitcher injuries (it is an expected-value model). So adding any injury-fill (replacement OR bench) would push pitcher means ABOVE ERoto -- an over-credit. `suppress_repl=True` + no fill makes the MC pitcher means match ERoto by construction. (Hitters DO get a small bench-fill premium above ERoto; pitchers get none -- the accepted, documented asymmetry, since pitcher rich-fill is deferred.)

**Architecture:** A new `_simulate_team_pitchers_ros_direct` helper (a pitcher analog of `_simulate_team_hitters_ros_direct`, minus the bench-fill loop). Independent `_ROS_DIRECT_PITCHERS` flag (Phase-6 fallback granularity, separate from `_ROS_DIRECT_HITTERS`). No pipeline/setup changes: `effective_rosters` is already threaded (Phase 4) and `build_effective_roster` already produces pitcher active bodies (Phase 2).

**Tech Stack:** Python, pytest. Touched: `src/fantasy_baseball/simulation.py`; test `tests/test_mc_integration.py`.

## Global Constraints

- ASCII-only; numeric defaults via `is not None`, never `x or default`; imports at top.
- Spec: `docs/superpowers/specs/2026-06-26-games-based-availability-mc-design.md` Component 5 (pitchers).
- `PITCHING_COUNTING = ["w","k","sv","ip","er","bb","h_allowed"]` (verified constants.py:128 -- the 7 roto stats; `g`/`gs` are NOT in it, so the sampler is unaffected).
- Pitchers and hitters are disjoint counting paths; the change must NOT regress the hitter path, the scalar path, or the `effective_rosters=None` fallback (byte-anchor).
- Phase 5 tests are MECHANISM-ONLY (SD calibration -> Phase 6).
- `simulation.py` is under `[tool.mypy].files`; run mypy on it.

---

### Task 1: pitcher ROS-direct helper + batch wiring + blend

**Files:** Modify `src/fantasy_baseball/simulation.py`; test `tests/test_mc_integration.py`.

**Interfaces:**
- Produces: `_simulate_team_pitchers_ros_direct(effective_roster: EffectiveRoster, fraction_remaining: float, rng: np.random.Generator, n_iter: int) -> dict[str, np.ndarray]` returning `{W, K, SV, ros_ip, ros_er, ros_bb, ros_ha}` (each shape `(n_iter,)`); `_ROS_DIRECT_PITCHERS: bool = True` module flag.

- [ ] **Step 1: Write the failing tests** in `tests/test_mc_integration.py` (mechanism-only; build a constructed `EffectiveRoster` with active pitcher bodies via `build_effective_roster` or directly):

```python
def test_pitcher_helper_samples_active_only_applies_factor():
    # Two active SP + one IL pitcher pool-displaced; a benched pitcher is NOT in
    # EffectiveRoster.active so it never contributes. Mechanism: K positive,
    # a factor<1 body contributes less than at factor 1.
    import numpy as np
    from fantasy_baseball.simulation import _simulate_team_pitchers_ros_direct
    eff = _make_effective_roster_with_pitchers()  # construct: 2 active P (factor 1.0), helper builds from EffectiveRoster
    out = _simulate_team_pitchers_ros_direct(eff, 0.5, np.random.default_rng(0), 500)
    assert set(out) >= {"W", "K", "SV", "ros_ip", "ros_er", "ros_bb", "ros_ha"}
    assert out["K"].mean() > 0 and out["ros_ip"].mean() > 0


def test_pitcher_helper_empty_active_returns_zeros():
    import numpy as np
    from fantasy_baseball.simulation import _simulate_team_pitchers_ros_direct
    eff = _make_effective_roster_hitters_only()  # no active pitchers
    out = _simulate_team_pitchers_ros_direct(eff, 0.5, np.random.default_rng(0), 100)
    assert all((out[c] == 0).all() for c in ("W", "K", "SV", "ros_ip"))


def test_effective_rosters_routes_pitchers_and_no_fallback_regression():
    # With effective_rosters, the team's pitcher cats come from the ROS-direct
    # helper (W/K/SV = YTD + ROS, no clamp); with None they use top-k (byte anchor
    # still holds). Assert the two PATHS differ for a team whose active-slot
    # pitchers != its top-k pitchers, and that effective_rosters=None is unchanged.
    ...  # follow the existing test_pitchers_unchanged / fallback pattern
```

(Read the existing `_simulate_team_hitters_ros_direct` tests + `tests/test_mc_roster.py` fixtures first; construct pitcher `Player`s with `PitcherStats` rest_of_season + a `LeagueContext` so `build_effective_roster` yields pitcher active bodies with pool factors. Pin any factor expectation to OBSERVED `_compute_pitcher_pool_factors` behavior.)

- [ ] **Step 2: Run, confirm FAIL.**

- [ ] **Step 3: Implement the helper** (mirror `_simulate_team_hitters_ros_direct`, lines ~772-873, MINUS the bench-fill loop):

```python
def _simulate_team_pitchers_ros_direct(
    effective_roster: EffectiveRoster,
    fraction_remaining: float,
    rng: np.random.Generator,
    n_iter: int,
) -> dict[str, np.ndarray]:
    """Return the team's ROS-ONLY pitcher arrays: {W, K, SV} + ros_ip/ros_er/
    ros_bb/ros_ha (for the ERA/WHIP recombine). Mirrors the hitter helper but with
    NO bench injury-fill (pitcher rich-fill deferred): samples the active PITCHER
    bodies' rest_of_season lines (pt_mean_fraction=1.0 = full haircut -> mean ~=
    ERoto; suppress_repl=True -> no built-in backfill, so the haircut mean stands
    and is NOT pushed above ERoto), applies each body's displacement factor, sums.
    Healthy bench pitchers are absent from EffectiveRoster.active -> excluded.
    Caller owns the YTD blend (team_total = YTD + ROS, no clamp; ERA/WHIP recombine).
    """
    active_p_bodies = [
        b for b in effective_roster.active if b.player.player_type == PlayerType.PITCHER
    ]
    cats = {"W": "w", "K": "k", "SV": "sv"}
    zeros = np.zeros(n_iter)
    if not active_p_bodies:
        out: dict[str, np.ndarray] = {cat: zeros.copy() for cat in cats}
        for k in ("ros_ip", "ros_er", "ros_bb", "ros_ha"):
            out[k] = zeros.copy()
        return out

    active_flats = [b.player.to_flat_dict() for b in active_p_bodies]
    vb = _apply_variance_batch(
        active_flats,
        PlayerType.PITCHER,
        rng,
        fraction_remaining,
        n_iter,
        pt_mean_fraction=1.0,
        suppress_repl=True,
    )
    factors = np.array([b.factor for b in active_p_bodies])  # (n_active,)
    realized = {col: vb.counts[col] * factors[None, :] for col in PITCHING_COUNTING}

    out = {cat: realized[col].sum(axis=1) for cat, col in cats.items()}
    out["ros_ip"] = realized["ip"].sum(axis=1)
    out["ros_er"] = realized["er"].sum(axis=1)
    out["ros_bb"] = realized["bb"].sum(axis=1)
    out["ros_ha"] = realized["h_allowed"].sum(axis=1)
    return out
```

Add `_ROS_DIRECT_PITCHERS: bool = True` next to `_ROS_DIRECT_HITTERS`.

- [ ] **Step 4: Wire the batch** (`simulate_remaining_season_batch`, ~924-1030). Add `use_ros_direct_pitchers = _ROS_DIRECT_PITCHERS and eff is not None`. When true, route pitchers through the helper and SKIP the `pb`/top-k pitcher sampling for this team:

```python
        use_ros_direct_pitchers = _ROS_DIRECT_PITCHERS and eff is not None

        if eff is not None and use_ros_direct_pitchers:
            pros = _simulate_team_pitchers_ros_direct(eff, fraction_remaining, rng, n_iter)
        elif pitchers:
            pb = _apply_variance_batch(pitchers, PlayerType.PITCHER, rng, fraction_remaining, n_iter).counts
            # ... existing top-k / active_cols pitcher selection + _gather_sum (UNCHANGED) ...
```

IMPORTANT rng-stream note: today `pb = _apply_variance_batch(pitchers, ...)` is sampled UNCONDITIONALLY (line 941-943, before the hitter selection). Moving it inside the `elif` changes when/whether pitchers draw rng. This is fine for the `effective_rosters` path (the new path) but verify the `effective_rosters=None` byte-anchor still holds (None -> use_ros_direct_pitchers False -> the `elif pitchers:` branch samples pb exactly as before, in the same order). Keep the hitter `hb`/`ros` sampling ordering identical for the None path. (The pitcher distributional test already tolerates the shifted stream on the effective_rosters path.)

Then the blend (~982-1029): when `use_ros_direct_pitchers`, W/K/SV = `actuals + pros["W"/"K"/"SV"]` (NO max-clamp, ROS>=0 structural floor), and `total_ip = actual_ip + pros["ros_ip"]`, `total_er = actual_er + pros["ros_er"]`, `total_h_plus_bb = actual_h_plus_bb + pros["ros_bb"] + pros["ros_ha"]`. When NOT use_ros_direct_pitchers, the existing pitcher blend (max-clamp at 989-991, sim_w/k/sv via _gather_sum) stays EXACTLY as-is. Mirror the hitter `if use_ros_direct:` blend branch structure (996-1009) for the W/K/SV + IP/ER components.

- [ ] **Step 5: Run, confirm PASS:** `pytest tests/test_mc_integration.py -v`.
- [ ] **Step 6: No regression:** `pytest tests/test_mc_integration.py tests/test_simulation.py tests/test_mc_selection.py tests/test_web/ -q`. The `effective_rosters=None` byte-anchor and the scalar path MUST still pass.
- [ ] **Step 7: ruff check + format --check + `mypy src/fantasy_baseball/simulation.py`** (clean; ignore pre-existing category_odds.py).
- [ ] **Step 8: Commit:**
```bash
git add src/fantasy_baseball/simulation.py tests/test_mc_integration.py
git commit -m "feat(sim): ROS-direct pitcher path (active-slot + IL displacement, no fill) (Phase 5)"
```

---

## Self-Review

**Spec coverage:** Component 5 -- pitchers mirror ERoto: active-slot + IL pool-displacement (reusing the Phase-2 `EffectiveRoster` pitcher bodies, whose factors come from `_compute_pitcher_pool_factors`), healthy bench excluded. NO bench injury-fill, NO closer-role SV (deferred). `pt_mean_fraction=1.0` + `suppress_repl=True` -> pitcher means ~= ERoto (no over-credit fill). Dual-path preserved: `effective_rosters=None` keeps the exact top-k pitcher path; pitchers/hitters disjoint counting.

**Placeholder scan:** Concrete helper code + wiring; the test bodies note "construct via build_effective_roster, pin factors to observed `_compute_pitcher_pool_factors`" -- genuine TDD instructions.

**Type consistency:** Helper returns `{W,K,SV,ros_ip,ros_er,ros_bb,ros_ha}`; caller blends them mirroring the hitter ROS-direct branch. `_ROS_DIRECT_PITCHERS` flag parallels `_ROS_DIRECT_HITTERS`. The pitcher counting iterate over the verified 7-stat `PITCHING_COUNTING`.
