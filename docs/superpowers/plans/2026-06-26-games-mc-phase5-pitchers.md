# Games-based MC -- Phase 5 (pitchers: ROS-direct active-slot + IL displacement, no fill) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development. Steps use checkbox (`- [ ]`).

**Goal:** Bring PITCHERS onto the ROS-direct engine, mirroring the hitter path MINUS the bench injury-fill: sample the `EffectiveRoster`'s active PITCHER bodies (active-slot + IL, already pool-displaced by Phase 2's `_compute_displacement_factors`) directly, apply each body's displacement `factor`, sum, and blend `team_total = YTD + ROS`. Healthy bench pitchers are EXCLUDED (already dropped from `EffectiveRoster`). This fixes the pitcher over-credit (top-k seating deep bullpens) that, left alone, tilts 5x5 standings toward pitching-deep teams. NO bench injury-fill, NO closer-role SV modeling (both deferred).

**Why no haircut AND no fill (the key design point -- corrected after plan-review).**
ERoto's pitcher MEAN is `rest_of_season * displacement_factor` with NO playing-time
haircut (`project_team_stats` / `_apply_displacement`; `playing_time_params` feeds
ERoto's SDs ONLY, never its means). So to MIRROR ERoto, the MC pitcher mean must be
`projection * factor` -- i.e. NO mean haircut. Pass **`pt_mean_fraction=0`** ->
`eff_mean = 1 - (1-mean_scale)*0 = 1` (no haircut, mean == projection == ERoto); the
SD term keeps `cv_pt * sqrt(fraction_remaining)` (variance horizon intact). Use
`suppress_repl=True` (no built-in backfill). NOTE the difference from the HITTER
helper (which uses `pt_mean_fraction=1.0` = FULL haircut, eff_mean=mean_scale): the
general rule is **apply the haircut ONLY when there is a fill to restore it**.
Hitters apply the haircut and `allocate_bench_fill` restores it (landing ~= ERoto +
a small bench premium, as the Phase-4 evidence validated). Pitchers have NO fill, so
applying the haircut (`pt_mean_fraction=1.0`) would leave it UNRESTORED and deflate
pitcher means ~15-24% BELOW ERoto (and below the current MC) -- the standings-
corrupting bug the plan-review caught. With `pt_mean_fraction=0` the pitcher mean
matches ERoto by construction (no haircut, no fill, no premium -- the accepted
asymmetry vs hitters, since pitcher bench-fill is deferred).

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


def test_pitcher_mean_matches_projection_no_haircut():
    # CRITICAL regression: pt_mean_fraction=0 => NO playing-time mean haircut, so
    # the helper's K/IP means ~= the summed ROS projections (== ERoto), NOT
    # mean_scale*projection (~0.8x, which pt_mean_fraction=1.0 would wrongly give).
    import numpy as np
    from fantasy_baseball.simulation import _simulate_team_pitchers_ros_direct
    # One active SP, factor 1.0, ROS K=150 / IP=180 (read off the constructed body).
    eff, proj_k, proj_ip = _one_active_pitcher_factor1(k=150.0, ip=180.0)
    out = _simulate_team_pitchers_ros_direct(eff, 0.5, np.random.default_rng(0), 4000)
    # Within ~6% of the projection (NegBin + PT variance noise), and NOT ~20% low.
    assert abs(out["K"].mean() - proj_k) / proj_k < 0.06, out["K"].mean()
    assert abs(out["ros_ip"].mean() - proj_ip) / proj_ip < 0.06, out["ros_ip"].mean()


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
    NO mean haircut and NO bench injury-fill (pitcher rich-fill deferred): samples
    the active PITCHER bodies' rest_of_season lines with pt_mean_fraction=0
    (eff_mean=1 -> NO playing-time haircut -> mean == projection == ERoto, which
    applies no haircut to means) and suppress_repl=True (no backfill), applies each
    body's displacement factor, sums. Healthy bench pitchers are absent from
    EffectiveRoster.active -> excluded. Caller owns the YTD blend (team_total =
    YTD + ROS, no clamp; ERA/WHIP recombine).
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
        pt_mean_fraction=0,  # eff_mean=1: NO haircut -> mean == projection == ERoto
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

Then the blend (~982-1029). PIN BOTH sites that consume the pitcher results (else NameError when the `sim_*` block is skipped): (1) the `total_ip`/`total_er`/`total_h_plus_bb` rate-component block (~989-991), and (2) the `out[team]` W/K/SV entries (~1024-1026). Move both into a `if use_ros_direct_pitchers: ... else: ...` and assign `total_ip`/`total_er`/`total_h_plus_bb` in BOTH branches (they feed ERA/WHIP at 1013-1014 on every path):

```python
        if use_ros_direct_pitchers:
            total_ip = actual_ip + pros["ros_ip"]
            total_er = actual_er + pros["ros_er"]
            total_h_plus_bb = actual_h_plus_bb + pros["ros_bb"] + pros["ros_ha"]
            sim_w_out = actuals.get("W", 0) + pros["W"]   # YTD + ROS, no clamp
            sim_k_out = actuals.get("K", 0) + pros["K"]
            sim_sv_out = actuals.get("SV", 0) + pros["SV"]
        else:
            # EXISTING pitcher blend, verbatim (max-clamp).
            total_ip = actual_ip + np.maximum(0.0, sim_ip - actual_ip)
            total_er = actual_er + np.maximum(0.0, sim_er - actual_er)
            total_h_plus_bb = actual_h_plus_bb + np.maximum(0.0, (sim_bb + sim_ha) - actual_h_plus_bb)
            sim_w_out = np.maximum(actuals.get("W", 0), sim_w)
            sim_k_out = np.maximum(actuals.get("K", 0), sim_k)
            sim_sv_out = np.maximum(actuals.get("SV", 0), sim_sv)
```

and the `out[team]` dict reads `"W": sim_w_out, "K": sim_k_out, "SV": sim_sv_out` (replacing the inline `np.maximum(...)` at 1024-1026). The `sim_w/k/sv/ip/er/bb/ha` and `pb` are only assigned in the `elif pitchers:` (non-ros-direct) branch, so they must NOT be referenced in the `use_ros_direct_pitchers` branch. Mirror the hitter `if use_ros_direct:` blend (996-1009).

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
