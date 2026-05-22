# deltaRoto Analytic Band Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the Monte-Carlo deltaRoto confidence band with a closed-form analytic band so it is cheap enough to run inline (restoring the free-tier refresh) and consistent with the EV deltaRoto, then clean up its display on all four surfaces.

**Architecture:** The band's `mean` reuses the existing EV deltaRoto (`compute_delta_roto(...).total`). The `sd` is computed analytically: propagate the swapped players' per-category stat variance (the same `STAT_VARIANCE`-CV + `cv_pt` quadrature `project_team_sds` already uses) through each category's Gaussian roto-points curve via a small fixed Gauss-Hermite node set, then sum per-category variances. No sampling, deterministic. Display: band `mean +/- sd` only on compare/trade; band only on recommended moves on the lineup page; single Trade Finder gains the band; the redundant multi-trade re-evaluation is removed.

**Tech Stack:** Python 3.11, numpy (Gauss-Hermite nodes via `numpy.polynomial.hermite_e` or hardcoded), `math.erf`-based normal CDF (reuse `scoring._pairwise_win_prob` style), pytest.

**Spec:** `docs/superpowers/specs/2026-05-22-deltaroto-analytic-band-design.md`

---

## File Structure

- `src/fantasy_baseball/lineup/delta_roto.py` — **rewrite** the band: analytic `compute_delta_roto_band` + `compute_one_for_one_band`; delete `_sum_realized` and numpy sampling. Add `_swap_category_variance` + `_category_points_band` helpers.
- `src/fantasy_baseball/scoring.py` — extract a reusable per-player per-category **variance** helper (factored from `project_team_sds`) so the band and `project_team_sds` share one source of truth.
- `src/fantasy_baseball/lineup/optimizer.py` — drop `n_draws`/`seed`; keep per-starter `roto_delta`, attach band only where a move is recommended (see Task 6).
- `src/fantasy_baseball/lineup/roster_audit.py` — drop `n_draws`/`seed` from the `compute_one_for_one_band` call.
- `src/fantasy_baseball/web/season_data.py` — `compute_comparison_standings`: stop returning the separate EV `delta_roto` for the headline; band is the headline.
- `src/fantasy_baseball/trades/multi_trade.py` — `evaluate_multi_trade` uses the analytic band (drop `n_draws`).
- `src/fantasy_baseball/web/season_routes.py` — `/api/optimize-trade-lineup` drops the redundant `evaluate_multi_trade`; `/api/trade-search` gains the band.
- Templates: `players.html` (band-only headline), `_lineup_hitters_tbody.html` / `_lineup_pitchers_tbody.html` (remove per-row band), `lineup.html` (band on moves only — already present), `waivers_trades.html` (band on single-trade cards).
- `tests/test_lineup/test_delta_roto_band.py` — **rewrite** for the analytic path.
- `tests/test_lineup/test_optimizer.py`, `tests/test_lineup/test_roster_audit.py`, `tests/test_web/test_season_data.py`, `tests/test_trades/test_multi_trade.py`, `tests/test_web/test_evaluate_trade_route.py` — update for new signatures/shape.

**Phasing (CLAUDE.md: <=5 files/phase, verify + approval between phases):**
- **Phase 1 (core):** scoring helper + `delta_roto.py` rewrite + its tests. Restores cheap band.
- **Phase 2 (refresh hot path):** `roster_audit.py`, `optimizer.py` + their tests. Restores the refresh.
- **Phase 3 (compare):** `season_data.py`, `players.html` + test.
- **Phase 4 (trade):** `multi_trade.py`, `season_routes.py`, `waivers_trades.html` + tests.

---

## Phase 1 — Core analytic band

### Task 1: Extract a per-player per-category variance helper in scoring.py

**Files:**
- Modify: `src/fantasy_baseball/scoring.py` (factor from `project_team_sds`, ~line 941)
- Test: `tests/test_scoring.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_scoring.py
def test_player_category_variance_counting_matches_team_sds():
    """A single hitter's category variance equals that player's contribution
    to project_team_sds (variances add across players)."""
    from fantasy_baseball.scoring import player_category_variance, project_team_sds
    from fantasy_baseball.utils.constants import Category
    # Two identical hitters -> team variance == 2 * single-player variance.
    players = [make_hitter(r=100, hr=30, rbi=90, sb=10, h=150, ab=550),
               make_hitter(r=100, hr=30, rbi=90, sb=10, h=150, ab=550)]
    team_sd_r = project_team_sds(players, displacement=False)[Category.R]
    one = player_category_variance(players[0])[Category.R]
    assert one > 0
    assert team_sd_r ** 2 == pytest.approx(2 * one, rel=1e-6)
```

(Use the existing hitter/pitcher factory in `tests/test_scoring.py`; if none, build a `Player` with a `HitterStats` rest_of_season as other tests there do.)

- [ ] **Step 2: Run to verify it fails** — `pytest tests/test_scoring.py::test_player_category_variance_counting_matches_team_sds -v` — Expected: FAIL (`player_category_variance` undefined).

- [ ] **Step 3: Implement** `player_category_variance(player) -> dict[Category, float]` in `scoring.py`, returning each category's variance contribution for ONE player. Factor the per-player body of `project_team_sds`'s loop so both call it. Counting cats: `v**2 * (STAT_VARIANCE[k]**2 + cv_pt**2)`. Rate cats need team denominators, so `player_category_variance` returns counting-cat variances and the rate **component** sums (`h_sq`, `er_sq`, `bb_sq`, `ha_sq`, plus `ab`, `ip`) needed to assemble rate variance at the team level. Keep `project_team_sds` behavior identical (it now sums these helper outputs).

- [ ] **Step 4: Run** — Expected: PASS. Also run `pytest tests/test_scoring.py -v` to confirm `project_team_sds` tests still pass (refactor preserved behavior).

- [ ] **Step 5: Commit** — `git commit -m "refactor(scoring): extract per-player category variance helper"`

### Task 2: Analytic `compute_delta_roto_band` — counting categories

**Files:**
- Modify: `src/fantasy_baseball/lineup/delta_roto.py`
- Test: `tests/test_lineup/test_delta_roto_band.py` (rewrite)

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_lineup/test_delta_roto_band.py (new content)
import pytest
from fantasy_baseball.lineup.delta_roto import compute_delta_roto_band, compute_delta_roto

def test_band_mean_equals_ev_delta(sample_swap):
    band = compute_delta_roto_band(**sample_swap.band_kwargs)
    point = compute_delta_roto(**sample_swap.point_kwargs)
    assert band.mean == pytest.approx(point.total, abs=1e-9)

def test_band_sd_positive_for_real_swap(sample_swap):
    band = compute_delta_roto_band(**sample_swap.band_kwargs)
    assert band.sd > 0
    assert 0.0 <= band.p_positive <= 1.0

def test_identity_swap_zero_band(identity_swap):
    band = compute_delta_roto_band(**identity_swap.band_kwargs)
    assert band.mean == pytest.approx(0.0, abs=1e-6)
    assert band.sd == pytest.approx(0.0, abs=1e-6)

def test_determinism(sample_swap):
    a = compute_delta_roto_band(**sample_swap.band_kwargs)
    b = compute_delta_roto_band(**sample_swap.band_kwargs)
    assert (a.mean, a.sd, a.p_positive) == (b.mean, b.sd, b.p_positive)
```

(Build `sample_swap` / `identity_swap` fixtures from the existing 2-team standings + roster helpers already used in `test_delta_roto_band.py`. `band_kwargs` and `point_kwargs` carry the before/after rosters, field, team_name, team_sds, fraction_remaining.)

- [ ] **Step 2: Run to verify fail** — Expected: FAIL (signature mismatch / mean not equal because old MC path).

- [ ] **Step 3: Implement the analytic band.** Replace the MC body of `compute_delta_roto_band`:

```python
def compute_delta_roto_band(
    before_players, after_players, field_stats, team_name,
    fraction_remaining, *, projected_standings, team_sds,
) -> DeltaRotoBand:
    # mean: reuse EV deltaRoto so the band is consistent everywhere.
    mean = _ev_delta_total(before_players, after_players, projected_standings,
                           team_name, team_sds)
    # sd: per-category Gaussian propagation of swapped-player variance.
    in_players, out_players = _swap_sets(before_players, after_players)
    var_total = 0.0
    for cat in ALL_CATEGORIES:
        sigma2 = _swap_category_variance(in_players, out_players, cat,
                                         before_players, after_players,
                                         fraction_remaining)
        if sigma2 <= 0:
            continue
        var_total += _category_delta_variance(cat, before_players, after_players,
                                               field_stats, team_sds, sigma2)
    sd = math.sqrt(var_total)
    p_positive = _normal_cdf(mean / sd) if sd > 0 else (1.0 if mean > 0 else 0.0)
    return DeltaRotoBand(mean=mean, sd=sd, p_positive=p_positive)
```

`_category_delta_variance` integrates `dpts_c(mu_b + dX) - dpts_c(mu_b)` over `dX ~ N(d_mu, sigma2)` with a fixed Gauss-Hermite node set, where `dpts_c(x) = sum_j Phi((x - mu_j)/s_cj)` over the 9 field teams (`mu_j` from `field_stats[cat]`, `s_cj = sqrt(sd_me^2 + sd_j^2)` from `team_sds`), flipping sign for `INVERSE_STATS` (ERA/WHIP). Return `E[dpts^2] - E[dpts]^2`. Hardcode 9 Gauss-Hermite nodes/weights (or `numpy.polynomial.hermite_e.hermegauss(9)`). `_normal_cdf(z) = 0.5*(1+erf(z/sqrt(2)))`.

- [ ] **Step 4: Run** — Expected: PASS for counting-driven swaps. (Rate categories handled in Task 3.)

- [ ] **Step 5: Commit** — `git commit -m "feat(delta_roto): analytic band core (counting cats)"`

### Task 3: Rate-category variance + honest-signal test

**Files:**
- Modify: `src/fantasy_baseball/lineup/delta_roto.py` (`_swap_category_variance` rate branch)
- Test: `tests/test_lineup/test_delta_roto_band.py`

- [ ] **Step 1: Write the failing test (the honest-signal lock):**

```python
def test_noisy_category_swap_has_wider_band(equal_mean_swaps):
    """An SB- or SV-driven swap (high league CV) yields a wider sd than an
    R- or K-driven swap of equal mean. This is the only player-independent
    thing the band legitimately encodes."""
    sb_swap, r_swap = equal_mean_swaps  # constructed with ~equal band.mean
    sb = compute_delta_roto_band(**sb_swap.band_kwargs)
    r = compute_delta_roto_band(**r_swap.band_kwargs)
    assert sb.mean == pytest.approx(r.mean, abs=0.25)
    assert sb.sd > r.sd
```

- [ ] **Step 2: Run to verify fail** — Expected: FAIL until rate/SB variance fully wired (or until the fixture's SV/SB variance is propagated).

- [ ] **Step 3: Implement rate-category variance.** For AVG/ERA/WHIP, `sigma2` for the swap comes from the change in the team's rate sd between before and after rosters using `project_team_sds`'s rate formula (numerator CV over the team denominator); use `project_team_sds(after)[cat]` and `project_team_sds(before)[cat]` to derive the swap's marginal rate variance (the rate variance is not additive per player because of the shared denominator — compute from before/after team sds). Counting cats keep the additive `player_category_variance` sum from Task 1.

- [ ] **Step 4: Run** — Expected: PASS. Run the whole file: `pytest tests/test_lineup/test_delta_roto_band.py -v`.

- [ ] **Step 5: Commit** — `git commit -m "feat(delta_roto): rate-category band variance + honest-signal test"`

### Task 4: `compute_one_for_one_band` + cleanup

**Files:**
- Modify: `src/fantasy_baseball/lineup/delta_roto.py`
- Test: `tests/test_lineup/test_delta_roto_band.py`

- [ ] **Step 1: Write failing test** for `compute_one_for_one_band(drop_name, add_player, active_players, field_stats, team_name, fraction_remaining, *, projected_standings, team_sds)` returning a band whose mean matches the 1-for-1 EV delta.
- [ ] **Step 2: Run — fail** (old signature has `n_draws`/`seed`).
- [ ] **Step 3: Implement** the wrapper (build before/after from drop/add, delegate to `compute_delta_roto_band`). **Delete** `_sum_realized` and the `numpy` / `_apply_variance` / `_flatten_full_season` imports.
- [ ] **Step 4: Run — pass.** Then `vulture src/fantasy_baseball/lineup/delta_roto.py` — no new dead code.
- [ ] **Step 5: Commit** — `git commit -m "feat(delta_roto): analytic one-for-one band; drop MC sampler"`

### Task 5: Phase 1 verification

- [ ] Run `pytest tests/test_lineup/test_delta_roto_band.py tests/test_scoring.py -v` — all pass.
- [ ] `ruff check src/fantasy_baseball/lineup/delta_roto.py src/fantasy_baseball/scoring.py` + `ruff format --check` — clean.
- [ ] `mypy src/fantasy_baseball/lineup/delta_roto.py src/fantasy_baseball/scoring.py` — clean (both in `[tool.mypy].files`).
- [ ] **STOP for user approval before Phase 2** (CLAUDE.md phased execution).

---

## Phase 2 — Refresh hot path (restores the refresh)

### Task 6: optimizer.py — drop MC params, band only on moves

**Files:**
- Modify: `src/fantasy_baseball/lineup/optimizer.py:255-267, 336-348`
- Test: `tests/test_lineup/test_optimizer.py`

- [ ] **Step 1: Write failing test** asserting `optimize_hitter_lineup(...)` assignments carry `roto_delta` for every starter but `band` is populated only when that starter corresponds to a recommended change (or: bands computed for all starters but cheap — see decision below), and that no `n_draws`/`seed` kwargs are accepted.
- [ ] **Step 2: Run — fail.**
- [ ] **Step 3: Implement.** Replace the `compute_delta_roto_band(..., n_draws=300, seed=0)` calls with the analytic call (no `n_draws`/`seed`). **Display decision (per spec):** the lineup page shows the band only on recommended moves, so the optimizer may keep computing the per-starter band cheaply (analytic) but the template (Task 9) only renders it on moves. Simplest: keep per-starter band (now cheap), change only the template. Keep this task to the signature change.
- [ ] **Step 4: Run — pass.**
- [ ] **Step 5: Commit** — `git commit -m "refactor(optimizer): analytic band, drop MC params"`

### Task 7: roster_audit.py — drop MC params

**Files:**
- Modify: `src/fantasy_baseball/lineup/roster_audit.py:351-360`
- Test: `tests/test_lineup/test_roster_audit.py`

- [ ] **Step 1: Write failing test** that `audit_roster(...)` candidates carry a `band` with `mean` == the candidate's `delta_roto.total` (consistency), using a small 2-team fixture.
- [ ] **Step 2: Run — fail.**
- [ ] **Step 3: Implement** — replace the `compute_one_for_one_band(..., n_draws=300, seed=0)` call with the analytic signature. No other audit logic changes.
- [ ] **Step 4: Run — pass.** Then a perf smoke: `audit_roster` over a ~50-FA fixture completes in well under 2s.
- [ ] **Step 5: Commit** — `git commit -m "refactor(roster_audit): analytic band, drop MC params"`

### Task 8: Phase 2 verification

- [ ] `pytest tests/test_lineup -v` — all pass.
- [ ] `ruff check`/`format --check` + `mypy` on `optimizer.py`, `roster_audit.py` — clean.
- [ ] Run the refresh integration test: `pytest tests/test_web/test_refresh_pipeline.py -v` — passes (no MC blowup).
- [ ] **STOP for user approval before Phase 3.**

---

## Phase 3 — Compare surface

### Task 9: season_data + players.html — band-only headline

**Files:**
- Modify: `src/fantasy_baseball/web/season_data.py:781-816` (`compute_comparison_standings`)
- Modify: `src/fantasy_baseball/web/templates/season/players.html:645-683` (`renderDeltaRoto`)
- Modify: lineup templates `_lineup_hitters_tbody.html:79`, `_lineup_pitchers_tbody.html:70` (remove per-row band cell; keep `roto_delta`)
- Test: `tests/test_web/test_season_data.py`

- [ ] **Step 1: Write failing test** that `compute_comparison_standings(...)` returns a `band` whose `mean` equals the EV `delta_roto.total` (so the headline is unambiguous), and still returns per-category deltas for the standings table.
- [ ] **Step 2: Run — fail.**
- [ ] **Step 3: Implement.** Backend: keep computing the band; the per-category `delta_roto` stays in the payload for the standings table, but `renderDeltaRoto` is changed to render **only** the band line (`mean +/- sd`, colored by `verdict`) and drop the separate "deltaRoto: X roto pts" EV line. Remove the per-row band cell from the two lineup tbody partials (settled rows show plain `roto_delta`).
- [ ] **Step 4: Run — pass.** Manual: `/players?compare=...` shows one band line, no contradictory number.
- [ ] **Step 5: Commit** — `git commit -m "feat(compare,lineup): band-only headline; declutter lineup rows"`

### Task 10: Phase 3 verification — `pytest tests/test_web/test_season_data.py -v`, ruff/mypy clean. **STOP for approval.**

---

## Phase 4 — Trade surfaces

### Task 11: multi_trade.py — analytic band

**Files:**
- Modify: `src/fantasy_baseball/trades/multi_trade.py:263-288`
- Test: `tests/test_trades/test_multi_trade.py`

- [ ] **Step 1: Write failing test** that `evaluate_multi_trade(...).band` has `mean` consistent with `ev_roto.delta_total` and is computed without `n_draws`.
- [ ] **Step 2: Run — fail.**
- [ ] **Step 3: Implement** — replace `compute_delta_roto_band(..., n_draws=400, seed=0)` with the analytic call.
- [ ] **Step 4: Run — pass.**
- [ ] **Step 5: Commit** — `git commit -m "refactor(multi_trade): analytic band"`

### Task 12: season_routes — drop redundant evaluate, wire band into trade-search

**Files:**
- Modify: `src/fantasy_baseball/web/season_routes.py:878-890` (`/api/optimize-trade-lineup`)
- Modify: `src/fantasy_baseball/web/season_routes.py` (`/api/trade-search` handler — add band per candidate)
- Modify: `src/fantasy_baseball/web/templates/season/waivers_trades.html:235-255` (`renderTradeCard` — show band)
- Test: `tests/test_web/test_evaluate_trade_route.py` (+ a trade-search test)

- [ ] **Step 1: Write failing tests:** (a) `/api/optimize-trade-lineup` returns slot assignments without computing a band (assert it does not call the band — or simply that it still returns `ok` and is fast); (b) `/api/trade-search` response candidates include a `band` dict with `mean/sd/verdict`.
- [ ] **Step 2: Run — fail.**
- [ ] **Step 3: Implement.** In `/api/optimize-trade-lineup`, replace the full `evaluate_multi_trade` legality call with the size-only `_can_roster_after`/`_target_size` check (import from `multi_trade`) — no band. In `/api/trade-search`, compute `compute_one_for_one_band` per candidate and include `band` in each card payload. In `waivers_trades.html` `renderTradeCard`, render the band line next to the `hart_delta`.
- [ ] **Step 4: Run — pass.**
- [ ] **Step 5: Commit** — `git commit -m "feat(trade): band on single-trade finder; drop redundant multi-trade evaluate"`

### Task 13: Phase 4 + full verification

- [ ] `pytest tests/test_trades tests/test_web -v` — pass.
- [ ] Full checklist at repo root: `pytest -n auto` (or affected subset, state which), `ruff check .`, `ruff format --check .`, `vulture`, `mypy`.
- [ ] Open PR.

---

## Self-Review notes

- Spec coverage: mean==EV (Tasks 2,7,9,11), sd analytic (Tasks 2-3), honest-signal (Task 3), refresh restored (Phase 2), compare band-only (Task 9), lineup declutter (Task 9), trade single+multi (Tasks 11-12). All covered.
- The exact numeric form of `_category_delta_variance` (curve softness `s_cj`, rate-category variance) is pinned by the Phase-1 tests; the executor refines the reference code against them under TDD.
- New symbols defined before use: `player_category_variance` (Task 1), `compute_delta_roto_band`/`compute_one_for_one_band` (Tasks 2-4) before their callers (Phases 2-4).
