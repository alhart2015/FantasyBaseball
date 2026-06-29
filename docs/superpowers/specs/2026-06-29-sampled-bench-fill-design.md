# Sampled bench injury-fill (de-bias the hitter-cat under-dispersion)

Date: 2026-06-29
Status: approved design, ready for implementation plan
Scope: HITTERS only (per decision below)

## Problem

The ROS-direct hitter MC (`simulation.py:_simulate_team_hitters_ros_direct`)
samples each active hitter with a stochastic playing-time scale, computes the
per-iteration shortfall `frac_missed = max(0, 1 - scale)`, and backfills those
missed games from the bench. The bench fill line is built **once**, deterministically:

```
per_game = base_ros_total / g_ros_full          # simulation.py:898-908
```

This zero-variance fill is the root of two documented biases (Phase 6
SD-calibration gate, `docs/superpowers/games-mc-phase6-sd-evidence-2026-06-27.md`):

1. **Under-dispersion of hitter counting cats.** Because the active mean is
   haircut below 1.0 (`eff_mean = mean_scale < 1`), the fill fires on ~6-25% of
   every hitter's games **every iteration** (not rare injuries). The deterministic
   fill anti-correlates with the active body's own (reduced) draw, draining
   team-total variance. Measured: R 0.72, RBI 0.70 (hitter median 0.77) of the
   analytic SD that was backtested vs realized 2022-2025.
2. **Pro-favorite downside asymmetry.** The fill triggers only when a starter
   draws below his mean; above-mean draws keep their full overage. So each hitter
   cat is right-skewed with a truncated lower tail, which firms up a leader's thin
   category leads and inflates first-place probability.

These are not bugs — they are the acknowledged `f^2` partial-fill approximation
and its deferred refinement. This spec implements that refinement.

A win-prob sanity check (2026-06-29) measured a live MC first-place probability of
95.6% for a thin-margin roto leader; the de-biased estimate is ~80%. This change
narrows that gap.

## Decisions (locked)

- **Hitters only.** The bench-fill engine is hitter-only by design (pitcher
  bench-fill was deferred at Phase 5). The audit found pitcher category SDs are
  already calibrated-to-slightly-over-dispersed (W 1.27, K 1.01, SV 1.13) — there
  is nothing to fix on the pitcher side, and touching it risks over-dispersing a
  calibrated path.
- **Aggregate game-budget capacity** (not a day-by-day schedule). A bench body can
  cover at most its ROS game budget, now further reduced each iteration by its own
  sampled availability. No per-day calendar simulation.
- **Sampling approach A** (sample bench bodies exactly like actives, then split
  the realized line into a per-game rate + a capacity). Chosen over a two-draw
  rate/availability split (more bespoke code, diverges from the active path) and
  over a capacity-only change (does not fix the rate-driven under-dispersion).

## Design

### Behavior

When a starter is simulated to miss games, those games are filled by a bench
hitter whose **own production is sampled with the same variance model the starter
uses** (counting-stat NegBin copula + playing-time availability). The bench
filler's own availability reduces how many games it can cover; any residual
cascades to the next eligible bench hitter (ordered by projected per-game value),
and a final residual falls to a deterministic replacement line. Position
eligibility and the one-body-at-a-time capacity cap are unchanged.

### Sampling (simulation.py `_simulate_team_hitters_ros_direct`)

Immediately AFTER the existing active draw (so active draws stay byte-identical;
the bench draw only appends to the rng stream), sample the HITTER bench pool with
the same call shape actives use:

```
bench_vb = _apply_variance_batch(
    bench_flats, PlayerType.HITTER, rng, fraction_remaining, n_iter,
    pt_mean_fraction=1.0, suppress_repl=True, pt_volumes=bench_full_season_volumes,
)
```

This yields `bench_vb.counts[col]` shape `(n_iter, n_bench)` and
`bench_vb.frac_missed` shape `(n_iter, n_bench)`. Bench bodies are healthy bench
(undisplaced) so no displacement factor is applied to their realized counts.

For each `(iter, bench body b)`:

```
capacity[it, b]      = g_ros_full[b] * (1 - frac_missed[it, b])
per_game_rate[it,b][col] = realized[it,b][col] / capacity[it,b]   if capacity > EPS else 0.0
```

- `(1 - frac_missed)` is `min(1, scale)`, so capacity is naturally capped at the
  full slate `g_ros_full` when the body overperforms (`scale > 1`).
- The division recovers a per-game RATE that carries the counting-stat noise of
  the realized draw while dividing the availability back out (availability is
  represented by `capacity`, so it is not double-counted).
- **EPS guard:** when `capacity <= EPS` (bench body essentially unavailable this
  iteration), `per_game_rate = 0` and `capacity = 0`, so the allocator skips this
  body and cascades to the next eligible bench body, then replacement.

**Invariant (sanity):** a fully-utilized bench body contributes exactly its
sampled line — `capacity * per_game_rate = realized_total` — and any partial slice
`assign * per_game_rate <= realized_total`. There is no division blow-up: even if
`capacity` is tiny, `realized_total` was drawn at `mu = base * scale` (small when
`scale` is small), so the rate stays ~`base / g_ros_full` in expectation, and the
`min(need, capacity)` cap bounds the contribution by `realized_total`.

`EPS` is a small absolute games threshold (e.g. `1e-9`); it guards division, not a
modeling cutoff.

### Allocation (mc_fill.py `allocate_bench_fill` + `BenchSample`)

- Add a field `capacity: float` to `BenchSample` (games this body can cover THIS
  iteration). `per_game_counts` is already per-iteration in shape; it now carries
  the sampled rate instead of the deterministic `base_ros / g_ros_full`.
- In `allocate_bench_fill`, initialize remaining capacity from `bs.capacity`
  instead of the static `bs.body.g_ros_full`. Everything else is unchanged:
  - shortfalls largest-first;
  - eligible bench ordered by **projected** `per_game_value` then player-id
    (deterministic) — ordering by projection, not by the iteration's realized
    rate, so "play your best bench bat first" stays stable across iterations;
  - one-body capacity cap; residual after the bench pool is exhausted goes to the
    deterministic replacement line (unchanged — replacement only fires when the
    whole bench is used up, which is rare and not a bias source).

### Vectorization / performance

One extra `_apply_variance_batch` call per team over the tiny bench pool
(typically <=2 hitters). The existing per-iteration allocation loop (already
present, `<=12` active, `<=2` bench) now reads per-iteration bench arrays. Cost is
negligible; this satisfies the TODO's "vectorize the fill draw" note — the draw is
batched, only the cheap allocation stays looped.

## Out of scope

- Pitcher bench-fill (still deferred).
- Sampling the terminal replacement line (kept deterministic; fires rarely and is
  not a bias driver). Could be a future refinement if a bench-thin team's tail
  matters.
- Day-by-day roster scheduling / the one-game-per-day calendar (aggregate game
  budget approximates it, per decision).
- Re-calibrating `STAT_DISPERSION` or the `cv_pt` curves (separate TODO items).

## Testing & acceptance

### Unit (`tests/test_mc_fill.py`)
- Capacity now comes from `BenchSample.capacity`, not `body.g_ros_full`: a body
  whose `capacity` is below its `g_ros_full` fills only up to `capacity`, and the
  residual cascades to the next eligible body, then replacement.
- A body with `capacity == 0` (sampled fully unavailable) is skipped entirely and
  the shortfall cascades.
- Existing allocation invariants (position eligibility, value ordering, one-body
  cap, replacement residual) still hold.

### Integration (`tests/test_mc_integration.py`)
- For a bench-deep team, the team R/RBI total **SD is strictly larger** than the
  deterministic-fill baseline (the de-bias is observable). A focused fixture with
  a known bench body and a fixed seed.
- Determinism: same seed -> identical team totals.
- Active-only assertions (where present) are unchanged, since actives are drawn
  before the bench append.

### Acceptance gate (evidence, not a unit test)
- Re-run the Phase 6 SD-calibration diagnostic (the selection-attribution
  diagnostic, `FB_SELECTION_ATTRIBUTION=1`) on the live snapshot and record the
  per-cat `mc_sd / analytic_sd` table. **Acceptance: R and RBI ratios move from
  ~0.70 toward 1.0 and the pooled ratio stays in the gate's `[0.8, 1.25]` band.**
  Save evidence under `docs/superpowers/` alongside the prior Phase 6 evidence.

## Expected fallout (intended, not regressions)

Adding the bench draw shifts the rng stream, so any seeded golden values that
assert ROS-direct **team totals** (or their SDs) will change. This is the intended
behavior change (the bench fill now carries variance), NOT a regression. Update
those expected values, and note the justification in the commit/test so the change
is auditable per the repo's "don't silently fix tests" rule. Per-active-body and
active-only assertions are unaffected (actives draw first; their rng draws are
byte-identical).

## Files

- `src/fantasy_baseball/mc_fill.py` — add `BenchSample.capacity`; use it for
  remaining-capacity init in `allocate_bench_fill`.
- `src/fantasy_baseball/simulation.py` — `_simulate_team_hitters_ros_direct`:
  sample the bench pool, build per-iteration `BenchSample`s (capacity + sampled
  per-game rate) with the EPS guard.
- `tests/test_mc_fill.py`, `tests/test_mc_integration.py` — new/updated tests per
  above.

## Verification checklist (per CLAUDE.md)

- `pytest tests/test_mc_fill.py tests/test_mc_integration.py tests/test_simulation.py -v`
  (and any golden updates).
- `ruff check .`, `ruff format --check .`, `vulture` (no new findings).
- `mypy` — `simulation.py` and `mc_fill.py` are under `[tool.mypy].files`; must pass.
- Acceptance-gate diagnostic re-run with the SD table recorded.
