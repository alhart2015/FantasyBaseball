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

Immediately AFTER the existing active draw (so the active draw is fully consumed
before the bench draw starts), sample the HITTER bench pool with the same call
shape actives use. First build the bench inputs (mirroring the active path at
`simulation.py:876-879` — `BenchBody` does not carry these, they are derived from
`bb.player`):

```
bench_flats = [bb.player.to_flat_dict() for bb in bench_h_bodies]
bench_full_season_volumes = np.array(
    [_full_season_pt_volume(bb.player, is_hitter=True) for bb in bench_h_bodies]
)
bench_vb = _apply_variance_batch(
    bench_flats, PlayerType.HITTER, rng, fraction_remaining, n_iter,
    pt_mean_fraction=1.0, suppress_repl=True, pt_volumes=bench_full_season_volumes,
)
```

`_full_season_pt_volume` falls back to ROS volume when a body lacks a full-season
projection — identical to the active path, so no special-casing.

**`_apply_variance_batch` must expose `scales`.** Today `VarianceBatch` carries
`counts` and `frac_missed = max(0, 1 - scales)`, which CLAMPS AWAY the
over-availability case (`scale > 1` all map to `frac_missed = 0`). The rate
decomposition below needs the unclamped `scale` (the games that actually generated
`realized`). So add a `scales: np.ndarray` field (shape `(n_iter, n_players)`) to
`VarianceBatch`, with **NO default**, and update BOTH of its construction sites:
the populated return at `simulation.py:820` passes the `scales` array already
computed at `:789`; the `n_players == 0` early return at `:759-763` (which runs
BEFORE `scales` is computed) must pass `scales=np.zeros((n_iter, 0))` to stay
shape-correct. Missing the early-return site is the trap — it is the exact path the
empty-bench no-op invariant (see Expected fallout) depends on. Existing consumers
(the active path, the top-k path) read only `counts`/`frac_missed`, so they are
unaffected. Note `frac_missed = max(0, 1 - scales)` is now derivable from `scales`;
keeping both is intentional (the active path at `:896` reads `frac_missed`), not an
oversight — `scales` is the authoritative field, `frac_missed` a convenience view.

So the bench draw yields `bench_vb.counts[col]`, `bench_vb.scales`, each shape
`(n_iter, n_bench)`. Bench bodies are healthy bench (undisplaced), so no
displacement factor is applied to their realized counts.

For each `(iter, bench body b)`:

```
games_played_sim[it, b]  = g_ros_full[b] * scales[it, b]          # UNCAPPED -- the games that generated realized
capacity[it, b]          = games_played_sim[it, b]                # how many of the starter's missed games b can cover
per_game_rate[it,b][col] = realized[it,b][col] / games_played_sim[it,b]   if games_played_sim > EPS else 0.0
```

- **Why divide by the UNCAPPED `g*scale`, not `g*(1-frac_missed)`:** `realized` is
  drawn at `mu = base*scale` over `g*scale` games. Dividing by the games that
  generated it recovers an UNBIASED per-game rate
  (`E[rate | scale] = base*scale/(g*scale) = base/g`). Dividing by a denominator
  capped at `g` (the `1 - frac_missed` form) would, for `scale > 1`, leave the
  >full-slate overage in the numerator while shrinking the denominator —
  inflating the rate by `max(1, scale)` and introducing a directional UPWARD mean
  shift in the fill. We want the change to be variance-correcting, mean-neutral,
  so divide by `g*scale`. (Resolving adversarial finding #1.)
- Capacity is the same `games_played_sim` (uncapped). A body sampled MORE available
  than projected (`scale > 1`) can legitimately cover more of an injured starter's
  games — that is the injury-insurance scenario — and it is bounded anyway by the
  starter's actual shortfall `need`. A body sampled LESS available (`scale < 1`)
  can cover fewer, cascading the residual.
- The rate carries the counting-stat noise of the `realized` draw while the
  availability is represented exactly once, in `capacity`/`games_played_sim` — not
  double-counted.
- **EPS guard:** when `games_played_sim <= EPS` (body essentially unavailable this
  iteration), `per_game_rate = 0` and `capacity = 0`, so the allocator skips this
  body and cascades to the next eligible bench body, then replacement. This also
  covers a `g_ros_full == 0` bench body (always-skipped, same end-state as today's
  zero per-game line).

**Invariant (sanity, holds for all `scale`):** a fully-utilized bench body
contributes exactly its sampled line — `capacity * per_game_rate =
games_played_sim * realized/games_played_sim = realized_total` — and any partial
slice `assign * per_game_rate <= realized_total` (the allocator decrements
`remaining` from `capacity`, so `sum(assign) <= capacity`). **No division blow-up
even for a single absurd draw:** a tiny `games_played_sim` can produce a large
per-game rate for one iteration (e.g. a NegBin `realized = 1` over `0.15` games),
but a body's TOTAL contribution is bounded by `capacity * per_game_rate =
realized_total` regardless of how large the rate is — the capacity cap, not the
rate magnitude, is what bounds the output. (This is the exact mechanism; do not
rely on "the rate stays small," which is false for individual draws.)

`EPS` is a small absolute games threshold (e.g. `1e-9`); it guards division, not a
modeling cutoff.

### Allocation (mc_fill.py `allocate_bench_fill` + `BenchSample`)

- Add a field `capacity: float` to `BenchSample` (games this body can cover THIS
  iteration), with **NO default** (force every call site to pass it explicitly;
  a `0.0` default would silently zero-fill a missed site). `BenchSample` is
  `@dataclass(frozen=True)`, so both constructors must be updated: the sampler in
  `simulation.py` and the `_bench_sample` factory in `tests/test_mc_fill.py`.
- **`BenchSample`s are now built PER ITERATION, not once.** Today they are a single
  list of deterministic samples built ONCE outside the `for it in range(n_iter)`
  loop (`simulation.py:900-908`) and reused. The change moves construction INSIDE
  the loop (or pre-builds per-iteration arrays and indexes them), reading
  `bench_vb.counts[col][it, b]`, `bench_vb.scales[it, b]` to compute that
  iteration's `capacity` and `per_game_rate`. (`per_game_counts` stays a scalar
  `dict[str, float]`; what changes is that a fresh `BenchSample` is constructed for
  each `it`. Resolving adversarial finding #4.) The indexed values are `np.float64`;
  wrap `capacity` and each `per_game_rate[col]` in `float(...)` so the
  `dict[str, float]` / `capacity: float` annotations stay accurate and `mypy`
  (which covers both files) passes.
- In `allocate_bench_fill`, initialize remaining capacity from `bs.capacity`
  instead of the static `bs.body.g_ros_full`. Everything else is unchanged:
  - shortfalls largest-first;
  - eligible bench ordered by **projected** `per_game_value` then player-id
    (deterministic) — ordering by projection, not by the iteration's realized
    rate, so "play your best bench bat first" stays stable across iterations;
  - one-body capacity cap; residual after the bench pool is exhausted goes to the
    deterministic replacement line (mechanism unchanged).

**Known residual — increased replacement firing (adversarial finding #5).** The
deterministic baseline gave each bench body capacity `g_ros_full`; the new capacity
is `g_ros_full * scale`, which is on average slightly below `g_ros_full` and can be
much smaller in iterations where the bench body is itself sampled injured —
precisely the heavy-injury tail this change targets. So more shortfall cascades to
the (still deterministic) replacement line than before, re-damping that tail. This
is acceptable for now: the user explicitly chose replacement-level as the terminal
floor, and sampling the replacement line is a listed future refinement. But because
it works against the de-bias, the acceptance evidence MUST measure the
replacement-fill share (fraction of filled games served by replacement) before vs
after; if it rises materially (e.g. the bench-thin teams' R/RBI ratios stall below
the band), sampling the replacement line is the next step.

### Vectorization / performance

One extra `_apply_variance_batch` call per team over the tiny bench pool
(typically <=2 hitters). The existing per-iteration allocation loop (already
present, `<=12` active, `<=2` bench) now reads per-iteration bench arrays and
rebuilds the (cheap) `BenchSample` dicts each iteration. The draw is batched
(satisfying the TODO's "vectorize the fill draw" note); only the cheap allocation
plus per-iteration dict-builds stay looped. Cost is negligible relative to the
copula draws and does not change asymptotics (the loop already ran per-iteration).

### AVG components

`per_game_rate` covers all of `HITTING_COUNTING` including `h` and `ab`, so the
AVG numerator/denominator the caller recombines (`ros_h`/`ros_ab`) carry the bench
draw's variance too. Note the asymmetry (consistent with the active path, not a
gap): `h` is in the NegBin copula so it carries count noise, while `ab` is only
playing-time-scaled (`base_ab * scale`, no count noise) — so AVG variance is
de-biased but not symmetric across numerator and denominator.

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
- A body with `capacity == 0` (sampled fully unavailable this iteration) is skipped
  entirely and the shortfall cascades.
- A `g_ros_full == 0` bench body (admitted by `build_effective_roster` with
  `per_game_value == 0`) reaches `capacity == 0` via the EPS guard and is
  always-skipped — same end-state as today's zero per-game line, but via a
  different path; test it explicitly alongside the sampled-unavailable case.
- Existing allocation invariants (position eligibility, value ordering by projected
  `per_game_value`, one-body cap, replacement residual) still hold.

### Integration (`tests/test_mc_integration.py`)
- For a bench-deep team, the team R/RBI total **SD is strictly larger** than the
  deterministic-fill baseline (the de-bias is observable). A focused fixture with
  a known bench body and a fixed seed.
- Determinism: same seed -> identical team totals.
- Active-only assertions (where present) are unchanged, since actives are drawn
  before the bench append.

### Acceptance gate (evidence, not a unit test)
Re-run the Phase 6 SD-calibration diagnostic (the selection-attribution
diagnostic, `FB_SELECTION_ATTRIBUTION=1`) on the live snapshot, seed=42,
n_iter=1000, and record the per-cat `mc_sd / analytic_sd` table plus hitter means.
Acceptance requires ALL of (resolving adversarial finding #2 — a falsifiable,
bounded, mean-aware gate):

1. **R and RBI rise materially toward 1.0** from the ~0.70-0.72 baseline.
   Pass/fail is two hard bounds plus a directional target, because the exact
   landing point is not modeled in advance (sampling the bench but not the still
   -deterministic replacement line — which fires MORE now, see #5 — could hold the
   ratios short of full calibration):
   - **HARD FAIL if either ratio exceeds `1.20`** (over-correction / over-dispersion
     — the prior pooled-only gate could mask this per-cat).
   - **HARD FAIL if either ratio does not improve by at least `+0.08`** over its
     baseline (i.e. no real de-bias).
   - **TARGET `>= 0.85`.** Landing in `[0.78, 0.85)` is a partial success: record it,
     and decide whether to also sample the replacement line (the #5 follow-up)
     rather than loosening this target post hoc.
2. **The pooled ratio stays in `[0.8, 1.25]`** (unchanged gate-wide band).
3. **Hitter category team-total MEANS stay within tolerance of the Phase 6
   baseline** (HITTERS = ERoto + the small bench-fill premium). The decomposition's
   per-game RATE is mean-neutral by construction (divide by `g*scale`), but the
   TOTAL fill is not perfectly mean-neutral: in the capacity-bound heavy-injury tail
   the reduced capacity shifts some games to the deterministic replacement line
   (the intended finding-#5 re-damping), giving a small DOWNWARD pressure. So the
   tolerance is deliberately asymmetric:
   - **HARD FAIL on any UPWARD drift `> +1%`** in a hitter cat's team-total mean —
     that is the finding-#1 mean-bias failure mode and must not occur.
   - **A DOWNWARD drift up to `~5%` is expected and acceptable** (the #5 effect); a
     downward drift `> 5%` means replacement firing is over-damping the mean and
     triggers the sample-the-replacement-line follow-up.
4. **Replacement-fill share** (fraction of filled games served by the deterministic
   replacement line) is recorded before vs after; a material rise is flagged (see
   the "increased replacement firing" residual above).

Save evidence under `docs/superpowers/` alongside the prior Phase 6 evidence.

## Expected fallout (intended, not regressions)

`simulate_remaining_season_batch` threads ONE shared `rng` Generator through the
team loop (`simulation.py:1045`), drawing each team's hitters then pitchers in
order. Inserting the bench draw inside `_simulate_team_hitters_ros_direct`
advances that shared stream, so it shifts **every draw sequenced after it** — the
SAME team's pitcher draw, and ALL draws (hitters and pitchers) of every team
processed later in the loop. So the real blast radius is broad: treat essentially
all seeded ROS-direct **team-total / SD goldens across the batch** as intentionally
changed (the bench fill now carries variance — the whole point), and update them
with justification noted in the commit/test per the repo's "don't silently fix
tests" rule. (Resolving adversarial finding #3 — the prior draft wrongly implied
only first-team hitters shift.)

Two things stay byte-identical and MUST be preserved (do not "fix" them):

- **The FIRST ROS-direct team's hitter ACTIVE draw** — actives are drawn before the
  appended bench draw, so the active rng draws for the first such team are
  unchanged. (Later teams' actives shift because an earlier team's bench draw
  already advanced the stream.)
- **Empty-bench ROS-direct teams** — `_apply_variance_batch` returns before any rng
  draw when `n_players == 0` (`simulation.py:759-763`), so a team with no bench
  hitters consumes ZERO extra rng and stays byte-identical end-to-end. The
  implementation MUST keep the empty-bench path a true no-op (don't introduce a draw
  that consumes rng for an empty pool), or these guardrail tests
  (`test_repl_not_double_counted_on_new_path`,
  `test_ros_direct_uses_full_season_volume_for_cv_pt`,
  `test_displacement_factor_scales_hitter_mean`, and similar empty-bench fixtures)
  will break. (Resolving adversarial finding #8.)

## Files

- `src/fantasy_baseball/mc_fill.py` —
  - add `BenchSample.capacity` (no default); use it for remaining-capacity init in
    `allocate_bench_fill`;
  - fix the now-stale/false docstring + comment: `allocate_bench_fill`'s docstring
    (`mc_fill.py:43-52`) asserts "One bench body's total assigned games `<=` its
    `g_ros_full`", which the design DELIBERATELY breaks (capacity = `g_ros_full*scale`
    can exceed `g_ros_full` when `scale > 1`, the injury-insurance case). Rewrite it
    to the new `bs.capacity` cap, and update the `remaining = {... g_ros_full ...}`
    init comment (`:55-56`).
- `src/fantasy_baseball/simulation.py` —
  - add `scales: np.ndarray` to `VarianceBatch` and populate it from the existing
    `scales` array in `_apply_variance_batch` (back-compatible);
  - `_simulate_team_hitters_ros_direct`: build `bench_flats` /
    `bench_full_season_volumes`, sample the bench pool, build per-iteration
    `BenchSample`s (capacity = `g_ros_full*scale`, rate = `realized/(g_ros_full*scale)`)
    with the EPS guard; keep the empty-bench path a true rng no-op;
  - update the helper's docstring (`simulation.py:860-862`) — the "Bench per-game
    lines are the clean DETERMINISTIC base ROS projection ... built once" sentence
    is now false and must describe the sampled per-iteration fill (resolving
    finding #10).
- `tests/test_mc_fill.py`, `tests/test_mc_integration.py` — new/updated tests per
  above (including the `_bench_sample` factory's new `capacity` arg). **Retarget the
  named test `test_fill_never_exceeds_bench_g_ros_full_capacity`** (`test_mc_fill.py`,
  ~line 105): it encodes the old `<= g_ros_full` cap, which the design intentionally
  replaces with the per-iteration `bs.capacity` cap. Rename/rewrite it to assert
  `sum(assigned) <= bs.capacity` (which CAN exceed `g_ros_full` when `scale > 1`),
  not the old static cap — otherwise it fails by design. State the justification in
  the commit per the "don't silently fix tests" rule (this is a deliberate contract
  change, not a loosened assertion).

## Verification checklist (per CLAUDE.md)

- `pytest tests/test_mc_fill.py tests/test_mc_integration.py tests/test_simulation.py -v`
  (and any golden updates).
- `ruff check .`, `ruff format --check .`, `vulture` (no new findings).
- `mypy` — `simulation.py` and `mc_fill.py` are under `[tool.mypy].files`; must pass.
- Acceptance-gate diagnostic re-run with the SD table recorded.
