# Decompose `luck` into playing-time and batted-ball components

- **Issue:** #277 (from the Rafaela investigation on #273)
- **Branch:** `feat/277-decompose-luck`, stacked on `feat/273-league-keeper-board` (PR #279, unmerged)
- **Status:** approved design, ready for planning

## Problem

The keeper composite blends four percentile families -- `skill`, `luck`, `future`,
`age` -- with `luck = value_pct - skill_pct` carrying a **positive** weight (0.8
hitters, 0.6 pitchers). That positive weight is justified because `luck` proxies
next-year playing time (`luck -> next-yr PT` +0.38). But a single scalar bundles
two things that behave differently out of sample:

1. **Playing time / role / durability** -- real, repeatable signal. An everyday
   player keeps his job; a closer keeps the ninth. This is what earns `luck` its
   weight.
2. **Batted-ball rate luck** -- AVG above xBA, ERA below FIP. This regresses and
   should NOT be rewarded. Ceddanne Rafaela (.278 AVG vs .242 xBA) and Otto Lopez
   are the motivating cases: everyday players whose keeper rank is inflated by
   rate overperformance the peripherals do not support.

Because the two were never entered as separate regressors, the current fit cannot
tell them apart, and the model oversells lucky everyday players.

## Goals

- Separate the playing-time signal (keep rewarding it) from the batted-ball rate
  luck (stop rewarding it) inside the `luck` family.
- Let the **holdout backtest decide** which parameterization ships, under the
  existing objective (Spearman rho vs next-year SGP percentile) -- #276 closed
  without changing that objective, so this refits under it.
- Produce regenerable evidence (a script flag, never a docstring assertion, per
  #272's history) for: whether the split improves the holdout, and whether the
  intended demotion of Rafaela/Otto happens without demoting genuinely skilled
  everyday players (Yordan Alvarez as the control).
- Ship a single winning parameterization -- do not carry two production configs.

## Non-goals

- **#278** (`sgp_sd` skill term for low-skill pitchers) -- separate issue. This
  spec re-fits `sgp_sd` only mechanically, because changing families moves the
  composite that `sgp_sd` is fit against; it does not add a skill term.
- **A new speed skill** (SB-rate percentile) to recover the SB signal. Noted as a
  fallback follow-up *if* the direct parameterization wins and speedsters visibly
  drop; not implemented here.
- The **`luck`x`future` collinearity** #277 flags -- explicitly "not a
  demonstrated bias" there; left alone.
- Regenerating the **stale published artifact** (a #273 leftover).
- Changing `scarcity` floors -- they live on the raw-SGP scale and are unaffected
  by the composite change.

## Chosen approach

Every candidate **adds a `pt` family** = within-pool percentile of PA (hitters) /
IP (pitchers), already carried on every row as `pt`. That is the "playing time is
real signal" half. Then two ways to handle what remains of `luck`, plus the
current model as a baseline:

### Baseline (current)
Families `(skill, luck, future, age)`. Reproduced unchanged so the bake-off has a
fair reference.

### Parameterization A -- keep `luck`, add `pt`
Families `(skill, pt, luck, future, age)`. Once `pt` is its own family, `luck` no
longer has to *proxy* playing time, so the grid search should **shrink `luck`'s
fitted weight** toward zero. That shrinkage is the direct test of #277's
hypothesis: if `pt` absorbs `luck`'s weight, the luck term was a confounded PT
stand-in. A keeps whatever SB/saves/role signal is real inside `luck`.

### Parameterization B -- replace `luck` with a direct batted-ball stat, add `pt`
Families `(skill, pt, batted_ball, future, age)`, where `batted_ball` is a
within-pool percentile of measured rate overperformance:
- **Hitters:** `AVG - xBA` (higher = luckier). `AVG` from `season_value`, `xBA`
  from the skills cache.
- **Pitchers:** `FIP - ERA` (higher = luckier; ERA below FIP = outperformed
  peripherals). `ERA` from `season_value`, `FIP` from the skills cache.

B targets exactly the regressing rate and nothing else. Its `batted_ball` fitted
weight directly answers #277's acceptance ("does the batted-ball component earn a
weight near zero or negative"). B's cost: it **drops** the SB/saves signal that
`luck` carries, because that value is neither a peripheral skill nor playing time.

Both `AVG`/`xBA` and `ERA`/`FIP` sit on a consistent park basis (none of the four
is park-adjusted in the skills layer), so the batted-ball measure carries no park
artifact.

### Why a bake-off rather than picking one
A and B occupy genuinely different model spaces: A can retain SB/role signal, B
cannot but is surgical about batted-ball luck. Ranking-equivalence note: for the
final re-ranked composite, `(skill, pt, luck)` and `(skill, pt, luck-residualized-
on-pt)` produce identical boards for mapped weights, so A does not need an explicit
residualization step -- the fitted `luck` weight is the readout. B is not
equivalent to A (its `batted_ball` excludes SB/saves), which is exactly why both
are worth running.

### Family machinery
Generalize the family machinery from a hardcoded 4-tuple to an explicit ordered
family set so the backtest can evaluate baseline/A/B without branching:
- A `KNOWN_FAMILIES` universe: `{skill, luck, pt, batted_ball, future, age}`.
- `composite()` and its weight tuples are driven by an explicit ordered
  `family_order`, validated against `KNOWN_FAMILIES`. Weight tuples become
  variable-length (`tuple[float, ...]`).
- **The family set is per-pool, like the weights.** `FAMILIES` becomes a
  `dict[pool, tuple[str, ...]]` (mirroring `FITTED_WEIGHTS`), so hitters and
  pitchers may ship different sets -- the pools already carry separate weights,
  fits, and residual quantiles, and B's batted-ball measure is pool-specific, so a
  split verdict (e.g. A for hitters, B for pitchers) is a legitimate and
  supported outcome, not an error to force into one global set.
- `_qualified_families` computes ALL candidate `*_pct` columns
  (`value_pct`, `skill_pct`, `luck_pct`, `pt_pct`, `batted_ball_pct`, `age_pct`);
  `future_pct` is still attached by the caller. The active pool's `family_order`
  selects the subset. Compute-all-select-active keeps the two call sites (ranking,
  backtest) from validating different feature definitions.

## Selection procedure

The bake-off runs in `--backtest` and reports, **per pool**, on the 2024 holdout:

- **Primary metric:** holdout Spearman rho vs next-year SGP percentile, for
  baseline, A, and B (each with its own grid-searched weights, `skill` pinned at
  1.0). Selection is per pool.

- **Grid ranges.** The tunable weights must span down through zero and slightly
  negative, or the hypothesis is unobservable: A's premise is `luck` shrinking
  toward zero and B's is `batted_ball` earning a near-zero-or-**negative** weight.
  Concretely, `luck` and `batted_ball` range over `{-0.4, -0.2, 0.0, 0.2, 0.4,
  0.6, 0.8, 1.0, 1.2}`, `pt` over `{0.0, 0.2, 0.4, 0.6, 0.8, 1.0, 1.2}`, `future`
  over `{0.0, 0.2, 0.4, 0.6, 0.8}`, `age` over `{0.0, 0.15, 0.3, 0.45}`. The
  baseline `luck` grid is extended to this same range (the shipped `0.4` floor
  would hide any shrinkage). Grids stay coarse deliberately -- two fit seasons
  cannot resolve a finer step (see the `composite` docstring).

- **Margin and tie-break (per pool).** rho differences at the third decimal are
  noise -- the module docstring states the top candidates separate by less than
  the two fit seasons separate. So A or B ships over baseline only if it beats
  baseline holdout rho by **more than the absolute rho spread between the two fit
  seasons for that pool** (the in-sample noise floor, computed and printed by the
  backtest). Within that band, treat candidates as tied and prefer, in order:
  (1) the candidate that passes the decision sniff test below, (2) `luck`-
  retaining A over batted-ball-only B (A preserves the SB/saves signal), (3)
  baseline (parsimony -- do not ship `pt` on intuition alone). A split verdict
  across pools is applied as-is via the per-pool family set. **The backtest
  computes and prints these numbers (per-candidate holdout rho, the fit-season
  noise floor, and the watchlist moves); the executor applies the margin,
  tie-break, and guard from that printed table and edits `FAMILIES`/
  `FITTED_WEIGHTS` by hand. No heuristic selection code ships.**
- **Mechanism checks (reported):** `pt_pct -> target_pt` and
  `batted_ball_pct -> target_rate` correlations (each transition already carries
  `target_pt` and `target_rate`), confirming `pt` predicts next-year PT and
  `batted_ball` does *not* predict next-year rate.
- **Shrinkage readout (reported):** `luck`'s fitted weight, baseline vs A.
- **Decision sniff test (reported + guard):** old->new hitter-pool rank moves for
  a watchlist on the live 2026 board -- Rafaela and Otto Lopez (should demote) and
  Alvarez (control, should not) -- plus mean realized next-year SGP of the
  winner's top-12 / top-20 on the holdout.

**Guard against a decision-nonsensical winner.** After the margin/tie-break picks
a per-pool winner, apply this guard on the hitter pool: compare each watchlist
player's composite rank under the winner vs under baseline. The guard **trips** if
either (a) Rafaela or Otto Lopez does not move to a strictly worse (higher) rank,
or (b) Alvarez moves worse by more than 3 ranks. When it trips, the selection
**stops and surfaces the full table** (rho, weights, watchlist moves) to the user
rather than shipping automatically -- a human decides. This preserves "let the
holdout decide" while guarding against the #276-style case where global rank
correlation is maximized but the top of the board is mis-ordered. The 3-rank
tolerance is a surface-to-human trigger, not a correctness threshold, so its exact
value is not load-bearing.

If baseline wins outright (neither A nor B beats it on holdout rho), that is a
valid #277 outcome -- "the single `luck` term is diluted rather than wrong." In
that case the model is left unchanged and the finding is reported; `pt` is not
shipped on intuition alone.

## Downstream refit

`projection.py`'s constants (`SGP_FIT`, `SGP_SD_FIT`, `STD_RESIDUAL_QUANTILES`)
are fit against the final re-ranked `composite_pct`. Changing the winning family
set moves that composite, so after selection the pipeline re-runs `--fit` to
regenerate all of them, paste-ready. `test_projection.py`'s monotonicity and
skew/tail assertions stay as guardrails: if regenerated constants violate them,
that is a real signal to investigate, not a test to loosen.

## Requirements

R1. A `pt` family exists: within-pool percentile of PA (hitters) / IP (pitchers),
    higher = better, NaN-safe.
R2. A `batted_ball` family exists: within-pool percentile of `AVG - xBA` (hitters)
    / `FIP - ERA` (pitchers), higher = luckier, NaN-safe.
R3. `season_value` emits the raw rate stat each pool's batted-ball measure needs
    (`avg` for hitters, `era` for pitchers) alongside `sgp`/`age`/`pt`.
R4. Family machinery is driven by an explicit ordered family set validated against
    a `KNOWN_FAMILIES` universe; weight tuples are variable-length. Default
    behavior (shipped `FAMILIES`/`FITTED_WEIGHTS`) is unchanged until selection.
R5. `--backtest` grid-searches weights for baseline, A, and B per pool and reports
    holdout rho for each, with `skill` pinned at 1.0.
R6. `--backtest` (or `--study`) reports the mechanism checks, shrinkage readout,
    and decision sniff test named in Selection procedure, all regenerable from the
    flag.
R7. The winning parameterization is written to `FAMILIES`/`FITTED_WEIGHTS`; the
    sniff-test guard is honored (surface, do not auto-ship, on failure).
R8. `projection.py` constants are regenerated via `--fit` against the winning
    composite; `test_projection.py` guardrails remain and pass.
R9. Display schema (`SHOWN`) and `composite.py` docstrings are updated to the
    winning family set. Grep-complete rename: every `luck`/`FAMILIES`/
    `FITTED_WEIGHTS` reference, type annotation, string literal, and test.
R10. If B wins (luck dropped), the SB/saves tradeoff and the speed-skill follow-up
     are documented in the module docstring and the handoff.
R11. Downstream consumers of the ranking output are enumerated and kept working:
     the per-pool CSV written under `SKILLS_DIR` (`keeper_rankings_{kind}_{year}
     .csv`), the `--league`/`--roster` stdout board, and any reader of those
     columns elsewhere in the repo. If the winning set drops a column a consumer
     reads (e.g. `luck_pct` under B), that consumer is updated in the same change;
     a grep for the column names must come back clean or handled.
R12. A written verdict is produced in EVERY outcome (not only if B wins): which
     family set shipped per pool, the fitted weights, the holdout rho table, and a
     pointer to the `--backtest` flag that regenerates the evidence. This is the
     #277 acceptance ("state whether the shipped weights change, and if they do
     not, explain why").

## Edge cases and failure modes

- **NaN inputs.** A hitter with no batted-ball event has NaN `xBA`; a sub-floor
  pitcher is filtered by `MIN_PT` before families are built. `percentile` keeps
  NaN as NaN and `composite` fills a missing family with its mean -- preserve that
  contract for `pt` and `batted_ball`.
- **Numeric-default trap.** `pt = 0` and `AVG - xBA = 0` are meaningful values,
  not "missing"; never sink them with `x or default` (CLAUDE.md).
- **Park basis.** Confirm `AVG`, `xBA`, `ERA`, `FIP` are all park-unadjusted at the
  point the batted-ball measure is computed, so it carries no park artifact.
- **Baseline reproduction.** Baseline rho must match the pre-change `--backtest`
  numbers for the shipped weights; a mismatch means the generalization changed
  behavior and must be fixed before trusting A/B.
- **Pitcher ERA basis.** The `ERA` used in `FIP - ERA` must come from the same
  BBRef pull as `FIP` (both BBRef-derived), so the two sit on a consistent basis;
  do not mix a Savant-derived ERA against a BBRef-derived FIP.
- **Rank-equivalence sanity.** A board built from `(skill, pt, luck)` and one from
  the same with `luck` residualized on `pt` (mapped weights) must match -- a check
  that the re-rank invariance holds.
- **`sgp_sd` shift is mechanical, not #278.** Do not add a skill term to `sgp_sd`
  here; only regenerate it against the new composite.

## Testing expectations

- Unit tests for the two new pure functions (`pt` percentile, `batted_ball`
  overperformance) covering the sign convention and NaN handling.
- A test that `composite()` honors an explicit `family_order` and rejects families
  outside `KNOWN_FAMILIES`.
- `test_projection.py` monotonicity + skew/tail assertions retained; regenerated
  constants pinned to the new fitted values.
- Existing `test_composite.py` / `test_scripts/test_keeper_rankings.py` updated
  deliberately for the family-set change (documented justification, per CLAUDE.md
  -- the requirement changed, the assertions did not silently loosen).
- **No-behavior-change gate for Phases 1-2.** Before touching production
  `FAMILIES`/`FITTED_WEIGHTS`, prove the generalization is inert. Phases 1-2 may
  ADD informational columns (`pt_pct`, `batted_ball_pct`, `avg`/`era`), so the gate
  is value-identity on the load-bearing outputs, not byte-identity of the whole
  CSV: capture a `build` result for one pool/year on the pre-change branch, run the
  same after the refactor, and require identical `rank`, `composite`, `proj_sgp`,
  `sd`, and `proj_var` for every player (added columns are fine). Also require the
  baseline candidate's `--backtest` holdout rho to equal the pre-change
  shipped-weights rho. Ship nothing to production until both hold.
- End-of-effort: `pytest -q -n auto`, `ruff check .`, `ruff format --check .`,
  `mypy` (composite/projection/keepers are under mypy coverage -- check the list),
  `vulture` clean of new findings.

## Prerequisites and risks

- **Skills cache.** `--backtest`/`--fit` read `data/cache/keeper_skills/` which is
  not committed and not present locally. Build it first with
  `python scripts/fetch_keeper_skills.py --year <Y>` for **2022-2025** (backtest
  transitions need year and year+1) and **2026** (live board). The Statcast pull
  is ~1 min/year and cached thereafter. If the fetch is blocked (Savant/BBRef
  unreachable), the bake-off cannot run against real data -- surface to the user
  rather than fabricating results.
- **Data provenance.** Validate against the branch's committed 2022-2026 inputs
  and the 2027/2028 out-year ZiPS the branch tracks; do not silently pull newer
  local scratch copies of the projection files.

## Phasing

- **Phase 0 -- prerequisites.** Build the skills cache for 2022-2026; abort/surface
  if the fetch is blocked.
- **Phase 1 -- family machinery (no behavior change).** Generalize `composite.py`
  to an explicit `family_order` + `KNOWN_FAMILIES`; add the `pt` and `batted_ball`
  pure functions; extend `season_value` (R3). Unit tests. Existing suite green,
  shipped board value-identical on the load-bearing columns (per the gate above).
- **Phase 2 -- bake-off tooling (no behavior change).** Compute the candidate
  columns in `_qualified_families`; generalize the `--backtest` grid to
  baseline/A/B; add the mechanism/shrinkage/sniff-test readouts. Run the bake-off.
- **Phase 3 -- select and ship.** Apply the guard; set `FAMILIES`/`FITTED_WEIGHTS`
  to the winner; regenerate `projection.py` via `--fit`; update `SHOWN`,
  docstrings, tests. Full verification. Sanity-check the live 2026 board and the
  watchlist.
- **Phase 4 -- document tradeoff (if B wins).** Record the SB/saves loss and the
  speed-skill follow-up.
