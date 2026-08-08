# Retire the keeper-value engine and the non-shape trajectory modes

Design for #325. Shape becomes the single trajectory engine; the keeper board is priced
off it instead of off the `keeper_forecast` -> `keeper_value` chain.

Source issue: #325. Read `docs/keeper-value-teardown-2026-08-01.md` first -- the
constraints it names are the reason this is sequenced the way it is.

## Status of the gate

Phase 0 of #325 is a hard gate: nothing is deleted until @alhart2015 has used the
trajectory UI (#321-#324) against real keeper decisions and says go. **That gate is
passed** -- this design was commissioned by him on 2026-08-08 with the instruction to
keep shape, retire the other trajectory methods, and verify with a backtest.

The one blocker inside the issue that was still open is also cleared: **#313 closed
2026-08-03** (PR #326) with shape validated on pitchers. Its verdict, which this work
builds on rather than re-derives:

- Shape beats level matching on every well-powered slice in both pools.
- On pitchers it wins by fixing calibration (bias -1.69 -> +0.17), not by
  discriminating (53% win rate on the big-drop slice -- a coin flip).
- Shape also beats `track`, and on pitchers `track` is *worse* than plain level
  matching. Shape adapts the weight on the prior season to the pool; track assumes it.

## What is actually being retired

Two chains, and a third thing that falls out of the second.

**The keeper-value scoring chain.** `scripts/keeper_forecast.py`,
`scripts/keeper_value.py`, `scripts/keeper_persistence.py`, and the library modules
they own: `keepers/persistence.py`, `keepers/playing_time.py`, `keepers/blend.py`.

**The non-shape trajectory matchers.** `trajectory/comps.py` and everything that
selects them: `--match current` / `--match track`, `--band`, `--prior-band`.

**The bake-off harness.** `scripts/backtest_trajectory.py` exists to race the matchers
against each other. Once there is one matcher and one keeper engine it has no
contender, and per the decision recorded below it is deleted rather than reduced to a
shape-only calibration tool.

`src/fantasy_baseball/keepers/` **stays as ingest and normalization**, as the teardown
left it. FanGraphs is Cloudflare-403 blocked league-wide, so MLB Stats + Savant + BBRef
is the working path and rebuilding it would be pure re-work.

## Why the replacement already exists

2a in #325 reads as "replace the scoring chain with a shape-priced board", which sounds
like a build. It is not. The shape-priced board already ships:

- `scripts/trajectory_board.py` ranks the whole pool on total VAR over the horizon,
  against the same position-aware waiver floors `keeper_value` nets against.
- `scripts/push_trajectory_board.py` writes `cache:trajectory_board`, and the web
  `/trajectory` route renders it (#321-#324).
- The keeper-value chain has **no consumers outside its own CLI scripts** -- no web
  route, no library import, no cache key.

So 2a is a deletion with a verification attached, not a migration.

## Decisions taken during design

| decision | choice | why |
|---|---|---|
| Backtest depth | Full historical head-to-head | Only thing that answers "is what we are deleting actually worse". Mechanism-only and present-day-disagreement alternatives were rejected. |
| Work split | Three PRs: evidence, then 2b, then 2a | The PR that deletes something should not also be the only place the evidence for deleting it lives. |
| Fate of the harness | Delete it in PR 3 | Per @alhart2015: "Keeping record of previous worse attempts is useless. If we really need the history we have the PR history in git." |
| Where the evidence lives | Issue comment on #325 + PR bodies | Same as how #313's verdict was recorded. No new `docs/` file. |
| Injury view | Second view censoring outcome years under 50% of anchor volume | Per @alhart2015: injury is close to random, and penalising an otherwise-correct keeper decision for it confounds the comparison. |
| Zero-volume outcomes | Censored too, in the injury-excluded view | Explicit call after being shown that volume alone cannot separate a season-ending April injury from a washout. The ALL view retains them, so nothing is lost. |

## Phase 1 -- the backtest (PR 1)

Teardown constraints 1 and 2 exist because the last two keeper-value attempts were
selected on pool-wide rank correlation, which "rewards being right about players nobody
would keep". This evaluation is on the decision.

### One scale for both estimators

`keeper_forecast.forecast_pool` emits the canonical rate/PT schema (`PA`, `ab_pa`,
`h_ab`, `r_pa`, ... / `IP`, `k_ip`, `er_ip`, ...). `trajectory.panel.score()` consumes
exactly that schema and produces SGP. So a keeper-value forecast, a shape forecast and a
realized season can all be scored by **one scorer**. That is what makes the numbers
comparable rather than merely same-unit.

The trap is era normalization. Shape lives entirely in the 2023-2025 reference run
environment (`trajectory/era.py`); keeper-value's inputs are raw. Both must be in one
environment or the comparison silently charges keeper-value for league-wide run-scoring
drift that shape was never exposed to.

Fix: normalize keeper-value's **inputs** by the season each belongs to -- actuals for
year Y by factor(Y), the ZiPS vintage projecting season Z by factor(Z) -- using the same
factor table `era.era_normalize` derives. Every season involved is historical, so every
factor exists. This requires extracting the factor computation out of `era_normalize`
into a reusable `era_factors()` so the two callers cannot disagree about what a factor
is.

### Making the keeper chain runnable out of sample

Four hardwirings, all in `scripts/keeper_forecast.py`:

| hardwiring | change |
|---|---|
| `BASE_YEAR = 2026` module constant | parameter threaded through `forecast_pool` and `volume_forecast` |
| `fetch_blend()` reads live Upstash | `forecast_pool` takes an `observed` rate frame; historical runs pass `keeper_persistence.load_rates(Y, kind, source="actual")` |
| `TRANSITIONS = ((2022,2023),(2023,2024),(2024,2025))` | `load_shares` takes the transition list; historical runs use **leave-one-transition-out** so the persistence fit never sees the transition it is predicting |
| playing-time panel spans 2010-2026 | censored to `season <= Y` before `lag_panel` / `fit_curve` |

Shape gets the symmetric treatment the harness already applies: query player removed
from the panel entirely (no self-matching), panel truncated to `season <= Y`.

### Two advantages keeper-value keeps

Stated, not removed, because removing them is impossible and pretending they are absent
would be worse:

1. **Out-year vintage leakage.** `centered_aging` wants the year-(Y+1) projection from
   the *preseason-Y* ZiPS run. Only the preseason-(Y+1) run exists on disk, and it
   already saw season Y.
2. **It reads ZiPS at all.** Shape reads only realized, era-normalized seasons.

Both flatter keeper-value. If shape wins anyway, that is the strong form of the result.

### Information set is otherwise symmetric

Both estimators see full realized seasons through year Y and nothing after. The live
chain's 2/3-season blend is replaced by full actuals for year Y, which matches what
shape's anchors are -- so this is the symmetric choice, not a concession.

### Target

Realized SGP summed over Y+1 and Y+2, and the same in VAR against `trajectory.value`
floors. A season not played scores as the 0 it is worth to a roster slot, per `played()`.

Available base years, bounded by actuals (2022-2025) and ZiPS vintages (2022-2026):

- **+1 horizon:** base 2022, 2023, 2024 (three cohorts)
- **+2 horizon:** base 2022, 2023 (**two** cohorts)

Two transition cohorts cannot support a fine distinction. That is teardown constraint 4
and it is expected to bind here; see Acceptance.

### Two views

Every slice below is reported twice.

**ALL** -- every scored query.

**INJURY-EXCLUDED** -- drops any query whose outcome-year volume (PA for hitters, IP for
pitchers) is under 50% of his **year-Y** volume. Rationale: injury is close to random,
and charging an otherwise-correct keeper decision for it confounds the comparison.

Rules that follow from that:

- The ratio is measured against year **Y** for both outcome years, never against Y+1. A
  wrecked Y+1 must not redefine "normal" for Y+2.
- Zero-volume outcome years are censored too, by explicit decision.
- Multi-year metric: a player censored in **either** outcome year leaves it entirely. A
  one-year sum and a two-year sum are not the same target.
- Keeper-triple slice: a censored player leaves both the candidate pool and the ex-post
  optimum -- the view asks "among players who were not wrecked, did the estimator pick
  right".
- Censoring is a property of the realized outcome, so both estimators lose identical
  rows. It cannot favour either.
- Every censored player is printed with name, year-Y volume, outcome volume, and is
  counted separately for zero vs non-zero volume. The same counts are also reported at a
  20% threshold, so the sensitivity of the cut is visible without a second run.

### Slices, in decreasing order of trust

1. **Keeper-triple regret** -- the actual decision. For each of the 10 teams in
   `data/historical_drafts_resolved.json` for draft years 2023 and 2024, each estimator
   picks the 3 of that team's 23 drafted players with the highest **forecast** multi-year
   VAR (the league keeps 3). Score the **realized** multi-year VAR of each triple against
   the ex-post best triple from the same roster; the shortfall is the regret. 20 real
   decisions rather than 2, because every team's roster is a genuine 23-player keeper
   pool.
2. **Top-of-board** -- realized multi-year VAR of each estimator's top 30 (3 keepers x
   10 teams), plus per-player error on the union of the two top-30s.
3. **Breakout** -- `now > 1.25 * prior`. This is where 2a's open question is answered:
   `persistence.S` regresses a breakout against ZiPS; shape regresses it against how
   comparable shapes actually played out. If shape is not worse here, the
   projection-gap mechanism has an equivalent and `persistence.py` can go.

A drafted roster for year Y approximates the roster from which the Y -> Y+1 keeper
decision is made; in-season adds and drops are not modelled. Stated in the writeup.

### Coverage

The two estimators do not cover the same players. `keeper_value` needs a year-Y volume
above its floor (300 PA / 50 IP), presence in the year-Y ZiPS vintage, and a
playing-time panel row; shape needs two anchor seasons in the panel. A slice scored on
each estimator's own covered set would compare two different populations.

So every slice runs on the **intersection** -- the players both can score -- and the
coverage gap is reported rather than absorbed: how many players each estimator alone
could score, and for the keeper-triple slice, how many of each 23-man roster survived
into the candidate pool. A roster left with fewer than 5 candidates is dropped from that
slice and named, because picking 3 from 4 is not the decision being measured.

### Noise floor

Bootstrap over query players on every headline number, reporting the interval on the
**difference** between estimators and the fraction of draws shape wins. With 20
team-decisions this is expected to say "cannot separate".

## Phase 2b -- non-shape trajectory modes (PR 2)

Three commits, in order. Commit 1 must land before anything is removed: `shape.py:67`
imports from `comps.py`, so deleting `comps.py` first breaks shape.

1. **`trajectory/model.py`, no behaviour change.** Move `PathPoint`, `Trajectory`,
   `collapse_split_seasons`, `played`, `DEFAULT_HORIZONS`, `MIN_LOCAL_SUPPORT` out of
   `comps.py`. Repoint `shape.py`, `web/trajectory_view.py`, `scripts/trajectory_board.py`,
   `scripts/push_trajectory_board.py`, `scripts/player_trajectory.py`,
   `tests/test_trajectory/test_board.py`. `DEFAULT_BAND` is comps-only and dies with it.
2. **Delete the matchers.** `trajectory/comps.py`; `comp_trajectory` from
   `trajectory/__init__.py`; `--match`, `--band`, `--prior-band` and the `parser.error`
   branches policing their combinations in `scripts/player_trajectory.py`;
   `tests/test_trajectory/test_comps.py`; the `current` and `track` contenders in
   `scripts/backtest_trajectory.py`.
3. **Convert, do not drop, `tests/test_trajectory/test_mode_parity.py`.** Its docstring
   records that three review rounds each found shape re-implementing something comps had
   already decided and silently disagreeing. Every assertion is triaged into either a
   shape-only invariant moved to `test_shape.py` -- negative season counts as played,
   split seasons collapse, empty-cohort behaviour, `mode` string -- or an explicit drop
   with the reason stated in the PR body. The file goes; the guarantees that still mean
   something do not.

`trajectory/comp_paths.py` is untouched. The player tab's comps (#324) come from MSE
against `Prepared.forward`, not from `comp_trajectory`.

## Phase 2a -- keeper value (PR 3)

Deletes:

- `scripts/keeper_forecast.py`, `scripts/keeper_value.py`, `scripts/keeper_persistence.py`
- `keepers/persistence.py`, `keepers/playing_time.py`, `keepers/blend.py`
- `tests/test_keepers/{test_persistence,test_playing_time,test_blend}.py`
- `scripts/backtest_trajectory.py` and `tests/test_trajectory/test_backtest_roles.py` --
  its last contender goes here, so it goes here too and each PR leaves a coherent tree

Two consequences the issue did not anticipate, both found by tracing consumers:

- **`keepers/vintages.py` becomes dead.** Its only importers are `blend.py`,
  `keeper_forecast.py` and `keeper_persistence.py`, all deleted. #325 lists it as ingest
  to keep, but that assumed a consumer. It is deleted; ZiPS ingest for the draft
  pipeline goes through `data/fangraphs`, not this module. `tests/test_keepers/test_vintages.py`
  goes with it.
- **`scripts/build_pt_panel.py` still defaults to `data/playing_time/`** and carries two
  guards written to stop a rebuild clobbering the keeper panel, both naming
  `keeper_forecast._panel_path`. With that consumer gone the trajectory panel is the only
  one left: the default `--out-dir` moves to `data/trajectory`, and the keeper-dir guard
  and `--allow-keeper-dir` flag are removed. Leaving a guard that names a deleted
  function is worse than the diff.

`keepers/` ingest otherwise stays untouched and all of it keeps a live consumer:
`cache`, `mlb_stats`, `savant`, `bref`, `appearances` (read by `trajectory/value.py` and
`trajectory/board.py` for position eligibility), `actuals` (read by `keepers/skills.py`,
`pt_model/panel.py`, `scripts/fetch_keeper_skills.py`), `skills`, `positions`.

## Docstrings

`trajectory/__init__.py` claims "**Hitters only.** The pitcher pool has never been
validated and shape is its default too -- #313." That has been stale since 2026-08-03
and is corrected in PR 2.

Every numeric claim whose harness is being deleted -- the tables in
`trajectory/__init__.py` and `shape.py` -- gets a `measured 2026-08-08 at <sha>`
citation naming the commit that still contained `backtest_trajectory.py`, so the code
that produced the number is findable in git.

## Acceptance

**PR 1**

- Historical head-to-head runs for base years 2022-2024 (+1) and 2022-2023 (+2), both
  pools, both views.
- Keeper-triple regret, top-of-board and breakout slices reported with bootstrap
  intervals on the difference.
- Censored-player list printed with volumes, split zero vs non-zero, at 50% and 20%.
- Verdict posted as a comment on #325 and in the PR body.

**A verdict of "cannot separate" is an acceptable outcome and does not block PR 2 or
PR 3.** Teardown constraint 4 is explicit that when the evaluation cannot separate two
candidates you say so and pick on other grounds. The other grounds are already on the
record and are not weakened by a null result:

- `keeper_value`'s own docstring: the persistence fit is a one-year transition applied
  twice, so 2028's rates are effectively 2027's and a 2026 breakout is over-credited
  against a steady veteran.
- Teardown constraint 3: indefinite, no-cost retention makes a multi-year horizon
  first-order. Shape is multi-year by construction and carries per-horizon survival and
  empirical p10..p90 at every year.
- Teardown constraint 6: shape reads only realized seasons, so no part of it is blind to
  the current season the way an out-year ZiPS baseline is.

The only result that **blocks** deletion is keeper-value beating shape on the
keeper-triple or top-of-board slice by a margin the bootstrap separates from zero, in
either view. That is stated up front so a null cannot be read after the fact as support.

**PR 2 and PR 3**

- `pytest -v`, `ruff check .`, `ruff format --check .`, `vulture`, and `mypy` for any
  touched file listed under `[tool.mypy].files`, all clean.
- No reference anywhere -- code, tests, docs, config -- to a deleted symbol. Grep is not
  an AST: check direct calls, annotations, string literals in dispatch dicts and config,
  `getattr`/`importlib`, `__all__` re-exports, tests and fixtures, and docs.

## Explicitly not in scope

- **Do not re-derive playing-time memory.** Teardown constraint 8, measured twice on
  #288: the prior year's PT gets a coefficient of +0.03 (hitters) / +0.08 (pitchers) and
  a predicted-next-year-PT family correlates with current PT alone at 0.999.
- **Do not resurrect the rejected variants.** PR #293 already rejected the reliability
  form and the two-stage product.
- **Closers stay unresolved.** #313 found opposite signs on two 27-row samples and
  deliberately did not fork the default. #306 is the real fix.
- No changes to the trajectory UI (#321-#324) or to `comp_paths.py`.
