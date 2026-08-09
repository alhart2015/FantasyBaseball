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
| Persistence fit leakage | Leave-one-transition-out, declared as a third advantage keeper-value keeps | A strictly causal fit leaves zero transitions for base 2022 and one for base 2023, deleting the +2 horizon and the multi-year claim with it. Base 2024 gets a causal sensitivity run so the size of the advantage is measured, not argued. |
| In-progress 2026 as an outcome year | Not admissible | `prorate_partial` is straight-line and assumes health, so pacing an outcome season scales an injured player up as if healthy -- the confound the injury view exists to remove. Costs draft 2024 its multi-year target; the triple slice is reported at both horizons instead. |

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
| `TRANSITIONS = ((2022,2023),(2023,2024),(2024,2025))` | `load_shares` takes the transition list; historical runs use **leave-one-transition-out** -- see the leakage note below, this is weaker than it sounds |
| playing-time panel spans 2010-2026 | censored to `season <= Y` before `lag_panel` / `fit_curve` |

Each base year needs **two** `forecast_pool` runs per pool, for target years Y+1 and
Y+2, exactly as the live tool is invoked separately for 2027 and 2028; the Y+2 run
iterates the playing-time curve twice. Shape is called once with `horizons=(1, 2)`. The
multi-year target is the sum of the two.

Shape gets the symmetric treatment the harness already applies: query player removed
from the panel entirely (no self-matching), panel truncated to `season <= Y`.

### When the playing-time curve cannot fit

`volume_forecast` already degrades rather than failing: no panel falls back to the
one-year gap model with a printed WARNING, and a player the curve cannot score falls
back per-player to `folded`. Censoring the panel to `season <= Y` makes both more likely
for early base years.

A gap-model fallback **still counts as keeper-value** -- it is what the engine does when
the data is thin, and substituting something better would be scoring an engine that does
not exist. But it is counted and reported per base year and pool. If more than 25% of a
base year's pool falls back, that base year is reported separately and excluded from the
headline number rather than folded in, because at that point the thing being measured is
mostly the gap model.

### Three advantages keeper-value keeps

Stated, not removed, because removing them is impossible and pretending they are absent
would be worse:

1. **Out-year vintage leakage.** `centered_aging` wants the year-(Y+1) projection from
   the *preseason-Y* ZiPS run. Only the preseason-(Y+1) run exists on disk, and it
   already saw season Y.
2. **It reads ZiPS at all.** Shape reads only realized, era-normalized seasons.
3. **Its persistence share is fit partly on seasons after the one it predicts.**
   `TRANSITIONS` is bounded by actuals coverage at (2022->23, 2023->24, 2024->25).
   Leave-one-transition-out guarantees the fit never sees the transition it is
   predicting, and nothing more: for base year 2022 the two remaining transitions are
   both *later* than the one being predicted. A strictly causal rule (transitions ending
   `<= Y`) leaves **zero** transitions for base 2022 and **one** for base 2023, which
   would delete the +2 horizon entirely and with it the multi-year claim this whole
   evaluation exists to make.

   So LOTO is kept and the leakage is declared rather than hidden. The number of
   future transitions used is printed per base year (2 for 2022, 1 for 2023, 0 for
   2024). **Base 2024 is additionally run strictly causally**, on the two transitions
   that precede it, as a sensitivity check -- it is the one base year where a causal fit
   and a LOTO fit can both be computed, so it is the only place the size of this
   advantage can be measured rather than argued about.

All three flatter keeper-value. If shape wins anyway, that is the strong form of the
result.

### Information set is otherwise symmetric

Both estimators see full realized seasons through year Y and nothing after. The live
chain's 2/3-season blend is replaced by full actuals for year Y, which matches what
shape's anchors are -- so this is the symmetric choice, not a concession.

### Target

Realized SGP summed over Y+1 and Y+2, and the same in VAR against `trajectory.value`
floors. A season not played scores as the 0 it is worth to a roster slot, per `played()`.

**2026 is not admissible as an outcome year.** It is in progress, and the only tool for
comparing it against full seasons -- `panel.prorate_partial` -- is straight-line and
explicitly assumes the player stays healthy. Pacing an outcome season would scale an
injured player's line up as if he had not been hurt, which is the exact confound the
injury-excluded view exists to remove. So outcome years stop at 2025.

Available base years, bounded by actuals (2022-2025), ZiPS vintages (2022-2026) and that
rule:

- **+1 horizon:** base 2022, 2023, 2024 (three cohorts)
- **+2 horizon:** base 2022, 2023 (**two** cohorts)

Two transition cohorts cannot support a fine distinction. That is teardown constraint 4
and it is expected to bind here; see Acceptance.

**Position eligibility for VAR comes from year Y**, for both estimators and both
horizons, via `keepers.appearances.season_eligibility` over
`data/cache/keeper_skills/mlb_fielding_{Y}.csv` (cached 2022-2026). Year Y is the
information set the keeper decision actually has; outcome-year eligibility would be
hindsight, and the catcher-to-outfield floor spread is 2.3 SGP a year -- larger than the
margins this backtest is trying to resolve. `board.py`'s existing fallback applies: a
missing or corrupt cache degrades a hitter to the UTIL floor, the highest one, so the
failure mode understates a player rather than inventing value.

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

1. **Keeper-triple regret** -- the actual decision. `data/historical_drafts_resolved.json`
   holds draft years 2023, 2024 and 2025, each with 10 teams of exactly 23 players. For
   each team, each estimator picks the 3 with the highest **forecast** VAR over the
   target horizon (the league keeps 3). Score the **realized** VAR of each triple against
   the ex-post best triple from the same roster; the shortfall is the regret.

   Because 2026 is not an admissible outcome year, this slice is reported at **both**
   horizons with honest counts rather than one blended number:

   | target | usable draft years | decisions |
   |---|---|---|
   | multi-year (Y+1 and Y+2) | 2023 | **10** |
   | one-year (Y+1) | 2023, 2024 | **20** |

   Draft 2025 is unusable at either horizon -- its outcome year is 2026. The multi-year
   table is the one that matches the keeper question; the one-year table is the one with
   twice the decisions. Neither is dropped and neither is presented as the other.
2. **Top-of-board** -- realized multi-year VAR of each estimator's top 30, taken from the
   **concatenated both-pools** intersection set ranked on forecast multi-year VAR (3
   keepers x 10 teams = 30 players kept league-wide). Per-pool top-15 tables are reported
   alongside, because hitters and pitchers net against different floors and a pooled
   ranking can hide a pool-specific failure. Plus per-player error on the union of the
   two top-30s.
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

Bootstrap on every headline number, 10,000 draws, reporting the 2.5-97.5 interval on the
**difference** between estimators and the fraction of draws shape wins.

The resampling unit differs by slice and getting it wrong makes the interval meaningless:

- **Keeper-triple regret:** resample **team-decisions** (a cluster bootstrap over the 10
  or 20 team-years). Resampling players inside a roster would change the roster, which
  changes the ex-post optimum and leaves regret undefined.
- **Top-of-board and breakout:** resample **players**.

With 10 to 20 team-decisions this is expected to say "cannot separate".

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
   `scripts/backtest_trajectory.py`. **`roles()` and `tests/test_trajectory/test_backtest_roles.py`
   survive this PR untouched** -- the pitcher role bucketing came from #313, not from the
   comp matchers, and it dies with the file in PR 3.
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
citation, so the code that produced the number is findable in git. `<sha>` is not an
unfilled placeholder: it is resolved at PR-3 time to the last commit that still
contained `scripts/backtest_trajectory.py`, which is only knowable once PR 3 is written.

## Acceptance

**PR 1 -- output**

- Historical head-to-head runs for base years 2022-2024 (+1) and 2022-2023 (+2), both
  pools, both views.
- Keeper-triple regret at both horizons (10 multi-year decisions, 20 one-year), plus
  top-of-board and breakout, each with the bootstrap interval on the difference and the
  resampling unit named.
- Censored-player list printed with volumes, split zero vs non-zero, at 50% and 20%.
- Per base year and pool: gap-model fallback counts, future-transition count used by the
  persistence fit, and the strictly-causal sensitivity run for base 2024.
- Verdict posted as a comment on #325 and in the PR body.

**PR 1 -- verification.** PR 1 introduces the most error-prone code in this design and
is not exempt from the repo's guardrail rule. Required tests:

- **Characterization:** `era_normalize`'s output is byte-identical before and after the
  `era_factors()` extraction, on a fixture panel. A behaviour-preserving extraction with
  no characterization test is the case `CLAUDE.md` names by hand.
- **Censor boundaries:** a query at exactly 50% of anchor volume, at zero volume, and
  with no outcome row at all -- each asserted to land on the intended side of the cut,
  and asserted to remove the identical row from both estimators.
- **Historical mode is actually historical:** the fitted panel contains no season > Y,
  and the query player is absent from it.
- **Regret:** computed against a hand-built 5-player roster with known realized values,
  including the case where an estimator's triple *is* the ex-post optimum (regret 0).
- **Multi-year censoring:** a player censored in only one of two outcome years is absent
  from the multi-year metric entirely.
- `pytest -v`, `ruff check .`, `ruff format --check .`, `vulture`, and `mypy` where
  applicable -- the same gate PR 2 and PR 3 carry.

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
