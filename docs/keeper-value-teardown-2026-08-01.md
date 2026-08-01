# Keeper value: teardown and reset (2026-08-01)

Three successive attempts to quantify keeper value were removed on this date. This
document records what was built, why each one was abandoned, and the constraints any
replacement has to satisfy. It exists so the fourth attempt does not rediscover the
same failure mode.

The code is all in git history. Do not restore it wholesale; read this first.

## The decision the feature is supposed to inform

League rules: **3 keepers per team, kept indefinitely, no cost.** No escalating price,
no round forfeit, no salary. So the question is:

> Is this player worth one of my 3 permanent retention slots, versus the best
> alternative use of that slot?

That is a **top-3 selection problem over an indefinite horizon**, scored against an
opportunity cost. Every attempt below optimized something else.

## What was built and removed

| Attempt | Code | Removed in |
|---|---|---|
| 1. Skill/luck breakout classifier + HR confirmation | `analysis/breakout.py`, `hr_confirm.py`, `breakout_backtest.py`, `data/skill_luck.py` | 99fb995c (#265) |
| 2. The keeper-value metric and the #266 out-year fold | `analysis/keeper_value.py`, `analysis/keeper_trades.py`, `keepers/{fold,coefficients,calibration}.py` | 2b6bf812 (#266) |
| 3. The percentile family composite | `keepers/{composite,projection,scarcity}.py`, `scripts/keeper_rankings.py` | this change |

Attempt 3's final shipped state (`main`, five families per pool):

```
FAMILIES       = {hitter:  (skill, luck, batted_ball, future, age),
                  pitcher: (skill, luck, batted_ball, future, age)}
FITTED_WEIGHTS = {hitter:  (1.0, 0.8, -0.2, 0.2, 0.30),
                  pitcher: (1.0, 0.8, -0.2, 0.2, 0.15)}
```

Its abandoned successor (PR #289, closed) replaced the `luck` residual with named
families and reached `(skill 1.0, speed 0.4, pt 2.2, batted_ball -0.2, future 0.4,
age 0.45)` for hitters.

## Why it was abandoned

### 1. The objective was a proxy, and the proxy was nearly a tautology

Every version fit **rank correlation against next-season realized SGP percentile**,
pooled across the whole player population. Two problems compound:

- SGP value is **counting-stat loaded**, so it is dominated by accumulation. The model
  therefore kept discovering that playing time predicts playing-time-loaded value.
  That is close to a tautology, and it is why the heaviest term kept growing: `luck`
  (a PT-loaded residual) at 0.8, then an explicit `pt` at **2.2** -- a fitted weight
  saying last season's observed plate appearances outrank every measure of talent.
- It grades ~400 players equally when only the ~30 at the top of the board, plus my
  own roster, are ever a real keeper decision.

### 2. The horizon was one year; the asset is held forever

A no-cost indefinite keeper is a multi-year asset. A one-season-ahead rank correlation
prices none of that, which is exactly why age could only ever enter as a small
afterthought adjustment (0.30-0.45) instead of as a driver.

### 3. The evaluation could not separate the candidates it was used to choose between

From the #277 bake-off (`docs/superpowers/keeper-277-verdict-2026-07-30.md`), holdout
2024, fit 2022-23:

| candidate | hitter | pitcher |
|---|---|---|
| baseline | 0.7085 | 0.4962 |
| A: pt + luck | 0.6878 | 0.4905 |
| B: pt + batted_ball | 0.6812 | 0.4932 |
| C: luck + batted_ball (shipped) | 0.7002 | 0.5094 |

Noise floor: hitters **0.0172**, pitchers **0.14**. Every pitcher gap is a statistical
tie, and most hitter gaps are marginal. Yet this table selected a family set and
weights to two decimals, and produced a written verdict. Eleven free weights were
being fit on **two** transition seasons.

### 4. It ended up defending a measured regression

The final state is self-documenting. From the deleted `composite.py` docstring:

> **This family set does NOT beat the residual it replaces.** Shipped holdout is
> 0.6935 (hitters) and 0.5006 (pitchers) against the pre-#288 model's 0.7002 and
> 0.5094 -- inside the fit-season noise band in both pools, but consistently a little
> below, not a tie in its favour. It ships on interpretability.

A 150-line module docstring arguing that a slightly worse model should ship is the
clearest signal available that the work had stopped being about the decision.

### 5. The baseline could not see the current season

`data/projections/{2027,2028}/zips-*.csv` are generated **pre-2026-season by design**
(ZiPS only updates the current year in-season; out-years refresh via a manual
~annual run). A mid-2026 keeper decision's entire information advantage is 2026 --
and the `future` family was built on files that have never seen it. #266 identified
this as the core problem and it was never solved; `future` was instead discounted to
a small weight to hide the staleness.

### 6. The identity join was broken underneath the model

Roster blobs carry a Yahoo `player_id` and `player_type` but **no `mlbam_id`**, while
the board is MLBAM-keyed. Joins fell back to `(normalized_name, player_type)`, which
is not unique. Fixing this in report logic churned repeatedly and never converged
(#282/#283). A model joined to the wrong player is wrong regardless of its weights.

**This is still open as #284** and is the one thread deliberately left alive; it is a
shared root with #230 (draft-value) and the closed #269.

## Constraints on any replacement

1. **Define the decision, then evaluate against the decision.** The objective is the
   value of my 3 retention slots against their opportunity cost -- not pool-wide rank
   correlation with next season.
2. **Evaluate where the decision happens.** Top-of-board and own-roster accuracy is
   what matters. Pool-wide rho rewards being right about players nobody would keep.
3. **Use a multi-year horizon.** Indefinite, no-cost retention means age and
   trajectory are first-order, not adjustments.
4. **Respect the noise floor.** Do not select between models whose holdout gap sits
   inside it, and do not report or tune weights to more precision than two transition
   seasons support. If the evaluation cannot separate two candidates, say so and pick
   on grounds other than the number -- or get more data first.
5. **Keep the parameter count small.** Eleven fitted weights on two seasons overfits
   by construction.
6. **Be explicit about each input's information set.** Anything built on out-year ZiPS
   has not seen the current season. Either fold the current season in properly or do
   not lean on it.
7. **Fix the identity join first (#284).** Populate `mlbam_id` on roster ingest via the
   existing Yahoo/MLBAM crosswalk (`lineup/yahoo_roster.py`, `analysis/draft_value.py`
   `_row_mlbam`, `streaks/data/load.py`) and join on the id.
8. **Playing-time memory is weak -- do not re-derive it.** Regressing next-season
   playing time on current plus prior season gives the prior a coefficient of +0.03
   (hitters) / +0.08 (pitchers). A predicted-next-year-PT family correlates with
   current PT alone at 0.999. Measured twice, on #288. Current-season PT already
   carries it.

## What survives

`src/fantasy_baseball/keepers/` is now **ingest and normalization only** -- no scoring:

- **Ingest:** `cache`, `mlb_stats` (MLB Stats API), `savant` (xBA/xSLG/barrels/xHR),
  `bref` (BBRef batting/pitching), `appearances`
- **Normalization:** `actuals`, `vintages` (ZiPS exports), `skills` (season rates),
  `positions` (eligible slots)
- **Entry point:** `scripts/fetch_keeper_skills.py` builds the skills cache

This layer is kept because it is real plumbing with real edge cases: FanGraphs is
Cloudflare-403 blocked across the board, so MLB Stats API + Savant + BBRef is the
working path, and rebuilding it would be pure re-work.

## Preserved artifacts

- `docs/superpowers/keeper-277-verdict-2026-07-30.md` -- the bake-off verdict and
  noise floors
- `docs/superpowers/keeper-277-bakeoff-2026-07-30.txt` -- the raw run
- `docs/superpowers/keeper-calibration-finding-2026-07-27.md`
- `docs/superpowers/keeper-scarcity-per-season-2026-07-30.txt`
- `docs/superpowers/specs/` and `plans/` -- the 2026-07-22 through 07-27 designs
- Closed as part of this reset: #257, #258, #260, #262, #266, #269, #275, #277, #278,
  #285, #288, #290, #291, and PR #289
