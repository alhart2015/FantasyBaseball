# HR-confirmation backtest: barrels/xHR vs the xSLG-gap proxy

Issue: #262 (child of #258 DARKO trajectory). Feature line: keeper breakout/mirage
diagnostic (`src/fantasy_baseball/analysis/breakout.py`, spec
`docs/superpowers/specs/2026-07-24-keeper-breakout-diagnostic-design.md`).

## Problem

`w_for_stat` confirms a hitter's HR is real via the **SLG-vs-xSLG gap**
(`_confirm_gap(row.slg, row.xslg, 0.150)`). That is a proxy: xSLG is expected
*total bases*, so it conflates HR with 2B/3B -- a hitter can beat xSLG on gap
doubles while his HR are legit, and the diagnostic would wrongly dock his home
runs. Two more HR-direct signals exist and are unused for confirmation:

- **barrel rate** -- already fetched onto `SkillLuckRow.barrel_pct` (and `brl_pa`
  is available in the same Savant leaderboard) but `w_for_stat` never reads it.
- **xHR** -- park-adjusted expected home runs, a separate Savant leaderboard, not
  fetched at all.

Directness for HR ranks **xHR > barrel > xSLG**; we shipped the weakest because it
came free alongside xwOBA/xBA/xSLG.

## Goal (this spec)

A backtest that answers: **does barrel rate and/or xHR confirm next-year HR better
than the SLG-vs-xSLG gap?** Wire a challenger into `w_for_stat` **only if it
clearly beats the proxy** (bootstrap CI on the head-to-head excludes 0). This is a
go/no-go gate, not an automation commitment. Per the feature line's hard-won
lesson, the fancier/more-direct signal has repeatedly *not* beaten the simpler one
-- barrels/xHR must earn their place here.

## Non-goals

- Pitchers (HR-against is out of scope; hitters only, matching the diagnostic v1).
- Changing the label boundary, the w-mapping's reliability/`confirm_weight`, or any
  non-HR confirmation. Only the **HR confirmation source** is the variable under test.
- Re-tuning `stat_stabilize["hr"]` (120) or `confirm_weight` (0.5); held fixed at
  shipped values so the test isolates the confirmation source.

## Data

New fetcher in `src/fantasy_baseball/data/skill_luck.py` for the Savant expected-HR
leaderboard, following the existing `load_statcast_hitters` pattern (cache-or-fetch,
`_rename_strict`, injectable fetcher for tests):

- URL: `https://baseballsavant.mlb.com/leaderboard/home-runs?type=batter&year={year}&min=1&csv=true`
- Columns kept: `player_id` -> `mlbam`, `xhr` (park-adjusted expected HR count),
  `hr_total` (Savant's HR count, used only for validation vs the MLB line).
- Cache file: `data/skill_luck/sc_hr_h_{year}.csv` (mirrors `sc_x_h_*`, `sc_brl_h_*`).
- **Coverage: 2016-2025** (2015 returns an empty leaderboard -- verified). `xhr` is a
  season *count*; converted to a rate as `xhr / pa` using PA from the MLB line.

`SkillLuckRow` gains one optional field `xhr: float | None` (season xHR count, None
pre-2016 or unmatched). `brl_pa` is added to the barrel fetch rename (already in the
cached CSVs) so the barrel candidate has a per-PA rate; `barrel_pct` stays as-is.

Barrel and xSLG ingredients are already on disk (2015-2025). No FanGraphs -- no
Cloudflare 403.

## The three candidates

Each candidate is a drop-in replacement for the HR branch of `w_for_stat`, i.e. it
produces `confirm in [0,1]`; everything else in the w-blend is held fixed:

| id | confirm | expected-HR source |
|----|---------|--------------------|
| `xslg` (incumbent) | `_confirm_gap(slg, xslg, scale)` | proxy (total bases) |
| `barrel` | `_confirm_gap(hr_rate, brl_expected_hr_rate, scale)` | `fit(brl_pa -> HR/PA)` on fit-years |
| `xhr` | `_confirm_gap(hr_rate, xhr/pa, scale)` | park-adjusted xHR |

`brl_expected_hr_rate` is a single OLS slope/intercept fit `HR/PA ~ brl_pa` on the
**fit-years only** (barrels are a rate skill, not an expected-HR count, so they need
one calibration step to become an expected-HR; xHR arrives pre-calibrated). The
calibration is fit over all fit-year hitter-seasons that clear the PA floor and have
`brl_pa` present (not just HR-candidates), so the slope is estimated on the full
skill range.

**Scale tuning (fair both ways).** Each candidate's `scale` (the `_confirm_gap`
denominator) is tuned on **fit-years only** to maximize the **same** metric the
verdict gates on -- fit-year forward-line Spearman -- so no candidate is tuned for
one metric and judged on another. Tuning is a **grid search** (not continuous
optimization, to stay reproducible and avoid degenerate flats): the HR/PA-gap
candidates (`barrel`, `xhr`) sweep scale over `{0.010, 0.015, ..., 0.060}` (2.5th-
to 97.5th-percentile-ish HR/PA gaps); the SLG-gap incumbent sweeps over `{0.075,
0.100, ..., 0.250}`, a grid bracketing its shipped 0.150. Ranking inside the tuner
reuses `breakout_backtest._spearman` (the ordinal tie ranker the memory flags as
load-bearing -- not scipy's tie-averaged `rankdata`). MAE is computed and reported
but is **not** a tuning target. The incumbent is additionally reported at its
**shipped 0.150** (untuned) so we can separate "re-tuning moved it" from "the signal
is better."

## Primary test -- matched to how the signal is used (go/no-go)

For each hitter-season on the common support, reusing `build_corpus` +
`marcel_prior` + `line_rates` (all existing):

- `prior_hr` = Marcel prior HR/PA; `surface_hr` = current HR/PA;
  `actual_hr` = realized *next-year* HR/PA; `reliability = pa / (pa + 120)`.
- For each candidate C: `forward_hr_C = prior_hr + reliability * ((1-cw) + cw*confirm_C) * (surface_hr - prior_hr)`, with `cw = 0.5` (shipped).
- Score `forward_hr_C` vs `actual_hr`:
  - **Spearman** (ranks next-year HR) -- reuse `breakout_backtest._spearman`.
  - **rate-MAE** (forecast accuracy) -- reuse `breakout_backtest.rate_mae`.
  - **Bootstrap CI** on `Spearman(C) - Spearman(xslg)` -- reuse `_bootstrap_diff`,
    with a **fixed RNG seed** (a script constant, e.g. 12345) so the reported CI is
    deterministic given the corpus. A decision-grade CI that flips the wire-in verdict
    must not depend on run-to-run resampling noise -- this bootstrap family is already
    flagged verdict-sensitive in the feature-line memory.

**Verdict rule (go/no-go gate):** the CI is the sole gate. A challenger is
**go/no-go-eligible** iff its Spearman-difference CI vs `xslg` **excludes 0** (lower
bound > 0). MAE is a **reported consistency check**, not part of the gate: a
challenger that clears the CI but whose MAE is worse than `xslg` by more than a stated
epsilon (0.0005 HR/PA, ~0.3 HR / 600 PA) is reported as "CI-positive but
MAE-inconsistent" rather than an unqualified win. This keeps the gate falsifiable
(the earlier "MAE no worse" had no tolerance). Same CI bar
`scripts/backtest_coefficient.py` uses. Report also includes a `surface` (no
regression) and `prior` (full regression) baseline for context.

## Level-control diagnostic (the anti-confound)

The naive "overperformed -> regresses" gap is level-confounded: extreme HR hitters
mean-revert regardless of signal (memory: elite overperformers regress ~85% vs base
~53%). So, alongside the go/no-go, stratify seasons into **prior-HR/PA terciles** and
within each tier report, per candidate, the Spearman between **that candidate's own
signed over/under-performance signal** and the next-year change
`(actual_hr - surface_hr)`. The signal differs by candidate because each confirms off
a different quantity: `(slg - xslg)` for `xslg` (SLG units), `(surface_hr -
brl_expected_hr)` for `barrel`, and `(surface_hr - xhr/pa)` for `xhr` (both HR/PA
units). A real luck signal shows persistent *negative* correlation within tiers (a
positive over-performance predicts a next-year decline), not only across them. A
candidate that only "works" by sorting on level dies here.

**Decision role (not decorative).** The go/no-go CI is the gate; this diagnostic is a
**confound veto** layered on top. A challenger is **wire-in eligible** only if it (a)
is go/no-go-eligible (CI excludes 0) **and** (b) survives level-control: its
signed-gap-vs-next-year-change Spearman keeps the expected negative sign in **at least
2 of the 3** prior-HR/PA tiers. A challenger that clears the CI but whose edge
collapses within tiers (expected sign in 0-1 tiers) is reported as
**"level-confounded -- do not wire in"**, not a win. This is exactly the trap the
feature line hit before, so it gets an explicit gate, not a footnote.

## Sample and splits

- Hitters, **common support: source years 2016-2024** (predicting 2017-2025; 2025 is
  cached). All three candidates scored on the identical player-seasons.
- **Fit years 2016-2020** (scale + barrel calibration only, never scored);
  **report years 2021-2024** (held-out; the CI/Spearman/MAE come from here). The script
  **prints the actual scored n**; if the held-out candidate count is too thin for a
  usable CI, widen report years (documented, not silent). ~4 report years of qualified
  HR candidates is expected to suffice, but n is reported, not assumed.
- Filters (all exposed as named script constants / sensitivity knobs, not magic
  numbers buried in the loop):
  - **PA floor** `PA_FLOOR = 150` on the current (source) year, and the player must
    also appear the following year (any PA) so a next-year HR/PA exists.
  - **HR-candidate filter** `HR_MOVE_MIN = 0.005` HR/PA (~3 HR over 600 PA): keep
    seasons where `|surface_hr - prior_hr| >= HR_MOVE_MIN`, i.e. players whose HR
    actually moved (the mirage/breakout population), not the stable middle.
  - The Savant HR leaderboard's own `min` gate is set to `min=1` (widest) so the
    modeling filters above -- not Savant's qualification rule -- define the population;
    the `min` semantics (batted-ball threshold) are noted under Risks.
- Robustness note (not the gate): barrel/xSLG additionally reported on 2015-2024 to
  confirm the common-support restriction didn't change their standing.

## Deliverables

- `src/fantasy_baseball/data/skill_luck.py`: `load_statcast_hr(cache_dir, year, *,
  fetcher=None)` + `xhr` on `SkillLuckRow` + `brl_pa` on the barrel rename, threaded
  through `build_hitter_skill_luck`. Pitcher rows set `xhr=None`.
- `scripts/backtest_hr_confirm.py`: builds the common-support corpus by reusing
  `backtest_breakout.build_corpus` as-is -- it carries xHR automatically once
  `SkillLuckRow.xhr` is threaded through `build_hitter_skill_luck` (no change to
  `build_corpus` itself). Marcel-prior reconstruction reuses
  `breakout_backtest.marcel_prior`; the private per-year helpers it needs
  (`_league_mean`, `_rates_to_line`) are either imported directly or first lifted to a
  small shared public function in `breakout_backtest` (an implementation choice for the
  plan, called out here so it is not invented silently). Runs the primary test +
  level-control, prints a verdict, writes `data/stats/hr_confirm_backtest_results.csv`.
- **No change to `breakout.py`'s `w_for_stat`** in this spec. Wiring a challenger in is
  a *follow-up* gated on the verdict (its own small diff + a pinned test), so the
  backtest result is reviewable before any behavior change.

## Testing

- Unit test the new fetcher's rename/rate conversion with an injected fake CSV
  (no network), mirroring existing `skill_luck` tests.
- Unit test the barrel `HR/PA ~ brl_pa` calibration and each `confirm_*` on a tiny
  synthetic corpus (monotonicity: a bigger over-performance gap -> lower confirm).
- Pin the full verdict logic on synthetic results so a future refactor can't silently
  flip the gate, matching the existing `test_spearman_tie_handling_pinned` discipline:
  (a) CI lower bound > 0 -> go/no-go-eligible; (b) the level-control veto -- a
  CI-positive candidate with the expected sign in <2 tiers is labeled
  "level-confounded"; (c) the MAE-consistency epsilon flag.
- Pin bootstrap **determinism**: two runs of `_bootstrap_diff` on the same records with
  the fixed seed return identical CI bounds.

## Risks / open points

- **xHR starts 2016**, costing one source year and the pre-2016 seasons; acceptable
  (barrels also effectively start 2015). Head-to-head is on common support by design.
- **Savant `min`/qualification** on the HR leaderboard differs from the xStats/barrel
  leaderboards; the join is left-join on MLBAM and a season missing any candidate's
  ingredient is dropped from the common-support set (counted and logged, never
  silently truncated).
- **Absolute-bias caveats that cancel in the relative gate:** requiring a next-year
  line is survivorship-biased (collapsed/retired players drop out), and repeated players
  across report years make bootstrap resamples mildly non-independent. Both inflate the
  *absolute* Spearman/MAE optimistically, but the gate is a **paired** challenger-vs-
  incumbent difference on identical seasons, so the bias is common-mode and largely
  cancels. Same structure as the existing `backtest_coefficient`; accepted, and noted
  so absolute numbers aren't over-read.
- If **no challenger clears the gate**, the deliverable is the negative result + the
  proxy stays; that is a valid, expected outcome per the guardrail.
