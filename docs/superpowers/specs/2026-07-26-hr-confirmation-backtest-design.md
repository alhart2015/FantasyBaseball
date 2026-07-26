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
one calibration step to become an expected-HR; xHR arrives pre-calibrated). Each
candidate's `scale` (the `_confirm_gap` denominator) is tuned on fit-years by
maximizing fit-year forward-line Spearman, so no candidate is handicapped by a bad
constant. The incumbent is additionally reported at its **shipped 0.150** so we see
whether re-tuning alone moved it.

## Primary test -- matched to how the signal is used (go/no-go)

For each hitter-season on the common support, reusing `build_corpus` +
`marcel_prior` + `line_rates` (all existing):

- `prior_hr` = Marcel prior HR/PA; `surface_hr` = current HR/PA;
  `actual_hr` = realized *next-year* HR/PA; `reliability = pa / (pa + 120)`.
- For each candidate C: `forward_hr_C = prior_hr + reliability * ((1-cw) + cw*confirm_C) * (surface_hr - prior_hr)`, with `cw = 0.5` (shipped).
- Score `forward_hr_C` vs `actual_hr`:
  - **Spearman** (ranks next-year HR) -- reuse `breakout_backtest._spearman`.
  - **rate-MAE** (forecast accuracy) -- reuse `breakout_backtest.rate_mae`.
  - **Bootstrap CI** on `Spearman(C) - Spearman(xslg)` -- reuse `_bootstrap_diff`.

**Verdict rule:** a challenger "clearly beats" the proxy iff its Spearman-difference
CI vs `xslg` **excludes 0** AND its MAE is no worse. Otherwise it stays out. Same bar
`scripts/backtest_coefficient.py` uses. Report also includes a `surface` (no
regression) and `prior` (full regression) baseline for context.

## Level-control diagnostic (the anti-confound)

The naive "overperformed -> regresses" gap is level-confounded: extreme HR hitters
mean-revert regardless of signal (memory: elite overperformers regress ~85% vs base
~53%). So, alongside the go/no-go, stratify seasons into **prior-HR/PA terciles** and
within each tier report, per candidate, the Spearman between the signed gap
`(surface_hr - expected_hr)` and the next-year change `(actual_hr - surface_hr)` (a
real luck signal shows persistent *negative* correlation within tiers, not only
across them). A candidate that only "works" by sorting on level dies here. Reported as
a diagnostic, not the gate.

## Sample and splits

- Hitters, **common support: source years 2016-2024** (predicting 2017-2025; 2025 is
  cached). All three candidates scored on the identical player-seasons.
- **Fit years 2016-2020** (scale + barrel calibration only, never scored);
  **report years 2021-2024** (held-out; the CI/Spearman/MAE come from here). ~4 report
  years x ~250 qualified HR candidates gives adequate CI power for a single stat.
- Filters: player present both years; `min` PA gate (reuse the leaderboard's `min=1`
  plus a modeling PA floor, e.g. >= 150 PA current year) ; **HR-candidate filter** --
  `|surface_hr - prior_hr|` above a small rate threshold, so we test on players whose
  HR actually moved (a mirage/breakout population), not the stable middle.
- Robustness note (not the gate): barrel/xSLG additionally reported on 2015-2024 to
  confirm the common-support restriction didn't change their standing.

## Deliverables

- `src/fantasy_baseball/data/skill_luck.py`: `load_statcast_hr(cache_dir, year, *,
  fetcher=None)` + `xhr` on `SkillLuckRow` + `brl_pa` on the barrel rename, threaded
  through `build_hitter_skill_luck`. Pitcher rows set `xhr=None`.
- `scripts/backtest_hr_confirm.py`: builds the common-support corpus (reusing
  `backtest_breakout.build_corpus`, extended to carry xHR), runs the primary test +
  level-control, prints a verdict, writes `data/stats/hr_confirm_backtest_results.csv`.
- **No change to `breakout.py`'s `w_for_stat`** in this spec. Wiring a challenger in is
  a *follow-up* gated on the verdict (its own small diff + a pinned test), so the
  backtest result is reviewable before any behavior change.

## Testing

- Unit test the new fetcher's rename/rate conversion with an injected fake CSV
  (no network), mirroring existing `skill_luck` tests.
- Unit test the barrel `HR/PA ~ brl_pa` calibration and each `confirm_*` on a tiny
  synthetic corpus (monotonicity: a bigger over-performance gap -> lower confirm).
- Pin the verdict-rule logic (CI excludes 0 -> "beats") on a synthetic result so a
  future refactor can't silently flip the gate, matching the existing
  `test_spearman_tie_handling_pinned` discipline.

## Risks / open points

- **xHR starts 2016**, costing one source year and the pre-2016 seasons; acceptable
  (barrels also effectively start 2015). Head-to-head is on common support by design.
- **Savant `min`/qualification** on the HR leaderboard differs from the xStats/barrel
  leaderboards; the join is left-join on MLBAM and a season missing any candidate's
  ingredient is dropped from the common-support set (counted and logged, never
  silently truncated).
- If **no challenger clears the gate**, the deliverable is the negative result + the
  proxy stays; that is a valid, expected outcome per the guardrail.
