# Trajectory band calibration (2026-09-04)

Reproduce:

```bash
python scripts/calibrate_band_coverage.py --pool both --sample 0 --out cov.csv
python scripts/calibrate_band_coverage.py --from-csv cov.csv   # re-analyse, no re-run
```

62,224 held-out query-horizons over the 2000-2025 panel, 250 bootstrap draws (the
board's own `SWEEP_DRAWS`). Query player held out for his whole career;
`last_complete_season` from the full panel so a holdout never moves censoring for anyone
else; a forward season he did not play is a real 0. Run on raw SGP, which is the same
question as VAR: `#331` made `y_var = y_sgp - replacement` affine, so prediction, both
band edges and the outcome all shift by the same floor and tail coverage is identical.

## Why it was run

`trajectory_board.py` told the reader to trust the band over the point estimate on a
`(!)` row, citing "measured on breakouts the interval is calibrated for hitters
(12%/11%)". That claim existed in exactly one place -- a comment string added in
`3d246cc6`, a docs-only commit -- and `scripts/backtest_trajectory.py`, the tool that
would reproduce it, was deleted in `55451f70` (#325). The coverage table that does carry
a sample size (`model.PathPoint.p10`, n=729 hitters at h=3) is general-population: 445 of
the live board's 602 hitters sit above 30% local support, so it says nothing about the
six below 5%. Nobody had ever measured coverage on the slice the `(!)` flag is drawn on.

## Result 1 -- the flag threshold was in the wrong place

Share of outcomes below p10 at horizon +3. Nominal is 10%.

| local support | hitters | pitchers |
|---|---|---|
| <5% (flagged) | 12% | 24% |
| 5-10% (flagged) | 20% | 24% |
| **10-30% (not flagged)** | **18%** | **24%** |
| >30% | 4% | 4% |

`MIN_LOCAL_SUPPORT = 0.10`, but the cliff is at ~30%. The unflagged 10-30% bucket is as
miscalibrated as the flagged rows beneath it -- 2,986 hitter and 2,612 pitcher rows the
board passed silently -- while above 30% the band is far too wide rather than too narrow.
A mark that fires on the wrong rows and stays quiet on equally bad ones is worse than the
`supp` column it decorates.

This is what retired `(!)`, `(!!)`, and the prose paragraphs beside them. Every surface
now prints `support` as a number: the league board always did, the per-team blocks and
the teams view did not, and the single-player views printed a glyph or a paragraph.

## Result 2 -- the per-year band is not the summed band

The board headlines a multi-year total whose band `sweep.totals` builds by SUMMING the
yearly bands, which assumes the years move together. Measured, that over-widening roughly
cancels the per-year narrowness:

| pool, support | below p10 | above p90 | reading |
|---|---|---|---|
| hitter <5% | 5% (3-9%) | 7% (4-11%) | too wide |
| hitter 5-10% | 9% (7-12%) | 7% (5-9%) | calibrated |
| hitter 10-30% | 11% (10-12%) | 7% (7-8%) | calibrated |
| pitcher <5% | 8% (6-12%) | 8% (6-12%) | calibrated |
| pitcher 5-10% | 13% (10-16%) | 9% (7-12%) | borderline |
| >30%, both | 3-4% | 7% | far too wide |

Median summed width is 36.2 against ~20.8 if the three years were independent, about
1.7x. Two errors cancelling is not a design, but the printed interval is honest.

**So: the summed range is usable at every support level. The single-year range beside it
understates the downside below ~30% support.** That distinction is the one a reader needs
and no surface had ever drawn it.

## Result 3 -- the point estimate is unbiased everywhere

`actual - predicted` on the summed h1-3 number; negative would mean the board overstates.
Bootstrap CI over 4,000 resamples.

| support | hitters | pitchers |
|---|---|---|
| <5% | +1.10 [-0.38, +2.53] | +0.82 [-0.70, +2.34] |
| 5-10% | -0.06 [-1.21, +1.06] | +0.60 [-0.59, +1.76] |
| 10-30% | +0.14 [-0.24, +0.52] | +0.14 [-0.25, +0.56] |
| >30% | -0.17 [-0.32, -0.02] | -0.10 [-0.22, +0.03] |

Every CI straddles zero except hitters above 30%, where the effect is -0.17 against a
mean prediction of 9.18. Low-support estimates come in slightly HIGH if anything -- the
model under-predicts them.

This kills the intuition that a thinly-supported breakout should be shrunk toward a prior
because "breakouts regress". Measured, they do not, and the `(!)` glyph's title -- "read
the band, not the point estimate" -- was asserting something untrue.

Note the median error is well below the mean in every bucket (hitters >30%: median -1.47,
mean -0.17). The outcome distribution is right-skewed: most players undershoot slightly,
a few overshoot enormously. That is the shape a keeper decision is buying, and it is why
an expected value understates a keeper whose downside you can walk away from.

## What is still not measured

- **Coverage of the summed band at horizons past 3.** Only h1-3 was run.
- **Whether the >30% band being far too wide costs anything.** It is the conservative
  direction, so it was left alone, but a tighter band there would separate supported
  players from each other rather than lumping them under one ~21-unit interval.
- **`band_fell_back` rows as their own slice.** They are carried in the CSV
  (`--out`) but were not bucketed separately; n is small.

---

# Update 2026-09-05: the band is corrected, not just measured

Everything above measured the problem. `trajectory.calibration` fixes it.

## What ships

`data/trajectory/band_calibration.json` -- split conformal multipliers, Mondrian on
`(pool, target, support bucket, side)`, built by `scripts/build_band_calibration.py`.
Each half-width is scaled so its tail holds the nominal 10% by construction.

Two rules make it correct rather than approximately helpful:

1. **Every displayed quantity is calibrated separately, each against the RAW band.** The
   year-`h` column takes `y{h}`; the 1..k total takes `s{k}`, fitted against the raw sum.
   Nobody asks how year 2 and year 3 covary, because the sum is its own target.
2. **The consumer sums raw and corrects once.** Correcting years and summing those was
   tried first and read 13-17% below p10 in-sample against a nominal 10% -- a
   miscalibration shaped like a correction. Pinned by
   `test_the_span_multiplier_is_fitted_against_a_raw_sum`.

## Coverage after, by rolling origin

Fit on every outcome observable before season Y, measure on those resolving in Y -- the
production refresh exactly. Spans containing the 60-game 2020 season excluded (see below).

| target | below p10 | above p90 | before |
|---|---|---|---|
| y1 | 10.0% | 9.7% | 8.1-20.5% below |
| s3 | 10.0% | 9.3% | 3.4-12.4% below |
| s5 | 10.8% | 9.2% | 1.3-14.6% below |

Worst deviation 0.8 points, against a spread of 1.3% to 26% before.

## Three parameters, each chosen by measurement

**One bucket edge, at 30%** (`SUPPORT_EDGES`). Fitting on 2001-2015 and again on
2016-2024 and comparing: 4 buckets drift by up to 0.68, 2 buckets by 0.10, with identical
median accuracy. At the thin end that drift INVERTS the correction -- the `<5%` cell
wanted x0.66 early and x1.34 late, and the wrong one took hitter s3 from 13.3% to 30.0%.

**An 8-year fit window** (`CALIBRATION_WINDOW_YEARS`). The required multiplier drifts, so
all-history lags it and leaves the upper tail cold (8.7-9.8%). Eight years halves the
worst deviation. Recency-weighting was also tried against the early/late split and does
nothing there (11.8% at 5, 10 and 15 years alike) -- the gap in that test is the nine-year
extrapolation, not stale multipliers.

**2020 reported, not corrected for.** A 60-game season scaled to 162 carries ~2.7x the
sampling variance of a full one, so spans touching it are genuinely more dispersed:
year-1 coverage there is 16.4%. Excluding it from the FIT moves multipliers by a median
0.005, so it buys nothing, and pricing a pandemic into every future band permanently is
the larger error.

## Effect on the live board (2027-29 VAR, 2026-09-04 panel)

Means and ranks are untouched -- calibration scales only `p10`/`p90`.

| pool | support | old width | new width | change |
|---|---|---|---|---|
| hitter | <30% | 32.4 | 29.5 | -9% |
| hitter | >30% | 22.3 | 17.4 | -22% |
| pitcher | <30% | 29.8 | 29.0 | -3% |
| pitcher | >30% | 17.1 | 13.3 | -22% |

The -22% is the useful part: supported players used to share one ~22-unit interval, so
the band said nothing about which of them was better understood. Now it does.

## Operational

* Re-run `scripts/build_band_calibration.py` whenever `data/trajectory/` is rebuilt. The
  artifact carries the panel filenames and `BandCalibration.load` REFUSES a mismatch.
* `--horizon` is capped at `MAX_HORIZON` (5). Past that there is no fitted `s{k}` and the
  board would print a band nothing measured.
* A missing artifact degrades to the uncorrected band rather than raising, so a fresh
  clone still renders.

## Still not measured

* Coverage past horizon 5.
* Whether the `>30%` band, still the widest thing on the board, could be tightened
  further by conditioning on something other than support.
* `band_fell_back` rows as their own slice -- carried in the CSV, never bucketed. It fires
  on ~15 of 90,555 held-out rows, so it is a rounding error either way.

---

# Update 2026-09-05b: probabilities are the product

The board now headlines P(elite) / P(keeper) / P(bust). The mean, band and support are
behind `--detail` on the CLI and `?detail=1` on the web.

## The bars are measured, not projected

`trajectory.value_bars` + `scripts/build_value_bars.py` -> `data/trajectory/value_bars.json`.

The bar a probability is compared against MUST be realized, because `exceedance` inverts
a distribution over realized outcomes. Using quantiles of the projected pool -- which is
what happened first -- understates by about two tiers:

| rank | projected 2027-29 | realized 2022-24 / 2023-25 |
|---|---|---|
| #10 | 11.8 | 19.9 / 21.1 |
| #30 | 4.6 | 14.3 / 14.2 |
| #100 | -2.2 | 5.0 / 4.9 |

SD(actual) is 1.36x SD(predicted) on held-out 3-year totals, and the gap widens with the
quantile (+6.0 at the 90th, +8.7 at the 99th). Only 76 of 1,277 projected players clear
replacement over three years; 157-170 actually did. The projected #30 bar is roughly the
realized #100 bar.

**`elite` and `keeper` are DERIVED from `league.yaml`** -- one per team, and every keeper
slot in the league -- so a rule change moves the bars. `bust` is rank 100, a judgement,
and it lands near where realized VAR crosses zero.

## Window coverage bounds the feature

Eligibility is cached from 2022 (`data/cache/keeper_skills/mlb_fielding_*.csv`), so a
k-year window needs an eligible start and k complete seasons:

| span | windows | elite | keeper | bust |
|---|---|---|---|---|
| s1 | 4 | 8.9 | 6.4 | 2.7 |
| s2 | 3 | 15.6 | 10.2 | 4.4 |
| s3 | 2 | 20.5 | 14.2 | 5.0 |
| s4 | 1 | 25.4 | 16.4 | 5.9 |
| s5 | **0** | -- | -- | -- |

A span with no window carries no bar and the board prints `--` rather than a number.
Bars are NOT linear in span -- the 3-year keeper bar is 2.2x the 1-year, not 3x, because
holding a rank for three years is harder -- so a missing span cannot be extrapolated. A
new season of fielding data extends the longest measurable span; re-run the build.

## `exceedance` takes the RAW band

The curve is fitted on `signed_scores` of the uncorrected held-out band, so the
half-widths passed in are its denominators. Handing it a calibrated band divides by an
already-scaled width. Measured: Caminero reads 23% bust on the raw band and 16% on the
corrected one. The parameters are named `raw_p10` / `raw_p90` for this reason, and
`sweep._probabilities` passes the raw sum while the corrected band is what gets displayed.

## What the board shows now

Sorted by P(keeper), not by the projected mean -- the mean is an input to the probability,
and the two order the pool differently wherever the bands differ. `bar_note()` prints the
bars and their window count above the table, so the thresholds are on the page rather than
in a docstring.

Effect on the top of the live board (2027-29): Caminero leads on P(keeper) at 43-44%
against a 23% bust, where Crow-Armstrong has the higher mean (18.9 vs 17.1) and a 26%
bust. That reordering is the feature.
