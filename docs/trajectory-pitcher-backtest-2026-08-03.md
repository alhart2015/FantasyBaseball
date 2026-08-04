# Trajectory: shape vs level matching on PITCHERS (#313)

Measured 2026-08-03 on the 2000-2026 panel. Reproduce with:

```bash
python scripts/backtest_trajectory.py --pool pitcher                  # elite slices
python scripts/backtest_trajectory.py --pool pitcher --sample 700     # role coverage
python scripts/backtest_trajectory.py --pool hitter                   # the reference
```

Every run prints two tables: `current -> shape` over all scored rows, and a three-way
including `track` over the subset track can score.

The query player is removed from the panel entirely before either estimator is built,
so neither can match him to himself.

## Verdict

**Shape stays the default for pitchers.** It beats level matching on every slice with
enough rows to measure. But it wins for a *different reason* than it wins for hitters,
and that distinction should survive into how the number is used.

- **Hitters: shape buys discrimination.** 61-64% of individual predictions are closer,
  RMSE falls 11-21%.
- **Pitchers: shape buys calibration.** 55-60% win rate is close to a coin flip and
  RMSE falls only 2-8%, but the systematic bias is eliminated: **-1.69 -> +0.17 SGP**
  on elite pitchers.

That bias fix is worth more than the modest RMSE gain suggests, because the board in
#321 ranks on VAR **summed across the horizon**. Level matching is pessimistic about
elite pitchers by ~1.7 SGP per year; over a five-year keeper horizon that compounds to
roughly 8.5 SGP of systematic understatement. Discrimination being near coin-flip means
shape will not reliably tell you which of two similar pitchers is better -- it will tell
you what a pitcher is worth without a thumb on the scale.

## The mechanism argument in #313 was correct

#313 predicted the advantage would be small or absent because `b(last)` never overtakes
`a(now)` for pitchers. The coefficients reproduce exactly (age 28):

| peak | HIT a(now) | HIT b(last) | PIT a(now) | PIT b(last) |
|---|---|---|---|---|
| 4 | +0.677 | +0.116 | +0.577 | +0.099 |
| 8 | +0.615 | +0.217 | +0.551 | +0.180 |
| 14 | +0.501 | +0.372 | +0.551 | +0.187 |
| 18 | +0.415 | +0.468 | +0.580 | +0.194 |

Hitters cross over by peak 18; pitchers stay flat at ~0.55 / ~0.19 across the range.

**And the prediction held where it mattered.** The decision-relevant slice -- an elite
arm coming off a big down year, where shape's whole value for hitters is discounting
the down year as noise -- is where pitchers fall furthest short:

| elite big drop (<70% of prior) | n | RMSE | shape wins |
|---|---|---|---|
| HITTERS | 239 | 6.03 -> 4.78 (-20.7%) | **64%** |
| PITCHERS | 207 | 5.91 -> 5.42 (-8.3%) | **53%** |

53% is a coin flip. The mechanism really is absent; shape simply is not *worse*, and it
still corrects the bias on that slice (-2.22 -> +0.24).

## Elite run (prior >= 14 SGP, ages 24-32, n=502)

| slice | n | RMSE | MAE | bias | shape wins |
|---|---|---|---|---|---|
| elite | 502 | 5.99 -> 5.53 | 4.79 -> 4.49 | -1.69 -> +0.17 | 56% |
| elite down year (<80%) | 266 | 5.86 -> 5.38 | 4.71 -> 4.49 | -2.24 -> +0.11 | 55% |
| elite big drop (<70%) | 207 | 5.91 -> 5.42 | 4.67 -> 4.54 | -2.22 -> +0.24 | 53% |
| elite holding steady | 236 | 6.13 -> 5.69 | 4.88 -> 4.49 | -1.07 -> +0.23 | 56% |
| SP | 466 | 6.05 -> 5.56 | 4.85 -> 4.53 | -1.76 -> +0.13 | 56% |
| closer | 27 | 4.63 -> 4.49 | 3.80 -> 3.39 | -0.82 -> +0.42 | 67% |
| RP | 9 | -- | -- | -- | too thin |

The `ALL` row the script prints is identical to `elite` here and is not a separate
finding: without `--sample` the query set IS the elite set.

Hitter reference, same run: elite n=775, RMSE 5.08 -> 4.50, MAE 4.08 -> 3.54,
bias -1.84 -> +0.00, shape wins 61%. This reproduces the table published in #313
(5.08 -> 4.55, wins 61%) to within bootstrap noise, which is what licenses trusting the
pitcher numbers from the same harness.

## Random sample (n=700, ages 24-32) -- for role coverage

| slice | n | RMSE | MAE | bias | shape wins |
|---|---|---|---|---|---|
| ALL | 700 | 3.48 -> 3.41 | 2.50 -> 2.43 | -0.09 -> -0.06 | 58% |
| SP | 226 | 4.70 -> 4.53 | 3.58 -> 3.46 | -0.39 -> -0.21 | 55% |
| RP | 447 | 2.63 -> 2.61 | 1.93 -> 1.86 | +0.01 -> -0.04 | 60% |
| closer | 27 | 3.92 -> **4.10** | 3.05 -> **3.22** | +0.62 -> +0.79 | **41%** |
| either anchor negative | 140 | 2.04 -> 2.04 | 1.55 -> 1.30 | +0.29 -> -0.14 | 72% |
| current season negative | 109 | 1.82 -> 1.82 | 1.40 -> 1.10 | +0.34 -> -0.20 | 75% |

## Three-way: shape vs level matching vs track

The tables above race shape against `current` (level matching) only, which is what the
harness had always done. `track` -- the same two anchors, bounded by a hard band on the
prior instead of kernel-weighted -- is the closer competitor, and #325 proposes retiring
it. It had never been raced. It has now.

Reported on the subset where track's hard band finds any cohort; the coverage is stated
because that subset is smaller than the full run.

**HITTERS** (elite, track scored 766/775):

| slice | current | track | shape | shape beats track |
|---|---|---|---|---|
| elite | 5.02 | 4.61 | **4.46** | 54% |
| elite big drop (<70%) | 5.88 | 4.81 | **4.68** | 56% |
| elite holding steady | 4.50 | 4.47 | **4.30** | 53% |

**PITCHERS** (elite, track scored 481/502):

| slice | current | track | shape | shape beats track |
|---|---|---|---|---|
| elite | 5.88 | **6.04** | **5.52** | 58% |
| elite big drop (<70%) | 5.93 | 5.69 | **5.46** | 51% |
| elite holding steady | 5.89 | **6.42** | **5.63** | 63% |

Shape is best on every slice in both pools. Two things behind that headline are worth
carrying forward:

**1. For hitters, most of shape's advantage is "use the prior anchor at all", not the
kernel.** Track captures 73% of shape's total RMSE gain over level matching on the elite
slice, and **89%** on the decision-relevant big-drop slice (5.88 -> 4.81 of the
5.88 -> 4.68 available). Shape's edge over track is real but modest: 3.3% RMSE and a 54%
win rate, which is close to a coin flip. The honest hitter claim is therefore *shape is
the best of the three, and clearly better than level matching -- but only slightly
better than the crude version of the same idea.*

**2. For pitchers the ordering inverts: track is the WORST of the three**, worse than
plain level matching (6.04 vs 5.88 elite, 6.42 vs 5.89 on holding-steady). This is the
same mechanism from the other direction. `b(last)` carries almost nothing for pitchers,
so a hard band on the prior shrinks the cohort without adding signal. Shape *fits* the
weight on that anchor (~0.19) and so degrades gracefully where track cannot, because
track imposes the constraint regardless of whether it pays.

That inversion is the strongest single argument for shape over track, and it is an
argument the hitter numbers alone could not make: **shape adapts the weight on the prior
season to the pool; track assumes it.** Note also that track is well calibrated in both
pools (bias +0.03 pitchers, -0.21 hitters) -- it is unbiased but noisier. Shape gets
both.

## Three findings from the pitcher-specific concerns in #313

**1. Role heterogeneity does NOT average two opposite effects into a null.** The worry
was that a pooled pitcher number hides a starter effect cancelling a closer effect. It
does not: SP wins 55-56% and RP wins 60%, both positive, in both runs.

**2. Closers are UNRESOLVED, and the two runs disagree.** Elite closers (n=27) show
shape winning 67% with RMSE improving. Random-sample closers (n=27) show shape winning
**41%** with RMSE and MAE getting *worse*. Different populations, both n=27, opposite
signs. **This is noise, not a finding in either direction.** Per constraint 4 of
`keeper-value-teardown-2026-08-01.md` -- do not select between models whose gap sits
inside the noise floor -- the honest statement is that we cannot say whether shape helps
or hurts closers, and the default should not be forked on 27 rows. #306 (match pitcher
comps on role) is the real fix; revisit after it lands.

**3. Negative anchors are where shape is clearest.** 15% of pitcher-seasons score below
replacement and the linear form had never been checked against a negative anchor. Shape
wins **72-75%** there, cutting MAE from 1.55 to 1.30 and flipping bias from +0.29 to
-0.14. Note RMSE is unchanged (2.04 -> 2.04): shape fixes the typical error and the
bias, not the tail.

## Harness changes made for this

`scripts/backtest_trajectory.py` gained:

- `roles()` -- `(mlbam_id, season) -> "SP" / "closer" / "RP"`. The cuts are borrowed,
  not invented: `trajectory.value.STARTER_SHARE` for the starts share and
  `utils.constants.CLOSER_SV_THRESHOLD` for saves, so this buckets pitchers the same way
  the replacement floor and the draft board do. Split seasons are re-summed first --
  `collapse_split_seasons` keeps only `sgp` and `age`, so a traded pitcher would
  otherwise read as two half-roles.
- Role and negative-anchor slices on the pitcher pool.
- `report()` now prints `(under 10, not reported)` instead of returning silently. A
  slice that vanishes reads as "not applicable" when it means "too thin to measure" --
  which for the closer split is the finding itself.

Covered by `tests/test_trajectory/test_backtest_roles.py`.

## What this does NOT establish

- **Track's coverage gap is not analysed.** Track cannot score 21/502 pitcher and 9/775
  hitter queries at all (empty cohort under the hard band). Those rows are excluded from
  the three-way rather than counted as a track failure. If anything that flatters track,
  since the queries it cannot reach are the unusual ones.
- **Horizon 1 only.** Every number here is a one-year-ahead prediction. The keeper board
  sums five. The bias argument above assumes the per-year bias persists across horizons,
  which is plausible but unmeasured.
- **Closers, as above.**
- **The 10-IP two-way floor** (`panel._in_role`) was not isolated. The pitcher pool
  contains genuine two-way seasons whose forward path may follow the bat; they are in
  these numbers, unflagged.
