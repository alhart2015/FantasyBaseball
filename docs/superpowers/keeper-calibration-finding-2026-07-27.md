# Keeper-value calibration study: finding

Issue #266, increment 1. Spec:
`docs/superpowers/specs/2026-07-27-keeper-value-definition-design.md`.
Plan: `docs/superpowers/plans/2026-07-27-keeper-calibration-study.md`.

**The question.** How much of a season's performance surprise should carry forward into a
projection that has not seen that season?

```
updated_2027 = ZiPS_2027 + k * shrink * (actual_2026 - ZiPS_2026)
```

`k = 0` ignores 2026 (today's behaviour); `k = 1` transfers the whole shrunk surprise. This
study measures `k`, once per folded coefficient.

---

## Part A -- Advance decisions (pre-registered)

Spec 6.2 requirements 9 and 11 require the error metric and the gate thresholds to be recorded
**before the first fit**, because under this descope the increment that has to clear the bar is
also the increment that chooses the bar. Everything in Part A was written and committed before
any fitting code existed. Part B, below, is filled in afterwards.

Justification draws only on Task 5's sample/survivorship measurement, which is a property of the
data and not of any estimator. Those numbers, measured on the real fit sample
(`data/analysis/keeper_calibration_survivorship.csv`):

```
player_type  threshold  year  n_matched  n_in_year  n_survived  survival_rate
     hitter      100.0  2022        678        465         351          0.755
     hitter      100.0  2023        644        458         356          0.777
     hitter      100.0  2024        649        454         361          0.795
     hitter      300.0  2022        678        275         201          0.731
     hitter      300.0  2023        644        293         207          0.706
     hitter      300.0  2024        649        286         199          0.696
    pitcher       50.0  2022        758        344         216          0.628
    pitcher       50.0  2023        773        355         213          0.600
    pitcher       50.0  2024        777        350         216          0.617
    pitcher      100.0  2022        758        139          78          0.561
    pitcher      100.0  2023        773        126          73          0.579
    pitcher      100.0  2024        777        125          73          0.584
```

The hitter survival rates at 100 PA (0.755 / 0.777 / 0.795) reproduce spec 6.3 exactly. That is
the check that pair membership is `ZiPS_Y INTERSECT actual_Y` and has not been preconditioned on
surviving into Y+1; a preconditioned sample reads 0.84-0.87.

### A.1 The error metric

**Playing-time-weighted mean squared error, computed on each coefficient's own native scale, and
never aggregated across coefficients.**

- *Scale.* Each coefficient is evaluated in the units of the column it governs -- `hr_pa` in home
  runs per plate appearance, `ip` in innings. No standardization, no pooling.
- *Aggregation across the eleven rate coefficients: none.* Spec 6.6 makes acceptance
  per-coefficient, so there is nothing to aggregate. A single pooled figure would be dominated by
  whichever column has the largest variance and would let one bad coefficient ride in on ten good
  ones (or sink them).
- *Aggregation across the three held-out pairs:* reported per pair, and the verdict is a
  **majority of pairs** (spec 6.6), not a mean of errors. Averaging three errors would let one
  pair with a large-variance year decide the verdict.

**Weight column, and why it differs between the rate coefficients and the playing-time
coefficient.** This is a consequence of spec 6.3's warning that survivorship "necessarily differs
by coefficient", and of 5.4's rule that absence is not zero:

- **The eleven rate coefficients: weight = `target_pt`** (realized playing time in year Y+1). A
  rate observed over 20 PA is nearly pure noise and must not dominate a rate observed over 600.
  Non-survivors carry a NaN target rate and are therefore excluded -- a player who did not appear
  in MLB in Y+1 has *no* Y+1 rate, only an absent one. **The rate coefficients are consequently
  measured on survivors only, and are conditional on continued play.** This is the survivorship
  restriction spec 6.3 requires be stated rather than silently absorbed.
- **The playing-time coefficient: weight = 1 (unweighted).** Weighting the PT target by
  `target_pt` would be circular -- each observation weighted by the very quantity being predicted
  -- and, worse, would assign weight 0 to every non-survivor and silently delete exactly the
  players whose lost playing time the coefficient exists to learn from. A non-survivor's Y+1 MLB
  playing time is genuinely ~0 and is a real observation, unlike his Y+1 rate. Note this is the
  *target* side; spec 5.4's "absence is not zero" governs the *residual* side, where a player
  absent from the year-Y leaderboard must pass through unfolded rather than acquire a large
  negative residual, and that rule is unaffected.

The fit objective is the same weighted least squares as the evaluation metric, per coefficient.
Fitting one loss and scoring another would make the held-out comparison meaningless.

### A.2 Acceptance is per coefficient

Per spec 6.6, verbatim: a coefficient **passes** if it beats *both* endpoints (`k=0` and `k=1`)
on held-out error on a **majority of the three pairs**, and otherwise **falls back** to whichever
endpoint performed better for that coefficient, recorded as the finding. No global verdict is
issued. Spec 6.3 already predicts one category (`sb_pa`) will misbehave for a known exogenous
reason -- the 2023 rules break -- and instructs against averaging it away.

A coefficient at or near zero is treated as a **suspected setup error** (most likely baseline
contamination per spec 6.1) and investigated before being reported as a result. That is a
diagnostic, not the bar.

### A.3 Gate thresholds

The gate has two distinct jobs (spec 5.4): selecting *training rows*, and deciding *passthrough*
in production. They are different objects and may legitimately differ. Pre-registered here:

| | Fit-sample threshold | Serve-time gate |
|---|---|---|
| Hitters | `PA >= 100` | `PA >= 100` |
| Pitchers | `IP >= 50` | `IP >= 50` |

**Hitters at 100 PA.** 465 / 458 / 454 rows in-year, ~1377 across the three pairs. Raising it to
300 PA costs 40% of the sample (285/pair) and buys nothing: survival at 300 PA is *lower*
(0.696-0.731) than at 100 PA (0.755-0.795), so the stricter gate does not even deliver a
cleaner-surviving sample -- it selects established regulars who have more room to fall off.
100 PA is also the threshold every figure in spec 6.3 is quoted at, which keeps the study
comparable to the spec's own numbers.

**Pitchers at 50 IP.** This is the choice spec 6.3 calls out as the live decision, and it swings
the sample nearly threefold: 344 / 355 / 350 rows in-year at `IP >= 50` (~1049 across three
pairs) versus 139 / 126 / 125 at `IP >= 100` (~390). Each coefficient is fit independently with
two parameters (slope plus a nuisance intercept, see B.1), so each leave-one-pair-out fold trains
on roughly 700 rows at `IP >= 50` versus roughly 260 at `IP >= 100`. Both are nominally
sufficient for two parameters, but 50 IP is chosen because (a) it triples the sample at no
methodological cost, (b) the pitcher pool this feature ranks is a starter pool -- and spec 5.1
already excludes relievers from the out-year ranking for lack of an `SV` column -- so a 50-IP
floor still admits the swingmen and injured starters whose playing-time signal matters most, and
(c) survival is *higher* at 50 IP (0.600-0.628) than at 100 IP (0.561-0.584), so the looser gate
is again the less survivorship-distorted one.

**Pitcher power, reported as spec 6.3 demands.** ~1049 pitcher rows across three pairs at
`IP >= 50` is adequate for six single-slope coefficients fit one at a time. It is *not* adequate
for anything richer -- per-category interaction terms, age curves, or a per-pair refit would each
be underpowered. The study therefore fits one slope per coefficient and nothing more, and that
constraint is a finding, not an accident.

### A.4 Constants chosen rather than fitted (conditioning, spec 6.2 requirement 6)

The coefficients below are reported as **conditional on** these, which are chosen, not fitted:

- **Shrink form:** `w = n / (n + n0)`, where `n` is *realized* year-Y playing time (spec 5.3:
  only realized opportunities carry sampling noise). Bounded in `[0, 1)`, so it can never amplify
  a residual -- spec 5.3's one hard requirement.
- **Shrink constant `n0`, headline:** **200 PA** for hitters, **50 IP** for pitchers. `n0` is the
  playing time at which the observation and the projection get equal weight; 200 PA is about a
  third of a regular's season, and 50 IP is the pitcher analogue. Because `k` and `w` enter the
  model multiplicatively, `k` is only interpretable against a stated `n0`.
- **Sensitivity grid, declared now and reported for sensitivity only** -- never used to select the
  headline number: `n0` in {100, 200, 400} PA and {25, 50, 100} IP.
- **The shrink is not applied to the playing-time coefficient** (`shrunk=False`). Spec 5.3:
  damping the PT residual by a function of the playing time an injury suppressed is circular and
  would make the PT coefficient structurally unable to learn from lost time.

### A.5 What is not pre-registered

The estimator's functional form. Spec 6.2 deliberately leaves it open -- three prose attempts
were each found broken -- so it is chosen against the data and documented in Part B, along with
which of the twelve requirements each design decision serves.

---

## Part B -- Results

Run: `python scripts/keeper_calibration.py`. Raw output in
`data/analysis/keeper_calibration_{hitter,pitcher}.csv`, the sensitivity grid in
`*_n0_sweep.csv`. Every number below is the full-sample fit on the **gated** sample, with
leave-one-pair-out errors from the same gate.

### B.1 The chosen estimator: `ShrunkTransfer`

```
pred = base + k * w * (actual_Y - base)          w = n / (n + n0), n = realized year-Y PT
```

`k` is fit per coefficient by weighted least squares of `target - base` on `w * residual`,
**with an additive nuisance intercept `c` that is fit but never applied**.

*Why an intercept, and why unshipped (requirements 1 and 12).* The intercept absorbs the
pool-wide level offset so it does not leak into the slope. Its production value is **0**,
because `ZiPS_Y` is a projection *for year Y* while the calibration target is year Y+1,
whereas `ZiPS_2027` is already aged forward to the year it is being folded into. `predict`
therefore computes the shipped form only, in held-out evaluation and at serve time alike --
so the comparison against the two endpoints is on identical footing and neither side gets a
free level correction.

*Why additive rather than a scale term.* A free scale `a` on the base rewrites
`a*Z + k*(A - Z)` as `(a - k)*Z + k*A`, which degenerates `k` into the plain OLS slope on
`actual_Y` and destroys the meaning of the `k=0` / `k=1` endpoints. That is the failure that
killed one earlier attempt. With an additive intercept the coefficient on the base stays
pinned at `1 - k*w` by construction, so both endpoints keep their meaning.

Per-player aging is not attempted: no ZiPS vintage carries an Age column (spec 11).
Confidence intervals are the weighted HC1 sandwich, not the classical formula -- year-over-year
baseball residuals are heavy-tailed and heteroskedastic in playing time.

### B.2 Fitted coefficients

**Hitters** -- gate `PA >= 100`, `n0 = 200 PA`. 1241 rate rows, 1377 playing-time rows across
the three pairs.

| coefficient | k (shipped) | 95% CI | ex-2022 | ex-2023 | ex-2024 | spread | beats k=0 | beats k=1 | verdict |
|---|---|---|---|---|---|---|---|---|---|
| `hr_pa`  | **0.494** | [0.403, 0.586] | 0.480 | 0.561 | 0.447 | 0.115 | 3/3 | 3/3 | **pass** |
| `r_pa`   | **0.531** | [0.432, 0.630] | 0.529 | 0.561 | 0.523 | 0.039 | 3/3 | 3/3 | **pass** |
| `rbi_pa` | **0.532** | [0.441, 0.622] | 0.498 | 0.612 | 0.500 | 0.114 | 3/3 | 3/3 | **pass** |
| `sb_pa`  | **0.637** | [0.480, 0.793] | 0.657 | 0.501 | 0.759 | 0.257 | 3/3 | 2/3 | **pass** |
| `h_ab`   | **0.428** | [0.334, 0.522] | 0.416 | 0.476 | 0.400 | 0.076 | 3/3 | 3/3 | **pass** |
| `ab_pa`  | **0.687** | [0.588, 0.785] | 0.661 | 0.666 | 0.743 | 0.082 | 3/3 | 3/3 | **pass** |
| `pa`     | 0.646 | [0.584, 0.708] | 0.624 | 0.614 | 0.702 | 0.088 | 3/3 | 1/3 | **fallback:k=1** |

**Pitchers** -- gate `IP >= 50`, `n0 = 50 IP`. 937 rate rows, 1049 playing-time rows.

| coefficient | k (shipped) | 95% CI | ex-2022 | ex-2023 | ex-2024 | spread | beats k=0 | beats k=1 | verdict |
|---|---|---|---|---|---|---|---|---|---|
| `k_ip`  | 0.970 | [0.866, 1.073] | 0.928 | 1.024 | 0.960 | 0.096 | 3/3 | 0/3 | **fallback:k=1** |
| `w_ip`  | **0.491** | [0.389, 0.593] | 0.459 | 0.515 | 0.493 | 0.056 | 3/3 | 3/3 | **pass** |
| `er_ip` | **0.343** | [0.238, 0.447] | 0.341 | 0.395 | 0.313 | 0.082 | 3/3 | 3/3 | **pass** |
| `bb_ip` | **0.697** | [0.582, 0.812] | 0.718 | 0.715 | 0.673 | 0.045 | 3/3 | 3/3 | **pass** |
| `h_ip`  | **0.385** | [0.281, 0.489] | 0.410 | 0.362 | 0.394 | 0.048 | 3/3 | 3/3 | **pass** |
| `ip`    | **0.631** | [0.517, 0.746] | 0.666 | 0.576 | 0.650 | 0.090 | 3/3 | 3/3 | **pass** |

**The headline answer: `k` is roughly 0.4-0.7, and it is decisively not zero.** **Eleven of the
thirteen** coefficients pass (six hitter rates + PA, five pitcher rates + IP); both fallbacks
land on `k=1`, never on `k=0`. Every coefficient beats the `k=0` endpoint on all three held-out
pairs, without exception. The feature is worth building.

**What `k=0` is and is not.** It is the do-nothing endpoint spec 6.2 requirement 3 names --
ignoring the season entirely. It is **not** what `main` currently ships: since PR #259,
`analysis/keeper_value.py` scales a current-season anchor by a ZiPS ratio and regresses toward
the ZiPS out-year at `DEFAULT_OUT_YEAR_REGRESSION = 0.6`, so the incumbent already carries
roughly 40% of the realized-season signal. **Beating `k=0` therefore does not establish that the
fold beats the shipped estimator.** Increment 1 cannot run that comparison -- spec 9 forbids
importing `fantasy_baseball.analysis` -- so it is scoped to increment 2, alongside the rank-level
check. Treat it as the study's largest unmeasured gap after B.3.

**The full-sample fits are all in range, but the clamp does bind elsewhere.** `k_ip` fits 0.970
full-sample with a CI including 1.0; its **ex-2023 fold fits 1.024** and clamps, which is why
that fold's held-out error is bit-identical to `k=1`. At `n0 = 100` the full-sample `k_ip` fit is
**1.276**, also clamped. Requirement 7's refusal clause never fires on a shipped coefficient, but
`out_of_range` now reports per-fold clamping (`folds_clamped`) rather than inspecting only the
full-sample fit, which is what previously made this invisible.

### B.3 The two fallbacks

**`k_ip` -- immaterial.** The fitted 0.970 loses to `k=1` on all three pairs, but by 0.4%
(mean held-out error 0.020179 versus 0.020107). The reading is that **strikeout rate carries
forward essentially in full**, and the pre-registered rule resolves a tie in the endpoint's
favour. Ship `k=1`.

**`pa` -- material, and the mechanism is understood.** This is requirement 12 in the flesh, and
it corrects a spec assumption.

Spec 6.2 states the playing-time residual has a large *positive* systematic mean, citing +58 PA
for 2025 regulars. **On the actual fit sample the sign is the other way:**

```
                mean ZiPS_Y PT   mean actual_Y PT   mean residual   mean(target - base)   non-survivors
hitters (>=100 PA)      470.0              379.0           -91.0                -142.0           9.9%
pitchers (>=50 IP)       94.1               97.0            +3.0                 -21.0          10.7%
```

ZiPS over-projects playing time by 91 PA on this population, because the ZiPS file is a full
pool and many of its projected regulars never get the plate appearances. The spec's +58 was
measured on a narrower "regulars" population and does not transfer.

The consequence is exactly what requirement 12 predicts. The fitted intercept for `pa` is
**-83.1 PA** (and `c = E[y] - k*E[x] = -142 - 0.646*(-91) = -83.2`, which reproduces it). With
the intercept unshipped, the `pa` prediction is systematically ~83 PA high, and `k=1` wins
out of sample not because the true slope is 1.0 -- the CI [0.584, 0.708] excludes it -- but
because moving further along a negative-mean residual absorbs part of the missing level
correction. The diagnostic is unambiguous:

```
hitter pa, mean held-out error:   k=0  58985    k=1  34852    fitted-k  36519    fitted-k+c  29611
pitcher ip, mean held-out error:  k=0   3335    k=1   3190    fitted-k   3001    fitted-k+c   2477
```

Regenerate with `python scripts/keeper_calibration.py`; the table above and the level figures
come from `data/analysis/keeper_calibration_{hitter,pitcher}_level_term.csv`. The `fitted-k+c`
row is a **diagnostic estimator that is never shipped** -- it exists so this finding's decisive
number is reproducible from the committed script rather than taken on trust.

Applying the intercept beats every shipped option on both player types. It is **not shipped**,
because doing so requires establishing that the level is a persistent ZiPS playing-time hedge
(which `ZiPS_2027` would carry too) rather than the year-Y-to-Y+1 aging and attrition gap
(which `ZiPS_2027` has already priced in). This study cannot separate the two -- that would need
a vintage pair where the base is aged forward, which does not exist on disk. **This is the
single largest open item handed to increment 2**, and it is worth taking: it is a 19% error
reduction on hitter PA and a 17% reduction on pitcher IP, and PA multiplies every counting stat.

Interim: ship `k=1` for `pa` and `k=0.631` for `ip` per the pre-registered rule.

### B.4 Stability, and the stolen-base rules break

`sb_pa` is the least stable coefficient by every measure: the widest per-fold spread (0.257
versus 0.039-0.115 for the other hitter rates), the widest CI ([0.480, 0.793]), and the only
rate coefficient that fails to beat `k=1` on all three pairs. That is consistent with spec
6.3's warning about MLB's 2023 rules package.

**But the three-pair sample cannot cleanly attribute it.** The pair that spans the break is
2022->2023. Folds that *include* it in training produce both the lowest fit (0.501, training on
2022+2024) and the highest (0.759, training on 2022+2023), so the instability is not a clean
level shift the fold structure can isolate. Per spec 6.3 it is reported rather than averaged
away: **`sb_pa = 0.637` is provisional**, carries the widest interval of any coefficient, and
should be re-fit first when the 2025->2026 pair opens after the 2026 season.

Every other coefficient is stable: per-fold spreads of 0.04-0.12, with all three folds inside
the full-sample CI.

### B.5 Conditioning on `n0` (requirement 6)

The pre-registered sensitivity grid, hitters:

```
n0     hr_pa   r_pa  rbi_pa  sb_pa   h_ab  ab_pa      mean held-out error (hr_pa)
100    0.407  0.437   0.439  0.535  0.348  0.574      0.000133
200    0.494  0.531   0.532  0.637  0.428  0.687      0.000133
400    0.655  0.701   0.701  0.824  0.574  0.895      0.000133
```

**`k` scales almost exactly inversely with the shrink, and the held-out error does not move.**
That is the honest statement of what this study identifies: **the product `k * w` is identified;
`k` alone is not.** Every coefficient above is meaningless without the stated `n0` beside it.
Verdicts are stable across the whole grid for all eleven rate coefficients and both PT
coefficients, with one exception: pitcher `k_ip` passes at `n0 = 25` (k = 0.806) and falls back
to `k=1` at 50 and 100 -- the same tie described in B.3, resolved differently by a hair.

Production must therefore use `n0 = 200 PA` / `n0 = 50 IP` with these coefficients, or refit.

### B.6 Survivorship (requirement 5)

Measured on the fit sample, in Part A. The treatment differs by coefficient, as spec 6.3
requires, and both halves are stated:

- **Rate coefficients are fit and scored on survivors only** (1241 of 1377 gated hitter rows,
  90.1%; 937 of 1049 pitcher rows, 89.4%). A non-survivor has no year-Y+1 rate at all, so there
  is nothing to fit against. These coefficients are therefore **conditional on continued play**,
  and would be biased upward if read as unconditional persistence.
- **The playing-time coefficients keep every non-survivor**, scored unweighted, with a target of
  0. That is why `pa` and `ip` have larger `n` than the rates in the same table. The 9.9% /
  10.7% of players who drop out of MLB entirely are the single largest source of playing-time
  error, and deleting them would have inflated the PT coefficient exactly as spec 6.3 warns.

### B.7 The gate discontinuity, and the ramp (requirement 9)

Because the playing-time term is unshrunk, the hard gate is a step. Quantified with the shipped
coefficients, for a regular lost to a May injury (`ZiPS_2026 = 400 PA`, `ZiPS_2027 = 380 PA`):

| realized 2026 PA | folded 2027 PA | |
|---|---|---|
| 99 (below gate) | 380 | unfolded passthrough |
| 101 (above gate) | 81 | `380 + 1.0 * (101 - 400)` |

**a 78.7% drop across two plate appearances** -- larger than the spec's 59% illustration,
because the shipped `pa` coefficient is 1.0 rather than 0.8. The pitcher analogue
(`ZiPS_2026 = 150 IP`, `ZiPS_2027 = 140 IP`, `k = 0.631`) is 140 IP at 49 IP realized versus
77.5 IP at 51 IP: a 44.6% drop.

That is too large to justify, so **a ramp is specified**, implemented as
`fold.gate_ramp(realized_pt, threshold, width)`: the fold weight ramps linearly from 0 at the
threshold to 1 at `threshold + width`. Chosen widths: **100 PA for hitters, 50 IP for pitchers**
-- i.e. full folding begins at twice the gate, and both ends of the ramp sit inside the fit
sample's support. The same example becomes 377.0 PA at 101 PA realized, 255 PA at 150, and
180 PA at 200, with no step anywhere. Below the threshold nothing changes: passthrough is
unaffected, and absence from the leaderboard still resolves to no fold at all.

`gate_mask` is retained and is what the fit sample uses -- `calibration.gated` calls it, so
there is one definition of the threshold rule and the fit sample and the serve path cannot
drift apart on what "enough playing time" means. A hard threshold is correct there because it
selects training rows rather than deciding a production value. **Increment 2 must apply
`gate_ramp`, not `gate_mask`, on the serve path.**

### B.8 The train/serve gap (requirement 8)

The coefficients are fit on a fully-realized `actual_Y`. Production applies them to a 2026 line
that is roughly 35% unrealized rest-of-season projection (spec 6.4, as of 2026-07-27). Analytic
attenuation of the *applied* move relative to the calibrated one:

- **Rate coefficients: ~0.57x.** Two compounding effects. The minuend blends in a ROS projection
  that carries little surprise of its own, diluting the residual to roughly the realized share
  (~0.65). And the shrink is computed on realized playing time, so a 600-PA-pace hitter sits at
  ~390 PA with `w = 0.661` rather than the calibration's `w = 0.750`, a further 0.88x. Product
  ~0.573.
- **Playing-time coefficients: ~0.65x.** Unshrunk, so only the dilution applies.

Both are **lower bounds on the transmitted signal**: FanGraphs ROS projections have themselves
been updated with 2026 performance, so the ROS portion is not literally surprise-free. The
attenuation shrinks weekly toward 1.0 as the season completes, while `k` stays frozen -- so the
metric drifts, and the drift is toward the calibrated behaviour rather than away from it.

Residual uncertainty that this study cannot remove: a re-fit with year-Y actuals truncated to a
comparable season fraction is infeasible (`rest_of_season/` exists only for 2026, FanGraphs does
not archive historical mid-season ROS files, and `game_logs` in `data/local.db` are 2026-only).
Separately, spec 6.4 notes that using a ZiPS-only 2026 minuend removes the 5-system blend offset
but covers only ~1,357 of 3,739 pool rows; increment 2 must state what happens to the uncovered
rows.

### B.9 Requirement checklist

| # | Requirement | Where |
|---|---|---|
| 1 | Same functional form in calibration and production | B.1 -- intercept fit, never applied, production value 0 |
| 2 | Out-of-sample, per-held-out-pair error | B.2, and the `err_<year>` columns in the CSVs |
| 3 | Compared against both endpoints | B.2 -- beats `k=0` 3/3 everywhere; two `k=1` fallbacks |
| 4 | Stability reported per category | B.4 -- per-fold fits and spreads |
| 5 | Survivorship handled explicitly | B.6, measured in A |
| 6 | Conditioning stated | B.5 -- `k*w` identified, `k` alone is not |
| 7 | Amplification bounded or refused | B.2 -- clamp to [0,1]; nothing out of range |
| 8 | Train/serve gap stated | B.8 -- ~0.57x rates, ~0.65x playing time |
| 9 | Gate discontinuity quantified; thresholds chosen | B.7 (78.7% / 44.6%, ramp specified); A.3 |
| 10 | Shrink form and constants chosen | A.4, conditioned in B.5 |
| 11 | Metric pinned before the first fit | A.1, committed before any fitting code |
| 12 | Systematic component of the PT residual treated | B.3 -- the sharpest finding in the study |

### B.10 What increment 2 inherits

1. **Ship the coefficients in B.2 with `n0 = 200 PA` / `n0 = 50 IP`.** They are not portable to
   another shrink constant. They are available in code as
   `keepers.coefficients.POLICIES[player_type]` -- a `FoldPolicy` carrying the per-column `k`
   alongside the `n0`, gate, ramp width and playing-time column they are conditional on. The
   whole policy is re-derived from the study CSVs by `coefficients.policy_from_study` and
   checked against the shipped constants in `tests/test_keepers/test_coefficients.py`, so a
   re-run that moves the numbers fails the suite rather than drifting silently. Import them;
   do not retype the table above.
2. **Compose the fold weights with `FoldPolicy.serve_weights`, never by hand** (B.7). It is the
   only place the three rules become mechanical rather than remembered:
   `gate_ramp` not `gate_mask`; rates shrunk but playing time **not** (spec 5.3); and the
   shrink using the policy's own `n0`. Then:

   ```python
   policy = POLICIES["hitter"]
   folded = fold_rates(base, residual, policy.serve_weights(realized_pt), policy.coefficients)
   ```

   `fold_rates` takes per-column weights *and* per-column coefficients because the study
   calibrated both per column. Passing a single weight for all thirteen reproduces neither the
   model nor the numbers in B.2 -- it folds playing time at the rate weight and silently
   attenuates the move (a 500-PA base with a -100 PA residual lands on 434 instead of 400).
3. **Resolve the playing-time level term** (B.3). Worth ~19% of hitter PA error, and PA
   multiplies every counting stat. Requires deciding whether the -83 PA level is a persistent
   ZiPS hedge or an aging gap `ZiPS_2027` has already priced.
4. **Re-fit `sb_pa` first** when the 2025->2026 pair opens after the 2026 season (B.4).
5. **Check at the rank level, and against the incumbent.** Spec 6.6 is explicit that this bar is
   measured on the rate/PT scale while the feature consumes VAR *rank*; rate error can improve
   while ranking degrades. Separately, B.2 notes the endpoints do not include the estimator
   `main` actually ships (`analysis/keeper_value.py`, anchor x ratio regressed at 0.6). Both
   checks need VAR, so both are increment 2's.
7. **Playing time is scored without saves.** `fold.reconstruct_pitcher` emits no `sv` -- spec 5.1
   excludes relievers from the out-year ranking because ZiPS populates `SV` in 0 of 1838 rows --
   but `analysis.keeper_value.PITCHER_FIELDS` scores `sv`. A missing column reads as zero saves,
   so increment 2 must supply it or exclude the category explicitly.
6. Spec 5.1's `role_ip` routing fix, the par curve, and the cross-team table remain increment 2.
