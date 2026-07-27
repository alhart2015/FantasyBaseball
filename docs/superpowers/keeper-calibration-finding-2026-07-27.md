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

*(To be completed after the fits are run. Nothing above this line may be amended after the first
fit without a separate, dated commit saying so.)*
