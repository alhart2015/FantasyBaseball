# Keeper value: feature definition and 2026 fold-in design

Issue: #266 (depends on #265, closed by PR #267)
Date: 2026-07-27
Status: design approved; hardened through two adversarial review rounds

**Supersedes** `2026-07-22-keeper-value-design.md` and
`2026-07-23-keeper-value-current-anchor-design.md` in full. Those describe the
discounted multi-year metric this design replaces (section 3) and are historical.
`2026-07-23-keeper-trade-generator-design.md` remains live and untouched.

## 1. The decision this serves

**Every team keeps exactly 3 players. Keeping is mandatory.** The 3 keepers consume
draft rounds 1-3, so the live 2027 draft begins at round 4. No escalating cost, no
salary, and the choice is re-made every offseason.

Because a team forfeits rounds 1-3 *regardless of whom it keeps*, the cost of keeping is
constant across every possible choice of 3 and cannot influence which 3 to keep. The
feature's work is therefore:

1. Projecting 2027 value well, from a baseline that predates the 2026 season.
2. Measuring the surplus a trio generates over the picks it costs.
3. Comparing the owner's trio against the other nine teams'.

## 2. What "keeper value" means

**Absolute keeper value.** Projected 2027 VAR. This alone answers "which 3 do I keep":
take the top 3.

**Known limitation, stated up front.** VAR is additive and blind to team context. This
is a roto league (`draft.scoring_mode: deltaroto_immediate`), where marginal category
value saturates -- three 40-SB players are worth less than the sum of their VARs -- and
nothing in a VAR sort prevents a positionally redundant trio. So "take the top 3" is a
strong default, not a proof of optimality. Mitigation: the output carries a **category
profile** for the selected trio (per-category SGP contribution) so saturation and
redundancy are visible to the owner. Replacing the sort with a delta-roto trio search is
a candidate later increment, deliberately not attempted here.

**Surplus over forfeited picks.** For a trio ranked 1st/2nd/3rd by absolute value, the
i-th keeper's surplus is `VAR_i - par_i`, where `par_i` is the expected VAR at the
owner's round-i pick had no keeper system existed. `par_i` is computed against the
**full, undepleted** 2027 pool: since keeping is mandatory the counterfactual is not
"what if I released him" but "what would rounds 1-3 have returned in a draft with no
keepers," and in that world no keeper set is removed. So `par_i` depends only on the
pool and the pick slot.

**The 2027 draft slot is not known** and the spec must not pretend otherwise.
`config/league.yaml`'s `draft.position: 8` is the **2026** slot, and 2027 order depends
on 2026 final standings that do not exist yet. Par is therefore computed at a **stated
assumed slot, defaulting to position 8**, and the output must label it as an assumption.
At position 8 in a 10-team snake the ordinals are picks **8, 13, 28** (R1 forward: 8;
R2 reversed: 11 + (10-8) = 13; R3 forward: 20 + 8 = 28). The report should show surplus
at a small range of plausible slots rather than a single number.

**Surplus is a diagnostic, not a decision rule** -- no keep/release decision exists under
mandatory keeping. Two further caveats, both required in the output:

- **Surplus is not field-relative.** `par_i` *rises* with a better draft slot, so holding
  trio quality fixed, surplus is larger for a team that picks late -- which in a snake
  league means the team that finished worse. Surplus measures "how much more than my own
  forfeited picks," never "better than the field." Cross-team comparison uses absolute
  value (section 7).
- **`par_i` assumes a draft in exact descending-VAR order.** Real drafts have positional
  runs and reaches, so VAR-rank ordinals overstate par and therefore understate surplus,
  systematically. `draft_value.ParCurve` exists precisely because real picks differ from
  VAR order; it is backward-looking and not reusable here, but its existence is the
  warning.

**Trajectory.** A separate 2028 column: "will he still be keep-worthy next year." Never
blended into the headline.

## 3. Horizon: single-year, and why

The metric is 2027. 2028 appears only as the unblended trajectory signal.

This breaks from `analysis/keeper_value.py`, which sums a discounted 3-year stream
(`DEFAULT_DISCOUNT = 0.80`, `DEFAULT_HORIZON = 3`, lines 28-29). That prices a multi-year
commitment the rules do not impose: the 2028 slot is re-chosen in 2028 with 2028
information. It also has a concrete defect -- with `base_year = 2026`, the largest weight
in the sum (1.0, ~41% of the total) lands on a season that is finished and immutable when
the decision is made. And it is epistemically backwards: with a pre-2026 baseline the
out-years are the least reliable inputs, so a discounted sum adds most noise where
confidence is lowest.

The young-player counter-case resolves without a sum. If a 21-year-old's 2027 line is
strong, single-year VAR captures him. If it is weak, he is inexpensive in the 2027 draft,
so a keeper slot -- an asset worth a round-1 through round-3 pick -- is the wrong
instrument.

## 4. The core problem: the out-year baseline is permanently stale

ZiPS 2027 and 2028 were generated 2026-03-25 and know nothing of 2026.

**Provenance, since the whole feature rests on it.** The out-year *pitcher* files on disk
were downloaded 2026-07-27 (this session) and installed under the March naming
convention; the *hitter* files were already present from March. The 2026-07-27 hitter
download matched the March export on all 1827 PlayerIds present in both, HR identical on
every one. Of 1901 rows, 74 did not join -- so the accurate claim is "identical on the
overlap," not "byte-identical," and whether those 74 are pool changes or a join artifact
is unresolved. The July hitter artifact was not retained, so that specific comparison is
not currently reproducible (section 13 requires scripting it).

The conclusion holds and is now verified on freshly downloaded pitcher files too:
FanGraphs has not regenerated ZiPS out-year projections, and re-downloading will not fold
2026 in.

## 5. The fold-in mechanism

### 5.1 Scope of the out-year valuation

**Relief pitchers are excluded from out-year ranking entirely.**

ZiPS 2027 and 2028 contain no saves data: `SV` is populated in **0 of 1838** rows in both
out-year pitcher files, against 1838/1838 (sum 1064) in ZiPS 2026. `HLD`/`QS`/`BS`
likewise; those are unscored. The failure is silent -- `safe_float`
(`utils/constants.py:372-377`) coerces NaN to 0.0 without error, and `sv` is present as
an all-empty column so the parser does not raise.

Scoring relievers without saves does not merely understate them; it **inverts the
ordering within the RP population**. Measured with the SV-free RP floor applied: an elite
closer goes +4.202 to +0.345 VAR (-92%), a mid closer +0.531 to -1.469, while a 3-save
setup man *gains* 0.714 (he sheds little save value but his floor drops 1.143). A ranking
built on that is not biased, it is wrong.

So: **out-year valuation and ranking cover hitters and starting pitchers only.** Relievers
are computed and reported in a separate not-comparable list, never ranked against hitters
or starters, and never placed in a projected trio. Where an opponent's actual keeper is a
reliever, the cross-team table flags that trio as incomplete rather than silently ranking
it.

Two consequences to accept and disclose:

- A genuine closer keeper cannot be evaluated by this feature. That is the cost of the
  missing data.
- At least one opponent trio is affected today (Pete Fairbanks, Send in the Cavalli).

**Implementation note: this requires no change to the scoring path.** Because relievers
are not ranked, SV need not be removed from the denominators or from the replacement
line. Both would have been infeasible as previously specified: `calculate_player_sgp`
indexes `denoms[Category.SV]` unconditionally (`sgp/player_value.py:82`, `:124`),
`get_sgp_denominators` offers override but never removal, and `_empirical_pitcher_floor`
(`sgp/replacement.py:218`) reads the module-global `REPLACEMENT_BY_POSITION` with no
injection seam. Instead: leave the ten-category path intact and let SV be 0 for the
hitters and starters that remain, which is accurate for them.

**SP/RP classification comes from the ZiPS out-year file itself**, not from position
eligibility: a pitcher is a reliever when `GS / G < 0.5` (both columns are present in the
out-year exports). This avoids depending on `cache:positions`, which is thin (section
5.6).

### 5.2 Decompose, then fold

Counting stats are products of a rate and playing time, and folding them raw conflates
"hit for less power" with "played half a season" -- different information with different
persistence. Of the ten scored categories (`utils/constants.py:13-22`), only AVG/ERA/WHIP
are rates; R/HR/RBI/SB/W/K/SV are counting; AB and IP are the scoring weights (PA is not
scored at all).

```
rate_2027     = ZiPS_2027_rate + k_rate * w * (rate_2026 - ZiPS_2026_rate)
pt_2027       = ZiPS_2027_pt   + k_pt       * (pt_2026   - ZiPS_2026_pt)
counting_2027 = rate_2027 * pt_2027
```

**Rates folded (one `k` each):** hitters HR/PA, R/PA, RBI/PA, SB/PA, H/AB, AB/PA;
pitchers K/IP, W/IP, ER/IP, BB/IP, H_allowed/IP.

**Playing time folded (one `k` per player type):** PA for hitters, IP for pitchers. AB is
derived as `PA_2027 * (AB/PA)_2027` rather than folded independently, which makes
`AB <= PA` structural instead of a constraint to police. `pt_2027` is clamped at `>= 0`.

**AVG/ERA/WHIP are recomputed from folded components**, never folded directly:
`AVG = H/AB`, `ERA = 9*ER/IP`, `WHIP = (BB + H_allowed)/IP`. BB and H_allowed are folded
**separately**, not as `(BB+H)/IP` -- `calculate_replacement_rates`
(`sgp/replacement.py:122-124`) needs them as distinct columns, as do `PITCHER_PROJ_KEYS`
and `PITCHER_CORR_STATS`.

*Honest note on why:* for the rates themselves this decomposition is algebraically
equivalent to folding the rate directly -- `H_2027/AB_2027 = (H/AB)_2027` because the
playing time cancels. The decomposition earns its keep on the seven **counting**
categories, where `count = rate * PT` and folding the count raw is genuinely wrong. It is
specified this way because the counting stats need the components anyway, so one
mechanism serves both.

Data availability is verified on both sides: `cache:full_season_projections` exposes
`h`/`ab`/`pa` and `er`/`bb`/`h_allowed`/`ip` at 100%, and `HITTING_COLUMN_MAP` /
`PITCHING_COLUMN_MAP` parse `H`/`AB` and `ER`/`BB`/`H`->`h_allowed` from every ZiPS
vintage including the out-years.

### 5.3 The sample-size shrink `w`

`w` applies to the **rate** residual only:

```
w = n / (n + n0)      n0 = 200 PA (hitters), 50 IP (pitchers)
```

Bounded in [0, 1), so it can never amplify. (A linear `PA/600` form was rejected: it
exceeds 1.0 above 600 PA, amplifying exactly where amplification is least warranted.)
`n0` is a stated default, not a finding, so **the fitted `k` is conditional on it** and
section 6.7 requires saying so.

`n` is the **full-season (YTD + ROS) playing time**, matching the minuend, so `w` sits at
its calibrated level rather than drifting weekly as the observed share grows.

Applying `w` to the playing-time residual would be a double-count -- damping an injury
signal in proportion to the very playing time the injury suppressed -- and would make
`k_pt` structurally unable to learn from players who lost time, which is the information a
keeper decision most needs. It is deliberately excluded there.

### 5.4 Gating: who gets folded at all

**The fold applies only to players with at least `n_min` realized 2026 MLB playing time
(default 50 PA / 15 IP). Everyone else passes through at raw ZiPS 2027, flagged
`unfolded`.**

This gate is load-bearing and replaces an earlier formulation that was wrong. Because `w`
does not apply to the PT residual (5.3), a player with zero MLB time would otherwise get
`pt_2027 = ZiPS_2027_pt + k_pt * (0 - ZiPS_2026_pt)` -- for a prospect projected at 300 PA
who never debuted and `k_pt = 0.8`, that is `350 - 240 = 110 PA`, gutting his line.
That would damage precisely the young high-upside case section 3 relies on.

Related and equally important: **absence from the MLB actuals is not zero playing time.**
`playerPool=all` returns MLB players only, so a player who spent 2026 in AAA is *missing
from the data*, not present with 0 PA. Absence must resolve to `unfolded` passthrough, not
to a large negative residual. The implementation must distinguish the two cases
explicitly.

A regular lost to a March injury also lands below the gate and passes through with his
stale 2027 line. That is a deliberate choice, not a clean fall-through, and the output
flags it.

### 5.5 Other edge cases

- **`0/0` rates.** A player with zero realized PT has an undefined rate (`NaN`), and
  `0 * NaN = NaN` in IEEE-754 -- so a `w = 0` multiply does **not** rescue it, and
  `safe_float` would later coerce the poisoned value to a silent 0.0. The 5.4 gate
  prevents this for the zero case; implementations must still guard `NaN` explicitly
  rather than relying on `w`.
- **In ZiPS 2027 but absent from ZiPS 2026: 95 of 1838 pitchers (5.2%).** No baseline, so
  no residual. Passthrough at raw ZiPS 2027, flagged `unfolded`. All 1901 hitters have a
  2026 counterpart.
- **Absent from ZiPS 2027 entirely** (2027 debutants): no projection. Reported separately
  as unrankable, never scored zero.
- **Two-way players.** Ohtani appears twice in the 2027 pool (once per type). Dedupe by
  MLBAMID for **both** trio selection and the par ordinal, counting one keeper slot.
  Exactly one MLBAMID currently appears in both files, so the practical risk is small
  today, but the rule must be written for the pool, not for Ohtani.
- The same `k`, `w`, gate and exclusions apply to ZiPS 2028 for trajectory. This is an
  approximation -- `k` is calibrated one-year-forward -- acceptable only because
  trajectory never enters the sort.

### 5.6 Joins and eligibility

Joins are by ID wherever an ID exists: `PlayerId` across ZiPS vintages; `MLBAMID` to
`player.id` for MLB actuals; `fg_id`/`mlbam_id` for `cache:full_season_projections`
(`fg_id` 100%; `mlbam_id` null on 1 hitter and 1 pitcher). **Normalized name matching is
required for the `config/league.yaml` keeper list** -- exact-name joining drops 3 of 30
(`Jose Ramirez`, `Julio Rodriguez`, `Ronald Acuna Jr.` vs ZiPS's accented spellings). Use
`draft/keepers.find_keeper_match` and `sgp/rankings.rank_key`.

**Position eligibility is the weak join and must be disclosed.** `calculate_var` routes on
`player["positions"]`, whose only source is `cache:positions` -- **706 entries, keyed by
normalized name**, against a 2027 pool of 3,739 rows. Roughly 81% have no eligibility
record; unmatched pitchers default to `SP` and unmatched hitters fall to the UTIL floor,
both of which move VAR. Since section 2's par construction ranks the *full* pool, this is
material. Requirements: rows using a defaulted position are flagged; the report states the
coverage rate; and par is computed both over the full pool and over the
resolvable-position subset so the sensitivity is visible. (The SP/RP split does **not**
depend on this -- it comes from `GS/G`, per 5.1.)

## 6. Calibrating k (increment 1)

### 6.1 The baseline must not already know year Y

Every ZiPS file on disk is a preseason projection for its own year, so `ZiPS_{Y+1}` was
built knowing year Y. Using it as the base would fit how much surprise ZiPS has *already
absorbed*, driving `k` toward zero by construction, and would ship the conclusion that
2026 tells us nothing about 2027 -- the opposite of the truth, since ZiPS 2027 has never
seen 2026.

The base is therefore `ZiPS_Y`, built knowing only through Y-1. Information sets align:
neither `ZiPS_Y` nor `ZiPS_2027` has absorbed the season whose surprise is being folded.

### 6.2 The fit: the base coefficient is pinned at 1

```
predicted_{Y+1} = norm(ZiPS_Y) + k * w * (actual_Y - ZiPS_Y)      [rates]
predicted_{Y+1} = norm(ZiPS_Y) + k     * (actual_Y - ZiPS_Y)      [playing time]
```

**The base coefficient is 1, identical to production (5.2).** An earlier draft fitted a
free scale term `a` on the base. That is rejected: production has no `a`, and dropping a
jointly-estimated `a` does not preserve `k`, since `a*Z + k*(A-Z) = (a-k)*Z + k*A` -- with
`a` free, `k` is merely the OLS coefficient on `actual_Y`, the endpoints `k=0` / `k=1`
lose their meaning, and section 6.7's diagnostic breaks because `a` can absorb the signal.
Calibration and production must fit and apply the same model.

`norm()` handles the one genuine gap: `ZiPS_Y` targets year Y while the target is Y+1, so
it is one year under-aged, whereas `ZiPS_2027` is already aged to 2027. `norm` is a
**single league-level scale factor per category**, computed from pool aggregates so that
`norm(ZiPS_Y)` matches the year-(Y+1) pool level. It is deliberately *not* per-player: a
per-player ratio would smuggle individual year-Y information back into the base
(reintroducing the 6.1 contamination), is unavailable for 57% of the 2022 cohort (650 of
1504 appear in both 2026 and 2027), and is unobtainable by age in any case -- **ZiPS CSVs
carry no Age column in any of the seven vintages on disk**. A league-level scalar leaks no
individual signal. The study must report `k`'s sensitivity to `norm`.

**Fit specification** (previously unstated, and every item changes the answer):

- Loss: mean squared error, **weighted by playing time** (PA for hitter rates, IP for
  pitcher rates) so a 20-PA player's HR/PA does not dominate the residual variance.
- No intercept; the base term is the offset.
- Fit in-sample on two pairs, evaluate on the held-out third (6.3). "Out of sample" refers
  to evaluation, not fitting.
- Fit sample: players clearing the 5.4 gate in year Y **and** with a defined `actual_{Y+1}`.
- `k` is unconstrained in the fit but **must be reported with its confidence interval, and
  any `k` outside [0, 1] flagged** -- `k * w` is otherwise unbounded above and would
  amplify residuals in production.

### 6.3 Three year-pairs, not four

The fit needs `ZiPS_Y`, `actual_Y`, `actual_{Y+1}`. A 2025 pair needs a complete 2026
season; `season_end` is 2026-09-28 and today is 2026-07-27. A 2021 pair is impossible
because `data/projections/` begins at 2022. So:

**2022->2023, 2023->2024, 2024->2025. Three.**

Leave-one-pair-out across the three, reporting per-held-out-pair error, the fitted `k` per
rate and per PT coefficient, comparison against `k=0` and `k=1`, and stability across
pairs. A fourth pair becomes available after the 2026 season; the study must be re-runnable
then.

**One pair spans a known structural break.** MLB's 2023 rules package (pitch clock, larger
bases, pickoff limits) raised league stolen bases sharply, so `SB/PA` has a level shift
inside a three-pair sample. Per-category stability must be reported, and an unstable
`k_SB` attributed to the break rather than averaged away.

### 6.4 Survivorship must be handled, and measured on the right sample

`actual_{Y+1}` exists only for players who kept playing. Of players with >=100 PA in year
Y, 75.5% / 77.7% / 79.5% reach >=100 PA in Y+1 across the three pairs (114, 102, 93
players lost). Fitting on survivors alone measures persistence *given continued play*,
biasing `k_pt` upward -- the coefficient most in need of honesty.

**Those rates were measured on the full MLB >=100 PA population (~460 per season), not on
the ZiPS-matched calibration sample (~350).** The ~110-player difference is
disproportionately callups and marginals, whose survival behavior differs. The study must
**re-measure survivorship on the actual fit sample** before using it.

Handling differs by coefficient and the earlier "report it both ways" was ill-posed:

- **`k_pt`:** include non-survivors at their actual (near-zero) Y+1 playing time. This is
  well defined and is the honest fit. Report survivors-only alongside it; the gap is a
  finding.
- **`k_rate`:** a non-survivor has no meaningful Y+1 rate, so inclusion is undefined. Fit
  on survivors and **state the restriction** rather than manufacturing a target.

Sample size must be reported: roughly 350 matched hitters per pair, ~1050 across three
pairs, before any per-category split.

### 6.5 The train/serve gap

`k` is fit on `actual_Y - ZiPS_Y` (fully observed, ZiPS vs ZiPS) but applied to
`full_season_2026 - ZiPS_2026`, which differs twice over:

- The minuend is ~34.9% unrealized rest-of-season projection across the 218 rostered
  players, so the applied residual is attenuated -- and that share shrinks weekly, drifting
  the metric with `k` frozen.
- The minuend is a 5-system blend (steamer/zips/atc/the-bat-x/oopsy at 0.20 each), so
  subtracting a ZiPS-only baseline injects inter-system level offsets as spurious signal.

**Required mitigation (feasible):** production uses the **ZiPS-only** 2026 full-season line
as the minuend, not the 5-system blend. Those files exist
(`data/projections/2026/rest_of_season/2026-07-27/zips-*.csv`), and this removes the second
mismatch entirely and matches the calibration's ZiPS-vs-ZiPS form.

**Deferred (infeasible):** a full re-fit with year-Y actuals truncated to a comparable
season fraction and blended forward. This was previously a required acceptance item and is
withdrawn as such: `rest_of_season/` exists only for 2026, historical mid-season ROS
projections are not archived by FanGraphs, and `game_logs` in `data/local.db` are 2026-only.
Reconstructing it would violate section 9's "increment 1 is standalone." Instead the study
reports the *analytic* attenuation implied by the current ROS share, and the residual
uncertainty is disclosed.

### 6.6 Data-ingest requirements

- **IP arrives as a baseball-notation string.** `stat.inningsPitched` returns `"1.2"` for
  1 2/3 innings. Naive `float()` gives 1.2. The error is small on a season total (bounded
  ~0.47 IP, ~0.3%) but the conversion is free and the units must match ZiPS's decimal IP.
  `stat.era` and `stat.whip` are also strings needing coercion with a zero-IP guard.
- **Assert page completeness, not exact row counts.** Hitting returns roughly 794 / 769 /
  742 / 765 rows for 2022-2025 at `playerPool=all`, but MLB revises historical rosters, so
  exact-equality assertions produce false failures. Assert a lower bound plus the actual
  failure mode: `_fetch_mlb_season` breaks on the first short page
  (`keepers/mlb_stats.py:51-52`), making silent truncation indistinguishable from a
  complete pull.
- **Coverage is limited and must be stated.** Against ZiPS 2023's 1716 hitters at most
  ~45% can match; the matched sample is ~500-530, or ~350 above the 100 PA threshold.
- **`fetch_or_cache` never invalidates** (`keepers/cache.py:23-38`). Correct for completed
  seasons. **In-progress seasons must not use it**, or must use a date-stamped path, or the
  first 2026 pull freezes permanently.

### 6.7 Acceptance criteria for increment 1

Complete when it reports, per folded rate and per PT coefficient:

1. Fitted `k` with confidence interval, flagged if outside [0, 1], with leave-one-pair-out
   PT-weighted MSE against the `k=0` and `k=1` baselines.
2. Stability of `k` across the three pairs, with `SB` interpreted against the 2023 break.
3. Survivorship re-measured on the fit sample, with the `k_pt` survivors-only comparison
   and the stated `k_rate` restriction (6.4).
4. Explicit statement that `k` is conditional on the chosen `n0` (5.3) and on `norm` (6.2),
   with `k`'s sensitivity to `norm`.
5. The analytic train/serve attenuation estimate (6.5).

**Passes** if held-out PT-weighted MSE at the fitted `k` beats both `k=0` and `k=1` on a
majority of held-out pairs. **Falls back** otherwise, shipping whichever endpoint performed
best, recorded as the finding.

Note the limit of this bar: it is measured on the rate/PT scale, while the feature consumes
VAR *rank*. Rate error can improve while ranking degrades. VAR is not built until increment
2, so a rank-level check is scoped there, and the human review point between increments
exists partly to catch this.

A fitted `k` near zero is a suspected setup error (most likely baseline contamination per
6.1) and must be investigated before being reported as a result. That is a diagnostic, not
the bar.

## 7. Cross-team comparison

Because "which 3" is a sort under mandatory keeping, cross-team comparison carries most of
the feature's value.

Universe: `cache:roster` plus `cache:opp_rosters` (nine opponents, 23-25 players each,
218 total, kept current by the refresh). Each team's projected trio is its top 3 by
absolute 2027 value among **hitters and starting pitchers**, deduped by MLBAMID. Trios
containing a reliever are flagged incomplete (5.1).

**Trios are compared on absolute 2027 VAR, not surplus** -- per-team surplus would need
each team's 2027 slot, which does not exist yet, and surplus is not field-relative anyway
(section 2). The owner's own surplus is reported separately at the assumed slot.

No circularity: `par_i` is computed against the full undepleted pool, so it does not depend
on which players are kept, and trio selection depends only on absolute value. A single pass.
**This guarantee is conditional on section 13's first open question resolving as "no."** If
positional floors were made a function of keeper depletion, floors would depend on which 30
are kept, which depend on VAR, which depends on floors -- a genuine fixed point. The answer
is expected to be "no" (see section 13), but the dependency must be stated.

## 8. Timing caveat

Rosters are live and current; the refresh keeps them so. The caveat is the calendar.
Keeper eligibility runs off the season-end roster and it is July, so trades and waiver
claims will change who is eligible. The ranking stays accurate as rosters move, but "my
projected trio" is provisional until rosters freeze. The 2026 residual also sharpens as the
rest-of-season share (6.5) shrinks toward zero.

## 9. Increments and delivery

**Increment 1: the calibration study.** Standalone -- ZiPS vintages and MLB actuals only;
no SGP, no VAR, no board.

- Delivery: a script under `scripts/`, results table to `data/analysis/`.
- Tests: IP notation conversion (6.6); `w` including zero-PT and high-PT bounds (5.3); the
  5.4 gate, covering both zero-PT and absent-from-actuals; rate/PT decomposition and
  recombination including the `AB <= PA` derivation and the `pt >= 0` clamp (5.2); `NaN`
  guarding (5.5); and the fit recovering a known planted `k` on synthetic data.
- Acceptance: section 6.7.

Review point: examine the findings and decide where to go next.

**Increment 2: the value pipeline and outputs.** Builds `updated_2027`/`updated_2028`,
scores through the existing SGP and VAR path (unmodified, per 5.1), constructs `par_1..3`
at the assumed slot, and emits the outputs.

Required output fields, so the shape is not deferred along with the surface: `player_id`,
`name`, `player_type`, `role` (SP/RP), `team`, `absolute_var_2027`, `surplus` (owner only),
`trajectory_var_2028`, `category_profile` (trio members), and the flags `unfolded`,
`unrankable`, `reliever_not_comparable`, `defaulted_position`, `injury_passthrough`. Surface
(CLI vs dashboard vs cache key), persistence, and the fate of `scripts/keeper_value.py` are
decided at the start of increment 2.

## 10. Relationship to existing code

| Path | Size | Disposition |
|---|---|---|
| `analysis/keeper_value.py` | 360 LOC | Superseded metric; plumbing reusable |
| `analysis/keeper_trades.py` | 192 LOC | Untouched |
| `scripts/keeper_value.py` | ~15 KB | Fate decided at increment 2 |
| `scripts/keeper_trades.py` | ~9 KB | Untouched |
| `tests/test_analysis/test_keeper_value.py` | 488 LOC | Reconcile in increment 2 |
| `tests/test_scripts/test_keeper_value_script.py` | 275 LOC | Reconcile in increment 2 |
| `tests/test_analysis/test_keeper_trades.py` | - | Untouched |
| `tests/test_scripts/test_keeper_trades_script.py` | 48 LOC | Untouched |
| `tests/test_keepers/` | - | Extended by increment 1 |

Written independently of `keeper_value.py`, so its choices are not inherited. Superseded:
the discounted horizon (section 3), and the ratio-scaling fold whose
`DEFAULT_OUT_YEAR_REGRESSION = 0.6` is justified in-code only by the comment
`0.6 = "mostly ZiPS"`.

One incumbent behaviour must be preserved in spirit: `_scale_line` (lines 88-96) holds the
anchor flat on a NaN out-year cell, which is the only reason closers are scoreable today.
Any replacement must treat missing out-year data deliberately rather than leaning on
`safe_float`'s silent NaN-to-0. Its SGP/VAR plumbing, position handling and playing-time
treatment remain reusable. Per CLAUDE.md, existing tests are guardrails: where increment 2
changes covered behaviour, justify explicitly rather than editing assertions.

## 11. Verified facts

- ZiPS out-year loads for both types: 2027 hitters (1901, 75) / pitchers (1838, 70)
  **post-parse** (raw CSVs are 74/69 columns; `parse_*_csv` appends `player_type`). 2028
  identical. Before the pitcher exports were added, `scripts/keeper_value.py` hard-failed.
- **`SV` is populated in 0 of 1838 rows** in both out-year pitcher files, against 1838/1838
  (sum 1064) in ZiPS 2026. `HLD`/`QS`/`BS` likewise, unscored.
- **95 of 1838 pitchers in ZiPS 2027 have no ZiPS 2026 counterpart**; all 1901 hitters do.
- **No Age column in any of the seven ZiPS vintages** (2022-2028).
- The 2027 and 2028 pitcher files are not swapped or duplicated: the 2027 file drifts less
  from ZiPS 2026 than the 2028 file (mean |dIP| 4.45 vs 7.81, |dSO| 4.26 vs 7.38), the
  expected one-year vs two-year aging pattern. **This verifies file identity against each
  other, not against a fresh download** -- for that, see section 4's provenance note.
- ZiPS 2026 is a preseason full-season projection (mean PA 400.9, max 696) that hedges
  playing time pool-wide: only 2.9% of rows exceed 600 PA, 18.2% fall below 300, and 2025
  regulars ran +58 mean PA versus projection. So the PT residual carries a large systematic
  component that is not surprise -- which is why `norm` (6.2) is league-level and why the
  fit sample's selection is called out in 6.4.
- `keepers/mlb_stats.fetch_mlb_season` works for arbitrary historical years, keyed by
  MLBAM. Pitching returns `stat.wins`, `stat.saves`, `stat.era`, `stat.whip`,
  `stat.strikeOuts`, `stat.inningsPitched`, with the string caveats in 6.6. No callers
  before this feature.
- `position_aware_replacement_levels` (`sgp/replacement.py:240-279`) is a function of the
  denominators, the AVG/ERA/WHIP rate baselines, the module-global
  `REPLACEMENT_BY_POSITION`, and `team_ab`/`team_ip` -- **not** pool depth.
  `calculate_replacement_rates` (line 93) and `find_replacement_players` (line 24) are the
  pool-derived pieces.
- `draft_value.ParCurve` / `par_for_slot` are backward-looking, built from actual historical
  picks; `keeper_par` is the mean VAR of kept players. Not the forward-looking par of
  section 2. Note its docstring (`draft_value.py:485-488`) deliberately keeps a two-way
  player's rows **separate**, which is the opposite of section 5.5's dedupe requirement --
  do not follow it as a model for that.
- `BASE_YEAR = 2026` at `scripts/keeper_value.py:48`.

## 12. Non-goals

- Rebuilding a projection system. ZiPS out-year stays the baseline.
- Per-player or classifier-based skill-versus-luck labeling. Coefficients apply uniformly,
  shrunk only by sample size. Finer structure is decided at the review point, on evidence.
- Modelling out-year saves, or ranking relievers (5.1).
- Delta-roto or roster-construction-aware trio selection (section 2). Candidate later
  increment.
- Trade evaluation. `keeper_trades.py` and its spec are untouched.

## 13. Open questions

- **Should keeper depletion shift the positional floors? Expected answer: no.** The floors
  are the SGP of empirical waiver lines (`REPLACEMENT_BY_POSITION`, calibrated from this
  league's free agents), not depth-derived, and `STARTERS_PER_POSITION =
  roster_slots x num_teams` measures roster *demand*, which is unchanged in 2027 since
  keepers still occupy slots. Confirm and record, because section 7's no-circularity
  guarantee depends on it.
- Whether `k_pt` needs a finer split than one coefficient per player type. Deferred to the
  increment 1 results.
- Whether increment 2 extends `scripts/keeper_value.py` or replaces it.
- Position eligibility for 2027 uses current positions; no position-change projection. See
  5.6 for the coverage problem this inherits.
- Script the section 4 vintage check and retain both artifacts so the staleness claim stays
  reproducible.
