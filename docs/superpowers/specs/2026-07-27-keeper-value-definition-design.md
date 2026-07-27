# Keeper value: feature definition and 2026 fold-in design

Issue: #266 (depends on #265, closed by PR #267)
Date: 2026-07-27
Status: design converged after three adversarial review rounds. The **estimator form is
deliberately not specified here** -- see section 6.

**Supersedes** `2026-07-22-keeper-value-design.md` and
`2026-07-23-keeper-value-current-anchor-design.md` in full.
`2026-07-23-keeper-trade-generator-design.md` remains live and untouched.

## 1. The decision this serves

**Every team keeps exactly 3 players. Keeping is mandatory.** The 3 keepers consume draft
rounds 1-3, so the live 2027 draft begins at round 4. No escalating cost, no salary; the
choice is re-made every offseason.

Because a team forfeits rounds 1-3 *regardless of whom it keeps*, the cost of keeping is
constant across every possible choice of 3 and cannot influence which 3 to keep. So the
feature's work is:

1. Projecting 2027 value well, from a baseline that predates the 2026 season.
2. Measuring the surplus a trio generates over the picks it costs.
3. Comparing the owner's trio against the other nine teams'.

## 2. What "keeper value" means

**Absolute keeper value.** Projected 2027 VAR. This alone answers "which 3 do I keep":
take the top 3.

*Known limitation, stated up front.* VAR is additive and blind to team context. This is a
roto league (`draft.scoring_mode: deltaroto_immediate`), where marginal category value
saturates -- three 40-SB players are worth less than the sum of their VARs -- and nothing
in a VAR sort prevents a positionally redundant trio. "Take the top 3" is a strong
default, not a proof of optimality. Mitigation: the output carries a **category profile**
for the selected trio so saturation and redundancy are visible. Delta-roto trio search is
a candidate later increment, deliberately not attempted here.

**Surplus over forfeited picks.** For a trio ranked 1st/2nd/3rd, the i-th keeper's surplus
is `VAR_i - par_i`, where `par_i` is the expected VAR at the owner's round-i pick had no
keeper system existed. `par_i` is computed against the **undepleted** 2027 pool: since
keeping is mandatory the counterfactual is "what would rounds 1-3 have returned in a draft
with no keepers," so no keeper set is removed and `par_i` depends only on the pool and the
slot.

**The 2027 draft slot is not known.** `config/league.yaml`'s `draft.position: 8` is the
**2026** slot; 2027 order depends on 2026 final standings that do not exist yet. Par is
computed at a **stated assumed slot, defaulting to position 8**, labelled as an assumption,
and reported across a range of plausible slots. At position 8 in a 10-team snake the
ordinals are picks **8, 13, 28** (R1 forward: 8; R2 reversed: 11 + (10-8) = 13; R3 forward:
20 + 8 = 28).

**Surplus is a diagnostic, not a decision rule** -- no keep/release decision exists under
mandatory keeping. Three caveats, all required in the output:

- **Surplus is not field-relative.** `par_i` rises with a better draft slot, so holding trio
  quality fixed, surplus is *larger* for a team that finished worse. It measures "more than
  my own forfeited picks," never "better than the field." Cross-team comparison uses
  absolute value (section 7).
- **`par_i` assumes a draft in exact descending-VAR order**, which real drafts are not
  (positional runs, reaches). The direction of the resulting bias is **not asserted here**:
  it depends on whether early picks deviate toward or away from best-available, and an
  earlier draft of this spec asserted a sign it could not derive. Report par alongside the
  assumption and leave the sign to observation.
- **The par pool excludes relievers** (section 5.1), which biases `par_3` down and surplus
  up. This is not hypothetical: with saves restored the best 2027 reliever scores +4.342
  against `par_3` of 4.397 -- correctly-valued closers sit essentially *at* the round-3
  ordinal. Disclose the direction.

**Trajectory.** A separate 2028 column: "will he still be keep-worthy next year." Never
blended into the headline.

## 3. Horizon: single-year, and why

The metric is 2027; 2028 appears only as the unblended trajectory signal.

This breaks from `analysis/keeper_value.py`, which sums a discounted 3-year stream
(`DEFAULT_DISCOUNT = 0.80`, `DEFAULT_HORIZON = 3`, lines 28-29). That prices a multi-year
commitment the rules do not impose: the 2028 slot is re-chosen in 2028 with 2028
information. It also has a concrete defect -- with `base_year = 2026` the largest weight in
the sum (1.0, ~41% of the total) lands on a season already finished when the decision is
made. And it is epistemically backwards: with a pre-2026 baseline the out-years are the
least reliable inputs, so a discounted sum adds most noise where confidence is lowest.

The young-player counter-case resolves without a sum. If a 21-year-old's 2027 line is
strong, single-year VAR captures him. If it is weak, he is inexpensive in the 2027 draft,
so a keeper slot -- worth a round-1 through round-3 pick -- is the wrong instrument.

## 4. The core problem: the out-year baseline is permanently stale

ZiPS 2027 and 2028 were generated 2026-03-25 and know nothing of 2026.

**Provenance.** The out-year *pitcher* files on disk were downloaded 2026-07-27 and
installed under the March naming convention; the *hitter* files were already present from
March. The 2026-07-27 hitter download matched the March export on all 1827 PlayerIds
present in both, HR identical on every one. Of 1901 rows, 74 did not join, so the accurate
claim is "identical on the overlap," not "byte-identical." The July hitter artifact was not
retained, so that comparison is not currently reproducible (section 13).

The conclusion holds, and is now confirmed on freshly downloaded pitcher files: FanGraphs
has not regenerated ZiPS out-year projections, and re-downloading will not fold 2026 in.

## 5. The fold-in mechanism

### 5.1 Scope: relievers are excluded from out-year ranking

ZiPS 2027 and 2028 contain no saves data: `SV` is populated in **0 of 1838** rows in both
out-year pitcher files, against 1838/1838 (sum 1064) in ZiPS 2026. The failure is silent --
`safe_float` (`utils/constants.py:372-377`) coerces NaN to 0.0, and `sv` exists as an
all-empty column so the parser does not raise.

Scoring relievers without saves is monotone in lost saves (every reliever loses `SV/7`
SGP), so it does not scramble the RP ordering wholesale -- rank correlation against a
saves-aware ordering is about 0.99. What it does is **compress the population and let
zero-save arms outrank genuine closers**: on the real 2027 pool a middle reliever scores
-0.689 against Mason Miller's -0.924 and Edwin Diaz's -1.895. A ranking that puts a
replaceable arm above two elite closers is not usable, and no disclosure repairs it.

**So out-year valuation and ranking cover hitters and starting pitchers only.** Relievers
are computed and reported in a separate not-comparable list, never ranked against hitters
or starters, never placed in a projected trio, and never included in the par pool.

Consequences to accept and disclose: a genuine closer keeper cannot be evaluated by this
feature, and at least one opponent trio is affected today (Pete Fairbanks, Send in the
Cavalli, the only `GS/G` reliever among the seven pitcher keepers).

**SP/RP classification** comes from the ZiPS out-year file: a pitcher is a reliever when
`GS / G < 0.5`. Both columns are parsed by `PITCHING_COLUMN_MAP` and are non-null in all
1838 rows. This avoids depending on `cache:positions`, which stores pitchers as a bare
`"P"` with no SP/RP eligibility and so could not do this job. Two caveats:

- **`G = 0` rows exist** (14 in ZiPS 2028: Verlander, Scherzer, Darvish, Morton, Yates and
  others). `0/0` is NaN and `NaN < 0.5` is `False`, so a naive rule silently classifies them
  as *starters*. Their `IP = 0` also makes every pitcher rate NaN in the base term. Guard
  explicitly. (The 2027 pitcher file is clean, `G` min 11, but section 5.5 applies the same
  machinery to 2028.)
- **The cut is a judgment call, not a fact read off the file.** 648 of 1838 pitchers (35.3%)
  fall in `0 < GS/G < 0.5` -- genuine swingmen -- and 42 of those project >= 80 IP. The cut
  is at least stable (1742/1743 agreement between the 2026 and 2027 files).

**This DOES require a scoring-path change, and it is mandatory.** An earlier draft of this
spec claimed the opposite; that claim was wrong and would have shipped a large silent error.
`calculate_var` routes a pitcher to its SP or RP floor via `_pitcher_floor_key`, which calls
`role_from_ip` -- `"SP" if ip >= 100.0 else "RP"` (`sgp/var.py:28-29`,
`utils/constants.py:390`). `GS` is never consulted; the docstring says the position token is
deliberately ignored because `PitcherStats` has no `GS` field at deployment.

ZiPS out-year files hedge innings hard, so this misroutes the majority of the population
this feature retains: of the 727 pitchers with `GS/G >= 0.5` in ZiPS 2027, **418 (57.5%)
project under 100 IP** (median 97.0) -- Snell 98.3, Glasnow 99.0, Bradish 99.3, Sasaki
97.7. Each is scored against the RP floor, which carries 8 free saves worth 1.143 SGP that
no out-year pitcher can earn; the SP/RP floor gap is roughly 1.9 SGP. Fifty pitchers cross
the threshold between the 2026 and 2027 vintages purely from out-year hedging.

**Requirement:** every pitcher must be routed to its floor by the `GS/G` role, not by
projected IP. `calculate_var` already exposes a `role_ip` keyword for exactly this class of
misrouting (`sgp/var.py:39, 58, 78`); increment 2 must pass a `role_ip` consistent with the
`GS/G` classification. Note `role_ip` takes an *IP value*, not a role, so the mapping from
role to the value passed must be stated in the implementation plan.

*(With relievers excluded, SV itself needs no surgery: it is simply 0 for the retained
hitters and starters, and `REPLACEMENT_BY_POSITION["SP"]` already carries `"sv": 0`, so a
correctly-routed starter nets against a save-free floor. Removing SV from the denominators
was previously specified and is infeasible: `calculate_player_sgp` indexes
`denoms[Category.SV]` unconditionally (`sgp/player_value.py:83, 126`),
`get_sgp_denominators` offers override but never removal, and `_empirical_pitcher_floor`
(`sgp/replacement.py:218`) reads a module global with no injection seam.)*

### 5.2 Decompose, then fold

Counting stats are products of a rate and playing time; folding them raw conflates "hit for
less power" with "played half a season." Of the ten scored categories
(`utils/constants.py:13-22`), only AVG/ERA/WHIP are rates; R/HR/RBI/SB/W/K/SV are counting;
AB and IP are the scoring weights (PA is not scored).

**Rates folded (one coefficient each):** hitters `HR/PA`, `R/PA`, `RBI/PA`, `SB/PA`,
`H/AB`, `AB/PA`; pitchers `K/IP`, `W/IP`, `ER/IP`, `BB/IP`, `H_allowed/IP`.

**Playing time folded:** `PA` for hitters, `IP` for pitchers, clamped at `>= 0`.

**Reconstruction -- each rate multiplies its OWN denominator.** This is not optional and an
earlier draft got it wrong by defining a single playing-time term per player type:

```
PA_2027  = folded PA
AB_2027  = PA_2027 * (AB/PA)_2027      # derived BEFORE any AB-denominated rate
H_2027   = AB_2027 * (H/AB)_2027       # AB, not PA
HR_2027  = PA_2027 * (HR/PA)_2027      # PA, and likewise R, RBI, SB
K_2027   = IP_2027 * (K/IP)_2027       # and likewise W, ER, BB, H_allowed
```

Multiplying `H/AB` by PA instead of AB inflates AVG by `1/0.8977 = 1.114` (mean `AB/PA` in
ZiPS 2027), turning a .250 hitter into .278 -- worth 11.2 SGP at the league's 0.0025 AVG
denominator, larger than the entire VAR range of the top-30 board. Deriving AB from PA also
makes `AB <= PA` structural rather than a constraint to police.

**AVG/ERA/WHIP are recomputed from folded components**, never folded directly:
`AVG = H/AB`, `ERA = 9*ER/IP`, `WHIP = (BB + H_allowed)/IP`. BB and H_allowed are folded
**separately**, not as `(BB+H)/IP` -- `calculate_replacement_rates`
(`sgp/replacement.py:121-124`) needs them as distinct columns, as do `PITCHER_PROJ_KEYS`
and `PITCHER_CORR_STATS`.

*Honest note:* for the rates themselves this decomposition is algebraically equivalent to
folding the rate directly, since the playing time cancels. It earns its keep on the seven
**counting** categories, where `count = rate * PT`, and on producing the components
`calculate_replacement_rates` requires. One mechanism serves all three needs.

Availability is verified on both sides: `cache:full_season_projections` exposes
`h`/`ab`/`pa` and `er`/`bb`/`h_allowed`/`ip` at 100%, and `HITTING_COLUMN_MAP` /
`PITCHING_COLUMN_MAP` parse the corresponding ZiPS columns in every vintage including the
out-years.

### 5.3 Shrinking noisy rates

A rate observed over few plate appearances is noisy and must be shrunk toward the
projection. A playing-time residual is not noisy in that way -- it is the observation
itself, and shrinking it in proportion to the playing time an injury suppressed would be
circular and would make the PT coefficient structurally unable to learn from players who
lost time. **So the shrink applies to rate residuals only.**

The functional form and constants are an increment-1 deliverable (section 6), subject to
two requirements: the shrink must be **bounded at or below 1** so it can never amplify a
residual, and whatever playing-time quantity feeds it must be the **same quantity** used by
the gate (5.4) and the minuend (6.4), which an earlier draft got wrong by using three
different bases.

### 5.4 Gating: who gets folded at all

**Players below a minimum realized 2026 MLB playing time pass through at raw ZiPS 2027,
flagged `unfolded`.**

This gate is load-bearing. Because the shrink does not apply to the playing-time residual
(5.3), an ungated player with little or no MLB time would have his 2027 playing time folded
toward that near-zero observation. For a prospect projected at 300 PA who never debuted and
a PT coefficient of 0.8, `350 + 0.8*(0 - 300) = 110 PA` -- gutting precisely the young
high-upside case section 3 relies on.

**Absence from the MLB actuals is not zero playing time.** `playerPool=all` returns MLB
players only, so a player who spent 2026 in AAA is *missing from the data*, not present with
0 PA. Absence must resolve to `unfolded` passthrough, never to a large negative residual.
The implementation must distinguish the two cases explicitly.

**The gate creates a discontinuity and increment 1 must quantify it.** Because the PT term
is unshrunk, the fold is fully on or fully off at the boundary. A regular who lost the season
to a May injury -- full-season line 120 PA, ZiPS 2026 400 PA, ZiPS 2027 380 PA, coefficient
0.8 -- passes through at 380 PA just below the gate and folds to 156 PA just above it: a 59%
jump across a couple of plate appearances. Increment 1 must either justify the cliff as
acceptable and publish its magnitude, or specify a ramp. It must not be left implicit, as an
earlier draft did.

A regular lost to a March injury also lands below the gate and passes through with his stale
line. That is a deliberate choice, not a clean fall-through, and the output flags it.

### 5.5 Other edge cases

- **`0/0` rates.** Zero realized playing time gives an undefined rate, and `0 * NaN = NaN`,
  so a zero shrink does **not** rescue it; `safe_float` would later coerce the poisoned value
  to a silent 0.0. The 5.4 gate covers the realized side; **zero-denominator rows also exist
  in the ZiPS base** (2 hitters with `PA = 0` in 2027, 3 in 2028; 14 pitchers with `IP = 0`
  in 2028). Guard NaN explicitly on both sides rather than relying on the shrink.
- **In ZiPS 2027 but absent from ZiPS 2026: 95 of 1838 pitchers (5.2%).** No baseline, so no
  residual: passthrough, flagged `unfolded`. All 1901 hitters have a counterpart.
- **Absent from ZiPS 2027 entirely** (2027 debutants): no projection. Reported separately as
  unrankable, never scored zero.
- **Two-way players.** Ohtani is the only MLBAMID in both 2027 files, and he breaks more than
  dedupe: `cache:positions` holds `'shohei ohtani': ['UTIL']` with **no `P` token**, so his
  *pitcher* row never enters the pitcher branch of `calculate_var` and is scored against the
  **UTIL hitter floor** (verified: pitcher row VAR -3.04 at the UTIL floor). He is also the
  one player where the two role classifiers disagree (`GS/G = 1.00` says SP, `IP = 95.3` says
  RP). Requirements: dedupe by MLBAMID for both trio selection and the par ordinal, counting
  one keeper slot; state which row wins (the higher-VAR row, consistent with the repo's
  namesake-collision convention); and ensure any pitcher row routes to a *pitcher* floor
  regardless of its position token.
- The same coefficients, shrink, gate and exclusions apply to ZiPS 2028 for trajectory. This
  is an approximation -- calibration is one-year-forward -- acceptable only because trajectory
  never enters the sort.

### 5.6 Joins and eligibility

Joins are by ID wherever one exists: `PlayerId` across ZiPS vintages; `MLBAMID` to
`player.id` for MLB actuals; `fg_id`/`mlbam_id` for `cache:full_season_projections` (`fg_id`
100%; `mlbam_id` null on exactly 1 hitter and 1 pitcher). **Normalized name matching is
required for the `config/league.yaml` keeper list** -- exact-name joining drops 3 of 30
(`Jose Ramirez`, `Julio Rodriguez`, `Ronald Acuna Jr.`). Use
`draft/keepers.find_keeper_match` and `sgp/rankings.rank_key`.

**Position eligibility is the weak join and must be disclosed.** `calculate_var` routes
hitters on `player["positions"]`, whose only source is `cache:positions` -- **706 entries,
keyed by bare normalized name**, against a 2027 pool of 3,739 rows, an 18.5% match rate.
Two distinct failure modes:

- **Unmatched (~81%).** Unmatched hitters fall to the UTIL floor; both move VAR. Flag every
  row using a defaulted position and report the coverage rate.
- **Mis-matched.** Bare-name keying violates the repo's `name::player_type` rule
  (CLAUDE.md: "Never key on bare names"), and the 2027 pool has **17 normalized-name
  collisions across the hitter and pitcher files** -- including `edwin diaz` and `jacob webb`,
  whose cache entries are `['P']`, so the *hitter* rows would route to a pitcher floor. These
  are silently wrong and currently unflagged. Increment 2 must key positions by
  `name::player_type` or flag collisions.

Since section 2's par ranks the pool, this is material: report par over both the full pool
and the resolvable-position subset so the sensitivity is visible. (The SP/RP split does not
depend on this -- it comes from `GS/G`, per 5.1.)

## 6. Increment 1: the calibration study

### 6.1 The question, and the one constraint that is not negotiable

**The question:** how much should a 2026 surprise move a 2027 baseline that has never seen
2026?

**The constraint:** *the calibration baseline must not already know the surprise season.*
Every ZiPS file on disk is a preseason projection for its own year, so `ZiPS_{Y+1}` was built
knowing year Y. Fitting against it would measure how much surprise ZiPS has *already
absorbed*, driving the coefficient toward zero by construction, and would ship the conclusion
that 2026 tells us nothing about 2027 -- the opposite of the truth, since ZiPS 2027 has never
seen 2026. The base must therefore be `ZiPS_Y`, built knowing only through Y-1, so that
neither it nor `ZiPS_2027` has absorbed the season whose surprise is being folded.

This single insight survived all three review rounds and is the reason the study is worth
running. Everything else about the estimator is deliberately left open below.

### 6.2 The estimator is an increment-1 deliverable, not a spec decision

Three review rounds each produced a confidently-specified estimator, and each was found
broken by the next round -- a free scale term that degenerated the coefficient's meaning, then
a normalization term that was unidentified and uncomputable at serve time, alongside a fit
sample definition that contradicted the survivorship rule. The lesson is that the exact
regression form cannot be pinned in prose ahead of contact with the data.

**So the spec fixes the question, the data, and the requirements. Increment 1 chooses,
justifies, and documents the estimator.** Its written finding is a deliverable in its own
right, not a footnote to a number.

**Requirements the chosen estimator must satisfy:**

1. **Calibration and production apply the same functional form.** Any term present only in
   calibration must be justified explicitly, and its production value stated. The known gap it
   would address: `ZiPS_Y` targets year Y while the target is Y+1, whereas `ZiPS_2027` is
   already aged to 2027. Resolving that gap is part of the deliverable; note that a per-player
   aging adjustment is unobtainable (**no Age column in any of the seven ZiPS vintages**, and
   a player's own 2026-to-2027 ratio is unavailable for 57% of the 2022 cohort).
2. **Evaluated out of sample** via leave-one-pair-out over the three usable pairs, with
   per-held-out-pair error reported.
3. **Compared against both endpoints** -- ignoring 2026 entirely, and transferring the full
   surprise -- since those are the two things the feature must beat to be worth building.
4. **Stability reported across pairs**, per category.
5. **Survivorship handled explicitly**, with its effect reported (6.3).
6. **Conditioning stated.** Any shrink constant, sample gate, or loss weighting that is chosen
   rather than fitted must be named, and the coefficient reported as conditional on it.
7. **Amplification bounded or refused.** A coefficient that would amplify residuals in
   production must not ship silently on the strength of held-out error alone.
8. **The train/serve gap stated** (6.4).
9. **The 5.4 gate discontinuity quantified** or a ramp specified.

### 6.3 The data

Three usable year-pairs: **2022->2023, 2023->2024, 2024->2025.** A 2025 pair needs a complete
2026 season (`season_end: 2026-09-28`; today is 2026-07-27); a 2021 pair is impossible because
`data/projections/` begins at 2022. A fourth pair opens after the 2026 season and the study
must be re-runnable then.

**One pair spans a structural break.** MLB's 2023 rules package (pitch clock, larger bases,
pickoff limits) raised league stolen bases sharply, so `SB/PA` has a level shift inside a
three-pair sample. Report per-category stability and attribute an unstable SB coefficient to
the break rather than averaging it away. Note that any league-level normalization computed
from ZiPS files would *pre-absorb* this break and make it unmeasurable -- an argument to
weigh when choosing the estimator.

**Survivorship.** `actual_{Y+1}` exists only for players who kept playing: of players with
>=100 PA in year Y, 75.5% / 77.7% / 79.5% reach >=100 PA in Y+1 across the three pairs.
Fitting on survivors alone measures persistence *given continued play*, biasing the
playing-time coefficient upward. Two cautions for whoever runs this:

- **Those rates were measured on the full MLB >=100 PA population (~460/season), not on the
  ZiPS-matched sample (~350).** Re-measure on the actual fit sample before using them.
- The treatment necessarily differs by coefficient, and a single "report it both ways"
  instruction is ill-posed: a non-survivor has a well-defined near-zero *playing time* but no
  meaningful *rate*. Whatever is chosen must be stated, and must be consistent with 5.4's rule
  that absence from the leaderboard is not zero.

**Sample size** must be reported: roughly 350 matched hitters per pair at the 100 PA
threshold, ~1050 across three pairs, before any per-category split. Against ZiPS 2023's 1716
hitters at most ~45% can ever match.

### 6.4 The train/serve gap

The coefficient is fit on a fully-observed `actual_Y` against `ZiPS_Y`, but applied to a 2026
line that is neither. Two mismatches:

- The production minuend is ~35% unrealized rest-of-season projection across rostered players,
  so the applied residual is attenuated -- and that share shrinks weekly, drifting the metric
  with the coefficient frozen.
- `cache:full_season_projections` is a 5-system blend (steamer/zips/atc/the-bat-x/oopsy at 0.20
  each), so subtracting a ZiPS-only baseline injects inter-system level offsets as spurious
  signal.

**Partial mitigation, with its limits stated.** Using a ZiPS-only 2026 line as the minuend
removes the multi-system offset. It does **not** remove the actual-vs-projection difference,
and the coverage is incomplete: `data/projections/2026/rest_of_season/2026-07-27/zips-*.csv`
are *rest-of-season* files (609 hitters, mean PA 107.1; 748 pitchers), so a full-season line
requires YTD actuals plus ROS. Coverage of the 243 rostered players is 242/243, but par is
computed over the full pool, where roughly 2,382 of 3,739 rows have no ZiPS ROS line. State the
coverage and what happens to uncovered rows.

**Deferred as infeasible:** a full re-fit with year-Y actuals truncated to a comparable season
fraction and blended forward. `rest_of_season/` exists only for 2026, FanGraphs does not archive
historical mid-season ROS projections, and `game_logs` in `data/local.db` are 2026-only. Report
the analytic attenuation implied by the current ROS share instead, and disclose the residual
uncertainty.

### 6.5 Data-ingest requirements

- **IP arrives as a baseball-notation string.** `stat.inningsPitched` returns `"1.2"` for 1 2/3
  innings; naive `float()` gives 1.2. The error is small on a season total (~0.3%) but the
  conversion is free and units must match ZiPS's decimal IP. `stat.era` and `stat.whip` are also
  strings needing coercion with a zero-IP guard.
- **Assert page completeness, not exact row counts.** Hitting returns roughly 794/769/742/765
  rows for 2022-2025 at `playerPool=all`, but MLB revises historical rosters, so exact-equality
  assertions produce false failures. Assert a lower bound plus the real failure mode:
  `_fetch_mlb_season` breaks on the first short page (`keepers/mlb_stats.py:51-52`), making silent
  truncation indistinguishable from a complete pull.
- **`fetch_or_cache` never invalidates** (`keepers/cache.py:23-38`). Correct for completed
  seasons; in-progress seasons must not use it, or must use a date-stamped path.

### 6.6 Acceptance

The study is complete when it delivers a written finding that documents the chosen estimator and
reports every item in 6.2's requirement list.

It **passes** if the chosen estimator beats both endpoints on held-out error on a majority of
pairs. It **falls back** otherwise, shipping whichever endpoint performed best, recorded as the
finding.

Note the limit of this bar: it is measured on the rate/playing-time scale, while the feature
consumes VAR *rank*. Rate error can improve while ranking degrades. VAR is not built until
increment 2, so a rank-level check is scoped there, and the review point between increments
exists partly to catch this.

A coefficient near zero is a suspected setup error -- most likely baseline contamination per 6.1
-- and must be investigated before being reported as a result. That is a diagnostic, not the bar.

## 7. Cross-team comparison

Because "which 3" is a sort under mandatory keeping, cross-team comparison carries most of the
feature's value.

Universe: `cache:roster` (the owner's 25) plus `cache:opp_rosters` (nine opponents, 23-25 each,
218) -- **243 players**. Each team's projected trio is its top 3 by absolute 2027 value among
**hitters and starting pitchers**, deduped by MLBAMID. The league table ranks the ten trios.

Because relievers are excluded from the selection universe, a trio can never *contain* one. The
flag that matters is different: **a team whose roster holds a reliever plausibly worth a keeper
slot is flagged as having an unevaluable candidate**, so its trio is read as "best among
evaluable players," not "best." One team qualifies today.

**Trios are compared on absolute 2027 VAR, not surplus** -- per-team surplus would need each
team's 2027 slot, which does not exist yet, and surplus is not field-relative anyway (section 2).
The owner's own surplus is reported separately at the assumed slot.

No circularity: `par_i` is computed against the undepleted pool, so it does not depend on which
players are kept, and trio selection depends only on absolute value. A single pass. **This is
conditional on two implementation choices being pinned in increment 2:** that positional floors
are not made a function of keeper depletion (section 13, expected "no"), and the replacement-rate
choice below.

**Pin the replacement-rate path.** `position_aware_replacement_levels` is a pure function of its
inputs, but one input, `repl_rates`, comes from `calculate_replacement_rates`
(`sgp/replacement.py:93-124`), which is **pool-derived** -- it takes a band around the 90th-ranked
pitcher in the live pool. This matters twice: whether `repl_rates` is passed at all swings the SP
floor by roughly 1.7 SGP on every pitcher, and the 5.1 reliever exclusion itself moves the floors
by about 0.11 SGP by changing the pool. Increment 2 must pin the choice explicitly, use the same
configuration as the draft board for comparability, and disclose it.

## 8. Timing caveat

Rosters are live and current; the refresh keeps them so. The caveat is the calendar. Keeper
eligibility runs off the season-end roster and it is July, so trades and waiver claims will change
who is eligible. The ranking stays accurate as rosters move, but "my projected trio" is provisional
until rosters freeze. The 2026 residual also sharpens as the rest-of-season share (6.4) shrinks
toward zero.

## 9. Increments and delivery

**Increment 1: the calibration study.** Standalone -- ZiPS vintages and MLB actuals only; no SGP,
no VAR, no board.

- Delivery: a script under `scripts/`, results to `data/analysis/`, plus the written finding
  (6.2) committed under `docs/`.
- Tests: IP notation conversion; the shrink function including its zero and high-playing-time
  bounds; the 5.4 gate, covering both zero playing time and absence from actuals; the 5.2
  decomposition and reconstruction, specifically that each rate multiplies its own denominator
  and that `AB <= PA` holds; NaN guarding on both the realized and ZiPS-base sides; and recovery
  of a known planted coefficient on synthetic data.
- Acceptance: section 6.6.

Review point: examine the findings and decide where to go next.

**Increment 2: the value pipeline and outputs.** Builds `updated_2027`/`updated_2028`, scores
through the existing SGP and VAR path with the `role_ip` routing fix (5.1), constructs `par_1..3`
at the assumed slot, and emits the outputs.

Required output fields: `player_id`, `name`, `player_type`, `role` (SP/RP), `team`,
`absolute_var_2027`, `surplus` (owner only), `trajectory_var_2028`, `category_profile` (trio
members), and the flags `unfolded`, `unrankable`, `reliever_not_comparable`,
`unevaluable_candidate`, `defaulted_position`, `injury_passthrough`. Surface (CLI vs dashboard vs
cache key), persistence, and the fate of `scripts/keeper_value.py` are decided at the start of
increment 2.

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

Written independently of `keeper_value.py`, so its choices are not inherited. Superseded: the
discounted horizon (section 3), and the ratio-scaling fold whose
`DEFAULT_OUT_YEAR_REGRESSION = 0.6` is justified in-code only by the comment `0.6 = "mostly
ZiPS"`.

One incumbent behaviour must be preserved in spirit: `_scale_line` (lines 88-96) holds the anchor
flat on a NaN out-year cell, which is the only reason closers are scoreable today. Any replacement
must treat missing out-year data deliberately rather than leaning on `safe_float`'s silent
NaN-to-0. Its SGP/VAR plumbing, position handling and playing-time treatment remain reusable. Per
CLAUDE.md, existing tests are guardrails: where increment 2 changes covered behaviour, justify
explicitly rather than editing assertions.

## 11. Verified facts

- ZiPS out-year loads for both types: 2027 hitters (1901, 75) / pitchers (1838, 70) **post-parse**
  (raw CSVs are 74/69 columns; `parse_*_csv` appends `player_type`). 2028 identical. Before the
  pitcher exports were added, `scripts/keeper_value.py` hard-failed.
- **`SV` populated in 0 of 1838 rows** in both out-year pitcher files, against 1838/1838 (sum
  1064) in ZiPS 2026. `HLD`/`QS`/`BS` likewise, unscored.
- **95 of 1838 pitchers in ZiPS 2027 have no ZiPS 2026 counterpart**; all 1901 hitters do.
- **No Age column in any of the seven ZiPS vintages** (2022-2028).
- **418 of 727 `GS/G >= 0.5` pitchers in ZiPS 2027 project under 100 IP** (median 97.0), so they
  misroute to the RP floor under `role_from_ip`. SP/RP floor gap ~1.9 SGP; the RP floor's 8 free
  saves are worth 1.143 SGP.
- 2027/2028 pitcher files are not swapped: the 2027 file drifts less from ZiPS 2026 than the 2028
  file (mean |dIP| 4.45 vs 7.81, |dSO| 4.26 vs 7.38). **This verifies the two files against each
  other, not against a fresh download** -- for that see section 4.
- ZiPS 2026 is a preseason full-season projection (mean PA 400.9, max 696) that hedges playing time
  pool-wide: only 2.9% exceed 600 PA, 18.2% fall below 300, and 2025 regulars ran +58 mean PA
  versus projection. The playing-time residual therefore carries a large systematic component that
  is not surprise -- a fact the estimator must contend with (6.2).
- Mean `AB/PA` in ZiPS 2027 is 0.8977.
- `keepers/mlb_stats.fetch_mlb_season` works for arbitrary historical years, keyed by MLBAM, and
  returns the raw response with no column dropped. Confirmed present: `stat.wins`, `stat.saves`,
  `stat.era`, `stat.whip`, `stat.strikeOuts`, `stat.inningsPitched`. The study additionally needs
  `ER`, `BB`, `H` (pitchers) and `PA`, `AB`, `H` (hitters) for the 5.2 decomposition -- confirm
  their exact field names against a live pull before planning, as section 6.5's caveats apply.
- `position_aware_replacement_levels` (`sgp/replacement.py:240-279`) is a function of the
  denominators, the AVG/ERA/WHIP rate baselines, the module-global `REPLACEMENT_BY_POSITION`, and
  `team_ab`/`team_ip` -- not pool depth. But `calculate_replacement_rates` (line 93) and
  `find_replacement_players` (line 24) are pool-derived; see section 7.
- `cache:positions` holds 706 bare-normalized-name entries; 18.5% match rate against the 3,739-row
  2027 pool; 17 normalized-name collisions across the hitter and pitcher files.
- `draft_value.ParCurve` / `par_for_slot` are backward-looking, built from historical picks;
  `keeper_par` is the mean VAR of kept players. Not the forward-looking par of section 2. Its
  docstring (`draft_value.py:485-488`) deliberately keeps a two-way player's rows **separate**,
  which is the opposite of section 5.5's dedupe requirement -- do not follow it as a model there.
- `BASE_YEAR = 2026` at `scripts/keeper_value.py:48`.

## 12. Non-goals

- Rebuilding a projection system. ZiPS out-year stays the baseline.
- Per-player or classifier-based skill-versus-luck labeling. Coefficients apply uniformly, shrunk
  only by sample size. Finer structure is decided at the review point, on evidence.
- Modelling out-year saves, or ranking relievers (5.1).
- Delta-roto or roster-construction-aware trio selection (section 2). Candidate later increment.
- Trade evaluation. `keeper_trades.py` and its spec are untouched.

## 13. Open questions

- **Should keeper depletion shift the positional floors? Expected answer: no.** The floors are the
  SGP of empirical waiver lines calibrated from this league's free agents, not depth-derived, and
  `STARTERS_PER_POSITION = roster_slots x num_teams` measures roster *demand*, unchanged in 2027
  since keepers still occupy slots. Confirm and record -- section 7's no-circularity guarantee
  depends on it.
- Whether the playing-time coefficient needs a finer split than one per player type. Deferred to
  the increment 1 results.
- Whether increment 2 extends `scripts/keeper_value.py` or replaces it.
- The exact `role_ip` value passed per role (5.1), to be settled in the increment 2 plan.
- Script the section 4 vintage check and retain both artifacts so the staleness claim stays
  reproducible.
