# Keeper value: feature definition and 2026 fold-in design

Issue: #266 (depends on #265, closed by PR #267)
Date: 2026-07-27
Status: design approved, implementation to follow

## 1. The decision this serves

Each offseason the owner retains 3 players from the end-of-season roster. Keeping is
free: no escalating cost, no forfeited draft pick, no salary. The choice is re-made
every year, so retaining a player in 2027 does not constrain the 2028 choice.

League-wide, 10 teams x 3 keepers withholds 30 players from the draft pool and the
draft runs 3 rounds shorter. Declining to keep a player means he re-enters the pool and
the roster spot is filled from that (depleted) pool instead.

The feature answers two questions:

1. Which 3 of my players are worth retaining, and by how much do they beat the
   alternative?
2. How does my retained trio compare to the other nine teams' trios?

## 2. What "keeper value" means

Three outputs, deliberately kept separate rather than blended into one score.

**Absolute keeper value.** A player's projected 2027 VAR, on the same scale the draft
board already uses. Answers "how good will he be."

**Relative keeper value (the headline number).** Absolute value minus *first-pick par*:
the VAR of the best player expected to be available at the owner's 2027 first-round
pick once the league's 30 keepers are off the board. At snake position 8 that is
roughly the 38th-best player overall (30 kept + 7 picks ahead).

A positive number means retaining the player beats using the pick. A negative number
means release him and take the pick. This is the headline because absolute VAR only
restates what is already known (Soto is good); the decision-relevant quantity is the
gap over the pick that retention actually costs. It also gives the 3-slot cap meaning:
if the 4th-ranked player has a larger gap than the 3rd, that is the swap to make.

**Trajectory.** A separate 2028 column answering "will he still be keep-worthy next
year." Never blended into the headline; it informs the eye, not the sort.

## 3. Horizon: single-year, and why

The metric is a single year (2027). 2028 appears only as the unblended trajectory
signal.

This is a deliberate break from the existing `analysis/keeper_value.py`, which sums a
discounted 3-year stream (discount 0.80, horizon 3). That structure prices a multi-year
commitment the league rules do not impose. Because keepers are free and re-chosen
annually, there is no lock-in: the 2028 decision is made fresh in 2028 with better
information. Summing the stream also lets an unexamined discount rate silently decide
how much a young ascending player is worth, which is exactly the kind of buried
judgment call this issue exists to surface.

The counter-case for a multi-year horizon is the 21-year-old whose 2027 line is
unremarkable but whose 2029 line is a star. That case is handled without a discounted
sum: if he is unremarkable in 2027 he will be inexpensive in the 2027 draft, so a
keeper slot is not required to retain access to him.

## 4. The core problem: the out-year baseline is permanently stale

The ZiPS 2027 and 2028 projections on disk were generated 2026-03-25 and know nothing
of the 2026 season.

This was verified, not assumed. A fresh 2026-07-27 download of the ZiPS 2027 hitters
file is identical to the March export (1827 matched PlayerIds, HR identical on 100% of
them). FanGraphs has not regenerated ZiPS out-year projections since March and there is
no indication it will mid-season. Re-downloading will never fold 2026 in. Whatever
adjustment this feature builds is the only path, permanently.

## 5. The fold-in mechanism

For every player, on every scored category:

```
residual_2026 = full_season_2026 - ZiPS_2026_preseason
updated_2027  = ZiPS_2027 + k * w * residual_2026
```

`residual_2026` is what the 2026 season revealed that ZiPS did not know when it built
the out-year lines. All of ZiPS 2026, 2027, and 2028 come from the same pre-2026
vintage, so the residual is a clean measure of surprise against that vintage's own
expectation.

**`full_season_2026`** is the YTD+ROS blend the season dashboard already computes, not
the raw year-to-date line. It is the best available estimate of the completed 2026
season, and it sharpens on its own as the season closes.

**`k` is the persistence coefficient** and is the substance of the whole feature.
`k = 0` ignores 2026 entirely (today's stale baseline). `k = 1` transfers the full
surprise. The truth is in between. Separate coefficients for rate stats and for playing
time: a lost half-season is different information than a lower batting average.

**`w` is a small-sample shrink** on the residual, proportional to realized playing time
against a full season. A 50-PA surprise moves the 2027 line about 1/12 as much as a
600-PA one. This is a stated default rather than a finding: without it a September
call-up's hot 40 PA rewrites his 2027 projection.

**Edge cases.**

- No 2026 MLB playing time: residual is zero, so `updated_2027 = ZiPS_2027` unchanged.
  Prospects fall through cleanly with no special case.
- Absent from ZiPS 2027 entirely (2027 debutants): no baseline exists. Reported
  separately as unrankable rather than silently scored as zero.
- The same `k` and `w` apply to ZiPS 2028 for the trajectory column. This is an
  approximation: `k` is calibrated on one-year-forward persistence (section 6), and a
  surprise plausibly persists differently two years out. Acceptable because the
  trajectory column is a directional signal that never enters the sort, but it should
  not be reused as if it were a calibrated 2028 valuation.

## 6. Calibrating k

Five ZiPS preseason vintages are on disk (2022 through 2026) and matching MLB actuals
are fetchable for each via the ingest kept from #265. That gives four year-pairs to fit
against.

**The baseline must not already know year Y.** This is the trap in the setup. Every
ZiPS file on disk is a preseason projection for its own year, so `ZiPS_{Y+1}` was built
knowing year Y. Using it as the base would mean fitting how much of a surprise ZiPS has
*already* absorbed, and `k` would come back near zero, wrongly implying 2026 carries no
information. Production is the opposite case: ZiPS 2027 has never seen 2026.

The correct analog uses `ZiPS_Y` as the base, since it was built knowing only through
Y-1. Predicting year Y+1 from it puts the baseline two years ahead of its last observed
season, matching production exactly (ZiPS 2027 saw through 2025 and is asked about
2027):

```
predicted_{Y+1} = ZiPS_Y * age_ratio + k * w * (actual_Y - ZiPS_Y)
```

`age_ratio` corrects for `ZiPS_Y` being targeted at year Y rather than Y+1, whereas
ZiPS 2027 already has one year of aging baked in. Take it from ZiPS's own 2026-to-2027
ratios, the same source the existing implementation uses for its out-year scaling.
Fit `k` per category by minimizing error against the observed `actual_{Y+1}`.

Fit out of sample (hold out a year-pair, or fit on three pairs and evaluate on the
fourth) so the reported `k` is not the in-sample optimum. Report:

- the fitted `k` per category, for rates and for playing time
- how it compares to `k = 0` (ignore the season) and `k = 1` (full transfer)
- how stable the fitted value is across the four year-pairs

Note the one substitution this makes: no genuine out-year ZiPS vintage exists for past
seasons (only the 2027/2028 pair generated in 2026), so the baseline is an
age-adjusted same-year projection standing in for a true out-year line. The physical
quantity being measured, how much of a single season's surprise persists beyond what
the baseline already knew, is the same. This limitation is stated rather than hidden,
and the sanity check on it is that the fitted `k` should land well away from zero; a
near-zero result is more likely a setup error than a finding.

The result is then reviewed before deciding where to go next: freeze it, split it
finer, or bring the Statcast/Savant signals in as a refinement. That decision is made
on the evidence, not pre-committed here.

## 7. Cross-team comparison

The universe is all rostered players across the ten teams: `cache:roster` (own team)
plus `cache:opp_rosters` (nine opponents, 23-25 players each, kept current by the
refresh). Roughly 245 players.

Each team's projected trio is its top 3 by absolute 2027 value. The league table ranks
the ten trios.

**There is no circularity to resolve.** First-pick par is a scalar subtracted uniformly
from every player, so it cannot reorder anyone, within a team or across teams. The
projected trios are therefore determined by absolute value alone, the 30 kept players
fall out of that, and par is computed once afterward. A single pass, no fixed point.

**Trios are compared on absolute 2027 VAR, not relative.** Comparing on relative value
would require each team's 2027 draft slot, which depends on final standings that do not
exist yet. Absolute gives a clean "how much talent is each team retaining." The owner's
own decision still uses relative-to-own-par.

## 8. Timing caveat

Rosters are live and current; the refresh keeps them so. The caveat is the calendar,
not data freshness. Keeper eligibility runs off whoever is on the roster at season's
end, and it is July. Trades and waiver claims between now and October will change who
is eligible. The ranking stays accurate as rosters move, but "my projected trio" is
provisional until rosters freeze, and a player ranked #2 today can be gone by
September.

## 9. Increments

**Increment 1: the calibration study.** Standalone. Needs only the five ZiPS vintages
and MLB actuals; no VAR machinery. Produces the fitted `k` values and the comparison
described in section 6. Calibration comes first because it is the part whose answer is
unknown.

Review point: look at what it found and decide together where to go next.

**Increment 2: the value pipeline and outputs.** Mechanical once `k` exists. Builds
`updated_2027` and `updated_2028`, scores them through the existing SGP and VAR path,
constructs first-pick par, and emits the three outputs plus the cross-team table.

## 10. Relationship to existing code

`analysis/keeper_value.py` (360 LOC) and `analysis/keeper_trades.py` (192 LOC) survived
the #265 purge and currently implement a discounted 3-year VAR sum with its own
out-year folding (anchor scaled by ZiPS year-over-year ratios, then regressed 60%
toward raw ZiPS). This design was written independently of that implementation rather
than as a refinement of it, so its choices are not inherited by default.

Two of its choices are superseded here: the multi-year discounted horizon (section 3)
and the ratio-scaling fold-in, which is a fixed full-transfer-then-regress rule with no
empirical basis for its 0.6 constant (section 5 replaces it with a calibrated residual
transfer). Its SGP and VAR plumbing, position handling, and playing-time treatment
remain useful and should be reused where they fit rather than rewritten.

What becomes of the existing module and of `keeper_trades.py` is settled in increment
2, once the new metric exists and the overlap is concrete.

## 11. Non-goals

- Rebuilding a projection system. ZiPS out-year stays the baseline; this feature
  adjusts it.
- Per-player or classifier-based skill-versus-luck labeling. The fold-in is a
  coefficient applied uniformly, shrunk only by sample size. Whether finer structure
  earns its complexity is a question for the review point after increment 1, decided on
  evidence.
- Trade evaluation. Out of scope here; `keeper_trades.py` is untouched by this design.

## 12. Verified facts underpinning this design

Confirmed during design, listed so implementation does not re-derive them:

- ZiPS out-year now loads for both player types after the missing pitcher exports were
  added: 2027 hitters (1901, 75) / pitchers (1838, 70); 2028 identical shape. Before
  this, `scripts/keeper_value.py` hard-failed on the missing files and could not run.
- Downloaded file identity was verified by content, not filename: the 2027 hitters
  download matches the existing March export exactly; of the two pitcher files, the
  2027 one drifts less from the 2026 ZiPS baseline than the 2028 one (mean |dIP| 4.45
  vs 7.81, |dSO| 4.26 vs 7.38), the expected one-year versus two-year aging pattern.
- ZiPS 2026 on disk is a preseason full-season projection (mean PA 400.9, max 696), the
  correct residual denominator.
- Every one of the 1901 players in ZiPS 2027 has a 2026 ZiPS counterpart, so the
  residual is defined league-wide.
- `keepers/mlb_stats.fetch_mlb_season` works for arbitrary historical years and returns
  the roto categories keyed by MLBAM (`stat.runs`, `stat.homeRuns`, `stat.avg`,
  `stat.atBats`, `stat.stolenBases`, `stat.rbi`, `player.id`). It had no callers before
  this feature.
- `draft_value.ParCurve` / `par_for_slot` are backward-looking, built from actual
  historical picks, and its `keeper_par` is the mean VAR of kept players. Neither is
  the forward-looking first-pick par this design needs; that is a small new
  construction, not a reuse.

## 13. Open questions for implementation

- Which VAR scale season to use for 2027 valuation: rebuild replacement levels from the
  updated 2027 projections, or reuse the 2026 scale. Rebuilding is more correct;
  confirm the cost.
- Whether playing-time persistence should be split further than one rate/one PT
  coefficient (for example, starters versus relievers). Deferred to the increment 1
  results.
- Position eligibility for 2027 is taken from current positions; no attempt is made to
  project position changes.
