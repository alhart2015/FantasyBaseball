# Split playing-time/durability out of the keeper `luck` family

- **Issue:** #288
- **Predecessor:** #277 (`2026-07-30-decompose-luck-design.md`) -- read it first;
  this supersedes its parameterization without contradicting its evidence.
- **Status:** IMPLEMENTED on `feat/288-pt-durability-split`. Section 3 records what
  shipped, which differs from what was designed -- read it, not the session notes.
- **Handoff superseded:** `docs/superpowers/HANDOFF-288-pt-durability-split.md`

## Problem

`luck = value_pct - skill_pct` **exactly**, so the shipped hitter composite
`(1.0*skill + 0.8*luck - 0.2*batted_ball + 0.2*future + 0.3*age) / 2.1` is an
algebraic rewrite of:

```
(0.8*value + 0.2*skill - 0.2*batted_ball + 0.2*future + 0.3*age) / 2.1
```

38% of a keeper's grade is last year's raw roto line and only 10% is his
peripherals. SGP is a counting total, so that 38% is rate and volume welded
together. Juan Soto took 359 PA in an injury-shortened 2026 and scores
`luck = -16` against a 96th-percentile skill line (.297 xBA, .422 xwOBA, 157
wRC+): the model charges an injury as a permanent demerit and ranks him 21st,
where zeroing `luck` ranks him 4th.

### `luck` is a residual, not a measurement

Nothing about `luck` is measured. It is whatever roto production the five hitter
skill columns (`barrel_pct, barrel_pa_pct, xwoba, xba, wrc_plus` -- all contact
quality) fail to explain. Measured against it (mean Spearman rho, 2022->23,
2023->24, 2024->25 hitters):

```
playing time            +0.662
SB rate                 +0.466
R rate                  +0.284
batted-ball (avg-xba)   +0.251   <- the only component that is actually luck
RBI rate                +0.031
HR rate                 -0.083
```

The BABIP-type luck the name implies is the **third** largest component. The two
largest are playing time and speed -- speed because the peripherals carry no
speed term at all (`corr(SB rate, skill_pct) = -0.129`), so the entire roto value
of a player's legs is booked as luck.

This makes the +0.8 weight defensible but its stated justification false. The
weight is paid for **durability and stolen bases**, both repeatable, not for
luck. Rewarding luck at +0.8 is indefensible; rewarding availability and speed at
+0.8 is obviously correct. Same coefficient, different model.

## Goals

- Replace the residual with families that each measure one named thing, so every
  fitted weight answers a question statable in English.
- Give playing time, speed and batted-ball luck their own homes; stop paying for
  lineup context (R/RBI rate), which is a team property that does not travel with
  a traded player.
- Stop charging injury-shortened elite seasons as permanent demerits.
- Keep **one** composite producing **one** ranking.

## Non-goals

- No IL-stint scrape or injury-propensity model (option B; stays deferred with
  #251/#253). Revealed PA/IP is the outcome of injuries and measures durability
  directly.
- No change to the backtest objective (Spearman rho vs next-year SGP percentile).
- No second production config; one parameterization ships per pool.
- No separate displayed availability number -- see "Corrections to the handoff".

## Design

### 1. Parameterization: drop the residual

```
hitter    skill, speed, durability, batted_ball, future, age
pitcher   skill,        pt,         batted_ball, future, age
```

Pitchers keep raw `pt` rather than `durability` -- a split verdict the backtest
forced; see Evidence.

- `skill` -- the existing five peripherals, unchanged, pinned at 1.0.
- `speed` -- **SB SGP per PA**, its own family with its own fitted weight.
  Hitters only.
- `durability` / `pt` -- the volume dimension. Hitters get the memory-carrying
  form (section 3); pitchers get raw same-season `pt_pct`.
- `batted_ball` -- `avg - xba` / `fip - era`, unchanged; the direct measurement
  of the true-luck component.
- `future`, `age` -- unchanged.
- **No `luck` family.** Nothing whose definition is "whatever is left over."

Speed must be its own family, not a sixth `SKILL_COLUMNS` entry. Equal weighting
within the skill family gives it 1/6 weight, which measurably fails to remove it
from the residual (`corr(residual, SB rate)` moves only +0.466 -> +0.461).

### 2. Why the residual cannot simply be cleaned

Playing time cannot be netted out of `luck` while `value_pct` is a counting total
and `skill_pct` is a rate: the leftover volume dimension *is* the +0.662
correlation. Redefining the residual against a rate (`rate_pct - skill_pct`,
both PT-neutral) does purge it (+0.662 -> +0.179), but what surfaces underneath
is lineup context (R rate +0.562, RBI +0.374) plus batted-ball (+0.502), and the
cleaned residual still predicts next season (0.279 SGP / 0.309 rate) -- i.e. it
still holds real signal we cannot name. Hence: no residual at all.

### 3. Durability estimator (AS SHIPPED -- differs from the design above)

The ZiPS-anchored estimator described in the design session was NOT what shipped.
It could not be validated: it needs historical out-year PA projections for
2022-2024 that do not exist. What shipped is simpler and testable on the real
holdout.

**`durability` = 0.75 * pt_pct(T) + 0.25 * pt_pct(T-1)**, both WITHIN-SEASON
percentiles over every player in that season.

- **Why percentiles**: seasons of different length become comparable for free. An
  in-progress season ranks its players against each other exactly as a completed
  one does, so nothing needs prorating and a COVID-2020 pull would need no
  rescale. This replaces the design's explicit proration step.
- **Why 0.75**: not tuned and not grid-searched -- within-family shape is fixed by
  argument here as it is for `skill_percentile`'s equal weighting. It is ZiPS's
  own revealed recency preference: regressing its out-year PA on the history it
  had seen recovers lag weights of 0.340 / 0.118, a 0.74 / 0.26 split. That is all
  that survives of the ZiPS-anchored design, and it survives as a constant rather
  than as a pipeline.
- **Whole-league base, not the qualified pool**: a veteran who took 100 PA while
  hurt must score genuinely low. Filtering him out would route him to the
  missing-prior fallback and hand him a clean slate -- the exact failure the family
  exists to prevent, and the same reasoning that killed the hard gate.
- **Missing prior falls back to the current season**: a rookie is judged on what
  he has shown (Sal Stewart), not charged for a career he has not had.
- **Enabled by one fetch**: BBRef 2021, pulled through the existing
  `fetch_or_cache` wrapper with `max_age=None`. Cached data was 2022-2026, so
  only the T=2022 transition was short of a prior season. 2020 is never touched.

**Known hole**: a veteran who missed an ENTIRE prior season does not appear in
that season's pull either, so he gets the rookie's benefit of the doubt instead of
the maximal penalty he deserves. That errs toward not penalizing, which is the
safer direction for a keeper board, but closing it needs a tenure signal from T-2.

**Tenure weighting did not ship.** The design's `w = min(1, tenure/2)` confidence
weight was part of the ZiPS-anchored estimator; with a two-season percentile blend
there is no absolute baseline to shrink toward, so it has no role. The rookie case
it protected is handled by the missing-prior fallback instead.

## Evidence

`--backtest`, fit 2022/2023, holdout 2024.

```
HITTER                  holdout      fit    noise
baseline                 0.7085   0.6770   0.0172
A: pt+luck               0.6878   0.6825   0.0217
B: pt+batted_ball        0.6812   0.6682   0.0073
C: luck-batted_ball      0.7002   0.6866   0.0241   (pre-#288 shipped)
D: direct                0.6958   0.6835   0.0147   skill=+1.00 speed=+0.40 pt=+1.00 bb=-0.20 future=+0.40 age=+0.45
E: durability (SHIPPED)  0.7097   0.6717   0.0126   skill=+1.00 speed=+0.40 durability=+1.20 bb=+0.00 future=+0.20 age=+0.45
skill only               0.5184                     (null floor)

PITCHER                 holdout      fit    noise
baseline                 0.4962   0.4937   0.1427
B / D (SHIPPED)          0.4932   0.4904   0.1078   skill=+1.00 pt=+0.60 bb=+0.00 future=+0.40 age=+0.15
C: luck-batted_ball      0.5094   0.5021   0.1321   (pre-#288 shipped)
E: durability            0.4873   0.4851   0.1232
skill only               0.2798                     (null floor)
```

**Split verdict, which `FAMILIES` being per-pool already anticipated.** Hitters
take `durability`; pitchers keep raw `pt`, where durability memory measurably
HURT (0.4873 vs 0.4932). Plausible: pitcher IP is role-driven (SP vs RP) and a
reliever's prior-year IP says little about his next-year workload.

Hitter-E is the best holdout of any candidate and has the lowest noise, so unlike
D it is not trading rho for interpretability. It still only edges baseline by
0.0012 against a 0.0126 noise band, so this is NOT a claim to have beaten the
residual -- the panel cannot resolve these. The case remains interpretability,
now without a predictive cost.

### Two deliberate overrides of the grid

1. **`batted_ball` pinned to -0.2 in both pools.** Left free, the grid zeroes it
   once `durability` absorbs the volume signal -- and the 2026 board then PROMOTES
   the two luckiest bats on it (Abrams bb=93 to rank 5, Otto bb=97 to rank 26),
   undoing what #277 shipped for. Pinning costs 0.0092 hitter rho (inside the
   0.0126 noise band) and restores the demotion (Abrams 10, Otto 39, Rafaela 78).
   For pitchers the same pin IMPROVES the holdout outright (0.4932 -> 0.5018), so
   the negative weight still replicates across both pools as #277 found. #277 set
   the precedent for this tiebreak by shipping C over a higher-scoring baseline on
   watchlist grounds.

2. **`durability` kept at 1.2, above `skill`'s pinned 1.0.** This breaks the old
   "skill leads" invariant deliberately: PT predicts next-year PT at 0.607 against
   skill's 0.464, so availability repeats better than talent. Capping at 1.0 was
   measured and costs 0.0012 -- unmeasurable -- so the test was rewritten to assert
   what the grid actually guarantees (skill pinned at 1.0 as the unit) rather than
   an aesthetic the data does not support. Extending the grid to 2.0 makes the fit
   pick 2.0 and the holdout DROP to 0.6968, so 1.2 is not a censored optimum.

### A latent bug this surfaced

`composite` skips zero-weight families BEFORE its `strict` all-NaN check, so
shipping a 0.0 weight silently disables that family's fail-loud guard -- a broken
xba feed would let `--backtest` decide the model on a degraded blend instead of
raising. Pre-existing, but only reachable once a fitted weight hit exactly zero.
Pinning `batted_ball` negative closes it in both pools; the short-circuit itself
is untouched and remains a trap for any future zero-weight family.

### Motivating case

```
2026 hitter board          OLD    NEW
Juan Soto                   20      9     durability=84 vs raw pt=46
Sal Stewart                  4     20     (predicted)
Ceddanne Rafaela            48     78     (#277 objective)
Otto Lopez                  38     39     (#277 objective)
CJ Abrams                   11     10
Yordan Alvarez              10     36     <- SEE OPEN QUESTIONS
```

Soto is fixed, and by the intended mechanism: raw `pt` puts him at the 46th
percentile because `MIN_PT` compresses the qualified pool, while `durability`
carries the memory of his 713/715 PA seasons and lifts him to 84.

## Corrections to the handoff

- **Decision 1 is misrecorded.** The handoff reads Hart's *"that seems like its
  own model and not something we should lump together with luck"* as a decision
  to display talent and availability as two side-by-side numbers. It was a
  statement about the **model term**, not the presentation. Confirmed 2026-08-01:
  one composite, one ranking, no adjustment column. Availability is priced by the
  fitted `pt` weight, not by a user-facing knob.
- **Soto's 359 PA is ~70% availability, not half a season.** 2026 was in progress
  (max 109 G / 513 PA) when the handoff was written. The `luck = -16` finding
  stands; the magnitude framing overstated it.
- **Option C (projected PT only) is dead** -- but not because ZiPS PA is flat. The
  out-year files have never seen the newest season: the 2027 file re-saved
  2026-07-30 is byte-identical in PA and WAR to its 2026-03-25 vintage (0 of 1,829
  matched players differ). ZiPS gets Soto "right" by ignorance, not by modeling.

## Open questions

1. **RESOLVED.** A memory-carrying durability term beats raw `pt` for hitters
   (0.7097 vs 0.6958) and loses for pitchers (0.4873 vs 0.4932). It did not need
   ZiPS history -- a two-season percentile blend plus one BBRef 2021 fetch was
   enough.

1b. **Yordan Alvarez falls 10 -> 36 on the 2026 board**, driven by the new `speed`
   family (sp=12, weight +0.40), NOT by durability. Arguably correct 5x5 logic --
   a 12th-percentile-speed DH really does forfeit SB category value that the old
   `luck` residual buried -- but he was #277's designated control-who-should-not-
   move, so the demotion is unadjudicated. Decide before this is built on.
2. **Backtest power, unchanged.** Two fit seasons and one holdout cannot resolve
   0.01, and every verdict here sits inside that band. Extending fit years needs
   BBRef 2019-2020; 2021 is now cached. Percentile-space durability means COVID
   2020 would need no rescale, removing the objection that blocked this before --
   but 2020's 60 games still make its PT percentiles mean something different.
3. **`MIN_PT` floor** (250 PA / 50 IP) keeps a star who missed most of a season
   off the board entirely, so the durability model cannot be evaluated on the
   players it most needs to get right.
4. **Durability vs role.** Low PT can mean fragile or benched. Both cut next-year
   PT so it may not matter for ranking, but the feature measures *missed time*,
   not *fragility*.
5. **Part-timer contamination** was a property of the ZiPS-anchored estimator that
   did not ship, so it no longer applies. Retained only as the reason that design
   was scoped: Hart's call was that the model must be good for stars and a
   part-timer leaking into the top tier would be dealt with then.
6. **Speed for pitchers** has no analogue; the pools stay asymmetric on two axes
   now (speed, and durability-vs-pt). `FAMILIES` is per-pool so this costs nothing
   structurally, but `--backtest` prints pitcher-D as "== B" because dropping
   `speed` makes them identical.

7. **`composite`'s zero-weight short-circuit** still bypasses the `strict` all-NaN
   guard. Closed here by pinning `batted_ball` negative, not fixed at the source;
   the next family the grid zeroes will reopen it.

8. **The whole-season-missed hole** in `durability` (section 3) -- a veteran who
   missed all of T-1 gets a rookie's benefit of the doubt.

## Reproducing

```bash
python scripts/keeper_rankings.py --study      # family correlation tables
python scripts/keeper_rankings.py --backtest   # candidate bake-off
```

Study scripts from the design session live in the session scratchpad
(`durability_pt_update.py`, `rate_study.py`, `reparam_study.py`,
`bakeoff_noresidual.py`); their logic must be folded into `--study` / `--backtest`
flags during implementation, per #272's regenerable-evidence rule.
