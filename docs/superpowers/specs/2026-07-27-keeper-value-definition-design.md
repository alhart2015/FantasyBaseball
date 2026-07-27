# Keeper value: feature definition and 2026 fold-in design

Issue: #266 (depends on #265, closed by PR #267)
Date: 2026-07-27
Status: design approved; hardened through one adversarial review round

**Supersedes** `2026-07-22-keeper-value-design.md` and
`2026-07-23-keeper-value-current-anchor-design.md` in full. Those two describe the
discounted multi-year metric this design replaces (section 3) and should be treated as
historical. `2026-07-23-keeper-trade-generator-design.md` remains live and untouched;
trade generation is out of scope here (section 12).

## 1. The decision this serves

**Every team keeps exactly 3 players. Keeping is mandatory, not optional.** The 3
keepers consume draft rounds 1-3, so the live 2027 draft begins at round 4. There is no
escalating cost, no salary, and the choice is re-made every offseason: retaining a
player in 2027 does not constrain the 2028 choice.

One consequence drives the whole design. Because a team forfeits rounds 1-3 *regardless
of whom it keeps*, the cost of keeping is a constant, identical across every possible
choice of 3. It therefore cannot influence which 3 to keep.

**So the "which 3" question has a trivial answer: the 3 highest projected 2027 values.**
The feature's real work is everything around that:

1. Projecting 2027 value well, from a baseline that predates the 2026 season.
2. Measuring how much surplus a trio generates over the picks it costs.
3. Comparing the owner's trio against the other nine teams'.

## 2. What "keeper value" means

Three outputs, deliberately kept separate rather than blended into one score.

**Absolute keeper value.** A player's projected 2027 VAR, on the same scale the draft
board uses. This alone answers "which 3 do I keep": take the top 3. It is the only
output that feeds that decision.

**Surplus over forfeited picks.** For a trio ranked 1st, 2nd, 3rd by absolute value,
the surplus of the i-th keeper is `VAR_i - par_i`, where `par_i` is the expected VAR of
the player the owner would have taken at his round-i pick had no keeper system existed.

`par_i` is computed against the **full, undepleted** 2027 pool. This is deliberate and
is what makes the quantity well defined: since keeping is mandatory, the counterfactual
is not "what if I released this player" (impossible) but "what would rounds 1-3 have
returned in a draft with no keepers." In that counterfactual every player is available,
so no keeper set is removed and `par_i` depends on nothing but the pool and the pick
slot.

At snake position 8 in a 10-team league the three slots are overall picks **8, 13, and
28** (round 1 forward: pick 8; round 2 reversed: 11 + (10-8) = 13; round 3 forward:
20 + 8 = 28). `par_i` is the i-th of those ordinals in the pool sorted descending by
projected 2027 VAR.

Surplus is **not a keep/release rule** -- no such decision exists under mandatory
keeping. It is a diagnostic. A 3rd keeper whose surplus is near zero is a weak slot
worth targeting for upgrade by trade; a trio with large surplus is a structural
advantage over the field.

**Trajectory.** A separate 2028 column answering "will he still be keep-worthy next
year." Never blended into the headline; it informs the eye, not the sort.

## 3. Horizon: single-year, and why

The metric is a single year (2027). 2028 appears only as the unblended trajectory
signal.

This is a deliberate break from the existing `analysis/keeper_value.py`, which sums a
discounted 3-year stream (`DEFAULT_DISCOUNT = 0.80`, `DEFAULT_HORIZON = 3`, lines
28-29). That structure prices a multi-year commitment the league rules do not impose:
the 2028 slot is re-chosen in 2028 with 2028 information, so summing years pays today
for an option that will be re-purchased for free.

It also has a concrete defect. With `base_year = 2026` (`scripts/keeper_value.py:49`),
the largest single weight in the sum -- 1.0, roughly 41% of the total -- lands on the
2026 season, which is finished and immutable at the moment the keeper decision is made.

A third argument is epistemic: with a baseline that predates 2026, the out-years are
the *least* reliable inputs. A discounted sum adds the most noise precisely where
confidence is lowest while presenting it as precision.

The counter-case is the 21-year-old whose 2027 line is unremarkable but whose 2029 line
is a star. It resolves without a discounted sum. If his 2027 line is strong, single-year
VAR already captures him. If it is weak, he will be inexpensive in the 2027 draft, so a
keeper slot -- an asset worth a round-1 through round-3 pick -- is the wrong instrument
for retaining access to him.

## 4. The core problem: the out-year baseline is permanently stale

The ZiPS 2027 and 2028 projections on disk were generated 2026-03-25 and know nothing
of the 2026 season.

This was checked, not assumed. A 2026-07-27 download of the ZiPS 2027 hitters file
matched the March export on every one of the 1827 PlayerIds present in both, with HR
identical on all of them. (Of the 1901 rows, 74 did not join; whether those are genuine
pool changes or a join artifact was not resolved, so the accurate claim is "identical
on the overlap," not "byte-identical." Implementation should script this check and
retain both artifacts so it can be re-run -- see section 13.)

The operative conclusion holds: FanGraphs has not regenerated ZiPS out-year projections
since March, and re-downloading will not fold 2026 in. Whatever adjustment this feature
builds is the only path.

## 5. The fold-in mechanism

### 5.1 Decompose first, then fold

Folding raw category residuals is wrong, for two separate reasons.

**Counting stats are not a bucket.** Of the ten scored categories
(`utils/constants.py:13-22`), only AVG/ERA/WHIP are rates. PA/IP are not scored at all;
they enter only as weights. The other seven -- R, HR, RBI, SB, W, K, SV -- are joint
products of a rate and playing time. A raw HR residual conflates "hit for less power"
with "played half a season," which are different information with different persistence.

**Rate residuals are not additive.** AVG enters scoring as marginal hits,
`(player_avg - replacement_avg) * player_ab` (`sgp/player_value.py:19-28`); ERA and WHIP
are weighted by `player_ip` (lines 31-43). Adding an AVG residual while AB also moves
re-weights the rate by a different denominator than the one it was measured on,
silently creating or destroying hits.

So the fold operates on a decomposed line:

```
rate_2027    = ZiPS_2027_rate + k_rate * w * (rate_2026_actual - ZiPS_2026_rate)
pt_2027      = ZiPS_2027_pt   + k_pt          * (pt_2026_actual   - ZiPS_2026_pt)
counting_2027 = rate_2027 * pt_2027
```

- **Rates** are per-PA for hitters (HR/PA, R/PA, RBI/PA, SB/PA, H/AB) and per-IP for
  pitchers (K/IP, W/IP, ER/IP, (BB+H)/IP). Playing time is PA/AB for hitters and IP for
  pitchers.
- **AVG/ERA/WHIP are never folded directly.** Fold their components (H and AB; ER, BB,
  H-allowed and IP) and recompute the rate. Both sides carry these:
  `cache:full_season_projections` exposes `h`/`ab` and `er`/`bb`/`h_allowed`/`ip`, and
  `data/fangraphs.py` parses `H`/`AB` (`HITTING_COLUMN_MAP`) and `ER`/`BB`/`H`
  (`PITCHING_COLUMN_MAP`) from the ZiPS CSVs.
- `k_rate` and `k_pt` are the persistence coefficients, fit per category-group in
  section 6. `k = 0` ignores 2026 (today's stale baseline); `k = 1` transfers the full
  surprise.

### 5.2 The sample-size shrink `w`

`w` applies to the **rate** residual only. A rate observed over few plate appearances
is noisy and must be shrunk toward the projection; a playing-time residual is not noisy
in that way -- it is the observation itself.

```
w = n / (n + n0)      n = realized PA (hitters) or IP (pitchers)
```

Bounded in [0, 1), so it can never amplify. `n0` is a stated default, not a finding:
**200 PA** for hitters, **50 IP** for pitchers. (A linear `PA/600` form was rejected: it
exceeds 1.0 above 600 PA and would amplify the residual for exactly the highest-PT
players.) Because section 6 fits `k` with `w` held fixed, **the fitted `k` is
conditional on these `n0` values**, and the calibration must report that.

Applying `w` to the playing-time residual would be a double-count -- shrinking an
injury signal in proportion to the very playing time the injury suppressed -- and would
make `k_pt` structurally unable to learn from players who lost time. That is exactly the
information a keeper decision most needs, so `w` is deliberately excluded there.

### 5.3 Saves are excluded from out-year valuation

**ZiPS 2027 and 2028 contain no saves data at all.** Verified: `SV` is NaN in 0 of 1838
rows -- that is, 100% missing -- in both out-year pitcher files, against 1838/1838
populated (sum 1064) in ZiPS 2026. `HLD`, `QS` and `BS` are likewise all-NaN but are
unscored.

This is silent, not loud: `safe_float` (`utils/constants.py:372-377`) coerces NaN to
0.0 without error, so an unguarded implementation produces wrong numbers rather than a
failure. Scored naively, every closer's out-year VAR collapses (Fairbanks +0.90 to
-3.08; Mason Miller +3.62 to -0.97) while starters are untouched, corrupting the SP/RP
ordering wholesale. Pete Fairbanks is on a current keeper list in `config/league.yaml`.
Note also that the fold-in cannot repair this: the residual is a *difference*, so with
`ZiPS_2027_SV = 0` the best reachable estimate for an elite closer is a few saves.

**Decision: SV is excluded from out-year (2027/2028) valuation entirely, and the
exclusion is disclosed in the output.** Out-year scoring runs on the nine remaining
categories.

Two requirements follow, and neither is optional:

- **Replacement levels must be recomputed without SV.** If relievers are scored on nine
  categories but measured against an SV-inclusive RP floor, they are compared to a
  standard they can no longer reach and every reliever looks uniformly terrible. The
  out-year scale must drop SV from the denominators and from the RP replacement line.
- **Every reliever row must carry an explicit `sv_excluded` flag**, and the report must
  state that closer value is understated. This is a known, disclosed bias, not a
  silent one.

Consequence to accept: relief pitchers are systematically undervalued in the out-years,
and a save-dependent keeper cannot be compared fairly against a hitter. That is the
cost of the missing data, and it is stated rather than papered over.

### 5.4 Edge cases

- **No 2026 MLB playing time.** The mechanism is `w = 0` (and a zero PT residual),
  which zeroes the correction *regardless of residual magnitude*. It is not that "the
  residual is zero" -- for a prospect ZiPS projected at 300 PA who never debuted, the PT
  residual is -300. State the mechanism correctly, because an implementer who applies
  the residual before the shrink, or who floors `w`, gets a silently different and badly
  wrong answer.
- **A regular lost to injury before the season** also lands at `w ~ 0` and sails through
  with his stale 2027 line unchanged. That is a defensible choice, not a clean fall-
  through, and the output should flag it.
- **In the 2027 file but absent from ZiPS 2026: 95 of 1838 pitchers (5.2%).** No
  baseline exists, so no residual is defined. Behavior: fall through to raw ZiPS 2027,
  and flag the row as unfolded rather than silently mixing it with folded players.
  (Hitters are unaffected: all 1901 have a 2026 counterpart.)
- **Absent from ZiPS 2027 entirely** (2027 debutants): no projection at all. Reported
  separately as unrankable rather than scored zero.
- **Two-way players.** Ohtani appears twice in the 2027 pool -- rank 1 as a hitter, and
  far down as a pitcher. Trio selection must dedupe by player, not by row, and must
  count him as one keeper slot. `config/league.yaml` annotates him "batter only";
  `analysis/draft_value.py:485-488` shows how the existing par curve solved the same
  problem.
- The same `k`, `w` and exclusions apply to ZiPS 2028 for the trajectory column. This
  is an approximation: `k` is calibrated on one-year-forward persistence and a surprise
  plausibly decays differently two years out. Acceptable because trajectory never enters
  the sort, but it must not be reused as a calibrated 2028 valuation.

### 5.5 Join keys

Every join is by ID; name matching is used only where no ID exists.

- ZiPS-to-ZiPS across vintages: `PlayerId` (FanGraphs).
- ZiPS to MLB actuals: `MLBAMID` (present in every ZiPS vintage) to `player.id`.
- ZiPS to `cache:full_season_projections`: `fg_id` / `mlbam_id`, populated on 100% of
  its rows.
- `config/league.yaml` keeper names to the pool: **normalized name matching is
  required.** Exact-name joining drops 3 of the 30 keepers today -- `Jose Ramirez`,
  `Julio Rodriguez`, `Ronald Acuna Jr.` versus ZiPS's accented spellings. Use the
  existing `draft/keepers.find_keeper_match` and `sgp/rankings.rank_key`, per the
  cross-cutting convention in CLAUDE.md.

## 6. Calibrating k (increment 1)

### 6.1 The baseline must not already know year Y

This is the trap in the setup. Every ZiPS file on disk is a preseason projection for its
own year, so `ZiPS_{Y+1}` was built knowing year Y. Using it as the base would fit how
much of a surprise ZiPS has *already absorbed*, driving `k` toward zero essentially by
construction -- and shipping the conclusion that 2026 carries no information about 2027,
the exact opposite of the truth, since ZiPS 2027 has never seen 2026.

The correct analog uses `ZiPS_Y` as the base, since it was built knowing only through
Y-1. The information sets then align exactly: `ZiPS_Y` has not absorbed year Y, and
`ZiPS_2027` has not absorbed 2026.

### 6.2 The fit

Per category-group, for rates and for playing time separately:

```
predicted_{Y+1} = a * ZiPS_Y + k * w * (actual_Y - ZiPS_Y)      [rates]
predicted_{Y+1} = a * ZiPS_Y + k     * (actual_Y - ZiPS_Y)      [playing time]
```

`a` is a **fitted per-category scale term**, not a supplied age ratio. This replaces an
earlier `age_ratio` construction that is not implementable: ZiPS CSVs carry **no Age
column** (verified across all six vintages), and the natural substitute -- each player's
own ZiPS 2026-to-2027 ratio -- is unavailable for 57% of the 2022 cohort (650 of 1504
appear in both) and is conceptually wrong anyway, since it encodes aging at the player's
2026 age rather than his year-Y age. Worse, it would smuggle year-Y information back
into the supposedly clean base, reintroducing the very contamination 6.1 exists to
prevent. A fitted `a` absorbs league-level drift and the same-year-versus-out-year
targeting difference, is estimated from the data, and needs no external age source.

Fit by minimizing error against observed `actual_{Y+1}`, out of sample.

### 6.3 Three year-pairs, not four

The fit needs `ZiPS_Y`, `actual_Y`, and `actual_{Y+1}`. A 2025 pair would require a
complete 2026 season; today is 2026-07-27 and `config/league.yaml` sets
`season_end: 2026-09-28`, with 2026 roughly two-thirds played. So the usable pairs are:

**2022->2023, 2023->2024, 2024->2025. Three.**

Protocol: leave-one-pair-out cross-validation across the three, reporting per-held-out-
pair error. Report the fitted `k` and `a` per category-group, their comparison against
`k = 0` and `k = 1`, and their stability across the three pairs. A fourth pair becomes
available after the 2026 season completes and the study should be re-runnable then.

### 6.4 Survivorship must be handled explicitly

`actual_{Y+1}` exists only for players who kept playing. Measured: of players with >=100
PA in year Y, 75.5% / 77.7% / 79.5% reach >=100 PA in Y+1 across the three pairs -- so
20-25% vanish annually (114, 102, 93 players). Those are the injuries, demotions and
washouts a keeper decision most needs priced.

Fitting on survivors alone measures "persistence *given* the player kept playing," which
biases `k_pt` upward -- the one coefficient most in need of honesty. The study must
report the fit both ways: survivors only, and with non-survivors included at their
actual (near-zero) playing time. The difference between them is itself a finding.

Sample size is thin and must be reported: roughly 350 matched hitters per pair at the
>=100 PA threshold, about 1050 across all three before any split by category.

### 6.5 The train/serve gap must be measured, not ignored

`k` is fit on `actual_Y - ZiPS_Y` -- fully observed, ZiPS against ZiPS. It is applied in
production to `full_season_2026 - ZiPS_2026`, which differs in two ways:

- **The minuend is not fully observed.** Across the 218 rostered players in
  `cache:opp_rosters`, the rest-of-season component is **34.9%** of full-season PA/IP.
  A third of the "surprise" is itself a regressed projection, so the applied residual is
  attenuated relative to the one `k` was fit on -- and that ratio shrinks weekly, making
  the metric drift without `k` changing.
- **The minuend is not ZiPS.** `config/league.yaml` blends steamer/zips/atc/the-bat-x/
  oopsy at 0.20 each. Subtracting a ZiPS-only baseline from a 5-system blend puts
  inter-system level differences into the residual as spurious signal.

Requirement: the study must include a sensitivity check that re-fits with the year-Y
actual truncated to a comparable fraction of the season and blended forward, quantifying
the attenuation. If it is material, production must either correct for it or use a
ZiPS-only 2026 line as the minuend. This is a required output of increment 1, not a
follow-up.

### 6.6 Data-ingest requirements

- **IP arrives as a string in baseball notation.** `stat.inningsPitched` returns values
  like `"1.2"` and `"0.1"`, meaning 1 2/3 and 1/3 innings. Naive `float()` yields 1.2
  instead of 1.667 -- a systematic understatement landing directly in the pitching PT
  residual and in every ERA/WHIP weighting. ZiPS IP is decimal, so differencing without
  conversion is invalid. `stat.era` and `stat.whip` are also strings and need explicit
  numeric coercion with a zero-IP guard.
- **Expected row counts, to assert against:** hitting returns 794 (2022), 769 (2023),
  742 (2024), 765 (2025) rows at `playerPool=all`. `_fetch_mlb_season`
  (`keepers/mlb_stats.py:51-52`) breaks on the first short page, so a silent truncation
  would be indistinguishable from a complete pull -- pin these counts as assertions.
- **Coverage is limited and must be stated.** Against ZiPS 2023's 1716 hitters, at most
  ~45% can ever match; the matched calibration sample is ~500-530, or ~350 at the >=100
  PA threshold.
- **`fetch_or_cache` never invalidates** (`keepers/cache.py:23-38` returns any non-empty
  cached CSV unconditionally). Correct for completed historical seasons. **In-progress
  seasons must not go through it**, or must use a date-stamped path -- otherwise the
  first 2026 pull freezes permanently and later runs silently reuse a stale mid-season
  snapshot.

### 6.7 Acceptance criteria for increment 1

The study is complete when it reports, per category-group:

1. Fitted `k` and `a`, with leave-one-pair-out error, against the `k = 0` and `k = 1`
   baselines.
2. Stability of `k` across the three pairs.
3. The survivorship comparison (6.4) and the train/serve sensitivity (6.5).
4. An explicit statement that `k` is conditional on the chosen `n0` values (5.2).

It **passes** if out-of-sample error at the fitted `k` beats both `k = 0` and `k = 1` on
a majority of held-out pairs. It **fails to a fallback** otherwise, shipping whichever
of `k = 0` / `k = 1` performed best, with that recorded as the finding.

A fitted `k` near zero should be treated as a suspected setup error -- most likely
baseline contamination per 6.1 -- and investigated before it is reported as a result.
This is a diagnostic, not the acceptance bar.

Whether to go further (finer category splits, Statcast signals) is decided at the review
point on this evidence, not pre-committed here.

## 7. Cross-team comparison

Because "which 3" is trivial under mandatory keeping (section 1), cross-team comparison
is where most of the feature's value sits.

The universe is all rostered players across the ten teams: `cache:roster` plus
`cache:opp_rosters` (nine opponents, 23-25 players each, kept current by the refresh),
roughly 245 players. Each team's projected trio is its top 3 by absolute 2027 value,
deduped by player (5.4). The league table ranks the ten trios.

**Trios are compared on absolute 2027 VAR, not surplus.** Per-team surplus would need
each team's 2027 draft slot, which depends on final standings that do not exist yet.
Absolute gives a clean "how much talent is each team retaining." The owner's own surplus
figures (section 2) use his known position-8 slots and are reported separately.

There is no circularity anywhere in this: `par_i` is computed against the full
undepleted pool (section 2), so it does not depend on which players are kept, and trio
selection depends only on absolute value. A single pass, nothing to converge.

## 8. Timing caveat

Rosters are live and current; the refresh keeps them so. The caveat is the calendar, not
data freshness. Keeper eligibility runs off whoever is on the roster at season's end, and
it is July. Trades and waiver claims between now and October will change who is
eligible. The ranking stays accurate as rosters move, but "my projected trio" is
provisional until rosters freeze, and a player ranked #2 today can be gone by September.

Separately, the 2026 residual itself sharpens as the season completes -- the 34.9%
rest-of-season share (6.5) shrinks toward zero.

## 9. Increments and delivery

**Increment 1: the calibration study.** Standalone -- needs only the ZiPS vintages and
MLB actuals; no SGP, no VAR, no board, no cache beyond the ingest's own.

- Delivery: a script under `scripts/`, writing a results table to `data/analysis/`.
- Tests: unit coverage for the IP notation conversion (6.6), the `w` function including
  its zero-PT and high-PT bounds (5.2), the rate/PT decomposition and recombination
  (5.1), and the fit on a synthetic dataset with a known planted `k`.
- Acceptance: section 6.7.

Review point: examine the findings and decide together where to go next.

**Increment 2: the value pipeline and outputs.** Builds `updated_2027` / `updated_2028`,
scores them through the existing SGP and VAR path with the SV exclusion and its
recomputed replacement levels (5.3), constructs `par_1..3`, and emits the three outputs
plus the cross-team table.

- Delivery, tests, and the fate of the existing `scripts/keeper_value.py` are specified
  at the start of increment 2, once the calibration result is known. It is an open
  question (section 13) whether increment 2 extends that script or replaces it.

## 10. Relationship to existing code

In scope for reconciliation, with LOC verified:

| Path | Size | Disposition |
|---|---|---|
| `analysis/keeper_value.py` | 360 LOC | Superseded metric; plumbing reusable |
| `analysis/keeper_trades.py` | 192 LOC | Untouched by this design |
| `scripts/keeper_value.py` | ~15 KB | Fate decided at increment 2 |
| `scripts/keeper_trades.py` | ~9 KB | Untouched |
| `tests/test_analysis/test_keeper_value.py` | - | Must be reconciled in increment 2 |
| `tests/test_scripts/test_keeper_value_script.py` | 275 LOC | Must be reconciled in increment 2 |
| `tests/test_analysis/test_keeper_trades.py` | - | Untouched |
| `tests/test_scripts/test_keeper_trades_script.py` | 48 LOC | Untouched |
| `tests/test_keepers/` | - | Extended by increment 1 |

This design was written independently of `keeper_value.py` rather than as a refinement,
so its choices are not inherited by default. Two are superseded: the multi-year
discounted horizon (section 3), and the ratio-scaling fold-in, whose
`DEFAULT_OUT_YEAR_REGRESSION = 0.6` is justified in-code only by the comment
`0.6 = "mostly ZiPS"` (lines 30-34).

One thing the incumbent does better and which must be preserved: `_scale_line`
(lines 88-96) explicitly holds the anchor flat on a NaN out-year cell. That guard is the
only reason closers are currently scoreable at all (5.3). Any replacement must handle
missing out-year data deliberately rather than relying on `safe_float`'s silent NaN-to-0.

Its SGP and VAR plumbing, position handling, and playing-time treatment remain useful
and should be reused where they fit rather than rewritten. Per CLAUDE.md, existing tests
are guardrails: where increment 2 changes behavior they cover, the change must be
justified explicitly, not absorbed by editing assertions.

## 11. Verified facts underpinning this design

Confirmed during design and review, so implementation need not re-derive them:

- ZiPS out-year loads for both player types after the missing pitcher exports were added:
  2027 hitters (1901, 75) / pitchers (1838, 70); 2028 identical. Before this,
  `scripts/keeper_value.py` hard-failed on the missing files.
- **`SV` is 100% NaN (0 of 1838 populated) in both out-year pitcher files**, against
  1838/1838 (sum 1064) in ZiPS 2026. `HLD`/`QS`/`BS` likewise, but unscored.
- **95 of 1838 pitchers in ZiPS 2027 have no ZiPS 2026 counterpart.** All 1901 hitters
  do.
- ZiPS CSVs carry **no Age column** in any vintage.
- Downloaded out-year file identity was verified by content: the 2027 pitcher file
  drifts less from the 2026 ZiPS baseline than the 2028 one (mean |dIP| 4.45 vs 7.81,
  |dSO| 4.26 vs 7.38), the expected one-year versus two-year aging pattern.
- ZiPS 2026 is a preseason full-season projection (mean PA 400.9, max 696). Note it
  hedges playing time pool-wide: only 2.9% of rows exceed 600 PA and 18.2% fall below
  300, so the PT residual carries a large systematic component (+58 mean PA for 2025
  regulars) that is not "surprise." This is why PT is fit with its own `a` and `k`.
- `keepers/mlb_stats.fetch_mlb_season` works for arbitrary historical years and returns
  the roto categories keyed by MLBAM. Hitting verified; pitching returns `stat.wins`,
  `stat.saves`, `stat.era`, `stat.whip`, `stat.strikeOuts`, `stat.inningsPitched`, with
  the string/notation caveats in 6.6. It had no callers before this feature.
- `position_aware_replacement_levels` (`sgp/replacement.py:240-279`) is a pure function
  of denominators plus the AVG/ERA/WHIP rate baselines and is **independent of the live
  pool**; only `calculate_replacement_rates` (line 93) is pool-derived.
- `draft_value.ParCurve` / `par_for_slot` are backward-looking, built from actual
  historical picks, and its `keeper_par` is the mean VAR of kept players. Neither is the
  forward-looking par of section 2; that is a small new construction, not a reuse.

## 12. Non-goals

- Rebuilding a projection system. ZiPS out-year stays the baseline; this feature adjusts
  it.
- Per-player or classifier-based skill-versus-luck labeling. The fold-in applies
  coefficients uniformly, shrunk only by sample size. Whether finer structure earns its
  complexity is decided at the review point after increment 1, on evidence.
- Modelling out-year saves. Section 5.3 excludes and discloses rather than models.
- Trade evaluation. `keeper_trades.py` and its spec are untouched.

## 13. Open questions for implementation

- **Should replacement rates be recomputed off the 2027 pool?** Only the three
  AVG/ERA/WHIP `repl_rates` are pool-derived (section 11), so this is narrower than it
  first appears. The substantive version of the question: the 2027 live draft is 3
  rounds shorter and 30 players shallower, while `STARTERS_PER_POSITION` is
  `roster_slots x num_teams` -- should keeper depletion shift the positional floors at
  all? Note this interacts with the SV-excluded RP floor (5.3).
- Whether playing-time persistence needs a finer split than one coefficient (starters
  versus relievers, for instance). Deferred to the increment 1 results.
- Whether increment 2 extends `scripts/keeper_value.py` or replaces it.
- Position eligibility for 2027 is taken from current positions; no attempt is made to
  project position changes.
- The section 4 vintage check should be scripted and both artifacts retained, so the
  "FanGraphs has not regenerated" claim can be re-verified rather than trusted.
