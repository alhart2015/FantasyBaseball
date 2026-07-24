# Keeper Breakout/Mirage Diagnostic - Design

Date: 2026-07-24
Issue: #258 (Keeper: DARKO-style true-talent trajectory + breakout/decline detection)
Status: approved design, pre-implementation

## Problem

The keeper metric (`src/fantasy_baseball/analysis/keeper_value.py`) separates a player's
current-season **level** (the anchor line) from the multi-year **aging shape**. The
current-anchor fix (PR #259, `--anchor current`) corrects the level to reflect this
season's YTD+ROS data -- but it believes the surface line at face value. A player on a
hot BABIP run gets an inflated anchor; a player whose power spike is backed by barrel
rate gets the same treatment as one whose spike is HR/FB luck. The aging shape, meanwhile,
is borrowed from ZiPS's 2027/2028 out-year projections, which were published before the
2026 season and therefore cannot see this year's breakouts.

The user does not trust this enough to make keeper decisions from it. The stated need is
**trust**, via one of two levers: (a) a backtest proving the approach separates real
breakouts from unsustainable mirages, or (b) fresher public data to decide from -- and a
new model only if those force it.

Critically, a keeper decision is **cardinal, not ordinal**. Knowing three players are all
"real breakouts" does not tell you which to keep. The deliverable must rank players by
forward value, not merely label them.

## Goals

1. Produce, per player, a **skill-adjusted current true-talent line**: the surface line
   with its luck-driven component regressed toward the preseason projection and its
   skill-backed component retained.
2. Rank players by the **forward keeper value** of that adjusted line, computed by reusing
   `keeper_value`'s existing SGP -> VAR -> discounted-multi-year machinery.
3. Attach an explainable **label** (`real breakout / lucky mirage / real decline / slump /
   stable`) and a one-line reason, as the "why" behind the number.
4. **Backtest** whether the adjusted line and its ranking beat two baselines --
   surface-believed and pure-ZiPS -- at predicting next-season outcomes, over historical
   data. This is the go/no-go evidence gate.
5. Ship a **report** (CSV + markdown) the user reads at keeper time.

## Non-goals (deferred, each to its own spec, gated on the backtest)

- **Wiring the signal into `keeper_value`** (replacing the surface-believed anchor or the
  ZiPS aging ratio in `_scale_line`). This spec does not modify `keeper_value.py`; it only
  *calls* its public functions.
- **A from-scratch DARKO aging/comps engine** (per-stat aging curves, reliability-weighted
  true-talent update, historical comparables). Only build this if the backtest shows the
  augmentation is insufficient.
- **Park/league-factor adjustment** of the underlying metrics.
- **The within-2026 first-half -> second-half backtest** (only becomes available once the
  2026 season completes).

## Chosen approach

Report-first, automate-only-if-proven. Four isolated units:

```
data/skill_luck/*.csv        (1) DATA LAYER: season xStats + rates + age, cached
        |
        v
analysis/breakout.py         (2) CLASSIFIER: pure; projection+surface+underlying
        |                        -> adjusted line + label + reason
        +--> scripts/run_breakout_report.py   (3) REPORT: current-season ranked resource
        |
        +--> scripts/backtest_breakout.py     (4) BACKTEST: historical, does it beat baselines?
```

Each unit is independently testable. The classifier is pure math (no I/O), mirroring how
`keeper_value.py` sits behind `draft.board`. The report and backtest are thin orchestrators.

### Key reuse

The report reduces to "compute a better anchor, then call existing machinery."
`keeper_value.keeper_value(player_id, name, anchor_line, positions, player_type,
zips_by_year, scale, ...)` already accepts the anchor line as an argument. The report
passes the **skill-adjusted** anchor where today's flow passes the surface-believed one.
We call `keeper_value`'s functions; we do not change them. The single new quantity is the
anchor line, which isolates the one variable being improved and keeps the report directly
comparable to today's number.

## Detailed design

### Unit 1: Season skill/luck data layer

Module: `src/fantasy_baseball/data/skill_luck.py`. Cache dir: `data/skill_luck/`.

Sources (via `pybaseball`, already a project dependency):

- **FanGraphs season** (`batting_stats(start, end, qual)` / `pitching_stats(...)`):
  K%, BB%, BABIP, HR/FB, contact%, and `Age`. `Age` supplies the age feature with no extra
  source. Keyed by FanGraphs id (`IDfg`).
- **Statcast season** (`statcast_batter_expected_stats(year, minPA)` /
  `statcast_pitcher_expected_stats(...)`, plus `statcast_batter_exitvelo_barrels(year)` for
  barrel rate): xwOBA, xBA, xSLG and their gaps vs actual, barrel%. Keyed by MLBAM id.

Coverage window: 2015-2026 (Statcast begins 2015). Hitters throughout; pitcher expected
stats where the leaderboard provides them.

Caching contract:

- Past seasons are immutable -> fetch once, write `data/skill_luck/{source}_{h|p}_{year}.csv`,
  never refetch.
- Current season -> refetch on demand, but **refuse to overwrite a good cache with an empty
  or failed pull** (the ROS-Cloudflare lesson, `project_ros_cloudflare_block_2026_06_04`).
- One function returns a per-player-season joined frame (rates + xStats + age); another
  handles fetch+cache with the fail-safe overwrite guard.

### Unit 1b: Player identity

FanGraphs id, MLBAM id, and the board's `MLBAMID` / `name::player_type` all differ. Build a
**mapping table once** via pybaseball's Chadwick register (`chadwick_register()` /
`playerid_reverse_lookup`), mapping `key_mlbam <-> key_fangraphs`; cache it. Join FanGraphs
and Statcast frames on it, then join to the board on `MLBAMID` (projections and actuals
already carry it). Reuse the existing accent-stripped/lowercased name normalization for
tie-breaks. **Unmatched players are reported and counted, never silently dropped** (the
name-collapse lesson, `project_data_pipeline_audit_2026_05_31`).

### Unit 2: Classifier (pure)

Module: `src/fantasy_baseball/analysis/breakout.py`. No I/O.

Operates in **rate space**, per roto-relevant component, to avoid confounding playing time
with talent. For each rate `r` (hitters: HR/PA, R/PA, RBI/PA, SB/PA, AVG=H/AB; pitchers:
K/IP, W, SV, ERA, WHIP):

```
adjusted_rate[r] = projection_rate[r] + w[r] * (surface_rate[r] - projection_rate[r])
```

- `projection_rate[r]` = the prior. **In the 2026 report** this is the blended preseason
  projection (the same blend the draft board uses, per `config/league.yaml`'s configured
  projection systems), not ZiPS alone. **In the historical backtest** (years lacking a
  blended preseason file) it is a reconstructed Marcel-style prior -- see Unit 4.
- `surface_rate[r]` = the rates of the **current YTD+ROS anchor line** -- the exact same
  anchor `keeper_value --anchor current` consumes (from `cache:full_season_projections`),
  not YTD actuals alone. This guarantees the report's surface column reconciles with today's
  number (Unit 3).
- `w[r]` in [0, 1] = how much of the deviation to believe. `w[r]` is an explicit,
  documented function of:
  - **reliability** (sample size: PA for hitters, IP/BF for pitchers) -- more sample -> higher `w`;
  - **confirmation** from the matching underlying signal -- power (HR, SLG) confirmed by
    barrel% and xSLG; AVG confirmed by xBA vs BABIP; pitcher ratios confirmed by K%/BB%;
    SB is role/health-driven and sticky; SV is role-driven but volatile (closer changes), so
    its confirmation leans on holds/role signals and its `w` stays conservative.

**Playing time is held, not adjusted.** Only rates are shrunk. The adjusted rates are
multiplied by a fixed PT estimate to produce the counting line: the report uses the
**PT-healed projected PA/IP that `keeper_value` already computes** (so an injury-shortened
season is not read as talent loss and PT matches the surface-believed line's PT); the
backtest uses the player's actual year-Y PA/IP. The result is the **skill-adjusted anchor
line** (a counting line in the same shape `keeper_value` consumes).

The **label** is derived from the aggregate, `w`-weighted signed deviation: large positive
believed delta -> `real breakout`; large positive *surface* delta with low `w` ->
`lucky mirage`; symmetric for `real decline` / `slump`; near-zero -> `stable`. The **reason**
is a short ASCII string naming the dominant driver (e.g. `"HR up on barrel 8->14%, xSLG
confirms"` vs `"AVG up on .380 BABIP, xBA flat -> mirage"`).

The `w`-mapping's parameters (reliability curve shape, confirmation weights, label
thresholds) are the knobs the backtest tunes. They live as named module constants with
documented defaults. Phase 3 ships with **domain-prior seed defaults** (chosen from known
stabilization points -- e.g. K% and barrel% stabilize fast, BABIP slow); until Phase 4
validates them, the report labels its numbers **provisional** and not yet trusted for final
decisions.

### Unit 3: Report

Script: `scripts/run_breakout_report.py`. Library entry point in `breakout.py` where
practical (keep I/O in the script, computation in the library, per repo convention).

The report lists **all keeper-eligible players**, sorted by skill-adjusted value, with a
`deviator` flag on those whose surface departs from projection beyond the candidate
threshold (so the breakout/decline cases surface at a glance without hiding the full board).
For each it emits: `skill-adjusted keeper value` (rank key), `surface-believed keeper value`,
the **delta** (the "mirage tax" in roto points), `label`, `reason`, and the key underlying
numbers (wOBA-xwOBA gap, BABIP, barrel%, K%/BB%). Outputs `breakout_report.csv` and
`breakout_report.md`, consistent with existing keeper report artifacts.

**Both value columns come from one `keeper_value.keeper_value(...)` call each -- same
positions, `zips_by_year`, and `scale` -- differing only in the anchor line passed
(surface-believed vs skill-adjusted).** Because the surface-believed call reuses the exact
`--anchor current` anchor-construction path (Unit 2), the `surface-believed keeper value`
column equals today's number **by construction**, not by assertion. Aging is `keeper_value`'s
existing shape, unchanged.

### Unit 4: Backtest

Script: `scripts/backtest_breakout.py`, following the existing backtest scaffold
(`scripts/backtest_recency.py`, `scripts/backtest_sd_calibration.py`: checkpoint / model /
target / metric rows to a results CSV).

Because no in-season projection history exists for past years (ROS is overwritten daily),
the backtest is **year-over-year**, not midseason. Corpus: player-seasons 2015-2024 with
FanGraphs rates + Statcast xStats + age, joined, min-PA/IP gated.

**Fixed-yardstick space (no historical `keeper_value` call).** The historical backtest does
NOT call `keeper_value` -- the repo has neither archived ZiPS out-years nor a per-season
league scale for past years. Instead each estimator's forward line is scored two ways:
(a) **per-stat rate MAE** vs realized year-Y+1 rates (no scale needed); and (b) a single
**forward SGP** computed with the CURRENT league `ScaleInputs` used as a constant ruler
across all seasons -- comparing estimators on one fixed yardstick, not reconstructing each
year's true keeper value. The 2026 report's `keeper_value` call is a separate path.

**The prior for the skill-adjusted and surface estimators.** For backtest years the
`projection_rate` prior is a **reconstructed Marcel-style projection**: a weighted blend of
the player's prior-N-year actual rates (recent-weighted), regressed toward the league mean
by sample, with a simple age adjustment. This same prior defines the candidate population
(year-Y surface deviating from prior beyond the threshold). Archived-ZiPS years (2022-2024)
additionally get the **pure-ZiPS** baseline; coverage is stated explicitly in the output --
no silent cap.

Estimators compared against realized year-Y+1 actuals:

- **surface-believed** (year-Y surface rates at face value),
- **skill-adjusted** (this design's adjusted line),
- **pure-ZiPS** (2022-2024 only).

Headline metric (decision-relevant): **rank correlation (Spearman) of each estimator's
forward SGP with realized year-Y+1 SGP** (same fixed ruler), on the candidate population.
To keep this a pure rate-quality test with no playing-time confound, **PT is held constant
across all three estimators AND the realized target** -- every line (predicted and actual)
is built at the same held PA/IP (the player's year-Y actual), so only the rates differ. SGP
aggregation then weights those rates by roto importance (what plain rate MAE cannot).
Does ranking by skill-adjusted value order next year's keepers better than ranking by
surface or by ZiPS? Secondary: per-stat rate MAE, and label discrimination (lift of the
sustainability score in predicting how much of a surface gain is retained). (Projecting PT
forward -- rewarding players likely to keep playing -- is a deferred backtest refinement.)

**Acceptance criterion (the go/no-go rule).** Automation stays deferred UNLESS the
skill-adjusted estimator's headline rank correlation exceeds BOTH baselines on the
**held-out** seasons (fit `w` on 2015-2022, report on 2023-2024) by a margin beyond sampling
noise -- concretely, the skill-adjusted vs surface improvement's bootstrap confidence
interval excludes zero. An in-sample-only or within-noise edge is a "not good enough"
verdict, and the report stays a manual resource. Overfitting is guarded by this held-out
split; min-PA/IP gate and candidate-deviation threshold are named tunable constants with
starting defaults (following the `DEFAULT_MIN_AB` pattern).

## Edge cases / failure modes

- **Small samples:** gate on min PA/IP; below the gate, reliability is low so `w` shrinks
  hard toward the projection -- never emit a confident line off a tiny sample. Mark
  low-confidence in the report.
- **Missing xStats** (insufficient batted balls; thinner pitcher expected-stats coverage):
  fall back to FanGraphs rates only, flag reduced confidence, do not drop the player.
- **Not park/league adjusted:** Statcast xStats are not park-adjusted; wOBA carries league
  context. Stated as a known limitation in the report; park factors deferred. Do not
  over-claim precision.
- **Fetch flakiness / rate limits:** cache-first; fail loud; never clobber good cached data
  with an empty or errored pull.
- **Identity-join misses:** emit and count unmatched players; never silently drop. A
  deliberate name-collision fixture guards the join.
- **Numeric-default trap:** follow the repo rule -- never `x or default` for numeric fields
  (`0` is falsy). Use explicit `is None` checks in rate math and sort keys.
- **ASCII-only** in all source, logs, and report renderers (Windows cp1252 stdout); reconfigure
  stdout to utf-8 only if a data-derived name forces it.

## Testing expectations

- **Classifier (pure):** fixtures for (a) barrel-backed HR breakout -> high `w`, `real`;
  (b) BABIP mirage -> low `w`, `lucky mirage`; (c) real decline; (d) small-sample
  low-confidence. Assert the ordering property: at equal surface delta, the skill-backed
  player's adjusted value exceeds the luck-backed player's.
- **Identity join:** fixture with a deliberate name collision -> correct MLBAM match and
  correct unmatched reporting.
- **Data layer:** mock `pybaseball`; assert a cache hit skips the network and a failed
  fetch does not overwrite an existing good cache. No network in tests.
- **Backtest:** deterministic smoke test on a tiny synthetic 2-year corpus; assert the
  discrimination metric computes and all three baselines are wired.
- Tests mirror `tests/` structure, run under pytest, ASCII-only, no network.

End-of-effort verification per CLAUDE.md: `pytest`, `ruff check .`, `ruff format --check .`,
`vulture`, and `mypy` for any file listed under `[tool.mypy].files`.

## Phasing (four increments, each independently verifiable)

1. **Data layer + identity** -- source/cache season xStats+rates+age, mapping table,
   coverage/unmatched report. Verify: cached CSVs + a coverage summary.
2. **Classifier** (pure) -- decomposition, `w`-mapping, adjusted line, label + reason.
   Verify: unit tests including the ordering property.
3. **Report** -- current-season ranked resource (skill-adjusted value, surface-believed
   value, delta, label, reason). Verify: report runs on 2026 data and ranks sensibly.
4. **Backtest** -- 2015-2024 year-over-year; adjusted vs surface vs pure-ZiPS; tune `w`;
   held-out reporting. Verify: results CSV + a summary; this is the go/no-go gate for any
   later automation.

## Known limitations

- The backtest is year-over-year, not midseason; it validates "does the adjusted line
  predict next year better," which is the decision-relevant question for keepers, but it
  cannot directly validate a midseason call.
- The pure-ZiPS baseline is limited to 2022-2024 (archived-ZiPS years); the surface baseline
  spans the full corpus.
- The historical backtest feeds the skill-adjusted estimator a **reconstructed Marcel prior**
  in place of the real preseason projection it uses in production; the surface-believed
  baseline needs no prior. So a rough reconstructed prior can only handicap the skill-adjusted
  estimator, making the comparison **conservative** -- clearing the gate under a rougher prior
  is stronger evidence, while failing it is ambiguous between method and prior. The 2022-2024
  pure-ZiPS baseline (a real projection) is the cleaner read on those years.
- Metrics are not park-adjusted in v1.
