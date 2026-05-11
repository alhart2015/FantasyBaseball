# Hot/Cold Streak Analysis — Design Spec

## Problem

Roster decisions in a Yahoo H2H roto league are made weekly: who starts, who sits, who to drop, who to pick up. The current toolkit (lineup optimizer, waiver scorer, trade evaluator) drives all of these off rest-of-season projections. Projections are the right answer in expectation, but they don't answer a different question fantasy managers ask constantly:

> *Player X has been on fire / ice cold for the last week. Should I expect that to continue?*

The intuition is that *some* short-term performance variation is real (a swing fix, a velocity bump, a new role) and some is luck (BABIP regression, sequencing, schedule). Acting on streaks indiscriminately is bad — chasing hot players overpays for noise. Ignoring streaks entirely is also bad — fantasy systems that are 100% projection-based miss real signal.

This project aims to:

1. Empirically define what "hot" and "cold" mean over short windows (3-14 days), with universal thresholds calibrated from history rather than imposed by gut feel.
2. Measure how often hot/cold streaks actually persist into the following week — establishing whether there's any predictive signal at all.
3. Identify peripheral metrics (BABIP, K%, BB%, exit velocity, barrel rate, xwOBA) that distinguish "real" streaks from luck-driven streaks.
4. Eventually fold the resulting predictions into roster decisions — first as a Sunday research report, later (if signal is real) as a dashboard panel and lineup-optimizer adjustment.

This is a long-running, research-first effort. Methodology may evolve. The Progress Log at the bottom of this document tracks what's been done and what's been learned.

## Scope: hitters only for v1

Pitchers are deferred. Hitter game cadence is uniform (4-7 games/week, 3-5 PA/game) and a 7-day window is a clean unit of analysis. Starting pitchers get 1-2 starts/week and need a per-start window — different baselines, different methodology. Relief pitchers are even noisier (small per-outing samples, role-driven save volatility). We will revisit pitcher methodology after the hitter pipeline is proven and shows signal.

## Decisions

| Decision | Choice | Rationale |
|----------|--------|-----------|
| Streak unit | Player-level composite + per-category labels | Composite drives roster decisions; per-category exposes which stat is hot/cold |
| Baseline | Empirical absolute thresholds from historical distributions | Projection-agnostic — 3 HR in a week is hot regardless of who was projected to hit them |
| Threshold method | 90th / 10th percentile within (category × window × playing-time bucket) | Lets data set the bar; stratification controls for opportunity |
| Playing-time strata | low (5-9 PA) / mid (10-19 PA) / high (20+ PA); skip <5 PA | Otherwise platoon hitters skew thresholds |
| Windows studied | 3-day, 7-day, 14-day | Calibrate which is most predictive; don't pre-commit |
| Categories tracked | HR, R, RBI, SB, AVG | The five hitter roto categories; nothing else relevant for decisions |
| Composite score | (# hot categories) − (# cold categories), range −5 to +5 | Simple, interpretable, no coefficients to tune |
| Predictive features | Counting + rate + Statcast peripherals (BABIP, K%, BB%, ISO, GB/FB, EV, barrel%, xwOBA) | Statcast gives the strongest "is this real?" signal |
| Historical scope | 2023, 2024, 2025 (3 seasons) | 2026 reserved as out-of-sample. 5 seasons would be more data but Statcast pulls scale linearly |
| Player qualification | ≥150 PA in the season (player-season-level filter) | Once a player qualifies, all their windows enter the threshold calibration, stratified by PT bucket — including their low-PA weeks. Filters out cup-of-coffee callups entirely without trimming legitimate platoon weeks |
| Storage | DuckDB at `data/streaks/streaks.duckdb`, gitignored | Analytical workload (window funcs, percentiles, joins on millions of rows); pandas/parquet integration; isolated from production cache |
| Statcast client | `pybaseball` against Baseball Savant | The standard; new dependency but well-maintained |
| Repo location | New `src/fantasy_baseball/streaks/` package + `scripts/streaks/` | Discoverable but isolated from production pipeline |
| Production coupling | None in v1 | `streaks/` does not import from `web/`, `lineup/`, or `data/redis_store.py` and vice versa |
| v1 deliverable | Jupyter notebooks + a CLI Sunday report | Lowest engineering bar for evolving research questions |
| Production integration | Deferred to v2/v3 (dashboard panel, optimizer adjustment) | Only worth building if v1 establishes real signal |

## Architecture

```
src/fantasy_baseball/streaks/
├── __init__.py
├── data/
│   ├── fetch_history.py    # MLB Stats API + pybaseball Statcast pulls
│   ├── schema.py           # DuckDB DDL
│   └── load.py             # idempotent inserts
├── windows.py              # rolling-window aggregation (per player, per date, per N days)
├── thresholds.py           # empirical percentile calibration
├── labels.py               # apply thresholds to compute hot/cold flags + composite
├── analysis/
│   ├── continuation.py     # base-rate calculations
│   └── predictors.py       # logistic regression / gbm fits
└── reports/
    └── weekly.py           # Sunday CLI report generator

scripts/streaks/
├── fetch_history.py        # entry point: fetch one season
├── compute_labels.py       # entry point: rebuild thresholds + label all windows
├── run_analysis.py         # entry point: regenerate analysis tables
└── weekly_report.py        # entry point: print/write current-week report

notebooks/streaks/          # gitignored
├── 01_distributions.ipynb
├── 02_continuation.ipynb
└── 03_predictors.ipynb

data/streaks/               # gitignored
└── streaks.duckdb
```

**Hard isolation from the production stack.** Nothing in `streaks/` imports from `web/refresh_pipeline.py`, `data/redis_store.py`, or `lineup/`. Conversely, the existing pipeline does not import from `streaks/`. We will revisit this boundary if and when v2 (dashboard panel) lands.

## Data Layer

### Sources

- **Per-game game logs**: MLB Stats API via the existing `analysis/game_logs.py` helpers. The current code fetches one season; we extend it to accept arbitrary seasons. Cost: one HTTP call per (player, season). Slow but free.
- **Per-PA Statcast**: `pybaseball.statcast()` against Baseball Savant. Returns one row per pitch, which we filter to terminal-PA events and aggregate. Cost: ~1 GB total for 3 seasons of qualifying hitters; pulled in chunks by date range.
- **Peripherals (BABIP, ISO, K%, BB%, GB/FB)**: computed from the same game logs and per-PA Statcast events. No separate FanGraphs scrape.

### DuckDB schema

```sql
CREATE TABLE hitter_games (
    player_id INTEGER,           -- mlbam_id
    name VARCHAR,
    team VARCHAR,
    season INTEGER,
    date DATE,
    pa INTEGER, ab INTEGER, h INTEGER, hr INTEGER,
    r INTEGER, rbi INTEGER, sb INTEGER, bb INTEGER, k INTEGER,
    PRIMARY KEY (player_id, date)
);

CREATE TABLE hitter_statcast_pa (
    player_id INTEGER,
    date DATE,
    pa_index INTEGER,           -- 1, 2, 3... within game
    event VARCHAR,              -- single, double, hr, strikeout, walk, ...
    launch_speed DOUBLE,        -- exit velocity
    launch_angle DOUBLE,
    estimated_woba_using_speedangle DOUBLE,  -- xwOBA
    barrel BOOLEAN,
    PRIMARY KEY (player_id, date, pa_index)
);

CREATE TABLE hitter_windows (
    player_id INTEGER,
    window_end DATE,
    window_days INTEGER,        -- 3, 7, or 14
    pa INTEGER, hr INTEGER, r INTEGER, rbi INTEGER, sb INTEGER,
    avg DOUBLE, babip DOUBLE, k_pct DOUBLE, bb_pct DOUBLE, iso DOUBLE,
    ev_avg DOUBLE, barrel_pct DOUBLE, xwoba_avg DOUBLE,
    pt_bucket VARCHAR,          -- 'low' | 'mid' | 'high'
    PRIMARY KEY (player_id, window_end, window_days)
);

CREATE TABLE thresholds (
    season_set VARCHAR,         -- e.g. '2023-2025' for the calibration set
    category VARCHAR,           -- 'hr' | 'r' | 'rbi' | 'sb' | 'avg'
    window_days INTEGER,
    pt_bucket VARCHAR,
    p10 DOUBLE,
    p90 DOUBLE,
    PRIMARY KEY (season_set, category, window_days, pt_bucket)
);

CREATE TABLE hitter_streak_labels (
    player_id INTEGER,
    window_end DATE,
    window_days INTEGER,
    category VARCHAR,
    label VARCHAR,              -- 'hot' | 'cold' | 'neutral'
    PRIMARY KEY (player_id, window_end, window_days, category)
);
```

### Idempotency

`fetch_history.py` skips player-seasons already present in `hitter_games`. Statcast pulls are by date range and skip dates already present in `hitter_statcast_pa`. Window aggregation rebuilds rows for any (player, date) that changed since last run. Threshold calibration always rebuilds (fast). Labels rebuild when thresholds or windows change.

## Streak Definition

For a hitter on date D, with a window of N days (3, 7, or 14):

1. Aggregate games in [D-N+1, D] → one row in `hitter_windows`.
2. Determine playing-time bucket from the window's PA count.
3. For each of the 5 categories, look up the threshold for (category, window, PT bucket).
4. Label = "hot" if the count (or AVG) exceeds the 90th-percentile threshold; "cold" if below the 10th; "neutral" otherwise.
5. Composite score = (# hot categories) − (# cold categories). Neutral categories contribute 0. Range −5 (cold across the board) to +5 (hot across the board).

Window-end-anchored definition (any 7-day window ending on D) — *not* run-based ("hit safely in 5 straight games"). Run-based streaks are noisier and the literature is sparse.

## Continuation Analysis (the research)

For each labeled (player, window_end, category) row in `hitter_streak_labels` from 2023-2024 (held-out test year: 2025):

- Compute the *next* window's outcome (the same player's category result over [window_end+1, window_end+N]).
- Tabulate:
  - P(next window above bucket median | current label = hot)
  - P(next window below bucket median | current label = cold)
  - Compare to unconditional rates. If hot → 50% and base rate is 50%, no signal.
- Stratify by streak strength (90th vs 95th vs 99th percentile crossings), by PT bucket, by category, by player's full-season skill level (e.g. quartile of season HR rate).

**Phase 3 go/no-go gate**: if hot/cold labels show no meaningful lift over base rates after stratification, stop and reframe before building the predictive model.

If signal exists, fit a logistic model — one model *per category* (HR, R, RBI, SB, AVG):

```
P(category continues hot/cold) ~ streak_strength + babip + k_pct + bb_pct + iso
                               + ev_avg + barrel_pct + xwoba_avg
                               + season_rate_in_category + pt_bucket
```

`season_rate_in_category` is the player's full-season rate for the predicted category (e.g. season HR/PA when modeling HR continuation) — proxies for "is this player capable of this kind of streak?" GB/FB is derived from `launch_angle` rather than stored separately.

Train on 2023-2024, validate on 2025. Report: coefficient signs, p-values, ROC-AUC, calibration plot, top features by importance.

## Outputs

### v1 (research artifacts)

- `notebooks/streaks/01_distributions.ipynb` — distribution plots, threshold calibration outputs, sanity checks (do thresholds make intuitive sense?).
- `notebooks/streaks/02_continuation.ipynb` — base-rate tables, plots of streak persistence by category and PT bucket. Lift over base rate per (category × window × strength) cell.
- `notebooks/streaks/03_predictors.ipynb` — logistic model fits, coefficient interpretation, calibration plots, held-out 2025 accuracy.
- `scripts/streaks/weekly_report.py` — CLI that pulls current-season game logs, applies the calibrated thresholds, runs the model, and emits a Sunday report:
  - Hot rostered hitters (ranked by composite score, then strength)
  - Cold rostered hitters (same)
  - Per-category labels for each
  - Continuation confidence score from the model
  - Top 1-2 peripheral drivers per player ("Soto: hot streak supported by 95th-percentile barrel rate and stable BABIP — likely real")
  - Suggested action lines (start/sit/stash) — heuristic on top of model score, no Yahoo integration

### v2 / v3 (deferred)

- Dashboard panel on the season dashboard surfacing the same data.
- Lineup-optimizer adjustment — multiplicative tweak to ROS projections for the next week based on continuation probability.

## Phasing

| Phase | Scope | Gate |
|-------|-------|------|
| 1 | Data layer: fetch + DuckDB schema + load 3 seasons of game logs and Statcast | Data quality checks pass; row counts plausible |
| 2 | Window aggregation + threshold calibration + labeling | Notebook 01 produces threshold tables that pass eyeball test |
| 3 | Continuation base rates | **Go/no-go**: do hot/cold labels show meaningful lift over base rates? |
| 4 | Predictive model | Held-out 2025 ROC-AUC ≥ 0.55 (i.e. better than coin flip) |
| 5 | Weekly Sunday report | Manually validated against 2-3 weeks of 2026 data |
| B (deferred) | Dashboard panel | Methodology proven through v1 |
| C (deferred) | Optimizer integration | Dashboard panel running for ≥4 weeks without surprises |

## Assumptions and Open Questions

- **90/10 percentiles**, not 95/5. More sample, less extreme labels. Will revisit if the analysis shows the bar is too low.
- **PT buckets at 5/10/20 PA.** Bucket boundaries are eyeballed; could be revisited based on the playing-time distribution.
- **Statcast coverage starts 2015** but quality (especially xwOBA) improved over time. 2023-2025 is well within the high-quality era.
- **`pybaseball` reliability**: the package occasionally breaks when Baseball Savant changes its endpoints. Keep version pinned in dev deps.
- **`hitter_statcast_pa` size**: ~600 PA/qualified-hitter × ~250 hitters × 3 seasons ≈ 450K rows. Manageable in DuckDB.
- **Player ID alignment**: this project keys on MLB AM ID (integer), not the project's `name::player_type` convention. We only need name-based joining at the report-generation step (matching streak players to Yahoo roster). Use `data/db.py` to map mlbam_id ↔ name when building the weekly report.
- **Test coverage**: research code has lighter test bar than production. Unit tests for `windows.py`, `thresholds.py`, `labels.py` (the core math). Notebooks not tested.
- **Lint / mypy**: streaks modules added to `[tool.mypy].files` and conform to project lint rules from day one. Cheaper than retrofitting.

## Dependencies (new)

- `duckdb` (Python package)
- `pybaseball`
- `polars` *(optional — pandas works but polars+duckdb is faster for the analytical queries; defer until performance is a problem)*

To be added to `pyproject.toml` `[project.optional-dependencies].dev` (research tooling, not production-required) when Phase 1 lands.

## Progress Log

This section is appended to as work happens. Each milestone gets a dated entry.

### 2026-05-06 — design spec written

- Brainstormed scope: hitters only, projection-agnostic empirical thresholds, Statcast in scope, isolated local DuckDB.
- Decisions captured in the table above. Branch `analysis/hot-streaks` created off main.
- Next: wait for spec review, then move to writing the implementation plan for Phase 1 (data layer).

### 2026-05-06 — Phase 1 (data layer) implemented

- All 11 plan tasks executed via subagent-driven TDD: DuckDB schema, idempotent loaders, qualified-hitter fetch, per-season game log fetch, per-PA Statcast fetch via pybaseball, fetch_season orchestrator, CLI script.
- 33 unit tests covering schema, upserts, existence queries, parsing, chunking, orchestration, and per-player exception handling. Full project suite passes.
- Streaks package added to mypy strict overrides.
- New gitignored directory `data/streaks/` for the local DuckDB; `notebooks/streaks/` reserved for Phase 2-3 research notebooks.

#### Known issues / follow-ups for Phase 1 acceptance

1. **Statcast partial-load handling.** The orchestrator skips Statcast for a season if *any* dates from that season are already loaded (all-or-nothing). An interrupted prior run could leave the DuckDB with a few days of Statcast and silently skip the rest on the next run. Mitigation today: row-count sanity check at acceptance time (~150K-200K rows expected per season; obvious if missing). Recovery: `DELETE FROM hitter_statcast_pa WHERE date BETWEEN '<season>-03-15' AND '<season>-11-15'` and re-run. Phase 2 cleanup: convert to gap-filling (chunk-by-chunk skip of already-loaded dates).
2. **Plan test bug (Task 8).** The plan's `test_pitches_to_pa_rows_assigns_pa_index_per_player_per_date` had inconsistent sort/assertion logic; fixed during implementation by splitting into per-player sub-lists. Original plan file not updated; future re-runs of this plan should use the fixed test.

#### Inputs for Phase 2 design (flagged by end-of-phase review)

3. **Schema gap for BABIP and ISO.** `hitter_windows` declares `babip` and `iso` columns, but `hitter_games` only stores h, hr, ab, pa, bb, k — no doubles, triples, sac flies, or HBP. Two paths for Phase 2: (a) extend `hitter_games` with `b2`, `b3`, `sf`, `hbp` columns and recompute the loader column tuple; (b) derive these metrics from per-PA Statcast event counts in `hitter_statcast_pa` (the `event` column has values like `"double"`, `"triple"`, `"sac_fly"`, `"hit_by_pitch"`). Path (b) couples Phase 2 windowing to Statcast availability, which we already require — likely the cleaner choice. Decide before writing the Phase 2 plan.
4. **`pa_index` stability.** `statcast.py::pitches_to_pa_rows` assigns `pa_index` from `groupby([batter, game_date]).cumcount()+1` after sorting by `[batter, game_date]` — no within-game secondary sort. Re-fetching the same date range can shuffle `pa_index` for the same plate appearance. For Phase 1 (counting / aggregating PAs) this is harmless. For Phase 2 if streak labels need to walk events in chronological order, sort by `[batter, game_date, at_bat_number]` (pybaseball exposes that column) before computing `pa_index`.
5. **`summary["players_fetched"]` semantics.** Currently counts attempted players (including those whose game-log fetch raised). The variable name reads like "succeeded." Either rename to `players_attempted` or add a separate `players_succeeded` field for log-readability. Cosmetic — pin the meaning in Phase 2 cleanup.

### 2026-05-07 — Phase 1 acceptance (with fixes)

The first acceptance run on 2025 surfaced three bugs, all fixed:

1. **numpy scalar binding crash** (PR #57). `_na_to_none` returned non-NaN values unchanged, so numpy scalars (int64 / float64) reached DuckDB's `executemany` binder, which doesn't auto-coerce them. Fix: `_na_to_none` now calls `.item()` on any object that has it (numpy scalars do; native Python types don't), unboxing to native Python.
2. **`/stats/leaders` 100-cap** (PR #58). The MLB Stats API leaderboard endpoint silently caps at 100 results regardless of the `limit` parameter. The bottom of the cap sits ~560 PA, well above our 150 PA cutoff. Switched to `/stats?stats=season&playerPool=All`, which returns every player who took ≥1 PA (~750/year). Filter client-side to ≥min_pa as before.
3. **Doubleheader collisions** (this PR). The PK was `(player_id, date)`, so when a player played two games on the same date, the second game's stats overwrote the first via `INSERT OR REPLACE`. Pre-fix, ~440 game-log rows per season silently disappeared. Fix: PK is now `(player_id, game_pk)`, where `game_pk` is the MLB Stats API gamePk integer (unique per game).

**Final acceptance numbers** (after re-fetch on the post-doubleheader-fix schema):

| Season | Qualified hitters | Game logs | Statcast PAs |
|--------|------------------:|----------:|-------------:|
| 2023   | 404 | 44,707 | 202,177 |
| 2024   | 410 | 45,472 | 197,983 |
| 2025   | 393 | 44,262 | 198,203 |
| **Total** | **1,207 player-seasons** | **134,441 game logs** | **598,363 Statcast PAs** |

DB size: **40.5 MB** for the full 3-season corpus (DuckDB columnar compression — uncompressed would be ~450 MB). The doubleheader fix recovered 1,377 game-log rows that the original `(player_id, date)` PK had silently overwritten.

The original spec estimates were off: ~400 hitters/season (not 150-200), ~44K game logs/season (not 25-30K), Statcast rows accurate (~200K). DB size much smaller than expected.

#### Next milestone

- **Phase 2 planning.** Window aggregation (`hitter_windows` population) + empirical threshold calibration (`thresholds`) + streak labeling (`hitter_streak_labels`). Decide BABIP/ISO sourcing (extend `hitter_games` with 2B/3B/SF/HBP, or derive from per-PA Statcast events). Address the all-or-nothing Statcast skip and `pa_index` chronological stability if Phase 2 needs them.

### 2026-05-08 — Phase 2 schema migration + re-fetch

Migrated the local DuckDB to the Phase 2 schema (DROP+`init_schema`, then re-ran `scripts/streaks/fetch_history.py` for all three seasons). The expanded `hitter_games` schema captures the full PA decomposition (`b2/b3/sf/hbp/ibb/cs/gidp/sh/ci`) plus `is_home`; the expanded `hitter_statcast_pa` adds `at_bat_number/bb_type/estimated_ba_using_speedangle/hit_distance_sc`.

Row counts match Phase 1 acceptance exactly (134,441 game logs / 598,363 Statcast PAs / 1,207 player-seasons across 2023-2025).

PA-identity check (`pa == ab + bb + hbp + sf + sh + ci`): 0 violations in 2024 and 2023; 2 in 2025 (José Ramírez 2025-09-03 game_pk=776474, gap +1; Dominic Canzone 2025-09-18 game_pk=776272, gap +1). 2/134,441 = 0.0015%, well under the plan's ~50/season investigation threshold. Likely a single rare component the MLB Stats API exposes under a slightly different field name; not blocking.

Drift sums for the new columns (per ~400 qualified hitters/season) all in expected ranges:
- doubles 7-8K, triples 600-700, SF ~1.2K, HBP ~1.7-1.9K, IBB ~500
- CS ~800-900, GIDP ~3K, SH ~350-450, CI ~80-90, `is_home` 50.0-50.2%

Statcast new-column non-null share:
- `at_bat_number`: 100% (always present on terminal PAs)
- `bb_type`: 67-68% (null on K/BB/HBP, which together are ~32% of PAs — math checks)
- `estimated_ba_using_speedangle`: 62-63% (subset of batted balls with measurable EV/angle)
- `launch_speed`: 64-66%

#### Next milestone

- **Phase 2 implementation continuing.** Tasks 8-14 of the Phase 2 plan: `windows.py` (rolling sums + rate stats + Statcast peripherals + PT bucket), `thresholds.py` (DuckDB `percentile_cont` over qualified-hitter rows), `labels.py` (hot/cold/neutral application), CLIs, and the Phase-2 acceptance notebook (`01_distributions.ipynb`).

### 2026-05-08 — Phase 2 (windows, thresholds, labels) accepted

All 14 plan tasks landed. `compute_windows` + `compute_thresholds` +
`apply_labels` chained via `scripts/streaks/compute_labels.py`. Real-data
run on the 2023-2025 corpus produced:

| Stage | Rows | Notes |
|-------|-----:|-------|
| `hitter_windows` | 521,127 | one row per (player, calendar_date in [first_played, last_played], window_days ∈ {3, 7, 14}) where PA ≥ 5 |
| `thresholds` | 45 | 5 categories × 3 windows × 3 PT buckets |
| `hitter_streak_labels` | 2,605,635 | 521,127 × 5 categories |

Threshold eyeball checklist (notebook `01_distributions.ipynb`):

- HR / 7d / high p90 = 2 (plan guess ~3 — slightly lower in reality, but 2 HR in a single 20+ PA week is genuinely the 90th percentile)
- AVG / 14d / high p90 = .348 (plan guess .375-.420 — author was slightly optimistic)
- AVG / 14d / high p10 = .150 (plan guess .150-.190) ✓
- SB / 7d / high p90 = 2 (plan guess ~2) ✓
- p10 ≤ p90 holds across all 45 strata ✓
- Bucket monotonicity for counting cats: rough but plateau-prone because the discrete distribution clamps p10 to 0

**Methodology surprise to revisit in Phase 3:** for sparse counting categories (HR, SB), p10 collapses to 0 in every bucket because most weeks have zero events even for high-PA hitters. The "cold" label therefore covers any window with zero events — a much wider net than "below the 10th percentile of nonzero counts." Phase 3 may want a different rule for sparse counts (e.g., a per-category lower bound that requires PA ≥ N before "cold").

Also flagged but deferred: `_add_statcast_peripherals` per-window mask loop is O(windows × daily_rows) and dominated wall-time on the real corpus (~5 minutes of the ~7-minute pipeline run); cumulative-sum-on-dense-calendar refactor noted in the function docstring.

#### Next milestone

- **Phase 3 — continuation analysis (the go/no-go gate).** For each labeled (player, window_end, category) row in 2023-2024, compute the next-window outcome and tabulate persistence rates. Stratify by streak strength, PT bucket, and player-season skill quartile. Compare to base rates. Hold 2025 out as the test set; 2026 is out-of-sample for production inference. The Phase 2 methodology surprise (p10=0 for sparse counts) is a Phase 3 design input — decide whether to redefine "cold" for sparse cats before fitting continuation models.

### 2026-05-09 — Phase 3 (continuation analysis) accepted

All 13 plan tasks landed. New schema: `hitter_projection_rates` (PK
`player_id, season`), `continuation_rates` (PK `season_set, category,
window_days, pt_bucket, strength_bucket, direction, cold_method`), and an
extended `hitter_streak_labels` PK that includes `cold_method`. Sparse-cat
(HR, SB) cold labels migrated from empirical p10 (which collapsed to
"0 = cold" for ~80% of windows) to skill-relative Poisson lower-tail rules
anchored on preseason projection rates. Skill baseline: mean blend of all
available preseason projection systems per season (Steamer + ZiPS for
2023-2025 → 5,101 projection-rate rows total). Two parallel rules —
`cold_method='poisson_p10'` and `cold_method='poisson_p20'` — are labeled
in the same pass; Phase 4 reads from whichever shows better lift.

Pipeline runtime on the real 2023-2025 corpus, after the perf-todo work in
Tasks 6+9 was vectorized (commit `7b8bb98`):

- `compute_windows`: ~22 sec (down from ~21 min before vectorization)
- `compute_labels` end-to-end: ~85 sec
- `run_continuation`: ~10 sec
- Total: ~95 sec wall time

Real-data row counts:

| Stage | Rows | Notes |
|-------|-----:|-------|
| `hitter_projection_rates` | 5,101 | Steamer + ZiPS, 2023-2025 |
| `hitter_windows` | 521,127 | 3d=143,988, 7d=181,427, 14d=195,712 |
| `thresholds` | 45 | 5 cats × 3 windows × 3 buckets |
| `hitter_streak_labels` | 3,642,065 | dense (R/RBI/AVG empirical): 521,127 each. Sparse (HR/SB) × 2 methods: 519,671 rows each |
| `continuation_rates` | 409 | per-stratum lift outputs after dropping neutral rows |

Sparse cold-label rates (the methodology surprise we predicted: Poisson
p10 makes cold labels very rare for sparse cats):

- HR poisson_p10: 1,025 cold / 119,079 hot / 399,567 neutral (0.2% cold rate)
- HR poisson_p20: 7,105 cold (1.4% cold rate)
- SB poisson_p10: 573 cold (0.1% cold rate)
- SB poisson_p20: 2,621 cold (0.5% cold rate)

This confirms the analysis from the plan's Q4 brainstorm: at our parameter
ranges, only the elite 14d-window cases produce cold-HR/SB labels under
Poisson p10.

**Go/no-go gate result: PASS (by a wide margin).**

- **Test 1:** ≥1 cell with N≥1000 and lift≥5pp. **96 cells qualify.** ✓
- **Test 2:** ≥3 of 5 categories show ≥2pp 7d lift. **All 5 categories qualify.** ✓

Headline lifts (max lift per cat × window × cold_method, restricted to
N≥1000):

| Category | Window | Cold method | Max lift | Where |
|----------|-------:|-------------|---------:|-------|
| AVG | 14d | empirical | +8.00pp | cold_q3, high bucket, below |
| HR | 14d | poisson_p10 / p20 | +12.84pp | hot_+3.0σ, high bucket, above |
| R | 14d | empirical | +24.22pp | hot_q5, high bucket, above |
| RBI | 14d | empirical | +19.92pp | hot_q5, high bucket, above |
| SB | 7d | poisson_p10 / p20 | +26.60pp | hot_+1.5σ, high bucket, above |

Hot persistence is the dominant signal across all categories. The 14d
window has the strongest signal for most cats (longer windows = less
noise). For SB the 7d window is comparable to 14d, suggesting SB hot
streaks are more transient.

Methodology surprises / notes for Phase 4:

- Sparse cold cells (HR/SB cold under Poisson) are too small (under 1k
  rows in most strata) to evaluate reliably — exactly as predicted.
  Phase 4 modeling should focus on hot persistence for sparse cats.
- Empirical p10 vs p20 produced essentially identical lift numbers in the
  cells where both fired — the difference is in row count, not in signal
  strength. The acceptance notebook's cell-size table shows p20 has more
  strata at higher N; Phase 4 may prefer p20 for sample-size reasons.
- 14d windows dominate the lift charts. Phase 4 may want to focus modeling
  on the 14d window first.
- Poisson calibration cell in the notebook overlays empirical PMF on
  Poisson(λ=2.0) for the `expected_HR ∈ [1.5, 2.5]` bin — log the result
  so Phase 4 knows whether to trust the rule unmodified or add an
  overdispersion correction.

Perf-todo work (commit `7b8bb98`). Three independent CPU-bound bottlenecks
were fixed before the acceptance run could finish:

- `compute_windows` final write: replaced `executemany` of 521K × 17-col
  rows with `INSERT ... FROM df` (DuckDB's pandas-scan path).
- `_apply_sparse_labels`: same fix, 3.64M label rows.
- `_dense_strength_bucket` / `_sparse_strength_bucket`: replaced per-row
  `df.apply` lambdas with vectorized `np.searchsorted` / numpy z-score
  quantization.

Pre-existing issues called out (not from Phase 3):

- `scripts/debug_eroto_avg.py` has 6 RUF100 noqa-directive lints +
  format drift (unrelated to streaks).
- `tests/test_data/test_mlb_schedule.py` has format drift (unrelated to
  streaks).
- `[tool.mypy].files` references a non-existent
  `src/fantasy_baseball/lineup/weighted_sgp.py` (config drift, unrelated
  to streaks).
- These are flagged for a separate cleanup pass; they don't gate Phase 3
  acceptance.

#### Next milestone

- **Phase 4 — predictive model.** Fit per-category logistic regressions
  with continuation as the target. Train on 2023-2024, validate on 2025.
  Calibration plot, ROC-AUC, top features by |coef|. Gate: held-out
  ROC-AUC ≥ 0.55 in at least 3 of 5 categories.

### 2026-05-10 — Phase 4 (predictive model) accepted

All 16 plan tasks landed (Task 15 perf-todo did **not** fire — real-data
wall time came in under the 120s budget). New schema: `model_fits` audit
table plus three new nullable rate columns on `hitter_projection_rates`
(`r_per_pa`, `rbi_per_pa`, `avg`). New library module
`streaks/analysis/predictors.py` carries the full pipeline:
`build_training_frame`, `fit_one_model` (player-grouped 5-fold CV over
`C ∈ {0.01, 0.1, 1, 10}`), `bootstrap_coef_ci` (200 player-grouped
resamples), `evaluate_model` (AUC + reliability), and
`permutation_feature_importance`. Orchestrated by `fit_all_models` and
CLI-exposed at `scripts/streaks/fit_models.py`.

**Methodology note (replaces spec's "p-values" language).** The spec's
original Phase 4 sketch mentioned "p-values" for coefficient inference.
Under L2 regularization, asymptotic p-values are not well-defined.
Phase 4 reports **200-resample player-grouped bootstrap CIs** instead.
This is the correct uncertainty quantification under regularization and
matches the design decision locked in the Phase 4 brainstorm.

Eight models fit on the 2023-2024 train / 2025 val corpus:

| Category | Direction | Chosen C | CV AUC | Val AUC | n_train | n_val |
|---|---|---:|---:|---:|---:|---:|
| R | above (hot) | 0.01 | 0.657 | 0.648 | 15,320 | 7,147 |
| R | below (cold) | 1.00 | 0.753 | 0.755 | 23,846 | 11,100 |
| RBI | above (hot) | 0.10 | 0.644 | 0.595 | 15,073 | 7,074 |
| RBI | below (cold) | 0.01 | 0.758 | 0.740 | 19,684 | 9,281 |
| AVG | above (hot) | 0.01 | 0.603 | 0.597 | 11,907 | 5,582 |
| AVG | below (cold) | 0.01 | 0.612 | 0.617 | 12,520 | 5,446 |
| HR | above (hot only) | 0.01 | 0.536 | 0.536 | 21,986 | 10,634 |
| SB | above (hot only) | 0.01 | 0.562 | 0.523 | 22,744 | 10,251 |

Per-category max(hot_auc, cold_auc):

| Category | Max AUC | Passes 0.55? |
|---|---:|---|
| R   | 0.755 | YES |
| RBI | 0.740 | YES |
| AVG | 0.617 | YES |
| HR  | 0.536 | NO |
| SB  | 0.523 | NO |

**Gate result: PASS** (3 of 5 categories at AUC ≥ 0.55).

Total pipeline runtime on the real 2023-2025 corpus: **~57s wall time**
(8 models × {player-grouped 5-fold CV over 4-value C grid + 200-resample
bootstrap + permutation importance}). Task 15's joblib parallelization
did not fire — budget was 120s; observed was under half of that.

#### Methodology surprises / Phase 5 inputs

- **Cold-direction models dominate hot for dense cats.** R/RBI/AVG cold
  AUCs (0.755 / 0.740 / 0.617) are materially higher than the
  corresponding hot AUCs (0.648 / 0.595 / 0.597). Cold persistence is
  more predictable from peripherals than hot persistence — likely
  because cold streaks have stronger luck-vs-skill signal in BABIP /
  K%, whereas hot streaks regress more uniformly. Phase 5's Sunday
  report should weight cold predictions more heavily than hot when
  composing the start/sit recommendation.
- **Sparse-cat (HR/SB) hot models barely beat coin flip** (0.536 /
  0.523). Phase 3's headline lift for these cats came almost entirely
  from `streak_strength` (which is already a function of HR/SB count);
  the additional peripherals add little signal at the AUC level. Phase 5
  should consume HR/SB hot probabilities conservatively — present them
  as a streak strength indicator, not a predictive lift driver.
- **Regularization preference is strong.** 7 of 8 models picked
  `C=0.01` (the strongest L2 in the grid); only RBI cold preferred
  `C=1.0`. Signal-to-noise is high relative to feature count; future
  iterations should consider tighter grid resolution at the low end.
- **Calibration:** reliability diagrams (notebook §3) — diagnostic-only.
  No isotonic correction applied in Phase 4. Phase 5 should re-examine
  per-model calibration after a season of Sunday reports and apply
  isotonic if any model is >5pp off at any bin center.
- **`barrel_pct` dropped from the feature set.** Task 14 acceptance
  found that `hitter_statcast_pa.barrel` is NULL across all 598K rows in
  the local Statcast corpus (Phase 1 fetch issue surfaced because
  Phase 3 didn't use barrel as a load-bearing feature). Phase 4 ships
  with 11 features instead of the spec's 12; `xwoba_avg` is in the set
  and captures most of barrel's predictive value anyway. See
  `feature_cols_with_nulls` in `predictors.build_training_frame` and the
  comment on `EXPECTED_FEATURE_COLUMNS`. **Phase 5 follow-up:**
  investigate why `pybaseball.statcast()` returns NULL barrel for our
  query shape; consider re-fetching when fixed.

#### Pre-existing issues unrelated to Phase 4

- **sklearn 1.8 `penalty='l2'` deprecation.** Every fit emits a
  `FutureWarning` ("'penalty' was deprecated in version 1.8 and will be
  removed in 1.10. Use `l1_ratio=0` instead"). The Phase 4 test run
  collects 358 of these warnings (one per fold/bootstrap fit). They are
  cosmetic now but **will break `_build_pipeline` on sklearn 1.10**.
  Migration: swap `penalty="l2"` for `l1_ratio=0` on `LogisticRegression`
  (the lbfgs solver semantics are equivalent under elastic-net with
  l1_ratio=0). One-line fix; defer until sklearn 1.10 release.
- **Vulture pre-existing finding:** `tests/test_streaks/test_fetch_history.py:272`
  unused variable `min_pa` (commit `be470a6`). Trivial cleanup, deferred.

#### Process notes from the Phase 4 execution

- **Task 1 required a follow-up commit** (`5651ff3`) to update Phase 3
  call sites + the `hitter_projection_rates` DDL when the dataclass
  shape changed. The plan's Task 1 file list was too narrow; future
  phases that change dataclass shapes should anticipate the cascade.
- **Task 9 dropped `pipeline` from `bootstrap_coef_ci`'s signature**
  (`9640cf2`). The function refits internally from `chosen_C` and
  never used the passed pipeline. Task 11 was dispatched with the
  updated signature in mind.
- **Task 11 added a disjoint-seasons guard** (`91f0556`) — current
  defaults (2023-2024 train, 2025 val) don't overlap, but the explicit
  ValueError prevents a silent leakage bug if a future caller passes
  overlapping ranges.
- **Task 14 surfaced the `barrel_pct=NULL` bug** during real-data
  acceptance. Fixup at `fe6276c`.

#### Next milestone

- **Phase 5 — weekly Sunday report.** Pull current-season game logs,
  apply calibrated thresholds, run the 8 models, emit a CLI report:
  hot rostered hitters (ranked by composite score then strength), cold
  rostered hitters, per-category labels, continuation probability from
  the model, top 1-2 peripheral drivers per player, suggested
  start/sit/stash lines. Refit models in-process each Sunday from the
  `2023-2024 + 2024-2025` training set (sliding window) — Phase 4's
  refit-on-demand decision means no joblib artifacts to chase.
- Phase 4 future work (deferred unless Phase 5 surfaces a need):
  per-category curated feature subsets (Phase 4 used a uniform set
  across all 8 models), the 7d-SB-hot model (Phase 4 used 14d only),
  isotonic calibration correction if Phase 5 probabilities consumed
  downstream need it, `barrel_pct` investigation + re-fetch.
