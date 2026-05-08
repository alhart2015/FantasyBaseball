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
