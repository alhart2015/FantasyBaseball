# Hot Streaks Phase 5 — Sunday Report — Design Spec

**Date:** 2026-05-11

## Problem

Phase 4 (`docs/superpowers/specs/2026-05-06-hot-streaks-design.md`) shipped a
working 14d streak-continuation model: 8 fitted models in `model_fits`,
gate passed at 3-of-5 categories with held-out 2025 val AUC ≥ 0.55. The
spec deferred production wiring to a Phase 5 deliverable — a weekly
"Sunday report" CLI that surfaces current hot/cold streaks for Hart's
roster and waiver candidates, with continuation probabilities, so he can
factor that signal into start/sit and waiver decisions during the live
2026 season.

This spec covers Phase 5 only. Production integration (dashboard panel,
lineup-optimizer adjustment) remains deferred to v2/v3.

## Goal

One command — `python scripts/streaks/run_sunday_report.py` — produces:

1. A dated markdown report at `data/streaks/reports/YYYY-MM-DD.md`
2. A pretty-printed terminal version of the same content

The report covers Hart's Yahoo roster + the top 10 free-agent hitters
in league 5652 (id from `config/league.yaml`), labels each player's
current 14-day window per category (HR/R/RBI/SB/AVG) using the
historical 2023-2025 thresholds, scores continuation probability via
the Phase 4 models, and attributes each prediction to its top 1-2
peripheral drivers.

## Decisions

| Decision | Choice | Rationale |
|----------|--------|-----------|
| Window | 14d only | Empirically cleaner signal than 7d on the local data (cold-direction lift +6.8pp at 14d vs +1.9pp at 7d, averaged across cats; 14d wins or ties on every dense-cat hot/cold split). Matches what the Phase 4 model was trained on — no time-window mismatch. |
| Output format | Markdown file + terminal pretty-print | Markdown enables retrospectives ("did the May 11 call on Player X pan out?"); terminal is the in-the-moment view. Same content rendered twice. |
| Scope | Roster + top 10 FAs in league 5652 | Roster covers start/sit; FAs cover pickup. League-wide trade view deferred. |
| Sort order | Composite score desc, tiebreaker max continuation probability | Composite = (# hot cats) − (# cold cats). Strongest sustained-hot players at top, sustained-cold at bottom. |
| Roster row format | All 5 categories shown, even when neutral | Uniform grid is scannable; lets Hart see at a glance which players are quiet vs active. |
| FA row format | Top 10 by `|composite|`, non-neutral cats concatenated | Mixed hot + cold pool — strongest absolute signals win. Compact since FAs are research, not roster. |
| Recommendations | None — streak data only | Start/sit depends on matchup, opponent SP, position scarcity, league scoring — knowledge that lives in `lineup/`, not `streaks/`. Cleanest separation; Hart (or the lineup optimizer later) consumes streak data for decisions. |
| Refit cadence | Each report run, on `season_set_train=2023-2025` | ~60s; the spec mandates "refit in-process each Sunday." All historical seasons in training since 2025 no longer needs to stay held out (we're scoring 2026 now). |
| Scoring qualifier | Drop ≥150 PA filter when scoring 2026 players | The 150 PA filter was for clean historical threshold calibration. For scoring, anyone with a valid 14d window (PA ≥ 5) gets predicted. |
| Yahoo integration | Reuse `lineup/yahoo_roster.py` + `lineup/waivers.py` | `fetch_roster` and `fetch_and_match_free_agents` already exist and are tested. Name → mlbam_id mapping reuses existing `models/player.py` infrastructure. |

## Pipeline orchestration

`run_sunday_report.py` runs end-to-end:

1. **Fetch 2026 game logs + Statcast** for any dates not yet in the DB
   (reuses `scripts/streaks/fetch_history.py`'s incremental skip logic;
   no `--force-statcast` needed in routine weekly use).
2. **Load 2026 projection rates** into `hitter_projection_rates`
   (reuses `scripts/streaks/load_projections.py --seasons 2026`).
   Source data already present at `data/projections/2026/`.
3. **Rebuild `hitter_windows`** from current `hitter_games` +
   `hitter_statcast_pa` data spanning 2023-2026 (one call to
   `compute_windows`).
4. **Re-compute `thresholds` + re-apply `hitter_streak_labels`** using
   historical calibration on `season_set=2023-2025`. The 2026 windows
   get labeled against the historical cutoffs — that's the design
   intent (projection-agnostic empirical thresholds).
5. **Refit the 8 Phase 4 models** on `season_set_train=2023-2025`
   (no held-out val set — 2026 is the live-scoring out-of-sample).
   Writes new `model_fits` rows.
6. **Pull Yahoo roster + FAs** for the league configured in
   `config/league.yaml`. Map every player to `mlbam_id` via existing
   roster-resolution helpers.
7. **Score current 14d windows** for every roster + FA hitter. For
   each (player, category) with a non-neutral label and a trained
   model, run `model.predict_proba`. Compute driver attributions.
8. **Format and emit** the markdown report + terminal pretty-print.

Total wall time estimate: ~6-10 min (dominated by the Statcast
incremental pull; the rest is seconds). On a Sunday with no new data
since the last run, ~60s (just the refit).

## New modules

```
src/fantasy_baseball/streaks/
├── inference.py          # NEW
└── reports/
    ├── __init__.py       # NEW
    └── sunday.py         # NEW

scripts/streaks/
└── run_sunday_report.py  # NEW
```

### `streaks/inference.py`

Pure inference layer — no Yahoo, no I/O beyond DuckDB.

- `score_player_windows(conn, models, player_ids, window_end_date) -> dict`:
  for each `(player_id, category)` in `player_ids × {hr,r,rbi,sb,avg}`,
  return label, continuation probability, and top-2 driver list.
- `top_drivers(model, feature_vector, feature_names, k=2) -> list`:
  rank features by `|coef × scaled_feature_value|`; return list of
  `(feature_name, z_score, sign)` tuples.
- `refit_models_for_report(conn, season_set_train='2023-2025') ->
  dict[(cat,direction), PerModelResult]`: thin wrapper over
  `fit_all_models` that writes to a new `model_fits` row but also
  returns the in-memory fitted pipelines (avoid re-loading from DB).

### `streaks/reports/sunday.py`

Orchestrator + formatter — knows about Yahoo and report layout.

- `build_report(conn, league_config, refit_models=True) -> Report`:
  drives steps 1-7 of the pipeline above, returns a structured
  `Report` dataclass containing roster rows + FA rows + driver detail
  lines.
- `render_markdown(report) -> str`
- `render_terminal(report) -> None` (uses `rich` or `tabulate` — pick
  whichever is already in use elsewhere in `web/` or `lineup/`).

### `scripts/streaks/run_sunday_report.py`

Thin CLI: flags for `--skip-fetch` (use cached data), `--skip-refit`
(use stored `model_fits`), `--output-dir` (default
`data/streaks/reports/`), `--league` override.

## Report layout

### Markdown skeleton

```markdown
# Streaks — Sunday Report — 2026-05-11
*Models refit on 2023-2025; data through 2026-05-10*

## Your Roster — Hart of the Order (League 5652)
Sorted by composite (#hot − #cold), tiebreak max continuation probability.

| Player        | Pos    | Comp | HR        | R         | RBI       | SB    | AVG       |
|---------------|--------|-----:|-----------|-----------|-----------|-------|-----------|
| Aaron Judge   | OF     |   +3 | hot 0.71  | hot 0.68  | hot 0.74  | —     | hot 0.62  |
| Mookie Betts  | 2B/OF  |   +1 | —         | hot 0.65  | —         | —     | —         |
| Mike Trout    | OF     |    0 | —         | —         | —         | —     | —         |
| Vlad Jr       | 1B     |   −2 | —         | —         | cold 0.78 | —     | cold 0.71 |

## Top 10 Free Agent Signals
Top 10 available hitters by |composite|, non-neutral cats only.

| Player          | Pos | Comp | Active Streaks                              |
|-----------------|-----|-----:|---------------------------------------------|
| Jorge Polanco   | 2B  |  +3  | hot 0.72 R, hot 0.69 RBI, hot 0.65 AVG     |
| Joc Pederson    | OF  |  +2  | hot 0.68 HR, hot 0.62 RBI                  |

## Drivers (top peripheral signal per active prediction)

**Aaron Judge — R hot 0.68**  →  xwoba_avg +1.8σ, barrel_pct +1.4σ
**Aaron Judge — RBI hot 0.74**  →  xwoba_avg +1.8σ, ev_avg +1.2σ
**Vlad Jr — RBI cold 0.78**  →  k_pct +1.6σ, babip −1.3σ
```

### Cell-level conventions

- `<label> <prob>`: `"hot 0.71"` / `"cold 0.78"`. Probability is `P(next 14d also hot|cold)` from the model, rounded to 2 decimals.
- `—` for neutral cells.
- `cold —` for HR-cold / SB-cold cells: the model was hot-only for sparse cats (Phase 4 design), so label is shown without a probability.
- Composite is signed (`+3`, `−2`, `0`) and right-aligned.

### Terminal rendering

Same content, table library (`rich` or `tabulate` — TBD during planning, follow whatever else in the codebase already uses). Hot labels colored green, cold red, composite signed and colored. Drivers detail rendered as plain text below the tables.

## Inference math

For each (roster_or_FA_player, category):

1. Pull the player's most recent 14d window row from `hitter_windows`
   (`WHERE window_days=14 AND window_end <= today ORDER BY window_end DESC LIMIT 1`).
2. Pull the player's current label from `hitter_streak_labels` joined
   on the same window. Dense cats use `cold_method='empirical'`;
   sparse cats use `cold_method='poisson_p20'` (matching Phase 4's
   training partition).
3. If label is hot or cold AND a model exists for `(cat, direction)`,
   assemble the 12-column feature vector matching
   `EXPECTED_FEATURE_COLUMNS`, run `model.predict_proba`, store
   `P(continuation)` = the probability of the labeled class.
4. **Driver attribution:** within the model's `Pipeline`, get the
   scaled feature vector (after `StandardScaler`); compute
   `|coef_j × x_scaled_j|` for each feature; return the top 2 by
   magnitude with their z-scores (signed) and feature names.

## Edge cases

- **Yahoo player not in `hitter_windows`** (call-up, prospect not in our
  qualified-hitter set): drop from the report with a one-line note at
  the bottom ("Skipped N players: not in streaks corpus").
- **2026 player with no projection rate**: drop their predictions for
  dense cats (those models depend on `season_rate_in_category`). For
  sparse cats (which also use the rate column), drop too. Player still
  appears in the roster table but cells render as `—`.
- **Statcast lag**: Baseball Savant has a 1-2 day publication lag. The
  most recent 14d window may end 1-2 days before "today." That's fine
  — windows are calendar-aligned, the report header notes "data
  through YYYY-MM-DD."
- **Player with insufficient PAs in last 14d** (PA < 5): no window row
  exists; skip the same way as no-corpus players.
- **HR/SB cold**: Phase 4 didn't train cold models for sparse cats.
  The label still computes (poisson-based), but cell shows `cold —`
  without a probability.

## Testing

- **Unit tests** for `inference.py`: seed an in-memory DuckDB with a
  small synthetic pipeline (reuse the Phase 4 test helper
  `_seed_pipeline`), fit a tiny model, call `score_player_windows`,
  assert label + probability + drivers come out correctly.
- **Snapshot test** for `render_markdown`: feed a known `Report`
  dataclass, compare output to a checked-in golden file.
- **End-to-end smoke** for `run_sunday_report.py`: mock the Yahoo
  fetchers, run against a small fixture DB, assert the report file is
  written and contains expected section headers.

## Open questions / deferred

- **Terminal library choice** (`rich` vs `tabulate`) — decide during
  planning by inspecting what `lineup/` or `web/` already imports.
- **Refit caching** — Phase 5 refits each run. If wall time becomes a
  pain point, add a `--skip-refit` flag (already in the CLI sketch
  above) and use the latest `model_fits` row. Not the default.
- **Composite ties** — current tiebreaker is "max continuation
  probability across cats." If that's also tied, fall back to player
  name alphabetical (stable, deterministic).
- **Pitcher streaks** — out of scope for Phase 5 (per the original
  design spec's "hitters only for v1" decision). Revisit after Phase 5
  ships.

## Done criteria

1. `python scripts/streaks/run_sunday_report.py` produces a markdown
   file at `data/streaks/reports/YYYY-MM-DD.md` and a terminal
   pretty-print, end-to-end against the real local DB + live Yahoo
   league 5652.
2. The report contains a Roster section listing every hitter on Hart
   of the Order with all 5 categories, sorted by composite-then-prob.
3. The report contains a Free Agents section listing the top 10 FAs
   by `|composite|`, showing only non-neutral cats.
4. The report contains a Drivers section listing top-2 peripheral
   drivers per active prediction.
5. Unit + snapshot tests pass; full pytest suite stays green.
6. Ruff, format, vulture all clean for touched files.
