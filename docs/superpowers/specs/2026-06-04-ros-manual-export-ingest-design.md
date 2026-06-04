# ROS manual-export ingest — design

Date: 2026-06-04
Status: approved (brainstorming) — ready for implementation plan

## Context / problem

The overnight FanGraphs ROS projection fetch broke on 2026-06-04: FanGraphs put
the whole `fangraphs.com` domain behind a Cloudflare managed/Turnstile challenge
("Just a moment..."), so the HTTP fetch (`_fetch_fangraphs_data` in
`data/fangraphs_fetch.py`) gets `403` for every system. Recon established that no
lightweight bypass works (plain `requests`, `curl_cffi` TLS impersonation, and
even headed Playwright with a cold profile all return the challenge); only a
warm, human-driven browser clears it.

FanGraphs' Contact Us page states their position explicitly:

> Outside of one click data exports for FanGraphs Members, we do not support
> exporting data in any other way. We do not support web scraping, API
> endpoints, importing data automatically to Excel or Google Sheets, web
> queries, etc.

So any automated extraction (scraping / API / web query / headless browser) is
off the table by the data owner's stated rules. The **one-click member export**
is the only supported path, and it is a human clicking Export in their own
browser.

## Goal

Make refreshing ROS projections a low-friction, repeatable local command that
stays entirely within FanGraphs' stated boundary: the human does the supported
one-click exports; a script only handles the resulting CSV files on the local
machine and pushes the blended result to prod Upstash. Target cadence: every few
days / on demand.

## Non-goals

- No automated extraction from FanGraphs (no scraping, API, headless browser,
  web query). Removed the Playwright / curl_cffi recon dependencies.
- No server-side (Render) fetch. The job runs locally; Render only consumes the
  cached blob.
- No change to the consumption side (refresh pipeline, dashboard) — the merged
  resilience work (PR #119: fetch-success gate, refuse-stale write guard,
  date-based snapshot selection, read-side staleness warning) already protects
  prod and is unchanged here.

## Approach

A single guided CLI, `scripts/ingest_ros_export.py`. The human exports the 5
systems x {hitters, pitchers} (10 one-click exports) in their browser; the
script walks them step by step, picks up each freshly-downloaded CSV, validates
it, stages it into today's dated snapshot dir under our naming convention, then
reuses the already-proven blend + push-to-prod tail.

Boundary rationale: the script never contacts FanGraphs. It reads files the user
legitimately exported via the supported one-click path. That is file handling on
the user's own machine — none of the prohibited extraction methods.

## Components

### 1. Guided ingest CLI — `scripts/ingest_ros_export.py`

- Resolves today's snapshot dir:
  `data/projections/{config.season_year}/rest_of_season/{local_today()}/`.
- Iterates the 10 steps in a fixed order (each system from
  `config.projection_systems`, hitters then pitchers). For each step:
  - Print a prompt: `Export {SYSTEM} {hitters|pitchers} from FanGraphs, then
    press Enter (s=skip, q=abort)`.
  - Record a timestamp when the prompt is shown.
  - On Enter: call the staging helper (below). On failure, re-prompt the same
    step. On `s`, skip the system (exclude from blend). On `q`, abort without
    pushing.
- After staging, run the push helper (below). Only push if at least one system
  is complete (both hitters + pitchers staged); otherwise abort and leave the
  last-good prod blob intact (mirrors the server-side fetch-success gate).
- Flags: `--source <dir>` (default `~/Downloads`), `--season <year>` (default
  from config), `--no-push` (stage only, for testing/dry runs).

### 2. Stage-newest-validated-export helper (pure, testable)

`stage_export(source_dir, since_ts, system, player_type, dest_dir) -> Path | None`

- Find the newest `*.csv` in `source_dir` with mtime `>= since_ts` (so it picks
  the file the user just exported, not a stale one). If none, return `None`
  (caller re-prompts).
- Validate type by parsing with the existing `parse_hitting_csv` /
  `parse_pitching_csv` (`data/fangraphs.py`): a hitters export must satisfy
  `REQUIRED_HITTING_COLS`, a pitchers export `REQUIRED_PITCHING_COLS`. A
  wrong-type or unparseable file returns `None` (caller re-prompts), so garbage
  is never staged.
- Copy the validated file to `dest_dir/{system}-{hitters|pitchers}.csv`.
- I/O (the prompt/print/input) is injected behind callables so the
  step-iteration core is unit-testable without real stdin.

### 3. Push-to-prod helper (reuse of the proven restore flow)

`push_ros_snapshot_to_prod()` — same shape as the 2026-06-04 manual restore:

- Flip `RENDER=true` and `kv_store._reset_singleton()` BEFORE importing the
  pipeline (the env gate is read once at import; this is the blessed
  local->prod pattern from `scripts/refresh_remote.py`).
- Load config, load roster names from prod (`get_latest_roster_names(get_kv())`).
- Call `blend_and_cache_ros(projections_dir, systems, weights, roster_names,
  season_year)` — writes `cache:ros_projections` + `cache:full_season_projections`
  to prod. The merged stale guard + date-based selection mean a today-dated
  snapshot blends and writes; a stale/missing one is refused (last-good kept).
- Verify via `build_explicit_upstash_kv`: print snapshot date, hitter/pitcher
  counts, and a spot-check (e.g. Aaron Judge ROS vs full-season HR) so a bad
  ingest is visible immediately.

## Data flow

```
human one-click exports (browser, supported)
  -> 10 CSVs land in ~/Downloads
  -> guided CLI stages each (newest + type-validated) into
     data/projections/{year}/rest_of_season/{today}/{system}-{hitters|pitchers}.csv
  -> RENDER-flip -> blend_and_cache_ros -> prod cache:ros_projections + full_season
  -> verify (snapshot date, counts, spot-check)
```

Dashboard standings/MC are computed by the separate refresh; after a push, the
user may run `scripts/refresh_remote.py` to propagate the fresh ROS into the
displayed standings (noted, not automated here).

## Error handling

- Wrong-type / missing export at a step -> re-prompt that step (never stages a
  bad file).
- Skipped system (`s`) -> excluded from the blend; `blend_projections` already
  tolerates absent systems.
- Abort (`q`) or zero complete systems -> do not push; last-good prod blob
  preserved.
- Push failure (KV error) -> propagates loudly (the blend path is fail-loud);
  the user re-runs. Prod is unchanged on failure because the write is the last
  step.

## Testing

- Unit: `stage_export` stages a valid hitters CSV, rejects a wrong-type
  (pitchers-into-hitters) CSV, and returns `None` when no file is newer than
  `since_ts` (temp dirs + fixture CSVs; reuse `tests/fixtures`).
- Unit: the step-iteration + success-gate logic with injected I/O (no real
  stdin) — verifies skip/abort/partial-complete decisions and that push is
  gated on >= 1 complete system.
- The blend + push tail is already covered by existing `test_ros_pipeline.py` /
  `test_ros_only_regression.py`; no live-network tests.

## How to run

```
# 1. In your browser, be ready to one-click export each system (hitters + pitchers).
python scripts/ingest_ros_export.py
# follow the 10 prompts; it stages, blends, and pushes to prod, then verifies.
# 2. (optional) propagate into dashboard standings:
python scripts/refresh_remote.py
```

## Open questions

None. (FanGraphs export default filenames are irrelevant — the guided flow
picks the newest download per step and names it, so no manual renaming and no
dependence on FanGraphs' filename scheme.)
