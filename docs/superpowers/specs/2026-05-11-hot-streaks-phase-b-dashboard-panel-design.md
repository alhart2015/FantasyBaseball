# Hot Streaks Phase B — Dashboard Panel — Design Spec

**Date:** 2026-05-11

## Problem

Phase 5 (`docs/superpowers/specs/2026-05-11-hot-streaks-phase-5-sunday-report-design.md`)
shipped a working weekly Sunday report — a CLI that surfaces hot/cold
streaks for Hart's roster and free agents via a markdown file plus a
terminal pretty-print. The methodology has been validated against live
2026 data and Hart is happy enough with it to move on.

The parent design spec
(`docs/superpowers/specs/2026-05-06-hot-streaks-design.md`) flagged a
deferred Phase B — "dashboard panel" — gated on "methodology proven
through v1." That gate is now passed. This spec covers Phase B only.
Optimizer integration (Phase C) remains deferred.

The pain point: the Sunday report is a once-a-week artifact. During the
week, the dashboard is where Hart actually makes lineup decisions, and
streak context is missing there. He has to swivel between a terminal
and the dashboard, or accept that streak signal is silently absent at
the moment of decision.

## Goal

Two new surfaces in the season dashboard, both fed by the same cached
inference output:

1. **`/streaks` page** — a dedicated dashboard view that ports the
   Sunday report's three sections (roster table, top free-agent
   signals, drivers) into HTML, with sortable columns and a
   configurable FA count (10 / 25 / 50).
2. **Lineup-page hitter indicator** — a streak chip on every row of
   the existing `/lineup` hitters table, showing the composite
   tone (`HOT` / `COLD` / `—`) and the single strongest hot-or-cold
   category when active (e.g., `HOT · HR`).

Both surfaces are read-only consumers of a new
`CacheKey.STREAK_SCORES` cache entry, populated by a new step in the
existing dashboard refresh pipeline. No streak math runs at request
time.

Scope is hitters only, inherited from Phase 5. The Lineup page's
pitcher table is unchanged.

## Decisions

| Decision | Choice | Rationale |
|----------|--------|-----------|
| Surface count | Two: dedicated page + Lineup indicator | Hart wants both a "destination" view (`/streaks`) and an in-flow signal where he already makes decisions (`/lineup`). |
| Lineup-indicator density | Composite chip + strongest hot/cold cat label | Densest possible signal in one row-cell. Five mini per-cat cells were ruled out as too wide for an already-wide table; composite-chip-only was ruled out as too lossy. |
| Streaks-page interaction | Sortable tables + FA-count selector (10/25/50) | Hart wants light interactivity but not full filtering. League-wide hot list deferred — would require pulling all opponent rosters into `build_report`. |
| Hitters only | Inherits from Phase 5 | Pitcher streak model isn't built. Pitcher rows on the Lineup page are unchanged. |
| Inference reuse | Refresh-pipeline step calls `build_report` from `streaks/reports/sunday.py` | `build_report` already returns a render-agnostic `Report` dataclass; both new surfaces serialize/deserialize it. No new inference code. |
| Cache key | New `CacheKey.STREAK_SCORES` | Matches the existing dashboard pattern — every page reads from a CacheKey populated by `refresh_pipeline`. |
| Cache holds top 50 FAs | Always | Dropdown is a pure client-side slice; switching 10↔50 never round-trips. |
| Refit cadence | Refit if `model_fits` is missing or ≥14 days old, else reuse | Refit is ~30s wall-time; doing it on every refresh adds variance for no signal benefit. Refresh CLI gets `--force-streaks-refit` for manual override. |
| Sort | Client-side JS on column headers | Datasets are small (≤50 rows per table); no htmx round-trip needed. Sort state is not persisted. |
| Indicator lookup | By normalized name | The Lineup page already has Yahoo player names; `Report` rows already carry `name`. Normalize on both sides (same `normalize_name` used by Phase 5). Unresolved rows render as `—`. |
| Top category | Cat whose label matches composite direction with highest probability | When composite is `HOT`, the hot cat with the highest continuation prob wins; same in reverse for `COLD`. Neutral composite has no label. |
| Empty-cache behavior | Streaks page shows "No streak data yet" empty state; Lineup chips render `—` everywhere | Dashboard works fine before first refresh — streak column is just inert. |

## Architecture

```
                ┌────────────────────────────────────────┐
                │  Refresh pipeline (existing dashboard) │
                │                                        │
                │   …                                    │
                │   ↓                                    │
                │   NEW: RefreshStreaks step             │
                │     · refit models if stale            │
                │     · build_report(top_n_fas=50)       │
                │     · serialize Report → JSON          │
                │     · write CacheKey.STREAK_SCORES     │
                │   ↓                                    │
                │   …                                    │
                └────────────────────────────────────────┘
                                  │
            ┌─────────────────────┴───────────────────┐
            ↓                                         ↓
   /streaks page                            /lineup page (hitters tbody)
   reads cache, renders                     reads cache, renders one chip
   3 sections + sort + selector             per roster-hitter row
```

The cache entry is the single source of streak truth. Both new pages
are pure read-from-cache; neither runs streak math at request time.

## New / changed components

| Kind | Name | Purpose |
|------|------|---------|
| New module | `streaks/pipeline.py` | `compute_streak_report(conn, *, config, projections_root, scoring_season, season_set_train, force_refit) -> Report`. Wraps the DB-refresh sequence (fetch → projections → windows → thresholds → labels), the refit-if-stale decision, the Yahoo fetch, and `build_report`. Returns the same `Report` dataclass Phase 5 ships. |
| New module | `streaks/dashboard.py` | `Report` ↔ JSON serialization, name-normalized lookup, indicator computation |
| Refactor | `scripts/streaks/run_sunday_report.py` | Replace inline orchestration in `main()` with a call to `compute_streak_report`. CLI flags (`--skip-fetch`, `--force-refit`) are forwarded. Markdown + terminal rendering stay in the CLI. |
| New pipeline method | `_compute_streaks` on `RefreshRun` in `web/refresh_pipeline.py` | Calls `compute_streak_report`, serializes the `Report`, writes to `CacheKey.STREAK_SCORES`. Wired into `RefreshRun.run()` near the end (after Yahoo + projections are loaded). |
| New CacheKey | `STREAK_SCORES` | Added to `CacheKey` enum in `data/cache_keys.py` |
| New route | `/streaks` in `web/season_routes.py` | Reads cache → renders template |
| New template | `web/templates/season/streaks.html` | Three sections (Roster, FAs, Drivers); sortable tables; FA-count `<select>` |
| Changed template | `web/templates/season/_lineup_hitters_tbody.html` | New rightmost "Streak" cell rendering the chip |
| Changed view | `lineup` route in `web/season_routes.py` | Inject per-hitter `streak_indicator` into context |
| Changed template | `web/templates/season/base.html` | New "Streaks" sidebar nav entry |
| New CSS | Additions to `web/static/season.css` | Chip styles (`hot` / `cold` / `neutral`), tiny tweaks to lineup hitters table for the new column |

No changes to `streaks/inference.py` or `streaks/reports/sunday.py` —
both stay as-is. The Phase 5 CLI continues to work unchanged in
behavior; only its `main()` body is refactored to delegate orchestration
to `streaks/pipeline.py`.

**Why extract `streaks/pipeline.py`?** Without it, the dashboard refresh
step would re-implement the same 7-step orchestration the Sunday CLI
already does (fetch logs/statcast → upsert projection rates → recompute
windows → recompute thresholds → apply labels → refit-or-reuse →
fetch Yahoo → `build_report`). Duplicating that logic invites drift
between the two surfaces — exactly the kind of code-quality problem
the senior-dev override calls out. Extracting once and re-using keeps
both consumers honest.

## Cache shape

`CacheKey.STREAK_SCORES` stores the serialized `Report`:

```json
{
  "generated_at": "2026-05-11T18:00:00Z",
  "report_date": "2026-05-11",
  "window_end": "2026-05-10",
  "team_name": "Hart of the Order",
  "league_id": 5652,
  "season_set_train": "2023-2025",
  "model_fit": {
    "refit_at": "2026-05-04T03:00:00Z",
    "n_train_seasons": 3,
    "reused": true
  },
  "roster_rows": [
    {
      "name": "Juan Soto",
      "positions": ["OF"],
      "player_id": 665742,
      "composite": 3,
      "max_probability": 0.62,
      "scores": {
        "HR":  {"player_id": 665742, "category": "HR", "label": "HOT",
                "probability": 0.62, "window_end": "2026-05-10",
                "drivers": [{"feature": "barrel_pct", "z_score": 1.8},
                            {"feature": "hardhit_pct", "z_score": 1.1}]},
        "R":   {"player_id": 665742, "category": "R", "label": "HOT",
                "probability": 0.58, "window_end": "2026-05-10",
                "drivers": [...]},
        "AVG": {"player_id": 665742, "category": "AVG", "label": "NEUTRAL",
                "probability": null, "window_end": "2026-05-10",
                "drivers": []}
      }
    }
  ],
  "fa_rows": [ /* up to 50 entries, same shape, sorted by |composite| desc */ ],
  "driver_lines": [
    {"player_name": "Juan Soto", "category": "HR", "label": "HOT",
     "probability": 0.62, "drivers": [...]}
  ],
  "skipped": ["Player Name — reason", ...]
}
```

Serialization rules:

- The JSON shape is a faithful 1:1 of the `Report` /
  `ReportRow` / `PlayerCategoryScore` / `Driver` / `DriverLine`
  dataclasses defined in `streaks/reports/sunday.py` and
  `streaks/inference.py`. No derived fields written to disk.
- `Report.report_date`, `Report.window_end`, and
  `PlayerCategoryScore.window_end` serialize as ISO date strings;
  deserialize back into `datetime.date`.
- `StreakCategory` and `StreakLabel` enums serialize as their
  string values.
- `PlayerCategoryScore.probability` is `null` when the label is
  neutral or no model was trained for that `(cat, direction)`.
- The serializer always writes the full payload — no incremental
  updates.
- Name normalization (`normalize_name`) happens at lookup time on
  the consumer side, not at serialization time — keeps the cache
  schema simple.

## Refit policy

`build_report` requires fitted models. Refitting all 8 models
(2 directions × 4 dense cats, plus a sparse fallback model for HR
hot-only) takes ~30s. Doing it on every dashboard refresh would add
variance for no model-accuracy benefit, since the underlying training
corpus (2023–2025) changes only when a season ends.

Policy in `compute_streak_report`:

1. Query `model_fits` for the most recent fit row (any `(category,
   direction)` will do — they're all refit together).
2. If `model_fits` is empty **or** the most recent `refit_at` is
   ≥14 days old → call `refit_models_for_report` and persist new fits.
3. Otherwise → load existing fits from `model_fits` (a new helper
   `load_models_from_fits(conn)` lives in `streaks/inference.py`
   alongside `refit_models_for_report`).
4. The function accepts a `force_refit: bool` parameter that bypasses
   the staleness check; `compute_streak_report` callers thread it
   through from their respective flags.

`scripts/streaks/run_sunday_report.py` gains `--force-refit` and the
existing `--skip-refit` flag is removed (the spec mentioned it was
deferred; `compute_streak_report` now implements the reuse path
properly, so the no-op stub is no longer needed). The dashboard's
refresh CLI doesn't get a new flag in v1 — manual refit override there
is rare enough to defer.

The `model_fit` block in the cache records whether the latest run
reused or refit, plus the `refit_at` timestamp — useful for debugging
"why does today's chip look the same as yesterday's."

## Streaks-page layout

Three sections, in order:

```
┌──────────────────────────────────────────────────────────────┐
│  Streaks · Hart of the Order · Week of May 5 – May 10        │
│  Model refit 7 days ago · 14d window                         │
│                                                              │
│  ─── Your Roster ──────────────────────────────────────────  │
│  [sortable: Name | Pos | AVG | HR | R | RBI | SB | Cmp ]    │
│  Juan Soto      OF   —    HOT  HOT  —    —    +2            │
│  Pete Alonso    1B   COLD —    COLD —    —    -2            │
│  …                                                           │
│                                                              │
│  ─── Top Free Agent Signals ──── [Show top: 10 ▼]            │
│  [sortable: Name | Pos | Active cats | |Cmp| ]              │
│  …                                                           │
│                                                              │
│  ─── Drivers ─────────────────────────────────────────────   │
│  Juan Soto · HR HOT (62%): barrel_pct +1.8z, hardhit% +1.1z  │
│  …                                                           │
└──────────────────────────────────────────────────────────────┘
```

- Each table header is `<th onclick="sortBy(this, 'cmp')">…</th>`.
  Pure JS sort; no server round-trip.
- The FA-count `<select>` is wired to JS that hides table rows
  beyond the chosen N. The cache always holds 50.
- Driver section is the same content as the Sunday report's drivers
  block, rendered as a `<ul>`. No sorting needed.
- Hot cells get the same color treatment as the lineup chip; cold
  cells likewise.

## Lineup-page integration

The hitters tbody template grows one rightmost cell per row:

```html
<td class="streak-cell">
  {% if hitter.streak_indicator %}
    <span class="streak-chip streak-{{ hitter.streak_indicator.tone }}"
          title="{{ hitter.streak_indicator.tooltip }}">
      {{ hitter.streak_indicator.label }}
    </span>
  {% else %}
    <span class="streak-chip streak-neutral">—</span>
  {% endif %}
</td>
```

The `streak_indicator` field is injected by the `lineup` route. For
each hitter the route does:

```python
indicator = build_indicator(
    name=hitter.name,
    streak_cache=read_cache(CacheKey.STREAK_SCORES),
)
```

`build_indicator` lives in `streaks/dashboard.py` and returns either
`None` (cache missing) or an `Indicator` dataclass:

```python
@dataclass(frozen=True)
class Indicator:
    tone: Literal["hot", "cold", "neutral"]
    label: str        # e.g. "HOT · HR", "COLD · AVG", "—"
    tooltip: str      # "composite=+2 · top: HR (62%), R (58%)"
```

The lookup normalizes both sides at runtime: it indexes the cache's
`roster_rows` (plus `fa_rows`, in case a roster move happened between
refreshes) by `normalize_name(row.name)`, then looks up
`normalize_name(hitter.name)`. Unresolved hitters get
`Indicator(tone="neutral", label="—", tooltip="No streak data")` so the
column never collapses mid-table.

The pitchers tbody template is untouched.

## Error handling / empty states

- **Cache missing** (first run after a fresh DB): `/streaks` renders an
  empty-state card ("No streak data yet — run a refresh"). Lineup
  chips render `—` for every row.
- **`RefreshStreaks` step fails** (DuckDB missing, model fit broken,
  Yahoo fetch error): the step logs at `WARNING` and returns; cache
  is left as-is (stale or absent). Other refresh steps continue.
  This matches how other "non-load-bearing" refresh steps degrade.
- **Unresolved player name** (Yahoo name doesn't match any projection
  CSV → no mlbam_id → not in `Report`): row appears with `—` chip and
  "No streak data" tooltip. Already-handled upstream in `build_report`.
- **`model_fits` query fails**: treat as "no fits available" → trigger
  a refit. If the refit itself fails, the step fails per above.

## Testing

- **Unit (`tests/test_streaks/test_pipeline.py`)**:
  - Stale-fit detection: empty `model_fits` → refit; recent fit →
    reuse; ≥14d-old fit → refit; `force_refit=True` → refit.
  - End-to-end with seeded DuckDB + mocked Yahoo: returns a `Report`
    with the expected roster rows and FA rows.
- **Unit (`tests/test_streaks/test_load_model_fits.py`, extension)**:
  - `load_models_from_fits` round-trips with a row written by
    `refit_models_for_report` and produces equivalent predictions on
    a fixed input (within numerical tolerance).
- **Unit (`tests/test_streaks/test_dashboard.py`)**:
  - `Report` ↔ JSON round-trip preserves every field.
  - `build_indicator`: hot composite → tone="hot" + top hot cat;
    cold composite → tone="cold" + top cold cat; neutral composite →
    tone="neutral" + no cat label; missing player → neutral, "—".
  - Tiebreak: when two cats share the top probability in the
    composite direction, alphabetical wins (stable, deterministic).
- **Unit (`tests/test_web/test_refresh_pipeline.py`, extension)**:
  - `_compute_streaks` writes through `write_cache(CacheKey.STREAK_SCORES, ...)`.
  - `_compute_streaks` failure is logged and swallowed; other steps
    continue. Cache is not overwritten on failure.
- **Integration (`tests/test_web/test_streaks_route.py`)**:
  - `GET /streaks` with a seeded cache returns 200, renders all three
    section headings, renders ≥1 roster row.
  - `GET /streaks` with no cache returns 200 and the empty-state copy.
- **Integration (`tests/test_web/test_season_routes.py`, extension)**:
  - `GET /lineup` with a seeded cache injects chips for every hitter.
  - `GET /lineup` with no cache renders `—` chips for every hitter.
- **Snapshot (`tests/test_web/test_streaks_snapshot.py`)**:
  - One canonical Streaks page HTML snapshot to catch unintentional
    visual regressions.
- **Refactor regression (`tests/test_scripts/test_run_sunday_report.py`, existing)**:
  - The Phase 5 tests must continue to pass after the CLI is refactored
    to delegate to `compute_streak_report` — they pin the public
    behavior, not the internals.
- All new modules added to `[tool.mypy].files`.

## Open / deferred

- **Per-player history / drill-down**. v1 has hover tooltip only;
  clicking a row does nothing new. Adding sparklines or a per-player
  modal is a natural Phase B+ extension once methodology has been
  validated in dashboard form.
- **League-wide hot list** (every rostered hitter in the league, not
  just our roster + FAs). Would require pulling opponent rosters into
  `build_report` (the dashboard already fetches them — wiring is the
  blocker, not data).
- **Pitcher streaks**. Out of scope until a pitcher model exists. The
  Lineup page's pitcher table is untouched.
- **Persisting FA-count or sort preference**. v1 forgets across
  reloads; cheap to redo. Persist via cookies if it becomes annoying.
- **Refresh-pipeline time impact**. `compute_streak_report` runs the
  same orchestration the Sunday CLI runs: incremental Statcast/game-log
  fetch (fast on warm DB), projection-rates upsert, windows / thresholds
  / labels recompute, refit-or-load, Yahoo roster + FA fetch, score,
  serialize. Without refit, end-to-end is ~10-15s on a warm DB. With
  refit (≤ once per 14 days), add ~30s. Acceptable for the cadence; if
  it isn't, move refit to a separate weekly job.

## Done criteria

1. **Sidebar nav** shows a new "Streaks" entry between Lineup and
   Roster Audit. Clicking it loads `/streaks`.
2. **`/streaks` page** renders Roster / Top FAs / Drivers tables driven
   by `CacheKey.STREAK_SCORES`, with sortable column headers and a
   working 10 / 25 / 50 FA-count dropdown.
3. **`/lineup` page** hitters table shows a streak chip per row
   (composite tone + strongest hot/cold cat label when active);
   chip is `—` when no signal or no resolution.
4. **A dashboard refresh** populates `CacheKey.STREAK_SCORES`. Models
   are refit only when `model_fits` is missing or ≥14 days old, or
   when `--force-streaks-refit` is passed.
5. **Tests** — unit, integration, snapshot all green; full project
   `pytest`, `ruff check .`, `ruff format --check .`, `mypy` (for
   touched files) all clean.

## Progress Log

This section is appended to as work happens. Each milestone gets a
dated entry.

### 2026-05-11 — design spec written

- Brainstormed scope after merging PR #68 (Phase 5).
- Two surfaces chosen: dedicated `/streaks` page + Lineup
  indicator chips.
- Inference reused via `build_report` from Phase 5; no new streak math.
- Next: write the implementation plan, then execute.
