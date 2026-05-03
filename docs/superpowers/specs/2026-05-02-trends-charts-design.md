# Trends Charts — Design

**Date**: 2026-05-02
**Page**: `/trends` (new)
**Scope**: Two interactive line charts showing standings movement over the season — actuals and projected ERoto — with per-stat tabs.

## Goals

1. Let the user see whether they're improving or sliding over time, instead of inferring it from a single-day standings snapshot.
2. Surface threats to the projected first-place finish as they develop, not the day they overtake.
3. Show every category, not just the headline roto total — so a sliding category is visible category-by-category.

## Non-goals

- Per-player time series.
- Mid-season trade timeline / annotation overlays.
- Streaks, rolling averages, or other derived analytics on top of the raw lines.
- Mobile-specific layout work — desktop dashboard parity is enough.

---

## Architecture

A new top-level page `/trends` in the season dashboard sidebar nav. Two stacked Chart.js line charts:

1. **Actual standings over time** — one line per team, x-axis = date, y-axis = roto points (default tab) or a single stat total (other tabs).
2. **Projected ERoto over time** — same shape, but the y-axis tracks projected end-of-season values.

Both charts share interactivity behavior (hover-to-highlight, click-to-toggle in the legend) and a shared color palette so the same line represents the same team across both charts.

---

## Data model

### Actual standings

The Redis hash `standings_history` already holds every snapshot, keyed by `effective_date`, with the canonical `Standings.to_json()` shape — no schema change needed. Per-stat values (`R`, `HR`, ..., `WHIP`) are already in `entry.stats`; `yahoo_points_for` already lives on each entry.

### Projected ERoto — new hash `projected_standings_history`

Mirrors `standings_history`:
- Hash key: `projected_standings_history`
- Field: `effective_date` ISO string (`YYYY-MM-DD`)
- Value: JSON of `ProjectedStandings.to_json()` (existing serializer — `{effective_date, teams: [{name, stats}]}`)

New helpers in `data/redis_store.py`, mirroring the existing `write_standings_snapshot` / `get_standings_history` pair:

```python
PROJECTED_STANDINGS_HISTORY_KEY = "projected_standings_history"

def write_projected_standings_snapshot(client, projected: ProjectedStandings) -> None: ...
def get_projected_standings_day(client, snapshot_date: str) -> ProjectedStandings | None: ...
def get_projected_standings_history(client) -> dict[str, ProjectedStandings]: ...
```

Same idempotent-overwrite semantics as `write_standings_snapshot` (last-write-wins per date).

### Refresh pipeline change

In `web/refresh_pipeline.py::_build_projected_standings`, after the existing `write_cache(CacheKey.PROJECTIONS, ...)`, also call `write_projected_standings_snapshot(client, self.projected_standings)`. From that point forward every refresh appends a snapshot.

`kv_sync.sync_remote_to_local` already replicates the matching set of history hashes; add `projected_standings_history` to its list so local SQLite stays in sync.

---

## Backfill

One-time script `scripts/backfill_projected_standings_history.py`:

```
1. Load today's preseason + ROS projection DataFrames from Redis (same path as
   refresh_pipeline._load_projections).
2. league = League.from_redis(season_year) — already loads every team's
   roster history per date.
3. Build a per-date rosters map by walking team.rosters across all teams:
       team_rosters_by_date[snap_date][team_name] = [RosterEntry, ...]
4. For each snapshot date D in team_rosters_by_date:
   a. For each team, hydrate roster entries against today's projections
      using data.projections.hydrate_roster_entries (same call refresh_pipeline
      uses — no new code).
   b. ProjectedStandings.from_rosters(team_rosters, effective_date=D)
      — uses projection_source="full_season_projection" by default.
   c. write_projected_standings_snapshot(client, projected_standings).
5. Idempotent — re-running overwrites; safe to re-run after the live pipeline
   has populated newer snapshots (last-write-wins per date).
```

### Math note on the approximation

The user's "extrapolate ROS backwards by linear scaling" framing is mathematically equivalent to using today's per-player `full_season_projection` (= today's ROS + today's YTD) under a constant-rate assumption: `day_D_full_season = day_D_YTD + day_D_ROS = rate × elapsed_D + rate × remaining_D = rate × season = today's full_season`.

Consequence: backfilled lines reflect **roster movement only**. Projection-drift signal (e.g., a slumping star getting projected lower mid-season) only appears in snapshots persisted forward from the day this lands. The user accepted this tradeoff during brainstorming — early-season, roster moves dominate anyway.

### Edge cases

- **Player on day-D roster missing from today's projections** (released, retired): hydration returns no match → contributes zero stats, same as live pipeline. Logged via the existing `hydrate_roster_entries` warning path.
- **Days with no roster snapshot**: skipped. Chart renders a gap. `weekly_rosters_history` has been populated daily, so this should not occur in practice.
- **Days with a standings snapshot but no roster snapshot** (or vice versa): each chart is independent — actuals chart shows the standings point, projected chart skips it.

---

## API

### `GET /trends`

Renders `season/trends.html`. No query parameters.

### `GET /api/trends/series`

Single endpoint serving both charts. Read-only, no auth (matches `/api/teams`).

**Response**:

```json
{
  "user_team": "Hart of the Order",
  "categories": ["R", "HR", "RBI", "SB", "AVG", "W", "K", "SV", "ERA", "WHIP"],
  "actual": {
    "dates": ["2026-03-29", "2026-03-30", "..."],
    "teams": {
      "Hart of the Order": {
        "roto_points": [60.0, 62.5, "..."],
        "stats": {
          "R": [120, 132, "..."],
          "HR": ["..."],
          "...": []
        }
      },
      "...": {}
    }
  },
  "projected": { "...": "same shape" }
}
```

**Server-side construction**:
- Read `standings_history` and `projected_standings_history`. Sort by `effective_date`.
- For each snapshot, run `score_roto` to get per-team CategoryPoints.
- For the actuals chart: prefer `entry.yahoo_points_for` for `roto_points` when present (matches `/standings`); fall back to `score_roto.total`.
- For the projected chart: always use `score_roto.total` (no Yahoo authority for projections).
- Transpose the sequence of snapshots into `{team_name: {roto_points: [...], stats: {cat: [...]}}}`.
- Teams that appear in some snapshots but not others (renames, mid-season team additions): include all team names; populate `null` for missing dates so Chart.js renders a gap.

One round trip is fine: 12 teams × 11 series × ~150 dates × ~4 bytes ≈ 80KB worst-case.

---

## Frontend

### Template — `web/templates/season/trends.html`

Extends `season/base.html`. Sidebar nav in `base.html` gains a new "Trends" link directly under "Standings" (the two views answer related "where do I stand" questions).

```html
{% block head_extra %}
<script src="https://cdn.jsdelivr.net/npm/chart.js@4"></script>
{% endblock %}

{% block content %}
<h1>Trends</h1>

<section class="trends-chart">
  <h2>Actual Standings</h2>
  <nav class="tab-strip" data-target="actual">
    <button class="active" data-tab="roto">Roto Points</button>
    <button data-tab="R">R</button>
    <!-- ... HR, RBI, SB, AVG, W, K, SV, ERA, WHIP -->
  </nav>
  <canvas id="chart-actual"></canvas>
</section>

<section class="trends-chart">
  <h2>Projected ERoto</h2>
  <nav class="tab-strip" data-target="projected">
    <!-- same tabs -->
  </nav>
  <canvas id="chart-projected"></canvas>
</section>

<script src="{{ url_for('static', filename='season_trends.js') }}"></script>
{% endblock %}
```

### JS — `web/static/season_trends.js`

Module-scoped, no build step:

1. On `DOMContentLoaded`, fetch `/api/trends/series`.
2. Build two Chart.js line charts. Datasets are one-per-team; labels are dates.
3. **Tab switching**: store the full series payload in module state; tab click swaps each dataset's `data` array (`stats[cat]` or `roto_points`) and calls `chart.update()`. No chart re-creation.
4. **Hover-to-highlight**: register an `onHover` handler that, when a single dataset is "nearest", lowers the alpha on every other dataset's `borderColor` and `backgroundColor`. Reset on mouseout.
5. **Click legend item**: rely on Chart.js default — clicking a legend item toggles dataset visibility.

### Color palette

12 distinct qualitative colors. The user's team gets a fixed color (not from the 12-color rotation) — bold red, `borderWidth: 4`. Other teams use a stable mapping (`team_name → palette[i]`) computed once from the sorted team list so the same team gets the same color across both charts and across page loads.

### Rate stats (AVG, ERA, WHIP)

Plotted directly off the stored stat (Yahoo's standings already store the rate). Y-axis adapts via Chart.js auto-scaling. No category inversion in the visualization — ERA/WHIP "down is good," but readers of fantasy charts already know that, and inversion would surprise.

---

## Testing

### Unit

- `tests/test_data/test_redis_store.py` — add tests for `write_projected_standings_snapshot`, `get_projected_standings_day`, `get_projected_standings_history`. Mirror the existing `write_standings_snapshot` test pattern.

### Integration

- `tests/test_web/test_refresh_pipeline.py` — assert `_build_projected_standings` writes to `projected_standings_history` after the cache write.
- `tests/test_web/test_season_routes.py` — `/api/trends/series` returns expected shape from a seeded mock KV. Cover:
  - Empty history (both hashes empty)
  - Actuals only (projected_standings_history empty)
  - Both populated, including a date that's missing from one of the two hashes (verify gap handling)
  - User team highlight (verify `user_team` field)

### Backfill script

- `tests/test_scripts/test_backfill_projected_standings_history.py` — seed a tiny `weekly_rosters_history` (3 dates × 2 teams) and minimal projection data; run the backfill; assert the resulting `projected_standings_history` matches expected per-team CategoryStats.

### Frontend

No automated tests — the repo has no JS testing infrastructure. Per CLAUDE.md, manually verify in the browser:
- Both charts load with 12 teams.
- Tab switching swaps the y-axis without flicker.
- Hover dims other lines; mouseout restores.
- Legend click toggles a team off; second click restores.
- User's team is visually distinct.

---

## Phasing

- **Phase 1** — Data model: redis_store helpers, `kv_sync` addition, refresh_pipeline write call. Tests.
- **Phase 2** — Backfill script + tests.
- **Phase 3** — `/trends` route, `/api/trends/series` endpoint, server-side data assembly. Route tests.
- **Phase 4** — Template + JS + sidebar nav link. Manual browser verification.

Each phase ≤ 5 files to satisfy the project's Phased Execution rule.

---

## Open follow-ups (not in scope)

- Future: also persist `team_sds` per snapshot so the projected chart can use EV-based scoring (currently uses rank-based `score_roto`). Today's chart matches the standings page; if we ever change `/standings` to default to EV scoring, revisit.
- Future: a "your team only" view that overlays opponents you're competing for top spot with — could be a simple multi-select filter on top of the same data.
