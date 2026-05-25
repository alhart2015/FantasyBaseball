# Category Bars: ranked projections with uncertainty

Date: 2026-05-25
Status: Approved (design); pending implementation plan
Branch: feat/category-bars-standings

## Problem

The season dashboard projects where each team will finish in all ten 5x5
roto categories, but it presents those projections only as point estimates
in tables. A point estimate hides uncertainty: a 5-run projected lead in
Runs is meaningless if the standard deviation on each team's total is 40
runs. We want a view that shows ranking AND uncertainty at a glance, so the
manager can tell "who is clearly ahead of whom" from "what is a toss-up."

The inspiration is a horizontal dot-plot with error bars: each competitor is
one row, sorted best-to-worst, with a central dot for the estimate and
whiskers for the uncertainty band. Overlapping whiskers between two rows mean
the gap is within the noise; a clear gap means a real edge.

## Goal

Add a "Category Bars" tab to the `/standings` page. At the top:

- a category selector (the 10 roto categories), and
- a projection toggle (Preseason / Current).

Below: a horizontal dot-plot of all teams for the selected category and
projection. Central dot = projected category total. Whiskers = +/-1 standard
deviation. Teams sorted best-on-top.

## Non-goals

- No new projection or variance math. Every number this view needs is
  already computed by the refresh pipeline and cached.
- No Monte Carlo per-category distributions. The Monte Carlo path only
  produces a per-category distribution for the USER's team, not all ten
  teams. The analytic team standard deviations are the only source that
  covers every team in every category, so they are what we plot.
- No new API endpoint. Data is embedded in the page (see Architecture).
- No "clearly ahead vs toss-up" auto-annotation in v1. The overlapping
  whiskers communicate this visually; an explicit callout can come later.

## Background: where the data already lives

The refresh pipeline (`src/fantasy_baseball/web/refresh_pipeline.py`,
end-of-season projection step) writes `CacheKey.PROJECTIONS` with four
relevant pieces:

- `projected_standings` -- current/ROS projected end-of-season totals
  (YTD actuals + rest-of-season projection).
- `team_sds` -- per-team, per-category standard deviations for the current
  projection, scaled by `sqrt(fraction_remaining)` so the band shrinks as
  the season is played out.
- `preseason_standings` -- preseason (Opening Day) full-season projected
  totals.
- `preseason_team_sds` -- per-team, per-category standard deviations for the
  preseason projection (full-season, `sd_scale = 1.0`).

The standings route (`src/fantasy_baseball/web/season_routes.py`, the
`/standings` handler) already reads this cache and builds two display dicts
via `format_standings_for_display`:

- `current_projected_data` (with `team_sds`)
- `preseason_data` (with `preseason_team_sds`)

Each display dict has a `teams` list, and each team entry already carries:

```python
{
    "name": str,
    "team_key": str,
    "is_user": bool,
    "stats": CategoryStats,        # central value per category (the dot)
    "sds": {Category: float},      # +/-1 SD per category (the whisker)
    "roto_points": {Category: float},
    "roto_total": float,
    "color_intensity": {Category: float},
    ...
}
```

So both the dots and the error bars for both projection flavors are already
in the template context. This feature is pure presentation.

Rate-stat variance (AVG, ERA, WHIP) is already handled correctly inside
`build_team_sds` -- those categories aggregate variance from numerator
components rather than treating the rate as additive. The view does not need
to know about this; it just reads the resulting `sds`.

## Architecture

### Component 1: backend formatter (new)

`format_category_bars_for_display(preseason_data, current_projected_data)`
in `src/fantasy_baseball/web/season_data.py`.

Input: the two existing display dicts (each a `{"teams": [...]}` structure as
described above).

Output: chart-ready, JSON-serializable nested dict:

```python
{
  "current": {
     "R":  [ {"team": str, "value": float, "sd": float, "is_user": bool}, ... ],
     "HR": [ ... ],
     ...   # all 10 categories
  },
  "preseason": {
     "R":  [ ... ],
     ...
  }
}
```

Rules:

- Category keys are the uppercase short names (R, HR, RBI, SB, AVG, W, K, SV,
  ERA, WHIP) so they are JSON-safe and match the existing trends payload
  convention.
- Each per-category list is sorted **best-to-worst**:
  - normal categories (the 8 not in `INVERSE_CATEGORIES`): descending by value;
  - inverse categories (ERA, WHIP, per the existing `INVERSE_CATEGORIES`
    constant): ascending by value.
  - Stable sort; ties keep input order (which is itself deterministic).
- `value` comes from the team entry's `stats` (indexed by the `Category`
  enum); `sd` from the team entry's `sds` (default 0.0 when the category is
  missing from `sds`, e.g. when no `team_sds` were cached).
- `is_user` is passed through unchanged.

This is the single place the inverse-category sort direction and the
SD-default logic live. It is independently unit-testable with plain dicts.

### Component 2: route wiring (small change)

In the `/standings` handler in `season_routes.py`, after `preseason_data` and
`current_projected_data` are built, call the new formatter and pass the result
to the template. The template embeds it as JSON in a script tag, mirroring the
existing `breakdown-data` pattern:

```html
<script id="category-bars-data" type="application/json">{{ category_bars | tojson }}</script>
```

When projections are absent (pre-refresh), pass an empty/None value and the
tab renders the same "Run a refresh first" empty state the other tabs use.

### Component 3: template + controls (new tab)

In `src/fantasy_baseball/web/templates/season/standings.html`, add a fourth
top-level tab ("Category Bars") next to Current / Projected / Monte Carlo,
using the existing `toggleTopView`-style tab machinery. The tab contains:

- a category selector (10 options -- buttons or a segmented control),
- a projection toggle (Preseason / Current),
- a single `<canvas>` for the Chart.js chart,
- a small "lower is better" note shown only for ERA / WHIP.

### Component 4: chart module (new)

`src/fantasy_baseball/web/static/season_category_bars.js`.

- Reads and parses `#category-bars-data`.
- Loads Chart.js 4 + the `chartjs-chart-error-bars` plugin
  (https://github.com/sgratzl/chartjs-chart-error-bars) on the standings
  page (Trends already loads Chart.js 4; this adds the horizontal
  error-bar controller).
- Renders a horizontal dot-plot for the currently selected
  (category, projection): each team is one row at `x = value` with horizontal
  error bars `xMin = value - sd`, `xMax = value + sd`; the Y axis is the team
  labels in sorted order (best on top). The exact plugin controller is a
  plan-level decision -- `barWithErrorBars` with `indexAxis: "y"` gives native
  horizontal error bars on a category axis and can be styled toward the
  dot look; `scatterWithErrorBars` gives a true dot but needs the category
  axis handled explicitly. The plan picks whichever renders the reference
  look with the least fighting of the library.
- Re-renders in place (update existing chart, no network call) when the
  category or projection control changes.

### Visuals

- Teams on Y, best on top. Value on X, scaled to the selected category's data
  (each category has its own scale -- categories are shown one at a time, so
  Runs ~800 and ERA ~3.80 never share an axis).
- X axis stays natural (low on the left, high on the right) for all
  categories. Best-on-top ordering carries the ranking; ERA/WHIP get a
  "lower is better" label. (Reversing the axis for the two inverse cats so
  "better is always the same side" was considered and deferred -- trivial to
  add later if desired.)
- The user's team is highlighted in the existing dashboard red (`#e15759`);
  other teams use a neutral dark dot; whiskers use a lighter accent. Matches
  the reference and the site palette.

## Edge cases

- **No SDs cached** (`sds` missing/empty for a team or category): `sd`
  defaults to 0.0, whiskers collapse to a bare dot, chart still renders.
- **No projections yet** (pre-refresh): tab shows the standard "Run a refresh
  first" empty state; no chart is drawn.
- **Ties** in the sorted value: stable sort preserves deterministic input
  order.
- **Rate-stat precision**: AVG to 3 decimals, ERA/WHIP to 2, counting stats
  as integers -- the chart's tick/tooltip formatting keys off whether the
  category is a rate stat (reuse the existing precision convention from the
  standings tables).

## Testing

Backend (`tests/test_web/test_season_data.py`):

- `format_category_bars_for_display` sorts a normal category (e.g. R)
  descending and an inverse category (e.g. ERA) ascending.
- `is_user` flag is preserved for the user's team.
- `sd` is passed through from `sds`, and defaults to 0.0 when a category is
  absent from `sds`.
- Both `preseason` and `current` keys are present and cover all 10 categories.
- Rate-stat values (AVG/ERA/WHIP) survive unrounded in the payload (rounding
  is a display concern, not a data concern).

Route (`tests/test_web/test_season_routes.py`):

- When `CacheKey.PROJECTIONS` is populated, the `/standings` response embeds a
  non-empty `#category-bars-data` JSON blob.
- When projections are absent, the page still renders (empty-state path) and
  does not error.

Frontend: covered by the existing manual/dashboard smoke path; no JS unit
harness exists in this repo, so chart rendering is verified by running the
dashboard locally (documented in the plan's verification step).

## Verification

- `pytest tests/test_web/ -v` (the touched area) green.
- `ruff check .`, `ruff format --check .`, `vulture` clean for touched files.
- `mypy` if any touched file is in the `[tool.mypy].files` list.
- Manual: run `python scripts/run_season_dashboard.py`, open `/standings`,
  switch to the Category Bars tab, confirm the chart renders for several
  categories and both projection toggles, and that the user's team is
  highlighted and ERA/WHIP sort lowest-on-top.
