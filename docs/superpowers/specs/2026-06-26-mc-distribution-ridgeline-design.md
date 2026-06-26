# MC Distribution Ridgeline View - Design

Date: 2026-06-26
Status: Approved (pending spec review)

## Problem

Between 6/25 and 6/26 the Monte Carlo (MC) standings flipped dramatically (user
~50% to win -> ~45/45 tossup with "hello peanuts") off one small roster move.
The cause is structural: the user is projected close to a rival in many
categories, so a small shift swings several category finishes at once and moves
final standings a lot. The dashboard currently exposes only point estimates and
a few percentiles (median, p10, p90, win%, top3%) plus a hard-to-read
category-bars scatter. None of these convey the *range and overlap* of outcomes,
which is exactly the uncertainty that makes the race a tossup.

Goal: quantify and visualize that uncertainty so the user can see, at a glance,
where their range of outcomes falls relative to every other team -- both for
overall roto points and for each individual category.

## Solution overview

A new **Distributions** view on the standings page (replacing the existing
"Category Bars" tab) that renders a **ridgeline plot**: one density curve per
team, stacked vertically and sorted by median, on a shared x-axis. The user's
row is highlighted. Overlap between teams reads as horizontal alignment of their
density mass -- contested races jump out visually.

A selector chooses what the x-axis measures:

- **Overall** -> x = total roto points (league range 10..120 for a 12-team,
  10-category league). This chart directly shows the tossup as overlapping mass
  between the user and the rival.
- **A category** (R, HR, RBI, SB, AVG, W, K, SV, ERA, WHIP) -> x = raw stat
  total for that category. A **Totals | Points** sub-toggle (shown only when a
  category is selected) swaps the raw-total bell curve for the discrete
  category roto-points (1..N) distribution.

```
[ Overall ] [ R ][ HR ][ RBI ][ SB ][ AVG ][ W ][ K ][ SV ][ ERA ][ WHIP ]
                                                    ( Totals | Points )

         x = total roto points (Overall)  /  raw stat total (category) ->
  1  Peanuts   ____/\______________
  2  YOU *     ______/\____________   <- highlighted (user color #e15759)
  3  Team C    _________/\_________
  4  Team D    ___________/\_______
     ...       (one row per team, sorted by median desc)
```

Rows are sorted by median descending (standings order), share one x-axis per
chart (so overlap is comparable), and carry a light median tick. The user's row
uses the existing user color (#e15759).

## Data model: retain a compact distribution

### What exists today

`run_ros_monte_carlo()` (in `src/fantasy_baseball/simulation.py`) runs
n_iterations (default 1000). Internally, per iteration it already computes every
team's total roto points, every team's raw category totals (from
`simulate_remaining_season_batch()`, shape `{team: {category: ndarray(n_iter)}}`),
and the per-iteration roto-points ranking across all teams. After computing
aggregates it **discards** all per-iteration arrays; only scalar summaries
survive to the cache (median_pts, p10, p90, first_pct, top3_pct, and per-category
risk for the user team).

So the raw material for distributions already exists per iteration -- it is just
thrown away.

### What we retain

Before discarding the per-iteration arrays, build compact **density curves** and
return them. We do NOT cache the raw 1000-length arrays (~250k numbers across
all teams/categories -- heavy to serialize and ship to the browser). Instead:

- **Continuous metrics** (overall total points; each category's raw totals):
  a lightweight Gaussian kernel density estimate (KDE), numpy-only (no scipy
  dependency), sampled at a fixed number of points (~60) on a **shared x-grid**
  per metric. Shared grid is required so ridgeline rows are horizontally
  comparable. Result per metric: `{ "x": [...], "teams": { team: [y...] } }`.
- **Discrete metric** (category roto-points): the exact probability mass over
  the discrete support of roto points. No smoothing -- roto points are genuinely
  discrete. Note roto ties split points, so the support is not strictly integers
  1..N: tied finishes produce half-points (e.g. two teams tied for the top get
  11.5 each in a 12-team league). The support is therefore the set of distinct
  point values that actually occur across iterations (a fixed half-integer grid
  from 1 to N is a safe superset). Result per category:
  `{ "x": [...point values...], "teams": { team: [p...] } }` where each team's
  `p` sums to 1.

Approximate cache cost: ~5k numbers for continuous curves + ~1.4k for discrete
PMFs ~= under 7k numbers total. JSON-serializable, no numpy in the payload.

### KDE detail

Implement a small numpy-only Gaussian KDE helper (a few lines: for each grid
point, sum Gaussian kernels centered on the samples; normalize). Bandwidth via a
standard rule of thumb (Silverman/Scott) with a floor to avoid degenerate
spikes when variance is tiny. The helper lives next to the simulation code (or a
small util module) and is unit-tested independently. Rationale for retaining the
real MC shape rather than a single Gaussian per team: close races can be skewed
or bimodal, and that shape is the whole point of the visualization.

### Return shape (new key on the MC result)

`run_ros_monte_carlo()` gains a `"distributions"` key alongside the existing
`team_results` / `category_risk`:

```python
"distributions": {
    "overall": {                 # continuous: total roto points
        "x": [float, ...],       # shared grid (~60 points)
        "teams": {team_name: [float, ...]}  # density at each x
    },
    "category_totals": {         # continuous: raw stat totals
        "R": {"x": [...], "teams": {team: [...]}},
        ...  # one per category
    },
    "category_points": {         # discrete: roto points (half-integer grid, ties split)
        "R": {"x": [1, 1.5, ..., N], "teams": {team: [p, ...]}},  # each p-list sums to 1
        ...
    },
    "user_team": str,            # for highlight convenience (optional; also known to frontend)
}
```

## Caching and data flow

Follows existing patterns exactly:

1. `run_ros_monte_carlo()` returns the new `distributions` key
   (`src/fantasy_baseball/simulation.py`).
2. `refresh_pipeline.py` folds it into the existing `cache:monte_carlo` payload
   (currently writes `{base, baseline_meta, rest_of_season}`); the distributions
   ride inside `rest_of_season`, so no new cache key is needed.
3. A new `format_distributions_for_display()` in
   `src/fantasy_baseball/web/season_data.py` reshapes the cached distributions
   into a template-ready structure (sort teams by median, mark the user team,
   attach labels). Mirrors the existing `format_*_for_display` functions.
4. The standings route (`src/fantasy_baseball/web/season_routes.py`) passes the
   formatted distributions to the template as `distributions`.
5. `standings.html` embeds it as `<script type="application/json"
   id="distributions-data">{{ distributions | tojson }}</script>` -- the same
   embedded-JSON pattern the category bars use today.

## Rendering

New static file `src/fantasy_baseball/web/static/season_distributions.js`.

Chart.js 4.4.4 is already loaded but has no native ridgeline. Render a
**custom ridgeline**: either a small Chart.js plugin or a direct canvas draw,
following the existing custom-plugin precedent (`userBounds` in
`season_category_bars.js`). For each team row: a filled density path with a
constant vertical offset (baseline per row), the user row highlighted, a median
tick. Shared x-axis across rows.

The selector (Overall + 10 category pills, plus the Totals|Points sub-toggle)
reuses the proven pill-toggle + destroy-and-rerender pattern already used by the
category-bars view.

## Scope

### v1 (this effort)

- Retain compact distributions in `run_ros_monte_carlo()` (overall, per-category
  totals, per-category points).
- New `format_distributions_for_display()` and route wiring.
- New Distributions view replacing the Category Bars tab; ridgeline rendering;
  Overall + 10 categories; Totals|Points toggle.
- **Delete** the old Category Bars scatter code (JS, the
  `format_category_bars_for_display` builder, template section, and the
  `category_bars` data plumbing) -- no soft-deprecate, per repo convention.

### Deferred (explicitly out of scope for v1)

- Opening-day / preseason comparison toggle (would require the preseason
  baseline MC freeze to also retain distributions).
- p10-p90 shaded band per row.
- Points-swing annotations (mapping a raw-total range to roto points earned).

## Testing

- Unit tests for the KDE / distribution-builder helper: shared-grid output,
  known input array -> expected curve shape, discrete PMF sums to 1, bandwidth
  floor behavior on near-zero variance.
- A test asserting `run_ros_monte_carlo()` returns the `distributions` key with
  the documented shape, and that it is JSON-serializable (no numpy) so it
  survives the cache round-trip.
- Frontend is covered via the data-contract tests above; the ridgeline draw
  itself is not unit-tested (consistent with existing chart JS).

## Verification (per repo CLAUDE.md)

Before declaring done: `pytest -v` (or the relevant subset), `ruff check .`,
`ruff format --check .`, `vulture` (no new findings), and `mypy` if any touched
file is under `[tool.mypy].files`. Additionally, exercise the refresh path
locally (`run_season_dashboard.py`, `--no-sync` when verifying not-yet-deployed
code) so the new distributions actually populate the cache, before any merge.

## Open questions / assumptions

- Assumes 12 teams / 10 categories (max overall points 120). Team count N is read
  from the data, not hard-coded.
- KDE bandwidth rule and grid resolution (~60 points) are tunable; final values
  settled during implementation against real cached MC output.
