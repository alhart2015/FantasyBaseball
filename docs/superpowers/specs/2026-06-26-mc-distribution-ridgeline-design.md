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
n_iterations (default 1000). Per iteration it ranks all teams via
`score_roto_dict()` and computes, for every team, both its total roto points and
its per-category roto points. What it currently **accumulates** across iterations
is narrower than what it computes:

- **Overall total points, all teams:** already accumulated in `all_totals[name]`
  (per-iteration list) before being collapsed to median/p10/p90. Reusable as-is.
- **Raw category totals, all teams:** present in the `batch` array returned by
  `simulate_remaining_season_batch()`, shape `{team: {category: ndarray(n_iter)}}`,
  which is in scope inside `run_ros_monte_carlo()`. Reusable as-is.
- **Per-category roto points:** accumulated for the **user team only**
  (`user_cat_pts`, guarded by `if name == user_team_name`). For all *other* teams
  the per-category points are computed each iteration and immediately discarded.

After computing aggregates the function discards every per-iteration array; only
scalar summaries survive to the cache (median_pts, p10, p90, first_pct, top3_pct,
and per-category risk for the user team).

**Implication for this work.** The `overall` and `category_totals` distributions
can be built from data already in scope (no accumulation-loop change). The
`category_points` distribution is the one piece that is genuinely new work: the
per-iteration loop must be extended to accumulate `pts[f"{cat}_pts"]` for **every**
team, not just the user. This is a small, contained change to the existing loop,
but it is not merely "stop discarding" -- the non-user per-category arrays do not
exist today. (Transient cost: 12 teams x 10 cats x 1000 iters of point values,
~1-2 MB of floats held during the run -- modest. Only the compact curves below
are cached.)

### What we retain

Before discarding the per-iteration arrays, build compact **density curves** and
return them. We do NOT cache the raw 1000-length arrays (~250k numbers across
all teams/categories -- heavy to serialize and ship to the browser). Instead:

- **Continuous metrics** (overall total points; each category's raw totals):
  a Gaussian kernel density estimate (KDE), sampled at a fixed number of points
  (~60) on a **shared x-grid** per metric. Shared grid is required so ridgeline
  rows are horizontally comparable. Result per metric:
  `{ "x": [...], "teams": { team: {"y": [...], "median": float} } }`, where
  `median` is that team's sample median (computed from the raw samples before
  KDE) and is used for row sorting and the row marker -- the density array alone
  carries no median, so it must be threaded into the payload here.
  - *Why `overall` (total points) is treated as continuous even though it is
    discrete:* a team's total is a sum of 10 per-category point values, so its
    support spans 10..120 in 0.5 steps (~220 possible values) and, over 1000
    iterations, clusters into a smooth-looking spread. KDE-smoothing it produces
    the intended bell curve. Category *points* (next bullet), by contrast, have
    only ~12-23 possible values and are kept as an exact discrete PMF.
  - *Bandwidth is per-team* (see KDE detail): each team's curve uses its own
    Silverman bandwidth (with the metric-relative floor below). This is
    deliberate -- a near-deterministic team *should* render as a narrow peak and
    an uncertain team as a wide spread; that per-team width difference IS the
    uncertainty signal the view exists to show, so we do not force one shared
    bandwidth across teams.
  - *Shared x-grid construction:* for each metric, pool the samples for **all**
    teams; take `lo = min(pooled)`, `hi = max(pooled)`; let `bw_max` be the
    largest per-team bandwidth among the teams for that metric; and use the grid
    `linspace(lo - 3*bw_max, hi + 3*bw_max, 60)`. Every team's KDE is sampled on
    this one shared grid (so rows are horizontally comparable), and padding by
    `3*bw_max` guarantees no team's tails are clipped (each team's own `bw <=
    bw_max`). The grid is shared; the bandwidth is per-team.
  - *Sentinel guard:* the `batch` arrays use a `99.0` sentinel for ERA/WHIP in
    degenerate zero-IP iterations (`simulate_remaining_season_batch`). Over a full
    rest-of-season a real team accumulates IP so this is rare, but the
    distribution builder must drop sentinel/`inf`/`nan` values before computing
    `lo`/`hi` and the KDE so a stray `99.0` cannot create a phantom tail. If a team
    has zero usable samples for a metric, omit its row for that metric.
- **Discrete metric** (category roto-points): the exact probability mass over the
  discrete support of roto points. Let `N` = team count (read from the data). A
  team's per-category points are an expected-value sum `1 + sum P(me > opp)` over
  the other `N-1` teams (`scoring.py`), so the value is bounded `1 <= pts <= N`
  (all losses -> 1; a clean win over everyone -> N). With `team_sds=None` an exact
  tie contributes 0.5, so the support is the half-integer grid `1, 1.5, ..., N`
  (at most `2N-1` distinct values; 23 for N=12). No smoothing -- with so few
  achievable values, smoothing would invent mass between them. Ties occur only on
  integer counting stats (R, HR, ...) and are occasional, not the common case;
  rate stats (AVG/ERA/WHIP) tie essentially never and are effectively integer.
  Build the PMF over the **distinct point values actually observed** across
  iterations (the `1..N` half-integer grid is a safe, possibly sparse, superset).
  Result per category:
  `{ "x": [...point values...], "teams": { team: {"p": [...], "mean": float} } }`
  where each team's `p` sums to 1 and `mean = sum(x*p)` is the expected points
  (used for row sorting and the row marker).

Approximate cache cost: ~5k numbers for continuous curves + ~1.4k for discrete
PMFs ~= under 7k numbers total. JSON-serializable, no numpy in the payload.

### KDE detail

Implement a small Gaussian KDE helper (a few lines: for each grid point, sum
Gaussian kernels centered on the samples; normalize). Bandwidth per team via
Silverman's rule (`bw = 0.9 * min(std, IQR/1.349) * n**(-1/5)`) with a
**metric-relative floor** to avoid a delta-spike when a team's variance is near
zero (e.g. a runaway category). The floor must scale with the metric's units --
an absolute floor good for R (counts in the hundreds) is nonsense for ERA
(~3.50). Define it relative to the metric's pooled spread, e.g.
`bw_floor = max(eps, f * (hi - lo))` with `f` a small fraction (~0.5-1% of the
pooled range, tuned at implementation), where `lo`/`hi` are the pooled
post-sentinel-drop min/max for that metric. The helper lives next to the
simulation code (or a small util module) and is unit-tested independently.

*Why hand-rolled rather than `scipy.stats.gaussian_kde`:* scipy is already a
hard dependency (`pyproject.toml`, imported in `simulation.py`), so availability
is not the issue. The hand-rolled helper is preferred for three concrete reasons:
(1) it lets us impose a metric-relative bandwidth floor (in the metric's data
units), which `gaussian_kde` does not expose cleanly (its `bw_method` scales by
the sample std, so a near-constant team still collapses to a spike); (2) it
raises no
`LinAlgError` on degenerate (zero-variance) input, which `gaussian_kde` does; and
(3) it emits plain Python floats directly, keeping the cached payload numpy-free
(see cache round-trip test). Retaining the real MC shape rather than a single
Gaussian per team matters because close races can be skewed or bimodal, and that
shape is the whole point of the visualization.

### Return shape (new key on the MC result)

`run_ros_monte_carlo()` gains a `"distributions"` key alongside the existing
`team_results` / `category_risk`:

```python
"distributions": {
    "overall": {                 # continuous: total roto points
        "x": [float, ...],       # shared grid (~60 points)
        # per team: KDE density on the grid + sample median (for sort + marker)
        "teams": {team_name: {"y": [float, ...], "median": float}}
    },
    "category_totals": {         # continuous: raw stat totals
        "R": {"x": [...], "teams": {team: {"y": [...], "median": float}}},
        ...  # one per category
    },
    "category_points": {         # discrete: roto points (half-integer grid, ties split)
        # per team: PMF over x + mean expected points (for sort + marker)
        "R": {"x": [1, 1.5, ..., N], "teams": {team: {"p": [...], "mean": float}}},
        ...                      # each p-list sums to 1
    },
    "user_team": str,            # team name; consumed by the formatter, see below
}
```

The raw simulation result keys teams by name (that is the team identifier at this
layer). `format_distributions_for_display()` resolves the highlight
**server-side**: it reads `user_team`, marks each team row with an `is_user`
boolean, and **drops** the raw `user_team` string -- it is not forwarded to the
template. The frontend keys off `is_user`, never off a name comparison.

Every per-team entry carries its own sort key (`median` for continuous, `mean`
for discrete), so the formatter can sort rows into standings order without
needing `team_results` or any separate median source.

## Caching and data flow

Follows existing patterns exactly:

1. `run_ros_monte_carlo()` returns the new `distributions` key
   (`src/fantasy_baseball/simulation.py`).
2. `src/fantasy_baseball/web/refresh_pipeline.py` folds it into the existing
   `cache:monte_carlo` payload (currently writes
   `{base, baseline_meta, rest_of_season}`); since
   `rest_of_season` **is** the `run_ros_monte_carlo()` return value, the new
   `distributions` key rides inside `rest_of_season` automatically -- no new
   cache key and no pipeline change.
3. A new `format_distributions_for_display()` in
   `src/fantasy_baseball/web/season_data.py` reshapes the distributions into a
   template-ready structure: sort each metric's team rows by the per-team sort key
   already in the payload (`median` for continuous, `mean` for discrete -- no
   separate median source needed), mark each row `is_user` (dropping the raw
   `user_team` string), attach labels. Mirrors the existing `format_*_for_display`
   functions.
4. **Route plumbing is required** (this does not happen for free). The existing
   path calls `format_monte_carlo_for_display(raw_mc["rest_of_season"], ...)`
   inside an `if raw_mc.get("rest_of_season"):` guard
   (`src/fantasy_baseball/web/season_routes.py`), because `rest_of_season` is
   `None` in early season (no ROS data). That formatter extracts only
   `team_results` / `category_risk` and ignores `distributions`. **Inside the same
   guard** (never read `.get(...)` on a possibly-`None` `rest_of_season`), the
   route must additionally read `raw_mc["rest_of_season"].get("distributions")`,
   pass it through the new `format_distributions_for_display()`, and hand the
   result to the template as `distributions`. Guard for the inner absence too
   (`rest_of_season` present but `distributions` key missing -> view renders an
   empty/"no data" state), since the MC cache is written with `required=False`
   and an older cache blob predating this change will not contain the key.
5. `standings.html` embeds it as `<script type="application/json"
   id="distributions-data">{{ distributions | tojson }}</script>` -- the same
   embedded-JSON pattern the category bars use today.

## Rendering

New static file `src/fantasy_baseball/web/static/season_distributions.js`.

Chart.js 4.4.4 is already loaded but has no native ridgeline. Render a
**custom ridgeline**: either a small Chart.js plugin or a direct canvas draw,
following the existing custom-plugin precedent (`userBounds` in
`season_category_bars.js`). For each team row: a filled density path with a
constant vertical offset (baseline per row), the user row highlighted, and a
central-tendency tick. The tick position comes straight from the payload -- the
per-team `median` for continuous metrics, the per-team `mean` for the discrete
Points metric -- so the renderer never has to integrate the grid or guess a
between-bins median. For the discrete Points metric the row is drawn as
stems/bars at the support values (not a filled curve), with the tick at `mean`.
Shared x-axis across rows.

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
- **Delete** the old Category Bars scatter code -- no soft-deprecate, per repo
  convention. The deletion is exhaustive; grep `category_bars` / `catbars` /
  `category-bars` and remove every site. Known inventory (verify with grep, do
  not trust this list as complete):
  - `season_data.py`: `format_category_bars_for_display`, its helper
    `_category_bars_one_flavor`, **and** the now-orphaned `_category_odds` helper
    (called only by `_category_bars_one_flavor`) plus its now-unused
    `category_finish_odds` import. (Keep the standalone `category_odds` module and
    `tests/test_category_odds.py` -- they are independent of the category-bars
    view; only the `season_data.py`-local `_category_odds` goes.)
  - `season_routes.py`: the `format_category_bars_for_display` import, its call,
    and the `category_bars=` template kwarg on the standings render.
  - `templates/season/standings.html`: the `#view-categorybars` block, the
    nav/toggle button, the `toggleTopView` branch for it, the
    `id="category-bars-data"` embedded-JSON node, the
    `<script src="...season_category_bars.js">` include, the now-dead
    `chartjs-chart-error-bars` CDN `<script>` (used only by the deleted JS; the
    new ridgeline draws on raw canvas and does not need it), and the
    category-bars-specific CSS rules (`.catbars-wrapper`, `#catbars-cat-toggle`,
    `.catbars-odds`, ...).
  - `static/season_category_bars.js`: delete the file.
  - `tests/`: the category-bars tests in `tests/test_web/test_season_data.py`
    (the block importing/exercising `format_category_bars_for_display`) and the
    category-bars assertions in `tests/test_web/test_season_routes.py`. A dangling
    import of the deleted function fails the whole test module, so these must be
    removed or rewritten as part of the deletion, not left for later.

### Deferred (explicitly out of scope for v1)

- Opening-day / preseason comparison toggle (would require the preseason baseline
  MC freeze to also retain distributions). **Note this is a regression:** the
  Category Bars view being replaced has a Current/Preseason sub-toggle, so v1
  drops the ability to compare a category's distribution against preseason
  projections in this view. The preseason ERoto comparison still exists under the
  Projected tab; only the category-distribution-vs-preseason view goes away until
  this toggle is added back. Flagged for user awareness.
- p10-p90 shaded band per row.
- Points-swing annotations (mapping a raw-total range to roto points earned).

## Testing

- Unit tests for the KDE / distribution-builder helper:
  - shared-grid construction: grid is `linspace(lo - 3*bw_max, hi + 3*bw_max, 60)`
    over all teams' pooled samples (`bw_max` = largest per-team bandwidth); every
    team sampled on the identical grid.
  - each team's density integrates to ~1 over the grid (trapezoid), for a known
    input -- valid because padding by `3*bw_max` keeps every team's tails on-grid.
  - bandwidth floor is metric-relative: a near-constant input does not collapse to
    a single-bin spike, and the floor scales with the metric (a tight-spread rate
    metric like ERA and a wide counting metric like R both produce sane,
    non-spike, non-oversmoothed curves -- covers the rate-stat case).
  - sentinel guard: `99.0`/`inf`/`nan` samples are dropped before `lo`/`hi`/KDE;
    a team with no usable samples is omitted for that metric.
  - per-team summary present: continuous entries carry `median`, discrete entries
    carry `mean = sum(x*p)`; values match the raw samples.
  - discrete PMF: sums to 1 per team; half-integer support handled (a constructed
    tie produces a 0.5-step point value in the support); support bounded `1..N`.
- A test asserting `run_ros_monte_carlo()` returns the `distributions` key with
  the documented shape, that `category_points` is populated for **all** teams
  (not just the user -- guards the C2 accumulation change), and that the whole
  payload is JSON-serializable (no numpy types) so it survives the cache
  round-trip.
- A route/formatter test: `format_distributions_for_display()` sorts rows by the
  per-team sort key and marks exactly one row `is_user` (and does not forward the
  raw `user_team` string); the standings route reads
  `rest_of_season["distributions"]` and embeds it; a cache blob with
  `rest_of_season=None` (early season) does not crash; and a blob whose
  `rest_of_season` **lacks** `distributions` (older payload) renders the
  empty-state without error.
- Frontend ridgeline draw is not unit-tested (consistent with existing chart JS);
  it is covered by the data-contract tests above plus the local refresh
  verification.

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
