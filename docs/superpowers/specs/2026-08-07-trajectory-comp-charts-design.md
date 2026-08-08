# Trajectory: paced current season, and per-comp career charts

Issue: [#346](https://github.com/alhart2015/FantasyBaseball/issues/346)
Status: approved design, ready to plan

## Problem

The player trajectory page (`/trajectory?view=player`) draws three things: a realized
career line, a dashed projection with a p10-p90 band, and the closest realized comp
paths as thin grey lines overlaid on the projection's ages. Four things are wrong with
it.

**1. There is a hole at the current season.** The career line is built from
`complete = live[~live["partial_season"]]` in `scripts/push_trajectory_board.py`, which
strips the in-progress season by construction. So the line stops at the last completed
season and the projection starts a year later, leaving a visible gap at exactly the age
the fit is anchored on -- and hiding the freshest evidence the model has. The current
page prints a sentence explaining that the gap is intentional. It should show the number
instead.

**2. It is not documented anywhere findable whether the current-season input is
prorated actuals or a projection blend.** Answering it required reading four modules.

**3. The pages carry prose explaining how to read the charts.** The reader does not need
it and it costs vertical space above the data.

**4. A comp is drawn as a forward path and nothing else.** If Xander Bogaerts's 2018 is
the closest realized path to CJ Abrams's projection, the stacked overlay shows five grey
points across Abrams's projected ages. It says nothing about Bogaerts: not what he did
before 2018, not what kind of player he was at that age, not where the match sits in his
arc. The stacking also makes the comps read as a cloud around the projection, which is
the exact over-confidence `comp_paths.py`'s module docstring warns about.

## Goals

- Draw the paced current season as the terminal point of the career line, visibly marked
  as a pace rather than a finished year.
- Record, in the issue tracker and in code comments, that the current-season input is
  straight-line-prorated realized stats.
- Remove reading-instruction prose from all three trajectory templates.
- Give every rendered comp its own small chart showing his **whole career**, with a
  vertical line at the age where he matches the subject, and the subject's own arc drawn
  faintly underneath.

## Non-goals

- **Changing the comp selection rule.** `closest_paths` still matches on exact age and
  ranks by RMSE over the projected horizons. This spec changes what is *drawn* about the
  selected comps, never which comps are selected.
- **Changing any fitted number.** No estimator, floor, era normalization, or band
  changes. The paced value already exists and already drives the fit; this only draws it.
- **Removing factual disclosure.** Vintage, staleness, excluded-player counts,
  chart-vintage-mismatch notes and roster-join banners all stay. Only interpretation
  prose goes.
- **Blending ROS projections into the trajectory model.** Explicitly out of scope; see
  Requirement 2.
- **Rewriting the stacked comp overlay on the main chart.** It stays as-is. The small
  multiples are additive.

## Requirement 1: the paced current-season point

### What exists

`board_inputs` (`trajectory/board.py`) computes `paced` via `_paced` ->
`prorate_partial(sgp, fraction)`, where `fraction = season_elapsed_fraction(hitter_panel,
season)`. That value is stored in the board payload as `now` and `build_player_view`
already reads it as `sp.now`. **No new stored data is required.**

### Design

`PlayerView` gains one field:

```python
#: The PACED current season as [age, value], floor-netted like `history`. `None` only
#: when the player was not found. Kept OUT of `history` deliberately: `history` means
#: "realized complete seasons", several template branches key on its emptiness to
#: report a missing/mismatched chart blob, and folding a board-sourced point into a
#: chart-blob-sourced list would make `not board.history` stop meaning what it says.
paced: list[float] | None
```

Value is `[sp.age, sp.now - floor]`, using the same `floor = sp.offset(scale)` every
other series is netted against.

**It is populated even when the chart blob is missing or vintage-mismatched.** `now`
comes from the board, which is by definition the board on screen, so it is never stale
relative to the projection. A player with no chart extras therefore renders a one-point
career anchor plus the projection, which is strictly more than today's projection alone.

### Rendering

The chart's career dataset is `history` concatenated with `paced`, drawn as one unbroken
line. The paced point is styled per-point via Chart.js array-valued point options:
`pointStyle: "circle"`, larger `pointRadius`, transparent `pointBackgroundColor` and a
2px border -- an open marker against the filled markers of the realized seasons. Its
tooltip label reads `<season> pace`.

`history` is sorted ascending by age server-side and `paced.age` is `sp.age`, which is
strictly greater than every realized season's age (the base season is in progress, so it
is not in `complete`). Concatenation therefore preserves ordering without a re-sort. A
test pins this: the last realized age is less than the paced age for a fixture player.

The numbers table gains a row for the paced season, marked `pace` in the third column
where realized rows read `actual`.

### Units

`now` is era-normalized paced SGP; `history` values are era-normalized realized SGP.
Both come off the same panel through the same `score()` path, and both have the same
floor subtracted. They are on one axis legitimately.

## Requirement 2: prorated actuals, not blended ROS

**Finding, no code change:** the trajectory model's current-season input is straight-line
prorated realized stats. `prorate_partial` divides realized SGP by the elapsed fraction;
the fraction is the busiest hitter's games over 162 (`season_elapsed_fraction`, hitter
panel only, guarded). No FanGraphs ROS projection is read anywhere in
`src/fantasy_baseball/trajectory/` or `scripts/push_trajectory_board.py`.

Deliverables:

- A test asserting no module under `src/fantasy_baseball/trajectory/` and not
  `scripts/push_trajectory_board.py` imports from `fantasy_baseball.data.ros` or any ROS
  projection loader. This is a boundary test: it is what stops a future "just blend in
  ROS for the current year" edit from silently changing what the chart's anchor means.
- The finding recorded in issue #346 (done) and referenced from the `paced` field's
  docstring so a reader of the chart code can find it.

## Requirement 3: remove interpretation prose

Delete exactly these:

| File | Text |
|---|---|
| `trajectory_player.html` | "Solid is what happened; dashed is projected, with the shaded band its p10-p90 range." |
| `trajectory_player.html` | "The career line above stops before {{ base_season }} on purpose..." (also factually obsolete after Requirement 1) |
| `trajectory_player.html` | The whole "Closest realized paths" honesty paragraph (`These are the closest realized paths ... none is recent.`) including its VAR-floor sentence |
| `trajectory.html` | The VAR-vs-SGP explainer (`Showing VAR -- value above the position-aware waiver floor...` / the raw-SGP branch) |
| `trajectory_teams.html` | "Ranks are LEAGUE ranks -- a block's top row reads its position among all N scored players, not its position on that roster." |

Keep:

- every `.trajectory-vintage` paragraph (build timestamp, panel filenames, elapsed season)
- the excluded-player counts on `trajectory.html`
- `mismatch_note` and the "None stored / predates the feature" notes
- the "your players are not highlighted" and "none of these blocks is yours" banners
- the extrapolation `(!)` warning on the player page
- `comp_paths.py`'s module docstring -- that is for whoever edits the code

The VAR-floor sentence removed from the comps paragraph asserts something a reader still
needs: on the VAR scale, comp values are netted against **the subject's** slot floor, not
each comp's own. That fact survives in `PlayerView.axis_label`, which the y-axis title
and the table header both render, and in `PlayerView.floor`'s docstring. Nothing is lost
except the sentence.

## Requirement 4: per-comp career charts

### Storage: a deduped `careers` map

`cache:trajectory_chart_data` gains a top-level `careers` key:

```json
{
  "generated_at": "...",
  "players": {"12345:hitter": {"history": [...], "comps": [...]}},
  "careers": {"67890:hitter": [[21, 3.4], [22, 7.1], ...]}
}
```

Keyed by `chart_key(mlbam_id, pool)` -- the same key `players` uses, and for the same
reason: a two-way player has one career per pool and a bare id would hand the hitter's
card the pitcher's arc.

**Deduped at the top level rather than nested under each comp.** Comps repeat heavily
across the board's 1,169 players. Measured against the live 2000-2026 panels
(2026-08-07): the complete panels hold 10,701 distinct (id, pool) careers at ~4.4 seasons
each, which is the hard ceiling on this map -- roughly 660 KB, and the realized comp
universe is a strict subset of it. The nested alternative writes up to
1,169 x 10 x 4.4 ~= 51,000 career entries, about 5x more, for the same information.

Only entries actually referenced by some stored comp are written.

### Comp entries gain an id

`player_comps` currently emits `{name, season, rmse, path}`. It gains `id`
(`c.mlbam_id`), which is the join key into `careers`. Without it the view has only a
display name, and joining a chart to a name is the exact defect class CLAUDE.md and #284
name.

### Push side

In `push_trajectory_board.build_payloads`, inside the existing per-pool loop, after the
`for player in produced` loop:

- collect `wanted = {c["id"] for comps in this pool's blocks for c in comps}`
- for each id in `wanted`, look it up in the **already-built `by_id`** map (collapsed,
  complete, era-normalized, per-pool) and write
  `careers[chart_key(id, kind)] = [[int(age), round(sgp, 4)], ...]` sorted by age

`by_id` is the same source the subject's own `history` comes from, so a comp card and the
subject overlay on it are on one scale by construction. An id absent from `by_id` writes
no entry (it cannot happen -- comps are drawn from the same `prepared`, built from the
same `complete` frame -- so its absence is a defect, and an absent card is the honest
rendering of it rather than a fabricated one).

`to_chart_payload` takes `careers` as a keyword argument and passes it through.

### View side

`_chart_extras` currently returns `(extras, mismatch)`. It returns `(extras, careers,
mismatch)`; `careers` is `{}` on every refusal path and whenever the key is absent or is
not a mapping (an old blob predating this feature).

`PlayerView` gains:

```python
#: One entry per rendered comp: his whole career, plus what to mark on it. Parallel to
#: `comps` and in the same order. `career` is empty when the blob carries no entry for
#: him -- an old blob, or a push that wrote comps without careers.
comp_careers: list[dict]
```

Each entry is:

```python
{
  "name": str, "season": int, "rmse": float,
  "match_age": int,                       # == the subject's age; see below
  "career": [[age, value], ...],          # floor-netted, ascending by age
}
```

`match_age` is `sp.age` for every comp, because `closest_paths` selects on
`prepared.age == float(age)` -- an EXACT age match. So the line sits at the same x on
every card, which is what makes the grid comparable. A test asserts every
`comp_careers[i]["match_age"] == view.age`, so if the matcher ever moves to a tolerance
window this stops silently drawing the line in the wrong place.

`career` values have the **subject's** floor subtracted on the VAR scale, identical to
the existing `comps[].path` treatment and for the identical reason: the card asks what
this arc would be worth in the subject's slot.

### Rendering

Below the existing comps table, a `.comp-grid` of `.comp-card`s, one per rendered comp
(so `n` controls both). Each card:

- header: comp name, match season, RMSE
- a ~180px canvas containing:
  - the comp's full career, solid, prominent
  - the subject's realized history + paced point + projection means, faint, underneath
  - a dashed vertical line at `match_age`
- when `career` is empty: the header and a one-line "no stored career -- re-run
  `push_trajectory_board.py`" instead of a canvas

The vertical line is drawn by a small inline Chart.js plugin (`afterDatasetsDraw`, reads
`chart.scales.x.getPixelForValue(matchAge)`, strokes across `chart.chartArea`), registered
per-chart in the card's own `plugins` array. Deliberately **not** the `chartjs-plugin-annotation`
CDN bundle: a second external script is a second thing that can fail to load, and
`season_trends.js`'s `ensureChartJs` loader would have to grow a dependency-ordering
concept to serve it. The plugin is about ten lines.

The x-axes are **not** shared across cards. Each card autoscales to its own comp's career
span. Forcing a common domain would compress a 20-season career into the same width as a
6-season one and shrink the region around the match line, which is the part being read.
The match line is the shared reference, not the axis.

The JS lives in `trajectory_chart.js` alongside the main chart rather than in a new file:
both read the same `#trajectory-chart-data` island, and the subject overlay is the same
series the main chart builds. A second file would re-parse the island and re-derive that
series.

## Failure modes

| Condition | Behavior |
|---|---|
| Blob has no `careers` key (predates this change) | Main chart, paced point, comps table all render. Cards show the "no stored career" line. No error. |
| `careers` is present but not a mapping | Treated as absent, same as above. Mirrors `_chart_extras`'s existing `players`-not-a-mapping refusal. |
| Chart vintage mismatch | `history`, `comps`, `comp_careers` all empty (existing behavior extended). `paced` still renders -- it is board data. Existing `mismatch_note` prints. |
| A comp id missing from `careers` | That one card renders headerless-canvas-free with the note; the others are unaffected. |
| A comp entry with no `id` (blob written by an older push) | `comp_careers` entry gets an empty `career`, same as a missing id. No `KeyError`. |
| Chart.js fails to load | Existing behavior: the table below is the fallback. The card grid renders headers with empty canvases; acceptable, and no worse than the main chart's current degradation. |
| Player not found / ambiguous name | `paced` is `None`, `comp_careers` is `[]`; the candidate list or not-found message renders as today. |

## Testing

New tests, by module:

- `tests/test_scripts/test_push_trajectory_board.py`
  - `player_comps` emits `id` on every comp
  - the built chart payload's `careers` contains an entry for every id referenced by any
    stored comp, and no entries beyond that
  - a career entry is ascending by age and matches `by_id`'s seasons for that player
- `tests/test_trajectory/test_sweep.py`
  - `to_chart_payload` round-trips `careers`
- `tests/test_web/test_trajectory_view.py`
  - `paced` is `[age, now - floor]` on the VAR scale and `[age, now]` on SGP
  - `paced` survives a chart vintage mismatch while `history`/`comps` do not
  - `paced` is `None` for a not-found player
  - `history`'s last age is strictly less than `paced`'s age
  - `comp_careers` is parallel to `comps` and same length under every `n`
  - every `match_age` equals `view.age`
  - `career` values are floor-netted on VAR
  - a blob with no `careers` key yields empty `career`s and no exception
  - a blob whose `careers` is a list (not a mapping) is refused the same way
- `tests/test_web/test_season_routes.py`
  - the player page renders with the new island fields present and 200s with a
    careers-less blob
- Boundary test (new, `tests/test_trajectory/`): no trajectory module imports a ROS
  projection loader (Requirement 2)

Prose removal is asserted negatively where a test already renders the page: the removed
sentences must not appear in the rendered HTML, and the kept vintage sentence must.

## Phasing

Three independently shippable steps, in dependency order:

1. **Paced point + prose removal.** No blob schema change, no re-push needed. Ships
   value immediately against the currently-deployed blob.
2. **Blob schema: comp `id` + `careers` map.** Push-side and view-side plumbing with no
   UI. Reader ships before the writer runs, per the standing rule that the reader must be
   deployed before a new blob format is pushed
   ([[feedback_deploy_before_pushing_blob_formats]]).
3. **The card grid.** JS + CSS + template, reading what step 2 stores.

Step 2's reader tolerates a blob without `careers`, and step 3's grid tolerates empty
careers, so any two of the three can be deployed in either order without a broken page.
