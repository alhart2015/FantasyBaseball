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
already reads it as `sp.now`. **The value itself needs no new storage** -- the only
addition is the `base_season_partial` flag described below, which labels it.

### Design

`PlayerView` gains one field:

```python
#: The PACED current season as [age, value], floor-netted like `history`. `None` when
#: the player was not found, or when the base season is ALREADY a realized row in
#: `history` -- see "When there is no paced point" below. Kept OUT of `history`
#: deliberately: `history` means "realized complete seasons", several template branches
#: key on its emptiness to report a missing/mismatched chart blob, and folding a
#: board-sourced point into a chart-blob-sourced list would make `not board.history`
#: stop meaning what it says.
paced: list[float] | None
```

Value is `[sp.age, sp.now - floor]`, using the same `floor = sp.offset(scale)` every
other series is netted against.

**It is populated even when the chart blob is missing or vintage-mismatched.** `now`
comes from the board, which is by definition the board on screen, so it is never stale
relative to the projection. A player with no chart extras therefore renders a one-point
career anchor plus the projection, which is strictly more than today's projection alone.

### When there is no paced point

The base season is not permanently in-progress, and the transition is silent.
`_live_seasons` (`scripts/build_pt_panel.py`) flags a season partial iff
`year >= date.today().year`. So a panel **rebuilt in January 2027** un-flags 2026: it
enters `complete`, and therefore enters `history` -- while `season =
calendar["season"].max()` is still 2026, so `sp.age` is the 2026 age and `sp.now` is the
2026 line (unprorated, since `needs_pacing` also goes False). Naively appending would
draw two points at one age, one of them labelled "pace", on a finished season.

Two mechanisms, both specified, because they fail in different directions:

1. **Suppression rule (authoritative, works on every blob).** `build_player_view` sets
   `paced = None` when `sp.age` already appears among `history`'s ages. That is exact:
   a collision means the base season is realized and already drawn, and no collision
   means it is not. It needs no schema change, so it protects blobs written before this
   spec as well as after.
2. **`base_season_partial` (new board-payload field, drives the LABEL).** The push
   writes `bool(calendar.loc[calendar["season"] == season, "partial_season"].any())` into
   `to_payload`'s meta. The suppression rule cannot distinguish "in progress" from
   "finished" when `history` is absent entirely (a mismatched or missing chart blob), and
   in that case the point is still drawn -- so the label must not assert a pace that may
   not be one. Absent on an old blob, it defaults to `True`, which is what every
   currently-deployed blob means.

`PlayerView` therefore also carries a server-built `paced_label: str` -- `"2026 pace"`
when `base_season_partial`, `"2026"` otherwise. Built server-side rather than
interpolated in JS, following the `axis_label` precedent: the finished string ships and
the two surfaces cannot disagree.

### Rendering

The chart's career dataset is `history` concatenated with `paced` (when present), drawn
as one unbroken line. The paced point is styled per-point via Chart.js array-valued point
options: `pointStyle: "circle"`, larger `pointRadius`, transparent `pointBackgroundColor`
and a 2px border -- an open marker against the filled markers of the realized seasons.
Its tooltip label is `paced_label`.

`history` is sorted ascending by age server-side, and given the suppression rule
`paced`'s age is strictly greater than every realized age whenever `paced` is not `None`.
Concatenation therefore preserves ordering without a re-sort. Two tests, one per branch:
a fixture whose base season is absent from `history` gets a paced point ordered last, and
a fixture whose base season is present in `history` gets `paced is None` and an unchanged
career line.

The numbers table gains a row for the paced season when there is one, marked `pace` in
the third column where realized rows read `actual`.

### Chart data island

`#trajectory-chart-data` currently carries `name`, `axis_label`, `history`, `projection`,
`comps`. It gains exactly three keys: `paced` (`[age, value]` or `null`), `paced_label`,
and `comp_careers`. No other field is added -- in particular `base_season` is not, since
`paced_label` is the only thing the JS needed it for.

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

- A boundary test. It scans every `.py` under `src/fantasy_baseball/trajectory/` plus
  `scripts/push_trajectory_board.py` and asserts none of them references any of these
  by name:

  | Surface | Why it is forbidden here |
  |---|---|
  | `fantasy_baseball.data.ros_pipeline` | builds the blended ROS projection set |
  | `fantasy_baseball.data.ros_export_ingest` | ingests the FanGraphs ROS export |
  | `fantasy_baseball.data.projections` | the projection-blend loader generally |
  | `CacheKey.ROS_PROJECTIONS` | the stored ROS blob |

  Enumerated by name rather than by a "ROS loader" concept, so the test asserts
  something. This is what stops a future "just blend in ROS for the current year" edit
  from silently changing what the chart's anchor means.

  Verified green against the current tree (2026-08-07): the only `fantasy_baseball.data`
  imports in that scope are `data.rosters`, `data.kv_store` and `data.cache_keys`, and
  `ROS_PROJECTIONS` appears in none of them. So the test starts passing and is a
  regression guard, not a task.
- The finding recorded in issue #346 (done) and referenced from the `paced` field's
  docstring so a reader of the chart code can find it.

## Requirement 3: remove interpretation prose

The removals are **clauses inside paragraphs that also hold content worth keeping**, so
each affected paragraph is classified in full below. Delete only what the DELETE column
names; everything in KEEP stays exactly as written.

| File | Paragraph | DELETE | KEEP |
|---|---|---|---|
| `trajectory_player.html` | the header `<p class="muted">` under the search form | "Solid is what happened; dashed is projected, with the shaded band its p10-p90 range." | "{{ board.name }} -- age {{ board.age }}, {{ board.slot }}." and the `board.extrapolated` `(!)` warning. The paragraph survives with those two. |
| `trajectory_player.html` | the `.trajectory-vintage` paragraph | the `{% if board.history %}` sentence, "The career line above stops before {{ base_season }} on purpose..." -- also factually obsolete after Requirement 1 | `ctl.vintage_note(board)`. The paragraph survives. |
| `trajectory_player.html` | the `<p class="muted">` under the "Closest realized paths" heading | the ENTIRE paragraph, `These are the closest realized paths ... none is recent.`, including the trailing `{% if board.scale == 'var' %}` floor sentence | the `<h3>Closest realized paths</h3>` heading, its count span, and **the comps table below it** -- none of which is prose |
| `trajectory.html` | the intro `<p class="muted">` | the whole `{% if board.scale == 'var' %}...{% else %}...{% endif %}` VAR-vs-SGP branch | "Of everyone you could hold, who is worth the most over {{ board.span }}." The paragraph survives with that one sentence. |
| `trajectory_teams.html` | the intro `<p class="muted">` | "Ranks are LEAGUE ranks -- a block's top row reads its position among all {{ board.ranked }} scored players, not its position on that roster." | "The best {{ board.per_team }} on every roster over {{ board.span }}, strongest team first." The paragraph survives with that one sentence. |

No paragraph is deleted outright and no table, heading, or control is touched.

Keep, elsewhere on those pages:

- every `.trajectory-vintage` paragraph (build timestamp, panel filenames, elapsed
  season), minus the one obsolete sentence named above
- the excluded-player counts on `trajectory.html`
- `mismatch_note` and the "None stored / predates the feature" notes
- the "your players are not highlighted" and "none of these blocks is yours" banners
- the `(!)`/`(!!)` extrapolation sentence on `trajectory_teams.html`'s vintage paragraph
- `comp_paths.py`'s module docstring -- that is for whoever edits the code, not for the
  reader of the page. Its selected-on-the-outcome warning is unchanged; one paragraph is
  amended, for the reason given below.

The VAR-floor sentence removed from the comps paragraph asserts something a reader still
needs: on the VAR scale, comp values are netted against **the subject's** slot floor, not
each comp's own. That fact survives in `PlayerView.axis_label`, which the y-axis title
and the table header both render, and in `PlayerView.floor`'s docstring. Nothing is lost
except the sentence.

### The band, the cards, and `comp_paths.py`'s docstring

`comp_paths.py`'s module docstring says a consumer that draws comp paths "without the
p10-p90 band beside them is making the forecast look more certain than it is." The card
grid is exactly such a consumer: ten comp arcs, no band. Left unrecorded, the grid reads
as code violating a standing in-repo instruction, and the next person to touch it
"fixes" it.

The decision, explicitly: **the cards are a shape-inspection surface, not a confidence
display.** They answer "who was this player and where does the match sit in his arc",
which is a question about the comp, not about the forecast's spread. The band remains on
the main chart directly above them, on the same page, at full size -- so the honest
uncertainty is not removed from the reader's view, only from each thumbnail.

`comp_paths.py`'s docstring is amended in this change to state the rule as it now
stands: the band must accompany any surface that presents comps *as the forecast's
range*, which the main chart does and the cards do not. This is the one deliberate
edit to that docstring; the selected-on-the-outcome warning it opens with is unchanged
and remains the reason the cards are secondary to the band rather than a replacement
for it.

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

`careers: dict[str, list]` is declared **outside** the `for kind in ("hitter",
"pitcher")` loop, beside the existing `extras` dict and for the same reason: it holds
both pools, and the key already carries the pool.

Inside the per-pool loop:

- during the existing `for player in produced` loop, accumulate a per-pool
  `wanted: set[int]` from each returned comp block (`c["id"]` for every comp; a block
  that came back `None` contributes nothing)
- after that loop, for each id in `wanted`, look it up in the **already-built `by_id`**
  map (collapsed, complete, era-normalized, and scoped to THIS pool) and write
  `careers[chart_key(id, kind)] = [[int(age), round(sgp, 4)], ...]` sorted ascending by
  age

`wanted` is per-pool and `by_id` is per-pool, so a two-way comp gets one entry per pool
under distinct keys and neither overwrites the other.

`by_id` is the same source the subject's own `history` comes from, so a comp card and the
subject overlay on it are on one scale by construction. An id absent from `by_id` writes
no entry (it cannot happen -- comps are drawn from the same `prepared`, built from the
same `complete` frame -- so its absence is a defect, and an absent card is the honest
rendering of it rather than a fabricated one).

`to_chart_payload` takes `careers` as a keyword argument and passes it through.

### View side

`_chart_extras` currently returns `(extras, mismatch)`. It returns `(extras, careers,
mismatch)`; `careers` is `{}` on every refusal path and whenever the key is absent or is
not a mapping (an old blob predating this feature). It is module-private with one caller
(`build_player_view`), but existing tests call it directly and must be updated to unpack
three values -- a mechanical change, listed here so the planner does not discover it as
a surprise failure.

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

`name`, `season` and `rmse` are re-carried from the parallel `comps` entry rather than
looked up by index in JS. One list, one card, no zip: a card that had to index into a
second array to title itself is one off-by-one away from putting Bogaerts's name over
Simmons's arc, and the duplication costs three short fields per rendered comp.

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
(so `n` controls both). **When `comps` is empty the entire grid section renders nothing**
-- no heading, no empty grid -- leaving the existing "None stored / predates the feature"
note as the only output. That is the short-observable-path case `player_comps` returns
`None` for and the push already counts and prints.

Each card:

- header: comp name, match season, RMSE
- a ~180px canvas containing:
  - the comp's full career, solid, prominent
  - the subject's realized history + paced point + projection means, faint, underneath
  - a dashed vertical line at `match_age`
- when that one comp's `career` is empty but others have one: the card keeps its header
  and shows a one-line "no stored career -- re-run `push_trajectory_board.py`" where the
  canvas would be
- when **every** rendered comp's `career` is empty -- the blob predates this feature, or
  a push wrote comps without careers -- the grid is replaced entirely by that one
  sentence. Ten identical notices in a ten-cell grid is noise about a single cause.

The vertical line is drawn by a small inline Chart.js plugin (`afterDatasetsDraw`, reads
`chart.scales.x.getPixelForValue(matchAge)`, strokes across `chart.chartArea`), registered
per-chart in the card's own `plugins` array. Deliberately **not** the `chartjs-plugin-annotation`
CDN bundle: a second external script is a second thing that can fail to load, and
`season_trends.js`'s `ensureChartJs` loader would have to grow a dependency-ordering
concept to serve it. The plugin is about ten lines.

**X-domain, per card: the union of the comp's career ages and the subject's drawn ages**
(history + paced + projection), which is Chart.js's own autoscale over the datasets the
card holds -- so it is the default, stated here because the alternative is a silent clip.
Clipping to the comp's career alone would cut off the subject's projection whenever the
comp retired young, removing the half of the card being compared. No domain is shared
across cards: forcing one would compress a 20-season career into the same width as a
6-season one and shrink the region around the match line, which is the part being read.
The match line is the shared reference, not the axis.

Card chrome uses the existing theme variables -- `var(--surface)`, `var(--line)`,
`var(--ink)`, `var(--text-secondary)` -- so the grid works in both themes without a
second palette.

The JS lives in `trajectory_chart.js` alongside the main chart rather than in a new file:
both read the same `#trajectory-chart-data` island, and the subject overlay is the same
series the main chart builds. A second file would re-parse the island and re-derive that
series.

## Failure modes

| Condition | Behavior |
|---|---|
| Blob has no `careers` key (predates this change) | Main chart, paced point, comps table all render. Every `career` is empty, so the grid collapses to the single "no stored career" sentence. No error. |
| `careers` is present but not a mapping | Treated as absent, same as above. Mirrors `_chart_extras`'s existing `players`-not-a-mapping refusal. |
| Chart vintage mismatch | `history`, `comps`, `comp_careers` all empty (existing behavior extended). `paced` still renders -- it is board data. Existing `mismatch_note` prints. |
| One comp's id missing from `careers`, others present | That card keeps its header and shows the note in place of its canvas; the other cards are unaffected. |
| A comp entry with no `id` (blob written by an older push) | `comp_careers` entry gets an empty `career`, same as a missing id. No `KeyError`. |
| Chart.js fails to load | Existing behavior: the table below is the fallback. The card grid renders headers with empty canvases; acceptable, and no worse than the main chart's current degradation. |
| Player not found / ambiguous name | `paced` is `None`, `comp_careers` is `[]`; the candidate list or not-found message renders as today. |
| `comps` is empty (short observable path) | No grid section at all. The existing "None stored" note is the only output. |
| Base season already realized in `history` (panel rebuilt after the season ended) | `paced is None`; the career line simply ends at that season and the projection starts the next year. No open marker, no duplicate point. |
| `base_season_partial` absent (old board blob) | Defaults to `True`, which is what every currently-deployed blob means. Affects only `paced_label`; suppression is decided by the `history` collision, which needs nothing stored. |

## Testing

New tests, by module:

- `tests/test_scripts/test_push_trajectory_board.py`
  - `player_comps` emits `id` on every comp
  - the built chart payload's `careers` contains an entry at `chart_key(id, pool)` for
    every `(id, pool)` referenced by a stored comp in that pool, and no entries beyond
    that set
  - a career entry is ascending by age and matches `by_id`'s seasons for that player
- `tests/test_trajectory/test_sweep.py`
  - `to_chart_payload` round-trips `careers`
- `tests/test_web/test_trajectory_view.py`
  - `paced` is `[age, now - floor]` on the VAR scale and `[age, now]` on SGP
  - `paced` survives a chart vintage mismatch while `history`/`comps` do not
  - `paced` is `None` for a not-found player
  - **base season absent from `history`:** `paced` is set and its age is strictly
    greater than every realized age
  - **base season present in `history`** (the January-rebuild case): `paced is None`,
    and `history` is returned unchanged -- no duplicate point at that age
  - `paced_label` reads `"<season> pace"` when `base_season_partial` is true or absent,
    and `"<season>"` when it is false
  - `comp_careers` is parallel to `comps` and same length under every `n`
  - every `match_age` equals `view.age`
  - `career` values are floor-netted on VAR
  - a blob with no `careers` key yields empty `career`s and no exception
  - a blob whose `careers` is a list (not a mapping) is refused the same way
  - a comp entry with no `id` yields an empty `career` rather than raising
  - a two-way comp's hitter and pitcher careers do not collide: `careers` entries under
    `"<id>:hitter"` and `"<id>:pitcher"` are both preserved and each pool's view reads
    its own
- `tests/test_web/test_season_routes.py`
  - the player page renders with the new island keys (`paced`, `paced_label`,
    `comp_careers`) present, and 200s with a careers-less blob
- Boundary test (new, `tests/test_trajectory/`): none of the modules listed in
  Requirement 2's table is referenced from `src/fantasy_baseball/trajectory/**` or
  `scripts/push_trajectory_board.py`

Prose removal is asserted negatively where a test already renders the page: the removed
sentences must not appear in the rendered HTML, and the kept vintage sentence must.

## Phasing

Three independently shippable steps, in dependency order:

1. **Paced point + prose removal.** The only blob change is `base_season_partial`, which
   is additive and defaults to `True` when absent, so this ships against the
   currently-deployed blob with no re-push. The suppression rule needs nothing stored.
2. **Blob schema: comp `id` + `careers` map.** Push-side and view-side plumbing with no
   UI.
3. **The card grid.** JS + CSS + template, reading what step 2 stores.

Step 2's reader tolerates a blob without `careers`, and step 3's grid tolerates empty
careers, so any two of the three can be deployed in either order without a broken page.

### Deploy gate on the first `careers`-writing push

**A local `push_trajectory_board.py` run writes to the same prod Upstash the live Render
build reads.** Phase ordering in the repo does not prevent that -- running the new push
from this machine before Render has the new reader is what broke `/trajectory` on
2026-08-06 ([[feedback_deploy_before_pushing_blob_formats]]).

So the first push that writes `careers` or comp `id`s is gated on, in order:

1. step 2's reader merged to `main`,
2. Render's deploy of that commit finished,
3. `/trajectory?view=player&player=<someone>` loaded against prod and rendering,
4. only then run the push.

Until step 4, prod keeps serving the old blob, which the new reader handles by design.
