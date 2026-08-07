# Trajectory board: player search and career chart -- Design

Date: 2026-08-07
Status: approved design (brainstorming complete)
Author: session (Hart)
Implements: #324, with the deviations recorded below
Builds on: the all-teams view (#340, merged as bdf58e95)

## Problem

The board ranks players and the team views group them, but neither shows one player's
**shape** -- what his career has done and what the model thinks it does next. That is the
question a keep-or-cut call actually turns on, and today it can only be answered from
`scripts/player_trajectory.py`, as a table of numbers, at a terminal.

A line is the right rendering for a trajectory. Numbers in a column make you reconstruct
the curve in your head.

## Goals

- **Search a player by name**, get his career to date and his projected path on one chart.
- **Career solid, projection dashed, p10-p90 as a shaded band** -- one rule the eye learns
  at a glance: line style says whether it happened.
- **His N closest comps' realized paths on the same axes**, N a control defaulting to 5.
- The chart must not let the comps read as evidence for the projection. See "The honesty
  requirement", which is a requirement and not a note.
- Works on the deployed dashboard, from a phone, with no panel on the server.

## Non-goals

- **No new estimator.** The line is `shape`, the same fit the board runs on, so one player
  never has two numbers in one app.
- **No comp-matcher comps.** `--match track`/`current` pick a cohort similar at the
  STARTING point. This picks paths that match the whole forward prediction, which is what
  the chart draws. Both exist; this view uses the second.
- **No full career arcs for comps.** See the deviation below.
- **No panel on Render.** `data/trajectory/` is gitignored and the board's design is
  "offline artifact, dashboard is a pure reader". That property is kept.
- **No vendoring of Chart.js in this change.** It is a real dependency risk and it is
  filed as #342; this spec depends on it, it does not fix it.

## Deviations from #324

**1. Line styles.** #324 specifies the projected point estimate SOLID with p10/p90 as two
dotted lines. This uses **solid for what happened, dashed for what is predicted, and a
shaded region for p10-p90**. Rationale: "solid" cannot mean both "observed" and "guessed"
on one chart, and a filled band stays dominant over five comp curves where two more dotted
lines would compete with them. This matters because of the honesty requirement below.

**2. The query player's career to date is drawn.** #324 charts only ages +1..+5. Without
the realized past there is nothing for "dashed" to contrast against, and the shape of the
career is half the signal.

**3. Comps are drawn across the forward ages only**, not their full career arcs -- #324
left this open. Their realized `+1..+5` aligns to the query's projected ages, so the
comparison sits exactly where the projection is and the left half of the chart stays
purely factual. It is also half the data.

**4. Comps are precomputed for all 1,169 players**, not a top-N subset. Measured: five
comps cost ~370 bytes per player, so limiting to the top 123 would save under 400 KB on a
multi-megabyte payload and cost a dead end every time a bench player, prospect or trade
target is searched -- which is when the view gets used. See "Payload" for the final size
with `MAX_COMPS = 10`.

**5. Two of #324's open questions are closed by decision**, both toward the simpler first
cut: candidates must be an **exact age match** (keeps the x-axis aligned), and MSE is
**unweighted** across the five horizons (weighting near years is a knob that needs its own
justification and evidence).

## Chosen approach

### Where the work happens

`push_trajectory_board.py` already computes the board offline, on the machine that has the
panel, and writes one blob to Upstash. Comps and career history are computed in the same
place, by the same run, and travel in the same blob. The dashboard stays a pure reader.

### `trajectory/comp_paths.py` -- new module

```python
def closest_paths(
    prepared: Prepared, predicted: Sequence[float], age: int, n: int
) -> list[CompPath]
```

`CompPath` carries `mlbam_id`, `season`, `rmse`, and `path` (the realized `+1..+5`).

**It returns ids, never names.** Name resolution happens once, in
`push_trajectory_board.py`, through the `player_names(PEOPLE_CACHE)` map it already loads
for the board -- so the module stays free of the people cache and testable without it, and
the blob carries names because the browser has no way to resolve one. An id with no name in
the cache renders as the id rather than being dropped: a comp is still a comp.

Selection rule, in full:

- **Candidates are history rows at EXACTLY `age`** with all five forward values realized.
- **Match on raw SGP**, never VAR: VAR's floor is slot-specific, so matching on it would
  compare the query's OF floor against each comp's own position. The display conversion is
  a separate decision with its own rule -- see "The chart".
- **RMSE over the five horizons, unweighted.** Ties break on `(rmse, mlbam_id, season)` so
  the result is a fixed function of the data rather than of row order.

`push_trajectory_board.py` calls it once per player with `n=MAX_COMPS`; the view slices
that list down to the requested `N`. The module itself has no opinion about either number.

It goes in its own module rather than in `shape.py`: `shape.py` produces a prediction, this
consumes one after the fact, and they share nothing but `Prepared`.

**`Prepared` gains `mlbam_id`.** `prepare()` already builds it and drops it. Adding it to
the dataclass is what lets a comp be named without rebuilding `build_history` alongside --
which is what the #324 prototype did, and which risks the two row orders diverging
silently.

### Requiring the full path, and saying so

A comp scored on two realized years competes against one scored on five, and the short path
wins for free. So all five are required -- which excludes **330 age-25 seasons from 2021
onward**, and means no comp is more recent than about 2020.

That is a rule, not an accident, and the page says so. Without the sentence a reader
notices the pattern and distrusts the feature.

### Payload

Each player gains:

| key | shape |
|---|---|
| `history` | `[[age, sgp], ...]`, realized seasons from the panel, ascending by age |
| `comps` | `[{name, season, rmse, path: [...]}, ...]`, best first, **`MAX_COMPS = 10` of them** |

**Ten are baked, not five.** `N` is a display control and the payload is computed hours
earlier by a different process, so the control can only ever slice what was stored -- a
control that asks for more than exists is a control that lies. Ten is the clamp ceiling
from requirement 5, so any legal `N` is servable from the blob. The chart still shows five
until asked otherwise.

Cost of the other five: ~370 bytes per player, so the payload lands near **1.84 MB** rather
than 1.40 MB. Measured on the same basis as the rest of this section.

762 KB -> ~1.84 MB, measured. There is no size ceiling enforced anywhere in this repo
(checked `kv_store`, the push script, and `docs/`); the free-tier headroom is orders of
magnitude above this.

**Both are read with `.get(..., [])`.** A payload pushed before this feature must render
what it can and say what is missing, not 500. This is #332's outage applied deliberately:
that incident took `/trajectory` down because a reader refused a blob it could largely
read.

### The view

`VIEWS` becomes `("board", "teams", "player")`; `select_view` already clamps it. The URL is
`?view=player&player=<name>`, so it is bookmarkable and shareable like the other two.

**Name resolution only, never a typed id.** An ambiguous name lists the candidates and
renders no chart -- the same rule `player_trajectory.py` enforces by refusing a
`--mlbam-id` that disagrees with `--player`. CLAUDE.md names a hand-carried id as a defect
class that has twice landed on a real row belonging to someone else.

### The chart

Chart.js, loaded through the existing `ensureChartJs()` promise in `season_trends.js` --
reused, not reimplemented; its injection-then-await exists to avoid a `Chart is not
defined` race and that reasoning is not worth rediscovering.

- **X axis: age.** **Y axis: VAR by default**, with an SGP toggle. Matching is on SGP
  regardless; the toggle is a display transform.
- **On the VAR axis, every line is netted against the QUERY player's floor** -- his career,
  his projection, and every comp's path alike. Not each comp's own slot floor: the chart's
  question is "what would this trajectory be worth in *his* slot", and per-comp floors
  would put lines on one axis that are not comparable to each other, which is the mixed-
  scale defect #331 was opened against. The query's own past uses his CURRENT floor too,
  even across seasons where he was slot-eligible elsewhere -- one bar, stated in the axis
  label, beats a bar that moves invisibly along the line.
- Career: solid. Projection: dashed. p10-p90: filled translucent band.
- Comps: thin, faint, solid (they happened), across the forward ages only.
- Below the chart, the same numbers as a table -- age, VAR, SGP, p10..p90 per row -- so the
  page is readable when the chart is not.

### The honesty requirement

**The comps are selected ON THE OUTCOME.** They are the five paths out of ~1,200 that
happened to land closest to a prediction. That makes them a fair illustration of what this
shape looked like when it played out, and it makes them **not evidence for the forecast**.

Drawn naively they hug the projected line and make it look far more certain than it is. The
honest uncertainty is the p10-p90 band, which is wider than the comp spread by
construction. Therefore:

- the band is visually dominant over the comp lines (that is what decides deviation 1);
- the block is labelled **closest realized paths**, never anything implying basis or
  support;
- each comp shows its RMSE, so the reader sees they were chosen for closeness;
- the pace note carries through (`2026 is 70% complete -- 8.7 SGP so far, pacing to 12.5`),
  because the whole query is built on the paced figure;
- the `(!)` extrapolation flag carries through. A chart renders a poorly-supported fit
  exactly as confidently as a well-supported one.

## Requirements

1. `?view=player&player=<name>` renders career, projection, band, and comps for a resolved
   player; the other two views are unaffected.
2. An unknown name renders a not-found message; an ambiguous one lists candidates and
   renders no chart.
3. Career is solid, projection dashed, p10-p90 a filled band, comps thin and faint.
4. Comps are drawn only across the projected ages.
5. `N` is a control defaulting to 5, clamped to a sane range.
6. The VAR/SGP toggle changes only the display; the same comps and the same ordering. On
   the VAR axis every series -- career, projection, band and comps -- is netted against the
   query player's floor, and the axis label says so.
7. Every comp shows its RMSE, and the five-realized-years rule is stated on the page.
8. The pace note and the `(!)` flag render when they apply.
9. A payload with no `history`/`comps` renders the rest of the page and says what is
   missing.
10. `closest_paths` is deterministic: same inputs, same order, regardless of row order.

## Edge cases and failure modes

| case | behaviour |
|---|---|
| Name matches nobody | "No player named X on this board." No chart. |
| Name matches several (two Max Muncys) | List the candidates with age and slot; render no chart. Never guess -- this is the #284 join being ambiguous, surfaced rather than resolved. |
| Player has no `history` (payload predates the feature) | Chart renders the projection only, with a line naming the stale payload and the fix (`push_trajectory_board.py`). |
| Player has fewer than `N` comps | Render what exists. Fewer than five is possible for extreme ages where few candidates have five realized years. |
| Player has NO comps | Chart renders career + projection; the comp block says no historical path matched at this age with five realized seasons. |
| Career shorter than one season (rookie) | `history` is one point or empty. A one-point line renders as a dot; nothing special-cased. |
| Age at the panel's edge (very old/young) | Candidate pool is small or empty. Falls into the no-comps case above. |
| `?n=` junk or out of range | `_clamp(n, 1, 10, 5)`. Ten is a chart-legibility ceiling, not a data limit. |
| Chart.js fails to load (CDN blocked, offline) | The table below the chart still renders. This is the exposure #342 is about; the table is what makes it survivable rather than blank. |

## Testing expectations

**`closest_paths` (unit).** Ordering by RMSE; exact-age filtering excludes a near-age row;
the full-path requirement excludes a two-year path that would otherwise win on a smaller
MSE; ties break deterministically and the result is identical when the input rows are
shuffled; matching is on SGP (a row whose VAR would rank differently still sorts by SGP).

**View (unit).** Name resolution, ambiguity, unknown; `.get` fallbacks for a payload with
no `history`/`comps`; `N` clamping, including that asking for more than `MAX_COMPS`
yields what exists rather than erroring; the VAR/SGP toggle leaves comp identity and order
unchanged; and **every series on the VAR axis is netted against the QUERY player's floor**
-- asserted with a comp whose own slot floor differs, so using the wrong one fails.

**Route.** The three views coexist; `?view=player` with no `player=` renders the search box
rather than an error; a bookmarked URL for a since-departed player degrades to not-found.

**Anti-vacuity.** The comp fixture needs at least one near-age row (so age filtering can
fail), one two-year path that would win on raw MSE (so the full-path rule can fail), and
two rows tied on RMSE (so the tie-break can fail). Every rule-pinning assertion gets a
mutation check -- seven vacuous tests were caught in this codebase in one recent session,
three of them by mutation alone.

## Phasing

One PR, four commits, each independently green:

1. `Prepared` gains `mlbam_id`; `comp_paths.py` and its unit tests. Nothing consumes it.
2. `push_trajectory_board.py` bakes `history` and `comps`; push-script tests.
3. `build_player_view` in `trajectory_view.py` + view tests. No route yet.
4. Route branch, template, chart JS, table, and the view pill -- together, since a control
   that renders a commit before the branch answering it would clamp straight back.

No commit changes a signature its callers still use. `c96cd79b` on an earlier branch left
`/trajectory` returning 500 at one revision by splitting a callee from its caller; commit 4
is deliberately whole for that reason.

**A push is required after this merges** -- the deployed payload has no `history`/`comps`.
Unlike #332, the page degrades rather than breaking, so the ordering is not load-bearing.
