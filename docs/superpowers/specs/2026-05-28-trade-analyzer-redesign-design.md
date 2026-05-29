# Trade Analyzer Redesign -- Click-to-Swap UX + Two-Sided Deltaroto

Date: 2026-05-28
Status: approved (design), pre-implementation
Touches: `src/fantasy_baseball/web/templates/season/waivers_trades.html`,
`src/fantasy_baseball/web/static/season.css`,
`src/fantasy_baseball/trades/multi_trade.py`,
`src/fantasy_baseball/web/season_routes.py`,
`tests/test_trades/`, `tests/test_web/`

## Problem

The "Build a Trade" panel in `waivers_trades.html` (the side-by-side roster
builder under the Trade Finder section) has two real problems for actual use:

1. **Drag-and-drop is hard on mobile.** The current implementation uses
   SortableJS with each positional slot as its own single-item drop zone
   (`waivers_trades.html:663-700`). On touch screens, the drag handles are
   tiny chips inside cramped slot rows; the two-column grid collapses to a
   single column at `<768px` but the rows stay desktop-sized. The result is
   un-usable on phones, which is when most trade negotiations actually
   happen.
2. **The evaluator only shows my side.** The Build-a-Trade evaluator's
   response (`season_routes.py:871-888`) carries my-side `roto`,
   `ev_roto`, `stat_totals`, and `band` -- but not the opponent's. The
   Trade Finder cards on the same page DO show both sides (via
   `trades/evaluate.py`'s `hart_cat_deltas` / `opp_cat_deltas`), so the
   feature exists in one code path and not the other. To pitch a trade
   to another manager, you need to show them the deltaroto **on their
   roster** -- "here's why you should take this."

## Decision

Replace SortableJS drag-and-drop with a **click-to-select-and-swap** state
machine, redo the panel layout for mobile-first, and extend
`evaluate_multi_trade` to return the opponent-side delta with the same
three views (roto / ev_roto / stat_totals) plus band.

Backend and frontend change together. Branch: `feat/trade-analyzer-click-swap`.

## Click interaction model

Two states. The state machine is keyed by `playerKey` strings (the same
`"<name>::<player_type>"` keys used in `state.placements`), not DOM refs,
so re-renders don't invalidate selection.

**States:**

- `idle` -- no selection, no halo, no indicator strip.
- `selected(A)` -- exactly one player marked selected. Sticky bottom strip
  reads `Selected: <name>` with a `cancel` link.

**Transitions:**

| From         | Tap target                                   | Result                                                      |
|--------------|----------------------------------------------|-------------------------------------------------------------|
| `idle`       | Player chip (not IL)                         | `selected(A)`                                               |
| `idle`       | Waiver suggestion                            | `selected(W)` (origin tagged `waiver`)                      |
| `idle`       | Slot / empty area / non-player               | no-op                                                       |
| `selected(A)`| A itself                                     | `idle` (deselect)                                           |
| `selected(A)`| Non-player area (header, gutter)             | `idle` (deselect)                                           |
| `selected(A)`| Player B, same side, different slot          | swap A<->B slots (intra-team reshuffle) -> `idle`           |
| `selected(A)`| Player B, opposite side                     | swap A<->B across sides (the trade) -> `idle`               |
| `selected(A)`| Empty positional slot, same side             | move A there; displaced incumbent (if any) -> bench -> `idle` |
| `selected(A)`| Empty positional slot, opposite side         | move A across; same displacement -> `idle`                  |
| `selected(A)`| Bench zone, either side                      | move A onto that bench -> `idle`                            |
| `selected(A)`| Drops zone, either side                      | drop A -> `idle`                                            |
| `selected(A)`| IL slot                                      | no-op (IL stays read-only)                                  |
| `selected(A)`| Another waiver suggestion W                  | `selected(W)` (replace selection; A returns to prior place) |

The displacement rule (incumbent on a positional slot goes to bench when
the slot is taken) is **exactly today's `placeAt()` behavior** -- reused
unchanged so legality semantics stay consistent.

Each tap-pair commits one move. A 2-for-2 is two tap-pairs in a row, same
as drag-drop today commits each drag individually. Changing the opponent
dropdown wipes `state.placements` AND clears the pending selection.

## Layout

Mobile-first. Mockup screenshots saved in
`.superpowers/brainstorm/217-1779977023/content/layout.html` while the
session was live; layout decisions below are the source of truth.

- **Stacked rosters on mobile**, my team on top, opp below. Side-by-side
  at `>=768px` (same breakpoint already used).
- **Sticky team header inside each panel** -- the header (team name + active
  count `N/23`) stays pinned at the top of its panel as the user scrolls.
- **Whole roster row is the tap target** (~36px tall). Slot label + chip in
  a 2-column grid (`28px 1fr`).
- **Selected state**: amber (`#fc4`) left-border (4px) + inset glow, row
  background brightens. Only one selection at a time; the color is loud
  on purpose so the user doesn't lose track on a long roster.
- **Bottom indicator strip** is `position: sticky; bottom: 0;` and only
  appears while a selection exists. Renders `Selected: <name> -- tap a
  slot to swap or move` with a `cancel` link.
- **Bench / IL / Drops** are three labeled zones below the slot grid per
  team (same structure as today; restyled for taller touch targets).
- **Result panel**: two cards (`My Team` / `Their Team`). Stacked on
  mobile, side-by-side at `>=768px`. Each card has its own three-mode
  toggle (`ev_roto` / `roto` / `stat_totals`) and its own band line
  (`+1.8 +/- 0.4 roto - 79% to help`).
- **Action row**: `Evaluate` and `Optimize Both` buttons (`Optimize Both`
  replaces today's `Optimize Lineup` -- see backend section).

## Backend changes

### `trades/multi_trade.py`

Current `MultiTradeResult` (verified in source) carries my-side view:

```python
@dataclass
class MultiTradeResult:
    legal: bool
    reason: str | None
    delta_total: float
    categories: dict[str, CategoryDelta]
    roto: ViewBlock
    ev_roto: ViewBlock
    stat_totals: ViewBlock
    band: dict
```

**Add opp-side fields** (parallel shape, `opp_` prefix):

```python
    opp_delta_total: float
    opp_categories: dict[str, CategoryDelta]
    opp_roto: ViewBlock
    opp_ev_roto: ViewBlock
    opp_stat_totals: ViewBlock
    opp_band: dict
```

The function already derives both teams' post-trade stat lines via
`aggregate_player_stats` and `apply_swap_delta` from
`trades/evaluate.py`; today only the my-side delta is surfaced. The new
work computes the parallel delta against opp's baseline (their
`projected_standings` row) and runs the same three views + Monte Carlo
band using `team_sds[opp_name]`. No new sampler -- same code path, opp's
variance vector.

Symmetry: a trade is mirrored on both sides (I send X, they receive X),
so the underlying stat math is the same join viewed from the other
baseline. Drops, IL, and waiver-adds are side-local -- `my_drops` only
affects my view, `opp_drops` only theirs.

### `TradeProposal`

Add `opp_active_ids` to mirror existing `my_active_ids`:

```python
    opp_active_ids: set[str] = field(default_factory=set)
```

JS reads opp's active set from `state.placements` the same way it does
for my side and posts it. **Backwards-compat:** if `opp_active_ids` is
absent in the payload, fall back to opp's current `selected_position`
from their roster (close to today's implicit behavior). Log a
deprecation warning on the fallback path so we can remove it later.

### `/api/evaluate-trade`

Additive response in `web/season_routes.py:800-889`. Re-use the local
`_serialize_view` for the new `opp_*` views and add them to the JSON.
Existing fields stay byte-identical -- the JS adds new readers without
losing the old ones.

### `/api/optimize-trade-lineup`

Add a `side` param to `web/season_routes.py:891+`. Values: `"my"` |
`"opp"` | `"both"`. **Default `"both"`** so the renamed `Optimize Both`
button does the right thing without further UI plumbing.

- `"my"` -- unchanged (today's behavior).
- `"opp"` -- run the optimizer against the opp's post-trade roster,
  return `opp_slots`.
- `"both"` -- both; return both `my_slots` and `opp_slots`.

The optimizer engine (`lineup/optimizer.py`'s
`optimize_hitter_lineup` / `optimize_pitcher_lineup`) does NOT change --
we call it twice. Validation: bad `side` -> 400 with
`{"error": "side must be 'my', 'opp', or 'both'"}`.

### Partial optimizer failure

If `side=both` and one side's optimizer raises, return whichever
succeeded plus a `partial: true` flag. UI renders "Couldn't optimize
opp lineup -- using their current arrangement."

## Error handling

### JS side

| Risk                                                    | Handling                                                       |
|---------------------------------------------------------|----------------------------------------------------------------|
| Rapid double-tap on same player                         | Synchronous state machine: tap A -> selected; tap A -> idle. No debounce. |
| Tap on a chip after panel re-rendered                   | Selection is keyed by `playerKey` string, not DOM ref. Safe.   |
| Tap on IL slot while selection exists                   | no-op, state unchanged                                          |
| Tap on opp roster with no opponent chosen yet           | Panel renders empty; defensive `state.opponent !== null` check |
| Opponent dropdown change with active selection          | `rebuildPlacements()` wipes everything; also clear selection   |
| Tap on a waiver suggestion when roster player selected  | Replace selection (state moves `selected(A)` -> `selected(W)`) |

### Server side

| Risk                                                    | Handling                                                                   |
|---------------------------------------------------------|----------------------------------------------------------------------------|
| `/api/evaluate-trade` raises in opp-side computation    | Existing try/except returns `{"error": ...}`; UI shows `bt-result-bad`      |
| Missing `opp_active_ids` in old clients                 | Fall back to opp roster's `selected_position`; log deprecation warning      |
| `opp_band` Monte Carlo fails (e.g. zero variance)       | Same fallback as my-side band today: return `None`; UI falls back to plain delta line |
| Optimizer fails on one side under `side=both`           | Return successful side + `partial: true`; UI surfaces partial-result note  |
| `side` not in {`my`, `opp`, `both`}                     | 400 with explicit error message                                            |

## Testing

### Backend (pytest)

`tests/test_trades/test_multi_trade.py` -- new cases:

- **Opp-delta basic**: 1-for-1 swap, no drops/adds. Assert `opp_delta_total`
  matches `trades/evaluate.py`'s single-player `opp_delta` to within
  rounding (cross-validation between the two code paths).
- **Opp-delta with opp drops**: trade + opp drops a player from their
  bench. Assert `opp_categories` reflects the drop.
- **Opp-delta zero-trade boundary**: `send=[]`, `receive=[]`, only
  `my_adds`. `opp_delta_total` should be exactly 0.
- **`opp_active_ids` respected**: same trade, two different
  `opp_active_ids` sets, different `opp_stat_totals`.
- **`opp_active_ids` fallback**: omit from proposal; opp post-trade
  roster equals "incoming slots in where outgoing came from".
- **Existing my-side tests**: assert all current my-side values are
  byte-identical (regression guard against the new code path).

`tests/test_web/` -- `/api/evaluate-trade` response schema: new
`opp_*` fields present, existing fields equal a recorded fixture,
`legal=false` flow still returns `reason`.

`tests/test_web/` -- `/api/optimize-trade-lineup`:

- `side=both` returns both slot dicts.
- `side=opp` returns only `opp_slots`.
- `side` not in allowed set -> 400.
- Partial-failure path sets `partial=true`.

### Frontend (no JS runner in this repo)

The existing code already exposes `window.bt._state`,
`window.bt._computeMovesSummary`, etc. for debug (`waivers_trades.html:891-894`).
The state machine extends the pattern: expose `window.bt._selection`
and `window.bt._tap(target)` so QA can exercise from the browser
console without clicking.

**QA checklist** (lives in PR description, run before merge):

1. Tap player -> indicator strip shows their name; tap again -> cleared.
2. Tap player A (mine), tap player B (opp) -> swap committed, indicator clears.
3. Same-side tap-tap -> intra-team reshuffle.
4. Tap empty bench slot with selection -> move there.
5. Tap drops zone with selection -> drop committed.
6. Tap IL slot with selection -> no-op.
7. Tap waiver suggestion with active selection -> selection replaced.
8. Opponent dropdown change with active selection -> selection cleared, rosters rebuilt.
9. Evaluate with valid trade -> both team cards render with band + cat deltas.
10. Optimize Both -> both rosters update; press Evaluate again -> cards refresh.
11. Mobile viewport (375px) -- sticky headers, touch targets >=36px, bottom strip visible.
12. Desktop viewport (>=768px) -- side-by-side rosters and result cards.

If we want Playwright coverage long-term, the `_tap` exposure makes it
cheap to add. Not in scope for this PR.

## Out of scope

- The Trade Finder search section (mobile-fine, already two-sided).
- `trades/evaluate.py` (Trade Finder's 1-for-1 backend) -- separate
  code path; leave alone.
- Yahoo sync flows.
- The roto / SGP / replacement-level math.
- The Monte Carlo sampler or band-construction code.
- The lineup optimizer engine (we call it twice; it does not change).

## Phasing

One branch (`feat/trade-analyzer-click-swap`), recommended as two
sub-commits so the rewrite is bisect-friendly:

1. **Backend:** extend `MultiTradeResult` and `TradeProposal`, update
   routes, add `opp_*` tests. Old UI keeps working (it ignores the new
   fields).
2. **Frontend:** click-state machine + layout redesign + result-panel
   redesign. Reads the new fields. QA checklist passes.

Both ship in one PR.
