# Multi-Player Trade Evaluator

## Overview

Add a "Build a Trade" section to `/waivers-trades` that lets the user construct an arbitrary N-for-M trade with a specific opponent and see the resulting delta-roto for their own team. The existing 1-for-1 Trade Finder stays in place — this is a complementary tool for evaluating specific proposals rather than discovering them.

## Motivation

The existing Trade Finder only handles 1-for-1 swaps and is optimized for discovery (given player X, find counterparts). Real trade discussions involve multi-player proposals, roster-balancing drops, and waiver-wire replacements for the open roster slot that can result. The user needs to evaluate whether "give up Soto + Acuna, get Judge + Trout + Alvarez, drop X to make room, pick up Y from waivers" is a net positive before making the offer.

## UI

All new UI lives in `src/fantasy_baseball/web/templates/season/waivers_trades.html`, below the existing Trade Finder, in a section titled "Build a Trade".

### Opponent selector

Dropdown populated from the cached `opp_rosters` keys. Changing it swaps the right-hand roster panel and resets all player selections on both sides.

### Roster panels (side-by-side)

**My team panel** (left) — one row per player on the user's active roster (not on IL). Each row shows:

- Player name, eligible positions, current ROS rank.
- Per-row controls:
  - Segmented control: `— / TRADE / DROP`
  - `Active / Bench` toggle (used only for delta-roto math; defaults to current Yahoo `selected_position` — anyone currently in BN defaults to Bench, everyone else Active)

**Opp team panel** (right) — same layout, but only the `— / TRADE / DROP` segmented control. No bench toggle since we do not compute the opponent's delta-roto.

Players on IL (either team) are shown greyed-out and cannot be traded, dropped, or toggled.

### Waiver pickups (my side only)

Autocomplete search input below the my-team panel. Scope: players with ROS projections who are NOT on any roster (mine or any opponent's). Selected pickups render as chips under the input, each with:

- Player name, positions, ROS rank.
- Remove button (`x`).
- `Active / Bench` toggle (defaults to Active).

### Legality banner

Live-updating text above the Evaluate button showing net player change per team:

> My team: +2 / -1 (net +1)   |   Opp: -2 / +1 (net -1)

The banner turns red if either team's resulting roster size (excluding IL) is not exactly **23** (12 hitters + 9 pitchers + 2 bench). Position coverage is NOT enforced; it is the user's responsibility to mark a legal active lineup. If the active set (my team only) does not have exactly 12 hitters + 9 pitchers, a yellow advisory warning appears but does not block evaluation.

### Evaluate button

Disabled until:

- An opponent is selected.
- At least one player is marked `TRADE` on each side.
- Both resulting rosters are exactly 23 (legality banner is not red).

On click, POSTs the proposal to `/api/evaluate-trade` and renders the result panel below.

### Result panel

Same visual style as existing trade cards (reuse CSS classes):

- Delta-roto total (signed, large).
- Per-category breakdown table: R, HR, RBI, SB, AVG, W, K, SV, ERA, WHIP — each showing before value, after value, and category delta.
- Before/after standings snapshot: my team's projected end-of-season roto points before vs. after the trade.

## Backend

### New module

`src/fantasy_baseball/trades/multi_trade.py`

```python
@dataclass
class TradeProposal:
    opponent: str
    send: list[str]         # my-side player_type keys "name::hitter"
    receive: list[str]      # opp-side player_type keys
    my_drops: list[str]
    opp_drops: list[str]
    my_adds: list[str]      # waiver pickup keys
    my_active_ids: set[str] # keys of players marked active post-trade

@dataclass
class CategoryDelta:
    before: float
    after: float
    delta: float  # roto points change for this category

@dataclass
class MultiTradeResult:
    legal: bool
    reason: str | None
    delta_total: float
    categories: dict[str, CategoryDelta]   # keyed by "R", "HR", "RBI", ...
    standings_before: float                # my roto total before
    standings_after: float                 # my roto total after

def evaluate_multi_trade(
    proposal: TradeProposal,
    hart_roster: list[Player],
    opp_rosters: dict[str, list[Player]],
    waiver_pool: dict[str, Player],
    standings: dict,
    projected_standings: dict,
    team_sds: dict | None,
    roster_slots: dict,
) -> MultiTradeResult: ...
```

### Math

1. **Resolve players** — look up each key in the appropriate source (my roster, opp roster, waiver pool). Fail fast with a clear reason if anything is missing.
2. **Legality check** — for each team, compute `new_size = current_non_il_size - drops - traded_away + traded_in + adds`, where `current_non_il_size` is the count of players NOT currently on IL. Both sides' `new_size` must equal 23. IL players are not touched by the trade and continue to sit on IL post-trade; they are not counted at any stage. If not equal, return `legal=False` with a reason naming which side failed and by how much.
3. **My-team delta-roto**:
   - Baseline active set: my current active roster (everyone not on IL and not in BN, per Yahoo `selected_position`), projected end-of-season stats from `projected_standings`.
   - Post-trade active set: `my_active_ids` intersected with (kept players + arrivals + adds). Members contribute their full ROS projection.
   - Compute before/after team stats using existing `apply_swap_delta` pool-assumption logic, generalized to lists (see refactor below).
   - Run both through `score_roto()` with `team_sds` (ERoto) or rank-based scoring. Return per-category and total deltas via `DeltaRotoResult`.
4. **Opp side** — no roto math. Just the size legality check.

### Refactor `_can_roster_without`

Currently scoped to removing a single player. Generalize to:

```python
def _can_roster_after(
    roster: list[Player],
    removals: list[str],
    additions: list[Player],
    roster_slots: dict,
) -> tuple[bool, str | None]: ...
```

Size-only check: `len(roster) - len(removals) + len(additions) == active_slots + bench_slots` (23 in this league). IL slots are always excluded. The existing callers in `evaluate.py` wrap this with single-element lists.

### Routes

**`POST /api/evaluate-trade`** — new route in `season_routes.py`.

Request:

```json
{
  "opponent": "Team Name",
  "send": ["Juan Soto::hitter", "Ronald Acuna Jr.::hitter"],
  "receive": ["Aaron Judge::hitter", "Mike Trout::hitter", "Yordan Alvarez::hitter"],
  "my_drops": ["Cal Raleigh::hitter"],
  "opp_drops": [],
  "my_adds": ["Jonah Heim::hitter"],
  "my_active_ids": ["Aaron Judge::hitter", "Mike Trout::hitter", ...]
}
```

Response:

```json
{
  "legal": true,
  "reason": null,
  "delta_total": 2.4,
  "categories": {
    "R":   {"before": 980, "after": 995, "delta": 1.0},
    "HR":  {"before": 245, "after": 252, "delta": 0.0},
    ...
  },
  "standings_before": 74.5,
  "standings_after": 76.9
}
```

**`GET /api/waiver-search?q=<query>`** — new lightweight endpoint. Returns up to 20 players with ROS projections whose name matches the query and who are on no roster. If an existing search endpoint already covers this, reuse it instead.

## Filter: waiver pool

Built once per evaluation (or cached): union of cached `ROS_PROJECTIONS` minus the union of (my roster ∪ every opp roster). Key by `name::player_type` for uniqueness.

## Testing

`tests/test_trades/test_multi_trade.py`:

- 2-for-2 trade, no drops, no adds — verify delta-roto matches two sequential single swaps (sanity check against existing `compute_trade_impact`).
- 2-for-3 trade with 1 drop + 0 adds on my side — verify legality passes and new arrival contributes only if marked active.
- 2-for-3 trade with 1 drop + 1 add — verify waiver pickup is included when active, excluded when benched.
- Illegal trade (wrong resulting size) — verify `legal=False` with descriptive reason.
- Bench toggle respected — verify a kept player toggled to bench drops out of the post-trade projection.
- IL player in roster does not count toward the 23 legality target.

## Out of scope

- Opponent's delta-roto in result panel (math works only for my team).
- Waiver pickups for the opponent.
- Position slot assignment (C, 1B, 2B, …) — just active/bench.
- Live preview on every checkbox change (evaluate is button-driven).
- Trade history / "what if we'd made this trade 3 weeks ago".
- Counter-offer suggestion generation.

## Files touched

- `src/fantasy_baseball/trades/multi_trade.py` (new)
- `src/fantasy_baseball/trades/evaluate.py` (refactor `_can_roster_without` → `_can_roster_after`)
- `src/fantasy_baseball/web/season_routes.py` (new `POST /api/evaluate-trade`, maybe new `GET /api/waiver-search`)
- `src/fantasy_baseball/web/templates/season/waivers_trades.html` (new "Build a Trade" section + JS)
- `tests/test_trades/test_multi_trade.py` (new)
