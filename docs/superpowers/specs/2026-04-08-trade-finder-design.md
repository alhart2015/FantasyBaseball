# Interactive Trade Finder

## Overview

Replace the batch Trade Recommendations section on the Waivers & Trades page with an interactive, player-driven trade search. Two modes: "Trade Away" (I have player X, find me a deal) and "Trade For" (I want player Y, what can I offer).

## Motivation

The current `find_trades()` runs during refresh, scanning all my players x all opponents and returning a generic top-5. This isn't useful for targeted trade exploration — the user wants to start from a specific player and see what's possible.

## Two Modes

### Trade Away

User names a player on their roster. The system:

1. Finds the player on the user's roster, gets their rank and wSGP (from user's leverage weights).
2. For each opponent team, scores **positional weakness** — how badly they need that position.
3. For each opponent player: checks rank proximity (within 5), positive wSGP gain for the user, roster legality both ways, and non-negative roto point impact.
4. Groups results by opponent, sorted by positional weakness (teams that need the user's player most appear first).
5. Within each team group, candidates sorted by wSGP gain.

### Trade For

User names a player on an opponent's roster. The system:

1. Identifies which opponent owns that player, gets their rank.
2. For each of the user's players: checks rank proximity (within 5 — user sends someone ranked within 5 of the target), positive wSGP gain for the user, roster legality, non-negative roto point impact.
3. Returns a single opponent group with viable send candidates sorted by wSGP gain.

## Positional Weakness Scoring

For Trade Away, surface teams that need the position being offered. For each opponent:

- Look at their current starter(s) at the sent player's position.
- Compute their starter's wSGP using that opponent's leverage weights.
- Compare against the league median wSGP at that slot across all teams.
- Teams whose starter is below median get a higher weakness score and sort first.

This is a lightweight heuristic using data already available from the refresh pipeline — no new computation required.

## Filtering Criteria

Both modes apply the same core filters (carried over from the existing trade logic):

- **Rank proximity**: `send_rank - receive_rank <= 5` (the player you send can be at most 5 ranks worse than what you receive — looks fair to the opponent). Sending a better-ranked player always passes.
- **Positive wSGP gain**: the user's weighted SGP must improve (hidden value).
- **Roster legality**: both rosters remain valid after the swap (incoming player can fill an active slot).
- **Non-negative roto impact**: `compute_trade_impact()` must show `hart_delta >= 0`.

## API

**Endpoint**: `POST /api/trade-search`

**Request body**:
```json
{
  "player_name": "Marcus Semien",
  "mode": "away"
}
```

**Response** (grouped by opponent):
```json
[
  {
    "opponent": "Team Name",
    "positional_weakness": 0.85,
    "candidates": [
      {
        "send": "Marcus Semien",
        "send_positions": ["2B", "SS"],
        "send_rank": 42,
        "receive": "Opponent Player",
        "receive_positions": ["SS", "3B"],
        "receive_rank": 38,
        "hart_wsgp_gain": 0.45,
        "hart_delta": 2,
        "opp_delta": 1,
        "hart_cat_deltas": {"R": 1, "HR": 0, "RBI": 1, ...},
        "opp_cat_deltas": {"R": -1, "SB": 2, ...}
      }
    ]
  }
]
```

Uses cached refresh data (rosters, standings, leverage, rankings). No Yahoo API calls at search time.

## Backend Changes

### `src/fantasy_baseball/trades/evaluate.py`

- **Delete** `find_trades()`.
- **Add** `search_trades_away(player_name, hart_name, hart_roster, opp_rosters, standings, leverage_by_team, roster_slots, rankings, projected_standings=None) -> list[dict]`
  - Locates the named player on the user's roster.
  - Computes positional weakness per opponent.
  - Iterates opponent rosters with standard filters.
  - Returns results grouped by opponent, sorted by positional weakness.
- **Add** `search_trades_for(player_name, hart_name, hart_roster, opp_rosters, standings, leverage_by_team, roster_slots, rankings, projected_standings=None) -> list[dict]`
  - Locates the named player across opponent rosters.
  - Iterates user's roster with standard filters.
  - Returns a single opponent group with viable send candidates.
- **Keep** `compute_trade_impact()`, `_project_team_stats()`, `_player_ros_stats()`, `_find_player_by_name()`, `_can_roster_without()`, and all helpers.

### `src/fantasy_baseball/web/season_routes.py`

- **Add** `POST /api/trade-search` endpoint.
  - Reads `player_name` and `mode` from request JSON.
  - Loads cached data: rosters, standings, leverage, rankings, projected standings.
  - Calls `search_trades_away()` or `search_trades_for()` based on mode.
  - Returns JSON response.
- **Update** the `/waivers-trades` route to stop passing `trades` to the template (no longer pre-computed).

### `src/fantasy_baseball/web/season_data.py`

- **Remove** the `find_trades()` call from `run_full_refresh()`.
- **Remove** the `"trades"` entry from `CACHE_FILES`.
- Ensure the data the search functions need is still cached: rosters (`roster`, opponent rosters), standings, leverage, rankings, projected standings.

## Frontend Changes

### `templates/season/waivers_trades.html`

- **Replace** the "Trade Recommendations" section with:
  - Text input for player name (with autocomplete from roster + opponent rosters).
  - Two buttons: "Trade Away" (search among my players) and "Trade For" (search among opponent players).
  - Results container that renders opponent-grouped cards.
- **Result cards** reuse the existing visual style:
  - Opponent header with positional weakness indicator (Trade Away mode).
  - Send/receive player names, positions, ranks.
  - wSGP gain badge.
  - Expandable per-category roto point deltas.
  - "Load Before/After Standings" drill-down (reuse existing `loadTradeStandings` pattern).
- Loading spinner during search.

## What's NOT Changing

- Waiver Wire section — untouched.
- Buy-Low sections — untouched.
- All existing helpers in `evaluate.py` — kept as-is.
- The refresh pipeline still caches rosters, standings, leverage, rankings — the trade search reads from these caches.
