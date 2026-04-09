# Pending Roster Moves

## Overview

Show pending roster transactions (waiver claims, adds/drops awaiting processing) on the Lineup page and account for them in waiver recommendations. In this league, roster moves submitted mid-week don't take effect until the next Tuesday. During that window, claimed players disappear from the FA pool but don't appear on anyone's roster, creating a blind spot.

## Problem

When you submit an add/drop (e.g., add Otto Lopez, drop Marcus Semien) on Wednesday:
- Yahoo removes Lopez from the free agent pool immediately
- Semien stays on your roster until Tuesday processing
- The dashboard still shows Semien as yours and doesn't mention Lopez
- The waiver scanner recommends a *second* move to drop Semien (for the next-best FA), which is redundant
- Other teams' pending claims are also invisible, so the scanner may recommend players that are already claimed

## Data Source

Yahoo Fantasy API exposes league transactions via `league/{league_id}/transactions`. Each transaction includes:
- `transaction_id`, `type` (add/drop, add, drop, commish, trade)
- `status` (successful, pending, etc.)
- `timestamp`
- Player details: name, player_id, positions, team
- Transaction data: source (freeagents/waivers), destination team, or vice versa

Pending transactions have `status != "successful"`. On processing day (Tuesday), this list is typically empty.

## Design

### Data fetch and caching

During `run_full_refresh()`, fetch all league transactions. Filter to pending ones (`status != "successful"`). Parse each into a normalized structure:

```python
{
    "transaction_id": str,
    "type": str,            # "add/drop", "add", "drop", "trade"
    "status": str,          # "pending", "waiver", etc.
    "timestamp": str,       # ISO datetime
    "team": str,            # team name
    "adds": [{"name": str, "player_id": str, "positions": [str]}],
    "drops": [{"name": str, "player_id": str, "positions": [str]}],
}
```

Cache as `pending_moves.json` via `write_cache("pending_moves", ...)` (persists to Redis for Render).

### Lineup page display

When pending moves exist for the user's team, show a "Pending Moves" banner at the top of the Lineup page, above the roster table. Each pending transaction displays as a compact card:

- "ADD Otto Lopez (2B, SS) / DROP Marcus Semien (2B, SS)"
- "Takes effect Tuesday"

Only shown when there are pending moves for the user's team. No interactivity — just awareness.

### Waiver scan adjustment

Before running `scan_waivers()` in the refresh pipeline, adjust the inputs using pending moves from *all* teams:

**User's roster:**
- Add pending-add players (match to projections, create Player objects)
- Remove pending-drop players
- This makes the optimizer evaluate your *future* roster, not your current one

**Free agent pool:**
- Remove any player that's a pending-add for any team (they're claimed, not available)

**Opponent rosters:**
- Add their pending-add players, remove their pending-drop players
- Keeps trade finder and other cross-team analysis accurate

### What's NOT included

- No new dashboard tab — just a banner on the existing Lineup page
- No transaction history or analyzer — separate feature (see `docs/feature_specs/transaction_analyzer.md`)
- No impact on leverage weights, lineup optimization, or rankings — only waiver recommendations
- No persistence to SQLite — pending moves are ephemeral by nature, cache-only is correct

## Files affected

- `src/fantasy_baseball/lineup/yahoo_roster.py` — new `fetch_pending_moves()` function
- `src/fantasy_baseball/web/season_data.py` — call fetch + cache during refresh, adjust waiver scan inputs
- `src/fantasy_baseball/web/season_routes.py` — pass pending moves to lineup template
- `src/fantasy_baseball/web/templates/season/lineup.html` — pending moves banner
