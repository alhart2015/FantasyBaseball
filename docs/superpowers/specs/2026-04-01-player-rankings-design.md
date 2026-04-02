# Player Rankings

**Date:** 2026-04-01
**Branch:** `player_rankings`

## Problem

No way to see how a player ranks relative to the full pool. wSGP tells you leverage-weighted value, but not overall production quality. When evaluating players across lineup, waivers, trades, and search, you need a universal "how good is this player" number.

## Design

### Ranking system

Three ordinal rankings per player, all based on `calculate_player_sgp` (unweighted, equal contribution from all 5 roto categories for their player type):

1. **ROS rank** — from ROS blended projections. The primary display rank.
2. **Preseason rank** — from preseason blended projections. Shows where the player was expected to be.
3. **Current rank** — from actual game log accumulated stats. Shows how they've performed so far. Null for players with no game log data.

Hitters and pitchers are ranked in **separate pools**. A hitter ranked #15 means 15th among all projected hitters.

### Backend: compute during refresh

In `run_full_refresh`, after projections and game logs are loaded:

1. Load all players from `ros_blended_projections` → compute SGP per player → sort descending → assign ordinal rank (1-based, within hitter/pitcher pools)
2. Same for `blended_projections` (preseason)
3. Load game log season totals → compute SGP from actual stats → sort descending → assign ordinal rank. Players with no game log data get no current rank.

Store as `rankings.json` cache: a dict keyed by normalized player name, each value is `{"ros": int, "preseason": int, "current": int | null}`.

### Backend: attach ranks to player data

After computing rankings, attach a `"rank"` dict to every player flowing through the pipeline:

- `roster_with_proj` entries (lineup page)
- Waiver scan results — both `add` and `drop` players
- Trade proposals — both `send` and `receive` players
- Buy-low candidates (trade targets and free agents)

Each gets: `"rank": {"ros": 15, "preseason": 22, "current": 8}`

For player search API, look up ranks from cached `rankings.json` at query time.

### Frontend: display

**Primary display:** ROS rank badge next to player name or wSGP. Styled as a compact pill (e.g., `#15`).

**Tooltip on hover:** Shows all three ranks:
```
ROS #15
Preseason #22
Current #8
```
Current shows "—" when no game log data exists.

### Surfaces to update

| Surface | Template file | Data source |
|---------|--------------|-------------|
| Lineup (hitters + pitchers) | `lineup.html` | `roster_with_proj` entries |
| Waiver wire | `waivers_trades.html` | waiver scan results |
| Trade recommendations | `waivers_trades.html` | trade proposals |
| Buy-low targets | `waivers_trades.html` | buy-low candidates |
| Buy-low free agents | `waivers_trades.html` | buy-low candidates |
| Player search | `players.html` | search API response |

### Ranking computation details

For each data source (preseason/ROS/current):
1. Run `calculate_player_sgp(pd.Series(player_dict))` for every player
2. Separate into hitters and pitchers
3. Sort each pool by SGP descending
4. Assign rank 1, 2, 3, ... (no tie-breaking needed — ties get arbitrary adjacent ranks)
5. Build lookup: `{normalized_name: rank}`

For current stats ranking, the player dict is built from game log totals with `player_type` set based on which table the stats came from. Rate stats (AVG, ERA, WHIP) are computed from components (H/AB, ER*9/IP, (BB+H)/IP).

## What this does NOT include

- Position-specific rankings (e.g., rank among all shortstops)
- Historical rank trends over time
- Rank-based filtering or sorting in the UI
