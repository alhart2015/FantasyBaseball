# Player Comparison Feature

## Overview

Add a player comparison section to the existing Players page that lets users select exactly two players — one on their roster, one not — and see a side-by-side stat comparison plus projected standings impact of the swap.

## User Flow

1. User browses the player table (existing functionality, unchanged)
2. Checkboxes appear on each row. User selects exactly 2 players.
3. Validation: one must be on the user's roster (`owner === "roster"`), one must not be. If both are on the roster or both are not, show an inline message explaining the constraint.
4. A comparison panel appears below the table showing:
   - Side-by-side ROS projections with diffs
   - SGP and wSGP
   - Projected standings: current team vs. team with the swap
5. A "Clear" button dismisses the comparison.

## Comparison Panel

### Stat Table

Side-by-side table with columns: Stat, Player A (roster), Player B (non-roster), Diff.

- **Hitter comparisons** show: ROS Rank, R, HR, RBI, SB, AVG, Total SGP, wSGP
- **Pitcher comparisons** show: ROS Rank, W, K, SV, ERA, WHIP, Total SGP, wSGP
- **Mixed type** (hitter vs pitcher): show both category sets with dashes for inapplicable stats — but this is an unusual case and doesn't need special handling beyond rendering blanks.

Diff column:
- Green text for improvements (higher is better for counting stats, lower for ERA/WHIP/rank)
- Red text for regressions
- The better value in each row gets bold weight

### Projected Standings

Two side-by-side standings tables: "Current Team" and "With [Player B]".

- Shows all 10 league teams sorted by total roto points
- User's team highlighted with accent color
- All 10 roto categories shown
- **Toggle** between "Roto Points" view (1-10 rankings per category) and "Stat Totals" view (raw projected stats). Default: Roto Points.
- In the "after" table, category point changes vs. before are annotated inline (e.g., `7 +1` in green, `5 -1` in red). In stat totals mode, show the raw delta instead.

## Data Flow

### Existing Data (no changes needed)

- `/api/players/browse` already returns all players with ROS stats, SGP, wSGP, positions, owner, rank. This powers the browse table and the stat comparison section.

### New API Endpoint

**`GET /api/players/compare?roster_player=<name>&other_player=<name>`**

Returns projected standings before/after the swap:

```json
{
  "before": {
    "stats": {"Team A": {"R": 750, ...}, ...},
    "roto": {"Team A": {"R_pts": 8, "total": 78}, ...}
  },
  "after": {
    "stats": {"Team A": {"R": 748, ...}, ...},
    "roto": {"Team A": {"R_pts": 8, "total": 79}, ...}
  },
  "categories": ["R", "HR", "RBI", "SB", "AVG", "W", "K", "SV", "ERA", "WHIP"]
}
```

### Backend Logic (new function in `season_data.py`)

**`compute_comparison_standings(roster_player_name, other_player_name, ...)`**

1. Load projected standings from cache (same source as trade standings).
2. Load the user's full roster from cache, plus the other player's ROS projections from the projections cache.
3. Compute "before" stats: `project_team_stats(roster)` for the user's team, other teams from projected standings as-is.
4. Compute "after" stats: rebuild the user's roster with the swap applied (remove roster player, add other player), then call `project_team_stats()` on the modified roster. This correctly handles rate stats (AVG, ERA, WHIP) which must be recomputed from components (H/AB, ER/IP, etc.), not added/subtracted.
5. Run `score_roto()` on both the before and after all-teams dicts to get roto point allocations.
6. Return both views.

This follows the same pattern as `compute_trade_standings_impact()` in season_data.py but uses `project_team_stats()` for correctness with rate stats.

## Frontend Implementation

All changes are in `players.html` (template + inline JS).

### Checkbox column
- Add a checkbox `<td>` as the first column in the browse table.
- Track selected players in a JS `Set` (by player name).
- Cap selection at 2. If a third is checked, uncheck the oldest.

### Comparison panel
- Hidden `<div id="comparison-panel">` below the table.
- When exactly 2 players are selected and one is roster / one is not:
  - Populate the stat table from the already-loaded `allPlayers` data (no API call needed for stats).
  - Fetch `/api/players/compare?roster_player=X&other_player=Y` for standings.
  - Render both sections.
- When selection is invalid (0-1 players, or both same ownership type):
  - Show a message: "Select one player from your roster and one free agent or opponent's player."

### Standings toggle
- Two small buttons or a toggle switch above the standings tables: "Roto Points" | "Stat Totals"
- Both datasets returned by the API; switching is purely client-side, no re-fetch.

## Files Modified

| File | Change |
|------|--------|
| `src/fantasy_baseball/web/templates/season/players.html` | Add checkbox column, comparison panel HTML, all JS logic |
| `src/fantasy_baseball/web/season_routes.py` | Add `/api/players/compare` route |
| `src/fantasy_baseball/web/season_data.py` | Add `compute_comparison_standings()` function |

## Testing

- Unit test for `compute_comparison_standings()`: verify that swapping a player correctly adjusts team stat totals and roto points.
- Existing tests for `score_roto()` and `project_team_stats()` already cover the scoring math.
