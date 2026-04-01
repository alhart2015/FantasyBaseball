# ROS Blending Fixes + Projected Standings Leverage

**Date:** 2026-04-01
**Branch:** `ros-blending-fixes`

## Problem

Two related data quality issues in the in-season pipeline:

### 1. Mixed ROS projection methodologies break the cross-system blend

The five projection systems use two different ROS methodologies:

| System | ROS type | Behavior |
|--------|----------|----------|
| Steamer | Full-season updated | PA stays ~constant, rates adjust with actual performance |
| The-Bat-X | Full-season updated | Same as Steamer |
| ZiPS | Remaining games only | PA decreases as season progresses |
| ATC | Remaining games only | Same as ZiPS |
| Oopsy | Remaining games only | Same as ZiPS |

`blend_projections()` averages counting stats across all systems per player. By mid-season, a remaining-games system might project ~22 HR for a player while a full-season system projects ~42 HR. The weighted average (~33 HR) is meaningless.

### 2. Leverage uses current standings instead of projected standings

`calculate_leverage()` computes category weights from gaps in current Yahoo standings. Early in the season, these gaps are noise. If you're behind in SB after one week but your roster projects to be strong in SB, leverage tells you to chase base stealers anyway.

The existing `season_progress` ramp partially mitigates this by blending toward uniform weights early season, but uniform weights aren't strategic -- they're just less wrong.

## Design

### Piece 1: Normalize remaining-games ROS projections to full-season

**Approach:** For remaining-games systems (ZiPS, ATC, Oopsy), add each player's actual accumulated stats from `game_logs` to their ROS projection before the cross-system blend. This normalizes all systems to full-season equivalents.

**System classification:** Hardcoded constant mapping system names to their ROS type. Based on empirical analysis comparing preseason vs ROS PA counts for Aaron Judge and Juan Soto across all five systems.

**Where in the code:** New normalization step in `load_ros_projections()` (in `db.py`), between loading individual system CSVs and calling `blend_projections()`. This function already has DB access for querying game logs.

**Implementation:**

1. New constant in `data/projections.py`:
   ```python
   FULL_SEASON_ROS_SYSTEMS: set[str] = {"steamer", "the-bat-x"}
   ```
   Systems not in this set are treated as remaining-games.

2. New function `normalize_ros_to_full_season(system_df, game_log_totals, player_type)`:
   - Takes a single system's ROS DataFrame (which has `mlbam_id` from the CSV)
   - Takes pre-queried game log season totals as a dict keyed by `mlbam_id`
   - For each player with a matching game log entry, adds actual counting stats to the ROS counting stats
   - Rate stats (avg, era, whip) are NOT touched here -- they get recomputed from components by the existing `_blend_hitters()` / `_blend_pitchers()` code after blending

3. New helper `get_season_totals(conn, season)` in `db.py`:
   - Queries `game_logs` for accumulated season totals per player: `SUM(pa), SUM(ab), SUM(h), SUM(r), SUM(hr), SUM(rbi), SUM(sb)` for hitters; `SUM(ip), SUM(k), SUM(er), SUM(bb), SUM(h_allowed), SUM(w), SUM(sv)` for pitchers
   - Returns `{mlbam_id: {stat: value}}` dict
   - Called once per refresh, reused across all systems

4. Modified `load_ros_projections()`:
   - After loading system CSVs (via `blend_projections`'s internal `load_projection_set`), normalize remaining-games systems before blending
   - This requires refactoring: either pass game_log_totals into `blend_projections()` via a new parameter, or pre-normalize the CSV files/DataFrames before `blend_projections` sees them

   **Preferred approach:** Add an optional `normalizer` callback to `blend_projections()`. After loading each system's DataFrame but before weighting, call `normalizer(system_name, hitters_df, pitchers_df)` if provided. `load_ros_projections` supplies a normalizer that adds actuals for remaining-games systems. This keeps `blend_projections` generic.

**Edge cases:**
- Player has no game log (rookie, injured, not yet played): leave ROS as-is
- Player in game_logs but missing `mlbam_id` in projections: fall back to normalized name match
- Game logs not yet populated (first run of season): skip normalization, blend proceeds as today

### Piece 2: Blended standings for leverage

**Approach:** New `blend_standings()` utility that interpolates between current and projected end-of-season standings. Updated `calculate_leverage()` that uses blended standings when projected standings are provided.

**`blend_standings(current, projected, progress)`** in `leverage.py`:
- Matches teams by name between the two standings lists
- For each stat category: `blended = progress * current + (1 - progress) * projected`
- Returns standings in the same `[{"name": str, "stats": dict, ...}]` format
- Teams present in only one list are included with their original stats

**`calculate_leverage()` changes:**
- New optional `projected_standings` parameter
- When provided: calls `blend_standings(current, projected, season_progress)`, then computes leverage from the blended result with full confidence (no uniform ramp)
- When `projected_standings` is None: current behavior preserved (uniform ramp fallback)

### Piece 3: Pipeline reorder in `season_data.py`

To use fresh projected standings for leverage, opponent rosters must be fetched earlier. Current order:

```
Step 4:  Load projections
Step 5:  Leverage (current standings only)
Step 6:  Match roster, compute wSGP
...
Step 11: Fetch opponent rosters
Step 12: Build projected standings
```

New order:

```
Step 4:  Load projections
Step 4b: Fetch opponent rosters (moved from Step 11)
Step 4c: Match all rosters to projections, build projected standings (moved from Step 12)
Step 5:  Leverage (with projected standings)
Step 6:  Match own roster, compute wSGP
...
Step 11: Trades (reuses already-fetched opponent rosters)
Step 12: (removed, absorbed into Step 4c)
```

Opponent roster fetching uses `ThreadPoolExecutor(max_workers=6)` for parallel Yahoo API calls (existing pattern). This adds ~5-10 seconds to the pipeline before leverage is computed, but the user has accepted this tradeoff for data quality.

### Caller updates

| Caller | Projected standings source | Change needed |
|--------|---------------------------|---------------|
| `season_data.py` | Fresh: built from opponent rosters + ROS projections in Step 4c | Pipeline reorder (Piece 3) |
| `summary.py` | Already builds `all_stats` via `project_team_stats` | Format into standings list, pass to `calculate_leverage` |
| `run_lineup.py` | Read dashboard cache if available, fall back to current-only | Add cache read, pass to `calculate_leverage` |
| `season_data.py` Step 11 (trade leverage) | Same projected standings from Step 4c | Wire through |

## Testing

- **Unit tests for `normalize_ros_to_full_season`:** Verify counting stats are added correctly, rate stats left alone, missing game logs handled gracefully
- **Unit tests for `blend_standings`:** Verify interpolation math, team name matching, edge cases (missing teams, progress=0, progress=1)
- **Unit test for `calculate_leverage` with projected standings:** Verify projected standings are used when provided, uniform ramp is bypassed
- **Integration test:** End-to-end pipeline produces different leverage weights with vs without projected standings
- **Regression:** Existing `calculate_leverage` tests still pass when `projected_standings` is None

## What this does NOT change

- The preseason projection blend (unaffected -- only ROS blend is normalized)
- The recency blend in `run_lineup.py` (that blends player projections with actual game log *rates* -- different from what we're doing here, which adds actual *totals* to remaining-games projections)
- Individual player wSGP calculation logic in `weighted_sgp.py`
- The SGP denominator or replacement level calculations
