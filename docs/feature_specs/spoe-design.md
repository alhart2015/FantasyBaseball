# Standings Points Over Expected (SPOE) — Design Spec

## Problem

In a roto league, standings reflect cumulative stats — but some teams are overperforming their rosters while others are underperforming. There's no way to quantify how much of a team's standings position is skill vs. luck. SPOE measures this by comparing what each team's roster *should* have produced (based on projections) to what it actually produced.

## Overview

For each week of the season, we project how many stats each team's roster should accumulate, sum those projected stats over time, convert to roto points, and compare to the actual roto standings. The difference is SPOE: positive means lucky, negative means unlucky.

## Decisions

| Decision | Choice | Rationale |
|----------|--------|-----------|
| Scope | All 10 teams, per-category detail | Full league view is more useful than just user's team |
| Projection source | 5-system blended (Steamer, ZiPS, ATC, The Bat X, Oopsy) | Consistency with all other calculations in the project |
| Games per week | Flat 6.5 assumption | Signal (luck) >> noise (schedule variance) |
| Weekly scaling | ROS projection × (7 / days_remaining_in_season) | Correct fraction of remaining projection consumed per week |
| Roster source | `weekly_rosters` table (all teams, saved by dashboard refresh) | Already populated automatically; 2 weeks of data exist |
| Actual stats source | `standings` table (cumulative per-category) | Already populated; no need to aggregate game logs per team |
| Rate stat accumulation | Accumulate components (H, AB, ER, IP, etc.), recompute rates at scoring time | Same approach as `project_team_stats()` |
| Storage | SQLite `spoe_results` table | Enables future time-series visualization; no Redis cache needed |
| Recomputation | Only current week; completed weeks are frozen | Inputs for completed weeks don't change |
| Snapshot date | Monday of scoring week (existing behavior) | Matches current `append_roster_snapshot` behavior |
| Output | New dashboard tab with team table + expandable per-category rows | Consistent with existing dashboard UX |

## Data Model

### New SQLite table: `spoe_results`

```sql
CREATE TABLE IF NOT EXISTS spoe_results (
    snapshot_date  TEXT NOT NULL,    -- Monday of the week
    team           TEXT NOT NULL,
    category       TEXT NOT NULL,    -- "R", "HR", ..., "WHIP", "SV", "total"
    projected_stat REAL,            -- accumulated projected stat value through this week
    actual_stat    REAL,            -- accumulated actual stat value through this week
    projected_pts  REAL,            -- roto points from projected stats
    actual_pts     REAL,            -- roto points from actual stats
    spoe           REAL,            -- actual_pts - projected_pts
    PRIMARY KEY (snapshot_date, team, category)
);
```

Each row = one team × one category × one week. The `"total"` category stores the sum across all 10 roto categories. Both stats and points are stored for debuggability — you can see *why* a team is lucky (e.g., 2.94 ERA vs. projected 3.80).

## Algorithm

### Core function: `compute_spoe(db_conn, config)`

New module: `src/fantasy_baseball/analysis/spoe.py`

```
Input:
  - SQLite connection (weekly_rosters, standings, game_logs tables)
  - League config (season_year, projection systems, season_end date, etc.)
  - ROS projection snapshots on disk (data/projections/{year}/ros/{date}/)

For each team:
    projected_components = {H: 0, AB: 0, R: 0, HR: 0, RBI: 0, SB: 0,
                            IP: 0, ER: 0, BB: 0, H_allowed: 0, W: 0, K: 0, SV: 0}

weeks = all distinct snapshot_dates from weekly_rosters for the current season

For each week W (by snapshot_date, chronological):

    If W already has rows in spoe_results AND W is not the current week:
        Load projected_components from stored projected_stat values
        Continue to next week    # frozen — don't recompute

    monday = snapshot_date for week W

    1. LOAD ROSTERS
       Query weekly_rosters for all teams where snapshot_date = monday
       Build {team_name: [{name, positions}, ...]}

    2. SELECT ROS PROJECTIONS
       Find latest ROS snapshot directory with date <= monday
       Blend 5 projection systems into (hitters_df, pitchers_df)

    3. NORMALIZE FULL-SEASON SYSTEMS
       For Steamer and The Bat X:
         Query game_logs for actual stats accumulated before monday
         Subtract actual stats from projections to get remaining-season stats
       ZiPS, ATC, Oopsy: use as-is (already pure ROS)

    4. PROJECT WEEKLY STATS PER TEAM
       days_remaining = (season_end - monday).days
       weekly_fraction = 7 / days_remaining

       For each team:
         Match roster player names to blended projections (normalized name matching)
         For each matched player:
           weekly_counting_stats = player_ros_stats × weekly_fraction
           (Scale component stats too: H, AB, IP, ER, BB, H_allowed)
         Sum across all players on roster → team weekly projected components
         Add to projected_components[team]

    5. COMPUTE PROJECTED RATES FROM ACCUMULATED COMPONENTS
       For each team:
         projected_stats = {
           R, HR, RBI, SB, W, K, SV: from accumulated components directly,
           AVG: accumulated_H / accumulated_AB,
           ERA: 9 * accumulated_ER / accumulated_IP,
           WHIP: (accumulated_BB + accumulated_H_allowed) / accumulated_IP
         }

    6. LOAD ACTUAL STATS
       Query standings table for snapshot_date = monday (or nearest)
       Build {team_name: {R: val, HR: val, ...}} from cumulative stats

    7. SCORE ROTO
       projected_roto = score_roto(projected_stats for all teams)
       actual_roto = score_roto(actual_stats for all teams)

    8. COMPUTE AND STORE SPOE
       For each team, for each category:
         spoe = actual_roto_pts - projected_roto_pts
       Also compute "total" = sum of all category SPOEs
       INSERT OR REPLACE into spoe_results
```

### Recovering accumulated components for frozen weeks

When a completed week is skipped (step: "Continue to next week"), we still need `projected_components` to be correct for the next week's accumulation. Two options:

- **Option A:** Store components in `spoe_results` alongside the final stats. This adds columns but is explicit.
- **Option B:** Store components as a separate JSON blob in a `spoe_components` table or column.

We'll go with **storing the component stats in `projected_stat`** for the component categories. Specifically, the `projected_stat` column for counting stats (R, HR, RBI, SB, W, K, SV) already *is* the accumulated component. For rate stats (AVG, ERA, WHIP), the `projected_stat` column stores the computed rate — but we also need the components. We'll add a `spoe_components` table:

```sql
CREATE TABLE IF NOT EXISTS spoe_components (
    snapshot_date  TEXT NOT NULL,
    team           TEXT NOT NULL,
    component      TEXT NOT NULL,   -- "H", "AB", "IP", "ER", "BB", "H_allowed"
    value          REAL NOT NULL,
    PRIMARY KEY (snapshot_date, team, component)
);
```

When resuming from a frozen week, load the latest `spoe_components` row to seed `projected_components`.

## Existing Code Reused

| Module | What we reuse |
|--------|--------------|
| `data/projections.py` | `blend_projections()`, `FULL_SEASON_ROS_SYSTEMS`, ROS loading |
| `data/projections.py` | `match_roster_to_projections()` (need variant for stored roster data) |
| `scoring.py` | `project_team_stats()` (pattern for rate stat computation), `score_roto()` |
| `data/db.py` | `get_connection()`, `create_tables()`, game log queries |
| `utils/name_utils.py` | `normalize_name()` for roster-to-projection matching |
| `utils/constants.py` | `ALL_CATEGORIES`, `RATE_STATS`, `INVERSE_STATS` |

## New Code

| File | Purpose |
|------|---------|
| `src/fantasy_baseball/analysis/spoe.py` | Core SPOE computation engine |
| `src/fantasy_baseball/data/db.py` | Schema additions (`spoe_results`, `spoe_components` tables) |
| `src/fantasy_baseball/web/season_data.py` | Call SPOE computation during refresh; read results for dashboard |
| `src/fantasy_baseball/web/season_routes.py` | New API endpoint for SPOE data |
| Dashboard frontend (HTML/JS) | New "Luck" tab with team table and expandable category rows |
| `tests/test_analysis/test_spoe.py` | Unit tests for the SPOE engine |

## Dashboard UI

### Main view: Team SPOE table

New tab labeled "Luck" (or "SPOE") in the dashboard nav.

Table sorted by actual standings rank:

| # | Team | Actual Pts | Projected Pts | SPOE |
|---|------|-----------|---------------|------|
| 1 | Jon's Underdogs | 78 | 71 | **+7.0** |
| 2 | Hart of the Order | 72 | 75 | **-3.0** |
| ... | | | | |

- Positive SPOE (lucky) = green
- Negative SPOE (unlucky) = red
- Rows are expandable (click to drill down)

### Expanded view: Per-category breakdown

| Category | Proj. Stat | Actual Stat | Proj. Pts | Actual Pts | SPOE |
|----------|-----------|-------------|-----------|------------|------|
| R | 52 | 67 | 4.0 | 7.0 | +3.0 |
| HR | 11 | 12 | 6.0 | 6.0 | 0.0 |
| ERA | 3.80 | 2.94 | 5.0 | 8.0 | +3.0 |
| ... | | | | | |

Shows where the luck is coming from — which categories are over/under performing.

## Edge Cases

1. **Missing roster snapshot for a week:** Skip that week. Accumulated projected stats carry forward unchanged.
2. **Missing ROS projection snapshot:** Fall back to the most recent available snapshot.
3. **Missing standings snapshot:** Skip that week — can't compute SPOE without actuals. Roster and standings snapshots are saved in the same refresh call, so in practice they always exist together.
4. **Player on roster but not in projections:** Exclude from projected stats. Their actual contribution still counts in standings, so this shows as positive SPOE (conservative).
5. **IL players:** Included if on the roster snapshot. Projection systems handle injuries by projecting reduced/zero stats.
6. **First week:** `projected_components` starts at zero for all teams.
7. **Mid-season first run:** Backfills all historical weeks, then only recomputes current week on subsequent runs.

## Tests

### Roster state tests
- Player A on roster week 1, dropped before week 2 → contributes to projected stats in week 1 only
- Player B picked up week 2 → contributes from week 2 onward, not week 1

### ROS projection selection tests
- Running for week of 4/8 with snapshots from 4/4, 4/6, 4/10 → selects 4/6
- No snapshot before the target date → falls back to earliest available

### Projection normalization tests
- Pure ROS systems (ZiPS, ATC, Oopsy) pass through unchanged
- Full-season systems (Steamer, The Bat X) have actual stats subtracted

### Roto scoring tests
- Projected stats and actual stats produce correct roto point assignments
- Tie-breaking works correctly (fractional points)

### SPOE computation tests
- Team with actual stats matching projections exactly → SPOE = 0
- Team outperforming in one counting category → positive SPOE in that category
- Rate stat accumulation: components accumulate correctly, rates recomputed at scoring time

### Incremental computation tests
- Completed weeks are not recomputed on subsequent runs
- Current week is always recomputed
- Backfill correctly processes all historical weeks on first run
