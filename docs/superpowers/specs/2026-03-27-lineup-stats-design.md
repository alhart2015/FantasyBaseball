# Lineup Stats with Performance vs Pace Color Coding

**Date:** 2026-03-27
**Status:** Approved

## Problem

The lineup page shows player names, positions, wSGP, and games but no actual stats. There's no way to see how players are performing relative to expectations without leaving the dashboard.

## Solution

Add season-to-date stat columns to the hitters and pitchers tables on the lineup page, color-coded by performance relative to pre-season projection pace. Tooltips on each stat cell show projection details.

## Data Sources

All data already exists in SQLite:

- **Pre-season projections:** `blended_projections` table — full-season counting stats and rates per player
- **Actual stats:** `game_logs` table — per-game stats, summed for season-to-date totals
- **Stat variance:** `STAT_VARIANCE` in `constants.py` — SD of actual/projected ratio residuals, calibrated from 2022-2024 data

## Stat Columns

**Hitters:** PA, R, HR, RBI, SB, AVG (PA is context/sample-size, not color-coded)

**Pitchers:** IP, W, K, SV, ERA, WHIP (IP is context/sample-size, not color-coded)

## Pace Calculation

### Counting stats (R, HR, RBI, SB, H, W, K, SV)

Scale pre-season projection by opportunity consumed:

For hitters: `expected_stat = projected_stat * (actual_pa / projected_pa)`
For pitchers: `expected_stat = projected_stat * (actual_ip / projected_ip)`

PA and IP directly measure opportunity — this accounts for IL stints, rest days, and varying usage patterns without needing to count games.

### Rate stats (AVG, ERA, WHIP)

Compare actual rate directly to projected rate. No pace scaling needed — rates don't accumulate.

### Z-score

**Counting stats:**
```
ratio = actual / expected
z_score = (ratio - 1.0) / STAT_VARIANCE[stat]
```

If `expected == 0`: show neutral color (cannot compute meaningful z-score). Do not attempt the ratio.

**Rate stats — component stat mapping:**

| Rate stat | Component stat for variance | Formula |
|-----------|----------------------------|---------|
| AVG | `h` (0.103) | `z = (actual_avg - proj_avg) / (STAT_VARIANCE["h"] * proj_avg)` |
| ERA | `er` (0.252) | `z = (actual_era - proj_era) / (STAT_VARIANCE["er"] * proj_era)` |
| WHIP | `h_allowed` (0.143) | `z = (actual_whip - proj_whip) / (STAT_VARIANCE["h_allowed"] * proj_whip)` |

For WHIP, `h_allowed` is used as the component stat because hits allowed are the dominant contributor to WHIP variance (BB variance is higher at 0.257 but BB is typically the smaller component). This is an approximation — a combined variance could be more precise but the visual color coding is a rough guide, not an exact statistical test.

For inverse stats (ERA, WHIP — lower is better), negate the z-score so green = good (outperforming = lower than projected).

### Sample size thresholds

| Condition | Behavior |
|-----------|----------|
| 0 PA / 0 IP (no games played) | Show dashes, neutral styling |
| < 10 PA (hitters) or < 5 IP (pitchers) | Show actuals, neutral color for all stats |
| 10-29 PA (hitters) or 5-9 IP (pitchers) | Color counting stats, neutral for rate stats |
| >= 30 PA (hitters) or >= 10 IP (pitchers) | Color all stats |

## Color Coding

Five levels based on z-score, with both text color and background tint:

| Z-score | Class | Text Color | Background |
|---------|-------|------------|------------|
| > +1.0 SD | `stat-hot-2` | Bright green (#22c55e) | Green tint (15% opacity) |
| +0.5 to +1.0 SD | `stat-hot-1` | Light green (#86efac) | Green tint (8% opacity) |
| -0.5 to +0.5 SD | `stat-neutral` | Default (#ccc) | None |
| -1.0 to -0.5 SD | `stat-cold-1` | Light red (#fca5a5) | Red tint (8% opacity) |
| < -1.0 SD | `stat-cold-2` | Bright red (#ef4444) | Red tint (15% opacity) |

PA and IP columns are always neutral (sample-size context only).

## Tooltip

On hover over any stat cell, show a tooltip with:

```
Player Name — Stat
─────────────────
Actual          12
Expected pace   11.4
Z-score         +0.3
─────────────────
Pre-season proj 72 R
ROS proj        Coming soon
```

- **ROS projection** is a placeholder for now — will show "Coming soon" with no arrow
- When ROS data is available later, it will show the ROS value and an up/down arrow comparing ROS to pre-season

## Architecture

### New module: `src/fantasy_baseball/analysis/pace.py`

Isolates the pace/z-score computation for testability:

```python
def compute_player_pace(
    actual_stats: dict,      # season-to-date from game_logs (lowercase keys: "r", "hr", etc.)
    projected_stats: dict,   # full-season from blended_projections (lowercase keys)
    player_type: str,        # "hitter" or "pitcher"
) -> dict:
    """Compute z-scores and color classes for each roto stat.

    Keys in input dicts use lowercase names matching game_logs columns
    and STAT_VARIANCE keys (e.g., "r", "hr", "rbi", "sb", "h", "ab", "pa").

    Returns dict with per-stat entries using UPPERCASE display keys:
    {
        "PA": {"actual": 102, "color_class": "stat-neutral"},
        "HR": {"actual": 5, "expected": 3.4, "z_score": 0.8, "color_class": "stat-hot-1",
               "projection": 22},
        ...
    }
    """
```

### Refresh pipeline changes: `season_data.py`

In `run_full_refresh()`, after matching roster to projections (Step 6) and before writing the roster cache:

1. Open a new DB connection (the Step 4 connection is already closed)
2. Bulk-load all game logs for the season in two queries:
   ```sql
   SELECT name, SUM(r) as r, SUM(hr) as hr, SUM(rbi) as rbi, SUM(sb) as sb,
          SUM(h) as h, SUM(ab) as ab, SUM(pa) as pa, COUNT(*) as games
   FROM game_logs WHERE season = ? AND player_type = 'hitter'
   GROUP BY name
   ```
   (Similar for pitchers with ip, k, w, sv, er, bb, h_allowed)
3. Build a lookup dict keyed by `normalize_name(name)` for efficient matching
4. For each roster player, look up actuals by normalized name, call `compute_player_pace()`
5. Attach the pace data as `entry["stats"]` to each player entry in the roster cache
6. Close DB connection

This uses 2 queries total (not N per-player queries) and matches names via `normalize_name()` in Python, consistent with how `match_roster_to_projections` works.

### Roster cache changes

Each player entry gains a `"stats"` dict:

```json
{
  "name": "Juan Soto",
  "wsgp": 3.12,
  "stats": {
    "PA": {"actual": 102, "color_class": "stat-neutral"},
    "R": {"actual": 19, "expected": 14.2, "z_score": 1.6, "color_class": "stat-hot-2", "projection": 95},
    "HR": {"actual": 9, "expected": 4.8, "z_score": 1.6, "color_class": "stat-hot-2", "projection": 35}
  }
}
```

### Template changes: `lineup.html`

- Replace current 6-column hitter table (Slot, Player, Elig, Games, wSGP, Status) with: Slot, Player, Elig, PA, R, HR, RBI, SB, AVG, wSGP, Status
- Replace current 5-column pitcher table with: Slot, Player, IP, W, K, SV, ERA, WHIP, wSGP, Status
- Each stat cell gets `class="{{ stat.color_class }}"` and a tooltip `<div>`
- CSS for the 5 color classes + tooltip styling added to the template's `<style>` block

### `format_lineup_for_display()` changes

Pass through the `stats` dict from the roster cache to the template data. No computation here — just forwarding.

## What doesn't change

- wSGP calculation (still there, still works)
- Suggested moves / optimize button
- Probable starters section
- Game log fetching (still triggered by "Fetch MLB Data" button)
- Refresh pipeline structure (new step inserted, doesn't alter existing steps)

## Edge cases

- **Player with no game logs:** Show dashes, neutral styling (no z-score to compute)
- **Player not matched to projections:** Show actuals if available, no color coding (no projection to compare against)
- **Division by zero:** If projected_pa or projected_ip is 0, show neutral
- **Rate stats with 0 AB or 0 IP:** Show dash instead of computing 0/0
- **Expected == 0 for counting stats:** Show neutral (not red — scoring when projected for 0 is not underperformance)

## Mockup Reference

See `.superpowers/brainstorm/1556-1774623635/lineup-stats-mockup.html` for the approved visual design.
