# Overall Pace Marker on Slot Column

## Summary

Add an overall pace indicator to each player's Slot column in the lineup view. The Slot cell (e.g., "SS", "OF", "DH") gets colored green/red based on the average of all per-category z-scores, giving a quick at-a-glance signal of whether a player is ahead or behind pace overall.

## Computation

- Average all per-category z-scores equally:
  - **Hitters**: R, HR, RBI, SB, AVG
  - **Pitchers**: W, K, SV, ERA, WHIP
- Skip categories where z-score is `None` (missing projection data), but include all others regardless of significance threshold
- No minimum PA/IP gate — color shows from day one
- Apply existing color thresholds to the average:
  - `avg_z >= 2.0` → `stat-hot-2` (bright green)
  - `avg_z >= 1.0` → `stat-hot-1` (light green)
  - `avg_z <= -2.0` → `stat-cold-2` (bright red)
  - `avg_z <= -1.0` → `stat-cold-1` (light red)
  - Otherwise → `stat-neutral` (no color)

## Display

- Slot column `<td>` receives the computed color class
- Tooltip on hover shows the average z-score (e.g., "+1.4" or "-0.8")
- Per-category stat cell coloring remains unchanged

## Implementation

### `pace.py` — new function

```python
def compute_overall_pace(pace_dict: dict) -> dict:
    """Average per-category z-scores into an overall pace summary."""
```

Returns `{"avg_z": float | None, "color_class": str}`.

### `season_data.py` — wire it up

Call `compute_overall_pace(player.pace)` during player data assembly. Add `overall_pace` key to the player dict passed to the template.

### `lineup.html` — apply color

Apply `overall_pace.color_class` to the Slot `<td>`. Add a `title` attribute with the formatted avg z-score.

## What doesn't change

- Per-category z-score computation in `compute_player_pace()`
- Existing CSS classes (`stat-hot-2`, `stat-hot-1`, `stat-neutral`, `stat-cold-1`, `stat-cold-2`)
- Significance markers on individual stat cells
- Optimizer logic, wSGP, leverage weights
