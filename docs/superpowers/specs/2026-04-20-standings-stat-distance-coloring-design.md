# Standings Stat-Distance Coloring

## Overview

Replace the five-bucket rank-based coloring on the standings page with a continuous red-to-green gradient driven by each team's distance from the category leader and trailer. Cells clustered at the top of a category (e.g., four teams within 1 HR of the leader) should all appear in the bright-green end of the spectrum rather than sliding through progressively darker buckets.

## Motivation

The current coloring in `src/fantasy_baseball/web/season_data.py:252-263` assigns a CSS class by rank position: ranks 1-2 → `rank-top` (bright green), 3-4 → `rank-high`, 5-6 → `rank-mid` (neutral), 7-8 → `rank-low`, 9-10 → `rank-bottom` (bright red). This misrepresents clusters. When four teams sit at 100/99/99/99 HR, only the first two render as bright green even though the third and fourth are tied with the second. Conversely, when a category spreads evenly across the 10 teams, the current display looks identical to the clustered case, hiding that information.

Coloring by actual stat distance surfaces how close categories are to gaining or losing roto points, which is the question a user is trying to answer when scanning the standings.

## Scope

Apply stat-distance coloring to:

- **Current view** — the "Roto Points / Stat Totals" table in `standings.html`, per-category cells and the Total column.
- **Projected / Preseason ERoto** — per-category cells and the Total column.
- **Projected / Current ERoto (ROS)** — per-category cells and the Total column.

**Not in scope:** the Monte Carlo view (structurally different — its cells are Median/P10/P90/1st%/Top3% per team, not per category). Its existing conditional `cat-top` / `cat-bottom` cues on the user-team row stay as-is.

## Coloring algorithm

For each category, across all 10 teams:

```
t = (value − min) / (max − min)        # linear position in [0, 1]
# For rate stats where lower is better (ERA, WHIP): t = 1 − t
intensity = 2 * t − 1                  # signed intensity in [−1, 1]
```

Where:

- `intensity = +1.0` — category leader (brightest green).
- `intensity = 0.0` — exactly halfway between leader and trailer (neutral, no background tint).
- `intensity = −1.0` — category trailer (brightest red).
- Tied-category case (`max == min`, every team equal in the category): the category key is omitted from `color_intensity` for every team, and the template falls back to a neutral cell.

The Total column uses the same formula applied to each team's total roto points (`team.roto_points["total"]`), leader-is-green (no inversion).

`ALL_CATEGORIES` and `INVERSE_CATS` already exist in `src/fantasy_baseball/scoring.py`; reuse `INVERSE_CATS` to drive the ERA/WHIP flip so this code does not re-encode the list of inverse categories.

## Data shape

In `season_data.py`, the per-team dict currently carries `color_classes: dict[str, str]`. Replace with:

```python
team["color_intensity"] = {
    "R":    float,        # signed, in [−1, 1]
    "HR":   float,
    ...
    "WHIP": float,
    "total": float,       # Total column
}
# Any category where all 10 teams tie is omitted from this dict (or stored as None).
```

Omitting a tied category from the dict is the simplest convention: the template only emits the `--intensity` style if a value exists.

All three table-building code paths in `season_data.py` — `build_standings_table` for the Current view, and the preseason / current-ERoto builders used by the Projected view — need to produce `color_intensity` identically. Factor the computation into one helper (e.g., `_compute_color_intensity(teams_with_stats, categories)`) so the three call sites stay aligned.

## Rendering

The template emits the intensity as a CSS custom property on the cell:

```html
<td class="stat-cell" style="--intensity: {{ team.color_intensity[cat] }}" ...>
```

Cells with no `style` attribute (tied-category cells) render neutral.

Colors stay in CSS so the palette lives in one place. In `season.css`, replace the five `rank-*` rules with two rules keyed on the sign of the intensity, using `color-mix` (widely supported in modern browsers) to interpolate between a transparent baseline and the green / red endpoints:

```css
/* Positive intensity → green, negative → red. Alpha scales with |intensity|. */
td.stat-cell {
  background: color-mix(in srgb, transparent,
    var(--standings-tint, transparent) calc(abs(var(--intensity, 0)) * 30%));
}
/* --standings-tint flips between green and red via a sign-selector below. */
td.stat-cell[style*="--intensity: -"] { --standings-tint: #ef4444; }  /* red half */
td.stat-cell                          { --standings-tint: #22c55e; }  /* green half default */
```

The `[style*="--intensity: -"]` attribute selector is a pragmatic way to discriminate sign without extra markup. If this proves too brittle during implementation (e.g., the leading minus is sometimes formatted differently), the fallback is to have the server emit two separate CSS variables (`--pos-intensity` and `--neg-intensity`, each in `[0, 1]` with the unused one `0`) and interpolate against both. The plan can pick whichever is cleaner once the implementer has the actual rendered output in hand.

Text color (`color:`) on the cell keeps the existing approach — darker foreground on low-intensity cells, brighter green/red foreground on the extremes. A simple rule: use the CSS custom property to pick one of two foreground colors by sign, scaled similarly. Exact values are a polish pass — match the existing visual weight of `rank-top` / `rank-bottom` at `|intensity| = 1.0`.

## Code touchpoints

- `src/fantasy_baseball/web/season_data.py` — replace the rank-bucket block in `build_standings_table` (~lines 251-263) with a call to the new `_compute_color_intensity` helper. Apply the same helper in the preseason / current-ERoto builders so the Projected tab matches.
- `src/fantasy_baseball/web/templates/season/standings.html` — change the template cells to emit `style="--intensity: …"` instead of `class="{{ team.color_classes[cat] }}"`. Update both the main Current table and the `eroto_table` macro for Projected. Add coloring for the Total column cells.
- `src/fantasy_baseball/web/static/season.css` — delete the five `td.rank-*` rules (lines 193-198) and add the `color-mix` gradient rules described above.

## Tests

Extend `tests/test_web/test_season_data.py`:

- Category leader receives `intensity == 1.0`; category trailer receives `intensity == −1.0`.
- ERA and WHIP are flipped — the team with the lowest ERA gets `intensity == 1.0`.
- When all 10 teams are tied in a category (e.g., every team has `SV == 0` on opening day), the category key is absent from every team's `color_intensity` dict.
- Clustered scenario: values `[100, 99, 99, 99, 70, 65, 55, 50, 45, 40]` → the three 99s all get the same intensity (≈ `+0.97`), not spread across different intensities.
- Total-column intensity tracks total roto points (leader = `+1.0`, trailer = `−1.0`).

No CSS-level visual regression test — the change is style-only for three tables and is verified by eye in the dev dashboard.

## Verification

Load the season dashboard locally, open `/standings`, and walk through each tab:

- Current (Roto Points and Stat Totals toggles)
- Projected → Preseason (both toggles)
- Projected → Current ERoto (both toggles)

Confirm clustered categories show uniform bright color, and that ERA/WHIP render the low-value teams as green. Monte Carlo view should render unchanged.

## Out of scope

- Redesigning the Monte Carlo table coloring.
- Changing the standings page's information architecture (gap-to-adjacent-team annotations, leverage callouts). Those remain on the TODO under "Standings page visual redesign" as a separate, larger piece of work.
- Tuning the exact palette. The initial implementation uses the same `#22c55e` / `#ef4444` anchor colors as the current `rank-*` rules.
