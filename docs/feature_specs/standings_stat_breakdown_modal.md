# Projected standings: per-player stat breakdown modal

**Status:** Spec (2026-04-22)
**Scope:** `src/fantasy_baseball/scoring.py`, `src/fantasy_baseball/web/refresh_pipeline.py`, `src/fantasy_baseball/data/cache_keys.py`, `src/fantasy_baseball/web/season_data.py`, `src/fantasy_baseball/web/season_routes.py`, `src/fantasy_baseball/web/templates/season/standings.html`.

## Problem

The Projected → Current → Stat Totals view shows each team's projected R/HR/RBI/SB/AVG/W/K/SV/ERA/WHIP totals for the rest of the season. A manager looking at those numbers cannot currently see *why* the total is what it is — which players are contributing, which are on the bench or IL, and which are being scaled down by displacement. That visibility matters both for self-assessment ("my HR total dropped — which player's projection fell?") and for opponent scouting ("why is that team's ERA so low?"). It is also a useful data-quality check: if an opponent's projection has many players with missing ROS data, their standing is artificially deflated and the user's relative position is overstated.

## Feature

Each team's projected stat cell in the Current → Stat Totals sub-view becomes clickable. On click, a modal opens showing that team's per-player breakdown for the clicked stat: every rostered player, their contribution (scaled for displacement), their scale factor, and a status tag explaining why they're contributing at the level shown.

## UX

**Trigger:** left-click on a stat cell in `#proj-current` table when the sub-view is in "Stat Totals" mode. Column-header clicks and cells in "ERoto Points" mode remain non-interactive.

**Visual affordance:** cells in stat-totals mode get a subtle pointer cursor and hover highlight. ERoto-points mode cells do not (the existing `toggleProjMode` JS is extended to toggle the affordance alongside the mode).

**Modal contents:**
- Header: `{team_name} — {stat} breakdown` (e.g., `Skelethor — SB breakdown`)
- Subheader: the team's total for that stat, matching the value displayed in the table cell
- Body: a table of per-player rows (see [Modal columns](#modal-columns))
- Dismiss: X button, click-outside, and ESC key

**Modal columns:**

| Stat | Columns | Player rows shown |
|---|---|---|
| R, HR, RBI, SB | Player / Status / Contribution / Scale% | all rostered hitters |
| W, K, SV | Player / Status / Contribution / Scale% | all rostered pitchers |
| AVG | Player / Status / H / AB / Scale% | all rostered hitters |
| ERA | Player / Status / ER / IP / Scale% | all rostered pitchers |
| WHIP | Player / Status / BB / H / IP / Scale% | all rostered pitchers |

Player type filter: HR/R/RBI/SB/AVG modals show hitters only; W/K/SV/ERA/WHIP modals show pitchers only. A hitter contributes 0 to K by construction — omitting them is removing structural noise, not hiding information.

Row order: contributing rows first, sorted by contribution descending; then 0-contribution rows (bench, no-projection) sorted by name.

Counting-stat contribution cells show `raw * scale_factor` rounded to 1 decimal. Rate-stat component cells show the scaled components (`raw_h * factor`, `raw_ab * factor`) rounded to 1 decimal. The team AVG/ERA/WHIP computed from the shown components should match the table cell value to within rounding.

## Status tags

Each row carries a status tag explaining why its scale factor is what it is:

| Tag | Meaning | Typical scale factor |
|---|---|---|
| `ACTIVE` | Player is in an active slot; counts at face value | 1.00 |
| `IL_FULL` | Player is in an IL slot or on BN with Yahoo IL status; counts at full ROS for displacement math | 1.00 |
| `DISPLACED` | Player is in an active slot but was picked as a displacement target by an IL-classified teammate; scaled down | (0, 1) |
| `BENCH` | Player is on BN with no IL status; excluded from the team total | 0.00 |
| `NO_PROJECTION` | Player's `rest_of_season` is `None` (data-quality miss); contributes 0 | n/a — display as `—` |

The `NO_PROJECTION` tag is the one the user specifically wants to be visible: surfacing missing-projection rows on opponents reveals when a team's total is understated due to hydration gaps.

## Data shape

New module-level dataclasses in `scoring.py` (co-located with `_apply_displacement` since they're derived from its classification):

```python
class ContributionStatus(StrEnum):
    ACTIVE = "active"
    IL_FULL = "il_full"
    DISPLACED = "displaced"
    BENCH = "bench"
    NO_PROJECTION = "no_projection"


@dataclass(frozen=True)
class PlayerContribution:
    name: str
    player_type: PlayerType
    status: ContributionStatus
    scale_factor: float              # 0.0 to 1.0; n/a for NO_PROJECTION → 0.0
    raw_stats: dict[str, float]      # pre-scale ROS values; empty for NO_PROJECTION


@dataclass(frozen=True)
class RosterBreakdown:
    team_name: str
    hitters: list[PlayerContribution]
    pitchers: list[PlayerContribution]
```

New function:

```python
def compute_roster_breakdown(team_name: str, roster: list[Player]) -> RosterBreakdown:
    """Partition roster via the same rules as _apply_displacement and
    return per-player contributions with status tags and scale factors.
    """
```

`compute_roster_breakdown` reuses the slot-first classification that `_apply_displacement` already performs (refactored as a shared helper if necessary — same classification, two outputs). Displacement factors are produced by the same inner loop; the breakdown just preserves the per-player view that `project_team_stats` currently aggregates away.

**Invariant (tested):** for each category, summing `raw_stats[cat] * scale_factor` across the breakdown matches `project_team_stats(roster, displacement=True)[cat]` to within floating-point tolerance. For rate stats, summing the components first and computing the rate from the totals matches.

## Data delivery

All team breakdowns are computed during the refresh pipeline's `_build_projected_standings` step (alongside `ProjectedStandings.from_rosters`) and written to a new cache key.

```python
class CacheKey(StrEnum):
    ...
    STANDINGS_BREAKDOWN = "standings_breakdown"
```

Payload shape:

```json
{
    "effective_date": "YYYY-MM-DD",
    "teams": {
        "<team_name>": {
            "hitters": [<PlayerContribution dicts>],
            "pitchers": [<PlayerContribution dicts>]
        },
        ...
    }
}
```

`season_data.py` loads this cache and passes it to the standings template. The template embeds it as `<script type="application/json" id="breakdown-data">` inside the `#proj-current` div, gated to `{% if current_projected %}` so the block is absent when ROS projections are unavailable.

No new HTTP endpoint. No on-demand fetch. Click-to-render reads from the embedded JSON.

## JS behavior

One new handler attached to `#proj-current` via event delegation:

```javascript
// Attach on DOMContentLoaded. Event delegation on the table.
document.getElementById('proj-current').addEventListener('click', function(e) {
    const cell = e.target.closest('td.proj-current-cell');
    if (!cell) return;
    if (!cell.classList.contains('mode-stats')) return;  // gated to stat-totals mode
    const teamName = cell.closest('tr').dataset.team;
    const cat = cell.dataset.cat;
    openBreakdownModal(teamName, cat);
});
```

The existing `toggleProjMode(el, selector)` is extended to add/remove `mode-stats` on the affected cells so the click handler can cheaply gate behavior without consulting other DOM state.

`openBreakdownModal(teamName, cat)` reads the embedded JSON, picks `teams[teamName]`, filters to the relevant player type for the stat, renders the table, and shows the modal. Dismiss handlers wire up ESC, click-outside, and the close button.

## Row data source for team name

The `eroto_table` macro adds a `data-team` attribute to each `<tr>`:

```jinja
<tr class="{% if team.is_user %}user-team{% endif %}" data-team="{{ team.name }}">
```

This is the only template change required outside the modal markup and header/click wiring.

## Rate-stat math notes

For AVG/ERA/WHIP, the modal must display values that tie to the table cell. Two non-obvious points:

1. **Scaled components, not scaled rates.** A player's rate doesn't change under displacement — their counts do. If Buxton is scaled to 60%, his H is `raw_h * 0.6` and his AB is `raw_ab * 0.6`; his personal AVG is unchanged. The modal shows the scaled components (not the rate) so the team AVG displayed in the subheader equals `sum(scaled_h) / sum(scaled_ab)`.
2. **`NO_PROJECTION` players contribute nothing to either numerator or denominator.** Their row shows `—` for components and `—` for scale, and they don't perturb the team rate.

## Non-goals

- No drilldown on Preseason sub-tab, Monte Carlo view, Current view (non-projected), or Projected → Current → ERoto Points mode.
- No interactive comparison between modals (open one at a time).
- No historical / time-series breakdown.
- No click-to-edit (the modal is read-only).
- No search / filter inside the modal; roster sizes are small enough that scrolling is fine.

## Rollout

Single feature branch, single PR. No feature flag. Cache-key addition is additive — the write path is new, and the read path gracefully shows a non-interactive view if the cache is missing (stale refresh, first-run on an existing deploy). The modal is not rendered until the data is present.

## Test plan

Unit tests (on `compute_roster_breakdown`):
- Classification parity: every player's `status` matches the branch `_apply_displacement` would have taken for them.
- Scale factor correctness: `DISPLACED` rows carry the same factor `_apply_displacement` would compute; `ACTIVE`, `IL_FULL`, `BENCH`, `NO_PROJECTION` carry 1.0, 1.0, 0.0, 0.0 respectively.
- Sum-to-team invariant: for each counting category, sum of scaled contributions equals `project_team_stats(roster, displacement=True)[cat]`.
- Rate-stat component invariant: AVG/ERA/WHIP computed from summed scaled components matches `project_team_stats`'s rate output.
- Missing projection: a player with `rest_of_season=None` yields `NO_PROJECTION`, scale 0, empty `raw_stats`, and does not perturb other rows' factors.

Integration tests (refresh pipeline):
- `_build_projected_standings` writes the `STANDINGS_BREAKDOWN` cache alongside `PROJECTIONS`.
- Cache payload round-trips through JSON without loss.

Template / route tests:
- When `STANDINGS_BREAKDOWN` cache is present, the standings page includes the embedded `<script type="application/json" id="breakdown-data">` tag.
- When the cache is absent, the page renders without the tag and without the click affordance.

Manual verification:
- Click each stat header in Stat Totals mode for own team and at least one opponent; confirm modal totals match cell values.
- Switch to ERoto Points mode and confirm cells are no longer interactive.
- Switch back to stat-totals and confirm interactivity returns.
- Confirm ESC, click-outside, and close button each dismiss the modal.
- Confirm at least one opponent's modal either has no `NO_PROJECTION` rows (clean) or flags the expected ones.
