# Roster Audit Design

## Problem

The waiver wire system evaluates free agents using raw projections while roster players get recency-blended data (projections + actual stats). This apples-to-oranges comparison means the system can't reliably answer: "Is Bryan Abreu still the best option for this roster slot, or is there someone better on the wire?"

More broadly, there's no systematic way to evaluate every roster spot against the available player pool. The user has to manually identify problems (like a hurt closer's underperforming replacement) and then hope the waiver scanner surfaces the right fix.

## Solution

A **roster audit** that runs during the daily refresh and produces a per-slot assessment of the entire roster. For each player, it answers: "Is there a better option available?" — using the same recency-blended methodology for both sides of the comparison.

## Design

### 1. Recency blending for all players

**Current state:** `run_lineup.py` applies `predict_reliability_blend` to roster players inline. `season_data.py` does not apply recency blending at all — roster players and FAs both use raw projections in the waiver scan.

**Change:** Extract a shared blending function and apply it to both roster players and FAs in `season_data.py` before the waiver scan and roster audit run.

#### New function: `blend_player_with_game_logs`

Location: `src/fantasy_baseball/lineup/blending.py` (new module)

```python
def blend_player_with_game_logs(
    player: Player,
    game_logs: list[dict],
    cutoff: str,
) -> Player:
    """Apply reliability-weighted recency blend to a Player's ROS stats.

    Converts the Player's ROS stats to per-PA/IP projection rates,
    runs predict_reliability_blend against game log entries, and
    returns a new Player with updated ROS stats.

    If game_logs is empty, returns the player unchanged.
    """
```

This function:
1. Converts `Player.ros` (HitterStats/PitcherStats dataclass) to the projection rates dict that `predict_reliability_blend` expects
2. Calls `predict_reliability_blend(proj_rates, game_logs, cutoff)`
3. Converts blended rates back to counting stats using the player's projected PA/IP
4. Returns a new `Player` with updated `.ros`

#### New function: `load_game_logs_by_name`

Location: same module

```python
def load_game_logs_by_name(
    season: int,
) -> dict[str, list[dict]]:
    """Load per-game log entries from SQLite, keyed by normalized name.

    Returns {normalized_name: [game_dicts]} where each game dict has
    the fields expected by predict_reliability_blend (date, pa/ab/h/...
    for hitters, ip/k/er/... for pitchers).
    """
```

Queries the `game_logs` table for all players in the season, groups by name, returns the per-game entries (with dates) needed by the reliability blend. This is different from `_load_game_log_totals` which returns only aggregated sums.

Note: the `game_logs` table has `gs` (games started) but no `g` (games) column. Since each row is one game appearance, the query should synthesize `g = 1` per row for pitcher game dicts. The recency blend needs `g` for `sv_per_g` calculation.

#### Integration into `season_data.py::run_full_refresh`

Two blend points in the pipeline:

1. **After step 6b** (fetch game logs): call `load_game_logs_by_name(season_year)` once to get per-game logs for all MLB players. Blend each roster `Player` in `roster_players`.
2. **At step 10** (after FAs are fetched from Yahoo): blend each FA `Player` returned by `fetch_and_match_free_agents`, then pass blended roster + blended FAs to both the roster audit and the waiver scan.

The game log dict is loaded once and reused for both blend points.

### 2. Roster audit module

Location: `src/fantasy_baseball/lineup/roster_audit.py`

#### Core function: `audit_roster`

```python
def audit_roster(
    roster: list[Player],
    free_agents: list[Player],
    leverage: dict[str, float],
    roster_slots: dict[str, int],
) -> list[dict]:
    """Evaluate every roster slot against the best available FA.

    For each roster player:
    1. Find the best FA who could replace them (any eligible type for that slot)
    2. Re-optimize the full team lineup with the swap
    3. Compute the wSGP delta (positive = FA is better)

    Returns a list of audit entries sorted by gap descending (biggest
    problems first), including entries where no upgrade is available.
    """
```

Each audit entry contains:

```python
{
    "player": str,              # Roster player name
    "player_type": str,         # "hitter" or "pitcher"
    "slot": str,                # Current optimized slot (C, 1B, P, BN, etc.)
    "player_wsgp": float,       # Recency-blended wSGP
    "best_fa": str | None,      # Best available replacement (None if no upgrade)
    "best_fa_type": str | None, # FA player type
    "best_fa_wsgp": float | None, # FA's recency-blended wSGP (None if no upgrade)
    "gap": float,               # Re-optimized team wSGP gain (0.0 if no upgrade)
    "categories": dict,         # Per-category breakdown of the gain
    "lineup_before": list,      # Current lineup with this player marked
    "lineup_after": list,       # Lineup after swap with FA, if applicable
}
```

#### How it works

1. Compute baseline team wSGP using `_compute_team_wsgp` (reuse from `waivers.py`)
2. For each roster player, determine their assigned slot from the optimized lineup
3. Find FA candidates: any FA whose wSGP exceeds the roster player's, filtered by position feasibility
4. For the top FA candidate, simulate the swap: build new roster, re-optimize, measure team wSGP delta
5. If delta > 0, record it as the gap. If no FA improves the team, record gap = 0 with `best_fa: None`
6. Sort by gap descending

Cross-type evaluation happens naturally: if a pitcher slot is occupied by a reliever and a starter would produce more team wSGP, the re-optimization will surface that because `_compute_team_wsgp` runs both hitter and pitcher optimizers on the full new roster.

#### Relationship to `scan_waivers`

The audit and the waiver scan serve different purposes:
- **Audit**: "For each of my players, is there a better option?" — covers the whole roster, shows "no upgrade" entries
- **Waiver scan**: "What are the top N swaps I should make?" — produces an action-ranked list

Both use `_compute_team_wsgp` for re-optimization. The audit runs first and its output is cached separately. The waiver scan continues to run as before but now benefits from recency-blended FA data.

### 3. Integration into refresh pipeline

In `season_data.py::run_full_refresh`:

```
Step 6b: Fetch game logs (existing)
Step 6c: Compute pace (existing)
NEW - Step 6d: Load per-game logs, apply recency blend to roster players
...
Step 10: Fetch FAs from Yahoo (existing)
NEW - Step 10b: Apply recency blend to FA players
NEW - Step 10c: Run roster audit, write cache:roster_audit
Step 10d: Scan waivers (existing, now with blended data on both sides)
```

`blend_player_with_game_logs` returns a new `Player` — roster and FA lists are rebuilt with blended copies, originals are not mutated.

### 4. Web page: `/roster-audit`

New route in `season_routes.py`, new template `season/roster_audit.html`.

#### Route

```python
@app.route("/roster-audit")
def roster_audit():
    meta = read_meta()
    audit_data = read_cache("roster_audit")
    return render_template(
        "season/roster_audit.html",
        meta=meta,
        active_page="roster_audit",
        audit=audit_data or [],
    )
```

#### Page layout

A single table showing all roster slots, sorted by gap (biggest problems first):

| Slot | Your Player | wSGP | Best Available | FA wSGP | Gap | Details |
|------|------------|------|----------------|---------|-----|---------|
| P | Bryan Abreu | 0.42 | Carlos Estevez | 0.71 | +0.29 | expand |
| OF_3 | Player X | 1.10 | Player Y | 1.25 | +0.15 | expand |
| ... | ... | ... | ... | ... | ... | ... |
| SS | Trea Turner | 2.85 | — | — | No better option | — |

**Visual treatment:**
- Rows with a positive gap get a subtle highlight (yellow/amber)
- "No better option" rows are normal/muted — they confirm the slot is strong
- Gap column is color-coded: larger gaps are warmer colors
- Expandable detail rows show per-category breakdown and lineup before/after

**Nav:** Add "Roster Audit" link to the sidebar in `base.html`, between "Lineup" and "Waivers & Trades".

### 5. What this does NOT include

- **No new API calls** — game logs for all MLB players are already fetched each refresh
- **No overperformer/underperformer alerting** — separate TODO
- **No changes to waiver scan logic** — `scan_waivers` stays as-is, just receives better input data
- **No changes to `recency.py`** — the reliability blend model is unchanged
- **No trade integration** — the existing trade evaluator is separate

## Files changed

| File | Change |
|------|--------|
| `src/fantasy_baseball/lineup/blending.py` | **New** — shared recency blend for Player objects |
| `src/fantasy_baseball/lineup/roster_audit.py` | **New** — audit logic |
| `src/fantasy_baseball/web/season_data.py` | Add recency blend step + audit step to refresh |
| `src/fantasy_baseball/web/season_routes.py` | Add `/roster-audit` route |
| `src/fantasy_baseball/web/templates/season/roster_audit.html` | **New** — audit page template |
| `src/fantasy_baseball/web/templates/season/base.html` | Add nav link |
| `tests/test_lineup/test_roster_audit.py` | **New** — audit logic tests |
| `tests/test_lineup/test_blending.py` | **New** — blending function tests |
