# Buy-Low Detector + Collapsible Sections

**Date:** 2026-03-27
**Status:** Approved

## Problem

The Waivers & Trades page shows waiver add/drop recommendations and trade proposals, but doesn't identify players who are underperforming their projections — prime buy-low candidates. The page is also getting long with no way to collapse sections.

## Solution

Add two new sections to the Waivers & Trades page: "Buy-Low Trade Targets" (opponent roster players underperforming projections) and "Buy-Low Free Agents" (waiver wire players underperforming projections). Make all sections on the page collapsible.

## Buy-Low Detection

### Qualification criteria

A player qualifies as buy-low when their **average z-score across roto categories** is < -1.0 (more than 1 standard deviation below projection pace).

- Hitter roto categories: R, HR, RBI, SB, AVG
- Pitcher roto categories: W, K, SV, ERA, WHIP
- Average the z-scores from `compute_player_pace()` (already implemented in `analysis/pace.py`)
- For rate stats below sample threshold (< 30 PA for AVG, < 10 IP for ERA/WHIP), exclude from the average rather than including a 0.0 z-score that dilutes the signal
- Players below minimum sample (< 10 PA or < 5 IP) are excluded entirely — not enough data

### Sorting

Sort by average z-score ascending (most underperforming first). Players > 1.5 SD below naturally appear at the top.

### Data sources

**Trade targets:** Opponent rosters are already fetched and matched to projections during the trade evaluation step (Step 11 in refresh pipeline). Run `compute_player_pace()` against game logs for each opponent player.

**Free agents:** Free agent players are already fetched and matched to projections during the waiver scan step (Step 10). Run `compute_player_pace()` against game logs for each free agent.

Game logs are already bulk-loaded from SQLite during Step 6b (pace computation for own roster). Reuse those lookup dicts (`hitter_logs`, `pitcher_logs`) — they contain all players' season totals, not just our roster.

## Display

### Card layout (collapsed — default)

```
[Player Name]  [Positions]  [Owner or "Free Agent"]     [-1.8 SD]
```

- Player name in bold
- Positions in secondary text
- Owner name (for trade targets) or "Free Agent" label
- Average z-score badge, color-coded using existing stat-cold-1/stat-cold-2 classes

### Card layout (expanded — on click)

Per-category stats table:

```
Cat    Actual    Expected    Z-score
R      12        18.4        -0.72
HR     2         5.1         -1.42
RBI    10        17.8        -0.88
SB     0         2.3         -1.40
AVG    .198      .271        -2.48
                       Avg:  -1.38
wSGP: 1.24
```

- "Expected" = pace-adjusted expected (from `compute_player_pace().expected`), not full-season projection. For counting stats this is projection scaled by PA/IP consumed. For rate stats this is the projected rate directly.
- Color code each z-score cell using the existing 5-level scheme
- wSGP = player's projected stats run through **your team's** leverage weights (how valuable they'd be to you)
- All numbers rounded to 2 significant figures (use 2 decimal places for z-scores, 3 for AVG/ERA/WHIP, integers for counting stats)

### Show more / show less

- Show 5 candidates per subsection by default
- "Show More" button at the bottom reveals all qualifying candidates
- Clicking again collapses back to 5
- Button text toggles: "Show More (N total)" / "Show Less" — N is the total count of qualifying candidates in that subsection

## Collapsible Sections

All sections on the Waivers & Trades page get collapsible behavior:

- **Waiver Wire** (existing)
- **Trade Recommendations** (existing)
- **Buy-Low Trade Targets** (new)
- **Buy-Low Free Agents** (new)

### Behavior

- Click the section header to toggle open/closed
- Chevron indicator (▼ open, ▶ closed)
- Default state: all sections expanded
- Simple JS `toggleSection()` function, no library needed

## Architecture

### New module: `src/fantasy_baseball/analysis/buy_low.py`

```python
def find_buy_low_candidates(
    players: list[dict],       # roster entries with projection stats (always dicts, not Series)
    game_log_lookup: dict,     # {normalized_name: {stat: value}} from bulk query
    leverage: dict,            # per-category leverage weights (for wSGP computation)
    owner: str = "Free Agent", # team name or "Free Agent"
) -> list[dict]:
    """Find players underperforming projections by > 1 SD.

    Returns list of candidate dicts sorted by avg_z ascending:
    {
        "name": str, "positions": list, "owner": str, "player_type": str,
        "avg_z": float, "stats": dict (from compute_player_pace),
        "wsgp": float,
    }
    """
```

**Input normalization:** `fa_players` from the waiver scan is `list[pd.Series]`. The caller (refresh pipeline) must convert to `list[dict]` via `.to_dict()` before passing to `find_buy_low_candidates()`. Opponent rosters from `opp_rosters` are already `list[dict]`.

**wSGP computation:** Call `calculate_weighted_sgp(pd.Series(player), leverage)` using the **projection stats** from the player entry and **your team's leverage weights** (not the opponent's). This tells you how valuable the player would be to you specifically.

This module:
1. For each player, looks up actuals via `normalize_name(player["name"])` against `game_log_lookup`
2. Extracts projection stats dict from the player entry (lowercase keys matching pace.py expectations)
3. Calls `compute_player_pace(actuals, projected, player_type)` to get per-stat z-scores
4. Averages z-scores across roto categories, **excluding stats where `z_score == 0.0` AND `color_class == "stat-neutral"`** (the signal that the stat was below sample threshold or had no projection — avoids diluting the average with non-informative zeros)
5. Filters to avg_z < -1.0
6. Computes wSGP via `calculate_weighted_sgp()` using projection stats and your leverage
7. Sorts ascending (most negative first)
8. Attaches owner name for display

**Important:** ERA and WHIP z-scores arrive already sign-inverted from `compute_player_pace()` (negative = bad for pitcher). Do not re-invert during averaging.

### Refresh pipeline changes: `season_data.py`

Insert a new step after Step 11 (trade evaluation), before Step 12 (standings projection):

```
Step 11b: Compute buy-low candidates
```

1. Reuse `hitter_logs` and `pitcher_logs` dicts from Step 6b (still in local scope — the connection is closed but the dicts are plain Python dicts with no connection dependency)
2. For opponent rosters: iterate `opp_rosters` dict, call `find_buy_low_candidates()` for each team with `owner=team_name`
3. For free agents: convert `fa_players` from `list[pd.Series]` to `list[dict]` via `[s.to_dict() for s in fa_players]`, then call `find_buy_low_candidates()` with `owner="Free Agent"`
4. Merge all opponent results into one list, re-sort by avg_z
5. Write `buy_low.json` cache with two keys: `{"trade_targets": [...], "free_agents": [...]}`

### Route changes: `season_routes.py`

Add `buy_low` to the data passed to the template:

```python
buy_low = read_cache("buy_low") or {}
```

### Template changes: `waivers_trades.html`

1. Wrap each existing section in a collapsible container
2. Add two new sections after Trade Recommendations
3. Add CSS for collapsible headers, chevrons, buy-low cards
4. Add JS for `toggleSection()` and `toggleShowMore()`

### Cache changes (REQUIRED — app will crash without this)

Add `"buy_low"` to `CACHE_FILES` dict in `season_data.py` (line 41-51). Without this, `read_cache("buy_low")` raises `KeyError`:

```python
"buy_low": "buy_low.json",
```

## Rounding

All numbers displayed with 2 significant figures:
- Z-scores: 2 decimal places (e.g., -1.38)
- Counting stats (actual/projected): integers
- Rate stats (AVG): 3 decimal places
- Rate stats (ERA/WHIP): 2 decimal places
- wSGP: 2 decimal places

## Edge cases

- **No game logs in DB yet:** No candidates qualify (all excluded by sample size). Section shows "No buy-low candidates yet — game log data needed."
- **No opponent rosters fetched:** Trade targets section empty. Free agents still computed.
- **Player with partial stats:** Only categories with enough sample count toward the average z-score.
- **All z-scores above -1.0:** Section shows "No buy-low candidates found — all players tracking near projections."

## What doesn't change

- Existing waiver wire recommendations (still computed by wSGP)
- Existing trade recommendations (still computed by roto point impact)
- Refresh pipeline structure (new step inserted, existing steps unmodified)
