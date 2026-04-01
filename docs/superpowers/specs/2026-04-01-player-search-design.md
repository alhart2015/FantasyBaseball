# Player Search

**Date:** 2026-04-01
**Branch:** `player-search`

## Problem

No way to look up an arbitrary player's stats, projections, and value in the dashboard. The waivers/trades page shows recommendations but doesn't let you explore the full player pool. When evaluating a potential pickup or trade, you need to manually cross-reference multiple data sources.

## Design

### New "Players" tab in dashboard nav

A dedicated page with a search box. Type a player name, get back a card with everything you need to evaluate them.

### Backend

**Route:** `GET /players` — renders the search page (empty state with search input).

**API:** `GET /api/players/search?q=<name>` — returns JSON array of matched players.

**Search logic:**
1. Query `ros_blended_projections` table with `name LIKE '%query%'` (case-insensitive, normalized)
2. Cap at 25 results, ordered by ADP (most relevant players first, fallback to name)
3. For each match, also query `blended_projections` for preseason stats
4. Compute wSGP on-the-fly using cached leverage weights (`read_cache("standings")` → `calculate_leverage`)
5. Look up game log totals for pace data (reuse `_load_game_log_totals` pattern)
6. Determine ownership: check cached roster + opponent rosters, else "Free Agent"

**Response shape per player:**
```json
{
  "name": "Aaron Judge",
  "team": "NYY",
  "positions": ["OF", "DH"],
  "player_type": "hitter",
  "ownership": "Free Agent",
  "wsgp": 4.23,
  "ros": {"r": 95, "hr": 38, "rbi": 92, "sb": 7, "avg": 0.272},
  "preseason": {"r": 110, "hr": 45, "rbi": 120, "sb": 5, "avg": 0.291},
  "pace": {
    "R": {"actual": 12, "expected": 14, "z_score": -0.8},
    "HR": {"actual": 5, "expected": 6, "z_score": -0.5}
  }
}
```

`pace` is null/omitted when no game log data exists for the player.

### Frontend

**Page:** Search input at top, results below. Empty state shows a prompt ("Search for any player by name").

**Interaction:**
- Debounced keyup (300ms) triggers fetch to `/api/players/search?q=...`
- Minimum 2 characters before searching
- Loading spinner while fetching
- Results render as cards

**Player card layout:**
- Header: name, positions, team, ownership badge ("Your roster" / "Team X" / "FA"), wSGP
- Stat table: one row per relevant category
  - Columns: Category | ROS Proj | Preseason | Actual | Pace Z
  - Hitters: R, HR, RBI, SB, AVG
  - Pitchers: W, K, SV, ERA, WHIP
  - Pace columns show "—" when no game log data

**Styling:** Follow existing dashboard patterns (dark theme, card layout, same CSS variables).

### Data sources

| Data | Source | Availability |
|------|--------|-------------|
| ROS projections | `ros_blended_projections` table | Always (required) |
| Preseason projections | `blended_projections` table | Always |
| Leverage weights | `read_cache("standings")` → `calculate_leverage` | After first refresh |
| Game log totals | `game_logs` table aggregated | After first refresh |
| Ownership | Cached roster + opponent rosters from `read_cache("roster")` and opponent data | After first refresh |
| Positions | From projection DataFrame or Yahoo roster data | Always in projections |

### Nav change

Add "Players" link to `base.html` sidebar, between "Waivers & Trades" and "SQL":
```html
<a href="{{ url_for('player_search') }}"
   class="nav-link {% if active_page == 'players' %}active{% endif %}">
    Players
</a>
```

## What this does NOT include

- Position filtering or advanced search (name only for now)
- Comparison mode (side-by-side players)
- Add/drop actions from the search results
- Caching of search results (queries are fast enough against SQLite)
