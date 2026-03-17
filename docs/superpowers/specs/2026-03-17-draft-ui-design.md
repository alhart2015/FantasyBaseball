# Draft Dashboard UI — Design Spec

## Overview

A browser-based live dashboard that visualizes the draft assistant's state in real-time. The existing CLI remains the input mechanism — the browser is a read-only visualization that updates automatically as picks are entered in the terminal.

## Architecture

**Communication flow:** CLI writes draft state to a shared JSON file (`data/draft_state.json`) after each pick. A Flask server serves a single-page dashboard that polls `/api/state` via htmx every 2 seconds. The browser updates panels without a full page reload.

**Launch:** `python scripts/run_draft.py` starts both the CLI loop and the Flask server (in a background thread). User opens `http://localhost:5000` in their browser.

## Layout: Sidebar

Two-column layout optimized for wide monitors.

**Left column (2/3 width): Draft Board**
- Table of all available players sorted by VAR descending
- Columns: Rank, Name, Position(s), VAR, key stats (HR/R/RBI/SB/AVG for hitters, W/K/SV/ERA/WHIP for pitchers)
- Position filter buttons: All / C / 1B / 2B / 3B / SS / OF / P
- Drafted players grayed out or hidden (toggle)
- Players on user's team highlighted

**Right column (1/3 width), stacked panels:**

1. **Status Bar** — current round, pick number, which team is picking, "YOUR PICK" indicator, picks until next user turn
2. **Recommendations** — top 5 picks with VAR, position, [NEED] flags, and scarcity notes
3. **Your Roster** — position-by-position grid (C, 1B, 2B, 3B, SS, IF, OF x4, UTIL x2, P x9, BN x2, IL x2) showing filled slots with player names and empty slots as dashes
4. **Category Balance** — 10 stat categories showing projected totals vs league-average targets. Each category displayed as a progress bar. Warnings (below 60% of target after 5+ hitters or 3+ pitchers) highlighted in red.

## State File Format

`data/draft_state.json` — written by the CLI after every pick:

```json
{
  "current_pick": 15,
  "current_round": 2,
  "picking_team": 6,
  "is_user_pick": false,
  "picks_until_user_turn": 4,
  "user_roster": ["Juan Soto", "Julio Rodriguez", "Junior Caminero"],
  "drafted_players": ["Elly De La Cruz", "..."],
  "recommendations": [
    {"name": "Logan Webb", "var": 5.9, "best_position": "P", "need_flag": false, "note": ""}
  ],
  "balance": {
    "totals": {"R": 210, "HR": 78, "RBI": 195, "SB": 12, "AVG": 0.280, "W": 0, "K": 0, "SV": 0, "ERA": 0.0, "WHIP": 0.0},
    "warnings": ["SB is low (12, target ~100)"]
  },
  "available_players": [
    {"name": "Logan Webb", "positions": ["SP"], "var": 5.9, "player_type": "pitcher", "hr": 0, "r": 0, "w": 14, "k": 180}
  ],
  "filled_positions": {"OF": 2, "SS": 1}
}
```

## Tech Stack

- **Flask** — lightweight Python web server
- **htmx** — declarative AJAX for live updates (loaded from CDN)
- **Vanilla CSS** — dark theme, no CSS framework needed

## Files

| File | Responsibility |
|------|---------------|
| `src/fantasy_baseball/web/__init__.py` | Empty package init |
| `src/fantasy_baseball/web/app.py` | Flask app with routes: `/` (dashboard), `/api/state` (JSON state) |
| `src/fantasy_baseball/web/templates/dashboard.html` | Single-page dashboard with htmx polling |
| `src/fantasy_baseball/web/static/style.css` | Dark theme dashboard styles |
| `src/fantasy_baseball/draft/state.py` | State serialization: read/write draft_state.json |
| `scripts/run_draft.py` (modify) | Write state file after each pick, start Flask in background thread |

## Dependencies

Add to `pyproject.toml`:
- `flask>=3.0`

## Dark Theme Colors

- Background: `#0f0f1a`
- Panel background: `#1a1a2e`
- Panel border: `#16213e`
- Accent/headers: `#e94560`
- Text primary: `#eee`
- Text secondary: `#888`
- Warning: `#e94560`
- Success/positive: `#4ecca3`
- User pick highlight: `#e94560` border glow

## Not in Scope

- Browser-based input (all input stays in CLI)
- Mobile layout
- Multiple browser clients
- Persistent state across restarts
