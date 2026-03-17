# Draft Dashboard UI — Design Spec

## Overview

A browser-based live dashboard that visualizes the draft assistant's state in real-time. The existing CLI remains the input mechanism — the browser is a read-only visualization that updates automatically as picks are entered in the terminal.

## Architecture

**Communication flow:** CLI writes draft state to a shared JSON file (`data/draft_state.json`) after each pick. A Flask server serves a single-page dashboard that polls `/api/state` via htmx every 2 seconds. The browser updates panels without a full page reload.

**Launch:** `python scripts/run_draft.py` starts both the CLI loop and the Flask server (in a background thread). User opens `http://localhost:5000` in their browser.

**Atomic writes:** `state.py` writes to a temp file then renames to `draft_state.json` to prevent the Flask thread from reading a partially-written file.

## Layout: Sidebar

Two-column layout optimized for wide monitors.

**Left column (2/3 width): Draft Board**
- Table of all available players sorted by VAR descending
- Columns: Rank, Name, Position(s), VAR, key stats (HR/R/RBI/SB/AVG for hitters, W/K/SV/ERA/WHIP for pitchers)
- Position filter buttons: All / C / 1B / 2B / 3B / SS / OF / P (P matches any player with SP or RP in positions)
- Drafted players grayed out or hidden (toggle)
- Players on user's team highlighted

**Right column (1/3 width), stacked panels:**

1. **Status Bar** — current round, pick number, which team is picking, "YOUR PICK" indicator, picks until next user turn
2. **Recommendations** — top 5 picks with VAR, positions list, [NEED] flags, and scarcity notes
3. **Your Roster** — position-by-position grid (C x1, 1B x1, 2B x1, 3B x1, SS x1, IF x1, OF x4, UTIL x2, P x9, BN x2, IL x2) showing filled slots with player names and empty slots as dashes. Players assigned by `best_position` from VAR calculation, not by slot optimization.
4. **Category Balance** — 10 stat categories showing projected totals vs league-average targets. Each category displayed as a progress bar. Warnings (below 60% of target after 5+ hitters or 3+ pitchers) highlighted in red.

## State File Format

`data/draft_state.json` — written atomically by the CLI after every pick.

### `state.py` interface

```python
def serialize_state(
    tracker: DraftTracker,
    balance: CategoryBalance,
    board: pd.DataFrame,
    recommendations: list[dict],
    filled_positions: dict[str, int],
) -> dict:
    """Convert all draft objects into a JSON-serializable dict."""

def write_state(state: dict, path: Path) -> None:
    """Atomically write state dict to JSON (write tmp + rename)."""

def read_state(path: Path) -> dict:
    """Read state dict from JSON. Returns empty dict on decode error."""
```

### Example state

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
    {"name": "Logan Webb", "var": 5.9, "best_position": "P", "positions": ["SP"], "need_flag": false, "note": ""}
  ],
  "balance": {
    "totals": {"R": 210, "HR": 78, "RBI": 195, "SB": 12, "AVG": 0.280, "W": 0, "K": 0, "SV": 0, "ERA": 0.0, "WHIP": 0.0},
    "warnings": ["SB is low (12, target ~100)"]
  },
  "available_players": [
    {"name": "Logan Webb", "positions": ["SP"], "var": 5.9, "player_type": "pitcher", "w": 14, "k": 180, "sv": 0, "era": 3.20, "whip": 1.10},
    {"name": "Pete Alonso", "positions": ["1B"], "var": 3.8, "player_type": "hitter", "r": 88, "hr": 35, "rbi": 95, "sb": 2, "avg": 0.254}
  ],
  "filled_positions": {"OF": 2, "SS": 1}
}
```

**Hitter fields:** `name`, `positions`, `var`, `player_type` ("hitter"), `r`, `hr`, `rbi`, `sb`, `avg`

**Pitcher fields:** `name`, `positions`, `var`, `player_type` ("pitcher"), `w`, `k`, `sv`, `era`, `whip`

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
| `src/fantasy_baseball/draft/state.py` | State serialization: serialize, atomic write, read |
| `scripts/run_draft.py` (modify) | Write state after each pick, start Flask in background thread |

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
- Warning: `#ff6b6b`
- Success/positive: `#4ecca3`
- User pick highlight: `#e94560` border glow

## Not in Scope

- Browser-based input (all input stays in CLI)
- Mobile layout
- Multiple browser clients
- Persistent state across restarts
