# Season Dashboard — Design Spec

In-season fantasy baseball dashboard for Yahoo 5x5 roto keeper league 5652. Provides standings visualization, lineup optimization, and waiver/trade recommendations through a browser-based UI.

## Architecture

### Tech Stack

- **Backend:** Flask + Python (new app, separate from draft dashboard)
- **Frontend:** htmx for interactivity, server-rendered HTML templates
- **Entry point:** `src/fantasy_baseball/web/season_app.py` (alongside existing `app.py`)
- **Templates:** `src/fantasy_baseball/web/templates/season/` (separate from draft templates)
- **Static assets:** shared `src/fantasy_baseball/web/static/` directory with draft dashboard

### Data Model — Cached with On-Demand Refresh

The dashboard loads instantly from cached JSON files. A "Refresh Data" button triggers a background fetch from Yahoo and re-runs all computations. This avoids 15-60 second page loads.

**Cache files** (stored in `data/cache/`):
- `standings.json` — current Yahoo standings (roto points + stat totals)
- `roster.json` — current Yahoo roster with positions and status
- `projections.json` — blended projections matched to roster
- `lineup_optimal.json` — optimizer output (hitter + pitcher assignments, suggested moves)
- `probable_starters.json` — weekly pitcher schedule with matchup quality
- `waivers.json` — top waiver pickups with add/drop pairs and category impact
- `trades.json` — 1-for-1 trade recommendations with per-team standings impact
- `monte_carlo.json` — simulation results (median/p10/p90, rank distribution, category risk)
- `meta.json` — last refresh timestamp, current week, scoring period

**Refresh flow:**
1. User clicks "Refresh Data" button
2. Backend kicks off Yahoo API calls + computation in a background thread
3. Frontend polls `/api/refresh-status` until complete (htmx polling)
4. Page reloads from fresh cache

### Authentication

Assumes existing OAuth token in `config/oauth.json`. If the token is expired or missing, the dashboard shows an error message directing the user to re-authenticate via CLI (`python scripts/run_lineup.py` triggers the OAuth flow).

**TODO:** Add browser-based OAuth flow — redirect to Yahoo login, callback to dashboard, store refreshed token. This enables remote/mobile access without CLI.

### API Endpoints

```
GET  /                          → Redirect to /standings
GET  /standings                 → Standings page
GET  /lineup                    → Lineup page
GET  /waivers-trades            → Waivers & Trades page
POST /api/refresh               → Trigger data refresh (returns job ID)
GET  /api/refresh-status        → Poll refresh progress
POST /api/optimize              → Run lineup optimizer, return suggested moves
GET  /api/trade/<idx>/standings  → Return before/after standings for a trade
```

### Existing Code Reuse

The dashboard reuses existing computation modules — no new math, just new presentation:

| Feature | Module | Key Function |
|---------|--------|-------------|
| Standings leverage | `lineup/leverage.py` | `calculate_leverage()` |
| Weighted SGP | `lineup/weighted_sgp.py` | `calculate_weighted_sgp()` |
| Lineup optimization | `lineup/optimizer.py` | `optimize_hitter_lineup()`, `optimize_pitcher_lineup()` |
| Yahoo data | `lineup/yahoo_roster.py` | `fetch_roster()`, `fetch_standings()`, `fetch_free_agents()` |
| Waiver scanning | `lineup/waivers.py` | `scan_waivers()` |
| Trade evaluation | `trades/evaluate.py` | `find_trades()` |
| Trade pitches | `trades/pitch.py` | `generate_pitch()` |
| Roto scoring | `scoring.py` | `project_team_stats()`, `score_roto()` |
| Monte Carlo | `simulation.py` | `simulate_season()` |
| Matchup quality | `lineup/matchups.py` | `calculate_matchup_factors()` |
| Projections | `data/projections.py` | `blend_projections()` |
| Game logs | `analysis/game_logs.py` | `fetch_all_game_logs()` |
| Schedule | `data/mlb_schedule.py` | `get_week_schedule()` |

## Layout

### Global Structure

Dark theme matching the existing draft dashboard aesthetic. Two-region layout:

- **Sidebar (fixed, left):** Team name ("Hart of the Order"), navigation links, last refresh timestamp, "Refresh Data" button. Collapses to a top navigation bar on mobile (responsive breakpoint).
- **Main content area:** Page header with current week/scoring period, pill-style toggles for view switching, content tables/cards.

### Navigation

Three sidebar links:
1. **Standings** — default landing page
2. **Lineup**
3. **Waivers & Trades**

Active page highlighted with a left border accent and bold text.

## Standings Page

Top-level pill toggle: **Current | Projected**. Sub-toggles appear contextually below based on the active top-level selection.

### Current Standings View

Sub-toggle between two views via pill-style radio buttons:

**Roto Points:** Table with all 10 teams, sorted by total roto points (descending). Columns: rank, team name, R, HR, RBI, SB, AVG, W, K, SV, ERA, WHIP, Total. Each cell shows the roto points (1-10) earned in that category.

**Stat Totals:** Same table layout, but cells show raw stat values (312 R, .271 AVG, 3.45 ERA, etc.). Still sorted by total roto points.

**Color coding (both sub-views):** For the user's team row:
- **Green** cells for categories ranked 1st, 2nd, or 3rd
- **Red** cells for categories ranked 8th, 9th, or 10th
- **Default** (white/gray) for categories ranked 4th-7th

The user's team row has a subtle blue background tint for quick identification regardless of standings position.

### Projected Standings View

Toggle between three sub-views via a second row of pill-style radio buttons (appears when "Projected" is selected):

**Static:** Same table format as Current Standings (roto points / stat totals toggle applies here too), but using full-season projected stats from blended projections. Shows where teams are expected to finish if current rosters hold and projections are accurate.

**Monte Carlo:** Table with columns: Team, Median Pts, P10, P90, 1st %, Top 3 %. Based on 1000-iteration simulation with injury variance and correlated stat variance. Below the main table, a **Category Risk** panel for the user's team showing per-category: Median Pts, P10, P90, Top 3 %, Bottom 3 %. Color coded: green for safe categories (high Top 3 %), red for risky categories (high Bottom 3 %).

**MC + Roster Mgmt:** Same format as Monte Carlo, but factors in active roster management adjustments (the simulation models teams making optimal lineup decisions week-to-week rather than setting-and-forgetting).

## Lineup Page

### Hitter Lineup

Table showing current hitter assignments. Columns: Slot (C, 1B, 2B, 3B, SS, IF, OF×4, UTIL×2), Player, Eligible Positions, Games This Week, Leverage-Weighted SGP (wSGP), Status.

- Active lineup players shown at full opacity
- Bench players shown below at reduced opacity, labeled "BN"
- IL players shown with a red "IL" badge and a red row tint if occupying an active slot
- Sorted by slot order (positional slots first, then UTIL, then BN, then IL)

### Pitcher Lineup

Same table format for pitchers. Columns: Slot (P×9), Player, Games This Week, wSGP, Status. Starting 9 ranked by wSGP, bench pitchers below.

### Optimize Button

Located in the top-right of each lineup section (hitters and pitchers).

- **Green "Optimize" button** when the current lineup is suboptimal — at least one move would improve total wSGP
- **Grayed out "Optimal ✓" button** when the current lineup is already the best possible assignment

Clicking "Optimize" triggers `/api/optimize` and displays a **Suggested Moves** banner below the button:
- Each move shown as: `START [player name] → [slot]` (green badge) or `BENCH [player name] → BN` / `IL [player name] → IL` (red badge)
- Shows the wSGP or reason (e.g. "IL-eligible") for each move
- Footer text: "Make these moves manually in Yahoo. Refresh data after to confirm."

### Probable Starters This Week

Table showing the user's pitchers who have scheduled starts. Columns: Pitcher, Day(s), Opponent, Matchup Quality, Starts Count.

**Matchup quality badges:**
- **Great** (green) — opponent has a weak lineup (low OPS)
- **Fair** (yellow) — average opponent
- **Tough** (red) — opponent has a strong lineup (high OPS)

**Expandable rows:** Clicking a row expands it to show the underlying matchup data:
- Opponent team OPS and league rank (e.g. "28th in OPS, .668")
- Opponent K% and league rank
- If a 2-start pitcher, both matchups shown separately

**Two-start pitchers** highlighted with a blue "2-start" badge in the Starts column.

## Waivers & Trades Page

### Waiver Wire

Card-based layout showing the top 5 recommended waiver pickups, ranked by leverage-weighted SGP gain.

Each waiver card contains:
- **Header:** Player name, position, ownership status (FA/waiver). wSGP gain as the headline number (right-aligned, green).
- **Projected stats:** Full-season projection line (e.g. ".262 / 28 HR / 75 RBI / 12 SB")
- **Add/Drop pair:** `ADD [player] → DROP [player]` with the roster slot
- **Category impact:** Per-category stat deltas with green (improvement) / red (regression) coloring (e.g. "HR +9, AVG −.008")

### Trade Recommendations

Card-based layout showing all viable 1-for-1 trade opportunities, sorted by the user's wSGP gain. Only win-win trades are shown (both sides must gain wSGP).

Each trade card contains:

**Collapsed view (default):**
- Opponent team name
- User's wSGP gain (right-aligned, green)
- Visual send/receive layout: `YOU SEND [player, position]` (red-tinted box) ⇄ `YOU GET [player, position]` (green-tinted box)
- "Show details & pitch" expand toggle

**Expanded view:**
- **"Why this works"** — category gains for both sides (e.g. "You gain: SV +18, ERA −0.20 / They gain: W +4, K +35")
- **Before/after standings** — per-category roto standings for both the user's team and the trade partner's team, showing how each category changes post-trade. Toggleable between Roto Points view and Stat Totals view (same toggle pattern as the Standings page).
- **Pre-written pitch** — generated trade pitch text in a styled quote block, ready to copy/paste to the other manager

**TODO:** Multi-player trade evaluation. Scope and UI TBD — will be added in a future iteration.

## File Structure

```
src/fantasy_baseball/web/
├── app.py                          # Existing draft dashboard (unchanged)
├── season_app.py                   # New: season dashboard Flask app
├── season_routes.py                # New: route handlers (standings, lineup, waivers-trades)
├── season_data.py                  # New: cache management, refresh logic, data assembly
├── templates/
│   ├── dashboard.html              # Existing draft template (unchanged)
│   └── season/
│       ├── base.html               # Layout: sidebar + main content area
│       ├── standings.html          # Standings page with toggle partials
│       ├── lineup.html             # Lineup page with optimize + starters
│       └── waivers_trades.html     # Waivers & trades page
├── static/
│   ├── style.css                   # Existing draft styles (unchanged)
│   └── season.css                  # New: season dashboard styles
data/cache/                         # New: cached computation results (add to .gitignore)
```

## Launch

```bash
python -m fantasy_baseball.web.season_app
# or
python scripts/run_season_dashboard.py
```

Starts Flask dev server on a configurable port (default 5001, to avoid conflict with draft dashboard on 5000). Prints URL to terminal.

## TODO (Future Iterations)

- **Browser-based OAuth flow** — redirect to Yahoo login from the dashboard, eliminating the need for CLI re-auth. Required for remote/mobile access.
- **Multi-player trade evaluation** — evaluate 2-for-1, 2-for-2, etc. trade packages. Scope and UI TBD.
- **Remote hosting** — deploy to a VPS or cloud service for phone access. Requires browser OAuth and HTTPS.
- **Auto-refresh** — optional periodic background refresh (e.g. every 30 minutes) instead of manual button clicks.
