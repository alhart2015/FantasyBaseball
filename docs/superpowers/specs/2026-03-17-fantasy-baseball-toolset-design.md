# Fantasy Baseball Toolset — Design Spec

## Overview

A CLI-based Fantasy Baseball toolset for a Yahoo 10-team 5x5 roto league (league ID 5652). Three subsystems built in order: (1) **shared foundation** (auth, data ingestion, SGP engine), (2) **draft assistant** using SGP value-based drafting, and (3) **in-season start/sit optimizer** that recommends lineups to maximize standings gain.

Each subsystem will get its own implementation plan. This spec covers the full design so all three plans share a coherent architecture.

## League Context

- **Platform:** Yahoo Fantasy Baseball (league 5652)
- **Format:** 10-team, 5x5 Rotisserie
- **Hitting categories:** R, RBI, HR, SB, AVG
- **Pitching categories:** W, K, ERA, WHIP, SV
- **Draft:** Snake draft, pick 8th, 3 keepers per team (30 total)
- **Roster:** C, 1B, 2B, 3B, SS, IF, 4 OF, 2 UTIL, 9 P, 2 BN, 2 IL (25 total slots)

## Project Structure

```
FantasyBaseball/
├── src/
│   ├── auth/           # Yahoo OAuth2 setup + credential management
│   ├── data/           # Data fetching & projection ingestion
│   ├── draft/          # Draft rankings and interactive draft assistant
│   ├── lineup/         # Weekly start/sit optimizer
│   ├── sgp/            # SGP engine (shared by draft + lineup)
│   └── utils/          # Shared helpers (stat mappings, position eligibility, constants)
├── data/               # Cached projections, keeper lists, SGP baselines
├── scripts/            # CLI entry points (run_draft.py, run_lineup.py)
├── tests/
└── config/             # League settings, OAuth credentials (gitignored)
```

### Module Boundaries

| Module | Responsibility | Interface | Dependencies |
|--------|---------------|-----------|-------------|
| `auth` | Manage Yahoo OAuth2 tokens | `get_authenticated_session() -> OAuth2` | `yahoo-oauth` |
| `data` | Fetch projections from Yahoo + FanGraphs, blend them | `get_blended_projections(sources, weights) -> DataFrame` | `auth`, `pandas`, `requests` |
| `sgp` | Calculate SGP denominators, replacement levels, player values | `calculate_sgp_values(projections, league_config) -> DataFrame` | `data`, `numpy` |
| `draft` | Interactive draft board, pick recommendations | CLI loop consuming `sgp` output | `sgp`, `data` |
| `lineup` | Lineup optimization, start/sit recommendations | CLI output consuming `sgp` + live roster | `sgp`, `data`, `auth` |
| `utils` | Constants, stat category mappings, position eligibility | Pure functions, no side effects | none |

## Authentication & Data Sources

### Yahoo OAuth2

- User registers a Yahoo Developer app at https://developer.yahoo.com/apps/
- `client_id` and `client_secret` stored in `config/oauth.json` (gitignored)
- On first run, OAuth2 flow opens browser for Yahoo login, token cached for reuse
- Library: `yahoo-fantasy-api` + `yahoo-oauth`
- **Rate limiting:** Yahoo enforces API rate limits. The auth module will implement simple backoff/retry logic to avoid hitting limits, especially important during live draft when the assistant queries frequently.

### Data Sources

1. **Yahoo Fantasy API** — league settings, rosters, keepers, live stats, free agents, Yahoo's own projections, and **position eligibility** (Yahoo assigns eligibility based on games played; this is the authoritative source for multi-position eligibility)
2. **FanGraphs projections** — Steamer, ZiPS, ATC downloaded as CSV files from FanGraphs' public projection pages (manual download, not scraping — FanGraphs provides CSV export buttons). User places CSV files in `data/projections/`. This means `beautifulsoup4` is **not** needed in the tech stack.
3. **Blended projections** — weighted average of available systems per player, configurable weights in `config/league.yaml` (default: equal weighting)

### Position Eligibility

The system uses Yahoo's eligibility data directly via the API. Each player's eligible positions are fetched and cached. This drives:
- **Draft model:** positional scarcity calculations (how deep each position is)
- **Lineup optimizer:** which players can fill which slots (especially IF and UTIL flexibility)

## SGP Engine (Shared Core)

### Standings Gain Points (SGP)

In a 10-team roto league, each stat category earns 1–10 standings points. SGP measures how many raw stats it takes to move up one standings place. A player's value is the sum of standings points they contribute across all 10 categories.

### SGP Denominators

For year one (no historical league data), use **published SGP denominators for 10-team 5x5 leagues** from established fantasy baseball analytics sources (e.g., Mr. Cheatsheet, Tanner Bell's work). Hardcoded defaults:

| Category | SGP Denominator (approx) |
|----------|-------------------------|
| R | 20 |
| HR | 9 |
| RBI | 20 |
| SB | 8 |
| AVG | .005 |
| W | 3 |
| K | 30 |
| ERA | 0.15 |
| WHIP | 0.015 |
| SV | 7 |

These are configurable in `config/league.yaml` and can be replaced with league-specific values after the first season completes. In future seasons, the system can calculate denominators from actual league standings data.

### Replacement Level Calculation

Replacement level is the projected stat line of the best freely available player at each position — i.e., the first player who goes undrafted.

**Algorithm:**
1. Rank all players by total SGP at each eligible position
2. The number of "above replacement" players per position equals the number of starting roster slots across all teams that the position fills:
   - C: 10 (1 per team × 10 teams)
   - 1B: 10
   - 2B: 10
   - 3B: 10
   - SS: 10
   - IF: 10 (filled by 1B/2B/3B/SS-eligible players not already counted)
   - OF: 40 (4 per team × 10 teams)
   - UTIL: 20 (filled by best remaining hitters)
   - P: 90 (9 per team × 10 teams)
3. The replacement-level player at each position is player N+1 in that ranking
4. A player's Value Above Replacement (VAR) = their total SGP minus the replacement-level SGP for their most valuable position

**Rate stats handling:** AVG is converted to "marginal hits" (H - AVG_replacement × AB) so it scales with playing time. ERA/WHIP similarly use marginal earned runs and marginal baserunners weighted by IP.

## Draft Model

### Pipeline

1. **Load blended projections** from `data` module
2. **Calculate SGP values** via `sgp` module (denominators + replacement levels)
3. **Remove keepers** — load the 30 keepers from `config/league.yaml`, remove from player pool, recalculate replacement levels with reduced pool
4. **Rank remaining players** by VAR

### Interactive Draft Assistant

The draft assistant runs as a terminal loop:

1. **Pre-draft:** Display top-50 ranked players by VAR with position eligibility
2. **Each pick:**
   - If it's the user's pick: recommend top 3 available players by VAR, highlighting positional needs and category balance
   - Prompt user to enter who they drafted (or auto-select recommendation)
   - If it's another team's pick: prompt user to enter who was picked (player name search with fuzzy matching)
3. **After each pick:**
   - Remove drafted player from pool
   - Update user's roster composition and category balance
   - Recalculate positional scarcity (optional — only if speed allows)
   - Display updated top-10 available with any shift in recommendations
4. **Category balance tracker:** Running tally of the user's projected totals across all 10 categories, with warnings if any category falls significantly below the league-average pace

**Snake draft awareness:** The assistant knows the user picks 8th and calculates the gap between current pick and next pick (e.g., 12 picks until next turn). It factors this into recommendations — if a scarce position (e.g., C) has a steep drop-off and the gap to next pick is large, it may recommend reaching slightly.

### Roto-Specific Modeling

- **Rate stats (AVG, ERA, WHIP):** Weighted by playing time via marginal stats approach (see SGP Engine above)
- **Positional scarcity:** Driven by the replacement level gap — positions where top players far exceed replacement get a natural value boost
- **Category balance:** Flags if drafted roster is lopsided (e.g., power-heavy but steal-weak) and suggests pivoting

## In-Season Start/Sit Optimizer

### Weekly/Daily Optimization

1. **Pull current state** — via Yahoo API: roster, season standings, each team's cumulative stats
2. **Identify high-leverage categories** — calculate the SGP gap between user and the team above in each category. Small gaps = high-leverage targets; large gaps = potential punts
3. **Fetch updated projections** — rest-of-season or weekly projections for rostered players, factoring in schedule (games per week for each player's MLB team)
4. **Optimize lineup** — solve a **linear assignment problem** using `scipy.optimize.linear_sum_assignment` or a simple brute-force enumeration (feasible for a single team's ~15 active hitters across ~12 slots). Assign players to roster slots to maximize expected SGP gain in high-leverage categories, respecting position eligibility. For pitchers (9 undifferentiated slots), rank by projected SGP contribution and start the top 9.
5. **Output recommendations** — show optimal lineup with reasoning (e.g., "Start X over Y at UTIL — you're close to gaining a point in SB")

### Waiver Wire Suggestions (v1)

Scan free agents via Yahoo API, calculate SGP value for each, and flag pickups that would improve the user's weakest categories. Simple comparison: "Dropping player X and adding player Y gains you ~Z SGP in [category]."

### Standings-Aware Strategy

As the season progresses, the optimizer shifts weights toward categories where gaining ground is realistic and away from categories that are locked in (either dominant or hopeless). This is built into the "high-leverage categories" step — not a separate system.

## Tech Stack

- **Python 3.11+**
- `yahoo-fantasy-api` + `yahoo-oauth` — Yahoo API access
- `pandas` — data manipulation and projection blending
- `numpy` — SGP calculations and stat math
- `scipy` — lineup optimization (linear assignment)
- `requests` — HTTP client (for any direct API calls if needed)

## Interface

CLI-first via two main scripts:

- `python scripts/run_draft.py` — interactive draft assistant with terminal-based pick loop (see Draft Model section for full interaction flow)
- `python scripts/run_lineup.py` — shows current roster and optimized lineup recommendation with reasoning

No web UI or database for v1. Projections cached as CSV in `data/`, config as YAML in `config/`.

## Configuration

`config/league.yaml`:
- League ID
- Keeper list (all 30, with team assignments)
- Draft position
- Projection source weights
- SGP denominators (overridable)
- Roster slot definitions

`config/oauth.json` (gitignored):
- Yahoo OAuth2 client_id and client_secret

## Implementation Order

1. **Shared foundation** — auth, data ingestion, FanGraphs CSV parsing, projection blending, SGP engine, position eligibility
2. **Draft assistant** — draft rankings, interactive terminal loop, category balance tracking
3. **In-season optimizer** — standings-aware lineup optimization, waiver wire suggestions

Each phase gets its own implementation plan.

## Future Enhancements (Out of Scope for v1)

- Web UI
- ML-based projection blending
- Monte Carlo draft simulation
- Trade analyzer
- Opponent modeling
- Automated keeper suggestions based on Yahoo API data
- Auto-calculated SGP denominators from historical league standings
