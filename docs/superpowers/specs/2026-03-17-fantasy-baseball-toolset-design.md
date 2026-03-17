# Fantasy Baseball Toolset — Design Spec

## Overview

A CLI-based Fantasy Baseball toolset for a Yahoo 10-team 5x5 roto league (league ID 5652). Two main components: a **draft assistant** using Standings Gain Points (SGP) value-based drafting, and an **in-season start/sit optimizer** that recommends lineups to maximize standings gain.

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
│   ├── draft/          # SGP calculations, draft rankings, draft assistant
│   ├── lineup/         # Weekly start/sit optimizer
│   └── utils/          # Shared helpers (stat mappings, constants)
├── data/               # Cached projections, keeper lists, historical SGP baselines
├── scripts/            # CLI entry points (run_draft.py, run_lineup.py)
├── tests/
└── config/             # League settings, OAuth credentials (gitignored)
```

## Authentication & Data Sources

### Yahoo OAuth2

- User registers a Yahoo Developer app at https://developer.yahoo.com/apps/
- `client_id` and `client_secret` stored in `config/oauth.json` (gitignored)
- On first run, OAuth2 flow opens browser for Yahoo login, token cached for reuse
- Library: `yahoo-fantasy-api` + `yahoo-oauth`

### Data Sources

1. **Yahoo Fantasy API** — league settings, rosters, keepers, live stats, free agents, Yahoo's own projections
2. **FanGraphs public projections** — Steamer, ZiPS, ATC (scraped or downloaded as CSV)
3. **Blended projections** — weighted average of available systems per player, configurable weights (default: equal)

## Draft Model: SGP-Based Value Drafting

### Standings Gain Points (SGP)

In a 10-team roto league, each stat category earns 1–10 standings points. SGP measures how many raw stats it takes to move up one standings place. A player's value is the sum of standings points they contribute across all 10 categories.

### Pipeline

1. **Calculate SGP denominators** — using historical league data or published baselines, determine the cost of one standings point per category (e.g., ~10 HR = 1 standings point, ~0.15 ERA = 1 standings point)
2. **Set replacement level** — based on league depth (250 rostered players), determine the baseline free-agent-level player at each position
3. **Score every player** — for each draftable player: take blended projections, convert each stat to SGP, sum across categories = total SGP value. Subtract replacement level for their position = Value Above Replacement (VAR)
4. **Remove keepers** — load the 30 keepers, remove from pool, recalculate replacement levels
5. **Draft assistant** — at each pick, show top available players ranked by VAR, flagging positional needs. Accounts for snake draft position (picks 8, 13, 28, 33, etc.) to suggest when to reach for scarce positions vs. take best available

### Roto-Specific Modeling

- **Rate stats (AVG, ERA, WHIP):** Weighted by playing time (AB for AVG, IP for ERA/WHIP) so a high-AVG part-timer isn't overvalued vs. a slightly-lower-AVG everyday player
- **Positional scarcity:** C and SS are typically shallow; the model inflates value for scarce positions where the top-to-replacement drop-off is steepest
- **Category balance:** Flags if drafted roster is lopsided (e.g., power-heavy but steal-weak) and suggests pivoting

## In-Season Start/Sit Optimizer

### Weekly/Daily Optimization

1. **Pull current state** — via Yahoo API: roster, season standings, each team's cumulative stats
2. **Identify high-leverage categories** — calculate the SGP gap between user and the team above in each category. Small gaps = high-leverage targets; large gaps = potential punts
3. **Fetch updated projections** — rest-of-season or weekly projections for rostered players, factoring in schedule (games per week for each player's MLB team)
4. **Optimize lineup** — assign players to roster slots to maximize expected SGP gain, respecting position eligibility. IF and UTIL slots provide flexibility. For pitchers (9 undifferentiated slots), balance SP upside (W, K) against RP stability (SV, low ratios)
5. **Output recommendations** — show optimal lineup with reasoning (e.g., "Start X over Y at UTIL — you're close to gaining a point in SB")

### Additional Features

- **Waiver wire suggestions** — scan free agents, flag pickups that improve weakest categories
- **Standings-aware strategy** — late-season, shift focus to categories where gaining ground is realistic

## Tech Stack

- **Python 3.11+**
- `yahoo-fantasy-api` + `yahoo-oauth` — Yahoo API access
- `pandas` — data manipulation and projection blending
- `numpy` — SGP calculations and stat math
- `requests` + `beautifulsoup4` — scraping FanGraphs projections

## Interface

CLI-first via two main scripts:

- `python scripts/run_draft.py` — interactive draft assistant. Shows ranked player board, mark players as drafted, real-time updated recommendations
- `python scripts/run_lineup.py` — shows current roster and optimized lineup recommendation with reasoning

No web UI or database for v1. Projections and config cached as flat files (CSV/JSON). Web interface planned for later.

## Configuration

`config/league.yaml`:
- League ID
- Keeper list (all 30)
- Draft position
- Projection source weights
- Roster slot definitions

`config/oauth.json` (gitignored):
- Yahoo OAuth2 client_id and client_secret

## Future Enhancements (Out of Scope for v1)

- Web UI
- ML-based projection blending
- Monte Carlo draft simulation
- Trade analyzer
- Opponent modeling
