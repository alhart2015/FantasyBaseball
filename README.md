# Fantasy Baseball Toolset

A data-driven Fantasy Baseball toolkit for Yahoo 5x5 roto leagues. Combines multiple projection systems, calculates Standings Gain Points (SGP), and recommends optimal draft picks and weekly lineups.

Built for the **Phantoms of the Outfield** league (Yahoo league 5652).

## Features

### Draft Assistant
An interactive CLI + live web dashboard for snake drafts:
- **SGP-based player rankings** — blends Steamer, ZiPS, and ATC projections, then ranks players by Value Above Replacement (VAR)
- **Positional scarcity** — flags when scarce positions (C, SS) are running thin
- **Category balance tracking** — monitors your roster's projected stats across all 10 categories and warns when you're falling behind
- **Fuzzy name search** — type partial or misspelled names and it finds the right player
- **Snake draft awareness** — knows your pick position and calculates the gap to your next turn
- **Live web dashboard** — Flask + htmx browser view that updates in real-time as you enter picks in the terminal

### In-Season Lineup Optimizer *(coming soon)*
- Weekly start/sit recommendations based on standings position
- Waiver wire suggestions targeting your weakest categories
- Standings-aware strategy that shifts focus as the season progresses

## Setup

### Prerequisites
- Python 3.11+
- A Yahoo Developer app ([register here](https://developer.yahoo.com/apps/))
  - Select "Confidential Client"
  - Enable "Fantasy Sports" with Read access
  - Redirect URI: `oob`

### Installation

```bash
git clone https://github.com/alhart2015/FantasyBaseball.git
cd FantasyBaseball
pip install -e ".[dev]"
```

### Yahoo Authentication

Create `config/oauth.json` with your Yahoo app credentials:

```json
{
    "consumer_key": "your-key-here",
    "consumer_secret": "your-secret-here"
}
```

Run any script that uses Yahoo's API — on first run, it will open a browser window for authorization. Paste the verification code back into the terminal. The token is cached for future use.

### Projection Data

Download projection CSVs from [FanGraphs](https://www.fangraphs.com/projections):

1. Select a projection system (Steamer, ZiPS, or ATC)
2. Export Hitters CSV → save to `data/projections/`
3. Export Pitchers CSV → save to `data/projections/`

Supported file naming:
- `steamer_hitters.csv` / `steamer_pitchers.csv`
- `fangraphs-leaderboard-projections-steamer-hitters.csv` (FanGraphs default export name)

Configure which systems to blend in `config/league.yaml`:

```yaml
projections:
  systems:
    - steamer
    - zips
    - atc
  weights:
    steamer: 0.33
    zips: 0.33
    atc: 0.34
```

### League Configuration

Copy the example config and customize:

```bash
cp config/league.yaml.example config/league.yaml
```

Key settings in `config/league.yaml`:
- `league.id` — your Yahoo league ID
- `league.team_name` — your team name (for keeper matching)
- `draft.position` — your snake draft pick position (1-indexed)
- `keepers` — all 30 keepers across the league
- `roster_slots` — your league's roster configuration
- `sgp_denominators` — tune these if default values don't match your league

## Draft Day

### 1. Fetch player positions (run once before draft)

```bash
python scripts/fetch_positions.py
```

This caches Yahoo position eligibility data to `data/player_positions.json`.

### 2. Launch the draft assistant

```bash
python scripts/run_draft.py
```

This starts the CLI in your terminal and a web dashboard at `http://localhost:5000`.

### 3. During the draft

- **Your pick:** The assistant shows top 5 recommendations with VAR scores, positional need flags, and category balance. Type a player name or enter a number (1-5) to select a recommendation.
- **Other teams' picks:** Type the drafted player's name. Fuzzy matching handles misspellings.
- **Commands:** `skip` to skip a pick, `quit` to exit.

The web dashboard updates automatically after each pick.

## How It Works

### Standings Gain Points (SGP)

In roto leagues, each stat category earns 1-10 standings points. SGP measures how many raw stats it takes to move up one place in the standings. Players are valued by how many standings points they contribute across all 10 categories.

**Counting stats** (R, HR, RBI, SB, W, K, SV): SGP = stat / denominator

**Rate stats** (AVG, ERA, WHIP): Converted to "marginal" counting stats weighted by playing time, so a part-timer with a high average isn't overvalued vs. an everyday player with a slightly lower average.

### Value Above Replacement (VAR)

VAR = Player's total SGP − replacement-level SGP at their position

Replacement level is the SGP of the best freely available player at each position (the first player who goes undrafted). Scarce positions like C and SS have lower replacement levels, which naturally inflates the value of good players at those positions.

### Projection Blending

Multiple projection systems are combined via weighted average. Counting stats are averaged directly. Rate stats (AVG, ERA, WHIP) are recomputed from blended component stats (H/AB for AVG, ER/IP for ERA) to maintain mathematical consistency.

## Project Structure

```
FantasyBaseball/
├── src/fantasy_baseball/
│   ├── auth/           # Yahoo OAuth2 authentication
│   ├── data/           # FanGraphs CSV parsing, projection blending, Yahoo player data
│   ├── draft/          # Draft board, tracker, balance, recommender, search
│   ├── sgp/            # SGP engine: denominators, player values, replacement levels, VAR
│   ├── utils/          # Constants, position helpers, name normalization
│   ├── web/            # Flask dashboard for draft visualization
│   └── config.py       # YAML config loading
├── scripts/
│   ├── run_draft.py    # Interactive draft assistant CLI
│   └── fetch_positions.py  # Cache Yahoo position data
├── data/
│   ├── projections/    # FanGraphs CSV files (not committed)
│   └── player_positions.json  # Cached Yahoo positions (not committed)
├── config/
│   ├── league.yaml     # League settings + keepers
│   └── oauth.json      # Yahoo credentials (gitignored)
└── tests/              # 127+ tests
```

## Running Tests

```bash
pytest -v
```

## Tech Stack

- **Python 3.11+** with pandas, numpy, scipy
- **yahoo-fantasy-api** + **yahoo-oauth** for Yahoo Fantasy API access
- **Flask** + **htmx** for the draft dashboard
- **pytest** for testing
