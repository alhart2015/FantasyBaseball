# Fantasy Baseball Toolset

A data-driven Fantasy Baseball toolkit for Yahoo 5x5 roto leagues. Combines multiple projection systems, calculates Standings Gain Points (SGP), and recommends optimal draft picks and weekly lineups.

Built for Yahoo league 5652 (10-team, keeper league).

## Features

### Draft Assistant
An interactive CLI + live web dashboard for snake drafts:
- **no_punt strategy** — leverage-weighted recommendations with AVG floor protection and closer deadline (R9 backstop), validated at ~18% win rate against realistic opponents (1.8x base rate)
- **SGP-based player rankings** — blends Steamer, ZiPS, and ATC projections, then ranks players by Value Above Replacement (VAR)
- **Category balance tracking** — monitors your roster's projected stats across all 10 categories and warns when you're falling behind
- **Closer alerts** — flags falling closers and enforces a deadline so you never punt saves
- **AVG floor** — demotes hitters that would tank your team batting average below .250
- **Positional scarcity** — flags when scarce positions (C, SS) are running thin
- **Traded pick support** — type `mine` on other teams' turns for picks traded to you, or `spacemen gausman` on your turn for picks traded away
- **Quoted team names** — `"crews control" chris sale` for ambiguous team names
- **Mock draft mode** — `--mock --position 6 --teams 10` for practice drafts
- **Live web dashboard** — Flask + htmx browser view with draft board (sortable by VAR or ADP), recommendations, projected standings, roster grid, and category balance bars
- **Fuzzy name search** — type partial or misspelled names and it finds the right player

### Draft Strategy Engine
Simulate and compare draft strategies with Monte Carlo analysis:
- **10 strategies** — default, no_punt, no_punt_opp, three_closers, avg_hedge, avg_anchor, closers_avg, balanced, nonzero_sv, anti_fragile
- **Opponent modeling** — assign strategies to specific opponents based on historical draft tendencies
- **ADP noise** — randomize opponent draft order to test robustness
- **Active roster modeling** — only counts stats from starting lineup, not bench
- **Monte Carlo projections** — injury model, stat variance, and roto scoring across 1000 simulated seasons

### In-Season Lineup Optimizer
A CLI tool that connects to Yahoo, analyzes your standings position, and recommends the optimal lineup + waiver moves:
- **Standings leverage** — identifies which categories are closest to gaining (or losing) a standings point
- **Optimal hitter lineup** — uses the Hungarian algorithm to assign hitters to roster slots, maximizing leverage-weighted SGP
- **Pitcher ranking** — ranks pitchers by leverage-weighted SGP and recommends who to start
- **Per-decision reasoning** — explains flex slot choices (e.g., "Start X over Y at UTIL — gains HR, RBI")
- **Waiver wire scanner** — scans free agents at every position, evaluates add/drop swaps, and ranks the top pickups by SGP gain
- **MLB schedule integration** — fetches weekly game counts and probable pitchers from the MLB Stats API, flags two-start pitchers
- **Schedule-aware projections** — scales counting stats by actual games per week (not a flat average)

## Setup

See [SETUP.md](SETUP.md) for detailed step-by-step instructions (including for non-technical users).

### Quick Start

```bash
git clone https://github.com/alhart2015/FantasyBaseball.git
cd FantasyBaseball
pip install -e ".[dev]"
```

Create `config/oauth.json` with Yahoo app credentials, download FanGraphs projection CSVs to `data/projections/`, and configure `config/league.yaml`.

## Draft Day

### 1. Fetch player positions (run once before draft)

```bash
python scripts/fetch_positions.py
```

### 2. Launch the draft assistant

```bash
python scripts/run_draft.py
```

For mock drafts:

```bash
python scripts/run_draft.py --mock --position 8 --teams 10
```

### 3. During the draft

- **Your pick:** Top 5 recommendations with VAR scores, closer alerts, and AVG warnings. Type a number (1-5) or player name.
- **Other teams' picks:** Type player name, optionally prefixed with team: `peanuts logan webb` or `"crews control" chris sale`
- **Traded picks to you:** Type `mine` on another team's turn to get your recommendations
- **Traded picks away:** Type `spacemen gausman` on your turn to assign to another team
- **Commands:** `skip` to skip, `quit` to exit

### 4. Simulate and compare strategies

```bash
python scripts/simulate_draft.py -s no_punt
python scripts/simulate_draft.py -s no_punt --opponent-strategies "1:three_closers,5:three_closers"
python scripts/monte_carlo.py -n 1000
```

## In-Season Usage

```bash
python scripts/run_lineup.py
```

Connects to Yahoo, fetches your roster, standings, and the MLB schedule, then prints:
1. **Category leverage** — which stats are most valuable to target this week
2. **Optimal hitter lineup** — slot assignments with reasoning on flex decisions
3. **Optimal pitcher lineup** — ranked by leverage-weighted SGP
4. **Probable starters** — matchups for your pitchers, flagging two-start pitchers
5. **Waiver recommendations** — top 5 add/drop swaps with category impact

## How It Works

### Standings Gain Points (SGP)

In roto leagues, each stat category earns 1-10 standings points. SGP measures how many raw stats it takes to move up one place in the standings. Players are valued by how many standings points they contribute across all 10 categories.

### Value Above Replacement (VAR)

VAR = Player's total SGP - replacement-level SGP at their position. Scarce positions like C and SS have lower replacement levels, which naturally inflates the value of good players at those positions.

### no_punt Strategy

The winning strategy: use leverage-weighted drafting to build a balanced roster, enforce an AVG floor of .250 to avoid punting batting average, and draft a closer by round 9 if the leverage engine hasn't already grabbed one. Validated through simulation against realistic opponent models based on 2024-2025 draft history.

### Monte Carlo Season Simulation

Each simulated season applies random injuries (45% of pitchers, 18% of hitters) and stat variance (12% std dev) to all players. Injured players are replaced proportionally by replacement-level waiver pickups. Only active roster players (13 hitters, 9 pitchers) contribute stats. Roto standings are scored across 1000 iterations to produce win probabilities and category risk profiles.

## Project Structure

```
FantasyBaseball/
├── src/fantasy_baseball/
│   ├── auth/           # Yahoo OAuth2 authentication
│   ├── data/           # FanGraphs CSV parsing, projection blending, MLB schedule
│   ├── draft/          # Draft board, tracker, balance, recommender, strategies, search
│   ├── lineup/         # In-season optimizer: leverage, weighted SGP, optimizer, waivers
│   ├── sgp/            # SGP engine: denominators, player values, replacement levels, VAR
│   ├── utils/          # Constants, position helpers, name normalization
│   ├── web/            # Flask dashboard for draft visualization
│   └── config.py       # YAML config loading
├── scripts/
│   ├── run_draft.py        # Interactive draft assistant CLI (+ mock mode)
│   ├── run_lineup.py       # In-season lineup optimizer CLI
│   ├── simulate_draft.py   # Draft simulation with configurable strategies
│   ├── monte_carlo.py      # Monte Carlo season projection
│   ├── fetch_positions.py  # Cache Yahoo position data
│   ├── analyze_history.py  # Historical draft tendency analysis
│   ├── analyze_mock.py     # Post-mock-draft projection analysis
│   └── backtest_2025.py    # Backtest simulation against 2025 actual results
├── data/
│   ├── projections/    # FanGraphs CSV files (not committed)
│   └── player_positions.json  # Cached Yahoo positions (not committed)
├── config/
│   ├── league.yaml     # League settings + keepers
│   └── oauth.json      # Yahoo credentials (gitignored)
├── docs/
│   └── superpowers/    # Design specs and implementation plans
├── SETUP.md            # Setup guide for new users
├── TODO.md             # In-season enhancement roadmap
└── tests/              # 208 tests
```

## Running Tests

```bash
pytest -v
```

## Tech Stack

- **Python 3.11+** with pandas, numpy, scipy
- **yahoo-fantasy-api** + **yahoo-oauth** for Yahoo Fantasy API access
- **MLB-StatsAPI** for weekly schedule and probable pitcher data
- **Flask** + **htmx** for the draft dashboard
- **pytest** for testing
