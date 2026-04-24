# Fantasy Baseball Toolset

A data-driven Fantasy Baseball toolkit for Yahoo 5x5 roto leagues. Combines multiple projection systems, calculates Standings Gain Points (SGP), and recommends optimal draft picks and weekly lineups.

Built for Yahoo league 5652 (10-team, keeper league).

## Features

### Draft Assistant
An interactive CLI + live web dashboard for snake drafts:
- **two_closers + VONA strategy** — leverage-weighted recommendations with VONA (Value Over Next Available) scoring, responsive rate-stat leverage for AVG/ERA/WHIP, and staggered closer deadlines at rounds 8 and 14
- **SGP-based player rankings** — blends 5 projection systems (Steamer, ZiPS, ATC, THE BAT X, Oopsy), then ranks players by Value Above Replacement (VAR) or VONA
- **Category balance tracking** — monitors your roster's projected stats across all 10 categories and warns when you're falling behind
- **Closer alerts** — flags falling closers and enforces deadlines so you never punt saves
- **AVG floor** — demotes hitters that would tank your team batting average below .250
- **Positional scarcity** — flags when scarce positions (C, SS) are running thin
- **Traded pick support** — type `mine` on other teams' turns for picks traded to you, or `spacemen gausman` on your turn for picks traded away
- **Quoted team names** — `"crews control" chris sale` for ambiguous team names
- **Mock draft mode** — `--mock --position 8 --teams 10` for practice drafts
- **Live web dashboard** — Flask + htmx browser view with draft board (sortable by VAR or ADP), recommendations, projected standings, roster grid, and category balance bars
- **Fuzzy name search** — type partial or misspelled names and it finds the right player

### Draft Strategy Engine
Simulate and compare draft strategies with Monte Carlo analysis:
- **14 strategies** — default, nonzero_sv, avg_hedge, two_closers, three_closers, four_closers, no_punt, no_punt_opp, no_punt_stagger, no_punt_cap3, avg_anchor, closers_avg, balanced, anti_fragile
- **Opponent modeling** — assign strategies to specific opponents based on historical draft tendencies
- **ADP noise** — randomize opponent draft order to test robustness
- **Active roster modeling** — only counts stats from starting lineup, not bench
- **Monte Carlo projections** — correlated injury model, empirical stat variance, and roto scoring across 1000 simulated seasons

### In-Season Tools
A season dashboard and CLI tools that connect to Yahoo, analyze your standings position, and recommend optimal lineups, waiver moves, and trades:
- **Season dashboard** — Flask web UI for in-season management (standings, rosters, projections, game logs, waivers, trades)
- **Standings leverage** — identifies which categories are closest to gaining (or losing) a standings point, using sample-size-aware z-scores derived from this league's historical standings
- **Optimal hitter lineup** — uses the Hungarian algorithm to assign hitters to roster slots, maximizing leverage-weighted SGP
- **Pitcher ranking** — ranks pitchers by leverage-weighted SGP with matchup quality adjustments
- **Per-decision reasoning** — explains flex slot choices (e.g., "Start X over Y at UTIL — gains HR, RBI")
- **Waiver wire scanner** — scans free agents at every position, evaluates add/drop swaps, and ranks the top pickups by SGP gain
- **MLB schedule integration** — fetches weekly game counts and probable pitchers from the MLB Stats API, flags two-start pitchers
- **Schedule-aware projections** — scales counting stats by actual games per week (not a flat average)
- **Trade recommender** — proposes 1-for-1 trades that improve both sides, using leverage-weighted SGP to find mutually beneficial swaps across all league opponents
- **Recency weighting** — blends ROS projections with recent performance for start/sit decisions (validated via backtest: +1.9% next-week accuracy)

### Data & Analytics
- **Upstash KV (Redis)** — runtime store for the deployed season dashboard: standings, rosters, projections, transactions, refresh state
- **Local SQLite mirror** — `fantasy.db` is synced from Upstash on demand for offline analysis and backs the draft tooling (projection blending, draft history, game logs)
- **Empirical variance model** — stat variance and correlation matrices calibrated from 2022-2024 projection-vs-actual residuals, used in Monte Carlo simulation
- **SGP denominators derived from this league's history** — not generic constants
- **Game log tracking** — fetches and stores per-game batting and pitching stats from the MLB Stats API

## Setup

### 1. Install and configure

```bash
git clone https://github.com/alhart2015/FantasyBaseball.git
cd FantasyBaseball
pip install -e ".[dev]"
```

Then:
- Create `config/oauth.json` with Yahoo app credentials (`consumer_key`, `consumer_secret`).
- Copy `config/league.yaml.example` → `config/league.yaml` and fill in league id, team name, draft position, keepers, and roster slots.
- Download FanGraphs projection CSVs into `data/projections/{season_year}/` (Steamer, ZiPS, ATC, THE BAT X, Oopsy — hitters and pitchers).

### 2. Authenticate with Yahoo

The first script that needs Yahoo data will open your browser for OAuth and cache the token under `config/`.

## Draft Day

### 1. Cache player positions (run before draft)

```bash
python scripts/fetch_positions_mlb.py
```

Fills position eligibility gaps via the MLB Stats API for any players Yahoo doesn't return positions for.

### 2. Launch the draft assistant

```bash
python scripts/run_draft.py
```

For mock drafts:

```bash
python scripts/run_draft.py --mock --position 8 --teams 10
```

### 3. During the draft

- **Your pick:** Top 5 recommendations with VAR/VONA scores, closer alerts, and AVG warnings. Type a number (1-5) or player name.
- **Other teams' picks:** Type player name, optionally prefixed with team: `peanuts logan webb` or `"crews control" chris sale`
- **Traded picks to you:** Type `mine` on another team's turn to get your recommendations
- **Traded picks away:** Type `spacemen gausman` on your turn to assign to another team
- **Commands:** `skip` to skip, `quit` to exit

### 4. Simulate and compare strategies

```bash
python scripts/simulate_draft.py -s two_closers --scoring-mode vona
python scripts/simulate_draft.py -s two_closers --opponent-strategies "1:two_closers,5:three_closers"
python scripts/compare_strategies.py   # Full comparison across strategies (slow, ~10min)
```

## In-Season Usage

```bash
python scripts/run_season_dashboard.py   # Launch dashboard at localhost:5001 (syncs from Upstash first)
python scripts/run_lineup.py             # CLI lineup optimizer + waiver recommendations
python scripts/refresh_remote.py         # Trigger a remote refresh on the Render-hosted dashboard
```

The season dashboard is the primary entry point — it covers standings leverage, lineup optimization, waiver scanning, and trade recommendations in one UI. `run_lineup.py` is the CLI equivalent for lineup + waivers, useful when you want a terminal-only view.

`run_lineup.py` connects to Yahoo, fetches your roster, standings, and the MLB schedule, then prints:
1. **Category leverage** — which stats are most valuable to target this week
2. **Optimal hitter lineup** — slot assignments with reasoning on flex decisions
3. **Optimal pitcher lineup** — ranked by leverage-weighted SGP
4. **Probable starters** — matchups for your pitchers, flagging two-start pitchers
5. **Waiver recommendations** — top add/drop swaps with category impact

## How It Works

### Standings Gain Points (SGP)

In roto leagues, each stat category earns 1-10 standings points. SGP measures how many raw stats it takes to move up one place in the standings. Players are valued by how many standings points they contribute across all 10 categories. Denominators are derived from this league's historical standings, not generic constants.

### Value Above Replacement (VAR)

VAR = Player's total SGP - replacement-level SGP at their position. Scarce positions like C and SS have lower replacement levels, which naturally inflates the value of good players at those positions. Replacement levels recalculate per pick from the available pool to reflect live positional scarcity.

### VONA (Value Over Next Available)

VONA measures urgency — how much value you lose by waiting. For each player, it estimates what the best remaining player in the same bucket (hitter / SP / closer) will be after opponents make their picks. High VONA means "draft now or lose significant value." Used alongside leverage weighting to balance urgency against team category needs.

### two_closers Strategy

The current strategy: use VONA + leverage-weighted drafting to build a balanced roster, with responsive rate-stat leverage that steers toward AVG/ERA/WHIP when the team falls behind target. Drafts 2 closers by rounds 8 and 14 if the VONA engine hasn't already grabbed them. Validated through simulation against realistic opponent models based on 2024-2025 draft history.

### Monte Carlo Season Simulation

Each simulated season applies random injuries (45% of pitchers, 18% of hitters) and correlated stat variance to all players. Variance is drawn from multivariate normal distributions with empirically calibrated per-stat standard deviations (HR: ±34.3%, SB: ±71.5%, AVG: ±10.3%, ERA: ±25.2%) and correlation matrices so that related stats (e.g., HR and RBI) move together realistically. Injured players are replaced proportionally by replacement-level waiver pickups. Only active roster players (13 hitters, 9 pitchers) contribute stats. Roto standings are scored across 1000 iterations to produce win probabilities and category risk profiles. Team-specific management adjustments model the impact of in-season waiver moves and streaming.

## Project Structure

```
FantasyBaseball/
├── src/fantasy_baseball/
│   ├── analysis/      # Game logs, recency weighting
│   ├── auth/          # Yahoo OAuth2 authentication
│   ├── data/          # FanGraphs CSV parsing, projection blending, MLB schedule, KV store (Upstash + SQLite mirror)
│   ├── draft/         # Draft board, tracker, balance, recommender, strategies, search
│   ├── lineup/        # In-season optimizer: leverage, weighted SGP, optimizer, waivers, matchups
│   ├── models/        # Domain models: player, team, league, roster, standings, free agents, positions
│   ├── sgp/           # SGP engine: denominators, player values, replacement levels, VAR
│   ├── trades/        # Trade evaluation and multi-team trade search
│   ├── utils/         # Constants (variance/correlation matrices), position helpers, name normalization
│   ├── web/           # Flask dashboards for draft and in-season management (refresh pipeline, season routes)
│   ├── scoring.py     # Shared roto scoring and team stat projection
│   ├── simulation.py  # Monte Carlo season simulation with correlated variance
│   └── config.py      # YAML config loading
├── scripts/
│   ├── run_draft.py                # Interactive draft assistant CLI (+ mock mode)
│   ├── run_lineup.py               # In-season lineup optimizer CLI
│   ├── run_season_dashboard.py     # In-season web dashboard (syncs from Upstash)
│   ├── simulate_draft.py           # Draft simulation with configurable strategies
│   ├── compare_strategies.py       # Side-by-side strategy comparison
│   ├── build_db.py                 # Rebuild local SQLite from source files
│   ├── calibrate_variance.py       # Calibrate stat variance from projection-vs-actual residuals
│   ├── derive_sgp_denominators.py  # Recompute SGP denominators from this league's standings history
│   ├── fetch_positions_mlb.py      # Fill position gaps via MLB Stats API
│   ├── fetch_actual_stats.py       # Fetch actual season stats for backtest/calibration
│   ├── refresh_remote.py           # Trigger refresh on the Render-hosted dashboard
│   ├── sync_redis.py               # Sync between local SQLite KV and remote Upstash
│   ├── freeze_preseason_baseline.py # Snapshot the preseason projection baseline
│   ├── save_roster.py              # Persist current Yahoo roster snapshot
│   ├── replay_picks.py             # Replay draft picks against the board
│   ├── rescore_transactions.py     # Re-evaluate historical waiver/trade transactions
│   ├── migrate_standings_history.py # Migrate / backfill standings history
│   ├── backfill_roster_history.py  # Backfill historical weekly rosters
│   ├── export_history.py           # Export league history for analysis
│   ├── compare_sgp_local_vs_remote.py # Diff SGP outputs across local vs remote KV
│   ├── analyze_draft.py            # Post-draft projection + Monte Carlo analysis
│   ├── analyze_mock.py             # Post-mock-draft projection analysis
│   ├── analyze_history.py          # Historical draft tendency analysis
│   ├── backtest_2025.py            # Backtest simulation against 2025 actual results
│   ├── backtest_recency.py         # Recency weighting backtest
│   └── smoke_test.py               # End-to-end smoke test of the refresh pipeline
├── data/
│   ├── projections/    # FanGraphs CSV files (not committed)
│   ├── fantasy.db      # Local SQLite mirror (not committed)
│   └── player_positions.json  # Cached Yahoo positions (not committed)
├── config/
│   ├── league.yaml         # League settings + keepers
│   ├── league.yaml.example # Template
│   └── oauth.json          # Yahoo credentials (gitignored)
├── CLAUDE.md           # Claude Code guidance (subsystem CLAUDE.md files live alongside the code)
├── TODO.md             # In-season enhancement roadmap
└── tests/              # 1267 tests
```

## Running Tests

```bash
pytest -v
```

## Tech Stack

- **Python 3.11+** with pandas, numpy, scipy
- **yahoo-fantasy-api** + **yahoo-oauth** for Yahoo Fantasy API access
- **MLB-StatsAPI** for weekly schedule, probable pitchers, and game logs
- **Upstash Redis** for the deployed dashboard's runtime KV store, with a local **SQLite** mirror for offline analysis and the draft tooling
- **Flask** + **htmx** for draft and season dashboards
- **Render** for hosting the season dashboard
- **pytest** for testing (1267 tests)
