# Projection Data

## Refreshing ROS Projections (In-Season)

Steamer and ZiPS update their Rest-of-Season projections daily once the season starts. Refreshing these files gives the lineup optimizer more accurate player valuations without any code changes.

### Steps

1. Go to [FanGraphs Projections](https://www.fangraphs.com/projections)
2. Select projection system (Steamer, ZiPS, or ATC)
3. Select **"Rest of Season"** from the time period dropdown (not "Pre-Season")
4. Export **Hitters** CSV, rename to `{system}-hitters.csv` (e.g., `steamer-hitters.csv`)
5. Export **Pitchers** CSV, rename to `{system}-pitchers.csv` (e.g., `steamer-pitchers.csv`)
6. Drop into this directory, overwriting the existing files

Repeat for each system you use (configured in `config/league.yaml` under `projections.systems`).

### Current systems

- `steamer-hitters.csv` / `steamer-pitchers.csv`
- `zips-hitters.csv` / `zips-pitchers.csv`
- `atc-hitters.csv` / `atc-pitchers.csv`

### How often?

Every 1-2 weeks during the season is plenty. More often doesn't hurt but has diminishing returns — the models update daily but week-to-week changes are small for most players.

### 2025 preseason files

The `*-2025.csv` files are archived preseason projections used by the backtest scripts. Don't overwrite these.
