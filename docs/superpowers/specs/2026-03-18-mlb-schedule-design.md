# MLB Schedule & Probable Pitchers Module

**Date:** 2026-03-18
**Status:** Approved

## Problem

The lineup optimizer has a `scale_by_schedule()` function that adjusts counting-stat projections by weekly game count, but `games_this_week` is never populated from real data — it always defaults to 6. Probable pitcher matchup data is also unavailable, so we can't identify two-start pitchers or show upcoming matchups.

## Solution

Add an MLB Stats API integration that fetches weekly schedule data (games per team + probable pitchers), caches it locally, and wires it into the existing lineup optimizer pipeline.

## Data Source

**MLB Stats API** via the `MLB-StatsAPI` Python package (`pip install MLB-StatsAPI`, import as `statsapi`).

- Free, no authentication required
- `statsapi.schedule(start_date, end_date)` returns games with probable pitcher fields
- `statsapi.get('teams', {'sportId': 1})` returns team name/abbreviation mappings
- **Must filter to `game_type == 'R'`** (regular season only) — the API also returns spring training (`S`) and exhibition (`E`) games

## Architecture

### New module: `src/fantasy_baseball/data/mlb_schedule.py`

Three layers:

1. **`fetch_week_schedule(start_date: str, end_date: str) -> dict`**
   - Calls `statsapi.schedule()` for the date range, filtering to regular-season games only (`game_type == 'R'`)
   - Calls `statsapi.get('teams', ...)` to build a full-name-to-abbreviation mapping
   - Normalizes all team abbreviations to FanGraphs format (see Team Abbreviation Normalization below)
   - Handles doubleheaders correctly: each game in a doubleheader is counted separately (both appear as individual entries from the API)
   - Probable pitcher fields may be empty strings (`""`) for unannounced starters — normalize these to `"TBD"`
   - Returns:
     ```python
     {
         "games_per_team": {"NYY": 7, "BOS": 6, ...},  # FanGraphs abbreviations
         "probable_pitchers": [
             {
                 "date": "2026-04-07",
                 "away_team": "NYY",
                 "home_team": "BOS",
                 "away_pitcher": "Gerrit Cole",
                 "home_pitcher": "Brayan Bello",
             },
             ...  # pitchers with "" names stored as "TBD"
         ],
         "team_abbrev_map": {"New York Yankees": "NYY", ...},  # cached for reuse
         "start_date": "2026-04-07",
         "end_date": "2026-04-13",
         "fetched_at": "2026-04-07T10:30:00",
     }
     ```

2. **`save_schedule_cache(data: dict, path: Path)` / `load_schedule_cache(path: Path) -> dict`**
   - JSON round-trip to `data/weekly_schedule.json`
   - Same pattern as existing `save_positions_cache` / `load_positions_cache`

3. **`get_week_schedule(start_date: str, end_date: str, cache_path: Path) -> dict`**
   - Main entry point
   - Tries live fetch; on success, saves to cache
   - On failure (network error, API down), falls back to cached data if `start_date`/`end_date` match
   - Returns `None` if no data available (API down + no matching cache). Callers degrade to defaults.

### Team Abbreviation Normalization

FanGraphs and MLB Stats API use different abbreviations for 7 of 30 teams:

| Team | FanGraphs | MLB Stats API |
|------|-----------|---------------|
| Arizona Diamondbacks | ARI | AZ |
| Chicago White Sox | CHW | CWS |
| Kansas City Royals | KCR | KC |
| San Diego Padres | SDP | SD |
| San Francisco Giants | SFG | SF |
| Tampa Bay Rays | TBR | TB |
| Washington Nationals | WSN | WSH |

A constant `MLB_TO_FANGRAPHS_ABBREV` dict in `mlb_schedule.py` maps MLB API abbreviations to FanGraphs abbreviations. All output from `fetch_week_schedule` uses FanGraphs abbreviations so that lookups against the projection DataFrames' `team` column work directly.

### New in `yahoo_roster.py`: `fetch_scoring_period(league) -> tuple[str, str]`

- Calls `league.current_week()` to get the current week number
- Calls `league.week_date_range(week)` to get `(start_date, end_date)` as `datetime.date` objects
- Converts to `"YYYY-MM-DD"` strings via `.isoformat()` before returning
- This ensures the schedule query matches Yahoo's exact scoring period boundaries, handling oddball lock days, short weeks, all-star break, etc.

### Integration in `run_lineup.py`

After connecting to Yahoo and before loading projections:

1. Call `fetch_scoring_period(league)` to get the week boundaries
2. Call `get_week_schedule(start_date, end_date, cache_path)` to get schedule data
3. After matching roster players to projections (using the projection row's `team` column), look up the team in `games_per_team` and pass the count to `scale_by_schedule()`. The `team` column in the projection DataFrames contains FanGraphs abbreviations (e.g., `"NYY"`, `"KCR"`), which match the normalized keys in `games_per_team` directly.
4. Apply the same schedule scaling to waiver wire free agent projections for consistency
5. After the pitcher lineup output, print a **"PROBABLE STARTERS THIS WEEK"** section:
   - List each roster pitcher's scheduled starts with date, opponent, and home/away indicator (`vs` for home, `@` for away)
   - Format dates as day-of-week abbreviations (Mon, Tue, etc.)
   - Flag two-start pitchers prominently (these are high-value weekly assets)

### Matching roster players to MLB teams

The blended projection DataFrames have a `team` column from FanGraphs containing abbreviations (e.g., `"NYY"`, `"KCR"`, `"SDP"`). The schedule module normalizes all abbreviations to FanGraphs format, so lookups against `games_per_team` are direct. If a player's team can't be matched, fall back to `DEFAULT_GAMES_PER_WEEK` (6).

## Output Example

```
PROBABLE STARTERS THIS WEEK (Apr 7-13)
  ** TWO-START PITCHERS **
    Gerrit Cole         NYY   Mon vs BOS, Sat @ TOR
    Corbin Burnes       BAL   Tue vs TBR, Sun @ NYY

  SINGLE START
    Max Fried           NYY   Wed vs BOS
    Spencer Strider     ATL   Fri @ MIA

  NO START ANNOUNCED
    Tyler Glasnow       LAD   TBD
```

## Error Handling

- MLB API unavailable: log warning, fall back to cache. If cache miss or stale dates, log warning and proceed with default 6 games/week for all teams. Never crash.
- Yahoo scoring period unavailable: fall back to Monday-Sunday of current week.
- Player team not found in schedule: use default 6 games/week.
- Empty probable pitcher names: display as "TBD" in output.

## Testing

- **Unit tests** for `fetch_week_schedule` response parsing (mock `statsapi` calls), including game_type filtering and doubleheader counting
- **Unit tests** for cache save/load/fallback logic
- **Unit tests** for team abbreviation normalization (all 7 divergent teams)
- **Unit tests** for `fetch_scoring_period` (mock Yahoo league object returning `datetime.date` objects)
- **New test file** `tests/test_data/test_mlb_schedule.py` for the above
- **New test file** `tests/test_lineup/test_scoring_period.py` for `fetch_scoring_period`

## Dependencies

- `MLB-StatsAPI` package — add `"MLB-StatsAPI"` to `pyproject.toml` under `[project.dependencies]`. Import as `statsapi`.

## Out of Scope

- Matchup quality adjustments (opponent batting stats affecting pitcher valuations) — tracked as a future TODO
- Recent performance weighting — separate TODO item
