# MLB Schedule & Probable Pitchers Implementation Plan

> **For agentic workers:** REQUIRED: Use superpowers:subagent-driven-development (if subagents available) or superpowers:executing-plans to implement this plan. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Populate `games_this_week` from real MLB schedule data and display probable pitcher matchups (flagging two-start pitchers) in the lineup optimizer.

**Architecture:** New `data/mlb_schedule.py` module fetches schedule data from the MLB Stats API (`MLB-StatsAPI` package), normalizes team abbreviations to FanGraphs format, and caches results to JSON. A new `fetch_scoring_period()` function in `yahoo_roster.py` gets the current Yahoo scoring week boundaries. `run_lineup.py` wires it all together.

**Tech Stack:** Python 3.11+, `MLB-StatsAPI` package (import as `statsapi`), `pytest`

**Spec:** `docs/superpowers/specs/2026-03-18-mlb-schedule-design.md`

---

### Task 1: Add MLB-StatsAPI dependency

**Files:**
- Modify: `pyproject.toml:9-19`

- [ ] **Step 1: Add the dependency**

In `pyproject.toml`, add `"MLB-StatsAPI>=1.7"` to the `dependencies` list:

```toml
dependencies = [
    "yahoo-fantasy-api>=2.12",
    "yahoo-oauth>=2.0",
    "pandas>=2.0",
    "numpy>=1.24",
    "scipy>=1.11",
    "requests>=2.31",
    "pyyaml>=6.0",
    "flask>=3.0",
    "waitress>=3.0",
    "MLB-StatsAPI>=1.7",
]
```

- [ ] **Step 2: Install the dependency**

Run: `pip install -e ".[dev]"`
Expected: Installs successfully, `statsapi` importable.

- [ ] **Step 3: Verify import works**

Run: `python -c "import statsapi; print(statsapi.__version__)"`
Expected: Prints a version number (e.g., `1.7.2`).

- [ ] **Step 4: Commit**

```bash
git add pyproject.toml
git commit -m "feat: add MLB-StatsAPI dependency for schedule data"
```

---

### Task 2: Team abbreviation normalization constant

**Files:**
- Create: `src/fantasy_baseball/data/mlb_schedule.py`
- Create: `tests/test_data/test_mlb_schedule.py`

- [ ] **Step 1: Write the failing test**

Create `tests/test_data/test_mlb_schedule.py`:

```python
import pytest
from fantasy_baseball.data.mlb_schedule import (
    MLB_TO_FANGRAPHS_ABBREV,
    normalize_team_abbrev,
)


class TestNormalizeTeamAbbrev:
    def test_passthrough_for_matching_abbrevs(self):
        assert normalize_team_abbrev("NYY") == "NYY"
        assert normalize_team_abbrev("BOS") == "BOS"
        assert normalize_team_abbrev("LAD") == "LAD"

    def test_converts_divergent_abbreviations(self):
        assert normalize_team_abbrev("AZ") == "ARI"
        assert normalize_team_abbrev("CWS") == "CHW"
        assert normalize_team_abbrev("KC") == "KCR"
        assert normalize_team_abbrev("SD") == "SDP"
        assert normalize_team_abbrev("SF") == "SFG"
        assert normalize_team_abbrev("TB") == "TBR"
        assert normalize_team_abbrev("WSH") == "WSN"

    def test_all_seven_divergent_teams_in_mapping(self):
        assert len(MLB_TO_FANGRAPHS_ABBREV) == 7
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_data/test_mlb_schedule.py -v`
Expected: FAIL with `ModuleNotFoundError` or `ImportError`.

- [ ] **Step 3: Write minimal implementation**

Create `src/fantasy_baseball/data/mlb_schedule.py`:

```python
"""Fetch and cache MLB weekly schedule and probable pitchers."""

import json
import logging
from datetime import datetime
from pathlib import Path

logger = logging.getLogger(__name__)

# MLB Stats API abbreviations that differ from FanGraphs
MLB_TO_FANGRAPHS_ABBREV: dict[str, str] = {
    "AZ": "ARI",
    "CWS": "CHW",
    "KC": "KCR",
    "SD": "SDP",
    "SF": "SFG",
    "TB": "TBR",
    "WSH": "WSN",
}


def normalize_team_abbrev(mlb_abbrev: str) -> str:
    """Convert an MLB Stats API team abbreviation to FanGraphs format."""
    return MLB_TO_FANGRAPHS_ABBREV.get(mlb_abbrev, mlb_abbrev)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_data/test_mlb_schedule.py -v`
Expected: All 3 tests PASS.

- [ ] **Step 5: Commit**

```bash
git add src/fantasy_baseball/data/mlb_schedule.py tests/test_data/test_mlb_schedule.py
git commit -m "feat: add team abbreviation normalization for MLB-to-FanGraphs mapping"
```

---

### Task 3: Schedule parsing logic (fetch_week_schedule)

**Files:**
- Modify: `src/fantasy_baseball/data/mlb_schedule.py`
- Modify: `tests/test_data/test_mlb_schedule.py`

This task adds `fetch_week_schedule()` which calls `statsapi.schedule()` and `statsapi.get('teams', ...)`, then parses the results into the normalized structure. We mock `statsapi` in tests.

- [ ] **Step 1: Write the failing test for game counting and filtering**

Add to `tests/test_data/test_mlb_schedule.py`:

```python
from unittest.mock import patch, MagicMock
from fantasy_baseball.data.mlb_schedule import fetch_week_schedule


def _make_game(away_name, home_name, game_date, game_type="R",
               away_pitcher="", home_pitcher=""):
    """Helper to build a fake statsapi.schedule() game entry."""
    return {
        "away_name": away_name,
        "home_name": home_name,
        "game_date": game_date,
        "game_type": game_type,
        "away_probable_pitcher": away_pitcher,
        "home_probable_pitcher": home_pitcher,
    }


def _mock_teams_response():
    """Fake response from statsapi.get('teams', ...)."""
    return {
        "teams": [
            {"name": "New York Yankees", "teamName": "Yankees", "abbreviation": "NYY"},
            {"name": "Boston Red Sox", "teamName": "Red Sox", "abbreviation": "BOS"},
            {"name": "Arizona Diamondbacks", "teamName": "Diamondbacks", "abbreviation": "AZ"},
            {"name": "Kansas City Royals", "teamName": "Royals", "abbreviation": "KC"},
            {"name": "Tampa Bay Rays", "teamName": "Rays", "abbreviation": "TB"},
            {"name": "Chicago White Sox", "teamName": "White Sox", "abbreviation": "CWS"},
            {"name": "San Diego Padres", "teamName": "Padres", "abbreviation": "SD"},
            {"name": "San Francisco Giants", "teamName": "Giants", "abbreviation": "SF"},
            {"name": "Washington Nationals", "teamName": "Nationals", "abbreviation": "WSH"},
        ],
    }


class TestFetchWeekSchedule:
    @patch("fantasy_baseball.data.mlb_schedule.statsapi")
    def test_counts_games_per_team_with_fangraphs_abbrevs(self, mock_api):
        mock_api.schedule.return_value = [
            _make_game("New York Yankees", "Boston Red Sox", "2026-04-07"),
            _make_game("New York Yankees", "Boston Red Sox", "2026-04-08"),
            _make_game("Kansas City Royals", "Boston Red Sox", "2026-04-09"),
        ]
        mock_api.get.return_value = _mock_teams_response()

        result = fetch_week_schedule("2026-04-07", "2026-04-13")

        assert result["games_per_team"]["NYY"] == 2
        assert result["games_per_team"]["BOS"] == 3
        assert result["games_per_team"]["KCR"] == 1  # KC -> KCR

    @patch("fantasy_baseball.data.mlb_schedule.statsapi")
    def test_filters_out_non_regular_season_games(self, mock_api):
        mock_api.schedule.return_value = [
            _make_game("New York Yankees", "Boston Red Sox", "2026-04-07", game_type="R"),
            _make_game("New York Yankees", "Boston Red Sox", "2026-04-07", game_type="S"),
            _make_game("New York Yankees", "Boston Red Sox", "2026-04-07", game_type="E"),
        ]
        mock_api.get.return_value = _mock_teams_response()

        result = fetch_week_schedule("2026-04-07", "2026-04-13")

        assert result["games_per_team"]["NYY"] == 1
        assert result["games_per_team"]["BOS"] == 1

    @patch("fantasy_baseball.data.mlb_schedule.statsapi")
    def test_counts_doubleheader_games_separately(self, mock_api):
        mock_api.schedule.return_value = [
            _make_game("New York Yankees", "Boston Red Sox", "2026-04-07"),
            _make_game("New York Yankees", "Boston Red Sox", "2026-04-07"),  # DH game 2
        ]
        mock_api.get.return_value = _mock_teams_response()

        result = fetch_week_schedule("2026-04-07", "2026-04-13")

        assert result["games_per_team"]["NYY"] == 2
        assert result["games_per_team"]["BOS"] == 2
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_data/test_mlb_schedule.py::TestFetchWeekSchedule -v`
Expected: FAIL with `ImportError` (function doesn't exist yet).

- [ ] **Step 3: Write the implementation**

Add to `src/fantasy_baseball/data/mlb_schedule.py`:

```python
import statsapi
from collections import defaultdict


def _build_team_name_map() -> dict[str, str]:
    """Build a mapping from full team name to FanGraphs abbreviation.

    Fetches team data from the MLB Stats API and normalizes abbreviations.
    """
    response = statsapi.get("teams", {"sportId": 1})
    name_map: dict[str, str] = {}
    for team in response.get("teams", []):
        full_name = team.get("teamName", "")
        # Also map the full franchise name (e.g., "New York Yankees")
        franchise_name = team.get("name", "")
        mlb_abbrev = team.get("abbreviation", "")
        fg_abbrev = normalize_team_abbrev(mlb_abbrev)
        if full_name:
            name_map[full_name] = fg_abbrev
        if franchise_name:
            name_map[franchise_name] = fg_abbrev
    return name_map


def fetch_week_schedule(start_date: str, end_date: str) -> dict:
    """Fetch weekly schedule and probable pitchers from MLB Stats API.

    Args:
        start_date: Start of scoring period as "YYYY-MM-DD".
        end_date: End of scoring period as "YYYY-MM-DD".

    Returns:
        Dict with games_per_team, probable_pitchers, metadata.
        All team abbreviations use FanGraphs format.
    """
    games = statsapi.schedule(start_date=start_date, end_date=end_date)
    team_name_map = _build_team_name_map()

    games_per_team: dict[str, int] = defaultdict(int)
    probable_pitchers: list[dict] = []

    for game in games:
        # Filter to regular season only
        if game.get("game_type", "") != "R":
            continue

        away_name = game.get("away_name", "")
        home_name = game.get("home_name", "")
        away_abbrev = team_name_map.get(away_name, away_name)
        home_abbrev = team_name_map.get(home_name, home_name)

        games_per_team[away_abbrev] += 1
        games_per_team[home_abbrev] += 1

        away_pitcher = game.get("away_probable_pitcher", "") or "TBD"
        home_pitcher = game.get("home_probable_pitcher", "") or "TBD"

        probable_pitchers.append({
            "date": game.get("game_date", ""),
            "away_team": away_abbrev,
            "home_team": home_abbrev,
            "away_pitcher": away_pitcher,
            "home_pitcher": home_pitcher,
        })

    return {
        "games_per_team": dict(games_per_team),
        "probable_pitchers": probable_pitchers,
        "team_abbrev_map": team_name_map,
        "start_date": start_date,
        "end_date": end_date,
        "fetched_at": datetime.now().isoformat(timespec="seconds"),
    }
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_data/test_mlb_schedule.py -v`
Expected: All tests PASS.

- [ ] **Step 5: Commit**

```bash
git add src/fantasy_baseball/data/mlb_schedule.py tests/test_data/test_mlb_schedule.py
git commit -m "feat: add fetch_week_schedule with game counting and game_type filtering"
```

---

### Task 4: Probable pitcher parsing and TBD handling

**Files:**
- Modify: `tests/test_data/test_mlb_schedule.py`

- [ ] **Step 1: Write the failing tests**

Add to `tests/test_data/test_mlb_schedule.py`:

```python
class TestProbablePitchers:
    @patch("fantasy_baseball.data.mlb_schedule.statsapi")
    def test_extracts_probable_pitchers(self, mock_api):
        mock_api.schedule.return_value = [
            _make_game("New York Yankees", "Boston Red Sox", "2026-04-07",
                       away_pitcher="Gerrit Cole", home_pitcher="Brayan Bello"),
        ]
        mock_api.get.return_value = _mock_teams_response()

        result = fetch_week_schedule("2026-04-07", "2026-04-13")

        assert len(result["probable_pitchers"]) == 1
        pp = result["probable_pitchers"][0]
        assert pp["away_pitcher"] == "Gerrit Cole"
        assert pp["home_pitcher"] == "Brayan Bello"
        assert pp["away_team"] == "NYY"
        assert pp["home_team"] == "BOS"
        assert pp["date"] == "2026-04-07"

    @patch("fantasy_baseball.data.mlb_schedule.statsapi")
    def test_empty_pitcher_names_become_tbd(self, mock_api):
        mock_api.schedule.return_value = [
            _make_game("New York Yankees", "Boston Red Sox", "2026-04-07",
                       away_pitcher="", home_pitcher=""),
        ]
        mock_api.get.return_value = _mock_teams_response()

        result = fetch_week_schedule("2026-04-07", "2026-04-13")

        pp = result["probable_pitchers"][0]
        assert pp["away_pitcher"] == "TBD"
        assert pp["home_pitcher"] == "TBD"

    @patch("fantasy_baseball.data.mlb_schedule.statsapi")
    def test_metadata_fields(self, mock_api):
        mock_api.schedule.return_value = []
        mock_api.get.return_value = _mock_teams_response()

        result = fetch_week_schedule("2026-04-07", "2026-04-13")

        assert result["start_date"] == "2026-04-07"
        assert result["end_date"] == "2026-04-13"
        assert "fetched_at" in result
        assert "team_abbrev_map" in result
```

- [ ] **Step 2: Run tests to verify they pass**

These should already pass with the Task 3 implementation. Run: `pytest tests/test_data/test_mlb_schedule.py -v`
Expected: All tests PASS.

- [ ] **Step 3: Commit**

```bash
git add tests/test_data/test_mlb_schedule.py
git commit -m "test: add probable pitcher parsing and TBD handling tests"
```

---

### Task 5: Cache save/load/fallback

**Files:**
- Modify: `src/fantasy_baseball/data/mlb_schedule.py`
- Modify: `tests/test_data/test_mlb_schedule.py`

- [ ] **Step 1: Write the failing tests**

Add to `tests/test_data/test_mlb_schedule.py`:

```python
from fantasy_baseball.data.mlb_schedule import (
    save_schedule_cache,
    load_schedule_cache,
    get_week_schedule,
)


class TestScheduleCache:
    def test_save_and_load_roundtrip(self, tmp_path):
        data = {
            "games_per_team": {"NYY": 7, "BOS": 6},
            "probable_pitchers": [],
            "team_abbrev_map": {},
            "start_date": "2026-04-07",
            "end_date": "2026-04-13",
            "fetched_at": "2026-04-07T10:30:00",
        }
        cache_path = tmp_path / "schedule.json"
        save_schedule_cache(data, cache_path)
        loaded = load_schedule_cache(cache_path)
        assert loaded == data

    def test_load_missing_file_returns_none(self, tmp_path):
        result = load_schedule_cache(tmp_path / "nope.json")
        assert result is None


class TestGetWeekSchedule:
    @patch("fantasy_baseball.data.mlb_schedule.fetch_week_schedule")
    def test_live_fetch_success_caches_result(self, mock_fetch, tmp_path):
        cache_path = tmp_path / "schedule.json"
        mock_fetch.return_value = {
            "games_per_team": {"NYY": 7},
            "probable_pitchers": [],
            "team_abbrev_map": {},
            "start_date": "2026-04-07",
            "end_date": "2026-04-13",
            "fetched_at": "2026-04-07T10:30:00",
        }

        result = get_week_schedule("2026-04-07", "2026-04-13", cache_path)

        assert result["games_per_team"]["NYY"] == 7
        assert cache_path.exists()

    @patch("fantasy_baseball.data.mlb_schedule.fetch_week_schedule")
    def test_falls_back_to_cache_on_api_failure(self, mock_fetch, tmp_path):
        cache_path = tmp_path / "schedule.json"
        # Pre-populate cache
        cached = {
            "games_per_team": {"NYY": 6},
            "probable_pitchers": [],
            "team_abbrev_map": {},
            "start_date": "2026-04-07",
            "end_date": "2026-04-13",
            "fetched_at": "2026-04-07T08:00:00",
        }
        save_schedule_cache(cached, cache_path)

        # API fails
        mock_fetch.side_effect = Exception("API down")

        result = get_week_schedule("2026-04-07", "2026-04-13", cache_path)

        assert result["games_per_team"]["NYY"] == 6

    @patch("fantasy_baseball.data.mlb_schedule.fetch_week_schedule")
    def test_ignores_stale_cache_with_wrong_dates(self, mock_fetch, tmp_path):
        cache_path = tmp_path / "schedule.json"
        # Cache from last week
        cached = {
            "games_per_team": {"NYY": 6},
            "probable_pitchers": [],
            "team_abbrev_map": {},
            "start_date": "2026-03-31",
            "end_date": "2026-04-06",
            "fetched_at": "2026-03-31T08:00:00",
        }
        save_schedule_cache(cached, cache_path)

        mock_fetch.side_effect = Exception("API down")

        result = get_week_schedule("2026-04-07", "2026-04-13", cache_path)

        assert result is None

    @patch("fantasy_baseball.data.mlb_schedule.fetch_week_schedule")
    def test_returns_none_with_no_cache_and_api_failure(self, mock_fetch, tmp_path):
        cache_path = tmp_path / "schedule.json"
        mock_fetch.side_effect = Exception("API down")

        result = get_week_schedule("2026-04-07", "2026-04-13", cache_path)

        assert result is None
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_data/test_mlb_schedule.py::TestScheduleCache -v`
Run: `pytest tests/test_data/test_mlb_schedule.py::TestGetWeekSchedule -v`
Expected: FAIL with `ImportError`.

- [ ] **Step 3: Write the implementation**

Add to `src/fantasy_baseball/data/mlb_schedule.py`:

```python
def save_schedule_cache(data: dict, path: Path) -> None:
    """Save schedule data to a JSON cache file."""
    with open(path, "w") as f:
        json.dump(data, f, indent=2)


def load_schedule_cache(path: Path) -> dict | None:
    """Load schedule data from a JSON cache file. Returns None if missing."""
    if not path.exists():
        return None
    with open(path) as f:
        return json.load(f)


def get_week_schedule(
    start_date: str, end_date: str, cache_path: Path
) -> dict | None:
    """Get weekly schedule, fetching live and falling back to cache.

    Returns None if both live fetch and cache fail.
    """
    try:
        data = fetch_week_schedule(start_date, end_date)
        save_schedule_cache(data, cache_path)
        return data
    except Exception:
        logger.warning(
            "MLB API fetch failed; checking cache", exc_info=True
        )

    cached = load_schedule_cache(cache_path)
    if cached and cached.get("start_date") == start_date and cached.get("end_date") == end_date:
        logger.info("Using cached schedule data from %s", cached.get("fetched_at"))
        return cached

    logger.warning("No valid schedule data available (API down, no matching cache)")
    return None
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_data/test_mlb_schedule.py -v`
Expected: All tests PASS.

- [ ] **Step 5: Commit**

```bash
git add src/fantasy_baseball/data/mlb_schedule.py tests/test_data/test_mlb_schedule.py
git commit -m "feat: add schedule cache save/load with API fallback"
```

---

### Task 6: fetch_scoring_period in yahoo_roster.py

**Files:**
- Modify: `src/fantasy_baseball/lineup/yahoo_roster.py:1-5`
- Create: `tests/test_lineup/test_scoring_period.py`

- [ ] **Step 1: Write the failing test**

Create `tests/test_lineup/test_scoring_period.py`:

```python
import datetime
from unittest.mock import MagicMock
from fantasy_baseball.lineup.yahoo_roster import fetch_scoring_period


class TestFetchScoringPeriod:
    def test_returns_date_strings_from_yahoo(self):
        mock_league = MagicMock()
        mock_league.current_week.return_value = 12
        mock_league.week_date_range.return_value = (
            datetime.date(2026, 6, 15),
            datetime.date(2026, 6, 21),
        )

        start, end = fetch_scoring_period(mock_league)

        assert start == "2026-06-15"
        assert end == "2026-06-21"
        mock_league.current_week.assert_called_once()
        mock_league.week_date_range.assert_called_once_with(12)

    def test_falls_back_to_current_week_on_error(self):
        mock_league = MagicMock()
        mock_league.current_week.side_effect = Exception("Yahoo down")

        start, end = fetch_scoring_period(mock_league)

        # Should return Mon-Sun of current week as fallback
        today = datetime.date.today()
        monday = today - datetime.timedelta(days=today.weekday())
        sunday = monday + datetime.timedelta(days=6)
        assert start == monday.isoformat()
        assert end == sunday.isoformat()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_lineup/test_scoring_period.py -v`
Expected: FAIL with `ImportError`.

- [ ] **Step 3: Write the implementation**

Add to `src/fantasy_baseball/lineup/yahoo_roster.py` (at the end of the file, add the import at the top):

Add `import datetime` to the imports at the top of the file.

Then add the function:

```python
def fetch_scoring_period(league) -> tuple[str, str]:
    """Get the current Yahoo scoring period date range.

    Returns (start_date, end_date) as "YYYY-MM-DD" strings.
    Falls back to Monday-Sunday of the current week on error.
    """
    try:
        week = league.current_week()
        start, end = league.week_date_range(week)
        return start.isoformat(), end.isoformat()
    except Exception:
        logger.warning(
            "Failed to get Yahoo scoring period; using Mon-Sun fallback",
            exc_info=True,
        )
        today = datetime.date.today()
        monday = today - datetime.timedelta(days=today.weekday())
        sunday = monday + datetime.timedelta(days=6)
        return monday.isoformat(), sunday.isoformat()
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_lineup/test_scoring_period.py -v`
Expected: All tests PASS.

- [ ] **Step 5: Run existing yahoo_roster tests to ensure no regressions**

Run: `pytest tests/test_lineup/test_yahoo_roster.py -v`
Expected: All existing tests PASS.

- [ ] **Step 6: Commit**

```bash
git add src/fantasy_baseball/lineup/yahoo_roster.py tests/test_lineup/test_scoring_period.py
git commit -m "feat: add fetch_scoring_period to get Yahoo scoring week boundaries"
```

---

### Task 7: Wire schedule data into run_lineup.py

**Files:**
- Modify: `scripts/run_lineup.py`

This task modifies `run_lineup.py` to:
1. Fetch the scoring period from Yahoo
2. Fetch the MLB schedule (with cache fallback)
3. Look up `games_this_week` from the projection row's `team` column
4. Apply schedule scaling to waiver wire free agents too

- [ ] **Step 1: Add imports and constants**

At the top of `scripts/run_lineup.py`, add these imports after the existing ones:

```python
from fantasy_baseball.lineup.yahoo_roster import fetch_scoring_period
from fantasy_baseball.data.mlb_schedule import get_week_schedule
```

Add a new constant after `PROJECTIONS_DIR`:

```python
SCHEDULE_PATH = PROJECT_ROOT / "data" / "weekly_schedule.json"
```

- [ ] **Step 2: Fetch scoring period and schedule after connecting to Yahoo**

In the `main()` function, after the standings fetch block (after line 89 `print()`), add:

```python
    # Fetch scoring period and MLB schedule
    print("Fetching weekly schedule...")
    period_start, period_end = fetch_scoring_period(league)
    schedule = get_week_schedule(period_start, period_end, SCHEDULE_PATH)
    games_per_team = schedule["games_per_team"] if schedule else {}
    if schedule:
        print(f"Scoring period: {period_start} to {period_end}")
        print(f"Schedule loaded for {len(games_per_team)} teams")
    else:
        print("Schedule unavailable — using default 6 games/week")
    print()
```

- [ ] **Step 3: Replace hardcoded games_this_week with schedule lookup**

In the roster matching loop, change the `games_this_week` lookup. Replace:

```python
        games_this_week = player.get("games_this_week", DEFAULT_GAMES_PER_WEEK)
```

With:

```python
        games_this_week = DEFAULT_GAMES_PER_WEEK  # updated after projection match
```

Then after each projection match, look up the team. In the hitter branch, after `hit_proj["player_type"] = "hitter"`, change the `scale_by_schedule` call:

```python
                hit_proj["player_type"] = "hitter"
                team = hit_proj.get("team", "")
                games_this_week = games_per_team.get(team, DEFAULT_GAMES_PER_WEEK)
                hit_proj = scale_by_schedule(hit_proj, games_this_week)
```

Same pattern in the pitcher branch, after `pit_proj["player_type"] = "pitcher"`:

```python
                pit_proj["player_type"] = "pitcher"
                team = pit_proj.get("team", "")
                games_this_week = games_per_team.get(team, DEFAULT_GAMES_PER_WEEK)
                pit_proj = scale_by_schedule(pit_proj, games_this_week)
```

And in the fallback branch, after `proj_row["player_type"] = ptype`:

```python
                    proj_row["player_type"] = ptype
                    team = proj_row.get("team", "")
                    games_this_week = games_per_team.get(team, DEFAULT_GAMES_PER_WEEK)
                    proj_row = scale_by_schedule(proj_row, games_this_week)
```

- [ ] **Step 4: Apply schedule scaling to waiver wire free agents**

In the waiver wire section (around line 242-244), after matching a free agent to projections:

```python
                if proj_row is not None:
                    proj_row["positions"] = fa["positions"]
                    team = proj_row.get("team", "")
                    games_this_week = games_per_team.get(team, DEFAULT_GAMES_PER_WEEK)
                    proj_row = scale_by_schedule(proj_row, games_this_week)
                    fa_players.append(proj_row)
```

- [ ] **Step 5: Run full test suite**

Run: `pytest tests/ -v`
Expected: All tests PASS. (The `run_lineup.py` changes are in the CLI script, so unit tests shouldn't break.)

- [ ] **Step 6: Commit**

```bash
git add scripts/run_lineup.py
git commit -m "feat: wire MLB schedule into lineup optimizer for real games-per-week data"
```

---

### Task 8: Probable starters display in run_lineup.py

**Files:**
- Modify: `scripts/run_lineup.py`

- [ ] **Step 1: Add the probable starters display function**

Add this function to `run_lineup.py` (before `main()`):

```python
from datetime import datetime as dt


def print_probable_starters(
    roster_pitchers: list[pd.Series],
    schedule: dict | None,
) -> None:
    """Print probable starter matchups, flagging two-start pitchers."""
    if not schedule or not roster_pitchers:
        return

    probable = schedule.get("probable_pitchers", [])
    if not probable:
        print("  No probable pitcher data available.")
        return

    # Build pitcher name -> list of starts
    pitcher_starts: dict[str, list[dict]] = {}
    roster_names = {normalize_name(p["name"]) for p in roster_pitchers}

    for game in probable:
        for side, team_key in [("away", "away_team"), ("home", "home_team")]:
            pitcher_name = game.get(f"{side}_pitcher", "TBD")
            if pitcher_name == "TBD":
                continue
            if normalize_name(pitcher_name) not in roster_names:
                continue

            opponent_key = "home_team" if side == "away" else "away_team"
            indicator = "@" if side == "away" else "vs"
            try:
                day = dt.strptime(game["date"], "%Y-%m-%d").strftime("%a")
            except (ValueError, KeyError):
                day = "?"

            if pitcher_name not in pitcher_starts:
                pitcher_starts[pitcher_name] = []
            pitcher_starts[pitcher_name].append({
                "day": day,
                "indicator": indicator,
                "opponent": game[opponent_key],
            })

    if not pitcher_starts:
        print("  No roster pitchers found in probable starters.")
        return

    two_start = {k: v for k, v in pitcher_starts.items() if len(v) >= 2}
    one_start = {k: v for k, v in pitcher_starts.items() if len(v) == 1}

    if two_start:
        print("  ** TWO-START PITCHERS **")
        for name, starts in sorted(two_start.items()):
            matchups = ", ".join(
                f"{s['day']} {s['indicator']} {s['opponent']}" for s in starts
            )
            print(f"    {name:<25} {matchups}")

    if one_start:
        print("  SINGLE START")
        for name, starts in sorted(one_start.items()):
            s = starts[0]
            print(f"    {name:<25} {s['day']} {s['indicator']} {s['opponent']}")

    # Roster pitchers with no announced start
    announced = {normalize_name(k) for k in pitcher_starts.keys()}
    unannounced = [
        p["name"] for p in roster_pitchers
        if normalize_name(p["name"]) not in announced
        and p.get("player_type") == "pitcher"
    ]
    if unannounced:
        print("  NO START ANNOUNCED")
        for name in sorted(unannounced):
            print(f"    {name:<25} TBD")
```

- [ ] **Step 2: Call it from main()**

After the pitcher lineup section (after the pitcher bench print, around line 218), add:

```python
    # Probable starters display
    if roster_pitchers:
        period_label = f"{period_start} to {period_end}" if schedule else ""
        print("=" * 60)
        print(f"PROBABLE STARTERS THIS WEEK ({period_label})")
        print("=" * 60)
        print_probable_starters(roster_pitchers, schedule)
        print()
```

- [ ] **Step 3: Run full test suite**

Run: `pytest tests/ -v`
Expected: All tests PASS.

- [ ] **Step 4: Commit**

```bash
git add scripts/run_lineup.py
git commit -m "feat: display probable starter matchups and flag two-start pitchers"
```

---

### Task 9: Final integration verification

- [ ] **Step 1: Run the full test suite**

Run: `pytest tests/ -v --tb=short`
Expected: All tests PASS.

- [ ] **Step 2: Verify the module imports cleanly**

Run: `python -c "from fantasy_baseball.data.mlb_schedule import get_week_schedule, fetch_week_schedule; print('OK')"`
Expected: Prints `OK`.

Run: `python -c "from fantasy_baseball.lineup.yahoo_roster import fetch_scoring_period; print('OK')"`
Expected: Prints `OK`.

- [ ] **Step 3: Update TODO.md**

Mark the first two items in the In-Season Enhancements section as done:

```
- [x] **Weekly schedule data via MLB Stats API** — ...
- [x] **Probable pitcher matchups** — ...
```

- [ ] **Step 4: Commit**

```bash
git add TODO.md
git commit -m "docs: mark schedule and probable pitcher TODOs as complete"
```
