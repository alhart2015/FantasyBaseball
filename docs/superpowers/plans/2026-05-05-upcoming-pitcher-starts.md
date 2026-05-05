# Upcoming Pitcher Starts Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the largely-empty "Probable Starters" table with full-week coverage built from MLB-announced starts plus projected starts from a 5-team-game rotation.

**Architecture:** Extend `data/mlb_schedule.py` to fetch a 14-day lookback alongside the scoring week. Add a new pure module `lineup/upcoming_starts.py` that, for each in-scope SP, finds the rotation anchor (most recent past start) and projects the next starts at +5 team-game-index steps, merging with MLB announcements. The existing `lineup/matchups.py::get_probable_starters` becomes a thin wrapper that calls the new module and decorates each start with the existing OPS-rank/quality detail. Cache shape gains an `announced: bool` per start; the template renders chips colored by Great/Fair/Tough with dotted borders for `announced=False`.

**Tech Stack:** Python 3.11+, `MLB-StatsAPI` (statsapi), pandas, Flask + Jinja2 templates. Tests via pytest, mocks via `unittest.mock.patch`.

**Spec:** `docs/superpowers/specs/2026-05-05-upcoming-pitcher-starts-design.md`

---

## File Map

**Create:**
- `src/fantasy_baseball/lineup/upcoming_starts.py` — pure functions for anchor/projection logic
- `tests/test_lineup/test_upcoming_starts.py` — unit tests for the new module

**Modify:**
- `src/fantasy_baseball/data/mlb_schedule.py` — add `lookback_days` parameter, add `game_number` field
- `src/fantasy_baseball/lineup/matchups.py` — rewrite `get_probable_starters` to delegate
- `src/fantasy_baseball/web/refresh_pipeline.py:843–875` — pre-filter SPs, pass `lookback_days=14`
- `src/fantasy_baseball/web/templates/season/lineup.html:155–212` — chip rendering
- `tests/test_data/test_mlb_schedule.py` — cover the lookback parameter
- `tests/test_lineup/test_matchups.py` — adjust the existing `get_probable_starters` test
- `tests/test_web/test_refresh_pipeline.py` — fixture seeds lookback + scoring-week games

**No DB / config / migration changes.** Cache shape is a superset; one refresh repopulates.

---

### Task 1: Extend `fetch_week_schedule` with `lookback_days` + capture `game_number`

**Files:**
- Modify: `src/fantasy_baseball/data/mlb_schedule.py:49–94`
- Test: `tests/test_data/test_mlb_schedule.py`

**Why:** The new logic needs the team's recent past games to find each pitcher's anchor. We also need `game_number` so doubleheader days produce two distinct game entries (rotation steps over each one independently).

- [ ] **Step 1: Write the failing tests**

Add to `tests/test_data/test_mlb_schedule.py` (above the `TestNormalizeTeamAbbrev` class, after the helper section):

```python
class TestFetchWeekScheduleLookback:
    @patch("fantasy_baseball.data.mlb_schedule.statsapi")
    def test_lookback_extends_start_date(self, mock_api):
        # Teams call (for _build_team_name_map)
        mock_api.get.return_value = _mock_teams_response()
        # Schedule call returns games across both lookback and window
        mock_api.schedule.return_value = [
            _make_game("New York Yankees", "Boston Red Sox", "2026-04-25",
                       away_pitcher="Gerrit Cole"),
            _make_game("New York Yankees", "Boston Red Sox", "2026-05-05",
                       away_pitcher="Carlos Rodon"),
        ]

        result = fetch_week_schedule("2026-05-05", "2026-05-11", lookback_days=14)

        # statsapi.schedule called with start = 2026-05-05 - 14d = 2026-04-21
        mock_api.schedule.assert_called_once_with("2026-04-21", "2026-05-11")
        # Both past and current games appear in probable_pitchers
        assert len(result["probable_pitchers"]) == 2
        # The returned start_date/end_date still reflect the scoring week
        assert result["start_date"] == "2026-05-05"
        assert result["end_date"] == "2026-05-11"

    @patch("fantasy_baseball.data.mlb_schedule.statsapi")
    def test_default_lookback_is_zero(self, mock_api):
        mock_api.get.return_value = _mock_teams_response()
        mock_api.schedule.return_value = []
        fetch_week_schedule("2026-05-05", "2026-05-11")
        mock_api.schedule.assert_called_once_with("2026-05-05", "2026-05-11")

    @patch("fantasy_baseball.data.mlb_schedule.statsapi")
    def test_game_number_captured(self, mock_api):
        mock_api.get.return_value = _mock_teams_response()
        mock_api.schedule.return_value = [
            {**_make_game("New York Yankees", "Boston Red Sox", "2026-05-05",
                          away_pitcher="A", home_pitcher="B"),
             "game_num": 1},
            {**_make_game("New York Yankees", "Boston Red Sox", "2026-05-05",
                          away_pitcher="C", home_pitcher="D"),
             "game_num": 2},
        ]
        result = fetch_week_schedule("2026-05-05", "2026-05-11")
        pps = result["probable_pitchers"]
        assert len(pps) == 2
        assert pps[0]["game_number"] == 1
        assert pps[1]["game_number"] == 2

    @patch("fantasy_baseball.data.mlb_schedule.statsapi")
    def test_game_number_defaults_to_one(self, mock_api):
        # statsapi may omit game_num for non-doubleheader games
        mock_api.get.return_value = _mock_teams_response()
        mock_api.schedule.return_value = [
            _make_game("New York Yankees", "Boston Red Sox", "2026-05-05"),
        ]
        result = fetch_week_schedule("2026-05-05", "2026-05-11")
        assert result["probable_pitchers"][0]["game_number"] == 1
```

- [ ] **Step 2: Run tests to confirm they fail**

```bash
pytest tests/test_data/test_mlb_schedule.py::TestFetchWeekScheduleLookback -v
```

Expected: FAIL — `lookback_days` is not a parameter, `game_number` key not in output.

- [ ] **Step 3: Implement**

Replace `fetch_week_schedule` in `src/fantasy_baseball/data/mlb_schedule.py`:

```python
from datetime import date as _date, timedelta as _timedelta


def fetch_week_schedule(start_date: str, end_date: str, lookback_days: int = 0) -> dict:
    """Fetch the MLB schedule for a date range and return structured data.

    Filters to regular-season games only (game_type == "R").
    Returns game counts per team (FanGraphs abbreviations), probable
    pitchers, team abbreviation map, and metadata.

    When ``lookback_days > 0``, the actual statsapi call covers
    ``(start_date - lookback_days) .. end_date``. The returned
    ``start_date`` and ``end_date`` fields still reflect the original
    scoring window, not the lookback span. ``games_per_team`` only
    counts games inside ``start_date..end_date`` so existing consumers
    that read it as the per-team count for the scoring week stay correct.

    Each ``probable_pitchers`` entry includes ``game_number`` (defaults
    to 1; >1 marks the second game of a doubleheader). Sorting by
    (date, game_number) gives a stable chronological order.
    """
    fetch_start = start_date
    if lookback_days > 0:
        fetch_start = (
            _date.fromisoformat(start_date) - _timedelta(days=lookback_days)
        ).isoformat()

    games = statsapi.schedule(fetch_start, end_date)
    team_name_map = _build_team_name_map()

    games_per_team: dict[str, int] = defaultdict(int)
    probable_pitchers: list[dict] = []

    for game in games:
        if game.get("game_type") != "R":
            continue

        away_name = game["away_name"]
        home_name = game["home_name"]
        game_date = game["game_date"]

        away_abbrev = team_name_map.get(away_name, away_name)
        home_abbrev = team_name_map.get(home_name, home_name)

        # Per-team game counts: scoring-week only, so other consumers
        # of games_per_team don't see lookback games.
        if game_date >= start_date:
            games_per_team[away_abbrev] += 1
            games_per_team[home_abbrev] += 1

        away_pitcher = game.get("away_probable_pitcher", "") or "TBD"
        home_pitcher = game.get("home_probable_pitcher", "") or "TBD"

        probable_pitchers.append({
            "date": game_date,
            "game_number": int(game.get("game_num", 1) or 1),
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
        "fetched_at": local_now().isoformat(timespec="seconds"),
    }
```

- [ ] **Step 4: Run tests to confirm they pass**

```bash
pytest tests/test_data/test_mlb_schedule.py -v
```

Expected: ALL PASS, including pre-existing tests.

- [ ] **Step 5: Commit**

```bash
git add src/fantasy_baseball/data/mlb_schedule.py tests/test_data/test_mlb_schedule.py
git commit -m "feat(schedule): add lookback_days + game_number to week schedule"
```

---

### Task 2: Plumb `lookback_days` through `get_week_schedule`

**Files:**
- Modify: `src/fantasy_baseball/data/mlb_schedule.py:114–143`
- Test: `tests/test_data/test_mlb_schedule.py`

**Why:** `get_week_schedule` is the cached entry point used by the refresh pipeline. Cache match logic must include `lookback_days` so a previously-cached zero-lookback payload doesn't silently get reused when the caller now wants 14 days of history.

- [ ] **Step 1: Write the failing tests**

Add a new test class to `tests/test_data/test_mlb_schedule.py`:

```python
class TestGetWeekScheduleLookback:
    @patch("fantasy_baseball.data.mlb_schedule.fetch_week_schedule")
    def test_passes_lookback_through(self, mock_fetch, tmp_path):
        mock_fetch.return_value = {
            "games_per_team": {},
            "probable_pitchers": [],
            "team_abbrev_map": {},
            "start_date": "2026-05-05",
            "end_date": "2026-05-11",
            "lookback_days": 14,
            "fetched_at": "2026-05-05T08:00:00",
        }
        cache_path = tmp_path / "schedule.json"

        result = get_week_schedule("2026-05-05", "2026-05-11", cache_path, lookback_days=14)

        mock_fetch.assert_called_once_with("2026-05-05", "2026-05-11", lookback_days=14)
        assert result["lookback_days"] == 14

    @patch("fantasy_baseball.data.mlb_schedule.fetch_week_schedule",
           side_effect=RuntimeError("api down"))
    def test_cache_match_includes_lookback(self, _mock_fetch, tmp_path):
        cache_path = tmp_path / "schedule.json"
        # Cache from a prior 0-lookback fetch
        save_schedule_cache(
            {
                "games_per_team": {},
                "probable_pitchers": [],
                "team_abbrev_map": {},
                "start_date": "2026-05-05",
                "end_date": "2026-05-11",
                "lookback_days": 0,
                "fetched_at": "2026-05-04T08:00:00",
            },
            cache_path,
        )

        # Caller now wants 14-day lookback — cache must NOT match.
        result = get_week_schedule("2026-05-05", "2026-05-11", cache_path, lookback_days=14)
        assert result is None
```

- [ ] **Step 2: Run tests to confirm they fail**

```bash
pytest tests/test_data/test_mlb_schedule.py::TestGetWeekScheduleLookback -v
```

Expected: FAIL — `lookback_days` is not a parameter of `get_week_schedule`.

- [ ] **Step 3: Implement**

Replace `get_week_schedule` and update `fetch_week_schedule`'s return dict to include `lookback_days` for cache-match purposes.

In `fetch_week_schedule`, add `"lookback_days": lookback_days,` to the returned dict (placed next to `start_date`/`end_date`).

Replace `get_week_schedule`:

```python
def get_week_schedule(
    start_date: str,
    end_date: str,
    cache_path: Path,
    lookback_days: int = 0,
) -> dict | None:
    """Main entry point for fetching the week schedule.

    Tries a live fetch first; on success, caches the result. On API
    failure, falls back to the cache if the cached date range AND
    ``lookback_days`` match the requested ones. Returns None if both
    live and cached data are unavailable or stale.
    """
    try:
        data = fetch_week_schedule(start_date, end_date, lookback_days=lookback_days)
        save_schedule_cache(data, cache_path)
        return data
    except Exception:
        logger.exception("Failed to fetch live week schedule; trying cache")

    cached = load_schedule_cache(cache_path)
    if cached is None:
        return None

    if (
        cached.get("start_date") != start_date
        or cached.get("end_date") != end_date
        or cached.get("lookback_days", 0) != lookback_days
    ):
        logger.warning(
            "Cached schedule (%s–%s, lookback=%s) does not match requested (%s–%s, lookback=%s); ignoring cache",
            cached.get("start_date"),
            cached.get("end_date"),
            cached.get("lookback_days", 0),
            start_date,
            end_date,
            lookback_days,
        )
        return None

    return cached
```

- [ ] **Step 4: Run tests to confirm they pass**

```bash
pytest tests/test_data/test_mlb_schedule.py -v
```

Expected: ALL PASS.

- [ ] **Step 5: Commit**

```bash
git add src/fantasy_baseball/data/mlb_schedule.py tests/test_data/test_mlb_schedule.py
git commit -m "feat(schedule): plumb lookback_days through get_week_schedule"
```

---

### Task 3: Create `upcoming_starts.py` skeleton with type definitions

**Files:**
- Create: `src/fantasy_baseball/lineup/upcoming_starts.py`
- Create: `tests/test_lineup/test_upcoming_starts.py`

**Why:** Lock in shared type definitions (`GameSlot`, `StartEntry`) and module shape before implementing logic, so subsequent tasks can be implemented and tested in isolation without redefining types.

- [ ] **Step 1: Write the failing tests**

Create `tests/test_lineup/test_upcoming_starts.py`:

```python
"""Unit tests for the rotation anchor + projection logic."""

from fantasy_baseball.lineup.upcoming_starts import (
    GameSlot,
    StartEntry,
)


def test_game_slot_fields():
    slot = GameSlot(
        date="2026-05-05",
        game_number=1,
        opponent="LAD",
        indicator="@",
        announced_starter="Bryan Woo",
    )
    assert slot.date == "2026-05-05"
    assert slot.game_number == 1
    assert slot.opponent == "LAD"
    assert slot.indicator == "@"
    assert slot.announced_starter == "Bryan Woo"


def test_start_entry_announced_default_false():
    entry = StartEntry(
        date="2026-05-05",
        day="Mon",
        opponent="LAD",
        indicator="@",
    )
    assert entry.announced is False


def test_start_entry_with_detail():
    entry = StartEntry(
        date="2026-05-05",
        day="Mon",
        opponent="LAD",
        indicator="@",
        announced=True,
        matchup_quality="Tough",
        detail={"ops": 0.789, "ops_rank": 4, "k_pct": 22.1, "k_rank": 18},
    )
    assert entry.announced is True
    assert entry.matchup_quality == "Tough"
    assert entry.detail["ops_rank"] == 4
```

- [ ] **Step 2: Run tests to confirm they fail**

```bash
pytest tests/test_lineup/test_upcoming_starts.py -v
```

Expected: FAIL — module does not exist.

- [ ] **Step 3: Implement**

Create `src/fantasy_baseball/lineup/upcoming_starts.py`:

```python
"""Build upcoming projected starts for roster pitchers.

For each in-scope starting pitcher we find the rotation anchor (their
most recent past start within a 14-day lookback) and project the next
starts at every 5th team game until past the scoring window. MLB-
announced probable starters override projections for the same game.

Public API:
    build_team_game_index(probable_pitchers, team_abbrev) -> list[GameSlot]
    find_anchor_index(team_games, pitcher_name, today) -> int | None
    project_start_indices(anchor_index, total_games, step=5) -> list[int]
    compose_pitcher_entries(...)  -> list[StartEntry]

All functions are pure — no I/O, no global state. The matchup/quality
decoration happens in lineup.matchups via existing helpers.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date
from typing import Any


@dataclass(frozen=True)
class GameSlot:
    """One scheduled team-game from the perspective of a single team.

    ``announced_starter`` is the name MLB has listed (or empty string if
    "TBD" / unset). For completed past games, MLB populates this with
    the actual starter, which is the signal used to find rotation anchors.
    """
    date: str  # YYYY-MM-DD
    game_number: int  # 1 for normal games, >1 for second game of doubleheader
    opponent: str  # FanGraphs-normalized opponent abbreviation
    indicator: str  # "@" if away, "vs" if home
    announced_starter: str = ""


@dataclass
class StartEntry:
    """One projected or announced start for a roster pitcher."""
    date: str
    day: str  # "Mon", "Tue", ...
    opponent: str
    indicator: str
    announced: bool = False
    matchup_quality: str = "Fair"  # "Great" | "Fair" | "Tough"
    detail: dict[str, Any] = field(default_factory=dict)


def build_team_game_index(
    probable_pitchers: list[dict[str, Any]],
    team_abbrev: str,
) -> list[GameSlot]:
    """Filter the league-wide probable_pitchers list to one team's games.

    Returns a chronological list (by date, then game_number). Each
    entry exposes the opponent and the announced starter for that team.
    """
    raise NotImplementedError("Implemented in Task 4")


def find_anchor_index(
    team_games: list[GameSlot],
    pitcher_name: str,
    today: date,
) -> int | None:
    """Most recent index in ``team_games`` where ``pitcher_name`` started.

    Only considers games strictly before ``today``. Name comparison is
    accent/case-insensitive (delegates to normalize_name). Returns
    ``None`` if the pitcher has no eligible past start in the index.
    """
    raise NotImplementedError("Implemented in Task 5")


def project_start_indices(
    anchor_index: int,
    total_games: int,
    step: int = 5,
) -> list[int]:
    """Return the projected start indices in the team's game stream.

    Starts at ``anchor_index + step`` and steps by ``step`` until
    exceeding ``total_games - 1``. Returns an empty list if anchor_index
    is negative.
    """
    raise NotImplementedError("Implemented in Task 6")


def compose_pitcher_entries(
    pitcher_name: str,
    team_games: list[GameSlot],
    today: date,
    window_start: date,
    window_end: date,
    matchup_factors: dict[str, dict[str, float]],
    team_stats: dict[str, dict[str, float]],
    ops_rank_map: dict[str, int],
    k_rank_map: dict[str, int],
) -> list[StartEntry]:
    """Build the full list of StartEntry rows for one pitcher.

    Combines:
      - announced starts in ``[window_start, window_end]`` where this
        pitcher is the starter,
      - projected starts (anchor + 5*N) that land inside the window,
        excluding any team-game whose announced starter is someone else.

    Each entry is decorated with the existing matchup_quality + detail
    payload by looking up the opponent in ``matchup_factors`` and
    ``team_stats``. Rows are sorted by date then game_number.
    """
    raise NotImplementedError("Implemented in Task 7")
```

- [ ] **Step 4: Run tests to confirm they pass**

```bash
pytest tests/test_lineup/test_upcoming_starts.py -v
```

Expected: PASS — only the dataclass tests run; NotImplementedError functions aren't invoked yet.

- [ ] **Step 5: Commit**

```bash
git add src/fantasy_baseball/lineup/upcoming_starts.py tests/test_lineup/test_upcoming_starts.py
git commit -m "feat(upcoming-starts): scaffold module with GameSlot and StartEntry types"
```

---

### Task 4: Implement `build_team_game_index`

**Files:**
- Modify: `src/fantasy_baseball/lineup/upcoming_starts.py`
- Test: `tests/test_lineup/test_upcoming_starts.py`

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_lineup/test_upcoming_starts.py`:

```python
from fantasy_baseball.lineup.upcoming_starts import build_team_game_index


def _pp(date_, away, home, awp="", hwp="", num=1):
    return {
        "date": date_,
        "game_number": num,
        "away_team": away,
        "home_team": home,
        "away_pitcher": awp or "TBD",
        "home_pitcher": hwp or "TBD",
    }


class TestBuildTeamGameIndex:
    def test_filters_to_target_team(self):
        pps = [
            _pp("2026-05-05", "SEA", "LAD", awp="Woo"),
            _pp("2026-05-05", "NYY", "BOS", awp="Cole"),
            _pp("2026-05-06", "TEX", "SEA", hwp="Castillo"),
        ]
        slots = build_team_game_index(pps, "SEA")
        assert len(slots) == 2
        assert slots[0].opponent == "LAD"
        assert slots[0].indicator == "@"
        assert slots[0].announced_starter == "Woo"
        assert slots[1].opponent == "TEX"
        assert slots[1].indicator == "vs"
        assert slots[1].announced_starter == "Castillo"

    def test_chronological_ordering(self):
        pps = [
            _pp("2026-05-07", "SEA", "TEX"),
            _pp("2026-05-05", "SEA", "LAD"),
            _pp("2026-05-06", "SEA", "TEX"),
        ]
        slots = build_team_game_index(pps, "SEA")
        assert [s.date for s in slots] == ["2026-05-05", "2026-05-06", "2026-05-07"]

    def test_doubleheader_sorts_by_game_number(self):
        pps = [
            _pp("2026-05-05", "SEA", "LAD", num=2, awp="Gilbert"),
            _pp("2026-05-05", "SEA", "LAD", num=1, awp="Woo"),
        ]
        slots = build_team_game_index(pps, "SEA")
        assert [s.game_number for s in slots] == [1, 2]
        assert slots[0].announced_starter == "Woo"
        assert slots[1].announced_starter == "Gilbert"

    def test_tbd_announced_starter_becomes_empty(self):
        pps = [_pp("2026-05-05", "SEA", "LAD", awp="TBD")]
        slots = build_team_game_index(pps, "SEA")
        assert slots[0].announced_starter == ""

    def test_empty_when_team_not_in_schedule(self):
        pps = [_pp("2026-05-05", "NYY", "BOS")]
        assert build_team_game_index(pps, "SEA") == []
```

- [ ] **Step 2: Run tests to confirm they fail**

```bash
pytest tests/test_lineup/test_upcoming_starts.py::TestBuildTeamGameIndex -v
```

Expected: FAIL — `NotImplementedError`.

- [ ] **Step 3: Implement**

Replace the stub in `src/fantasy_baseball/lineup/upcoming_starts.py`:

```python
def build_team_game_index(
    probable_pitchers: list[dict[str, Any]],
    team_abbrev: str,
) -> list[GameSlot]:
    slots: list[GameSlot] = []
    for game in probable_pitchers:
        if game["away_team"] == team_abbrev:
            opponent = game["home_team"]
            indicator = "@"
            starter = game.get("away_pitcher", "") or ""
        elif game["home_team"] == team_abbrev:
            opponent = game["away_team"]
            indicator = "vs"
            starter = game.get("home_pitcher", "") or ""
        else:
            continue

        if starter == "TBD":
            starter = ""

        slots.append(
            GameSlot(
                date=game["date"],
                game_number=int(game.get("game_number", 1) or 1),
                opponent=opponent,
                indicator=indicator,
                announced_starter=starter,
            )
        )

    slots.sort(key=lambda s: (s.date, s.game_number))
    return slots
```

- [ ] **Step 4: Run tests to confirm they pass**

```bash
pytest tests/test_lineup/test_upcoming_starts.py -v
```

Expected: ALL PASS.

- [ ] **Step 5: Commit**

```bash
git add src/fantasy_baseball/lineup/upcoming_starts.py tests/test_lineup/test_upcoming_starts.py
git commit -m "feat(upcoming-starts): implement build_team_game_index"
```

---

### Task 5: Implement `find_anchor_index`

**Files:**
- Modify: `src/fantasy_baseball/lineup/upcoming_starts.py`
- Test: `tests/test_lineup/test_upcoming_starts.py`

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_lineup/test_upcoming_starts.py`:

```python
from datetime import date as _date

from fantasy_baseball.lineup.upcoming_starts import (
    GameSlot,
    find_anchor_index,
)


def _slot(d, opp, ann="", ind="@", num=1):
    return GameSlot(date=d, game_number=num, opponent=opp, indicator=ind, announced_starter=ann)


class TestFindAnchorIndex:
    def test_finds_most_recent_past_start(self):
        games = [
            _slot("2026-05-01", "TEX", ann="Bryan Woo"),
            _slot("2026-05-03", "LAD", ann="Castillo"),
            _slot("2026-05-06", "TEX", ann="Bryan Woo"),
        ]
        idx = find_anchor_index(games, "Bryan Woo", today=_date(2026, 5, 7))
        assert idx == 2  # the May 6 start

    def test_excludes_today_and_future(self):
        games = [
            _slot("2026-05-01", "TEX", ann="Bryan Woo"),
            _slot("2026-05-07", "LAD", ann="Bryan Woo"),  # today — excluded
            _slot("2026-05-08", "LAD", ann="Bryan Woo"),  # future — excluded
        ]
        idx = find_anchor_index(games, "Bryan Woo", today=_date(2026, 5, 7))
        assert idx == 0

    def test_returns_none_when_no_match(self):
        games = [_slot("2026-05-01", "TEX", ann="Castillo")]
        idx = find_anchor_index(games, "Bryan Woo", today=_date(2026, 5, 7))
        assert idx is None

    def test_returns_none_when_pitcher_has_only_future_starts(self):
        games = [_slot("2026-05-08", "TEX", ann="Bryan Woo")]
        idx = find_anchor_index(games, "Bryan Woo", today=_date(2026, 5, 7))
        assert idx is None

    def test_name_match_is_accent_insensitive(self):
        # normalize_name strips accents, so "José Berríos" and "Jose Berrios" match.
        games = [_slot("2026-05-01", "TEX", ann="José Berríos")]
        idx = find_anchor_index(games, "Jose Berrios", today=_date(2026, 5, 7))
        assert idx == 0
```

- [ ] **Step 2: Run tests to confirm they fail**

```bash
pytest tests/test_lineup/test_upcoming_starts.py::TestFindAnchorIndex -v
```

Expected: FAIL — `NotImplementedError`.

- [ ] **Step 3: Implement**

Add an import and replace the stub:

```python
from fantasy_baseball.utils.name_utils import normalize_name


def find_anchor_index(
    team_games: list[GameSlot],
    pitcher_name: str,
    today: date,
) -> int | None:
    target = normalize_name(pitcher_name)
    today_iso = today.isoformat()
    anchor: int | None = None
    for i, slot in enumerate(team_games):
        if slot.date >= today_iso:
            continue
        if not slot.announced_starter:
            continue
        if normalize_name(slot.announced_starter) == target:
            anchor = i
    return anchor
```

- [ ] **Step 4: Run tests to confirm they pass**

```bash
pytest tests/test_lineup/test_upcoming_starts.py -v
```

Expected: ALL PASS.

- [ ] **Step 5: Commit**

```bash
git add src/fantasy_baseball/lineup/upcoming_starts.py tests/test_lineup/test_upcoming_starts.py
git commit -m "feat(upcoming-starts): implement find_anchor_index"
```

---

### Task 6: Implement `project_start_indices`

**Files:**
- Modify: `src/fantasy_baseball/lineup/upcoming_starts.py`
- Test: `tests/test_lineup/test_upcoming_starts.py`

- [ ] **Step 1: Write the failing tests**

Append:

```python
from fantasy_baseball.lineup.upcoming_starts import project_start_indices


class TestProjectStartIndices:
    def test_simple_rotation_one_projection(self):
        # 10 games total, anchor at index 2 -> projections at 7
        assert project_start_indices(anchor_index=2, total_games=10, step=5) == [7]

    def test_two_projections_within_window(self):
        # anchor 0, total 12 -> 5, 10
        assert project_start_indices(anchor_index=0, total_games=12, step=5) == [5, 10]

    def test_no_projection_when_anchor_at_end(self):
        assert project_start_indices(anchor_index=7, total_games=10, step=5) == []

    def test_anchor_index_negative_returns_empty(self):
        assert project_start_indices(anchor_index=-1, total_games=10, step=5) == []

    def test_step_other_than_five(self):
        # 6-man rotation = step 6
        assert project_start_indices(anchor_index=0, total_games=20, step=6) == [6, 12, 18]
```

- [ ] **Step 2: Run tests to confirm they fail**

```bash
pytest tests/test_lineup/test_upcoming_starts.py::TestProjectStartIndices -v
```

Expected: FAIL — `NotImplementedError`.

- [ ] **Step 3: Implement**

Replace the stub:

```python
def project_start_indices(
    anchor_index: int,
    total_games: int,
    step: int = 5,
) -> list[int]:
    if anchor_index < 0:
        return []
    indices: list[int] = []
    nxt = anchor_index + step
    while nxt < total_games:
        indices.append(nxt)
        nxt += step
    return indices
```

- [ ] **Step 4: Run tests to confirm they pass**

```bash
pytest tests/test_lineup/test_upcoming_starts.py -v
```

Expected: ALL PASS.

- [ ] **Step 5: Commit**

```bash
git add src/fantasy_baseball/lineup/upcoming_starts.py tests/test_lineup/test_upcoming_starts.py
git commit -m "feat(upcoming-starts): implement project_start_indices"
```

---

### Task 7: Implement `compose_pitcher_entries` (merge announced + projected)

**Files:**
- Modify: `src/fantasy_baseball/lineup/upcoming_starts.py`
- Test: `tests/test_lineup/test_upcoming_starts.py`

**Why:** This is the central composition step — merging announced starts with rotation projections, dropping projection collisions where MLB has already announced someone else, and decorating with matchup quality.

- [ ] **Step 1: Write the failing tests**

Append:

```python
from datetime import date as _date

from fantasy_baseball.lineup.upcoming_starts import compose_pitcher_entries


def _seq(*specs):
    """Build a team game index from compact specs: (date, opp, ann, indicator, num)."""
    out = []
    for spec in specs:
        d, opp, ann = spec[:3]
        ind = spec[3] if len(spec) > 3 else "@"
        num = spec[4] if len(spec) > 4 else 1
        out.append(
            GameSlot(
                date=d, game_number=num, opponent=opp, indicator=ind, announced_starter=ann
            )
        )
    return out


_FACTORS = {
    "TEX": {"era_whip_factor": 0.90, "k_factor": 1.05},  # Great
    "LAD": {"era_whip_factor": 1.10, "k_factor": 0.95},  # Tough
    "HOU": {"era_whip_factor": 1.00, "k_factor": 1.00},  # Fair
}
_TEAM_STATS = {
    "TEX": {"ops": 0.700, "k_pct": 0.24},
    "LAD": {"ops": 0.800, "k_pct": 0.20},
    "HOU": {"ops": 0.750, "k_pct": 0.22},
}
_OPS_RANK = {"TEX": 25, "LAD": 4, "HOU": 14}
_K_RANK = {"TEX": 8, "LAD": 26, "HOU": 14}


class TestComposePitcherEntries:
    def test_simple_5_day_rotation_no_off_day(self):
        # Mon..Sun. Anchor: pitcher started Mon -> projected Sat.
        team_games = _seq(
            ("2026-05-04", "TEX", "Bryan Woo", "@"),  # past anchor (yesterday)
            ("2026-05-05", "TEX", "", "@"),
            ("2026-05-06", "TEX", "", "@"),
            ("2026-05-07", "TEX", "", "@"),
            ("2026-05-08", "TEX", "", "@"),
            ("2026-05-09", "TEX", "", "@"),  # +5 -> projected
            ("2026-05-10", "TEX", "", "@"),
        )
        entries = compose_pitcher_entries(
            "Bryan Woo",
            team_games,
            today=_date(2026, 5, 5),
            window_start=_date(2026, 5, 5),
            window_end=_date(2026, 5, 11),
            matchup_factors=_FACTORS,
            team_stats=_TEAM_STATS,
            ops_rank_map=_OPS_RANK,
            k_rank_map=_K_RANK,
        )
        assert len(entries) == 1
        e = entries[0]
        assert e.date == "2026-05-09"
        assert e.day == "Sat"
        assert e.opponent == "TEX"
        assert e.announced is False
        assert e.matchup_quality == "Great"
        assert e.detail["ops_rank"] == 25

    def test_off_day_extends_calendar_gap(self):
        # Anchor Mon (idx 0). Team games after anchor: Tue, Wed, [off Thu],
        # Fri, Sat, Sun. Sun is the 5th team-game post-anchor (idx 5),
        # so the projected start lands on Sun = 2026-05-10. Compare with
        # test_simple_5_day_rotation_no_off_day: same anchor, but no off-day
        # so the 5th team-game is Sat = 2026-05-09. The off-day pushes the
        # next start one calendar day later — that's the gap "extension."
        team_games = _seq(
            ("2026-05-04", "TEX", "Bryan Woo", "@"),  # anchor (idx 0)
            ("2026-05-05", "TEX", "", "@"),           # idx 1
            ("2026-05-06", "TEX", "", "@"),           # idx 2
            # No 2026-05-07 entry — off day
            ("2026-05-08", "TEX", "", "@"),           # idx 3
            ("2026-05-09", "TEX", "", "@"),           # idx 4
            ("2026-05-10", "TEX", "", "@"),           # idx 5 — 5th team-game post-anchor
        )
        entries = compose_pitcher_entries(
            "Bryan Woo",
            team_games,
            today=_date(2026, 5, 5),
            window_start=_date(2026, 5, 5),
            window_end=_date(2026, 5, 11),
            matchup_factors=_FACTORS,
            team_stats=_TEAM_STATS,
            ops_rank_map=_OPS_RANK,
            k_rank_map=_K_RANK,
        )
        assert len(entries) == 1
        assert entries[0].date == "2026-05-10"

    def test_announced_start_takes_precedence_over_projection(self):
        # Anchor Mon, MLB announces same pitcher Sat. Result: 1 entry, announced=True.
        team_games = _seq(
            ("2026-05-04", "TEX", "Bryan Woo", "@"),  # anchor
            *[("2026-05-0" + str(d), "TEX", "", "@") for d in range(5, 9)],
            ("2026-05-09", "TEX", "Bryan Woo", "@"),  # announced same date as projection
            ("2026-05-10", "TEX", "", "@"),
        )
        entries = compose_pitcher_entries(
            "Bryan Woo",
            team_games,
            today=_date(2026, 5, 5),
            window_start=_date(2026, 5, 5),
            window_end=_date(2026, 5, 11),
            matchup_factors=_FACTORS,
            team_stats=_TEAM_STATS,
            ops_rank_map=_OPS_RANK,
            k_rank_map=_K_RANK,
        )
        assert len(entries) == 1
        assert entries[0].date == "2026-05-09"
        assert entries[0].announced is True

    def test_announced_other_pitcher_drops_projection(self):
        # Projection lands on a game where MLB has someone else announced.
        team_games = _seq(
            ("2026-05-04", "TEX", "Bryan Woo", "@"),  # anchor
            *[("2026-05-0" + str(d), "TEX", "", "@") for d in range(5, 9)],
            ("2026-05-09", "TEX", "Castillo", "@"),  # someone else announced
            ("2026-05-10", "TEX", "", "@"),
        )
        entries = compose_pitcher_entries(
            "Bryan Woo",
            team_games,
            today=_date(2026, 5, 5),
            window_start=_date(2026, 5, 5),
            window_end=_date(2026, 5, 11),
            matchup_factors=_FACTORS,
            team_stats=_TEAM_STATS,
            ops_rank_map=_OPS_RANK,
            k_rank_map=_K_RANK,
        )
        assert entries == []

    def test_two_start_week_from_old_anchor(self):
        # Build a 21-game team stream where the anchor sits at index 0
        # (2026-04-25, ten days before the window opens). Anchor + 5/10/15/20
        # land at indices 5, 10, 15, 20 — i.e. dates 04-30, 05-05, 05-10, 05-15.
        # Window is 2026-05-05..2026-05-11, so projections at 05-05 and 05-10
        # both fall inside it; 04-30 (before) and 05-15 (after) are excluded.
        from datetime import timedelta as _td
        base = _date(2026, 4, 25)
        team_games: list[GameSlot] = []
        for i in range(21):
            d = (base + _td(days=i)).isoformat()
            ann = "Bryan Woo" if i == 0 else ""
            opp = "TEX" if i < 14 else "LAD"
            team_games.append(
                GameSlot(date=d, game_number=1, opponent=opp,
                         indicator="@", announced_starter=ann)
            )

        entries = compose_pitcher_entries(
            "Bryan Woo",
            team_games,
            today=_date(2026, 5, 5),
            window_start=_date(2026, 5, 5),
            window_end=_date(2026, 5, 11),
            matchup_factors=_FACTORS,
            team_stats=_TEAM_STATS,
            ops_rank_map=_OPS_RANK,
            k_rank_map=_K_RANK,
        )
        assert len(entries) == 2
        assert entries[0].date == "2026-05-05"
        assert entries[1].date == "2026-05-10"
        assert all(e.announced is False for e in entries)

    def test_no_anchor_yields_only_announced(self):
        # No past start by this pitcher; MLB announces them mid-week.
        team_games = _seq(
            ("2026-05-04", "TEX", "OtherGuy", "@"),
            *[("2026-05-0" + str(d), "TEX", "", "@") for d in range(5, 9)],
            ("2026-05-09", "TEX", "Bryan Woo", "@"),
        )
        entries = compose_pitcher_entries(
            "Bryan Woo",
            team_games,
            today=_date(2026, 5, 5),
            window_start=_date(2026, 5, 5),
            window_end=_date(2026, 5, 11),
            matchup_factors=_FACTORS,
            team_stats=_TEAM_STATS,
            ops_rank_map=_OPS_RANK,
            k_rank_map=_K_RANK,
        )
        assert len(entries) == 1
        assert entries[0].announced is True

    def test_empty_when_pitcher_has_no_starts(self):
        team_games = _seq(("2026-05-05", "TEX", "OtherGuy", "@"))
        entries = compose_pitcher_entries(
            "Bryan Woo",
            team_games,
            today=_date(2026, 5, 5),
            window_start=_date(2026, 5, 5),
            window_end=_date(2026, 5, 11),
            matchup_factors=_FACTORS,
            team_stats=_TEAM_STATS,
            ops_rank_map=_OPS_RANK,
            k_rank_map=_K_RANK,
        )
        assert entries == []
```

- [ ] **Step 2: Run tests to confirm they fail**

```bash
pytest tests/test_lineup/test_upcoming_starts.py::TestComposePitcherEntries -v
```

Expected: FAIL — `NotImplementedError`.

- [ ] **Step 3: Implement**

Add datetime import + helper in `src/fantasy_baseball/lineup/upcoming_starts.py`:

```python
from datetime import datetime as _datetime


def _day_name(date_iso: str) -> str:
    return _datetime.strptime(date_iso, "%Y-%m-%d").strftime("%a")


def _matchup_quality(
    factors: dict[str, dict[str, float]],
    team_stats: dict[str, dict[str, float]],
    opponent: str,
) -> str:
    if opponent in factors:
        f = factors[opponent]["era_whip_factor"]
        if f <= 0.93:
            return "Great"
        if f >= 1.03:
            return "Tough"
        return "Fair"
    if team_stats:
        avg_ops = sum(s["ops"] for s in team_stats.values()) / max(len(team_stats), 1)
        ops = team_stats.get(opponent, {}).get("ops", avg_ops)
        if ops < avg_ops * 0.95:
            return "Great"
        if ops > avg_ops * 1.05:
            return "Tough"
    return "Fair"


def _build_detail(
    team_stats: dict[str, dict[str, float]],
    ops_rank_map: dict[str, int],
    k_rank_map: dict[str, int],
    opponent: str,
) -> dict[str, Any]:
    opp = team_stats.get(opponent, {})
    raw_k = opp.get("k_pct", 0.0)
    k_display = round(raw_k * 100, 1) if raw_k < 1 else round(raw_k, 1)
    return {
        "ops": round(opp.get("ops", 0.0), 3),
        "ops_rank": ops_rank_map.get(opponent, 0),
        "k_pct": k_display,
        "k_rank": k_rank_map.get(opponent, 0),
    }


def compose_pitcher_entries(
    pitcher_name: str,
    team_games: list[GameSlot],
    today: date,
    window_start: date,
    window_end: date,
    matchup_factors: dict[str, dict[str, float]],
    team_stats: dict[str, dict[str, float]],
    ops_rank_map: dict[str, int],
    k_rank_map: dict[str, int],
) -> list[StartEntry]:
    target = normalize_name(pitcher_name)
    win_start_iso = window_start.isoformat()
    win_end_iso = window_end.isoformat()

    in_window = lambda d: win_start_iso <= d <= win_end_iso  # noqa: E731

    used_indices: set[int] = set()
    entries: list[StartEntry] = []

    # 1. Announced starts inside the window.
    for i, slot in enumerate(team_games):
        if not in_window(slot.date):
            continue
        if not slot.announced_starter:
            continue
        if normalize_name(slot.announced_starter) != target:
            continue
        used_indices.add(i)
        entries.append(
            StartEntry(
                date=slot.date,
                day=_day_name(slot.date),
                opponent=slot.opponent,
                indicator=slot.indicator,
                announced=True,
                matchup_quality=_matchup_quality(matchup_factors, team_stats, slot.opponent),
                detail=_build_detail(team_stats, ops_rank_map, k_rank_map, slot.opponent),
            )
        )

    # 2. Projected starts from the anchor.
    anchor = find_anchor_index(team_games, pitcher_name, today)
    if anchor is not None:
        for idx in project_start_indices(anchor, len(team_games), step=5):
            if idx in used_indices:
                continue
            slot = team_games[idx]
            if not in_window(slot.date):
                continue
            # Drop projection if MLB announced a different starter for this game.
            if slot.announced_starter and normalize_name(slot.announced_starter) != target:
                continue
            entries.append(
                StartEntry(
                    date=slot.date,
                    day=_day_name(slot.date),
                    opponent=slot.opponent,
                    indicator=slot.indicator,
                    announced=False,
                    matchup_quality=_matchup_quality(matchup_factors, team_stats, slot.opponent),
                    detail=_build_detail(team_stats, ops_rank_map, k_rank_map, slot.opponent),
                )
            )

    entries.sort(key=lambda e: e.date)
    return entries
```

- [ ] **Step 4: Run tests to confirm they pass**

```bash
pytest tests/test_lineup/test_upcoming_starts.py -v
```

Expected: ALL PASS.

- [ ] **Step 5: Commit**

```bash
git add src/fantasy_baseball/lineup/upcoming_starts.py tests/test_lineup/test_upcoming_starts.py
git commit -m "feat(upcoming-starts): merge announced + projected starts with matchup decoration"
```

---

### Task 8: SP-eligible + projected-GS filter helper

**Files:**
- Modify: `src/fantasy_baseball/lineup/upcoming_starts.py`
- Test: `tests/test_lineup/test_upcoming_starts.py`

**Why:** The refresh pipeline needs a single function it can call to filter the roster down to "real starting pitchers". Keeping it in `upcoming_starts.py` keeps all the new logic together.

- [ ] **Step 1: Write the failing tests**

Append:

```python
import pandas as pd

from fantasy_baseball.lineup.upcoming_starts import filter_starting_pitchers
from fantasy_baseball.models.player import Player, PlayerType
from fantasy_baseball.models.positions import Position


def _player(name, positions):
    return Player(name=name, player_type=PlayerType.PITCHER, positions=positions)


class TestFilterStartingPitchers:
    def _proj(self, rows):
        df = pd.DataFrame(rows)
        # Tests provide _name_norm explicitly to avoid coupling to normalize_name internals
        return df

    def test_keeps_sp_eligible_with_positive_gs(self):
        roster = [
            _player("Bryan Woo", [Position.SP]),
            _player("Mason Miller", [Position.RP]),
        ]
        proj = self._proj([
            {"_name_norm": "bryan woo", "gs": 28.0},
            {"_name_norm": "mason miller", "gs": 0.0},
        ])
        kept = filter_starting_pitchers(roster, proj)
        assert [p.name for p in kept] == ["Bryan Woo"]

    def test_drops_sp_eligible_with_zero_gs(self):
        # Swingman: SP+RP eligible but projected as a reliever.
        roster = [_player("AJ Puk", [Position.SP, Position.RP])]
        proj = self._proj([{"_name_norm": "aj puk", "gs": 0.0}])
        assert filter_starting_pitchers(roster, proj) == []

    def test_drops_sp_eligible_missing_from_projections(self):
        # No projection row at all -> excluded (can't verify gs > 0).
        roster = [_player("Unknown Guy", [Position.SP])]
        proj = self._proj([])
        assert filter_starting_pitchers(roster, proj) == []

    def test_drops_pure_rp(self):
        roster = [_player("Mason Miller", [Position.RP])]
        proj = self._proj([{"_name_norm": "mason miller", "gs": 0.0}])
        assert filter_starting_pitchers(roster, proj) == []

    def test_handles_missing_gs_column(self):
        # Older projection blob without a gs column -> exclude all (defensive).
        roster = [_player("Bryan Woo", [Position.SP])]
        proj = self._proj([{"_name_norm": "bryan woo", "ip": 180.0}])
        assert filter_starting_pitchers(roster, proj) == []
```

- [ ] **Step 2: Run tests to confirm they fail**

```bash
pytest tests/test_lineup/test_upcoming_starts.py::TestFilterStartingPitchers -v
```

Expected: FAIL — `filter_starting_pitchers` not defined.

- [ ] **Step 3: Implement**

Add to `src/fantasy_baseball/lineup/upcoming_starts.py`:

```python
import pandas as pd

from fantasy_baseball.models.positions import Position


def filter_starting_pitchers(
    roster: list[Any],
    pitchers_proj: pd.DataFrame,
) -> list[Any]:
    """Keep only roster members who are SP-eligible AND projected gs > 0.

    Players missing from the projection frame, or with a projection row
    that has no ``gs`` column / non-positive gs, are dropped.
    """
    if pitchers_proj is None or pitchers_proj.empty or "gs" not in pitchers_proj.columns:
        return []
    if "_name_norm" not in pitchers_proj.columns:
        # Defensive: refresh pipeline always attaches _name_norm, but tests may not.
        pitchers_proj = pitchers_proj.copy()
        pitchers_proj["_name_norm"] = pitchers_proj["name"].apply(normalize_name)

    gs_by_name = dict(zip(pitchers_proj["_name_norm"], pitchers_proj["gs"]))

    kept: list[Any] = []
    for player in roster:
        if Position.SP not in player.positions:
            continue
        gs = gs_by_name.get(normalize_name(player.name), 0.0) or 0.0
        if gs > 0:
            kept.append(player)
    return kept
```

- [ ] **Step 4: Run tests to confirm they pass**

```bash
pytest tests/test_lineup/test_upcoming_starts.py -v
```

Expected: ALL PASS.

- [ ] **Step 5: Commit**

```bash
git add src/fantasy_baseball/lineup/upcoming_starts.py tests/test_lineup/test_upcoming_starts.py
git commit -m "feat(upcoming-starts): add SP filter (eligible + projected gs > 0)"
```

---

### Task 9: Rewrite `get_probable_starters` to use the new module

**Files:**
- Modify: `src/fantasy_baseball/lineup/matchups.py:281–406`
- Test: `tests/test_lineup/test_matchups.py`

**Why:** `get_probable_starters` is the cache-write-side function the refresh pipeline calls. Rewrite it to use `compose_pitcher_entries`, keeping the existing per-pitcher rollup shape but adding the `announced` flag per start.

- [ ] **Step 1: Write the failing tests**

Append a new test class to `tests/test_lineup/test_matchups.py`:

```python
from datetime import date

from fantasy_baseball.lineup.matchups import get_probable_starters
from fantasy_baseball.models.player import Player, PlayerType
from fantasy_baseball.models.positions import Position


def _make_sched(pps):
    return {"probable_pitchers": pps}


def _pitcher(name, team="SEA"):
    p = Player(name=name, player_type=PlayerType.PITCHER, positions=[Position.SP])
    p.team = team
    return p


class TestGetProbableStartersV2:
    def test_announced_only_passes_through(self):
        sched = _make_sched([
            {"date": "2026-05-05", "game_number": 1, "away_team": "SEA",
             "home_team": "LAD", "away_pitcher": "Bryan Woo", "home_pitcher": "TBD"},
        ])
        team_stats = {"LAD": {"ops": 0.800, "k_pct": 0.20}}
        out = get_probable_starters(
            pitcher_roster=[_pitcher("Bryan Woo")],
            schedule=sched,
            matchup_factors={"LAD": {"era_whip_factor": 1.10, "k_factor": 0.95}},
            team_stats=team_stats,
            today=date(2026, 5, 5),
            window_start=date(2026, 5, 5),
            window_end=date(2026, 5, 11),
        )
        assert len(out) == 1
        assert out[0]["pitcher"] == "Bryan Woo"
        assert out[0]["starts"] == 1
        assert out[0]["matchups"][0]["announced"] is True

    def test_projected_added_when_no_announcement(self):
        # Anchor in lookback (May 1), team has no off-day; projection -> May 6.
        pps = [
            {"date": d, "game_number": 1, "away_team": "SEA",
             "home_team": "LAD", "away_pitcher": ann, "home_pitcher": "TBD"}
            for d, ann in [
                ("2026-05-01", "Bryan Woo"),  # anchor
                ("2026-05-02", ""),
                ("2026-05-03", ""),
                ("2026-05-04", ""),
                ("2026-05-05", ""),
                ("2026-05-06", ""),  # +5 -> projected
            ]
        ]
        out = get_probable_starters(
            pitcher_roster=[_pitcher("Bryan Woo")],
            schedule={"probable_pitchers": pps},
            matchup_factors={"LAD": {"era_whip_factor": 1.0, "k_factor": 1.0}},
            team_stats={"LAD": {"ops": 0.750, "k_pct": 0.22}},
            today=date(2026, 5, 5),
            window_start=date(2026, 5, 5),
            window_end=date(2026, 5, 11),
        )
        assert len(out) == 1
        assert out[0]["starts"] == 1
        assert out[0]["matchups"][0]["date"] == "2026-05-06"
        assert out[0]["matchups"][0]["announced"] is False

    def test_pitcher_with_no_starts_is_excluded(self):
        out = get_probable_starters(
            pitcher_roster=[_pitcher("Ghost Pitcher")],
            schedule={"probable_pitchers": [
                {"date": "2026-05-05", "game_number": 1, "away_team": "SEA",
                 "home_team": "LAD", "away_pitcher": "TBD", "home_pitcher": "TBD"},
            ]},
            matchup_factors={},
            team_stats={},
            today=date(2026, 5, 5),
            window_start=date(2026, 5, 5),
            window_end=date(2026, 5, 11),
        )
        assert out == []

    def test_player_team_used_to_select_team_games(self):
        # Pitcher's team is NYY; their LAD game shouldn't be considered.
        sched = _make_sched([
            {"date": "2026-05-05", "game_number": 1, "away_team": "SEA",
             "home_team": "LAD", "away_pitcher": "Bryan Woo", "home_pitcher": "TBD"},
            {"date": "2026-05-05", "game_number": 1, "away_team": "NYY",
             "home_team": "BOS", "away_pitcher": "Gerrit Cole", "home_pitcher": "TBD"},
        ])
        out = get_probable_starters(
            pitcher_roster=[_pitcher("Gerrit Cole", team="NYY")],
            schedule=sched,
            matchup_factors={"BOS": {"era_whip_factor": 1.0, "k_factor": 1.0}},
            team_stats={"BOS": {"ops": 0.750, "k_pct": 0.22}},
            today=date(2026, 5, 5),
            window_start=date(2026, 5, 5),
            window_end=date(2026, 5, 11),
        )
        assert len(out) == 1
        assert out[0]["matchups"][0]["opponent"] == "BOS"
```

- [ ] **Step 2: Run tests to confirm they fail**

```bash
pytest tests/test_lineup/test_matchups.py::TestGetProbableStartersV2 -v
```

Expected: FAIL — current `get_probable_starters` signature does not accept `today`/`window_start`/`window_end`.

- [ ] **Step 3: Implement**

Replace `get_probable_starters` (lines 281–406) in `src/fantasy_baseball/lineup/matchups.py`:

```python
def get_probable_starters(
    pitcher_roster: list[Any],
    schedule: dict[str, Any],
    matchup_factors: dict[str, dict[str, float]] | None = None,
    team_stats: dict[str, dict[str, float]] | None = None,
    today: "date | None" = None,
    window_start: "date | None" = None,
    window_end: "date | None" = None,
) -> list[dict[str, Any]]:
    """Build per-pitcher rollups of upcoming starts in the scoring week.

    Combines MLB-announced probables with rotation projections. Each
    rollup row carries:
        pitcher, starts, days, opponents, matchup_quality (worst-case),
        matchups (list of per-start StartEntry dicts including ``announced``).

    Args:
        pitcher_roster: roster pitchers (must have .name and .team).
        schedule: result of get_week_schedule() containing probable_pitchers.
        matchup_factors: result of calculate_matchup_factors().
        team_stats: raw team batting stats {abbrev: {ops, k_pct}}.
        today: cutoff for anchor lookup. Defaults to local_today().
        window_start, window_end: scoring-week bounds. Default to the
            schedule dict's start_date/end_date.
    """
    from datetime import date as _date

    from fantasy_baseball.lineup.upcoming_starts import (
        build_team_game_index,
        compose_pitcher_entries,
    )
    from fantasy_baseball.utils.time_utils import local_today

    if not schedule or not schedule.get("probable_pitchers"):
        return []

    pps = schedule["probable_pitchers"]

    if today is None:
        today = local_today()
    if window_start is None:
        window_start = _date.fromisoformat(schedule["start_date"])
    if window_end is None:
        window_end = _date.fromisoformat(schedule["end_date"])

    matchup_factors = matchup_factors or {}
    team_stats = team_stats or {}

    if team_stats:
        ops_ranked = sorted(team_stats.items(), key=lambda x: x[1]["ops"], reverse=True)
        k_ranked = sorted(team_stats.items(), key=lambda x: x[1]["k_pct"])
        ops_rank_map = {abbrev: i + 1 for i, (abbrev, _) in enumerate(ops_ranked)}
        k_rank_map = {abbrev: i + 1 for i, (abbrev, _) in enumerate(k_ranked)}
    else:
        ops_rank_map = {}
        k_rank_map = {}

    rollups: list[dict[str, Any]] = []
    for pitcher in pitcher_roster:
        team_abbrev = getattr(pitcher, "team", "") or ""
        if not team_abbrev:
            continue

        team_games = build_team_game_index(pps, team_abbrev)
        if not team_games:
            continue

        entries = compose_pitcher_entries(
            pitcher.name,
            team_games,
            today=today,
            window_start=window_start,
            window_end=window_end,
            matchup_factors=matchup_factors,
            team_stats=team_stats,
            ops_rank_map=ops_rank_map,
            k_rank_map=k_rank_map,
        )
        if not entries:
            continue

        matchups = [
            {
                "date": e.date,
                "day": e.day,
                "opponent": e.opponent,
                "indicator": e.indicator,
                "announced": e.announced,
                "matchup_quality": e.matchup_quality,
                "detail": e.detail,
            }
            for e in entries
        ]
        # Worst-of rollup quality: Tough > Fair > Great
        if any(m["matchup_quality"] == "Tough" for m in matchups):
            roll_quality = "Tough"
        elif any(m["matchup_quality"] == "Fair" for m in matchups):
            roll_quality = "Fair"
        else:
            roll_quality = "Great"

        rollups.append(
            {
                "pitcher": pitcher.name,
                "starts": len(matchups),
                "days": ", ".join(m["day"] for m in matchups),
                "opponents": ", ".join(f"{m['indicator']} {m['opponent']}" for m in matchups),
                "matchup_quality": roll_quality,
                "matchups": matchups,
            }
        )

    rollups.sort(key=lambda s: (-s["starts"], s["pitcher"]))
    return rollups
```

- [ ] **Step 4: Run tests to confirm they pass**

```bash
pytest tests/test_lineup/test_matchups.py tests/test_lineup/test_upcoming_starts.py -v
```

Expected: ALL PASS. Pre-existing matchups tests should also still pass.

- [ ] **Step 5: Commit**

```bash
git add src/fantasy_baseball/lineup/matchups.py tests/test_lineup/test_matchups.py
git commit -m "feat(matchups): rewrite get_probable_starters via upcoming_starts module"
```

---

### Task 10: Wire the refresh pipeline (lookback + SP filter)

**Files:**
- Modify: `src/fantasy_baseball/web/refresh_pipeline.py:843–875`
- Test: `tests/test_web/test_refresh_pipeline.py`

**Why:** The pipeline is the only production caller of `get_week_schedule` + `get_probable_starters`. It needs to (a) request `lookback_days=14`, (b) pre-filter the roster to SP+gs>0, and (c) pass through the cache without further changes.

- [ ] **Step 1: Read the existing refresh test fixture to understand the shape**

```bash
grep -n "PROBABLE_STARTERS\|probable_pitchers\|_fetch_probable_starters" tests/test_web/test_refresh_pipeline.py
```

Read the relevant fixture in `tests/test_web/_refresh_fixture.py` and the existing test before writing new ones.

- [ ] **Step 2: Add the failing test**

Append to `tests/test_web/test_refresh_pipeline.py` (adapt to whatever fixture pattern the file uses — the snippet below assumes `mock_statsapi_schedule` + a `RefreshRun` instance fixture; mirror the patterns already in that file):

```python
def test_fetch_probable_starters_passes_lookback_and_filters_sp(monkeypatch, refresh_run):
    """The refresh step must (a) request lookback_days=14 from get_week_schedule
    and (b) pass only SP-eligible + gs > 0 pitchers into get_probable_starters."""
    captured = {}

    def fake_get_week_schedule(start_date, end_date, cache_path, lookback_days=0):
        captured["lookback_days"] = lookback_days
        return {
            "probable_pitchers": [],
            "start_date": start_date,
            "end_date": end_date,
        }

    def fake_get_probable_starters(pitcher_roster, schedule, **kwargs):
        captured["roster_names"] = [p.name for p in pitcher_roster]
        return []

    monkeypatch.setattr(
        "fantasy_baseball.web.refresh_pipeline.get_week_schedule",
        fake_get_week_schedule,
        raising=False,
    )
    # Patch the imported symbol inside _fetch_probable_starters
    from fantasy_baseball.lineup import matchups as _matchups
    monkeypatch.setattr(_matchups, "get_probable_starters", fake_get_probable_starters)

    # ... seed refresh_run with: a mix of SP/RP roster players + a pitchers_proj
    # DataFrame containing gs values that exclude the RPs.
    # Then invoke refresh_run._fetch_probable_starters().

    refresh_run._fetch_probable_starters()

    assert captured["lookback_days"] == 14
    assert "Bryan Woo" in captured["roster_names"]
    assert "Mason Miller" not in captured["roster_names"]
```

(If the existing test file uses VCR cassettes or a different mocking pattern, mirror that pattern instead; the behavior assertions are what matter.)

- [ ] **Step 3: Run the test to confirm it fails**

```bash
pytest tests/test_web/test_refresh_pipeline.py -k "lookback" -v
```

Expected: FAIL — `lookback_days` is still 0; SP filter not applied.

- [ ] **Step 4: Implement**

Replace `_fetch_probable_starters` in `src/fantasy_baseball/web/refresh_pipeline.py` (~line 843):

```python
    # --- Step 9: Probable starters ---
    def _fetch_probable_starters(self):
        from fantasy_baseball.data.mlb_schedule import get_week_schedule
        from fantasy_baseball.lineup.matchups import (
            calculate_matchup_factors,
            get_probable_starters,
            get_team_batting_stats,
        )
        from fantasy_baseball.lineup.upcoming_starts import filter_starting_pitchers

        assert self.start_date is not None
        assert self.end_date is not None
        assert self.roster_players is not None
        assert self.pitchers_proj is not None

        self._progress("Fetching schedule and matchup data...")
        project_root = Path(__file__).resolve().parents[3]
        schedule_cache_path = project_root / "data" / "weekly_schedule.json"
        # 14-day lookback: the upcoming-starts module needs each pitcher's
        # most recent start as the rotation anchor for projecting forward.
        schedule = get_week_schedule(
            self.start_date,
            self.end_date,
            schedule_cache_path,
            lookback_days=14,
        )

        batting_stats_cache_path = project_root / "data" / "team_batting_stats.json"
        team_stats = get_team_batting_stats(batting_stats_cache_path)
        matchup_factors = calculate_matchup_factors(team_stats)

        sp_roster = filter_starting_pitchers(self.roster_players, self.pitchers_proj)

        probable_starters = get_probable_starters(
            sp_roster,
            schedule or {},
            matchup_factors=matchup_factors,
            team_stats=team_stats,
        )
        write_cache(CacheKey.PROBABLE_STARTERS, probable_starters)
```

- [ ] **Step 5: Run the test to confirm it passes**

```bash
pytest tests/test_web/test_refresh_pipeline.py -v
```

Expected: ALL PASS.

- [ ] **Step 6: Commit**

```bash
git add src/fantasy_baseball/web/refresh_pipeline.py tests/test_web/test_refresh_pipeline.py
git commit -m "feat(refresh): wire 14-day lookback + SP filter into probable-starters step"
```

---

### Task 11: Update the `lineup.html` template (chip rendering)

**Files:**
- Modify: `src/fantasy_baseball/web/templates/season/lineup.html:155–212`

**Why:** Each pitcher row should now render a chip per start, with chip background carrying the matchup-quality color and a dotted border for `announced=False`. The matchup column collapses into chips. The starts column keeps the 2-start badge.

- [ ] **Step 1: Replace the Probable Starters block**

In `src/fantasy_baseball/web/templates/season/lineup.html`, replace the block from `{# Probable starters section #}` through the closing `</table>` (lines 155–211 inclusive):

```html
{# Probable starters section #}
{% if starters %}
<div class="section-header" style="margin-top: 24px;">
    <h3>Upcoming Starts</h3>
</div>

<style>
  .upcoming-chip {
    display: inline-block;
    padding: 2px 8px;
    margin: 2px 4px 2px 0;
    border-radius: 12px;
    font-size: 0.85em;
    border: 1px solid transparent;
    white-space: nowrap;
  }
  .upcoming-chip.q-great {
    background: var(--badge-success-bg, #1b3b1b);
    color: var(--badge-success-fg, #c4f3c4);
    border-color: var(--badge-success-fg, #4caf50);
  }
  .upcoming-chip.q-fair {
    background: var(--badge-warning-bg, #3b3b1b);
    color: var(--badge-warning-fg, #f3e3c4);
    border-color: var(--badge-warning-fg, #d3a64c);
  }
  .upcoming-chip.q-tough {
    background: var(--badge-danger-bg, #3b1b1b);
    color: var(--badge-danger-fg, #f3c4c4);
    border-color: var(--badge-danger-fg, #d34c4c);
  }
  .upcoming-chip.projected {
    border-style: dashed;
    opacity: 0.85;
  }
</style>

<table class="data-table">
    <thead>
        <tr>
            <th style="text-align: left;">Pitcher</th>
            <th style="text-align: left;">Upcoming Starts</th>
            <th>Total</th>
        </tr>
    </thead>
    <tbody>
    {% for s in starters %}
        <tr>
            <td style="text-align: left; font-weight: 500;">{{ s.pitcher }}</td>
            <td style="text-align: left;">
                {% for m in s.matchups %}
                    <span class="upcoming-chip
                              {% if m.matchup_quality == 'Great' %}q-great{% elif m.matchup_quality == 'Tough' %}q-tough{% else %}q-fair{% endif %}
                              {% if not m.announced %}projected{% endif %}"
                          title="{{ m.day }} {{ m.indicator }} {{ m.opponent }}{% if m.detail and m.detail.ops %} — OPS {{ m.detail.ops }} ({{ m.detail.ops_rank }}th), K% {{ m.detail.k_pct }} ({{ m.detail.k_rank }}th){% endif %}{% if not m.announced %} · projected from rotation{% endif %}">
                        {{ m.day }} {{ m.indicator }}{{ m.opponent }}
                        {% if m.detail and m.detail.ops_rank %}
                          <span style="opacity: 0.75;">({{ m.detail.ops_rank }})</span>
                        {% endif %}
                    </span>
                {% endfor %}
            </td>
            <td>
                {{ s.starts }}
                {% if s.starts and s.starts | int >= 2 %}
                <span class="badge badge-info" style="margin-left: 4px;">2-start</span>
                {% endif %}
            </td>
        </tr>
    {% endfor %}
    </tbody>
</table>
{% endif %}
```

(Note: the existing `expand-content` row + `toggleExpand` JS handler is no longer needed for this table — the per-start detail now lives in the chip's `title` tooltip. The `toggleExpand` function may still be referenced by other tables in the same template; do NOT delete it without grep-confirming.)

- [ ] **Step 2: Confirm `toggleExpand` is still used elsewhere**

```bash
grep -n "toggleExpand\|expand-content\|expandable" src/fantasy_baseball/web/templates/season/lineup.html
```

If there are no remaining references after the edit, the function definition can stay (harmless) or be removed in a follow-up. Either way, leave it for this task — no scope creep.

- [ ] **Step 3: Manual visual verification**

Start the dev server (per CLAUDE.md "For UI or frontend changes, start the dev server and use the feature in a browser"):

```bash
python scripts/run_season_dashboard.py
```

Open `http://localhost:5050/lineup` (port from the script default) and confirm:

- The "Upcoming Starts" section appears with one row per starting pitcher.
- Chips show day + opponent + OPS rank.
- Announced starts have solid-bordered chips; projected starts have dashed-border chips.
- Hovering a chip shows the tooltip with full OPS/K% rank detail.
- The 2-start badge appears on rows where `starts >= 2`.
- No JavaScript console errors.

If you cannot run the dev server (e.g. no Yahoo auth available), say so explicitly in the commit message — do not silently claim visual verification.

- [ ] **Step 4: Commit**

```bash
git add src/fantasy_baseball/web/templates/season/lineup.html
git commit -m "feat(web): chip-style Upcoming Starts table with dashed-border projections"
```

---

### Task 12: Final verification

**Files:** none (verification only)

- [ ] **Step 1: Run the full test suite**

```bash
pytest -v
```

Expected: ALL PASS. If any tests fail that look unrelated, do not edit them — investigate the cause first per the CLAUDE.md "tests are the guardrail" rule.

- [ ] **Step 2: Run lint and format**

```bash
ruff check .
ruff format --check .
vulture
```

Expected: zero new violations. Pre-existing vulture findings unrelated to this work are acceptable; call them out if you see them.

- [ ] **Step 3: Run the live refresh against Yahoo**

```bash
python scripts/run_season_dashboard.py
```

Then trigger a refresh from the web UI (or POST `/api/refresh`). Confirm:

- The progress log shows "Fetching schedule and matchup data..." without errors.
- After completion, `/lineup` renders the new Upcoming Starts table with both announced and projected chips.
- `data/weekly_schedule.json` cache file exists and contains a `lookback_days: 14` field plus `probable_pitchers` entries dating back ~2 weeks before the scoring week start.

This step satisfies the "Run refresh before merge" memory rule for refresh-path changes.

- [ ] **Step 4: Final commit (only if any fixups were needed)**

If steps 1–3 surfaced fixes, commit them:

```bash
git add -p
git commit -m "fix(upcoming-starts): <specific fix>"
```

Otherwise no commit needed. Push the branch and open a PR (do not merge without asking the user — per the "no merge without asking" memory rule).

---

## Plan Self-Review

**Spec coverage:**
- § Scope (SP-eligible + gs > 0): Task 8 (filter helper), Task 10 (wired into refresh).
- § Window (Mon-Sun scoring week): inherited from existing `start_date`/`end_date`; passed through Task 9.
- § Lookback (14 days): Task 1 (param), Task 2 (cached entry point), Task 10 (refresh passes 14).
- § Anchor + projection logic: Tasks 4 (team game index), 5 (anchor), 6 (projection indices), 7 (composition + collision rules).
- § Cache shape with `announced` flag: Task 7 (StartEntry has `announced`), Task 9 (rollup carries it through).
- § Code organization (new module, refresh wiring, template): Tasks 3–11 cover each.
- § Testing: every code task includes failing test → impl → passing test loop. Task 12 is end-to-end + lint + live refresh.
- § Rollout (no migration, single refresh repopulates): inherent — only the refresh-side writer changes; readers see a superset.
- § Out of scope (opponent SPs, handedness, 6-man, recency filter): not introduced anywhere — confirmed by absence.

**Placeholder scan:** No "TBD"/"TODO"/"implement later" in any task. Every code step has the exact code.

**Type consistency:** `GameSlot`, `StartEntry`, `find_anchor_index`, `project_start_indices`, `compose_pitcher_entries`, `filter_starting_pitchers`, `build_team_game_index` — names match across Tasks 3 (definitions), 4–8 (implementations), and 9 (consumer). `lookback_days` parameter name matches across Tasks 1, 2, 10.

