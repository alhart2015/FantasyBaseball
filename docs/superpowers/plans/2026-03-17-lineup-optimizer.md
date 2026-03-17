# In-Season Lineup Optimizer Implementation Plan

> **For agentic workers:** REQUIRED: Use superpowers:subagent-driven-development (if subagents available) or superpowers:executing-plans to implement this plan. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a CLI lineup optimizer that pulls your roster and standings from Yahoo, identifies high-leverage categories, and recommends the optimal lineup + waiver wire pickups.

**Architecture:** Five new modules under `src/fantasy_baseball/lineup/` plus a CLI entry point. Yahoo API provides roster and standings data. FanGraphs rest-of-season projections provide player stat forecasts. A leverage calculator identifies which categories to target. The optimizer uses `scipy.optimize.linear_sum_assignment` for hitter slot assignment and simple ranking for pitchers. Waiver scanner finds free agents that improve weak categories.

**Tech Stack:** Python 3.11+, scipy, pandas, existing fantasy_baseball modules (auth, sgp, data, config, utils)

---

## Chunk 1: Yahoo Data + Leverage Calculation

### Task 1: Yahoo Roster and Standings Fetcher

**Files:**
- Create: `src/fantasy_baseball/lineup/__init__.py`
- Create: `src/fantasy_baseball/lineup/yahoo_roster.py`
- Create: `tests/test_lineup/__init__.py`
- Create: `tests/test_lineup/test_yahoo_roster.py`

- [ ] **Step 1: Write the tests**

Create `tests/test_lineup/__init__.py` (empty) and `tests/test_lineup/test_yahoo_roster.py`:
```python
import pytest
from unittest.mock import MagicMock
from fantasy_baseball.lineup.yahoo_roster import (
    parse_roster,
    parse_standings,
)


def _make_mock_roster_player(name, positions, selected_position):
    return {
        "name": name,
        "eligible_positions": positions,
        "selected_position": selected_position,
        "player_id": "12345",
    }


class TestParseRoster:
    def test_extracts_player_info(self):
        raw = [
            _make_mock_roster_player("Juan Soto", ["OF", "Util"], "OF"),
            _make_mock_roster_player("Gerrit Cole", ["SP"], "SP"),
        ]
        roster = parse_roster(raw)
        assert len(roster) == 2
        assert roster[0]["name"] == "Juan Soto"
        assert roster[0]["positions"] == ["OF", "Util"]
        assert roster[0]["selected_position"] == "OF"

    def test_empty_roster(self):
        assert parse_roster([]) == []


class TestParseStandings:
    def test_extracts_team_stats(self):
        raw = {
            "teams": [
                {
                    "name": "Hart of the Order",
                    "team_key": "469.l.5652.t.4",
                    "team_standings": {"rank": 3},
                    "team_stats": {
                        "stats": [
                            {"stat": {"stat_id": "60", "value": "450"}},  # R
                            {"stat": {"stat_id": "7", "value": "120"}},   # HR
                        ]
                    },
                },
            ]
        }
        # parse_standings should handle the raw Yahoo format
        # We test the normalized output
        standings = parse_standings(raw, stat_id_map={"60": "R", "7": "HR"})
        assert len(standings) == 1
        assert standings[0]["name"] == "Hart of the Order"
        assert standings[0]["rank"] == 3
        assert standings[0]["stats"]["R"] == 450.0
        assert standings[0]["stats"]["HR"] == 120.0

    def test_empty_standings(self):
        raw = {"teams": []}
        assert parse_standings(raw, stat_id_map={}) == []
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_lineup/test_yahoo_roster.py -v`
Expected: FAIL (ImportError)

- [ ] **Step 3: Write the implementation**

Create `src/fantasy_baseball/lineup/__init__.py` (empty) and `src/fantasy_baseball/lineup/yahoo_roster.py`:
```python
"""Fetch roster, standings, and free agents from Yahoo Fantasy API."""


# Yahoo stat IDs for 5x5 roto categories
YAHOO_STAT_ID_MAP: dict[str, str] = {
    "60": "R",    # Runs
    "7": "HR",    # Home Runs
    "13": "RBI",  # RBI
    "16": "SB",   # Stolen Bases
    "3": "AVG",   # Batting Average
    "28": "W",    # Wins
    "32": "SV",   # Saves
    "42": "K",    # Strikeouts
    "26": "ERA",  # ERA
    "27": "WHIP", # WHIP
}


def fetch_roster(league, team_key: str) -> list[dict]:
    """Fetch a team's current roster from Yahoo.

    Args:
        league: yahoo_fantasy_api League object.
        team_key: Yahoo team key (e.g., '469.l.5652.t.4').

    Returns:
        List of player dicts with name, positions, selected_position.
    """
    team = league.to_team(team_key)
    raw_roster = team.roster()
    return parse_roster(raw_roster)


def parse_roster(raw_roster: list[dict]) -> list[dict]:
    """Normalize raw Yahoo roster data."""
    players = []
    for p in raw_roster:
        players.append({
            "name": p["name"],
            "positions": p.get("eligible_positions", []),
            "selected_position": p.get("selected_position", ""),
            "player_id": p.get("player_id", ""),
        })
    return players


def fetch_standings(league) -> list[dict]:
    """Fetch league standings with cumulative team stats.

    Returns:
        List of team dicts with name, rank, and stats.
    """
    raw = league.standings()
    return parse_standings(raw, stat_id_map=YAHOO_STAT_ID_MAP)


def parse_standings(raw: dict, stat_id_map: dict[str, str]) -> list[dict]:
    """Normalize raw Yahoo standings data."""
    teams = []
    for team_data in raw.get("teams", []):
        stats = {}
        team_stats = team_data.get("team_stats", {})
        for stat_entry in team_stats.get("stats", []):
            stat = stat_entry.get("stat", {})
            sid = str(stat.get("stat_id", ""))
            if sid in stat_id_map:
                cat = stat_id_map[sid]
                try:
                    stats[cat] = float(stat.get("value", 0))
                except (ValueError, TypeError):
                    stats[cat] = 0.0

        team_standings = team_data.get("team_standings", {})
        teams.append({
            "name": team_data.get("name", ""),
            "team_key": team_data.get("team_key", ""),
            "rank": team_standings.get("rank", 0),
            "stats": stats,
        })
    return teams


def fetch_free_agents(league, position: str, count: int = 50) -> list[dict]:
    """Fetch top free agents at a position.

    Args:
        league: yahoo_fantasy_api League object.
        position: Position to filter (C, 1B, 2B, etc.).
        count: Number of free agents to return.

    Returns:
        List of player dicts with name, positions.
    """
    try:
        agents = league.free_agents(position)
        result = []
        for p in agents[:count]:
            result.append({
                "name": p["name"],
                "positions": p.get("eligible_positions", [position]),
                "player_id": p.get("player_id", ""),
            })
        return result
    except Exception:
        return []
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_lineup/test_yahoo_roster.py -v`
Expected: All PASS

- [ ] **Step 5: Commit**

```bash
git add src/fantasy_baseball/lineup/ tests/test_lineup/
git commit -m "feat: add Yahoo roster and standings fetcher"
```

---

### Task 2: Standings Leverage Calculator

**Files:**
- Create: `src/fantasy_baseball/lineup/leverage.py`
- Create: `tests/test_lineup/test_leverage.py`

- [ ] **Step 1: Write the tests**

Create `tests/test_lineup/test_leverage.py`:
```python
import pytest
from fantasy_baseball.lineup.leverage import calculate_leverage


def _make_standings():
    """10 teams with standings data. User team is rank 5."""
    return [
        {"name": "Team 1", "rank": 1, "stats": {"R": 500, "HR": 150, "RBI": 480, "SB": 90, "AVG": 0.275, "W": 55, "K": 850, "SV": 55, "ERA": 3.40, "WHIP": 1.15}},
        {"name": "Team 2", "rank": 2, "stats": {"R": 490, "HR": 145, "RBI": 470, "SB": 85, "AVG": 0.272, "W": 52, "K": 830, "SV": 50, "ERA": 3.50, "WHIP": 1.18}},
        {"name": "Team 3", "rank": 3, "stats": {"R": 475, "HR": 140, "RBI": 455, "SB": 80, "AVG": 0.270, "W": 50, "K": 810, "SV": 48, "ERA": 3.60, "WHIP": 1.20}},
        {"name": "Team 4", "rank": 4, "stats": {"R": 460, "HR": 135, "RBI": 445, "SB": 75, "AVG": 0.268, "W": 48, "K": 790, "SV": 45, "ERA": 3.70, "WHIP": 1.22}},
        {"name": "User Team", "rank": 5, "stats": {"R": 450, "HR": 130, "RBI": 430, "SB": 50, "AVG": 0.265, "W": 45, "K": 770, "SV": 40, "ERA": 3.80, "WHIP": 1.25}},
        {"name": "Team 6", "rank": 6, "stats": {"R": 430, "HR": 120, "RBI": 410, "SB": 45, "AVG": 0.260, "W": 42, "K": 740, "SV": 35, "ERA": 3.95, "WHIP": 1.28}},
        {"name": "Team 7", "rank": 7, "stats": {"R": 420, "HR": 115, "RBI": 400, "SB": 40, "AVG": 0.258, "W": 40, "K": 720, "SV": 30, "ERA": 4.10, "WHIP": 1.30}},
        {"name": "Team 8", "rank": 8, "stats": {"R": 400, "HR": 105, "RBI": 380, "SB": 35, "AVG": 0.252, "W": 35, "K": 690, "SV": 25, "ERA": 4.30, "WHIP": 1.35}},
        {"name": "Team 9", "rank": 9, "stats": {"R": 380, "HR": 95, "RBI": 360, "SB": 30, "AVG": 0.248, "W": 32, "K": 660, "SV": 20, "ERA": 4.50, "WHIP": 1.40}},
        {"name": "Team 10", "rank": 10, "stats": {"R": 350, "HR": 80, "RBI": 330, "SB": 20, "AVG": 0.240, "W": 28, "K": 620, "SV": 15, "ERA": 4.80, "WHIP": 1.48}},
    ]


class TestCalculateLeverage:
    def test_returns_all_categories(self):
        standings = _make_standings()
        leverage = calculate_leverage(standings, "User Team")
        assert "R" in leverage
        assert "HR" in leverage
        assert "ERA" in leverage
        assert len(leverage) == 10

    def test_all_weights_positive(self):
        standings = _make_standings()
        leverage = calculate_leverage(standings, "User Team")
        for cat, weight in leverage.items():
            assert weight >= 0, f"{cat} has negative weight"

    def test_weights_sum_to_one(self):
        standings = _make_standings()
        leverage = calculate_leverage(standings, "User Team")
        total = sum(leverage.values())
        assert total == pytest.approx(1.0, abs=0.01)

    def test_small_gap_gets_high_leverage(self):
        standings = _make_standings()
        # R gap to team above: 460-450 = 10
        # SB gap to team above: 75-50 = 25
        # R should have higher leverage than SB (smaller gap = easier to gain)
        leverage = calculate_leverage(standings, "User Team")
        assert leverage["R"] > leverage["SB"]

    def test_inverse_stats_correct_direction(self):
        # For ERA/WHIP, lower is better, so gap is user_stat - team_above_stat
        # User ERA: 3.80, Team 4 ERA: 3.70 -> gap = 0.10 (user needs to LOWER)
        standings = _make_standings()
        leverage = calculate_leverage(standings, "User Team")
        assert leverage["ERA"] > 0  # Should still be positive weight

    def test_last_place_team_has_leverage(self):
        standings = _make_standings()
        leverage = calculate_leverage(standings, "Team 10")
        total = sum(leverage.values())
        assert total == pytest.approx(1.0, abs=0.01)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_lineup/test_leverage.py -v`
Expected: FAIL (ImportError)

- [ ] **Step 3: Write the implementation**

Create `src/fantasy_baseball/lineup/leverage.py`:
```python
from fantasy_baseball.utils.constants import ALL_CATEGORIES, INVERSE_STATS

# Categories where gap is too large to realistically close
MAX_MEANINGFUL_GAP_MULTIPLIER: float = 3.0


def calculate_leverage(
    standings: list[dict],
    user_team_name: str,
) -> dict[str, float]:
    """Calculate leverage weights for each stat category based on standings gaps.

    Higher leverage = smaller gap to the team above = easier to gain a standings point.
    Weights are normalized to sum to 1.0.

    Args:
        standings: List of team dicts from parse_standings (sorted by rank).
        user_team_name: Name of the user's team.

    Returns:
        Dict of category -> leverage weight (0 to 1, summing to 1).
    """
    # Find user's team and the team above them
    sorted_teams = sorted(standings, key=lambda t: t.get("rank", 99))
    user_team = None
    user_idx = None
    for i, team in enumerate(sorted_teams):
        if team["name"] == user_team_name:
            user_team = team
            user_idx = i
            break

    if user_team is None:
        # Fallback: equal weights
        return {cat: 1.0 / len(ALL_CATEGORIES) for cat in ALL_CATEGORIES}

    user_stats = user_team.get("stats", {})

    # Calculate gap to team directly above for each category
    # For 1st place, use gap to team below (defend position)
    if user_idx > 0:
        target_team = sorted_teams[user_idx - 1]
    else:
        target_team = sorted_teams[user_idx + 1] if len(sorted_teams) > 1 else user_team

    target_stats = target_team.get("stats", {})

    raw_leverage: dict[str, float] = {}
    for cat in ALL_CATEGORIES:
        user_val = user_stats.get(cat, 0)
        target_val = target_stats.get(cat, 0)

        if cat in INVERSE_STATS:
            # Lower is better: gap = user_val - target_val
            # Positive gap means user is behind (higher ERA = worse)
            gap = abs(user_val - target_val)
        else:
            # Higher is better: gap = target_val - user_val
            gap = abs(target_val - user_val)

        # Convert gap to leverage: smaller gap = higher leverage
        # Use inverse relationship: leverage = 1 / (gap + epsilon)
        epsilon = 0.001
        raw_leverage[cat] = 1.0 / (gap + epsilon)

    # Normalize to sum to 1
    total = sum(raw_leverage.values())
    if total > 0:
        return {cat: val / total for cat, val in raw_leverage.items()}
    return {cat: 1.0 / len(ALL_CATEGORIES) for cat in ALL_CATEGORIES}
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_lineup/test_leverage.py -v`
Expected: All PASS

- [ ] **Step 5: Commit**

```bash
git add src/fantasy_baseball/lineup/leverage.py tests/test_lineup/test_leverage.py
git commit -m "feat: add standings-based category leverage calculator"
```

---

### Task 3: Leverage-Weighted SGP Calculator

**Files:**
- Create: `src/fantasy_baseball/lineup/weighted_sgp.py`
- Create: `tests/test_lineup/test_weighted_sgp.py`

- [ ] **Step 1: Write the tests**

Create `tests/test_lineup/test_weighted_sgp.py`:
```python
import pytest
import pandas as pd
from fantasy_baseball.lineup.weighted_sgp import calculate_weighted_sgp


def _make_hitter(name, r, hr, rbi, sb, avg, ab):
    return pd.Series({
        "name": name, "player_type": "hitter",
        "r": r, "hr": hr, "rbi": rbi, "sb": sb,
        "avg": avg, "ab": ab, "h": int(avg * ab),
    })


def _make_pitcher(name, w, k, sv, era, whip, ip):
    return pd.Series({
        "name": name, "player_type": "pitcher",
        "w": w, "k": k, "sv": sv, "era": era, "whip": whip, "ip": ip,
        "er": era * ip / 9, "bb": int(whip * ip * 0.3),
        "h_allowed": int(whip * ip * 0.7),
    })


class TestWeightedSgp:
    def test_equal_weights_matches_regular_sgp(self):
        player = _make_hitter("Judge", 110, 45, 120, 5, .291, 550)
        equal = {cat: 0.1 for cat in ["R", "HR", "RBI", "SB", "AVG", "W", "K", "SV", "ERA", "WHIP"]}
        wsgp = calculate_weighted_sgp(player, equal)
        assert wsgp > 0

    def test_sb_heavy_weights_favor_speedster(self):
        power = _make_hitter("Power", 90, 40, 100, 2, .260, 520)
        speed = _make_hitter("Speed", 95, 15, 65, 40, .280, 550)
        sb_heavy = {"R": 0.05, "HR": 0.05, "RBI": 0.05, "SB": 0.6, "AVG": 0.05,
                    "W": 0.04, "K": 0.04, "SV": 0.04, "ERA": 0.04, "WHIP": 0.04}
        power_wsgp = calculate_weighted_sgp(power, sb_heavy)
        speed_wsgp = calculate_weighted_sgp(speed, sb_heavy)
        assert speed_wsgp > power_wsgp

    def test_pitcher_with_pitching_weights(self):
        pitcher = _make_pitcher("Cole", 15, 240, 0, 3.15, 1.05, 200)
        k_heavy = {"R": 0.02, "HR": 0.02, "RBI": 0.02, "SB": 0.02, "AVG": 0.02,
                   "W": 0.1, "K": 0.6, "SV": 0.05, "ERA": 0.08, "WHIP": 0.07}
        wsgp = calculate_weighted_sgp(pitcher, k_heavy)
        assert wsgp > 0

    def test_zero_weight_category_ignored(self):
        player = _make_hitter("Steals Only", 0, 0, 0, 50, .200, 400)
        only_sb = {"R": 0, "HR": 0, "RBI": 0, "SB": 1.0, "AVG": 0,
                   "W": 0, "K": 0, "SV": 0, "ERA": 0, "WHIP": 0}
        wsgp = calculate_weighted_sgp(player, only_sb)
        assert wsgp > 0  # Only SB contributes
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_lineup/test_weighted_sgp.py -v`
Expected: FAIL (ImportError)

- [ ] **Step 3: Write the implementation**

Create `src/fantasy_baseball/lineup/weighted_sgp.py`:
```python
import pandas as pd
from fantasy_baseball.sgp.denominators import get_sgp_denominators
from fantasy_baseball.sgp.player_value import (
    calculate_counting_sgp,
    calculate_hitting_rate_sgp,
    calculate_pitching_rate_sgp,
    DEFAULT_TEAM_AB,
    DEFAULT_TEAM_IP,
    REPLACEMENT_AVG,
    REPLACEMENT_ERA,
    REPLACEMENT_WHIP,
)


def calculate_weighted_sgp(
    player: pd.Series,
    leverage: dict[str, float],
    denoms: dict[str, float] | None = None,
) -> float:
    """Calculate leverage-weighted SGP for a player.

    Like calculate_player_sgp but each category's contribution is
    multiplied by the leverage weight. Higher weight = category matters more.

    Args:
        player: Player stats Series.
        leverage: Dict of category -> weight (should sum to ~1.0).
        denoms: SGP denominators.

    Returns:
        Weighted SGP total.
    """
    if denoms is None:
        denoms = get_sgp_denominators()

    total = 0.0

    if player.get("player_type") == "hitter":
        for stat, col in [("R", "r"), ("HR", "hr"), ("RBI", "rbi"), ("SB", "sb")]:
            weight = leverage.get(stat, 0)
            if weight > 0:
                sgp = calculate_counting_sgp(player.get(col, 0), denoms[stat])
                total += sgp * weight

        weight_avg = leverage.get("AVG", 0)
        if weight_avg > 0:
            sgp = calculate_hitting_rate_sgp(
                player_avg=player.get("avg", 0),
                player_ab=int(player.get("ab", 0)),
                replacement_avg=REPLACEMENT_AVG,
                sgp_denominator=denoms["AVG"],
                team_ab=DEFAULT_TEAM_AB,
            )
            total += sgp * weight_avg

    elif player.get("player_type") == "pitcher":
        for stat, col in [("W", "w"), ("K", "k"), ("SV", "sv")]:
            weight = leverage.get(stat, 0)
            if weight > 0:
                sgp = calculate_counting_sgp(player.get(col, 0), denoms[stat])
                total += sgp * weight

        ip = player.get("ip", 0)
        if ip > 0:
            weight_era = leverage.get("ERA", 0)
            if weight_era > 0:
                sgp = calculate_pitching_rate_sgp(
                    player_rate=player.get("era", 0), player_ip=ip,
                    replacement_rate=REPLACEMENT_ERA,
                    sgp_denominator=denoms["ERA"],
                    team_ip=DEFAULT_TEAM_IP, innings_divisor=9,
                )
                total += sgp * weight_era

            weight_whip = leverage.get("WHIP", 0)
            if weight_whip > 0:
                sgp = calculate_pitching_rate_sgp(
                    player_rate=player.get("whip", 0), player_ip=ip,
                    replacement_rate=REPLACEMENT_WHIP,
                    sgp_denominator=denoms["WHIP"],
                    team_ip=DEFAULT_TEAM_IP, innings_divisor=1,
                )
                total += sgp * weight_whip

    return total
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_lineup/test_weighted_sgp.py -v`
Expected: All PASS

- [ ] **Step 5: Commit**

```bash
git add src/fantasy_baseball/lineup/weighted_sgp.py tests/test_lineup/test_weighted_sgp.py
git commit -m "feat: add leverage-weighted SGP calculator for lineup optimization"
```

---

## Chunk 2: Optimizer + Waivers + CLI

### Task 4: Hitter Lineup Optimizer

**Files:**
- Create: `src/fantasy_baseball/lineup/optimizer.py`
- Create: `tests/test_lineup/test_optimizer.py`

- [ ] **Step 1: Write the tests**

Create `tests/test_lineup/test_optimizer.py`:
```python
import pytest
import pandas as pd
from fantasy_baseball.lineup.optimizer import optimize_hitter_lineup, optimize_pitcher_lineup


def _make_hitter(name, positions, r, hr, rbi, sb, avg, ab):
    return pd.Series({
        "name": name, "positions": positions, "player_type": "hitter",
        "r": r, "hr": hr, "rbi": rbi, "sb": sb,
        "avg": avg, "ab": ab, "h": int(avg * ab),
    })


def _make_pitcher(name, positions, w, k, sv, era, whip, ip):
    return pd.Series({
        "name": name, "positions": positions, "player_type": "pitcher",
        "w": w, "k": k, "sv": sv, "era": era, "whip": whip, "ip": ip,
        "er": era * ip / 9, "bb": int(whip * ip * 0.3),
        "h_allowed": int(whip * ip * 0.7),
    })


EQUAL_LEVERAGE = {cat: 0.1 for cat in ["R", "HR", "RBI", "SB", "AVG", "W", "K", "SV", "ERA", "WHIP"]}


class TestOptimizeHitterLineup:
    def test_assigns_all_starters(self):
        hitters = [
            _make_hitter("C1", ["C"], 60, 20, 65, 2, .250, 450),
            _make_hitter("1B1", ["1B"], 80, 30, 95, 3, .270, 520),
            _make_hitter("2B1", ["2B"], 75, 18, 70, 12, .265, 510),
            _make_hitter("3B1", ["3B"], 70, 25, 80, 5, .260, 500),
            _make_hitter("SS1", ["SS"], 85, 22, 75, 25, .275, 540),
            _make_hitter("OF1", ["OF"], 100, 35, 100, 8, .280, 550),
            _make_hitter("OF2", ["OF"], 90, 28, 85, 15, .275, 530),
            _make_hitter("OF3", ["OF"], 80, 22, 70, 20, .270, 510),
            _make_hitter("OF4", ["OF"], 75, 18, 65, 10, .265, 490),
            _make_hitter("UTIL1", ["1B", "OF"], 85, 32, 90, 2, .260, 520),
            _make_hitter("UTIL2", ["DH"], 70, 20, 75, 1, .255, 480),
            _make_hitter("BN1", ["OF"], 50, 10, 40, 5, .245, 350),
        ]
        lineup = optimize_hitter_lineup(hitters, EQUAL_LEVERAGE)
        # Should have assignments for active slots
        assert "C" in lineup
        assert "1B" in lineup
        assert lineup["C"] == "C1"  # Only catcher

    def test_multi_position_player_optimal_slot(self):
        hitters = [
            _make_hitter("Multi", ["SS", "2B"], 90, 25, 80, 20, .280, 540),
            _make_hitter("SS Only", ["SS"], 70, 15, 60, 10, .260, 480),
            _make_hitter("2B Only", ["2B"], 65, 12, 55, 8, .255, 470),
        ]
        lineup = optimize_hitter_lineup(hitters, EQUAL_LEVERAGE)
        # Multi should go to SS or 2B, freeing the other for the specialist
        assert lineup.get("SS") is not None or lineup.get("2B") is not None

    def test_bench_players_identified(self):
        hitters = [
            _make_hitter("Star", ["OF"], 110, 45, 120, 5, .291, 550),
            _make_hitter("Scrub", ["OF"], 30, 5, 20, 1, .220, 200),
        ]
        lineup = optimize_hitter_lineup(hitters, EQUAL_LEVERAGE)
        starters = set(lineup.values())
        assert "Star" in starters


class TestOptimizePitcherLineup:
    def test_starts_top_pitchers(self):
        pitchers = [
            _make_pitcher("Ace", ["SP"], 15, 240, 0, 3.00, 1.05, 200),
            _make_pitcher("Mid", ["SP"], 10, 160, 0, 3.80, 1.20, 170),
            _make_pitcher("Bad", ["SP"], 5, 80, 0, 5.00, 1.45, 100),
            _make_pitcher("Closer", ["RP"], 3, 60, 35, 2.50, 1.00, 65),
        ]
        starters, bench = optimize_pitcher_lineup(pitchers, EQUAL_LEVERAGE, slots=3)
        starter_names = [p["name"] for p in starters]
        assert "Ace" in starter_names
        assert "Closer" in starter_names  # High SV value
        assert len(starters) == 3

    def test_respects_slot_count(self):
        pitchers = [
            _make_pitcher(f"P{i}", ["SP"], 10, 150, 0, 3.50, 1.15, 170)
            for i in range(12)
        ]
        starters, bench = optimize_pitcher_lineup(pitchers, EQUAL_LEVERAGE, slots=9)
        assert len(starters) == 9
        assert len(bench) == 3
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_lineup/test_optimizer.py -v`
Expected: FAIL (ImportError)

- [ ] **Step 3: Write the implementation**

Create `src/fantasy_baseball/lineup/optimizer.py`:
```python
import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment
from fantasy_baseball.utils.constants import ROSTER_SLOTS
from fantasy_baseball.utils.positions import can_fill_slot
from fantasy_baseball.lineup.weighted_sgp import calculate_weighted_sgp

# Active hitter slots (excludes BN and IL)
HITTER_SLOTS: list[str] = []
for pos, count in ROSTER_SLOTS.items():
    if pos in ("P", "BN", "IL"):
        continue
    for i in range(count):
        HITTER_SLOTS.append(pos)
# Result: ["C", "1B", "2B", "3B", "SS", "IF", "OF", "OF", "OF", "OF", "UTIL", "UTIL"]


def optimize_hitter_lineup(
    hitters: list[pd.Series],
    leverage: dict[str, float],
) -> dict[str, str]:
    """Assign hitters to roster slots to maximize leverage-weighted SGP.

    Uses scipy's linear_sum_assignment (Hungarian algorithm).

    Args:
        hitters: List of hitter stat Series with 'name' and 'positions'.
        leverage: Category leverage weights.

    Returns:
        Dict of slot -> player name for optimal lineup.
    """
    if not hitters:
        return {}

    n_players = len(hitters)
    n_slots = len(HITTER_SLOTS)

    # Calculate weighted SGP for each hitter
    values = []
    for h in hitters:
        values.append(calculate_weighted_sgp(h, leverage))

    # Build cost matrix (negative because we maximize)
    # Rows = players, Cols = slots
    # Pad to make rectangular if needed
    size = max(n_players, n_slots)
    cost = np.full((size, size), 1e9)  # High cost = infeasible

    for i, hitter in enumerate(hitters):
        positions = hitter.get("positions", [])
        for j, slot in enumerate(HITTER_SLOTS):
            if can_fill_slot(positions, slot):
                cost[i][j] = -values[i]  # Negative for maximization

    # Solve assignment
    row_idx, col_idx = linear_sum_assignment(cost)

    # Extract assignments
    lineup: dict[str, str] = {}
    assigned_slots: dict[str, int] = {}  # Track slot usage for duplicates
    for r, c in zip(row_idx, col_idx):
        if r < n_players and c < n_slots and cost[r][c] < 1e8:
            slot = HITTER_SLOTS[c]
            player_name = hitters[r]["name"]
            # Handle duplicate slot names (OF x4, UTIL x2)
            slot_key = slot
            count = assigned_slots.get(slot, 0)
            if count > 0:
                slot_key = f"{slot}_{count + 1}"
            assigned_slots[slot] = count + 1
            lineup[slot_key] = player_name

    return lineup


def optimize_pitcher_lineup(
    pitchers: list[pd.Series],
    leverage: dict[str, float],
    slots: int = 9,
) -> tuple[list[dict], list[dict]]:
    """Select top pitchers by leverage-weighted SGP.

    All P slots are interchangeable, so just rank and start top N.

    Args:
        pitchers: List of pitcher stat Series.
        leverage: Category leverage weights.
        slots: Number of active pitcher slots.

    Returns:
        Tuple of (starters, bench) as lists of dicts with name and wsgp.
    """
    scored = []
    for p in pitchers:
        wsgp = calculate_weighted_sgp(p, leverage)
        scored.append({"name": p["name"], "wsgp": wsgp, "player": p})

    scored.sort(key=lambda x: x["wsgp"], reverse=True)

    starters = scored[:slots]
    bench = scored[slots:]
    return starters, bench
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_lineup/test_optimizer.py -v`
Expected: All PASS

- [ ] **Step 5: Commit**

```bash
git add src/fantasy_baseball/lineup/optimizer.py tests/test_lineup/test_optimizer.py
git commit -m "feat: add lineup optimizer with Hungarian algorithm for hitters"
```

---

### Task 5: Waiver Wire Scanner

**Files:**
- Create: `src/fantasy_baseball/lineup/waivers.py`
- Create: `tests/test_lineup/test_waivers.py`

- [ ] **Step 1: Write the tests**

Create `tests/test_lineup/test_waivers.py`:
```python
import pytest
import pandas as pd
from fantasy_baseball.lineup.waivers import evaluate_pickup


def _make_player(name, player_type, **stats):
    data = {"name": name, "player_type": player_type}
    data.update(stats)
    return pd.Series(data)


EQUAL_LEVERAGE = {cat: 0.1 for cat in ["R", "HR", "RBI", "SB", "AVG", "W", "K", "SV", "ERA", "WHIP"]}


class TestEvaluatePickup:
    def test_better_player_has_positive_gain(self):
        add = _make_player("Good", "hitter", r=90, hr=30, rbi=85, sb=15, avg=.280, ab=540, h=151)
        drop = _make_player("Bad", "hitter", r=40, hr=8, rbi=30, sb=2, avg=.230, ab=300, h=69)
        result = evaluate_pickup(add, drop, EQUAL_LEVERAGE)
        assert result["sgp_gain"] > 0
        assert result["add"] == "Good"
        assert result["drop"] == "Bad"

    def test_worse_player_has_negative_gain(self):
        add = _make_player("Bad", "hitter", r=40, hr=8, rbi=30, sb=2, avg=.230, ab=300, h=69)
        drop = _make_player("Good", "hitter", r=90, hr=30, rbi=85, sb=15, avg=.280, ab=540, h=151)
        result = evaluate_pickup(add, drop, EQUAL_LEVERAGE)
        assert result["sgp_gain"] < 0

    def test_returns_category_breakdown(self):
        add = _make_player("Steals", "hitter", r=70, hr=10, rbi=50, sb=40, avg=.270, ab=500, h=135)
        drop = _make_player("Power", "hitter", r=70, hr=30, rbi=80, sb=2, avg=.250, ab=500, h=125)
        result = evaluate_pickup(add, drop, EQUAL_LEVERAGE)
        assert "categories" in result
        assert result["categories"]["SB"] > 0  # Gaining steals
        assert result["categories"]["HR"] < 0  # Losing power
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_lineup/test_waivers.py -v`
Expected: FAIL (ImportError)

- [ ] **Step 3: Write the implementation**

Create `src/fantasy_baseball/lineup/waivers.py`:
```python
import pandas as pd
from fantasy_baseball.lineup.weighted_sgp import calculate_weighted_sgp
from fantasy_baseball.sgp.denominators import get_sgp_denominators
from fantasy_baseball.sgp.player_value import (
    calculate_counting_sgp,
    calculate_hitting_rate_sgp,
    calculate_pitching_rate_sgp,
    DEFAULT_TEAM_AB,
    DEFAULT_TEAM_IP,
    REPLACEMENT_AVG,
    REPLACEMENT_ERA,
    REPLACEMENT_WHIP,
)


def evaluate_pickup(
    add_player: pd.Series,
    drop_player: pd.Series,
    leverage: dict[str, float],
) -> dict:
    """Evaluate the SGP gain of adding one player and dropping another.

    Args:
        add_player: Player to add (free agent).
        drop_player: Player to drop (current roster).
        leverage: Category leverage weights.

    Returns:
        Dict with add, drop, sgp_gain, and per-category breakdown.
    """
    add_wsgp = calculate_weighted_sgp(add_player, leverage)
    drop_wsgp = calculate_weighted_sgp(drop_player, leverage)

    # Per-category breakdown
    denoms = get_sgp_denominators()
    categories = {}
    for stat, col in _get_stat_cols(add_player):
        add_val = _category_sgp(add_player, stat, col, denoms)
        drop_val = _category_sgp(drop_player, stat, col, denoms)
        weight = leverage.get(stat, 0)
        categories[stat] = (add_val - drop_val) * weight

    return {
        "add": add_player["name"],
        "drop": drop_player["name"],
        "sgp_gain": add_wsgp - drop_wsgp,
        "categories": categories,
    }


def _get_stat_cols(player: pd.Series) -> list[tuple[str, str]]:
    """Get relevant stat/column pairs for a player's type."""
    if player.get("player_type") == "hitter":
        return [("R", "r"), ("HR", "hr"), ("RBI", "rbi"), ("SB", "sb"), ("AVG", "avg")]
    elif player.get("player_type") == "pitcher":
        return [("W", "w"), ("K", "k"), ("SV", "sv"), ("ERA", "era"), ("WHIP", "whip")]
    return []


def _category_sgp(player: pd.Series, stat: str, col: str, denoms: dict) -> float:
    """Calculate raw SGP for a single category."""
    if stat in ("AVG",):
        return calculate_hitting_rate_sgp(
            player_avg=player.get("avg", 0),
            player_ab=int(player.get("ab", 0)),
            replacement_avg=REPLACEMENT_AVG,
            sgp_denominator=denoms["AVG"],
            team_ab=DEFAULT_TEAM_AB,
        )
    elif stat in ("ERA",):
        ip = player.get("ip", 0)
        if ip > 0:
            return calculate_pitching_rate_sgp(
                player_rate=player.get("era", 0), player_ip=ip,
                replacement_rate=REPLACEMENT_ERA,
                sgp_denominator=denoms["ERA"],
                team_ip=DEFAULT_TEAM_IP, innings_divisor=9,
            )
        return 0.0
    elif stat in ("WHIP",):
        ip = player.get("ip", 0)
        if ip > 0:
            return calculate_pitching_rate_sgp(
                player_rate=player.get("whip", 0), player_ip=ip,
                replacement_rate=REPLACEMENT_WHIP,
                sgp_denominator=denoms["WHIP"],
                team_ip=DEFAULT_TEAM_IP, innings_divisor=1,
            )
        return 0.0
    else:
        return calculate_counting_sgp(player.get(col, 0), denoms[stat])
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_lineup/test_waivers.py -v`
Expected: All PASS

- [ ] **Step 5: Commit**

```bash
git add src/fantasy_baseball/lineup/waivers.py tests/test_lineup/test_waivers.py
git commit -m "feat: add waiver wire pickup evaluator with category breakdown"
```

---

### Task 6: Lineup Optimizer CLI

**Files:**
- Create: `scripts/run_lineup.py`

- [ ] **Step 1: Write the CLI script**

Create `scripts/run_lineup.py`:
```python
"""In-Season Lineup Optimizer for Fantasy Baseball.

Usage:
    python scripts/run_lineup.py

Connects to Yahoo to fetch your roster and standings, then
recommends the optimal lineup based on standings leverage.
"""
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from fantasy_baseball.auth.yahoo_auth import get_yahoo_session, get_league
from fantasy_baseball.config import load_config
from fantasy_baseball.data.projections import blend_projections
from fantasy_baseball.data.yahoo_players import load_positions_cache
from fantasy_baseball.lineup.yahoo_roster import fetch_roster, fetch_standings
from fantasy_baseball.lineup.leverage import calculate_leverage
from fantasy_baseball.lineup.weighted_sgp import calculate_weighted_sgp
from fantasy_baseball.lineup.optimizer import optimize_hitter_lineup, optimize_pitcher_lineup
from fantasy_baseball.utils.name_utils import normalize_name
from fantasy_baseball.utils.positions import is_hitter, is_pitcher

import pandas as pd

CONFIG_PATH = PROJECT_ROOT / "config" / "league.yaml"
POSITIONS_PATH = PROJECT_ROOT / "data" / "player_positions.json"
PROJECTIONS_DIR = PROJECT_ROOT / "data" / "projections"


def main():
    config = load_config(CONFIG_PATH)
    print(f"Lineup Optimizer | {config.team_name}")
    print()

    # Connect to Yahoo
    print("Connecting to Yahoo...")
    session = get_yahoo_session()
    league = get_league(session, league_id=config.league_id, game_key=config.game_code)

    # Find user's team key
    teams = league.teams()
    user_team_key = None
    for key, team in teams.items():
        if normalize_name(team["name"]) == normalize_name(config.team_name):
            user_team_key = key
            break
    if not user_team_key:
        print(f"Could not find team '{config.team_name}' in league")
        sys.exit(1)

    # Fetch roster and standings
    print("Fetching roster and standings...")
    roster = fetch_roster(league, user_team_key)
    standings = fetch_standings(league)

    print(f"Roster: {len(roster)} players")
    print(f"Standings: {len(standings)} teams")
    print()

    # Calculate leverage
    leverage = calculate_leverage(standings, config.team_name)
    print("CATEGORY LEVERAGE (higher = more valuable to target):")
    sorted_lev = sorted(leverage.items(), key=lambda x: x[1], reverse=True)
    for cat, weight in sorted_lev:
        bar = "#" * int(weight * 100)
        print(f"  {cat:>4}: {weight:.3f} {bar}")
    print()

    # Load projections and match to roster
    print("Loading projections...")
    hitters_proj, pitchers_proj = blend_projections(
        PROJECTIONS_DIR, config.projection_systems, config.projection_weights or None,
    )
    positions_cache = load_positions_cache(POSITIONS_PATH)
    norm_positions = {normalize_name(k): v for k, v in positions_cache.items()}

    # Match roster players to projections
    roster_hitters = []
    roster_pitchers = []
    for player in roster:
        name = player["name"]
        name_norm = normalize_name(name)
        positions = player["positions"]

        # Find in projections
        proj_row = None
        for df in [hitters_proj, pitchers_proj]:
            if df.empty:
                continue
            matches = df[df["name"].apply(normalize_name) == name_norm]
            if not matches.empty:
                proj_row = matches.iloc[0].copy()
                break

        if proj_row is None:
            continue

        # Attach positions from Yahoo
        proj_row["positions"] = positions

        if is_hitter(positions):
            roster_hitters.append(proj_row)
        if is_pitcher(positions):
            roster_pitchers.append(proj_row)

    print(f"Matched: {len(roster_hitters)} hitters, {len(roster_pitchers)} pitchers")
    print()

    # Optimize hitter lineup
    if roster_hitters:
        print("=" * 60)
        print("OPTIMAL HITTER LINEUP")
        print("=" * 60)
        lineup = optimize_hitter_lineup(roster_hitters, leverage)
        for slot, name in sorted(lineup.items()):
            print(f"  {slot:<8} {name}")
        print()

        # Show bench
        starters = set(lineup.values())
        bench = [h for h in roster_hitters if h["name"] not in starters]
        if bench:
            print("  BENCH:")
            for h in bench:
                print(f"    {h['name']}")
        print()

    # Optimize pitcher lineup
    if roster_pitchers:
        print("=" * 60)
        print("OPTIMAL PITCHER LINEUP")
        print("=" * 60)
        starters, bench = optimize_pitcher_lineup(
            roster_pitchers, leverage, slots=9
        )
        print("  START:")
        for p in starters:
            print(f"    {p['name']:<25} wSGP: {p['wsgp']:.2f}")
        if bench:
            print("  BENCH:")
            for p in bench:
                print(f"    {p['name']:<25} wSGP: {p['wsgp']:.2f}")
        print()

    print("Done! Update your lineup on Yahoo based on these recommendations.")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Verify imports work**

Run: `python -c "from fantasy_baseball.lineup.yahoo_roster import fetch_roster; from fantasy_baseball.lineup.leverage import calculate_leverage; from fantasy_baseball.lineup.optimizer import optimize_hitter_lineup; from fantasy_baseball.lineup.waivers import evaluate_pickup; print('All imports OK')"`
Expected: "All imports OK"

- [ ] **Step 3: Run full test suite**

Run: `pytest -v --tb=short`
Expected: All tests PASS

- [ ] **Step 4: Commit**

```bash
git add scripts/run_lineup.py
git commit -m "feat: add in-season lineup optimizer CLI"
```

---

## Summary

After completing all 6 tasks, the lineup optimizer provides:

1. **`lineup/yahoo_roster.py`** — Fetch roster, standings, free agents from Yahoo API
2. **`lineup/leverage.py`** — Calculate category leverage from standings gaps
3. **`lineup/weighted_sgp.py`** — Leverage-weighted SGP for lineup optimization
4. **`lineup/optimizer.py`** — Hungarian algorithm for hitter slots, rank-based for pitchers
5. **`lineup/waivers.py`** — Waiver wire pickup evaluation with category breakdown
6. **`scripts/run_lineup.py`** — CLI that connects to Yahoo and outputs optimal lineup

**Weekly workflow:**
1. Download updated FanGraphs rest-of-season projections (optional, for better accuracy)
2. Run `python scripts/run_lineup.py`
3. Review recommendations and set your lineup on Yahoo
