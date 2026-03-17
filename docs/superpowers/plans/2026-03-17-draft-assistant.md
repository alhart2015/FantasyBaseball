# Draft Assistant Implementation Plan

> **For agentic workers:** REQUIRED: Use superpowers:subagent-driven-development (if subagents available) or superpowers:executing-plans to implement this plan. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build an interactive CLI draft assistant that recommends picks during a live snake draft, tracking positional needs and category balance in real-time.

**Architecture:** Six new modules under `src/fantasy_baseball/draft/` plus a CLI entry point. The draft board is pre-computed from blended projections + Yahoo position data. During the draft, a tracker manages pick state, a recommender suggests picks factoring VAR + positional scarcity + category balance, and fuzzy search handles name input. All existing Phase 1 modules (data, sgp, config) are consumed unchanged.

**Tech Stack:** Python 3.11+, pandas, difflib (stdlib), existing fantasy_baseball modules

**Config change required:** Add `team_name` field to `LeagueConfig` and `config/league.yaml` (under `league:` section) so the CLI knows which team is the user's. This avoids hardcoding team names.

**Pre-requisites before draft day:**
1. Download FanGraphs Steamer/ZiPS hitter + pitcher CSVs → `data/projections/`
2. Run `python scripts/fetch_positions.py` to cache Yahoo position data
3. Run `python scripts/run_draft.py` to start

---

## Chunk 1: Data Preparation + Board Building

### Task 1: Yahoo Position Fetcher

**Files:**
- Create: `src/fantasy_baseball/data/yahoo_players.py`
- Create: `tests/test_data/test_yahoo_players.py`
- Create: `scripts/fetch_positions.py`

- [ ] **Step 1: Write the tests**

Create `tests/test_data/test_yahoo_players.py`:
```python
import pytest
import json
from pathlib import Path
from fantasy_baseball.data.yahoo_players import (
    merge_position_maps,
    load_positions_cache,
    save_positions_cache,
)


class TestMergePositionMaps:
    def test_merges_two_positions(self):
        maps = [
            {"Player A": ["C"], "Player B": ["1B"]},
            {"Player A": ["1B"], "Player C": ["OF"]},
        ]
        merged = merge_position_maps(maps)
        assert set(merged["Player A"]) == {"C", "1B"}
        assert merged["Player B"] == ["1B"]
        assert merged["Player C"] == ["OF"]

    def test_deduplicates(self):
        maps = [
            {"Player A": ["SS"]},
            {"Player A": ["SS", "2B"]},
        ]
        merged = merge_position_maps(maps)
        assert sorted(merged["Player A"]) == ["2B", "SS"]

    def test_empty_input(self):
        assert merge_position_maps([]) == {}


class TestPositionsCache:
    def test_save_and_load_roundtrip(self, tmp_path):
        positions = {"Aaron Judge": ["OF"], "Gerrit Cole": ["SP"]}
        cache_path = tmp_path / "positions.json"
        save_positions_cache(positions, cache_path)
        loaded = load_positions_cache(cache_path)
        assert loaded == positions

    def test_load_missing_file_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            load_positions_cache(tmp_path / "nope.json")
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_data/test_yahoo_players.py -v`
Expected: FAIL (ImportError)

- [ ] **Step 3: Write the implementation**

Create `src/fantasy_baseball/data/yahoo_players.py`:
```python
import json
from pathlib import Path


YAHOO_POSITIONS = ["C", "1B", "2B", "3B", "SS", "OF", "SP", "RP"]


def fetch_positions_from_yahoo(league) -> dict[str, list[str]]:
    """Fetch player position eligibility from Yahoo Fantasy API.

    Iterates through each position and collects eligible players.
    Pre-draft, all players are free agents, so this captures everyone.

    Args:
        league: yahoo_fantasy_api League object.

    Returns:
        Dict of player_name -> list of eligible positions.
    """
    position_maps = []
    for pos in YAHOO_POSITIONS:
        try:
            agents = league.free_agents(pos)
            pos_map = {}
            for player in agents:
                name = player["name"]
                eligible = player.get("eligible_positions", [pos])
                pos_map[name] = eligible
            position_maps.append(pos_map)
        except Exception:
            continue
    return merge_position_maps(position_maps)


def merge_position_maps(maps: list[dict[str, list[str]]]) -> dict[str, list[str]]:
    """Merge multiple position maps into one, deduplicating positions."""
    merged: dict[str, list[str]] = {}
    for pos_map in maps:
        for name, positions in pos_map.items():
            if name not in merged:
                merged[name] = []
            for pos in positions:
                if pos not in merged[name]:
                    merged[name].append(pos)
    return merged


def save_positions_cache(positions: dict[str, list[str]], path: Path) -> None:
    """Save position data to a JSON cache file."""
    with open(path, "w") as f:
        json.dump(positions, f, indent=2)


def load_positions_cache(path: Path) -> dict[str, list[str]]:
    """Load position data from a JSON cache file."""
    if not path.exists():
        raise FileNotFoundError(f"Position cache not found: {path}")
    with open(path) as f:
        return json.load(f)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_data/test_yahoo_players.py -v`
Expected: All PASS

- [ ] **Step 5: Create the fetch script**

Create `scripts/fetch_positions.py`:
```python
"""Fetch player position eligibility from Yahoo and cache to JSON.

Run this once before draft day:
    python scripts/fetch_positions.py
"""
from pathlib import Path
from fantasy_baseball.auth.yahoo_auth import get_yahoo_session, get_league
from fantasy_baseball.config import load_config
from fantasy_baseball.data.yahoo_players import (
    fetch_positions_from_yahoo,
    save_positions_cache,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = PROJECT_ROOT / "config" / "league.yaml"
CACHE_PATH = PROJECT_ROOT / "data" / "player_positions.json"


def main():
    config = load_config(CONFIG_PATH)
    print(f"Connecting to Yahoo Fantasy (league {config.league_id})...")
    session = get_yahoo_session()
    league = get_league(session, config.league_id, config.game_code)
    print("Fetching player positions (this may take a minute)...")
    positions = fetch_positions_from_yahoo(league)
    save_positions_cache(positions, CACHE_PATH)
    print(f"Cached {len(positions)} players to {CACHE_PATH}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 6: Commit**

```bash
git add src/fantasy_baseball/data/yahoo_players.py tests/test_data/test_yahoo_players.py scripts/fetch_positions.py
git commit -m "feat: add Yahoo position fetcher with JSON caching"
```

---

### Task 2: Draft Board Builder

**Files:**
- Create: `src/fantasy_baseball/draft/__init__.py`
- Create: `src/fantasy_baseball/draft/board.py`
- Create: `tests/test_draft/__init__.py`
- Create: `tests/test_draft/test_board.py`

- [ ] **Step 1: Write the tests**

Create `tests/test_draft/__init__.py` (empty) and `tests/test_draft/test_board.py`:
```python
import pytest
import pandas as pd
from pathlib import Path
from fantasy_baseball.draft.board import build_draft_board, apply_keepers


@pytest.fixture
def position_cache(tmp_path):
    """Create a mock position cache matching our fixture players."""
    import json
    positions = {
        "Aaron Judge": ["OF", "DH"],
        "Mookie Betts": ["OF", "SS"],
        "Adley Rutschman": ["C"],
        "Marcus Semien": ["2B", "SS"],
        "Gerrit Cole": ["SP"],
        "Emmanuel Clase": ["RP"],
        "Corbin Burnes": ["SP"],
    }
    cache_path = tmp_path / "positions.json"
    with open(cache_path, "w") as f:
        json.dump(positions, f)
    return cache_path


class TestBuildDraftBoard:
    def test_returns_dataframe_with_required_columns(self, fixtures_dir, position_cache):
        board = build_draft_board(
            projections_dir=fixtures_dir,
            positions_path=position_cache,
            systems=["steamer"],
        )
        assert "name" in board.columns
        assert "positions" in board.columns
        assert "total_sgp" in board.columns
        assert "var" in board.columns
        assert "best_position" in board.columns

    def test_players_ranked_by_var_descending(self, fixtures_dir, position_cache):
        board = build_draft_board(
            projections_dir=fixtures_dir,
            positions_path=position_cache,
            systems=["steamer"],
        )
        vars_list = board["var"].tolist()
        assert vars_list == sorted(vars_list, reverse=True)

    def test_all_fixture_players_present(self, fixtures_dir, position_cache):
        board = build_draft_board(
            projections_dir=fixtures_dir,
            positions_path=position_cache,
            systems=["steamer"],
        )
        assert len(board) == 7

    def test_positions_from_cache(self, fixtures_dir, position_cache):
        board = build_draft_board(
            projections_dir=fixtures_dir,
            positions_path=position_cache,
            systems=["steamer"],
        )
        judge = board[board["name"] == "Aaron Judge"].iloc[0]
        assert "OF" in judge["positions"]


class TestApplyKeepers:
    def test_removes_keepers_from_board(self, fixtures_dir, position_cache):
        board = build_draft_board(
            projections_dir=fixtures_dir,
            positions_path=position_cache,
            systems=["steamer"],
        )
        keepers = [{"name": "Aaron Judge", "team": "Spacemen"}]
        filtered = apply_keepers(board, keepers)
        assert "Aaron Judge" not in filtered["name"].values
        assert len(filtered) == len(board) - 1

    def test_keeper_not_in_projections_is_ignored(self, fixtures_dir, position_cache):
        board = build_draft_board(
            projections_dir=fixtures_dir,
            positions_path=position_cache,
            systems=["steamer"],
        )
        keepers = [{"name": "Nonexistent Player", "team": "Nobody"}]
        filtered = apply_keepers(board, keepers)
        assert len(filtered) == len(board)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_draft/test_board.py -v`
Expected: FAIL (ImportError)

- [ ] **Step 3: Write the implementation**

Create `src/fantasy_baseball/draft/__init__.py` (empty) and `src/fantasy_baseball/draft/board.py`:
```python
import pandas as pd
from pathlib import Path
from fantasy_baseball.data.projections import blend_projections
from fantasy_baseball.data.yahoo_players import load_positions_cache
from fantasy_baseball.sgp.denominators import get_sgp_denominators
from fantasy_baseball.sgp.player_value import calculate_player_sgp
from fantasy_baseball.sgp.replacement import calculate_replacement_levels
from fantasy_baseball.sgp.var import calculate_var


def build_draft_board(
    projections_dir: Path,
    positions_path: Path,
    systems: list[str],
    weights: dict[str, float] | None = None,
    sgp_overrides: dict[str, float] | None = None,
) -> pd.DataFrame:
    """Build a ranked draft board from projections and position data.

    Pipeline:
    1. Blend projections from multiple systems
    2. Merge with Yahoo position eligibility
    3. Calculate SGP for each player
    4. Calculate replacement levels and VAR
    5. Return sorted by VAR descending

    Args:
        projections_dir: Path to directory with projection CSVs.
        positions_path: Path to cached player_positions.json.
        systems: Projection systems to blend.
        weights: Optional projection weights.
        sgp_overrides: Optional SGP denominator overrides.

    Returns:
        DataFrame sorted by VAR descending.
    """
    # Step 1: Blend projections
    hitters, pitchers = blend_projections(projections_dir, systems, weights)

    # Step 2: Load positions and merge
    positions = load_positions_cache(positions_path)
    hitters = _attach_positions(hitters, positions, default_type="hitter")
    pitchers = _attach_positions(pitchers, positions, default_type="pitcher")

    # Step 3: Calculate SGP
    denoms = get_sgp_denominators(sgp_overrides)
    pool = pd.concat([hitters, pitchers], ignore_index=True)
    pool["total_sgp"] = pool.apply(
        lambda row: calculate_player_sgp(row, denoms=denoms), axis=1
    )

    # Step 4: Replacement levels and VAR
    replacement_levels = calculate_replacement_levels(pool)
    pool["var"] = 0.0
    pool["best_position"] = ""
    for idx, row in pool.iterrows():
        var, pos = calculate_var(row, replacement_levels, return_position=True)
        pool.at[idx, "var"] = var
        pool.at[idx, "best_position"] = pos

    # Step 5: Sort and return
    return pool.sort_values("var", ascending=False).reset_index(drop=True)


def apply_keepers(board: pd.DataFrame, keepers: list[dict]) -> pd.DataFrame:
    """Remove keeper players from the draft board.

    Args:
        board: Draft board DataFrame.
        keepers: List of keeper dicts with 'name' key.

    Returns:
        Filtered DataFrame with keepers removed.
    """
    keeper_names = {k["name"] for k in keepers}
    return board[~board["name"].isin(keeper_names)].reset_index(drop=True)


def _attach_positions(
    df: pd.DataFrame,
    positions: dict[str, list[str]],
    default_type: str,
) -> pd.DataFrame:
    """Attach position eligibility from Yahoo cache to projection data."""
    if df.empty:
        return df
    df = df.copy()
    default_positions = ["OF"] if default_type == "hitter" else ["SP"]
    df["positions"] = df["name"].apply(
        lambda name: positions.get(name, default_positions)
    )
    return df
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_draft/test_board.py -v`
Expected: All PASS

- [ ] **Step 5: Commit**

```bash
git add src/fantasy_baseball/draft/ tests/test_draft/
git commit -m "feat: add draft board builder with SGP/VAR pipeline"
```

---

### Task 3: Fuzzy Player Search

**Files:**
- Create: `src/fantasy_baseball/draft/search.py`
- Create: `tests/test_draft/test_search.py`

- [ ] **Step 1: Write the tests**

Create `tests/test_draft/test_search.py`:
```python
import pytest
from fantasy_baseball.draft.search import find_player


PLAYER_NAMES = [
    "Aaron Judge",
    "Mookie Betts",
    "Adley Rutschman",
    "Marcus Semien",
    "Gerrit Cole",
    "Emmanuel Clase",
    "Corbin Burnes",
    "Juan Soto",
    "Julio Rodriguez",
]


class TestFindPlayer:
    def test_exact_match(self):
        match = find_player("Aaron Judge", PLAYER_NAMES)
        assert match == "Aaron Judge"

    def test_case_insensitive(self):
        match = find_player("aaron judge", PLAYER_NAMES)
        assert match == "Aaron Judge"

    def test_partial_match(self):
        match = find_player("judge", PLAYER_NAMES)
        assert match == "Aaron Judge"

    def test_misspelling(self):
        match = find_player("aron juge", PLAYER_NAMES)
        assert match == "Aaron Judge"

    def test_last_name_only(self):
        match = find_player("rutschman", PLAYER_NAMES)
        assert match == "Adley Rutschman"

    def test_no_match_returns_none(self):
        match = find_player("zzzzzzz", PLAYER_NAMES)
        assert match is None

    def test_close_match_with_threshold(self):
        match = find_player("Corbin Burns", PLAYER_NAMES)
        assert match == "Corbin Burnes"

    def test_find_multiple_candidates(self):
        candidates = find_player("ju", PLAYER_NAMES, return_top_n=3)
        assert isinstance(candidates, list)
        assert len(candidates) <= 3
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_draft/test_search.py -v`
Expected: FAIL (ImportError)

- [ ] **Step 3: Write the implementation**

Create `src/fantasy_baseball/draft/search.py`:
```python
from difflib import SequenceMatcher


def find_player(
    query: str,
    player_names: list[str],
    threshold: float = 0.4,
    return_top_n: int | None = None,
) -> str | list[str] | None:
    """Find the best matching player name using fuzzy search.

    Matches against full name, last name, and substring. Uses
    SequenceMatcher for fuzzy scoring.

    Args:
        query: Search string (can be partial, misspelled, any case).
        player_names: List of canonical player names to search.
        threshold: Minimum similarity score (0-1) to accept a match.
        return_top_n: If set, return top N matches as a list.

    Returns:
        Best matching name, list of top matches, or None if no match.
    """
    query_lower = query.lower().strip()
    scored: list[tuple[float, str]] = []

    for name in player_names:
        name_lower = name.lower()
        # Score full name match
        full_score = SequenceMatcher(None, query_lower, name_lower).ratio()
        # Score against last name only
        last_name = name_lower.split()[-1] if " " in name_lower else name_lower
        last_score = SequenceMatcher(None, query_lower, last_name).ratio()
        # Substring bonus: if query is contained in the name
        substring_bonus = 0.3 if query_lower in name_lower else 0.0
        best = max(full_score, last_score) + substring_bonus
        scored.append((best, name))

    scored.sort(key=lambda x: x[0], reverse=True)

    if return_top_n is not None:
        return [name for score, name in scored[:return_top_n] if score >= threshold]

    if scored and scored[0][0] >= threshold:
        return scored[0][1]
    return None
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_draft/test_search.py -v`
Expected: All PASS

- [ ] **Step 5: Commit**

```bash
git add src/fantasy_baseball/draft/search.py tests/test_draft/test_search.py
git commit -m "feat: add fuzzy player name search for draft input"
```

---

## Chunk 2: Draft Engine + CLI

### Task 4: Snake Draft Tracker

**Files:**
- Create: `src/fantasy_baseball/draft/tracker.py`
- Create: `tests/test_draft/test_tracker.py`

- [ ] **Step 1: Write the tests**

Create `tests/test_draft/test_tracker.py`:
```python
import pytest
from fantasy_baseball.draft.tracker import DraftTracker


class TestDraftTracker:
    def make_tracker(self):
        return DraftTracker(num_teams=10, user_position=8)

    def test_initial_state(self):
        t = self.make_tracker()
        assert t.current_pick == 1
        assert t.current_round == 1
        assert t.picking_team == 1

    def test_round_1_order(self):
        t = self.make_tracker()
        # Round 1 is picks 1-10, teams 1-10
        teams = [t.picking_team]
        for _ in range(9):
            t.advance()
            teams.append(t.picking_team)
        assert teams == [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]

    def test_round_2_reverses(self):
        t = self.make_tracker()
        # Advance through round 1
        for _ in range(10):
            t.advance()
        # Round 2 pick 1 should be team 10
        assert t.current_round == 2
        assert t.picking_team == 10

    def test_snake_pattern(self):
        t = self.make_tracker()
        # Advance to pick 11 (round 2, team 10)
        for _ in range(10):
            t.advance()
        assert t.picking_team == 10
        # Pick 20 should be team 1
        for _ in range(9):
            t.advance()
        assert t.picking_team == 1

    def test_is_user_pick(self):
        t = self.make_tracker()
        # Advance to pick 8 (user's first pick)
        for _ in range(7):
            t.advance()
        assert t.is_user_pick is True

    def test_is_not_user_pick(self):
        t = self.make_tracker()
        assert t.is_user_pick is False  # Pick 1 is team 1

    def test_picks_until_next_user_turn(self):
        t = self.make_tracker()
        # At pick 1, user picks at 8: 7 picks away
        assert t.picks_until_user_turn == 7

    def test_picks_until_next_after_user_picks(self):
        t = self.make_tracker()
        # Advance to pick 8 (user's pick)
        for _ in range(7):
            t.advance()
        assert t.is_user_pick is True
        # After user picks, advance to pick 9
        t.advance()
        # Next user pick is 13 (round 2, position 3 from end = 10-8+1=3)
        # picks 9,10,11,12,13 -> 4 picks away
        assert t.picks_until_user_turn == 4

    def test_user_roster_tracking(self):
        t = self.make_tracker()
        t.draft_player("Juan Soto", is_user=True)
        assert "Juan Soto" in t.user_roster
        assert "Juan Soto" in t.drafted_players

    def test_other_team_draft(self):
        t = self.make_tracker()
        t.draft_player("Random Guy", is_user=False)
        assert "Random Guy" not in t.user_roster
        assert "Random Guy" in t.drafted_players

    def test_total_picks(self):
        t = DraftTracker(num_teams=10, user_position=8, rounds=22)
        assert t.total_picks == 220
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_draft/test_tracker.py -v`
Expected: FAIL (ImportError)

- [ ] **Step 3: Write the implementation**

Create `src/fantasy_baseball/draft/tracker.py`:
```python
class DraftTracker:
    """Track state of a snake draft.

    In a snake draft with N teams, odd rounds go 1→N,
    even rounds go N→1.
    """

    def __init__(self, num_teams: int, user_position: int, rounds: int = 22):
        self.num_teams = num_teams
        self.user_position = user_position  # 1-indexed
        self.rounds = rounds
        self.current_pick = 1  # Global pick number (1-indexed)
        self.drafted_players: list[str] = []
        self.user_roster: list[str] = []

    @property
    def total_picks(self) -> int:
        return self.num_teams * self.rounds

    @property
    def current_round(self) -> int:
        return (self.current_pick - 1) // self.num_teams + 1

    @property
    def pick_in_round(self) -> int:
        """1-indexed position within the current round."""
        return (self.current_pick - 1) % self.num_teams + 1

    @property
    def picking_team(self) -> int:
        """Which team (1-indexed) is currently picking."""
        pos = self.pick_in_round
        if self.current_round % 2 == 1:
            return pos  # Odd rounds: 1→N
        else:
            return self.num_teams - pos + 1  # Even rounds: N→1

    @property
    def is_user_pick(self) -> bool:
        return self.picking_team == self.user_position

    @property
    def picks_until_user_turn(self) -> int:
        """Number of picks until the user's next turn (0 if it's their turn)."""
        if self.is_user_pick:
            return 0
        # Simulate forward to find next user pick
        save_pick = self.current_pick
        count = 0
        temp_pick = self.current_pick
        while temp_pick <= self.total_picks:
            temp_pick += 1
            count += 1
            temp_round = (temp_pick - 1) // self.num_teams + 1
            temp_pos = (temp_pick - 1) % self.num_teams + 1
            if temp_round % 2 == 1:
                team = temp_pos
            else:
                team = self.num_teams - temp_pos + 1
            if team == self.user_position:
                return count
        return count  # End of draft

    def advance(self) -> None:
        """Move to the next pick."""
        self.current_pick += 1

    def draft_player(self, name: str, is_user: bool = False) -> None:
        """Record a drafted player."""
        self.drafted_players.append(name)
        if is_user:
            self.user_roster.append(name)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_draft/test_tracker.py -v`
Expected: All PASS

- [ ] **Step 5: Commit**

```bash
git add src/fantasy_baseball/draft/tracker.py tests/test_draft/test_tracker.py
git commit -m "feat: add snake draft state tracker"
```

---

### Task 5: Category Balance Tracker

**Files:**
- Create: `src/fantasy_baseball/draft/balance.py`
- Create: `tests/test_draft/test_balance.py`

- [ ] **Step 1: Write the tests**

Create `tests/test_draft/test_balance.py`:
```python
import pytest
import pandas as pd
from fantasy_baseball.draft.balance import CategoryBalance


def _make_hitter(name, r, hr, rbi, sb, avg, ab):
    return pd.Series({
        "name": name, "player_type": "hitter",
        "r": r, "hr": hr, "rbi": rbi, "sb": sb, "avg": avg, "ab": ab, "h": int(avg * ab),
    })


def _make_pitcher(name, w, k, sv, era, whip, ip):
    return pd.Series({
        "name": name, "player_type": "pitcher",
        "w": w, "k": k, "sv": sv, "era": era, "whip": whip, "ip": ip,
        "er": era * ip / 9, "bb": int(whip * ip * 0.3), "h_allowed": int(whip * ip * 0.7),
    })


class TestCategoryBalance:
    def test_empty_roster(self):
        bal = CategoryBalance()
        totals = bal.get_totals()
        assert totals["HR"] == 0
        assert totals["K"] == 0

    def test_add_hitter(self):
        bal = CategoryBalance()
        bal.add_player(_make_hitter("Judge", 110, 45, 120, 5, .291, 550))
        totals = bal.get_totals()
        assert totals["HR"] == 45
        assert totals["R"] == 110
        assert totals["RBI"] == 120
        assert totals["SB"] == 5

    def test_add_multiple_hitters_sums(self):
        bal = CategoryBalance()
        bal.add_player(_make_hitter("Judge", 110, 45, 120, 5, .291, 550))
        bal.add_player(_make_hitter("Betts", 105, 28, 85, 15, .287, 540))
        totals = bal.get_totals()
        assert totals["HR"] == 73
        assert totals["SB"] == 20

    def test_avg_is_weighted(self):
        bal = CategoryBalance()
        bal.add_player(_make_hitter("Judge", 110, 45, 120, 5, .291, 550))
        bal.add_player(_make_hitter("Betts", 105, 28, 85, 15, .287, 540))
        totals = bal.get_totals()
        # Weighted avg = (550*.291 + 540*.287) / (550+540)
        expected = (550 * .291 + 540 * .287) / (550 + 540)
        assert totals["AVG"] == pytest.approx(expected, abs=0.001)

    def test_add_pitcher(self):
        bal = CategoryBalance()
        bal.add_player(_make_pitcher("Cole", 15, 240, 0, 3.15, 1.05, 200))
        totals = bal.get_totals()
        assert totals["W"] == 15
        assert totals["K"] == 240
        assert totals["SV"] == 0

    def test_era_whip_weighted_by_ip(self):
        bal = CategoryBalance()
        bal.add_player(_make_pitcher("Cole", 15, 240, 0, 3.00, 1.00, 200))
        bal.add_player(_make_pitcher("Clase", 4, 70, 40, 2.00, 0.90, 70))
        totals = bal.get_totals()
        # ERA = total_ER * 9 / total_IP
        total_er = 3.00 * 200 / 9 + 2.00 * 70 / 9
        expected_era = total_er * 9 / 270
        assert totals["ERA"] == pytest.approx(expected_era, abs=0.01)

    def test_get_warnings_flags_weak_categories(self):
        bal = CategoryBalance()
        # Add 5 low-power, no-speed hitters to pass min threshold
        for i in range(5):
            bal.add_player(_make_hitter(f"Slappy{i}", 40, 2, 30, 0, .260, 400))
        warnings = bal.get_warnings()
        # Should flag SB as weak (0 steals)
        assert any("SB" in w for w in warnings)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_draft/test_balance.py -v`
Expected: FAIL (ImportError)

- [ ] **Step 3: Write the implementation**

Create `src/fantasy_baseball/draft/balance.py`:
```python
import pandas as pd
from fantasy_baseball.utils.constants import HITTING_CATEGORIES, PITCHING_CATEGORIES

# Rough per-team targets for a competitive 10-team 5x5 roto roster
# These represent what a "middle of the pack" team might accumulate
TEAM_TARGETS: dict[str, float] = {
    "R": 850, "HR": 220, "RBI": 830, "SB": 100, "AVG": 0.265,
    "W": 75, "K": 1200, "ERA": 3.80, "WHIP": 1.20, "SV": 80,
}

# What fraction of the target triggers a warning
WARNING_THRESHOLD: float = 0.6


class CategoryBalance:
    """Track projected stat totals for a fantasy roster under construction."""

    def __init__(self):
        self._hitters: list[pd.Series] = []
        self._pitchers: list[pd.Series] = []

    def add_player(self, player: pd.Series) -> None:
        """Add a drafted player's projections to the balance tracker."""
        if player.get("player_type") == "hitter":
            self._hitters.append(player)
        elif player.get("player_type") == "pitcher":
            self._pitchers.append(player)

    def get_totals(self) -> dict[str, float]:
        """Calculate projected totals across all rostered players."""
        totals: dict[str, float] = {}

        # Hitting counting stats
        for stat, col in [("R", "r"), ("HR", "hr"), ("RBI", "rbi"), ("SB", "sb")]:
            totals[stat] = sum(h.get(col, 0) for h in self._hitters)

        # AVG: weighted by AB
        total_h = sum(h.get("h", 0) for h in self._hitters)
        total_ab = sum(h.get("ab", 0) for h in self._hitters)
        totals["AVG"] = total_h / total_ab if total_ab > 0 else 0.0

        # Pitching counting stats
        for stat, col in [("W", "w"), ("K", "k"), ("SV", "sv")]:
            totals[stat] = sum(p.get(col, 0) for p in self._pitchers)

        # ERA and WHIP: weighted by IP
        total_ip = sum(p.get("ip", 0) for p in self._pitchers)
        if total_ip > 0:
            total_er = sum(p.get("er", 0) for p in self._pitchers)
            totals["ERA"] = total_er * 9 / total_ip
            total_bb = sum(p.get("bb", 0) for p in self._pitchers)
            total_ha = sum(p.get("h_allowed", 0) for p in self._pitchers)
            totals["WHIP"] = (total_bb + total_ha) / total_ip
        else:
            totals["ERA"] = 0.0
            totals["WHIP"] = 0.0

        return totals

    def get_warnings(self) -> list[str]:
        """Return warnings for categories significantly below targets."""
        totals = self.get_totals()
        warnings = []
        num_hitters = len(self._hitters)
        num_pitchers = len(self._pitchers)

        if num_hitters == 0 and num_pitchers == 0:
            return []

        # Only warn after enough players are drafted to be meaningful
        min_hitters = 5
        min_pitchers = 3

        for cat in HITTING_CATEGORIES:
            if cat == "AVG":
                continue  # Rate stat — skip for counting-style warnings
            target = TEAM_TARGETS[cat]
            if num_hitters >= min_hitters and totals[cat] < target * WARNING_THRESHOLD:
                warnings.append(f"{cat} is low ({totals[cat]:.0f}, target ~{target:.0f})")

        for cat in PITCHING_CATEGORIES:
            if cat in ("ERA", "WHIP"):
                continue  # Rate stats — skip
            target = TEAM_TARGETS[cat]
            if num_pitchers >= min_pitchers and totals[cat] < target * WARNING_THRESHOLD:
                warnings.append(f"{cat} is low ({totals[cat]:.0f}, target ~{target:.0f})")

        return warnings
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_draft/test_balance.py -v`
Expected: All PASS

- [ ] **Step 5: Commit**

```bash
git add src/fantasy_baseball/draft/balance.py tests/test_draft/test_balance.py
git commit -m "feat: add category balance tracker with team targets and warnings"
```

---

### Task 6: Pick Recommender

**Files:**
- Create: `src/fantasy_baseball/draft/recommender.py`
- Create: `tests/test_draft/test_recommender.py`

- [ ] **Step 1: Write the tests**

Create `tests/test_draft/test_recommender.py`:
```python
import pytest
import pandas as pd
from fantasy_baseball.draft.recommender import get_recommendations


def _make_board():
    """Create a small draft board for testing."""
    players = [
        {"name": "Player A", "var": 15.0, "best_position": "OF", "positions": ["OF"],
         "player_type": "hitter", "r": 100, "hr": 35, "rbi": 100, "sb": 20, "avg": .280, "ab": 550, "h": 154},
        {"name": "Player B", "var": 14.0, "best_position": "SS", "positions": ["SS"],
         "player_type": "hitter", "r": 90, "hr": 25, "rbi": 80, "sb": 30, "avg": .275, "ab": 530, "h": 146},
        {"name": "Player C", "var": 13.0, "best_position": "C", "positions": ["C"],
         "player_type": "hitter", "r": 70, "hr": 22, "rbi": 75, "sb": 2, "avg": .260, "ab": 480, "h": 125},
        {"name": "Player D", "var": 12.0, "best_position": "P", "positions": ["SP"],
         "player_type": "pitcher", "w": 14, "k": 210, "sv": 0, "era": 3.20, "whip": 1.10, "ip": 195,
         "er": 69, "bb": 50, "h_allowed": 165},
        {"name": "Player E", "var": 11.0, "best_position": "OF", "positions": ["OF"],
         "player_type": "hitter", "r": 80, "hr": 20, "rbi": 70, "sb": 15, "avg": .270, "ab": 500, "h": 135},
    ]
    return pd.DataFrame(players)


class TestGetRecommendations:
    def test_returns_top_n(self):
        board = _make_board()
        recs = get_recommendations(board, drafted=[], user_roster=[], n=3)
        assert len(recs) == 3

    def test_excludes_drafted_players(self):
        board = _make_board()
        recs = get_recommendations(board, drafted=["Player A"], user_roster=[], n=3)
        names = [r["name"] for r in recs]
        assert "Player A" not in names

    def test_recommendations_sorted_by_var(self):
        board = _make_board()
        recs = get_recommendations(board, drafted=[], user_roster=[], n=5)
        vars_list = [r["var"] for r in recs]
        assert vars_list == sorted(vars_list, reverse=True)

    def test_flags_positional_need(self):
        board = _make_board()
        # User has no catcher — catcher should get flagged
        filled = {"OF": 1}
        recs = get_recommendations(board, drafted=[], user_roster=[], n=5, filled_positions=filled)
        catcher_rec = next(r for r in recs if r["name"] == "Player C")
        assert catcher_rec.get("need_flag") is True

    def test_includes_player_stats(self):
        board = _make_board()
        recs = get_recommendations(board, drafted=[], user_roster=[], n=1)
        assert "var" in recs[0]
        assert "best_position" in recs[0]
        assert "name" in recs[0]
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_draft/test_recommender.py -v`
Expected: FAIL (ImportError)

- [ ] **Step 3: Write the implementation**

Create `src/fantasy_baseball/draft/recommender.py`:
```python
import pandas as pd
from fantasy_baseball.utils.constants import ROSTER_SLOTS
from fantasy_baseball.utils.positions import can_fill_slot

# Positions that must be filled (excluding BN, IL, and flex slots)
REQUIRED_POSITIONS = ["C", "1B", "2B", "3B", "SS", "OF", "P"]


def get_recommendations(
    board: pd.DataFrame,
    drafted: list[str],
    user_roster: list[str],
    n: int = 5,
    filled_positions: dict[str, int] | None = None,
    picks_until_next: int | None = None,
) -> list[dict]:
    """Get top draft pick recommendations.

    Combines VAR rankings with positional need flags.

    Args:
        board: Full draft board (sorted by VAR).
        drafted: Names of all drafted players (keepers + picks).
        user_roster: Names of players on user's team.
        n: Number of recommendations to return.
        filled_positions: Dict of position -> count of user's filled slots.
        picks_until_next: Picks until user's next turn (for scarcity notes).

    Returns:
        List of recommendation dicts with name, var, position, need_flag, note.
    """
    available = board[~board["name"].isin(drafted)].head(n * 3)

    if filled_positions is None:
        filled_positions = {}

    unfilled = _get_unfilled_positions(filled_positions)

    recs = []
    for _, player in available.iterrows():
        rec = {
            "name": player["name"],
            "var": player["var"],
            "best_position": player["best_position"],
            "positions": player["positions"],
            "player_type": player["player_type"],
            "need_flag": False,
            "note": "",
        }

        # Check if this player fills an unfilled position
        for pos in player["positions"]:
            slot_pos = "P" if pos in ("SP", "RP") else pos
            if slot_pos in unfilled:
                rec["need_flag"] = True
                rec["note"] = f"fills {slot_pos} need"
                break

        # Scarcity note: if few players at this position remain
        if picks_until_next and picks_until_next > 8:
            pos = player["best_position"]
            remaining_at_pos = len(available[available["best_position"] == pos])
            if remaining_at_pos <= 3:
                rec["note"] = f"scarce position — only {remaining_at_pos} left in top tier"

        recs.append(rec)

    recs.sort(key=lambda r: r["var"], reverse=True)
    return recs[:n]


def _get_unfilled_positions(filled: dict[str, int]) -> set[str]:
    """Determine which roster positions still need to be filled."""
    unfilled = set()
    for pos, slots in ROSTER_SLOTS.items():
        if pos in ("BN", "IL", "UTIL", "IF"):
            continue
        current = filled.get(pos, 0)
        if current < slots:
            unfilled.add(pos)
    return unfilled


def get_filled_positions(
    user_roster_names: list[str], board: pd.DataFrame
) -> dict[str, int]:
    """Count how many of each position the user has filled."""
    filled: dict[str, int] = {}
    for name in user_roster_names:
        rows = board[board["name"] == name]
        if rows.empty:
            continue
        player = rows.iloc[0]
        pos = player["best_position"]
        filled[pos] = filled.get(pos, 0) + 1
    return filled
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_draft/test_recommender.py -v`
Expected: All PASS

- [ ] **Step 5: Commit**

```bash
git add src/fantasy_baseball/draft/recommender.py tests/test_draft/test_recommender.py
git commit -m "feat: add pick recommender with positional need and scarcity flags"
```

---

### Task 7: Interactive Draft CLI

**Files:**
- Create: `scripts/run_draft.py`

This is the CLI entry point that ties everything together. It's primarily I/O and display logic, so it's tested via a manual run rather than unit tests.

- [ ] **Step 1: Write the CLI script**

Create `scripts/run_draft.py`:
```python
"""Interactive Draft Assistant for Fantasy Baseball.

Usage:
    python scripts/run_draft.py

Pre-requisites:
    1. FanGraphs projection CSVs in data/projections/
    2. Run: python scripts/fetch_positions.py
    3. config/league.yaml with keepers and settings
"""
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from fantasy_baseball.config import load_config
from fantasy_baseball.draft.board import build_draft_board, apply_keepers
from fantasy_baseball.draft.tracker import DraftTracker
from fantasy_baseball.draft.balance import CategoryBalance
from fantasy_baseball.draft.search import find_player
from fantasy_baseball.draft.recommender import (
    get_recommendations,
    get_filled_positions,
)

CONFIG_PATH = PROJECT_ROOT / "config" / "league.yaml"
POSITIONS_PATH = PROJECT_ROOT / "data" / "player_positions.json"
PROJECTIONS_DIR = PROJECT_ROOT / "data" / "projections"


def main():
    # Load config
    config = load_config(CONFIG_PATH)
    print(f"League {config.league_id} | Draft position: {config.draft_position}")
    print(f"Keepers: {len(config.keepers)} players across {config.num_teams} teams")
    print()

    # Build draft board (keep full board for keeper lookups)
    print("Building draft board...")
    full_board = build_draft_board(
        projections_dir=PROJECTIONS_DIR,
        positions_path=POSITIONS_PATH,
        systems=config.projection_systems,
        weights=config.projection_weights or None,
        sgp_overrides=config.sgp_overrides or None,
    )
    board = apply_keepers(full_board, config.keepers)
    print(f"Draft pool: {len(board)} players (after removing {len(config.keepers)} keepers)")
    print()

    # Initialize tracker and balance
    user_keepers = [k for k in config.keepers if k.get("team") == config.team_name]
    rounds = sum(config.roster_slots.values()) - len(user_keepers)
    tracker = DraftTracker(
        num_teams=config.num_teams,
        user_position=config.draft_position,
        rounds=rounds,
    )
    balance = CategoryBalance()

    # Add keeper projections to balance and mark all keepers as drafted
    for keeper in config.keepers:
        is_user = keeper.get("team") == config.team_name
        if is_user:
            rows = full_board[full_board["name"] == keeper["name"]]
            if not rows.empty:
                balance.add_player(rows.iloc[0])
        tracker.draft_player(keeper["name"], is_user=is_user)

    # Show pre-draft rankings
    print("=" * 70)
    print("TOP 25 AVAILABLE PLAYERS")
    print("=" * 70)
    _show_top_players(board, tracker.drafted_players, 25)
    print()

    # Main draft loop
    while tracker.current_pick <= tracker.total_picks:
        print("=" * 70)
        print(f"ROUND {tracker.current_round} | Pick {tracker.current_pick} "
              f"| Team {tracker.picking_team}", end="")
        if tracker.is_user_pick:
            print(" *** YOUR PICK ***")
        else:
            print()
        print("=" * 70)

        if tracker.is_user_pick:
            _handle_user_pick(board, tracker, balance)
        else:
            _handle_other_pick(board, tracker)

        # Show updated top 10
        print()
        _show_top_players(board, tracker.drafted_players, 10)
        print()

        tracker.advance()

    print("\nDraft complete!")
    print("\nYour roster:")
    for name in tracker.user_roster:
        print(f"  {name}")


def _handle_user_pick(board, tracker, balance):
    """Handle the user's draft pick with recommendations."""
    filled = get_filled_positions(tracker.user_roster, board)
    # Calculate gap to NEXT user turn after this one
    # Simulate: advance, find next, then use that gap
    save = tracker.current_pick
    tracker.current_pick += 1
    picks_gap = tracker.picks_until_user_turn + 1
    tracker.current_pick = save

    recs = get_recommendations(
        board,
        drafted=tracker.drafted_players,
        user_roster=tracker.user_roster,
        n=5,
        filled_positions=filled,
        picks_until_next=picks_gap,
    )

    # Show recommendations
    print(f"\nPicks until next turn: {picks_gap}")
    print("\nRECOMMENDATIONS:")
    for i, rec in enumerate(recs, 1):
        flag = " [NEED]" if rec["need_flag"] else ""
        note = f" ({rec['note']})" if rec["note"] else ""
        print(f"  {i}. {rec['name']} ({rec['best_position']}) "
              f"VAR: {rec['var']:.1f}{flag}{note}")

    # Show category balance
    totals = balance.get_totals()
    warnings = balance.get_warnings()
    print(f"\nROSTER BALANCE:")
    print(f"  R:{totals['R']:.0f} HR:{totals['HR']:.0f} RBI:{totals['RBI']:.0f} "
          f"SB:{totals['SB']:.0f} AVG:{totals['AVG']:.3f}")
    print(f"  W:{totals['W']:.0f} K:{totals['K']:.0f} SV:{totals['SV']:.0f} "
          f"ERA:{totals['ERA']:.2f} WHIP:{totals['WHIP']:.3f}")
    if warnings:
        print(f"  ⚠ {', '.join(warnings)}")

    # Get user input
    name = _get_player_input(board, tracker)
    if name:
        tracker.draft_player(name, is_user=True)
        rows = board[board["name"] == name]
        if not rows.empty:
            balance.add_player(rows.iloc[0])
        print(f"  → Drafted: {name}")


def _handle_other_pick(board, tracker):
    """Handle another team's pick."""
    name = _get_player_input(board, tracker)
    if name:
        tracker.draft_player(name, is_user=False)
        print(f"  → Drafted: {name}")


def _get_player_input(board, tracker):
    """Get and fuzzy-match a player name from user input."""
    available_names = board[~board["name"].isin(tracker.drafted_players)]["name"].tolist()
    while True:
        raw = input("\nEnter player name (or 'skip' to skip): ").strip()
        if raw.lower() == "skip":
            return None
        if raw.lower() == "quit":
            sys.exit(0)

        # Try number selection (for recommendations)
        if raw.isdigit():
            idx = int(raw) - 1
            filled = get_filled_positions(tracker.user_roster, board)
            recs = get_recommendations(board, tracker.drafted_players, tracker.user_roster,
                                       n=5, filled_positions=filled)
            if 0 <= idx < len(recs):
                return recs[idx]["name"]

        # Fuzzy search
        match = find_player(raw, available_names)
        if match:
            confirm = input(f"  → {match}? (y/n): ").strip().lower()
            if confirm in ("y", "yes", ""):
                return match
            # Show alternatives
            alts = find_player(raw, available_names, return_top_n=5)
            if alts:
                print("  Alternatives:")
                for i, alt in enumerate(alts, 1):
                    print(f"    {i}. {alt}")
                choice = input("  Pick # (or type again): ").strip()
                if choice.isdigit() and 1 <= int(choice) <= len(alts):
                    return alts[int(choice) - 1]
        else:
            print("  No match found. Try again.")


def _show_top_players(board, drafted, n):
    """Display top N available players."""
    available = board[~board["name"].isin(drafted)]
    for i, (_, p) in enumerate(available.head(n).iterrows(), 1):
        pos_str = "/".join(p["positions"][:3]) if isinstance(p["positions"], list) else p["best_position"]
        print(f"  {i:>3}. {p['name']:<25} {pos_str:<12} VAR: {p['var']:>6.1f}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Verify all imports work (smoke test)**

Run: `python -c "from fantasy_baseball.draft.board import build_draft_board; from fantasy_baseball.draft.tracker import DraftTracker; from fantasy_baseball.draft.balance import CategoryBalance; from fantasy_baseball.draft.recommender import get_recommendations; from fantasy_baseball.draft.search import find_player; print('All imports OK')"`
Expected: "All imports OK"

- [ ] **Step 3: Run full test suite**

Run: `pytest -v --tb=short`
Expected: All tests PASS

- [ ] **Step 4: Commit**

```bash
git add scripts/run_draft.py
git commit -m "feat: add interactive draft assistant CLI"
```

---

## Summary

After completing all 7 tasks, the draft assistant provides:

1. **`data/yahoo_players.py`** — Fetch and cache Yahoo position eligibility
2. **`draft/board.py`** — Build ranked draft board (projections → SGP → VAR)
3. **`draft/search.py`** — Fuzzy player name matching
4. **`draft/tracker.py`** — Snake draft state tracking
5. **`draft/balance.py`** — Category balance monitoring with warnings
6. **`draft/recommender.py`** — Pick recommendations with positional need flags
7. **`scripts/run_draft.py`** — Interactive CLI tying it all together

**Draft day workflow:**
1. Download FanGraphs CSVs → `data/projections/steamer_hitters.csv`, etc.
2. `python scripts/fetch_positions.py` (caches Yahoo positions)
3. `python scripts/run_draft.py` (launches assistant)
