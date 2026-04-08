# Interactive Trade Finder Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the batch trade recommendation system with an interactive, player-driven trade search that supports "Trade Away" and "Trade For" modes.

**Architecture:** Two new search functions in `evaluate.py` replace `find_trades()`. A new POST API endpoint calls them on demand using cached refresh data. The frontend replaces the Trade Recommendations section with a search input + mode toggle. Opponent rosters and leverage weights get cached during refresh so the search endpoint can load them without re-fetching from Yahoo.

**Tech Stack:** Python (Flask, dataclasses), vanilla JS, existing wSGP/leverage/ranking infrastructure.

---

### Task 1: Cache opponent rosters and leverage during refresh

Opponent rosters (as Player objects) and per-team leverage weights are computed during `run_full_refresh()` but never persisted. The on-demand search endpoint needs them. Add two new cache entries.

**Files:**
- Modify: `src/fantasy_baseball/web/season_data.py:79-92` (CACHE_FILES dict)
- Modify: `src/fantasy_baseball/web/season_data.py:1109` (after leverage computation)
- Modify: `src/fantasy_baseball/web/season_data.py:849` (after opponent roster fetch)

- [ ] **Step 1: Add cache keys to CACHE_FILES**

In `season_data.py`, add two entries to the `CACHE_FILES` dict at line 91 (before the closing brace):

```python
    "roster_audit": "roster_audit.json",
    "opp_rosters": "opp_rosters.json",
    "leverage": "leverage.json",
```

- [ ] **Step 2: Cache opponent rosters after fetch**

After line 849 (`_progress(f"Fetched {len(opp_rosters)} opponent rosters")`), add:

```python
        # Cache opponent rosters for on-demand trade search
        opp_rosters_flat = {
            tname: [p.to_dict() for p in roster]
            for tname, roster in opp_rosters.items()
        }
        write_cache("opp_rosters", opp_rosters_flat, cache_dir)
```

- [ ] **Step 3: Cache leverage weights after computation**

After line 1109 (the end of the leverage computation loop), add:

```python
        write_cache("leverage", leverage_by_team, cache_dir)
```

- [ ] **Step 4: Run tests to verify no regressions**

Run: `pytest tests/ -v -x --timeout=30 -q`
Expected: All existing tests pass (no test changes needed — this is additive).

- [ ] **Step 5: Commit**

```bash
git add src/fantasy_baseball/web/season_data.py
git commit -m "feat(trades): cache opponent rosters and leverage for on-demand search"
```

---

### Task 2: Write `search_trades_away` with tests

The core "Trade Away" search: given a player on the user's roster, find trade candidates across all opponents, grouped by opponent and sorted by positional weakness.

**Files:**
- Modify: `src/fantasy_baseball/trades/evaluate.py`
- Create: `tests/test_trades/test_trade_search.py`

- [ ] **Step 1: Write failing tests for `search_trades_away`**

Create `tests/test_trades/test_trade_search.py`:

```python
import pytest
from fantasy_baseball.models.player import Player, HitterStats, PitcherStats
from fantasy_baseball.sgp.rankings import rank_key
from fantasy_baseball.trades.evaluate import search_trades_away

ALL_CATS = ["R", "HR", "RBI", "SB", "AVG", "W", "K", "SV", "ERA", "WHIP"]
_EQUAL_LEVERAGE = {cat: 0.1 for cat in ALL_CATS}

ROSTER_SLOTS = {"C": 1, "1B": 1, "2B": 1, "3B": 1, "SS": 1, "IF": 1,
                "OF": 4, "UTIL": 2, "P": 9, "BN": 2, "IL": 2}

SAMPLE_STANDINGS = [
    {"name": "Hart", "team_key": "t.1", "rank": 3,
     "stats": {"R": 900, "HR": 280, "RBI": 880, "SB": 120,
               "AVG": .260, "W": 80, "K": 1300, "SV": 80, "ERA": 3.50, "WHIP": 1.15}},
    {"name": "Rival", "team_key": "t.2", "rank": 5,
     "stats": {"R": 850, "HR": 250, "RBI": 870, "SB": 180,
               "AVG": .255, "W": 85, "K": 1400, "SV": 40, "ERA": 3.80, "WHIP": 1.20}},
    {"name": "Rival A", "team_key": "t.3", "rank": 4,
     "stats": {"R": 870, "HR": 260, "RBI": 860, "SB": 140,
               "AVG": .258, "W": 82, "K": 1350, "SV": 50, "ERA": 3.60, "WHIP": 1.18}},
]


def _make_hitter(name, positions, r=70, hr=20, rbi=65, sb=8, avg=.270, ab=500):
    h = int(avg * ab)
    return Player(name=name, player_type="hitter", positions=positions,
                  ros=HitterStats(pa=int(ab * 1.15), ab=ab, h=h,
                                  r=r, hr=hr, rbi=rbi, sb=sb, avg=avg))


def _make_pitcher(name, positions, ip=150, w=9, k=140, sv=0, era=3.80, whip=1.25):
    er = int(era * ip / 9)
    bb = int((whip * ip - ip * 0.8) / 1)
    h_allowed = int(whip * ip - bb)
    return Player(name=name, player_type="pitcher", positions=positions,
                  ros=PitcherStats(ip=ip, w=w, k=k, sv=sv, era=era, whip=whip,
                                   er=er, bb=bb, h_allowed=h_allowed))


class TestSearchTradesAway:
    def test_returns_grouped_by_opponent(self):
        """Results should be a list of opponent groups with 'opponent' and 'candidates' keys."""
        hart_roster = [_make_hitter("Hart OF", ["OF"], hr=15, sb=5)]
        opp_rosters = {
            "Rival": [_make_hitter("Opp OF", ["OF"], hr=25, sb=15)],
            "Rival A": [_make_hitter("Opp A OF", ["OF"], hr=22, sb=12)],
        }
        rankings = {
            rank_key("Hart OF", "hitter"): 55,
            rank_key("Opp OF", "hitter"): 50,
            rank_key("Opp A OF", "hitter"): 52,
        }
        results = search_trades_away(
            player_name="Hart OF",
            hart_name="Hart", hart_roster=hart_roster, opp_rosters=opp_rosters,
            standings=SAMPLE_STANDINGS,
            leverage_by_team={"Hart": _EQUAL_LEVERAGE, "Rival": _EQUAL_LEVERAGE, "Rival A": _EQUAL_LEVERAGE},
            roster_slots=ROSTER_SLOTS, rankings=rankings,
        )
        assert isinstance(results, list)
        for group in results:
            assert "opponent" in group
            assert "candidates" in group
            assert isinstance(group["candidates"], list)

    def test_player_not_found_returns_empty(self):
        """Searching for a player not on the roster should return empty list."""
        hart_roster = [_make_hitter("Hart OF", ["OF"])]
        results = search_trades_away(
            player_name="Nonexistent Player",
            hart_name="Hart", hart_roster=hart_roster, opp_rosters={},
            standings=SAMPLE_STANDINGS,
            leverage_by_team={"Hart": _EQUAL_LEVERAGE},
            roster_slots=ROSTER_SLOTS, rankings={},
        )
        assert results == []

    def test_candidates_have_required_fields(self):
        """Each candidate should include send, receive, ranks, wSGP gain, and deltas."""
        hart_roster = [_make_hitter("Hart OF", ["OF"], hr=15, sb=5)]
        opp_rosters = {"Rival": [_make_hitter("Opp OF", ["OF"], hr=25, sb=15)]}
        rankings = {
            rank_key("Hart OF", "hitter"): 55,
            rank_key("Opp OF", "hitter"): 50,
        }
        results = search_trades_away(
            player_name="Hart OF",
            hart_name="Hart", hart_roster=hart_roster, opp_rosters=opp_rosters,
            standings=SAMPLE_STANDINGS,
            leverage_by_team={"Hart": _EQUAL_LEVERAGE, "Rival": _EQUAL_LEVERAGE},
            roster_slots=ROSTER_SLOTS, rankings=rankings,
        )
        assert len(results) > 0
        candidate = results[0]["candidates"][0]
        for key in ("send", "receive", "send_rank", "receive_rank",
                    "send_positions", "receive_positions",
                    "hart_wsgp_gain", "hart_delta", "opp_delta",
                    "hart_cat_deltas", "opp_cat_deltas"):
            assert key in candidate, f"Missing key: {key}"

    def test_rank_filter_applied(self):
        """Trades where send_rank - receive_rank > 5 should be excluded."""
        hart_roster = [_make_hitter("Hart OF", ["OF"], hr=15, sb=5)]
        opp_rosters = {"Rival": [_make_hitter("Opp OF", ["OF"], hr=25, sb=15)]}
        rankings = {
            rank_key("Hart OF", "hitter"): 60,
            rank_key("Opp OF", "hitter"): 50,
        }
        results = search_trades_away(
            player_name="Hart OF",
            hart_name="Hart", hart_roster=hart_roster, opp_rosters=opp_rosters,
            standings=SAMPLE_STANDINGS,
            leverage_by_team={"Hart": _EQUAL_LEVERAGE, "Rival": _EQUAL_LEVERAGE},
            roster_slots=ROSTER_SLOTS, rankings=rankings,
        )
        # rank gap = 10 > 5, should be excluded
        all_candidates = [c for g in results for c in g["candidates"]]
        assert not any(c["receive"] == "Opp OF" for c in all_candidates)

    def test_positional_weakness_included(self):
        """Each opponent group should include a positional_weakness score."""
        hart_roster = [_make_hitter("Hart SS", ["SS"], hr=15, sb=5)]
        opp_rosters = {"Rival": [_make_hitter("Opp SS", ["SS"], hr=25, sb=15)]}
        rankings = {
            rank_key("Hart SS", "hitter"): 55,
            rank_key("Opp SS", "hitter"): 50,
        }
        results = search_trades_away(
            player_name="Hart SS",
            hart_name="Hart", hart_roster=hart_roster, opp_rosters=opp_rosters,
            standings=SAMPLE_STANDINGS,
            leverage_by_team={"Hart": _EQUAL_LEVERAGE, "Rival": _EQUAL_LEVERAGE},
            roster_slots=ROSTER_SLOTS, rankings=rankings,
        )
        for group in results:
            assert "positional_weakness" in group
```

- [ ] **Step 2: Run tests to confirm they fail**

Run: `pytest tests/test_trades/test_trade_search.py -v`
Expected: FAIL — `ImportError: cannot import name 'search_trades_away'`

- [ ] **Step 3: Implement `search_trades_away`**

Add to `src/fantasy_baseball/trades/evaluate.py`, after the existing helper functions (after `_can_roster_without` at line 263):

```python
def _score_positional_weakness(
    player_positions: list[str],
    opp_roster: list[Player],
    opp_leverage: dict[str, float],
    all_opp_rosters: dict[str, list[Player]],
    all_leverage: dict[str, dict[str, float]],
) -> float:
    """Score how badly an opponent needs the offered positions.

    Compares the opponent's best starter at the offered position (by wSGP)
    against the league median at that position. Returns a score where higher
    means the opponent has a bigger need.
    """
    # Find the opponent's best player at any of the offered positions
    opp_best_wsgp = None
    for p in opp_roster:
        if set(p.positions) & set(player_positions):
            wsgp = calculate_weighted_sgp(p.ros, opp_leverage) if p.ros else 0.0
            if opp_best_wsgp is None or wsgp < opp_best_wsgp:
                opp_best_wsgp = wsgp

    if opp_best_wsgp is None:
        # Opponent has no one at this position — maximum weakness
        return 999.0

    # Gather wSGP at this position across all teams for the median
    all_wsgps = []
    for tname, roster in all_opp_rosters.items():
        team_lev = all_leverage.get(tname, {})
        for p in roster:
            if set(p.positions) & set(player_positions) and p.ros:
                all_wsgps.append(calculate_weighted_sgp(p.ros, team_lev))

    if not all_wsgps:
        return 0.0

    all_wsgps.sort()
    median_wsgp = all_wsgps[len(all_wsgps) // 2]

    # Weakness = how far below median the opponent's best is
    # Positive means they're weak; higher is weaker
    return median_wsgp - opp_best_wsgp


def search_trades_away(
    player_name: str,
    hart_name: str,
    hart_roster: list[Player],
    opp_rosters: dict[str, list[Player]],
    standings: list[dict],
    leverage_by_team: dict[str, dict],
    roster_slots: dict[str, int],
    rankings: dict[str, int],
    projected_standings: list[dict] | None = None,
) -> list[dict]:
    """Find trade candidates for a player the user wants to trade away.

    Searches all opponent rosters for players the user could receive in
    exchange. Results are grouped by opponent and sorted by positional
    weakness (teams that need the offered position most appear first).

    Args:
        player_name: name of the player to trade away (on user's roster).
        hart_name: user's team name in standings.
        hart_roster: user's roster as Player objects.
        opp_rosters: {opponent_name: [Player]} for each opponent.
        standings: current league standings.
        leverage_by_team: {team_name: {cat: weight}} leverage weights.
        roster_slots: league roster slot configuration.
        rankings: {rank_key: int} unweighted SGP ROS rankings.
        projected_standings: optional projected end-of-season standings.

    Returns:
        List of opponent groups:
        [{"opponent": str, "positional_weakness": float, "candidates": [...]}, ...]
        Groups sorted by positional_weakness descending (neediest teams first).
        Candidates sorted by hart_wsgp_gain descending within each group.
    """
    hart_player = _find_player_by_name(player_name, hart_roster)
    if hart_player is None:
        return []

    send_rank = rankings.get(
        rank_key_from_positions(hart_player.name, hart_player.positions))
    if send_rank is None:
        return []

    hart_leverage = leverage_by_team.get(hart_name, {})
    hart_wsgp = calculate_weighted_sgp(hart_player.ros, hart_leverage)

    grouped: dict[str, list[dict]] = {}

    for opp_name, opp_roster in opp_rosters.items():
        for opp_player in opp_roster:
            receive_rank = rankings.get(
                rank_key_from_positions(opp_player.name, opp_player.positions))
            if receive_rank is None:
                continue

            if not _can_roster_without(hart_roster, hart_player, opp_player, roster_slots):
                continue
            if not _can_roster_without(opp_roster, opp_player, hart_player, roster_slots):
                continue

            rank_gap = send_rank - receive_rank
            if rank_gap > MAX_RANK_GAP:
                continue

            gain_wsgp = calculate_weighted_sgp(opp_player.ros, hart_leverage)
            hart_wsgp_gain = gain_wsgp - hart_wsgp
            if hart_wsgp_gain <= 0:
                continue

            hart_ros = _player_ros_stats(hart_player)
            opp_ros = _player_ros_stats(opp_player)

            impact = compute_trade_impact(
                standings, hart_name, opp_name,
                hart_ros, opp_ros, opp_ros, hart_ros,
                projected_standings=projected_standings,
            )

            if impact["hart_delta"] < 0:
                continue

            grouped.setdefault(opp_name, []).append({
                "send": hart_player.name,
                "send_positions": hart_player.positions,
                "send_rank": send_rank,
                "receive": opp_player.name,
                "receive_positions": opp_player.positions,
                "receive_rank": receive_rank,
                "hart_wsgp_gain": round(hart_wsgp_gain, 2),
                "hart_delta": impact["hart_delta"],
                "opp_delta": impact["opp_delta"],
                "hart_cat_deltas": impact["hart_cat_deltas"],
                "opp_cat_deltas": impact["opp_cat_deltas"],
            })

    # Sort candidates within each group by wSGP gain descending
    for candidates in grouped.values():
        candidates.sort(key=lambda c: -c["hart_wsgp_gain"])

    # Score positional weakness per opponent and build result
    results = []
    for opp_name, candidates in grouped.items():
        opp_roster = opp_rosters[opp_name]
        opp_leverage = leverage_by_team.get(opp_name, {})
        weakness = _score_positional_weakness(
            hart_player.positions, opp_roster, opp_leverage,
            opp_rosters, leverage_by_team,
        )
        results.append({
            "opponent": opp_name,
            "positional_weakness": round(weakness, 2),
            "candidates": candidates,
        })

    # Sort groups by positional weakness descending (neediest first)
    results.sort(key=lambda g: -g["positional_weakness"])
    return results
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_trades/test_trade_search.py -v`
Expected: All 5 tests PASS.

- [ ] **Step 5: Commit**

```bash
git add src/fantasy_baseball/trades/evaluate.py tests/test_trades/test_trade_search.py
git commit -m "feat(trades): add search_trades_away with positional weakness scoring"
```

---

### Task 3: Write `search_trades_for` with tests

The "Trade For" search: given a player on an opponent's roster, find which of the user's players could be offered as a reasonable trade.

**Files:**
- Modify: `src/fantasy_baseball/trades/evaluate.py`
- Modify: `tests/test_trades/test_trade_search.py`

- [ ] **Step 1: Write failing tests for `search_trades_for`**

Add to `tests/test_trades/test_trade_search.py`:

```python
from fantasy_baseball.trades.evaluate import search_trades_for


class TestSearchTradesFor:
    def test_returns_single_opponent_group(self):
        """Results should contain exactly one group for the opponent who owns the target."""
        hart_roster = [
            _make_hitter("Hart OF", ["OF"], hr=25, sb=15),
            _make_hitter("Hart SS", ["SS"], hr=20, sb=10),
        ]
        opp_rosters = {"Rival": [_make_hitter("Target", ["OF"], hr=20, sb=20)]}
        rankings = {
            rank_key("Hart OF", "hitter"): 40,
            rank_key("Hart SS", "hitter"): 45,
            rank_key("Target", "hitter"): 48,
        }
        results = search_trades_for(
            player_name="Target",
            hart_name="Hart", hart_roster=hart_roster, opp_rosters=opp_rosters,
            standings=SAMPLE_STANDINGS,
            leverage_by_team={"Hart": _EQUAL_LEVERAGE, "Rival": _EQUAL_LEVERAGE},
            roster_slots=ROSTER_SLOTS, rankings=rankings,
        )
        assert len(results) == 1
        assert results[0]["opponent"] == "Rival"

    def test_player_not_found_returns_empty(self):
        """Searching for a player not on any opponent roster should return empty list."""
        hart_roster = [_make_hitter("Hart OF", ["OF"])]
        opp_rosters = {"Rival": [_make_hitter("Other", ["OF"])]}
        results = search_trades_for(
            player_name="Nonexistent",
            hart_name="Hart", hart_roster=hart_roster, opp_rosters=opp_rosters,
            standings=SAMPLE_STANDINGS,
            leverage_by_team={"Hart": _EQUAL_LEVERAGE, "Rival": _EQUAL_LEVERAGE},
            roster_slots=ROSTER_SLOTS, rankings={},
        )
        assert results == []

    def test_candidates_sorted_by_wsgp_gain(self):
        """Candidates should be sorted by wSGP gain descending."""
        hart_roster = [
            _make_hitter("Hart A", ["OF"], hr=25, sb=5),
            _make_hitter("Hart B", ["OF"], hr=22, sb=3),
        ]
        opp_rosters = {"Rival": [_make_hitter("Target", ["OF"], hr=15, sb=25)]}
        rankings = {
            rank_key("Hart A", "hitter"): 40,
            rank_key("Hart B", "hitter"): 42,
            rank_key("Target", "hitter"): 46,
        }
        leverage = {"Hart": {"R": .05, "HR": .05, "RBI": .05, "SB": .3, "AVG": .05,
                             "W": .1, "K": .1, "SV": .1, "ERA": .1, "WHIP": .1},
                    "Rival": _EQUAL_LEVERAGE}
        results = search_trades_for(
            player_name="Target",
            hart_name="Hart", hart_roster=hart_roster, opp_rosters=opp_rosters,
            standings=SAMPLE_STANDINGS,
            leverage_by_team=leverage,
            roster_slots=ROSTER_SLOTS, rankings=rankings,
        )
        if results and len(results[0]["candidates"]) >= 2:
            gains = [c["hart_wsgp_gain"] for c in results[0]["candidates"]]
            assert gains == sorted(gains, reverse=True)

    def test_candidates_have_required_fields(self):
        """Each candidate should include the standard trade proposal fields."""
        hart_roster = [_make_hitter("Hart OF", ["OF"], hr=25, sb=15)]
        opp_rosters = {"Rival": [_make_hitter("Target", ["OF"], hr=20, sb=20)]}
        rankings = {
            rank_key("Hart OF", "hitter"): 40,
            rank_key("Target", "hitter"): 45,
        }
        results = search_trades_for(
            player_name="Target",
            hart_name="Hart", hart_roster=hart_roster, opp_rosters=opp_rosters,
            standings=SAMPLE_STANDINGS,
            leverage_by_team={"Hart": _EQUAL_LEVERAGE, "Rival": _EQUAL_LEVERAGE},
            roster_slots=ROSTER_SLOTS, rankings=rankings,
        )
        assert len(results) > 0
        candidate = results[0]["candidates"][0]
        for key in ("send", "receive", "send_rank", "receive_rank",
                    "send_positions", "receive_positions",
                    "hart_wsgp_gain", "hart_delta", "opp_delta",
                    "hart_cat_deltas", "opp_cat_deltas"):
            assert key in candidate, f"Missing key: {key}"
```

- [ ] **Step 2: Run tests to confirm they fail**

Run: `pytest tests/test_trades/test_trade_search.py::TestSearchTradesFor -v`
Expected: FAIL — `ImportError: cannot import name 'search_trades_for'`

- [ ] **Step 3: Implement `search_trades_for`**

Add to `src/fantasy_baseball/trades/evaluate.py`, after `search_trades_away`:

```python
def search_trades_for(
    player_name: str,
    hart_name: str,
    hart_roster: list[Player],
    opp_rosters: dict[str, list[Player]],
    standings: list[dict],
    leverage_by_team: dict[str, dict],
    roster_slots: dict[str, int],
    rankings: dict[str, int],
    projected_standings: list[dict] | None = None,
) -> list[dict]:
    """Find trade offers the user can make to acquire a specific opponent player.

    Searches the user's roster for players they could send that pass
    the rank proximity filter and produce positive wSGP gain.

    Args:
        player_name: name of the player to acquire (on an opponent's roster).
        hart_name: user's team name in standings.
        hart_roster: user's roster as Player objects.
        opp_rosters: {opponent_name: [Player]} for each opponent.
        standings: current league standings.
        leverage_by_team: {team_name: {cat: weight}} leverage weights.
        roster_slots: league roster slot configuration.
        rankings: {rank_key: int} unweighted SGP ROS rankings.
        projected_standings: optional projected end-of-season standings.

    Returns:
        List with a single opponent group (or empty if player not found):
        [{"opponent": str, "candidates": [...]}]
        Candidates sorted by hart_wsgp_gain descending.
    """
    # Find which opponent owns the target player
    target_player = None
    target_opp = None
    for opp_name, opp_roster in opp_rosters.items():
        found = _find_player_by_name(player_name, opp_roster)
        if found is not None:
            target_player = found
            target_opp = opp_name
            break

    if target_player is None:
        return []

    receive_rank = rankings.get(
        rank_key_from_positions(target_player.name, target_player.positions))
    if receive_rank is None:
        return []

    hart_leverage = leverage_by_team.get(hart_name, {})
    gain_wsgp = calculate_weighted_sgp(target_player.ros, hart_leverage)
    opp_roster = opp_rosters[target_opp]

    candidates = []
    for hart_player in hart_roster:
        send_rank = rankings.get(
            rank_key_from_positions(hart_player.name, hart_player.positions))
        if send_rank is None:
            continue

        if not _can_roster_without(hart_roster, hart_player, target_player, roster_slots):
            continue
        if not _can_roster_without(opp_roster, target_player, hart_player, roster_slots):
            continue

        rank_gap = send_rank - receive_rank
        if rank_gap > MAX_RANK_GAP:
            continue

        hart_wsgp = calculate_weighted_sgp(hart_player.ros, hart_leverage)
        hart_wsgp_gain = gain_wsgp - hart_wsgp
        if hart_wsgp_gain <= 0:
            continue

        hart_ros = _player_ros_stats(hart_player)
        target_ros = _player_ros_stats(target_player)

        impact = compute_trade_impact(
            standings, hart_name, target_opp,
            hart_ros, target_ros, target_ros, hart_ros,
            projected_standings=projected_standings,
        )

        if impact["hart_delta"] < 0:
            continue

        candidates.append({
            "send": hart_player.name,
            "send_positions": hart_player.positions,
            "send_rank": send_rank,
            "receive": target_player.name,
            "receive_positions": target_player.positions,
            "receive_rank": receive_rank,
            "hart_wsgp_gain": round(hart_wsgp_gain, 2),
            "hart_delta": impact["hart_delta"],
            "opp_delta": impact["opp_delta"],
            "hart_cat_deltas": impact["hart_cat_deltas"],
            "opp_cat_deltas": impact["opp_cat_deltas"],
        })

    candidates.sort(key=lambda c: -c["hart_wsgp_gain"])

    if not candidates:
        return []

    return [{"opponent": target_opp, "candidates": candidates}]
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_trades/test_trade_search.py -v`
Expected: All 9 tests PASS (5 from Task 2 + 4 new).

- [ ] **Step 5: Commit**

```bash
git add src/fantasy_baseball/trades/evaluate.py tests/test_trades/test_trade_search.py
git commit -m "feat(trades): add search_trades_for for targeted trade acquisition"
```

---

### Task 4: Delete `find_trades` and update existing tests

Remove the old batch function and update the test file that imports it.

**Files:**
- Modify: `src/fantasy_baseball/trades/evaluate.py:266-367` (delete `find_trades`)
- Modify: `tests/test_trades/test_evaluate.py`
- Modify: `src/fantasy_baseball/web/season_data.py:723` (remove import)
- Modify: `src/fantasy_baseball/web/season_data.py:1111-1141` (remove call + pitch generation)

- [ ] **Step 1: Delete `find_trades` from evaluate.py**

Remove the entire `find_trades` function (lines 266-367 — the exact lines after `search_trades_for` is added will shift; it's the function starting with `def find_trades(`).

- [ ] **Step 2: Update test_evaluate.py to remove find_trades tests**

In `tests/test_trades/test_evaluate.py`:

Remove the `find_trades` import from line 8:
```python
# Change this import:
from fantasy_baseball.trades.evaluate import (
    compute_roto_points,
    compute_roto_points_by_cat,
    compute_trade_impact,
    find_trades,
)
# To this:
from fantasy_baseball.trades.evaluate import (
    compute_roto_points,
    compute_roto_points_by_cat,
    compute_trade_impact,
)
```

Delete these test functions that call `find_trades`:
- `test_find_trades_returns_ranked_list` (lines 107-150)
- `test_rank_filter_accepts_within_threshold` (lines 171-184)
- `test_rank_filter_rejects_beyond_threshold` (lines 187-200)
- `test_rank_filter_accepts_sending_better_ranked` (lines 203-221)
- `test_rejects_trade_with_no_wsgp_gain` (lines 224-237)
- `test_sort_by_wsgp_gain_descending` (lines 240-264)
- `test_sort_tiebreaker_by_rank_generosity` (lines 267-290)
- `test_roster_legality_still_enforced` (lines 293-307)
- `test_trades_include_rank_data` (lines 310-326)

Also delete the shared fixtures that are only used by these tests:
- `_EQUAL_LEVERAGE` (line 154)
- `_make_hitter` (lines 156-159)
- `_make_pitcher` (lines 162-168)
- `ROSTER_SLOTS` (lines 65-66)
- `SAMPLE_STANDINGS` (lines 68-84)

Keep the following tests that test `compute_roto_points`, `compute_roto_points_by_cat`, and `compute_trade_impact`:
- `test_compute_roto_points` (lines 23-27)
- `test_compute_trade_impact` (lines 30-49)
- `test_trade_impact_zero_for_identical_players` (lines 52-62)
- `test_compute_roto_points_by_cat_missing_stats` (lines 87-104)

- [ ] **Step 3: Remove find_trades call from season_data.py**

In `src/fantasy_baseball/web/season_data.py`:

Remove the `find_trades` import (line 723):
```python
# Delete this line:
        from fantasy_baseball.trades.evaluate import find_trades
```

Remove the `generate_pitch` import (line 724):
```python
# Delete this line:
        from fantasy_baseball.trades.pitch import generate_pitch
```

Remove the entire trade evaluation block (lines 1111-1141). This is the section starting with `hart_roster_for_trades = [` through `write_cache("trades", trade_proposals, cache_dir)`. Also remove the progress message at line 1103 (`_progress("Evaluating trades...")`).

Remove `"trades": "trades.json",` from the CACHE_FILES dict (line 86).

- [ ] **Step 4: Run all tests**

Run: `pytest tests/ -v -x --timeout=30 -q`
Expected: All tests PASS. No remaining references to `find_trades`.

- [ ] **Step 5: Verify no remaining references to find_trades**

Run: `grep -r "find_trades" src/ tests/`
Expected: No matches.

- [ ] **Step 6: Commit**

```bash
git add src/fantasy_baseball/trades/evaluate.py tests/test_trades/test_evaluate.py src/fantasy_baseball/web/season_data.py
git commit -m "refactor(trades): remove batch find_trades in favor of targeted search"
```

---

### Task 5: Add the `/api/trade-search` endpoint

Wire up the Flask endpoint that calls the search functions on demand using cached data.

**Files:**
- Modify: `src/fantasy_baseball/web/season_routes.py`

- [ ] **Step 1: Add the trade search API endpoint**

In `src/fantasy_baseball/web/season_routes.py`, after the existing `/api/trade/<int:idx>/standings` route (around line 382), add:

```python
    @app.route("/api/trade-search", methods=["POST"])
    def api_trade_search():
        from fantasy_baseball.models.player import Player
        from fantasy_baseball.trades.evaluate import (
            search_trades_away, search_trades_for,
        )

        data = request.get_json(silent=True) or {}
        player_name = data.get("player_name", "").strip()
        mode = data.get("mode", "")

        if not player_name:
            return jsonify({"error": "player_name is required"}), 400
        if mode not in ("away", "for"):
            return jsonify({"error": "mode must be 'away' or 'for'"}), 400

        # Load cached data from last refresh
        config = _load_config()
        standings_raw = read_cache("standings")
        if not standings_raw:
            return jsonify({"error": "No standings data. Run a refresh first."}), 404

        roster_raw = read_cache("roster")
        if not roster_raw:
            return jsonify({"error": "No roster data. Run a refresh first."}), 404

        opp_rosters_raw = read_cache("opp_rosters")
        if not opp_rosters_raw:
            return jsonify({"error": "No opponent roster data. Run a refresh first."}), 404

        leverage_raw = read_cache("leverage")
        if not leverage_raw:
            return jsonify({"error": "No leverage data. Run a refresh first."}), 404

        rankings_raw = read_cache("rankings")
        if not rankings_raw:
            return jsonify({"error": "No rankings data. Run a refresh first."}), 404

        proj_cache = read_cache("projections") or {}
        projected_standings = proj_cache.get("projected_standings")

        # Reconstruct Player objects from cached dicts
        hart_roster = [Player.from_dict(p) for p in roster_raw]
        opp_rosters = {
            tname: [Player.from_dict(p) for p in players]
            for tname, players in opp_rosters_raw.items()
        }

        # Rankings cache stores {key: {ros: int, preseason: int, current: int}}
        # The search functions expect {key: int} (ROS rank only)
        flat_rankings = {}
        for key, val in rankings_raw.items():
            if isinstance(val, dict):
                ros = val.get("ros")
                if ros is not None:
                    flat_rankings[key] = ros
            elif isinstance(val, int):
                flat_rankings[key] = val

        kwargs = dict(
            player_name=player_name,
            hart_name=config.team_name,
            hart_roster=hart_roster,
            opp_rosters=opp_rosters,
            standings=standings_raw,
            leverage_by_team=leverage_raw,
            roster_slots=config.roster_slots,
            rankings=flat_rankings,
            projected_standings=projected_standings,
        )

        if mode == "away":
            results = search_trades_away(**kwargs)
        else:
            results = search_trades_for(**kwargs)

        return jsonify(results)
```

- [ ] **Step 2: Remove the old trade standings endpoint**

Delete the `/api/trade/<int:idx>/standings` route (lines 362-382). This endpoint loads trades by index from the cache, which no longer exists.

- [ ] **Step 3: Update the waivers-trades route to stop passing trades**

Modify the `/waivers-trades` route handler. Change:
```python
        trades_raw = read_cache("trades")
```
to remove that line, and remove `trades=trades_raw or [],` from the `render_template` call.

The route should become:
```python
    @app.route("/waivers-trades")
    def waivers_trades():
        meta = read_meta()
        waivers_raw = read_cache("waivers")
        buy_low_raw = read_cache("buy_low") or {}

        # Build player name list for trade search autocomplete
        roster_raw = read_cache("roster") or []
        opp_rosters_raw = read_cache("opp_rosters") or {}
        my_players = sorted(set(p.get("name", "") for p in roster_raw if p.get("name")))
        opp_players = sorted(set(
            p.get("name", "")
            for players in opp_rosters_raw.values()
            for p in players
            if p.get("name")
        ))

        return render_template(
            "season/waivers_trades.html",
            meta=meta,
            active_page="waivers_trades",
            waivers=waivers_raw or [],
            buy_low_targets=buy_low_raw.get("trade_targets", []),
            buy_low_free_agents=buy_low_raw.get("free_agents", []),
            categories=ALL_CATEGORIES,
            my_players=my_players,
            opp_players=opp_players,
        )
```

- [ ] **Step 4: Run tests**

Run: `pytest tests/ -v -x --timeout=30 -q`
Expected: All tests PASS.

- [ ] **Step 5: Commit**

```bash
git add src/fantasy_baseball/web/season_routes.py
git commit -m "feat(trades): add /api/trade-search endpoint and wire up cached data"
```

---

### Task 6: Replace frontend Trade Recommendations with interactive search

Replace the static trade section in the template with a search input, mode toggle, and dynamic results rendering.

**Files:**
- Modify: `src/fantasy_baseball/web/templates/season/waivers_trades.html`

- [ ] **Step 1: Replace the Trade Recommendations HTML section**

In `waivers_trades.html`, replace everything from line 174 (`{# === Trade Recommendations === #}`) through line 257 (the closing `</div>` of `section-trades`) with:

```html
{# === Trade Finder === #}
<div class="section-toggle" onclick="toggleSection('trades')">
    <span>Trade Finder</span>
    <span class="chevron" id="chevron-trades">&#9660;</span>
</div>
<div class="section-body" id="section-trades">

<div style="display: flex; gap: 8px; align-items: center; margin-bottom: 16px; flex-wrap: wrap;">
    <input type="text" id="trade-search-input" list="trade-player-list"
           placeholder="Type a player name..."
           style="flex: 1; min-width: 200px; padding: 8px 12px; border-radius: 6px;
                  border: 1px solid var(--panel-border); background: var(--panel-bg);
                  color: var(--text); font-size: 14px;">
    <datalist id="trade-player-list">
        {% for name in my_players %}
        <option value="{{ name }}">
        {% endfor %}
        {% for name in opp_players %}
        <option value="{{ name }}">
        {% endfor %}
    </datalist>
    <button class="pill" onclick="tradeSearch('away')" id="btn-trade-away">Trade Away</button>
    <button class="pill" onclick="tradeSearch('for')" id="btn-trade-for">Trade For</button>
</div>

<div id="trade-search-results"></div>

</div>
```

- [ ] **Step 2: Replace the trade-related JavaScript**

Remove the old `toggleTradeDetails`, `loadTradeStandings`, `renderTradeStandings`, and `toggleTradeStandingsView` functions (lines 410-521). Replace with:

```javascript
function tradeSearch(mode) {
    var input = document.getElementById('trade-search-input');
    var name = input.value.trim();
    if (!name) return;

    var container = document.getElementById('trade-search-results');
    container.innerHTML = '<p style="color: var(--text-secondary); font-size: 13px;">Searching...</p>';

    // Highlight active button
    document.getElementById('btn-trade-away').classList.remove('active');
    document.getElementById('btn-trade-for').classList.remove('active');
    document.getElementById('btn-trade-' + mode).classList.add('active');

    fetch('/api/trade-search', {
        method: 'POST',
        headers: {'Content-Type': 'application/json'},
        body: JSON.stringify({player_name: name, mode: mode})
    })
    .then(function(r) { return r.json(); })
    .then(function(data) {
        if (data.error) {
            container.innerHTML = '<p style="color: var(--danger); font-size: 13px;">' + esc(data.error) + '</p>';
            return;
        }
        container.innerHTML = renderTradeResults(data, mode);
    })
    .catch(function(e) {
        container.innerHTML = '<p style="color: var(--danger); font-size: 13px;">Search failed.</p>';
    });
}

function renderTradeResults(groups, mode) {
    if (!groups || groups.length === 0) {
        return '<p style="color: var(--text-secondary); font-size: 13px;">No trade candidates found.</p>';
    }

    var html = '';
    for (var g = 0; g < groups.length; g++) {
        var group = groups[g];
        html += '<div style="margin-bottom: 20px;">';
        html += '<div style="font-weight: 600; font-size: 15px; margin-bottom: 8px;">';
        html += esc(group.opponent);
        if (mode === 'away' && group.positional_weakness != null) {
            var wk = group.positional_weakness;
            var wkClass = wk > 0 ? 'cat-gain' : 'cat-loss';
            html += ' <span class="' + wkClass + '" style="font-size: 11px; font-weight: 500;">';
            html += wk > 0 ? 'Needs this position' : 'Stacked here';
            html += '</span>';
        }
        html += '</div>';

        for (var c = 0; c < group.candidates.length; c++) {
            var t = group.candidates[c];
            html += renderTradeCard(t, g, c);
        }
        html += '</div>';
    }
    return html;
}

function renderTradeCard(t, gIdx, cIdx) {
    var id = 'tc-' + gIdx + '-' + cIdx;
    var html = '<div class="card" id="' + id + '">';
    html += '<div class="card-header" style="cursor: pointer;" onclick="toggleEl(\'' + id + '-detail\')">';
    html += '<div>';
    html += '<span style="font-weight: 600; font-size: 14px;">Send ' + esc(t.send) + '</span>';
    html += ' <span style="color: var(--text-secondary); font-size: 11px;">#' + t.send_rank + '</span>';
    html += ' <span style="color: var(--text-secondary); font-size: 11px; margin: 0 6px;">&#8644;</span>';
    html += '<span style="font-weight: 600; font-size: 14px;">Get ' + esc(t.receive) + '</span>';
    html += ' <span style="color: var(--text-secondary); font-size: 11px;">#' + t.receive_rank + '</span>';
    html += '</div>';
    html += '<div style="display: flex; gap: 12px; align-items: center;">';
    html += '<span style="color: var(--success); font-weight: bold; font-size: 13px;">+' + t.hart_wsgp_gain.toFixed(2) + ' wSGP</span>';
    if (t.hart_delta !== 0) {
        var dc = t.hart_delta > 0 ? 'var(--success)' : 'var(--danger)';
        html += '<span style="color: ' + dc + '; font-weight: bold; font-size: 13px;">' + (t.hart_delta > 0 ? '+' : '') + t.hart_delta + ' roto</span>';
    }
    html += '</div></div>';

    // Expandable detail
    html += '<div id="' + id + '-detail" class="trade-details">';
    html += '<div style="display: flex; gap: 24px; flex-wrap: wrap; margin-top: 8px;">';

    // Your category impact
    html += '<div style="flex: 1; min-width: 160px;">';
    html += '<div style="font-size: 11px; color: var(--text-secondary); font-weight: 600; margin-bottom: 6px; text-transform: uppercase; letter-spacing: 0.5px;">Your Impact</div>';
    html += '<div class="cat-impact">';
    html += renderCatDeltas(t.hart_cat_deltas);
    html += '</div></div>';

    // Opponent category impact
    html += '<div style="flex: 1; min-width: 160px;">';
    html += '<div style="font-size: 11px; color: var(--text-secondary); font-weight: 600; margin-bottom: 6px; text-transform: uppercase; letter-spacing: 0.5px;">Their Impact</div>';
    html += '<div class="cat-impact">';
    html += renderCatDeltas(t.opp_cat_deltas);
    html += '</div></div>';

    html += '</div></div></div>';
    return html;
}

function renderCatDeltas(deltas) {
    var cats = ['R','HR','RBI','SB','AVG','W','K','SV','ERA','WHIP'];
    var html = '';
    for (var i = 0; i < cats.length; i++) {
        var cat = cats[i];
        var d = deltas[cat] || 0;
        if (d > 0) html += '<span class="cat-gain">' + cat + ' +' + d + '</span>';
        else if (d < 0) html += '<span class="cat-loss">' + cat + ' ' + d + '</span>';
    }
    return html;
}

function toggleEl(id) {
    var el = document.getElementById(id);
    if (el) el.classList.toggle('open');
}
```

Also add enter-key support inside the existing `<script>` tag:

```javascript
document.addEventListener('DOMContentLoaded', function() {
    var input = document.getElementById('trade-search-input');
    if (input) {
        input.addEventListener('keydown', function(e) {
            if (e.key === 'Enter') {
                // Default to "away" if user's player, "for" if opponent's player
                var myPlayers = [{% for name in my_players %}'{{ name }}',{% endfor %}];
                var mode = myPlayers.indexOf(this.value.trim()) >= 0 ? 'away' : 'for';
                tradeSearch(mode);
            }
        });
    }
});
```

- [ ] **Step 3: Verify the template renders without errors**

Run the app briefly to check for template syntax errors:

```bash
cd C:/Users/alden/FantasyBaseball && python -c "
from fantasy_baseball.web.season_app import create_app
app = create_app()
with app.test_client() as c:
    resp = c.get('/waivers-trades')
    print(f'Status: {resp.status_code}')
    assert resp.status_code == 200, f'Got {resp.status_code}'
    print('Template renders OK')
"
```

Expected: `Status: 200` and `Template renders OK`.

- [ ] **Step 4: Commit**

```bash
git add src/fantasy_baseball/web/templates/season/waivers_trades.html
git commit -m "feat(trades): replace static trade recs with interactive trade finder UI"
```

---

### Task 7: End-to-end smoke test and cleanup

Verify the full flow works: search input → API call → results rendered. Clean up any dead code.

**Files:**
- Possibly modify: `src/fantasy_baseball/trades/pitch.py` (check if still used)
- Possibly modify: `src/fantasy_baseball/web/season_data.py` (remove `compute_trade_standings_impact` if dead)

- [ ] **Step 1: Check for dead code**

Run:
```bash
grep -r "generate_pitch\|compute_trade_standings_impact\|find_trades" src/ tests/
```

If `generate_pitch` is no longer imported anywhere in `src/` or `tests/`, note it for potential removal but leave it (it may be useful for the trade finder later).

If `compute_trade_standings_impact` is no longer referenced (the `/api/trade/<idx>/standings` endpoint was removed), remove it from `season_data.py`.

- [ ] **Step 2: Remove `compute_trade_standings_impact` if dead**

If grep confirms it's unused, delete the function from `season_data.py` (lines 556-610).

- [ ] **Step 3: Run all tests**

Run: `pytest tests/ -v --timeout=30 -q`
Expected: All tests PASS. No import errors, no broken references.

- [ ] **Step 4: Commit cleanup**

```bash
git add -A
git commit -m "chore(trades): remove dead code from old batch trade system"
```

---

Plan complete and saved to `docs/superpowers/plans/2026-04-08-trade-finder.md`. Two execution options:

**1. Subagent-Driven (recommended)** - I dispatch a fresh subagent per task, review between tasks, fast iteration

**2. Inline Execution** - Execute tasks in this session using executing-plans, batch execution with checkpoints

Which approach?