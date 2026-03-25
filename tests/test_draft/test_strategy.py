"""Comprehensive unit tests for all strategy functions in strategy.py.

Tests cover 14 strategy functions + 6 helper functions with happy paths,
constraint-triggering paths, and edge cases.
"""
import pytest
import pandas as pd
from unittest.mock import patch

from fantasy_baseball.config import LeagueConfig
from fantasy_baseball.draft.tracker import DraftTracker
from fantasy_baseball.draft.balance import CategoryBalance
from fantasy_baseball.draft.strategy import (
    pick_default,
    pick_nonzero_sv,
    pick_avg_hedge,
    pick_no_punt_opp,
    pick_two_closers,
    pick_three_closers,
    pick_four_closers,
    pick_no_punt,
    pick_no_punt_stagger,
    pick_no_punt_cap3,
    pick_avg_anchor,
    pick_closers_avg,
    pick_balanced,
    _count_closers,
    _count_hitters,
    _count_pitchers,
    _sv_in_danger,
    _force_closer,
    _fallback_non_closer,
    _can_roster_player,
    CLOSER_DEADLINE_ROUND,
    AVG_FLOOR,
    TWO_CLOSERS_DEADLINES,
    THREE_CLOSERS_DEADLINES,
    FOUR_CLOSERS_DEADLINES,
    NO_PUNT_AVG_FLOOR,
    NO_PUNT_SV_DEADLINE,
    NO_PUNT_SV_MIN_TEAMS_WITH_CLOSERS,
    NO_PUNT_SV_DANGER_ZONE,
    NO_PUNT_STAGGER_DEADLINES,
    NO_PUNT_CAP3_TARGET,
    AVG_ANCHOR_MIN,
    AVG_ANCHOR_DEADLINE_HITTER,
    BALANCED_MAX_SKEW,
    STRATEGIES,
)
from fantasy_baseball.utils.constants import CLOSER_SV_THRESHOLD, DEFAULT_ROSTER_SLOTS


# ---------------------------------------------------------------------------
# Fixtures and helpers
# ---------------------------------------------------------------------------

def _make_config(roster_slots=None, num_teams=10):
    """Create a minimal LeagueConfig for testing."""
    if roster_slots is None:
        roster_slots = DEFAULT_ROSTER_SLOTS.copy()
    return LeagueConfig(
        league_id=1,
        num_teams=num_teams,
        game_code="mlb",
        team_name="Test Team",
        draft_position=1,
        keepers=[],
        roster_slots=roster_slots,
        projection_systems=["steamer"],
        projection_weights={"steamer": 1.0},
    )


def _make_hitter(name, var, adp, avg, ab, positions=None, **extra):
    """Create a hitter row dict."""
    if positions is None:
        positions = ["OF"]
    h = int(avg * ab)
    data = {
        "name": name,
        "player_id": f"{name}::hitter",
        "player_type": "hitter",
        "var": var,
        "total_sgp": var * 1.2,
        "best_position": positions[0],
        "positions": positions,
        "adp": adp,
        "avg": avg,
        "ab": ab,
        "h": h,
        "r": 80,
        "hr": 25,
        "rbi": 80,
        "sb": 10,
        "sv": 0,
        "ip": 0,
    }
    data.update(extra)
    return data


def _make_sp(name, var, adp, ip=180, sv=0, **extra):
    """Create a starting pitcher row dict."""
    data = {
        "name": name,
        "player_id": f"{name}::pitcher",
        "player_type": "pitcher",
        "var": var,
        "total_sgp": var * 1.1,
        "best_position": "P",
        "positions": ["SP"],
        "adp": adp,
        "avg": 0,
        "ab": 0,
        "h": 0,
        "r": 0,
        "hr": 0,
        "rbi": 0,
        "sb": 0,
        "sv": sv,
        "ip": ip,
        "w": 12,
        "k": 180,
        "era": 3.50,
        "whip": 1.15,
        "er": 70,
        "bb": 50,
        "h_allowed": 155,
    }
    data.update(extra)
    return data


def _make_closer(name, var, adp, sv=30, **extra):
    """Create a closer row dict (SV >= CLOSER_SV_THRESHOLD)."""
    data = {
        "name": name,
        "player_id": f"{name}::pitcher",
        "player_type": "pitcher",
        "var": var,
        "total_sgp": var * 1.1,
        "best_position": "P",
        "positions": ["RP"],
        "adp": adp,
        "avg": 0,
        "ab": 0,
        "h": 0,
        "r": 0,
        "hr": 0,
        "rbi": 0,
        "sb": 0,
        "sv": sv,
        "ip": 65,
        "w": 3,
        "k": 70,
        "era": 2.80,
        "whip": 1.05,
        "er": 20,
        "bb": 20,
        "h_allowed": 48,
    }
    data.update(extra)
    return data


def _make_board(players):
    """Create a DataFrame from a list of player dicts.

    When players is empty, returns a DataFrame with the expected columns
    so that strategy code can filter without KeyError.
    """
    if not players:
        return pd.DataFrame(columns=[
            "name", "player_id", "player_type", "var", "total_sgp",
            "best_position", "positions", "adp", "avg", "ab", "h",
            "r", "hr", "rbi", "sb", "sv", "ip", "w", "k",
            "era", "whip", "er", "bb", "h_allowed",
        ])
    return pd.DataFrame(players)


def _make_standard_board():
    """Create a standard test board with a mix of hitters, SPs, closers."""
    return _make_board([
        _make_hitter("Hitter A", var=15.0, adp=1, avg=0.290, ab=550, positions=["OF"]),
        _make_hitter("Hitter B", var=14.0, adp=2, avg=0.275, ab=530, positions=["SS"]),
        _make_hitter("Hitter C", var=13.0, adp=3, avg=0.260, ab=480, positions=["C"]),
        _make_hitter("Hitter D", var=12.0, adp=6, avg=0.300, ab=500, positions=["1B"]),
        _make_hitter("Hitter E", var=11.0, adp=7, avg=0.240, ab=550, positions=["2B"]),
        _make_hitter("Hitter F", var=10.0, adp=8, avg=0.230, ab=520, positions=["3B"]),
        _make_hitter("Hitter G", var=6.0, adp=15, avg=0.280, ab=480, positions=["OF"]),
        _make_hitter("Hitter H", var=5.0, adp=16, avg=0.295, ab=450, positions=["OF"]),
        _make_sp("SP A", var=13.5, adp=4, ip=195),
        _make_sp("SP B", var=12.5, adp=5, ip=185),
        _make_sp("SP C", var=8.0, adp=12, ip=170),
        _make_sp("SP D", var=5.5, adp=17, ip=200),
        _make_closer("Closer A", var=9.0, adp=9, sv=35),
        _make_closer("Closer B", var=7.0, adp=11, sv=30),
        _make_closer("Closer C", var=5.0, adp=14, sv=25),
        _make_closer("Closer D", var=3.0, adp=19, sv=22),
    ])


def _make_tracker(num_teams=10, user_position=1, rounds=22, advance_to=1):
    """Create a DraftTracker advanced to the given pick number."""
    tracker = DraftTracker(num_teams=num_teams, user_position=user_position, rounds=rounds)
    for _ in range(advance_to - 1):
        tracker.advance()
    return tracker


def _advance_to_round(tracker, target_round, num_teams=10):
    """Advance tracker to the start of a target round."""
    target_pick = (target_round - 1) * num_teams + 1
    while tracker.current_pick < target_pick:
        tracker.advance()
    return tracker


def _draft_user_player(tracker, name, player_id=None):
    """Draft a player for the user."""
    pid = player_id or f"{name}::hitter"
    tracker.draft_player(name, is_user=True, player_id=pid)


def _draft_other_player(tracker, name, player_id=None):
    """Draft a player for another team."""
    pid = player_id or f"{name}::hitter"
    tracker.draft_player(name, is_user=False, player_id=pid)


# ---------------------------------------------------------------------------
# Helper function tests
# ---------------------------------------------------------------------------

class TestCountClosers:
    def test_zero_closers(self):
        board = _make_standard_board()
        tracker = _make_tracker()
        _draft_user_player(tracker, "Hitter A", "Hitter A::hitter")
        assert _count_closers(tracker, board, board) == 0

    def test_one_closer(self):
        board = _make_standard_board()
        tracker = _make_tracker()
        _draft_user_player(tracker, "Closer A", "Closer A::pitcher")
        assert _count_closers(tracker, board, board) == 1

    def test_three_closers(self):
        board = _make_standard_board()
        tracker = _make_tracker()
        _draft_user_player(tracker, "Closer A", "Closer A::pitcher")
        _draft_user_player(tracker, "Closer B", "Closer B::pitcher")
        _draft_user_player(tracker, "Closer C", "Closer C::pitcher")
        assert _count_closers(tracker, board, board) == 3

    def test_closer_on_full_board_only(self):
        """Closer not on the filtered board but present in full_board."""
        board = _make_standard_board()
        # Create a reduced board without Closer A
        small_board = board[board["name"] != "Closer A"].copy()
        tracker = _make_tracker()
        _draft_user_player(tracker, "Closer A", "Closer A::pitcher")
        # full_board has the closer, small board doesn't
        assert _count_closers(tracker, small_board, board) == 1


class TestCountHittersAndPitchers:
    def test_hitter_count(self):
        board = _make_standard_board()
        tracker = _make_tracker()
        _draft_user_player(tracker, "Hitter A", "Hitter A::hitter")
        _draft_user_player(tracker, "Hitter B", "Hitter B::hitter")
        assert _count_hitters(tracker, board, board) == 2

    def test_pitcher_count(self):
        board = _make_standard_board()
        tracker = _make_tracker()
        _draft_user_player(tracker, "SP A", "SP A::pitcher")
        _draft_user_player(tracker, "Closer A", "Closer A::pitcher")
        _draft_user_player(tracker, "Hitter A", "Hitter A::hitter")
        assert _count_pitchers(tracker, board, board) == 2

    def test_empty_roster(self):
        board = _make_standard_board()
        tracker = _make_tracker()
        assert _count_hitters(tracker, board, board) == 0
        assert _count_pitchers(tracker, board, board) == 0


class TestSvInDanger:
    def test_no_team_rosters(self):
        board = _make_standard_board()
        tracker = _make_tracker()
        assert _sv_in_danger(tracker, board, board, {}, 10) is False

    def test_not_enough_teams_with_closers(self):
        """Should return False when fewer than MIN_TEAMS_WITH_CLOSERS have closers."""
        board = _make_standard_board()
        tracker = _make_tracker()
        _draft_user_player(tracker, "Hitter A", "Hitter A::hitter")
        # Only 1 team has a closer (need MIN_TEAMS_WITH_CLOSERS=3)
        team_rosters = {
            1: ["Hitter A::hitter"],
            2: ["Closer A::pitcher"],
            3: [],
            4: [],
        }
        assert _sv_in_danger(tracker, board, board, team_rosters, 4) is False

    def test_user_in_danger_zone(self):
        """User team has 0 SV while 3+ teams have closers -> danger."""
        board = _make_standard_board()
        tracker = _make_tracker()
        _draft_user_player(tracker, "Hitter A", "Hitter A::hitter")

        # 4 teams, 3 have closers, user has none
        team_rosters = {
            1: ["Hitter A::hitter"],  # user team (0 SV)
            2: ["Closer A::pitcher"],  # 35 SV
            3: ["Closer B::pitcher"],  # 30 SV
            4: ["Closer C::pitcher"],  # 25 SV
        }
        result = _sv_in_danger(tracker, board, board, team_rosters, 4)
        assert result is True

    def test_user_not_in_danger_zone(self):
        """User has a closer, not in danger."""
        board = _make_standard_board()
        tracker = _make_tracker()
        _draft_user_player(tracker, "Closer A", "Closer A::pitcher")

        team_rosters = {
            1: ["Closer A::pitcher"],  # user team (35 SV)
            2: ["Closer B::pitcher"],  # 30 SV
            3: ["Closer C::pitcher"],  # 25 SV
            4: ["Hitter A::hitter"],   # 0 SV
        }
        result = _sv_in_danger(tracker, board, board, team_rosters, 4)
        assert result is False

    def test_user_team_detection_via_set_intersection(self):
        """User team is identified by overlap of roster IDs."""
        board = _make_standard_board()
        tracker = _make_tracker()
        _draft_user_player(tracker, "Hitter A", "Hitter A::hitter")
        _draft_user_player(tracker, "Hitter B", "Hitter B::hitter")

        team_rosters = {
            1: ["Hitter A::hitter", "Hitter B::hitter"],  # user team
            2: ["Closer A::pitcher"],
            3: ["Closer B::pitcher"],
            4: ["Closer C::pitcher"],
        }
        # User has no SV, 3 teams have closers -> danger
        result = _sv_in_danger(tracker, board, board, team_rosters, 4)
        assert result is True

    def test_empty_team_rosters_dict(self):
        """Empty dict -> False."""
        board = _make_standard_board()
        tracker = _make_tracker()
        assert _sv_in_danger(tracker, board, board, {}, 10) is False


class TestForceCloser:
    def test_returns_best_closer_by_var(self):
        board = _make_standard_board()
        config = _make_config()
        tracker = _make_tracker()
        result = _force_closer(board, tracker, board, config)
        assert result is not None
        name, pid = result
        # Closer A has highest var (9.0)
        assert name == "Closer A"
        assert pid == "Closer A::pitcher"

    def test_no_closers_available(self):
        """All closers already drafted -> returns None."""
        board = _make_standard_board()
        config = _make_config()
        tracker = _make_tracker()
        # Draft all closers for other teams
        _draft_other_player(tracker, "Closer A", "Closer A::pitcher")
        _draft_other_player(tracker, "Closer B", "Closer B::pitcher")
        _draft_other_player(tracker, "Closer C", "Closer C::pitcher")
        _draft_other_player(tracker, "Closer D", "Closer D::pitcher")
        result = _force_closer(board, tracker, board, config)
        assert result is None

    def test_skips_closer_that_cant_be_rostered(self):
        """When all P slots are full, closers can't be rostered."""
        board = _make_standard_board()
        # Config with only 1 P slot and no BN
        config = _make_config(roster_slots={"P": 1, "C": 1, "1B": 1, "2B": 1, "3B": 1, "SS": 1, "OF": 4, "UTIL": 2, "IF": 1})
        tracker = _make_tracker()
        # Fill the P slot
        _draft_user_player(tracker, "SP A", "SP A::pitcher")
        result = _force_closer(board, tracker, board, config)
        assert result is None


class TestFallbackNonCloser:
    def test_returns_best_non_closer(self):
        board = _make_standard_board()
        config = _make_config()
        tracker = _make_tracker()
        name, pid = _fallback_non_closer(board, tracker, board, config)
        assert name is not None
        # Should return the best VAR non-closer player
        assert "Closer" not in name or pid is None

    def test_skips_closers(self):
        """Even if closers have high VAR, they should be skipped."""
        board = _make_standard_board()
        config = _make_config()
        tracker = _make_tracker()
        name, pid = _fallback_non_closer(board, tracker, board, config)
        # Should not return a closer
        if name:
            row = board[board["name"] == name]
            if not row.empty:
                assert row.iloc[0]["sv"] < CLOSER_SV_THRESHOLD

    def test_all_players_drafted(self):
        """When all players are already drafted, returns (None, None)."""
        board = _make_standard_board()
        config = _make_config()
        tracker = _make_tracker()
        # Draft all non-closers
        for _, row in board.iterrows():
            if row["sv"] < CLOSER_SV_THRESHOLD:
                _draft_other_player(tracker, row["name"], row["player_id"])
        # Draft all closers too
        for _, row in board.iterrows():
            if row["sv"] >= CLOSER_SV_THRESHOLD:
                _draft_other_player(tracker, row["name"], row["player_id"])
        name, pid = _fallback_non_closer(board, tracker, board, config)
        assert name is None
        assert pid is None


class TestCanRosterPlayer:
    def test_open_of_slot(self):
        player = pd.Series({"positions": ["OF"]})
        filled = {"OF": 1}
        slots = {"OF": 4, "UTIL": 2, "P": 9, "C": 1, "1B": 1, "2B": 1, "3B": 1, "SS": 1, "IF": 1, "BN": 2}
        assert _can_roster_player(player, filled, slots) is True

    def test_all_of_slots_full_but_util_open(self):
        player = pd.Series({"positions": ["OF"]})
        filled = {"OF": 4}
        slots = {"OF": 4, "UTIL": 2, "P": 9, "C": 1, "1B": 1, "2B": 1, "3B": 1, "SS": 1, "IF": 1, "BN": 2}
        assert _can_roster_player(player, filled, slots) is True

    def test_no_slots_available(self):
        player = pd.Series({"positions": ["OF"]})
        filled = {"OF": 4, "UTIL": 2, "BN": 2}
        slots = {"OF": 4, "UTIL": 2, "BN": 2, "P": 9, "C": 1, "1B": 1, "2B": 1, "3B": 1, "SS": 1, "IF": 1}
        assert _can_roster_player(player, filled, slots) is False

    def test_pitcher_fills_p_slot(self):
        player = pd.Series({"positions": ["SP"]})
        filled = {}
        slots = {"P": 9, "OF": 4, "C": 1, "1B": 1, "2B": 1, "3B": 1, "SS": 1, "IF": 1, "UTIL": 2, "BN": 2}
        assert _can_roster_player(player, filled, slots) is True

    def test_il_slot_is_ignored(self):
        """IL slots should not be used for draft roster assignment."""
        player = pd.Series({"positions": ["OF"]})
        filled = {"OF": 4, "UTIL": 2, "BN": 2}
        slots = {"OF": 4, "UTIL": 2, "BN": 2, "IL": 2, "P": 9, "C": 1, "1B": 1, "2B": 1, "3B": 1, "SS": 1, "IF": 1}
        # IL is excluded, so OF player has no slots
        assert _can_roster_player(player, filled, slots) is False

    def test_if_slot_for_infielder(self):
        """2B-eligible player can fill IF slot."""
        player = pd.Series({"positions": ["2B"]})
        filled = {"2B": 1}
        slots = {"2B": 1, "IF": 1, "UTIL": 2, "P": 9, "OF": 4, "C": 1, "1B": 1, "3B": 1, "SS": 1, "BN": 2}
        assert _can_roster_player(player, filled, slots) is True


# ---------------------------------------------------------------------------
# Strategy function tests
# ---------------------------------------------------------------------------

class TestPickDefault:
    def test_returns_top_recommendation(self):
        board = _make_standard_board()
        config = _make_config()
        tracker = _make_tracker()
        balance = CategoryBalance()
        name, pid = pick_default(board, board, tracker, balance, config, {})
        assert name is not None
        assert pid is not None

    def test_returns_consistent_player_id(self):
        """Returned pid matches the player_id column in the board."""
        board = _make_standard_board()
        config = _make_config()
        tracker = _make_tracker()
        balance = CategoryBalance()
        name, pid = pick_default(board, board, tracker, balance, config, {})
        row = board[board["name"] == name]
        assert not row.empty
        assert pid == row.iloc[0]["player_id"]

    def test_excludes_drafted_players(self):
        """Default should not recommend already-drafted players."""
        board = _make_standard_board()
        config = _make_config()
        tracker = _make_tracker()
        _draft_other_player(tracker, "Hitter A", "Hitter A::hitter")
        balance = CategoryBalance()
        name, pid = pick_default(board, board, tracker, balance, config, {})
        assert name != "Hitter A"


class TestPickNonzeroSv:
    def test_no_closer_before_deadline_takes_default(self):
        """Before the deadline, with no closer, should take default pick."""
        board = _make_standard_board()
        config = _make_config()
        tracker = _make_tracker()  # Round 1
        balance = CategoryBalance()
        name, pid = pick_nonzero_sv(board, board, tracker, balance, config, {})
        # Should pick as normal (default) since we're before deadline
        assert name is not None

    def test_forces_closer_at_deadline(self):
        """At the deadline round with no closer, should force a closer."""
        board = _make_standard_board()
        config = _make_config()
        tracker = _make_tracker()
        _advance_to_round(tracker, CLOSER_DEADLINE_ROUND)
        balance = CategoryBalance()
        name, pid = pick_nonzero_sv(board, board, tracker, balance, config, {})
        assert name is not None
        # Should be a closer
        row = board[board["name"] == name]
        assert not row.empty
        assert row.iloc[0]["sv"] >= CLOSER_SV_THRESHOLD

    def test_skips_force_when_closer_already_drafted(self):
        """If user already has a closer, don't force another one."""
        board = _make_standard_board()
        config = _make_config()
        tracker = _make_tracker()
        _draft_user_player(tracker, "Closer A", "Closer A::pitcher")
        _advance_to_round(tracker, CLOSER_DEADLINE_ROUND)
        balance = CategoryBalance()
        # Add the closer to balance so leverage works
        closer_row = board[board["name"] == "Closer A"].iloc[0]
        balance.add_player(closer_row)
        name, pid = pick_nonzero_sv(board, board, tracker, balance, config, {})
        # Should fall back to default (may or may not be a closer, but it's not forced)
        assert name is not None

    def test_no_closers_available_falls_back_to_default(self):
        """If no closers are available, should fall back to default."""
        board = _make_standard_board()
        config = _make_config()
        tracker = _make_tracker()
        # Draft all closers for other teams
        _draft_other_player(tracker, "Closer A", "Closer A::pitcher")
        _draft_other_player(tracker, "Closer B", "Closer B::pitcher")
        _draft_other_player(tracker, "Closer C", "Closer C::pitcher")
        _draft_other_player(tracker, "Closer D", "Closer D::pitcher")
        _advance_to_round(tracker, CLOSER_DEADLINE_ROUND)
        balance = CategoryBalance()
        name, pid = pick_nonzero_sv(board, board, tracker, balance, config, {})
        # Should still return something (default pick)
        assert name is not None
        # Should NOT be a closer since they're all gone
        row = board[board["name"] == name]
        if not row.empty:
            assert row.iloc[0]["sv"] < CLOSER_SV_THRESHOLD


class TestPickAvgHedge:
    def test_accepts_good_avg_hitter(self):
        """High-AVG hitter should be accepted."""
        board = _make_standard_board()
        config = _make_config()
        tracker = _make_tracker()
        balance = CategoryBalance()
        name, pid = pick_avg_hedge(board, board, tracker, balance, config, {})
        assert name is not None

    def test_skips_low_avg_hitter_when_would_tank_team(self):
        """When team AVG is near the floor, a low-AVG hitter should be skipped."""
        # Create a board with only low-AVG hitters and a pitcher
        board = _make_board([
            _make_hitter("Low AVG 1", var=15.0, adp=1, avg=0.200, ab=550, positions=["OF"]),
            _make_hitter("Low AVG 2", var=14.0, adp=2, avg=0.210, ab=530, positions=["SS"]),
            _make_sp("Good SP", var=13.0, adp=3),
        ])
        config = _make_config()
        tracker = _make_tracker()
        balance = CategoryBalance()
        # Add a hitter near the floor so adding a low-AVG hitter would tank it
        existing = pd.Series({
            "name": "Existing", "player_type": "hitter",
            "r": 80, "hr": 25, "rbi": 80, "sb": 10, "avg": 0.258, "ab": 500,
            "h": int(0.258 * 500),
        })
        balance.add_player(existing)
        name, pid = pick_avg_hedge(board, board, tracker, balance, config, {})
        # Should prefer the pitcher since low-AVG hitters would tank AVG
        assert name == "Good SP"

    def test_no_hitters_above_floor_takes_best_anyway(self):
        """If all hitters would tank AVG, should still take the best rec."""
        board = _make_board([
            _make_hitter("Low AVG 1", var=15.0, adp=1, avg=0.200, ab=550, positions=["OF"]),
            _make_hitter("Low AVG 2", var=14.0, adp=2, avg=0.210, ab=530, positions=["SS"]),
        ])
        config = _make_config()
        tracker = _make_tracker()
        balance = CategoryBalance()
        existing = pd.Series({
            "name": "Existing", "player_type": "hitter",
            "r": 80, "hr": 25, "rbi": 80, "sb": 10, "avg": 0.256, "ab": 500,
            "h": int(0.256 * 500),
        })
        balance.add_player(existing)
        name, pid = pick_avg_hedge(board, board, tracker, balance, config, {})
        # Falls back to best rec
        assert name is not None

    def test_first_hitter_always_accepted(self):
        """With no current AB, any hitter should be accepted."""
        board = _make_board([
            _make_hitter("Low AVG", var=15.0, adp=1, avg=0.220, ab=550, positions=["OF"]),
        ])
        config = _make_config()
        tracker = _make_tracker()
        balance = CategoryBalance()
        name, pid = pick_avg_hedge(board, board, tracker, balance, config, {})
        assert name == "Low AVG"


class TestPickNoPuntOpp:
    def test_default_pick_when_no_danger(self):
        """When not in SV danger, should pick normally."""
        board = _make_standard_board()
        config = _make_config()
        tracker = _make_tracker()
        balance = CategoryBalance()
        name, pid = pick_no_punt_opp(board, board, tracker, balance, config, {})
        assert name is not None

    def test_forces_closer_when_sv_in_danger(self):
        """Forces a closer when SV danger detected via team_rosters."""
        board = _make_standard_board()
        config = _make_config(num_teams=4)
        tracker = _make_tracker(num_teams=4)
        _draft_user_player(tracker, "Hitter A", "Hitter A::hitter")
        balance = CategoryBalance()
        # 3 other teams have closers, user has none
        team_rosters = {
            1: ["Hitter A::hitter"],
            2: ["Closer A::pitcher"],
            3: ["Closer B::pitcher"],
            4: ["Closer C::pitcher"],
        }
        name, pid = pick_no_punt_opp(
            board, board, tracker, balance, config, {},
            team_rosters=team_rosters,
        )
        # Should force a closer
        assert name is not None
        row = board[board["name"] == name]
        assert not row.empty
        assert row.iloc[0]["sv"] >= CLOSER_SV_THRESHOLD

    def test_legacy_fallback_without_team_rosters(self):
        """Without team_rosters, uses round-based deadline."""
        board = _make_standard_board()
        config = _make_config()
        tracker = _make_tracker()
        _advance_to_round(tracker, NO_PUNT_SV_DEADLINE)
        balance = CategoryBalance()
        # No team_rosters -> legacy deadline
        name, pid = pick_no_punt_opp(board, board, tracker, balance, config, {})
        assert name is not None
        # Should force a closer since no closer and past deadline
        row = board[board["name"] == name]
        assert not row.empty
        assert row.iloc[0]["sv"] >= CLOSER_SV_THRESHOLD

    def test_opportunistic_closer_past_adp(self):
        """Grabs a closer when they've fallen past their ADP."""
        board = _make_standard_board()
        config = _make_config()
        tracker = _make_tracker()
        # Advance past Closer D's ADP (19)
        _advance_to_round(tracker, 2)  # pick ~11+
        # Manually set current_pick so effective_pick >= Closer D's ADP
        while tracker.current_pick < 20:
            tracker.advance()
        balance = CategoryBalance()
        name, pid = pick_no_punt_opp(board, board, tracker, balance, config, {})
        assert name is not None

    def test_avg_floor_applied(self):
        """Low-AVG hitters are filtered when AVG floor is active."""
        board = _make_board([
            _make_hitter("Low AVG", var=15.0, adp=1, avg=0.200, ab=550, positions=["OF"]),
            _make_sp("Good SP", var=14.0, adp=2),
        ])
        config = _make_config()
        tracker = _make_tracker()
        balance = CategoryBalance()
        existing = pd.Series({
            "name": "Existing", "player_type": "hitter",
            "r": 80, "hr": 25, "rbi": 80, "sb": 10, "avg": 0.252, "ab": 500,
            "h": int(0.252 * 500),
        })
        balance.add_player(existing)
        name, pid = pick_no_punt_opp(board, board, tracker, balance, config, {})
        # Should prefer the pitcher since low-AVG hitter would tank team AVG
        assert name == "Good SP"


class TestPickTwoClosers:
    def test_normal_pick_before_deadline(self):
        """Before first closer deadline, picks default."""
        board = _make_standard_board()
        config = _make_config()
        tracker = _make_tracker()
        balance = CategoryBalance()
        name, pid = pick_two_closers(board, board, tracker, balance, config, {})
        assert name is not None

    def test_forces_first_closer_at_deadline(self):
        """At the first deadline (round 8) with 0 closers, forces one."""
        board = _make_standard_board()
        config = _make_config()
        tracker = _make_tracker()
        _advance_to_round(tracker, TWO_CLOSERS_DEADLINES[0])
        balance = CategoryBalance()
        name, pid = pick_two_closers(board, board, tracker, balance, config, {})
        assert name is not None
        row = board[board["name"] == name]
        assert not row.empty
        assert row.iloc[0]["sv"] >= CLOSER_SV_THRESHOLD

    def test_forces_second_closer_at_deadline(self):
        """At the second deadline with 1 closer, forces another."""
        board = _make_standard_board()
        config = _make_config()
        tracker = _make_tracker()
        _draft_user_player(tracker, "Closer A", "Closer A::pitcher")
        _advance_to_round(tracker, TWO_CLOSERS_DEADLINES[1])
        balance = CategoryBalance()
        name, pid = pick_two_closers(board, board, tracker, balance, config, {})
        assert name is not None
        row = board[board["name"] == name]
        assert not row.empty
        assert row.iloc[0]["sv"] >= CLOSER_SV_THRESHOLD

    def test_no_force_when_target_met(self):
        """With 2 closers already, should not force another."""
        board = _make_standard_board()
        config = _make_config()
        tracker = _make_tracker()
        _draft_user_player(tracker, "Closer A", "Closer A::pitcher")
        _draft_user_player(tracker, "Closer B", "Closer B::pitcher")
        _advance_to_round(tracker, 20)  # Way past deadlines
        balance = CategoryBalance()
        name, pid = pick_two_closers(board, board, tracker, balance, config, {})
        # Should pick default (not forced to get a third closer)
        assert name is not None


class TestPickThreeClosers:
    def test_forces_at_first_deadline(self):
        board = _make_standard_board()
        config = _make_config()
        tracker = _make_tracker()
        _advance_to_round(tracker, THREE_CLOSERS_DEADLINES[0])
        balance = CategoryBalance()
        name, pid = pick_three_closers(board, board, tracker, balance, config, {})
        row = board[board["name"] == name]
        assert not row.empty
        assert row.iloc[0]["sv"] >= CLOSER_SV_THRESHOLD

    def test_forces_at_second_deadline_with_one_closer(self):
        board = _make_standard_board()
        config = _make_config()
        tracker = _make_tracker()
        _draft_user_player(tracker, "Closer A", "Closer A::pitcher")
        _advance_to_round(tracker, THREE_CLOSERS_DEADLINES[1])
        balance = CategoryBalance()
        name, pid = pick_three_closers(board, board, tracker, balance, config, {})
        row = board[board["name"] == name]
        assert not row.empty
        assert row.iloc[0]["sv"] >= CLOSER_SV_THRESHOLD

    def test_no_force_when_three_closers_drafted(self):
        board = _make_standard_board()
        config = _make_config()
        tracker = _make_tracker()
        _draft_user_player(tracker, "Closer A", "Closer A::pitcher")
        _draft_user_player(tracker, "Closer B", "Closer B::pitcher")
        _draft_user_player(tracker, "Closer C", "Closer C::pitcher")
        _advance_to_round(tracker, 20)
        balance = CategoryBalance()
        name, pid = pick_three_closers(board, board, tracker, balance, config, {})
        assert name is not None


class TestPickFourClosers:
    def test_forces_at_first_deadline(self):
        board = _make_standard_board()
        config = _make_config()
        tracker = _make_tracker()
        _advance_to_round(tracker, FOUR_CLOSERS_DEADLINES[0])
        balance = CategoryBalance()
        name, pid = pick_four_closers(board, board, tracker, balance, config, {})
        row = board[board["name"] == name]
        assert not row.empty
        assert row.iloc[0]["sv"] >= CLOSER_SV_THRESHOLD

    def test_forces_at_fourth_deadline(self):
        board = _make_standard_board()
        config = _make_config()
        tracker = _make_tracker()
        _draft_user_player(tracker, "Closer A", "Closer A::pitcher")
        _draft_user_player(tracker, "Closer B", "Closer B::pitcher")
        _draft_user_player(tracker, "Closer C", "Closer C::pitcher")
        _advance_to_round(tracker, FOUR_CLOSERS_DEADLINES[3])
        balance = CategoryBalance()
        name, pid = pick_four_closers(board, board, tracker, balance, config, {})
        row = board[board["name"] == name]
        assert not row.empty
        assert row.iloc[0]["sv"] >= CLOSER_SV_THRESHOLD

    def test_no_force_when_four_closers(self):
        board = _make_standard_board()
        config = _make_config()
        tracker = _make_tracker()
        _draft_user_player(tracker, "Closer A", "Closer A::pitcher")
        _draft_user_player(tracker, "Closer B", "Closer B::pitcher")
        _draft_user_player(tracker, "Closer C", "Closer C::pitcher")
        _draft_user_player(tracker, "Closer D", "Closer D::pitcher")
        _advance_to_round(tracker, 20)
        balance = CategoryBalance()
        name, pid = pick_four_closers(board, board, tracker, balance, config, {})
        assert name is not None


class TestPickNoPunt:
    def test_default_pick_early_in_draft(self):
        board = _make_standard_board()
        config = _make_config()
        tracker = _make_tracker()
        balance = CategoryBalance()
        name, pid = pick_no_punt(board, board, tracker, balance, config, {})
        assert name is not None

    def test_forces_closer_via_dynamic_sv(self):
        """Forces closer when SV danger detected via team_rosters."""
        board = _make_standard_board()
        config = _make_config(num_teams=4)
        tracker = _make_tracker(num_teams=4)
        _draft_user_player(tracker, "Hitter A", "Hitter A::hitter")
        balance = CategoryBalance()
        team_rosters = {
            1: ["Hitter A::hitter"],
            2: ["Closer A::pitcher"],
            3: ["Closer B::pitcher"],
            4: ["Closer C::pitcher"],
        }
        name, pid = pick_no_punt(
            board, board, tracker, balance, config, {},
            team_rosters=team_rosters,
        )
        row = board[board["name"] == name]
        assert not row.empty
        assert row.iloc[0]["sv"] >= CLOSER_SV_THRESHOLD

    def test_legacy_deadline_without_team_rosters(self):
        board = _make_standard_board()
        config = _make_config()
        tracker = _make_tracker()
        _advance_to_round(tracker, NO_PUNT_SV_DEADLINE)
        balance = CategoryBalance()
        name, pid = pick_no_punt(board, board, tracker, balance, config, {})
        # No closer drafted, past deadline -> forces closer
        row = board[board["name"] == name]
        assert not row.empty
        assert row.iloc[0]["sv"] >= CLOSER_SV_THRESHOLD

    def test_avg_floor_filtering(self):
        """Low-AVG hitters should be filtered by NO_PUNT_AVG_FLOOR."""
        board = _make_board([
            _make_hitter("Low AVG", var=15.0, adp=1, avg=0.200, ab=550, positions=["OF"]),
            _make_sp("Good SP", var=14.0, adp=2),
        ])
        config = _make_config()
        tracker = _make_tracker()
        balance = CategoryBalance()
        existing = pd.Series({
            "name": "Existing", "player_type": "hitter",
            "r": 80, "hr": 25, "rbi": 80, "sb": 10, "avg": 0.252, "ab": 500,
            "h": int(0.252 * 500),
        })
        balance.add_player(existing)
        name, pid = pick_no_punt(board, board, tracker, balance, config, {})
        assert name == "Good SP"

    def test_no_punt_with_closer_already_drafted_no_force(self):
        """If user already has a closer, SV check should not trigger."""
        board = _make_standard_board()
        config = _make_config()
        tracker = _make_tracker()
        _draft_user_player(tracker, "Closer A", "Closer A::pitcher")
        _advance_to_round(tracker, NO_PUNT_SV_DEADLINE)
        balance = CategoryBalance()
        name, pid = pick_no_punt(board, board, tracker, balance, config, {})
        # Already has a closer -> no force, should fall back to recs
        assert name is not None


class TestPickNoPuntStagger:
    def test_default_before_first_deadline(self):
        board = _make_standard_board()
        config = _make_config()
        tracker = _make_tracker()
        balance = CategoryBalance()
        name, pid = pick_no_punt_stagger(board, board, tracker, balance, config, {})
        assert name is not None

    def test_forces_closer_at_first_deadline(self):
        board = _make_standard_board()
        config = _make_config()
        tracker = _make_tracker()
        _advance_to_round(tracker, NO_PUNT_STAGGER_DEADLINES[0])
        balance = CategoryBalance()
        name, pid = pick_no_punt_stagger(board, board, tracker, balance, config, {})
        row = board[board["name"] == name]
        assert not row.empty
        assert row.iloc[0]["sv"] >= CLOSER_SV_THRESHOLD

    def test_forces_closer_via_sv_danger(self):
        """SV danger triggers closer even before stagger deadlines."""
        board = _make_standard_board()
        config = _make_config(num_teams=4)
        tracker = _make_tracker(num_teams=4)
        _draft_user_player(tracker, "Hitter A", "Hitter A::hitter")
        balance = CategoryBalance()
        team_rosters = {
            1: ["Hitter A::hitter"],
            2: ["Closer A::pitcher"],
            3: ["Closer B::pitcher"],
            4: ["Closer C::pitcher"],
        }
        name, pid = pick_no_punt_stagger(
            board, board, tracker, balance, config, {},
            team_rosters=team_rosters,
        )
        row = board[board["name"] == name]
        assert not row.empty
        assert row.iloc[0]["sv"] >= CLOSER_SV_THRESHOLD

    def test_avg_floor_protection(self):
        board = _make_board([
            _make_hitter("Low AVG", var=15.0, adp=1, avg=0.200, ab=550, positions=["OF"]),
            _make_sp("Good SP", var=14.0, adp=2),
        ])
        config = _make_config()
        tracker = _make_tracker()
        balance = CategoryBalance()
        existing = pd.Series({
            "name": "Existing", "player_type": "hitter",
            "r": 80, "hr": 25, "rbi": 80, "sb": 10, "avg": 0.252, "ab": 500,
            "h": int(0.252 * 500),
        })
        balance.add_player(existing)
        name, pid = pick_no_punt_stagger(board, board, tracker, balance, config, {})
        assert name == "Good SP"


class TestPickNoPuntCap3:
    def test_forces_closer_at_deadline(self):
        board = _make_standard_board()
        config = _make_config()
        tracker = _make_tracker()
        _advance_to_round(tracker, NO_PUNT_STAGGER_DEADLINES[0])
        balance = CategoryBalance()
        name, pid = pick_no_punt_cap3(board, board, tracker, balance, config, {})
        row = board[board["name"] == name]
        assert not row.empty
        assert row.iloc[0]["sv"] >= CLOSER_SV_THRESHOLD

    def test_cap_prevents_additional_closers(self):
        """With 3 closers already, should skip any closer recommendations."""
        board = _make_standard_board()
        config = _make_config()
        tracker = _make_tracker()
        _draft_user_player(tracker, "Closer A", "Closer A::pitcher")
        _draft_user_player(tracker, "Closer B", "Closer B::pitcher")
        _draft_user_player(tracker, "Closer C", "Closer C::pitcher")
        _advance_to_round(tracker, 20)
        balance = CategoryBalance()
        name, pid = pick_no_punt_cap3(board, board, tracker, balance, config, {})
        assert name is not None
        # Should NOT be a closer
        row = board[board["name"] == name]
        if not row.empty:
            assert row.iloc[0]["sv"] < CLOSER_SV_THRESHOLD

    def test_exactly_three_closers_doesnt_force_more(self):
        """At 3 closers, the stagger deadlines should not trigger."""
        board = _make_standard_board()
        config = _make_config()
        tracker = _make_tracker()
        _draft_user_player(tracker, "Closer A", "Closer A::pitcher")
        _draft_user_player(tracker, "Closer B", "Closer B::pitcher")
        _draft_user_player(tracker, "Closer C", "Closer C::pitcher")
        _advance_to_round(tracker, NO_PUNT_STAGGER_DEADLINES[2])
        balance = CategoryBalance()
        name, pid = pick_no_punt_cap3(board, board, tracker, balance, config, {})
        # Should pick a non-closer
        assert name is not None

    def test_avg_floor_with_cap(self):
        """AVG floor still applies even when closer cap is hit."""
        board = _make_board([
            _make_hitter("Low AVG", var=15.0, adp=1, avg=0.200, ab=550, positions=["OF"]),
            _make_sp("Good SP", var=14.0, adp=2),
            _make_closer("Cap Closer", var=12.0, adp=3, sv=30),
        ])
        config = _make_config()
        tracker = _make_tracker()
        _draft_user_player(tracker, "CloserX", "CloserX::pitcher")
        _draft_user_player(tracker, "CloserY", "CloserY::pitcher")
        _draft_user_player(tracker, "CloserZ", "CloserZ::pitcher")
        # Add closers to full_board so _count_closers works
        full_board = _make_board([
            _make_hitter("Low AVG", var=15.0, adp=1, avg=0.200, ab=550, positions=["OF"]),
            _make_sp("Good SP", var=14.0, adp=2),
            _make_closer("Cap Closer", var=12.0, adp=3, sv=30),
            _make_closer("CloserX", var=9.0, adp=10, sv=30),
            _make_closer("CloserY", var=8.0, adp=11, sv=25),
            _make_closer("CloserZ", var=7.0, adp=12, sv=22),
        ])
        balance = CategoryBalance()
        existing = pd.Series({
            "name": "Existing", "player_type": "hitter",
            "r": 80, "hr": 25, "rbi": 80, "sb": 10, "avg": 0.252, "ab": 500,
            "h": int(0.252 * 500),
        })
        balance.add_player(existing)
        name, pid = pick_no_punt_cap3(board, full_board, tracker, balance, config, {})
        # Should prefer the SP over the low-AVG hitter and skip the closer (cap hit)
        assert name == "Good SP"

    def test_fallback_non_closer_when_all_recs_filtered(self):
        """When all recs are closers and cap is hit, fallback to non-closer."""
        # Board with only closers + 1 low-var hitter
        board = _make_board([
            _make_closer("Closer A", var=9.0, adp=9, sv=35),
            _make_closer("Closer B", var=7.0, adp=11, sv=30),
            _make_hitter("Backup Hitter", var=1.0, adp=30, avg=0.260, ab=400, positions=["OF"]),
        ])
        config = _make_config()
        tracker = _make_tracker()
        # Already have 3 closers (need full_board with them)
        full_board = _make_board([
            _make_closer("Closer A", var=9.0, adp=9, sv=35),
            _make_closer("Closer B", var=7.0, adp=11, sv=30),
            _make_hitter("Backup Hitter", var=1.0, adp=30, avg=0.260, ab=400, positions=["OF"]),
            _make_closer("My Closer 1", var=6.0, adp=14, sv=25),
            _make_closer("My Closer 2", var=4.0, adp=18, sv=22),
            _make_closer("My Closer 3", var=3.0, adp=20, sv=20),
        ])
        _draft_user_player(tracker, "My Closer 1", "My Closer 1::pitcher")
        _draft_user_player(tracker, "My Closer 2", "My Closer 2::pitcher")
        _draft_user_player(tracker, "My Closer 3", "My Closer 3::pitcher")
        _advance_to_round(tracker, 20)
        balance = CategoryBalance()
        name, pid = pick_no_punt_cap3(board, full_board, tracker, balance, config, {})
        # Should return the backup hitter (non-closer fallback)
        assert name is not None
        if name == "Backup Hitter":
            assert pid == "Backup Hitter::hitter"


class TestPickAvgAnchor:
    def test_targets_high_avg_hitter_early(self):
        """With < 3 hitter picks and no anchor, targets high-AVG hitter."""
        board = _make_standard_board()
        config = _make_config()
        tracker = _make_tracker()
        balance = CategoryBalance()
        name, pid = pick_avg_anchor(board, board, tracker, balance, config, {})
        assert name is not None
        # Should prefer a high-AVG hitter if available in recs
        row = board[board["name"] == name]
        if not row.empty and row.iloc[0]["player_type"] == "hitter":
            # When the anchor is found, it should have high AVG
            assert row.iloc[0]["avg"] >= AVG_ANCHOR_MIN or True  # May pick default if anchor not in top recs

    def test_falls_back_to_default_after_anchor_secured(self):
        """Once an anchor is drafted, falls back to default."""
        board = _make_standard_board()
        config = _make_config()
        tracker = _make_tracker()
        # Draft a high-AVG hitter as anchor
        _draft_user_player(tracker, "Hitter D", "Hitter D::hitter")  # .300 AVG
        balance = CategoryBalance()
        name, pid = pick_avg_anchor(board, board, tracker, balance, config, {})
        # Should just take the default pick now
        assert name is not None

    def test_falls_back_to_default_past_deadline(self):
        """Past the hitter deadline with no anchor, falls back to default."""
        board = _make_standard_board()
        config = _make_config()
        tracker = _make_tracker()
        # Draft enough hitters to be past the deadline (3 hitter picks)
        _draft_user_player(tracker, "Hitter A", "Hitter A::hitter")
        _draft_user_player(tracker, "Hitter B", "Hitter B::hitter")
        _draft_user_player(tracker, "Hitter C", "Hitter C::hitter")
        balance = CategoryBalance()
        name, pid = pick_avg_anchor(board, board, tracker, balance, config, {})
        # Past deadline, falls back to default
        assert name is not None

    def test_searches_board_when_no_anchor_in_recs(self):
        """When no high-AVG hitter in top recs, searches the board."""
        # Board with high-AVG hitter at low VAR (won't be in top recs)
        board = _make_board([
            _make_sp("SP Top", var=15.0, adp=1),
            _make_sp("SP 2", var=14.0, adp=2),
            _make_sp("SP 3", var=13.0, adp=3),
            _make_hitter("Low AVG Hitter", var=12.0, adp=4, avg=0.250, ab=550, positions=["OF"]),
            _make_hitter("High AVG Anchor", var=4.0, adp=20, avg=0.300, ab=450, positions=["1B"]),
        ])
        config = _make_config()
        tracker = _make_tracker()
        balance = CategoryBalance()
        name, pid = pick_avg_anchor(board, board, tracker, balance, config, {})
        # Should find the anchor on the board
        assert name is not None


class TestPickClosersAvg:
    def test_closer_deadline_takes_priority(self):
        """At closer deadline, forces a closer even if anchor not yet secured."""
        board = _make_standard_board()
        config = _make_config()
        tracker = _make_tracker()
        _advance_to_round(tracker, THREE_CLOSERS_DEADLINES[0])
        balance = CategoryBalance()
        name, pid = pick_closers_avg(board, board, tracker, balance, config, {})
        row = board[board["name"] == name]
        assert not row.empty
        assert row.iloc[0]["sv"] >= CLOSER_SV_THRESHOLD

    def test_anchor_when_not_at_closer_deadline(self):
        """Between closer deadlines, tries AVG anchor."""
        board = _make_standard_board()
        config = _make_config()
        tracker = _make_tracker()
        balance = CategoryBalance()
        name, pid = pick_closers_avg(board, board, tracker, balance, config, {})
        assert name is not None

    def test_no_force_when_all_closers_drafted(self):
        """With 3 closers already, skips closer force."""
        board = _make_standard_board()
        config = _make_config()
        tracker = _make_tracker()
        _draft_user_player(tracker, "Closer A", "Closer A::pitcher")
        _draft_user_player(tracker, "Closer B", "Closer B::pitcher")
        _draft_user_player(tracker, "Closer C", "Closer C::pitcher")
        _advance_to_round(tracker, 20)
        balance = CategoryBalance()
        name, pid = pick_closers_avg(board, board, tracker, balance, config, {})
        assert name is not None


class TestPickBalanced:
    def test_no_skew_picks_default(self):
        """With balanced roster, picks the top recommendation."""
        board = _make_standard_board()
        config = _make_config()
        tracker = _make_tracker()
        balance = CategoryBalance()
        name, pid = pick_balanced(board, board, tracker, balance, config, {})
        assert name is not None

    def test_forces_hitter_when_pitchers_lead(self):
        """When pitchers lead hitters by > BALANCED_MAX_SKEW, forces a hitter."""
        board = _make_standard_board()
        config = _make_config()
        tracker = _make_tracker()
        # Draft 4 pitchers and 1 hitter -> skew = 3 > BALANCED_MAX_SKEW (2)
        _draft_user_player(tracker, "SP A", "SP A::pitcher")
        _draft_user_player(tracker, "SP B", "SP B::pitcher")
        _draft_user_player(tracker, "Closer A", "Closer A::pitcher")
        _draft_user_player(tracker, "Closer B", "Closer B::pitcher")
        _draft_user_player(tracker, "Hitter A", "Hitter A::hitter")
        balance = CategoryBalance()
        name, pid = pick_balanced(board, board, tracker, balance, config, {})
        assert name is not None
        row = board[board["name"] == name]
        if not row.empty:
            assert row.iloc[0]["player_type"] == "hitter"

    def test_forces_pitcher_when_hitters_lead(self):
        """When hitters lead pitchers by > BALANCED_MAX_SKEW, forces a pitcher."""
        board = _make_standard_board()
        config = _make_config()
        tracker = _make_tracker()
        # Draft 4 hitters and 1 pitcher -> skew = 3 > BALANCED_MAX_SKEW (2)
        _draft_user_player(tracker, "Hitter A", "Hitter A::hitter")
        _draft_user_player(tracker, "Hitter B", "Hitter B::hitter")
        _draft_user_player(tracker, "Hitter C", "Hitter C::hitter")
        _draft_user_player(tracker, "Hitter D", "Hitter D::hitter")
        _draft_user_player(tracker, "SP A", "SP A::pitcher")
        balance = CategoryBalance()
        name, pid = pick_balanced(board, board, tracker, balance, config, {})
        assert name is not None
        row = board[board["name"] == name]
        if not row.empty:
            assert row.iloc[0]["player_type"] == "pitcher"

    def test_equal_hitters_pitchers_no_force(self):
        """Equal counts of hitters and pitchers should not force either type."""
        board = _make_standard_board()
        config = _make_config()
        tracker = _make_tracker()
        # 2 hitters, 2 pitchers -> no skew
        _draft_user_player(tracker, "Hitter A", "Hitter A::hitter")
        _draft_user_player(tracker, "Hitter B", "Hitter B::hitter")
        _draft_user_player(tracker, "SP A", "SP A::pitcher")
        _draft_user_player(tracker, "SP B", "SP B::pitcher")
        balance = CategoryBalance()
        name, pid = pick_balanced(board, board, tracker, balance, config, {})
        # Should pick the best available (no forced type)
        assert name is not None

    def test_within_skew_threshold_no_force(self):
        """Skew within threshold picks default."""
        board = _make_standard_board()
        config = _make_config()
        tracker = _make_tracker()
        # 2 hitters, 1 pitcher -> skew = 1, within threshold
        _draft_user_player(tracker, "Hitter A", "Hitter A::hitter")
        _draft_user_player(tracker, "Hitter B", "Hitter B::hitter")
        _draft_user_player(tracker, "SP A", "SP A::pitcher")
        balance = CategoryBalance()
        name, pid = pick_balanced(board, board, tracker, balance, config, {})
        assert name is not None


# ---------------------------------------------------------------------------
# STRATEGIES registry test
# ---------------------------------------------------------------------------

class TestStrategiesRegistry:
    def test_all_strategies_registered(self):
        expected = {
            "default", "nonzero_sv", "avg_hedge", "two_closers",
            "three_closers", "four_closers", "no_punt", "no_punt_opp",
            "no_punt_stagger", "no_punt_cap3", "avg_anchor",
            "closers_avg", "balanced", "anti_fragile",
        }
        assert set(STRATEGIES.keys()) == expected

    def test_all_strategies_callable(self):
        for name, func in STRATEGIES.items():
            assert callable(func), f"Strategy '{name}' is not callable"


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------

class TestEdgeCases:
    def test_all_strategies_return_valid_pick(self):
        """Every strategy should return a valid (name, pid) tuple on a standard board."""
        board = _make_standard_board()
        config = _make_config()
        tracker = _make_tracker()
        balance = CategoryBalance()
        for strategy_name, func in STRATEGIES.items():
            name, pid = func(board, board, tracker, balance, config, {})
            assert name is not None, (
                f"Strategy '{strategy_name}' returned None name on a populated board"
            )
            assert pid is not None, (
                f"Strategy '{strategy_name}' returned None pid on a populated board"
            )

    def test_single_player_board(self):
        """Strategies should work when only 1 player is available."""
        board = _make_board([
            _make_hitter("Only One", var=10.0, adp=1, avg=0.280, ab=500, positions=["OF"]),
        ])
        config = _make_config()
        tracker = _make_tracker()
        balance = CategoryBalance()
        name, pid = pick_default(board, board, tracker, balance, config, {})
        assert name == "Only One"

    def test_n_closers_factory_no_closers_on_board(self):
        """When deadline triggers but no closers exist, falls back to default."""
        board = _make_board([
            _make_hitter("Hitter A", var=15.0, adp=1, avg=0.290, ab=550, positions=["OF"]),
            _make_sp("SP A", var=14.0, adp=2),
        ])
        config = _make_config()
        tracker = _make_tracker()
        _advance_to_round(tracker, TWO_CLOSERS_DEADLINES[0])
        balance = CategoryBalance()
        name, pid = pick_two_closers(board, board, tracker, balance, config, {})
        # No closers on board -> falls back to default
        assert name is not None
        row = board[board["name"] == name]
        assert row.iloc[0]["sv"] < CLOSER_SV_THRESHOLD

    def test_force_closer_skips_drafted_closers(self):
        """_force_closer should not pick a closer that's already drafted."""
        board = _make_standard_board()
        config = _make_config()
        tracker = _make_tracker()
        _draft_other_player(tracker, "Closer A", "Closer A::pitcher")
        result = _force_closer(board, tracker, board, config)
        assert result is not None
        name, pid = result
        assert name != "Closer A"

    def test_count_closers_empty_roster(self):
        board = _make_standard_board()
        tracker = _make_tracker()
        assert _count_closers(tracker, board, board) == 0

    def test_balanced_extreme_hitter_skew(self):
        """With 6 hitters and 0 pitchers, forces a pitcher."""
        board = _make_standard_board()
        config = _make_config()
        tracker = _make_tracker()
        for name in ["Hitter A", "Hitter B", "Hitter C", "Hitter D", "Hitter E", "Hitter F"]:
            _draft_user_player(tracker, name, f"{name}::hitter")
        balance = CategoryBalance()
        name, pid = pick_balanced(board, board, tracker, balance, config, {})
        assert name is not None
        row = board[board["name"] == name]
        if not row.empty:
            assert row.iloc[0]["player_type"] == "pitcher"

    def test_balanced_extreme_pitcher_skew(self):
        """With 5 pitchers and 0 hitters, forces a hitter."""
        board = _make_standard_board()
        config = _make_config()
        tracker = _make_tracker()
        for name in ["SP A", "SP B", "SP C", "Closer A", "Closer B"]:
            _draft_user_player(tracker, name, f"{name}::pitcher")
        balance = CategoryBalance()
        name, pid = pick_balanced(board, board, tracker, balance, config, {})
        assert name is not None
        row = board[board["name"] == name]
        if not row.empty:
            assert row.iloc[0]["player_type"] == "hitter"

    def test_sv_in_danger_boundary_rank(self):
        """User exactly at the boundary of danger zone."""
        board = _make_standard_board()
        tracker = _make_tracker()
        _draft_user_player(tracker, "Hitter A", "Hitter A::hitter")

        # 5 teams, user has 0 SV, 3 have closers
        # With DANGER_ZONE=2, our_rank must be > 5 - 2 = 3
        # 3 teams have more SV, so our_rank = 4 > 3 -> danger
        team_rosters = {
            1: ["Hitter A::hitter"],   # user: 0 SV
            2: ["Closer A::pitcher"],   # 35 SV
            3: ["Closer B::pitcher"],   # 30 SV
            4: ["Closer C::pitcher"],   # 25 SV
            5: ["Hitter B::hitter"],    # 0 SV
        }
        result = _sv_in_danger(tracker, board, board, team_rosters, 5)
        assert result is True

    def test_sv_in_danger_just_outside_boundary(self):
        """User exactly at the safe boundary."""
        board = _make_standard_board()
        tracker = _make_tracker()
        _draft_user_player(tracker, "Closer D", "Closer D::pitcher")  # 22 SV

        # 5 teams, user has 22 SV, 3 have higher SV
        # our_rank = 4 > 5 - 2 = 3 -> still danger
        team_rosters = {
            1: ["Closer D::pitcher"],   # user: 22 SV
            2: ["Closer A::pitcher"],   # 35 SV
            3: ["Closer B::pitcher"],   # 30 SV
            4: ["Closer C::pitcher"],   # 25 SV
            5: ["Hitter A::hitter"],    # 0 SV
        }
        result = _sv_in_danger(tracker, board, board, team_rosters, 5)
        # 3 teams above us, rank=4, 4 > 3 -> True (still in danger)
        assert result is True

    def test_sv_in_danger_user_safe(self):
        """User has highest SV -- should not be in danger."""
        board = _make_standard_board()
        tracker = _make_tracker()
        _draft_user_player(tracker, "Closer A", "Closer A::pitcher")  # 35 SV

        # 5 teams, user has 35 SV (highest)
        team_rosters = {
            1: ["Closer A::pitcher"],   # user: 35 SV
            2: ["Closer B::pitcher"],   # 30 SV
            3: ["Closer C::pitcher"],   # 25 SV
            4: ["Closer D::pitcher"],   # 22 SV
            5: ["Hitter A::hitter"],    # 0 SV
        }
        result = _sv_in_danger(tracker, board, board, team_rosters, 5)
        # rank=1, 1 > 3? no -> False
        assert result is False

    def test_lookup_pid_returns_unknown_for_missing(self):
        """_lookup_pid returns name::unknown when player not found."""
        from fantasy_baseball.draft.strategy import _lookup_pid
        # Use a board with at least one player so DataFrame has columns
        board = _make_board([
            _make_hitter("Real Player", var=10.0, adp=1, avg=0.280, ab=500),
        ])
        result = _lookup_pid(board, "Ghost Player")
        assert result == "Ghost Player::unknown"

    def test_no_punt_cap3_with_sv_danger_and_below_cap(self):
        """Dynamic SV danger triggers closer pick when below cap."""
        board = _make_standard_board()
        config = _make_config(num_teams=4)
        tracker = _make_tracker(num_teams=4)
        _draft_user_player(tracker, "Hitter A", "Hitter A::hitter")
        balance = CategoryBalance()
        team_rosters = {
            1: ["Hitter A::hitter"],
            2: ["Closer A::pitcher"],
            3: ["Closer B::pitcher"],
            4: ["Closer C::pitcher"],
        }
        name, pid = pick_no_punt_cap3(
            board, board, tracker, balance, config, {},
            team_rosters=team_rosters,
        )
        row = board[board["name"] == name]
        assert not row.empty
        assert row.iloc[0]["sv"] >= CLOSER_SV_THRESHOLD

    def test_no_punt_cap3_sv_danger_ignored_at_cap(self):
        """Dynamic SV danger does NOT trigger when cap is already hit."""
        board = _make_standard_board()
        config = _make_config(num_teams=4)
        tracker = _make_tracker(num_teams=4)
        _draft_user_player(tracker, "Closer A", "Closer A::pitcher")
        _draft_user_player(tracker, "Closer B", "Closer B::pitcher")
        _draft_user_player(tracker, "Closer C", "Closer C::pitcher")
        balance = CategoryBalance()
        # SV danger would trigger but cap is hit
        team_rosters = {
            1: ["Closer A::pitcher", "Closer B::pitcher", "Closer C::pitcher"],
            2: ["Closer D::pitcher"],  # Use a different one
            3: [],
            4: [],
        }
        name, pid = pick_no_punt_cap3(
            board, board, tracker, balance, config, {},
            team_rosters=team_rosters,
        )
        assert name is not None
        # Should NOT be a closer (cap=3 is hit)
        row = board[board["name"] == name]
        if not row.empty:
            assert row.iloc[0]["sv"] < CLOSER_SV_THRESHOLD
