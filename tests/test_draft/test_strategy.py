"""Unit tests for surviving strategy helpers in strategy.py.

The legacy pick_* functions were removed when STRATEGIES was aliased to OVERLAYS.
This file covers the board-level helpers that the overlays depend on:
_count_closers, _count_hitters, _count_pitchers, _sv_in_danger,
_force_closer, _fallback_non_closer, _lookup_pid, and select_from_ranked.

Overlay behavior is tested in test_strategy_overlays.py.
"""

from typing import ClassVar

import pandas as pd

from fantasy_baseball.config import LeagueConfig
from fantasy_baseball.draft.strategy import (
    STRATEGIES,
    _count_closers,
    _count_hitters,
    _count_pitchers,
    _fallback_non_closer,
    _force_closer,
    _sv_in_danger,
)
from fantasy_baseball.draft.tracker import DraftTracker
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
        return pd.DataFrame(
            columns=[
                "name",
                "player_id",
                "player_type",
                "var",
                "total_sgp",
                "best_position",
                "positions",
                "adp",
                "avg",
                "ab",
                "h",
                "r",
                "hr",
                "rbi",
                "sb",
                "sv",
                "ip",
                "w",
                "k",
                "era",
                "whip",
                "er",
                "bb",
                "h_allowed",
            ]
        )
    return pd.DataFrame(players)


def _make_standard_board():
    """Create a standard test board with a mix of hitters, SPs, closers."""
    return _make_board(
        [
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
        ]
    )


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

    def test_count_closers_handles_none_or_nan_sv(self):
        """_count_closers must not crash when a roster row's sv is None or NaN.

        Regression: row.get("sv", 0) on a pandas Series returns None when the
        index has the key with a null value -- NOT the default. None >= 20
        raises TypeError; NaN >= 20 silently returns False. Use _safe_float
        to guard against both.
        """
        import math

        board = pd.DataFrame(
            [
                {
                    "player_id": "h1",
                    "name": "Hitter",
                    "player_type": "hitter",
                    "sv": None,
                    "var": 5.0,
                },
                {
                    "player_id": "p1",
                    "name": "Closer",
                    "player_type": "pitcher",
                    "sv": 35,
                    "var": 4.0,
                },
                {
                    "player_id": "p2",
                    "name": "Pitcher Nan SV",
                    "player_type": "pitcher",
                    "sv": math.nan,
                    "var": 3.0,
                },
            ]
        )

        class _StubTracker:
            user_roster_ids: ClassVar[list[str]] = ["h1", "p1", "p2"]

        count = _count_closers(_StubTracker(), board, board)
        # Only p1 (sv=35) qualifies; h1 has sv=None, p2 has sv=NaN.
        assert count == 1


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
            4: ["Hitter A::hitter"],  # 0 SV
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
        config = _make_config(
            roster_slots={
                "P": 1,
                "C": 1,
                "1B": 1,
                "2B": 1,
                "3B": 1,
                "SS": 1,
                "OF": 4,
                "UTIL": 2,
                "IF": 1,
            }
        )
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
        name, _pid = _fallback_non_closer(board, tracker, board, config)
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


# ---------------------------------------------------------------------------
# STRATEGIES registry test (STRATEGIES = OVERLAYS alias)
# ---------------------------------------------------------------------------


class TestStrategiesRegistry:
    def test_all_strategies_registered(self):
        expected = {
            "default",
            "nonzero_sv",
            "avg_hedge",
            "two_closers",
            "three_closers",
            "four_closers",
            "no_punt",
            "no_punt_opp",
            "no_punt_stagger",
            "no_punt_cap3",
            "avg_anchor",
            "closers_avg",
            "balanced",
            "anti_fragile",
        }
        assert set(STRATEGIES.keys()) == expected

    def test_all_strategies_callable(self):
        for name, func in STRATEGIES.items():
            assert callable(func), f"Strategy '{name}' is not callable"


# ---------------------------------------------------------------------------
# Edge cases for surviving helpers
# ---------------------------------------------------------------------------


class TestEdgeCases:
    def test_force_closer_skips_drafted_closers(self):
        """_force_closer should not pick a closer that's already drafted."""
        board = _make_standard_board()
        config = _make_config()
        tracker = _make_tracker()
        _draft_other_player(tracker, "Closer A", "Closer A::pitcher")
        result = _force_closer(board, tracker, board, config)
        assert result is not None
        name, _pid = result
        assert name != "Closer A"

    def test_count_closers_empty_roster(self):
        board = _make_standard_board()
        tracker = _make_tracker()
        assert _count_closers(tracker, board, board) == 0

    def test_sv_in_danger_boundary_rank(self):
        """User exactly at the boundary of danger zone."""
        board = _make_standard_board()
        tracker = _make_tracker()
        _draft_user_player(tracker, "Hitter A", "Hitter A::hitter")

        # 5 teams, user has 0 SV, 3 have closers
        # With DANGER_ZONE=2, our_rank must be > 5 - 2 = 3
        # 3 teams have more SV, so our_rank = 4 > 3 -> danger
        team_rosters = {
            1: ["Hitter A::hitter"],  # user: 0 SV
            2: ["Closer A::pitcher"],  # 35 SV
            3: ["Closer B::pitcher"],  # 30 SV
            4: ["Closer C::pitcher"],  # 25 SV
            5: ["Hitter B::hitter"],  # 0 SV
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
            1: ["Closer D::pitcher"],  # user: 22 SV
            2: ["Closer A::pitcher"],  # 35 SV
            3: ["Closer B::pitcher"],  # 30 SV
            4: ["Closer C::pitcher"],  # 25 SV
            5: ["Hitter A::hitter"],  # 0 SV
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
            1: ["Closer A::pitcher"],  # user: 35 SV
            2: ["Closer B::pitcher"],  # 30 SV
            3: ["Closer C::pitcher"],  # 25 SV
            4: ["Closer D::pitcher"],  # 22 SV
            5: ["Hitter A::hitter"],  # 0 SV
        }
        result = _sv_in_danger(tracker, board, board, team_rosters, 5)
        # rank=1, 1 > 3? no -> False
        assert result is False

    def test_lookup_pid_returns_unknown_for_missing(self):
        """_lookup_pid returns name::unknown when player not found."""
        from fantasy_baseball.draft.strategy import _lookup_pid

        # Use a board with at least one player so DataFrame has columns
        board = _make_board(
            [
                _make_hitter("Real Player", var=10.0, adp=1, avg=0.280, ab=500),
            ]
        )
        result = _lookup_pid(board, "Ghost Player")
        assert result == "Ghost Player::unknown"
