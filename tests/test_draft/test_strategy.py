"""Unit tests for surviving strategy helpers in strategy.py.

The legacy pick_* functions were removed when STRATEGIES was aliased to OVERLAYS.
This file covers the board-level helpers that the overlays depend on:
_count_closers and select_from_ranked.

Overlay behavior is tested in test_strategy_overlays.py.
"""

from typing import ClassVar

import pandas as pd

from fantasy_baseball.draft.strategy import (
    STRATEGIES,
    _count_closers,
)
from fantasy_baseball.draft.tracker import DraftTracker

# ---------------------------------------------------------------------------
# Fixtures and helpers
# ---------------------------------------------------------------------------


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
    def test_count_closers_empty_roster(self):
        board = _make_standard_board()
        tracker = _make_tracker()
        assert _count_closers(tracker, board, board) == 0


def test_fillers_or_all_gates_recs_to_open_starters():
    """fillers_or_all restricts a ranked list to candidates that fill an open
    STARTER slot, so the dashboard /api/recs gate stops recommending pitchers
    once a team's pitching slots are full. Falls back to the full list when no
    starter slots are open (or none of the ranked items can fill one).
    """
    from types import SimpleNamespace

    from fantasy_baseball.draft.strategy import fillers_or_all
    from fantasy_baseball.models.positions import Position

    pitcher = SimpleNamespace(positions=[Position.P])
    outfielder = SimpleNamespace(positions=[Position.OF])
    catcher = SimpleNamespace(positions=[Position.C])
    ranked = [pitcher, outfielder, catcher]

    # P full; C/OF/UTIL open -> the pitcher is gated out (can't start a 10th arm).
    open_starters = {Position.C, Position.OF, Position.UTIL}
    assert fillers_or_all(ranked, open_starters) == [outfielder, catcher]

    # No open starter slots -> don't hide anything (return the full list).
    assert fillers_or_all(ranked, set()) == ranked

    # Open slot exists but nothing in the list fills it -> fall back to all.
    assert fillers_or_all([pitcher], {Position.C}) == [pitcher]
