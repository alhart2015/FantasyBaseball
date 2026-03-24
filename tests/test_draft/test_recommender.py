import pytest
from unittest.mock import patch
import pandas as pd
from fantasy_baseball.draft.recommender import get_recommendations, calculate_vona_scores, _vona_leverage_weight
from fantasy_baseball.utils.constants import ALL_CATEGORIES


def _make_board():
    players = [
        {"name": "Player A", "var": 15.0, "total_sgp": 20.0, "best_position": "OF", "positions": ["OF"],
         "player_type": "hitter", "player_id": "Player A::hitter",
         "r": 100, "hr": 35, "rbi": 100, "sb": 20, "avg": .280, "ab": 550, "h": 154,
         "adp": 1},
        {"name": "Player B", "var": 14.0, "total_sgp": 19.0, "best_position": "SS", "positions": ["SS"],
         "player_type": "hitter", "player_id": "Player B::hitter",
         "r": 90, "hr": 25, "rbi": 80, "sb": 30, "avg": .275, "ab": 530, "h": 146,
         "adp": 2},
        {"name": "Player C", "var": 13.0, "total_sgp": 18.0, "best_position": "C", "positions": ["C"],
         "player_type": "hitter", "player_id": "Player C::hitter",
         "r": 70, "hr": 22, "rbi": 75, "sb": 2, "avg": .260, "ab": 480, "h": 125,
         "adp": 3},
        {"name": "Player D", "var": 12.0, "total_sgp": 17.0, "best_position": "P", "positions": ["SP"],
         "player_type": "pitcher", "player_id": "Player D::pitcher",
         "w": 14, "k": 210, "sv": 0, "era": 3.20, "whip": 1.10, "ip": 195,
         "er": 69, "bb": 50, "h_allowed": 165,
         "adp": 4},
        {"name": "Player E", "var": 11.0, "total_sgp": 16.0, "best_position": "OF", "positions": ["OF"],
         "player_type": "hitter", "player_id": "Player E::hitter",
         "r": 80, "hr": 20, "rbi": 70, "sb": 15, "avg": .270, "ab": 500, "h": 135,
         "adp": 5},
    ]
    return pd.DataFrame(players)


class TestGetRecommendations:
    def test_returns_top_n(self):
        board = _make_board()
        recs = get_recommendations(board, drafted=[], user_roster=[], n=3)
        assert len(recs) == 3

    def test_excludes_drafted_players(self):
        board = _make_board()
        recs = get_recommendations(board, drafted=["Player A::hitter"], user_roster=[], n=3)
        names = [r["name"] for r in recs]
        assert "Player A" not in names

    def test_recommendations_sorted_by_var(self):
        board = _make_board()
        recs = get_recommendations(board, drafted=[], user_roster=[], n=5)
        vars_list = [r["var"] for r in recs]
        assert vars_list == sorted(vars_list, reverse=True)

    def test_flags_positional_need(self):
        board = _make_board()
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


class TestVonaPicksUntilNext:
    """Verify that picks_until_next is passed through to calculate_vona_scores."""

    def test_vona_receives_picks_until_next(self):
        """When picks_until_next is passed, calculate_vona_scores uses it."""
        board = _make_board()
        with patch(
            "fantasy_baseball.draft.recommender.calculate_vona_scores",
            wraps=calculate_vona_scores,
        ) as mock_vona:
            get_recommendations(
                board, drafted=[], user_roster=[], n=3,
                scoring_mode="vona", picks_until_next=4,
            )
            mock_vona.assert_called_once()
            _, kwargs = mock_vona.call_args
            # picks_until_next should be passed as positional arg #2
            args = mock_vona.call_args[0]
            assert args[1] == 4

    def test_vona_default_when_none(self):
        """When picks_until_next is None, VONA falls back to default of 10."""
        board = _make_board()
        with patch(
            "fantasy_baseball.draft.recommender.calculate_vona_scores",
            wraps=calculate_vona_scores,
        ) as mock_vona:
            get_recommendations(
                board, drafted=[], user_roster=[], n=3,
                scoring_mode="vona", picks_until_next=None,
            )
            mock_vona.assert_called_once()
            args = mock_vona.call_args[0]
            assert args[1] is None  # None passed through; default applied inside

    def test_vona_not_called_in_var_mode(self):
        """In VAR scoring mode, calculate_vona_scores should not be called."""
        board = _make_board()
        with patch(
            "fantasy_baseball.draft.recommender.calculate_vona_scores",
        ) as mock_vona:
            get_recommendations(
                board, drafted=[], user_roster=[], n=3,
                scoring_mode="var", picks_until_next=4,
            )
            mock_vona.assert_not_called()

    def test_calculate_vona_scores_different_picks(self):
        """VONA scores differ based on picks_until_next value."""
        board = _make_board()
        scores_4 = calculate_vona_scores(board, picks_until_next=4)
        scores_14 = calculate_vona_scores(board, picks_until_next=14)
        # With only 5 players, picks_until_next=14 removes all of them,
        # so all VONA scores should be non-negative (nothing better remaining).
        # With picks_until_next=4, the 5th player remains, providing a
        # comparison point. The scores should differ.
        # At minimum, the set of scores should not be identical.
        assert scores_4 != scores_14

    def test_calculate_vona_scores_respects_picks_count(self):
        """With picks_until_next=2, only 2 players are removed from the pool."""
        board = _make_board()
        scores = calculate_vona_scores(board, picks_until_next=2)
        # With 5 hitters+pitchers and 2 removed (top 2 by ADP),
        # the remaining pool has 3 players. The top-ADP player (Player A)
        # should have positive VONA since they'd be gone if not picked now.
        assert scores["Player A::hitter"] > 0


def _make_closer(name, sv=35, w=3, k=70, era=2.50, whip=1.00, ip=65):
    """Helper to build a closer pd.Series for _vona_leverage_weight tests."""
    return pd.Series({
        "name": name, "player_type": "pitcher",
        "w": w, "k": k, "sv": sv, "era": era, "whip": whip, "ip": ip,
    })


def _make_sp(name, w=14, k=210, sv=0, era=3.20, whip=1.10, ip=195):
    """Helper to build a starting pitcher pd.Series."""
    return pd.Series({
        "name": name, "player_type": "pitcher",
        "w": w, "k": k, "sv": sv, "era": era, "whip": whip, "ip": ip,
    })


def _make_test_hitter(name, r=90, hr=30, rbi=85, sb=15, avg=0.275, ab=530):
    """Helper to build a hitter pd.Series for _vona_leverage_weight tests."""
    return pd.Series({
        "name": name, "player_type": "hitter",
        "r": r, "hr": hr, "rbi": rbi, "sb": sb, "avg": avg, "ab": ab,
    })


def _uniform_leverage():
    """Leverage dict where all categories have equal weight."""
    return {cat: 1.0 / len(ALL_CATEGORIES) for cat in ALL_CATEGORIES}


class TestVonaLeverageWeight:
    """Tests for _vona_leverage_weight."""

    def test_closer_high_sv_leverage(self):
        """A closer when SV leverage is high should get a high multiplier."""
        closer = _make_closer("Clase", sv=35)
        # Heavily weight SV — team desperately needs saves
        leverage = {cat: 0.02 for cat in ALL_CATEGORIES}
        leverage["SV"] = 0.82  # dominating the leverage
        result = _vona_leverage_weight(closer, leverage)
        assert result > 1.3

    def test_closer_low_sv_leverage(self):
        """A closer when SV leverage is low should get a low multiplier."""
        closer = _make_closer("Clase", sv=35)
        # SV well-satisfied, other categories need help
        leverage = {cat: 0.11 for cat in ALL_CATEGORIES}
        leverage["SV"] = 0.01
        result = _vona_leverage_weight(closer, leverage)
        assert result < 0.7

    def test_hitter_boosted_when_hitting_needed(self):
        """A hitter when hitting categories are needed should get boosted."""
        hitter = _make_test_hitter("Judge", r=110, hr=45, rbi=120, sb=5, avg=0.291, ab=550)
        # Heavy hitting leverage, low pitching leverage
        leverage = {
            "R": 0.18, "HR": 0.18, "RBI": 0.18, "SB": 0.18, "AVG": 0.18,
            "W": 0.02, "K": 0.02, "ERA": 0.02, "WHIP": 0.02, "SV": 0.02,
        }
        result = _vona_leverage_weight(hitter, leverage)
        assert result > 1.3

    def test_pitcher_boosted_when_pitching_needed(self):
        """An SP when pitching categories are needed should get boosted."""
        sp = _make_sp("Cole", w=15, k=240, sv=0, era=3.00, whip=1.05, ip=200)
        # Heavy pitching leverage, low hitting leverage
        leverage = {
            "R": 0.02, "HR": 0.02, "RBI": 0.02, "SB": 0.02, "AVG": 0.02,
            "W": 0.18, "K": 0.18, "ERA": 0.18, "WHIP": 0.18, "SV": 0.18,
        }
        result = _vona_leverage_weight(sp, leverage)
        assert result > 1.3

    def test_uniform_leverage_gives_roughly_one(self):
        """When all categories are equally weighted, multiplier should be ~1.0."""
        hitter = _make_test_hitter("Betts")
        leverage = _uniform_leverage()
        result = _vona_leverage_weight(hitter, leverage)
        assert result == pytest.approx(1.0, abs=0.15)

    def test_uniform_leverage_closer_roughly_one(self):
        """Uniform leverage should also give ~1.0 for a closer."""
        closer = _make_closer("Clase")
        leverage = _uniform_leverage()
        result = _vona_leverage_weight(closer, leverage)
        assert result == pytest.approx(1.0, abs=0.15)

    def test_uniform_leverage_sp_roughly_one(self):
        """Uniform leverage should also give ~1.0 for an SP."""
        sp = _make_sp("Cole")
        leverage = _uniform_leverage()
        result = _vona_leverage_weight(sp, leverage)
        assert result == pytest.approx(1.0, abs=0.15)

    def test_empty_leverage_returns_one(self):
        """Empty leverage dict should return 1.0 gracefully."""
        hitter = _make_test_hitter("Judge")
        result = _vona_leverage_weight(hitter, {})
        assert result == 1.0

    def test_none_leverage_returns_one(self):
        """None leverage should return 1.0 gracefully."""
        hitter = _make_test_hitter("Judge")
        result = _vona_leverage_weight(hitter, None)
        assert result == 1.0

    def test_zero_contribution_player_returns_one(self):
        """A player with all-zero stats should return 1.0."""
        zero_hitter = pd.Series({
            "name": "Nobody", "player_type": "hitter",
            "r": 0, "hr": 0, "rbi": 0, "sb": 0, "avg": 0.250, "ab": 0,
        })
        leverage = _uniform_leverage()
        result = _vona_leverage_weight(zero_hitter, leverage)
        assert result == 1.0

    def test_result_is_positive(self):
        """Multiplier should always be positive regardless of leverage distribution."""
        closer = _make_closer("Clase", sv=35)
        # Extreme leverage: only hitting matters
        leverage = {
            "R": 0.20, "HR": 0.20, "RBI": 0.20, "SB": 0.20, "AVG": 0.20,
            "W": 0.0, "K": 0.0, "ERA": 0.0, "WHIP": 0.0, "SV": 0.0,
        }
        result = _vona_leverage_weight(closer, leverage)
        assert result >= 0
