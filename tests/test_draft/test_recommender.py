from unittest.mock import patch

import pandas as pd

from fantasy_baseball.draft.recommender import calculate_vona_scores, get_recommendations


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
        names = [r.name for r in recs]
        assert "Player A" not in names

    def test_recommendations_sorted_by_var(self):
        board = _make_board()
        recs = get_recommendations(board, drafted=[], user_roster=[], n=5)
        vars_list = [r.var for r in recs]
        assert vars_list == sorted(vars_list, reverse=True)

    def test_flags_positional_need(self):
        board = _make_board()
        filled = {"OF": 1}
        recs = get_recommendations(board, drafted=[], user_roster=[], n=5, filled_positions=filled)
        catcher_rec = next(r for r in recs if r.name == "Player C")
        assert catcher_rec.need_flag is True

    def test_includes_player_stats(self):
        board = _make_board()
        recs = get_recommendations(board, drafted=[], user_roster=[], n=1)
        assert recs[0].var is not None
        assert recs[0].best_position
        assert recs[0].name


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
            _, _kwargs = mock_vona.call_args
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


