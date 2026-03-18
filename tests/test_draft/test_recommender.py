import pytest
import pandas as pd
from fantasy_baseball.draft.recommender import get_recommendations


def _make_board():
    players = [
        {"name": "Player A", "var": 15.0, "total_sgp": 20.0, "best_position": "OF", "positions": ["OF"],
         "player_type": "hitter", "player_id": "Player A::hitter",
         "r": 100, "hr": 35, "rbi": 100, "sb": 20, "avg": .280, "ab": 550, "h": 154},
        {"name": "Player B", "var": 14.0, "total_sgp": 19.0, "best_position": "SS", "positions": ["SS"],
         "player_type": "hitter", "player_id": "Player B::hitter",
         "r": 90, "hr": 25, "rbi": 80, "sb": 30, "avg": .275, "ab": 530, "h": 146},
        {"name": "Player C", "var": 13.0, "total_sgp": 18.0, "best_position": "C", "positions": ["C"],
         "player_type": "hitter", "player_id": "Player C::hitter",
         "r": 70, "hr": 22, "rbi": 75, "sb": 2, "avg": .260, "ab": 480, "h": 125},
        {"name": "Player D", "var": 12.0, "total_sgp": 17.0, "best_position": "P", "positions": ["SP"],
         "player_type": "pitcher", "player_id": "Player D::pitcher",
         "w": 14, "k": 210, "sv": 0, "era": 3.20, "whip": 1.10, "ip": 195,
         "er": 69, "bb": 50, "h_allowed": 165},
        {"name": "Player E", "var": 11.0, "total_sgp": 16.0, "best_position": "OF", "positions": ["OF"],
         "player_type": "hitter", "player_id": "Player E::hitter",
         "r": 80, "hr": 20, "rbi": 70, "sb": 15, "avg": .270, "ab": 500, "h": 135},
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
