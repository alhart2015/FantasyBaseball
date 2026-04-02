import pytest
import pandas as pd
from fantasy_baseball.sgp.rankings import compute_sgp_rankings


class TestComputeSgpRankings:
    def _make_hitters_df(self):
        return pd.DataFrame([
            {"name": "Aaron Judge", "player_type": "hitter", "r": 100, "hr": 40, "rbi": 100, "sb": 5, "h": 160, "ab": 550, "avg": 0.291, "pa": 650},
            {"name": "Juan Soto", "player_type": "hitter", "r": 110, "hr": 35, "rbi": 90, "sb": 10, "h": 155, "ab": 540, "avg": 0.287, "pa": 680},
            {"name": "Marcus Semien", "player_type": "hitter", "r": 80, "hr": 20, "rbi": 70, "sb": 12, "h": 140, "ab": 600, "avg": 0.233, "pa": 660},
        ])

    def _make_pitchers_df(self):
        return pd.DataFrame([
            {"name": "Gerrit Cole", "player_type": "pitcher", "w": 15, "k": 220, "sv": 0, "ip": 200, "era": 2.80, "whip": 0.95, "er": 62, "bb": 40, "h_allowed": 150},
            {"name": "Emmanuel Clase", "player_type": "pitcher", "w": 3, "k": 70, "sv": 40, "ip": 70, "era": 2.50, "whip": 0.90, "er": 19, "bb": 15, "h_allowed": 48},
        ])

    def test_returns_dict_keyed_by_normalized_name(self):
        from fantasy_baseball.utils.name_utils import normalize_name
        rankings = compute_sgp_rankings(self._make_hitters_df(), self._make_pitchers_df())
        assert normalize_name("Aaron Judge") in rankings
        assert normalize_name("Gerrit Cole") in rankings

    def test_hitters_ranked_separately_from_pitchers(self):
        from fantasy_baseball.utils.name_utils import normalize_name
        rankings = compute_sgp_rankings(self._make_hitters_df(), self._make_pitchers_df())
        hitter_ranks = [rankings[normalize_name(n)] for n in ["Aaron Judge", "Juan Soto", "Marcus Semien"]]
        pitcher_ranks = [rankings[normalize_name(n)] for n in ["Gerrit Cole", "Emmanuel Clase"]]
        assert 1 in hitter_ranks
        assert 1 in pitcher_ranks

    def test_ranks_are_ordinal_1_based(self):
        from fantasy_baseball.utils.name_utils import normalize_name
        rankings = compute_sgp_rankings(self._make_hitters_df(), self._make_pitchers_df())
        hitter_ranks = sorted([rankings[normalize_name(n)] for n in ["Aaron Judge", "Juan Soto", "Marcus Semien"]])
        assert hitter_ranks == [1, 2, 3]

    def test_higher_sgp_gets_lower_rank_number(self):
        from fantasy_baseball.utils.name_utils import normalize_name
        rankings = compute_sgp_rankings(self._make_hitters_df(), self._make_pitchers_df())
        assert rankings[normalize_name("Aaron Judge")] < rankings[normalize_name("Marcus Semien")]

    def test_empty_dataframes_return_empty_dict(self):
        rankings = compute_sgp_rankings(pd.DataFrame(), pd.DataFrame())
        assert rankings == {}


class TestRankingsFromGameLogs:
    def test_ranks_from_game_log_totals(self):
        from fantasy_baseball.sgp.rankings import compute_rankings_from_game_logs

        hitter_logs = {
            "aaron judge": {"pa": 100, "ab": 80, "h": 25, "r": 15, "hr": 8, "rbi": 20, "sb": 1},
            "juan soto": {"pa": 110, "ab": 90, "h": 30, "r": 18, "hr": 6, "rbi": 15, "sb": 3},
        }
        pitcher_logs = {
            "gerrit cole": {"ip": 30, "k": 35, "w": 3, "sv": 0, "er": 8, "bb": 5, "h_allowed": 20},
        }
        rankings = compute_rankings_from_game_logs(hitter_logs, pitcher_logs)
        assert "aaron judge" in rankings
        assert "gerrit cole" in rankings
        assert rankings["aaron judge"] in (1, 2)
        assert rankings["gerrit cole"] == 1

    def test_empty_logs_return_empty_dict(self):
        from fantasy_baseball.sgp.rankings import compute_rankings_from_game_logs
        assert compute_rankings_from_game_logs({}, {}) == {}
