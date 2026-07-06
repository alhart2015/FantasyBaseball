import pandas as pd

from fantasy_baseball.sgp.rankings import (
    build_rankings_lookup,
    compute_sgp_rankings,
    lookup_rank,
    rank_key,
)


class TestComputeSgpRankings:
    def _make_hitters_df(self):
        return pd.DataFrame(
            [
                {
                    "name": "Aaron Judge",
                    "player_type": "hitter",
                    "r": 100,
                    "hr": 40,
                    "rbi": 100,
                    "sb": 5,
                    "h": 160,
                    "ab": 550,
                    "avg": 0.291,
                    "pa": 650,
                },
                {
                    "name": "Juan Soto",
                    "player_type": "hitter",
                    "r": 110,
                    "hr": 35,
                    "rbi": 90,
                    "sb": 10,
                    "h": 155,
                    "ab": 540,
                    "avg": 0.287,
                    "pa": 680,
                },
                {
                    "name": "Marcus Semien",
                    "player_type": "hitter",
                    "r": 80,
                    "hr": 20,
                    "rbi": 70,
                    "sb": 12,
                    "h": 140,
                    "ab": 600,
                    "avg": 0.233,
                    "pa": 660,
                },
            ]
        )

    def _make_pitchers_df(self):
        return pd.DataFrame(
            [
                {
                    "name": "Gerrit Cole",
                    "player_type": "pitcher",
                    "w": 15,
                    "k": 220,
                    "sv": 0,
                    "ip": 200,
                    "era": 2.80,
                    "whip": 0.95,
                    "er": 62,
                    "bb": 40,
                    "h_allowed": 150,
                },
                {
                    "name": "Emmanuel Clase",
                    "player_type": "pitcher",
                    "w": 3,
                    "k": 70,
                    "sv": 40,
                    "ip": 70,
                    "era": 2.50,
                    "whip": 0.90,
                    "er": 19,
                    "bb": 15,
                    "h_allowed": 48,
                },
            ]
        )

    def test_returns_dict_keyed_by_name_and_type(self):
        rankings = compute_sgp_rankings(self._make_hitters_df(), self._make_pitchers_df())
        assert rank_key("Aaron Judge", "hitter") in rankings
        assert rank_key("Gerrit Cole", "pitcher") in rankings

    def test_hitters_ranked_separately_from_pitchers(self):
        rankings = compute_sgp_rankings(self._make_hitters_df(), self._make_pitchers_df())
        hitter_ranks = [
            rankings[rank_key(n, "hitter")] for n in ["Aaron Judge", "Juan Soto", "Marcus Semien"]
        ]
        pitcher_ranks = [
            rankings[rank_key(n, "pitcher")] for n in ["Gerrit Cole", "Emmanuel Clase"]
        ]
        assert 1 in hitter_ranks
        assert 1 in pitcher_ranks

    def test_ranks_are_ordinal_1_based(self):
        rankings = compute_sgp_rankings(self._make_hitters_df(), self._make_pitchers_df())
        hitter_ranks = sorted(
            [rankings[rank_key(n, "hitter")] for n in ["Aaron Judge", "Juan Soto", "Marcus Semien"]]
        )
        assert hitter_ranks == [1, 2, 3]

    def test_higher_sgp_gets_lower_rank_number(self):
        rankings = compute_sgp_rankings(self._make_hitters_df(), self._make_pitchers_df())
        assert (
            rankings[rank_key("Aaron Judge", "hitter")]
            < rankings[rank_key("Marcus Semien", "hitter")]
        )

    def test_empty_dataframes_return_empty_dict(self):
        rankings = compute_sgp_rankings(pd.DataFrame(), pd.DataFrame())
        assert rankings == {}

    def test_denoms_override_reaches_ranking_math(self):
        """League denominator overrides must change the ranking basis."""
        from fantasy_baseball.sgp.denominators import get_sgp_denominators

        hitters, pitchers = self._make_hitters_df(), self._make_pitchers_df()
        default = compute_sgp_rankings(hitters, pitchers)
        assert (
            default[rank_key("Aaron Judge", "hitter")]
            < default[rank_key("Marcus Semien", "hitter")]
        )
        # 0.1 SB per standings place makes SB dominate: Semien (12 SB)
        # overtakes Judge (5 SB) despite Judge's power edge.
        overridden = compute_sgp_rankings(
            hitters, pitchers, denoms=get_sgp_denominators({"SB": 0.1})
        )
        assert (
            overridden[rank_key("Marcus Semien", "hitter")]
            < overridden[rank_key("Aaron Judge", "hitter")]
        )

    def test_same_name_hitter_and_pitcher_get_separate_ranks(self):
        """Juan Soto the hitter and Juan Soto the pitcher get independent ranks."""
        hitters = pd.DataFrame(
            [
                {
                    "name": "Juan Soto",
                    "player_type": "hitter",
                    "r": 110,
                    "hr": 35,
                    "rbi": 90,
                    "sb": 10,
                    "h": 155,
                    "ab": 540,
                    "avg": 0.287,
                    "pa": 680,
                },
            ]
        )
        pitchers = pd.DataFrame(
            [
                {
                    "name": "Juan Soto",
                    "player_type": "pitcher",
                    "w": 0,
                    "k": 1,
                    "sv": 0,
                    "ip": 2,
                    "era": 4.50,
                    "whip": 1.50,
                    "er": 1,
                    "bb": 1,
                    "h_allowed": 2,
                },
            ]
        )
        rankings = compute_sgp_rankings(hitters, pitchers)
        assert rank_key("Juan Soto", "hitter") in rankings
        assert rank_key("Juan Soto", "pitcher") in rankings
        assert rankings[rank_key("Juan Soto", "hitter")] == 1
        assert rankings[rank_key("Juan Soto", "pitcher")] == 1

    def test_same_name_same_type_disambiguated_by_fg_id(self):
        """Two pitchers named Mason Miller get distinct ranks via fg_id."""
        pitchers = pd.DataFrame(
            [
                {
                    "name": "Mason Miller",
                    "player_type": "pitcher",
                    "fg_id": "31757",
                    "w": 3,
                    "k": 99,
                    "sv": 32,
                    "ip": 63,
                    "era": 2.50,
                    "whip": 0.90,
                    "er": 17,
                    "bb": 15,
                    "h_allowed": 42,
                },
                {
                    "name": "Mason Miller",
                    "player_type": "pitcher",
                    "fg_id": "sa3023658",
                    "w": 0,
                    "k": 1,
                    "sv": 0,
                    "ip": 2,
                    "era": 4.50,
                    "whip": 1.50,
                    "er": 1,
                    "bb": 1,
                    "h_allowed": 2,
                },
            ]
        )
        rankings = compute_sgp_rankings(pd.DataFrame(), pitchers)
        # Each fg_id gets its own (pool-namespaced) rank
        assert "31757::pitcher" in rankings
        assert "sa3023658::pitcher" in rankings
        assert rankings["31757::pitcher"] < rankings["sa3023658::pitcher"]  # real Miller higher
        # Name key gets the better (lower) rank
        assert rankings[rank_key("Mason Miller", "pitcher")] == rankings["31757::pitcher"]

    def test_shared_fg_id_across_pools_keeps_separate_ranks(self):
        """A two-way player (one fg_id in both pools) keeps a rank per pool.

        Regression: the pitcher pass used to overwrite the bare fg_id key,
        so a position player's mop-up-innings line (or a two-way star's
        pitching line) buried his real hitter rank. Namespacing the fg_id
        key by pool keeps each pool-relative ordinal separate; lookup_rank
        selects the right one by player_type.
        """
        two_way_id = "660271"
        hitters = pd.DataFrame(
            [
                {
                    "name": "Shohei Ohtani",
                    "player_type": "hitter",
                    "fg_id": two_way_id,
                    "r": 130,
                    "hr": 50,
                    "rbi": 120,
                    "sb": 20,
                    "h": 175,
                    "ab": 560,
                    "avg": 0.313,
                    "pa": 660,
                },
                {
                    "name": "Weak Bat",
                    "player_type": "hitter",
                    "fg_id": "111",
                    "r": 40,
                    "hr": 5,
                    "rbi": 35,
                    "sb": 1,
                    "h": 100,
                    "ab": 480,
                    "avg": 0.208,
                    "pa": 520,
                },
            ]
        )
        pitchers = pd.DataFrame(
            [
                {
                    "name": "Ace Pitcher",
                    "player_type": "pitcher",
                    "fg_id": "222",
                    "w": 15,
                    "k": 220,
                    "sv": 0,
                    "ip": 200,
                    "era": 2.80,
                    "whip": 0.95,
                    "er": 62,
                    "bb": 40,
                    "h_allowed": 150,
                },
                {
                    "name": "Shohei Ohtani",
                    "player_type": "pitcher",
                    "fg_id": two_way_id,
                    "w": 0,
                    "k": 1,
                    "sv": 0,
                    "ip": 2,
                    "era": 4.50,
                    "whip": 1.50,
                    "er": 1,
                    "bb": 1,
                    "h_allowed": 2,
                },
            ]
        )
        rankings = compute_sgp_rankings(hitters, pitchers)
        hitter_rank = rankings[rank_key("Shohei Ohtani", "hitter")]
        pitcher_rank = rankings[rank_key("Shohei Ohtani", "pitcher")]
        assert hitter_rank == 1  # elite bat leads the hitter pool
        assert pitcher_rank > hitter_rank  # junk 1-IP line ranks low among pitchers
        # Both pool ranks survive under type-namespaced fg_id keys (no overwrite).
        assert rankings[f"{two_way_id}::hitter"] == hitter_rank
        assert rankings[f"{two_way_id}::pitcher"] == pitcher_rank

        # Consumer path: lookup_rank resolves the pool-correct rank by type.
        merged = build_rankings_lookup(rankings, {}, {})
        got_hitter = lookup_rank(merged, two_way_id, "Shohei Ohtani", "hitter")
        got_pitcher = lookup_rank(merged, two_way_id, "Shohei Ohtani", "pitcher")
        assert got_hitter["rest_of_season"] == hitter_rank
        assert got_pitcher["rest_of_season"] == pitcher_rank


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
        assert "aaron judge::hitter" in rankings
        assert "gerrit cole::pitcher" in rankings
        assert rankings["aaron judge::hitter"] in (1, 2)
        assert rankings["gerrit cole::pitcher"] == 1

    def test_empty_logs_return_empty_dict(self):
        from fantasy_baseball.sgp.rankings import compute_rankings_from_game_logs

        assert compute_rankings_from_game_logs({}, {}) == {}

    def test_denoms_override_reaches_game_log_ranking_math(self):
        from fantasy_baseball.sgp.denominators import get_sgp_denominators
        from fantasy_baseball.sgp.rankings import compute_rankings_from_game_logs

        hitter_logs = {
            "aaron judge": {"pa": 100, "ab": 80, "h": 25, "r": 15, "hr": 8, "rbi": 20, "sb": 1},
            "juan soto": {"pa": 110, "ab": 90, "h": 30, "r": 18, "hr": 6, "rbi": 15, "sb": 3},
        }
        # SB-dominated denominators rank Soto first (3 SB vs Judge's 1);
        # HR-dominated denominators rank Judge first (8 HR vs Soto's 6).
        # Together they prove the denoms actually reach the SGP math.
        sb_ranked = compute_rankings_from_game_logs(
            hitter_logs, {}, denoms=get_sgp_denominators({"SB": 0.01})
        )
        assert sb_ranked["juan soto::hitter"] == 1
        hr_ranked = compute_rankings_from_game_logs(
            hitter_logs, {}, denoms=get_sgp_denominators({"HR": 0.01})
        )
        assert hr_ranked["aaron judge::hitter"] == 1


class TestBuildRankingsLookup:
    def test_player_in_all_three(self):
        ros = {"Soto::hitter": {"overall": 5}}
        pre = {"Soto::hitter": {"overall": 3}}
        cur = {"Soto::hitter": {"overall": 7}}
        result = build_rankings_lookup(ros, pre, cur)
        assert result["Soto::hitter"] == {
            "rest_of_season": {"overall": 5},
            "preseason": {"overall": 3},
            "current": {"overall": 7},
            "total": None,
        }

    def test_player_only_in_ros_has_none_for_others(self):
        result = build_rankings_lookup(
            ros={"Newbie::hitter": {"overall": 100}},
            preseason={},
            current={},
        )
        assert result["Newbie::hitter"] == {
            "rest_of_season": {"overall": 100},
            "preseason": None,
            "current": None,
            "total": None,
        }

    def test_player_only_in_preseason_has_none_for_others(self):
        # E.g. preseason hype guy who didn't end up on the ROS list
        result = build_rankings_lookup(
            ros={},
            preseason={"Bust::hitter": {"overall": 50}},
            current={},
        )
        assert result["Bust::hitter"] == {
            "rest_of_season": None,
            "preseason": {"overall": 50},
            "current": None,
            "total": None,
        }

    def test_player_only_in_current_has_none_for_others(self):
        # Surprise breakout with no projection on either side
        result = build_rankings_lookup(
            ros={},
            preseason={},
            current={"Surprise::hitter": {"overall": 25}},
        )
        assert result["Surprise::hitter"] == {
            "rest_of_season": None,
            "preseason": None,
            "current": {"overall": 25},
            "total": None,
        }

    def test_union_includes_keys_from_all_three(self):
        result = build_rankings_lookup(
            ros={"A::hitter": {"o": 1}},
            preseason={"B::hitter": {"o": 2}},
            current={"C::hitter": {"o": 3}},
        )
        assert set(result.keys()) == {"A::hitter", "B::hitter", "C::hitter"}

    def test_empty_inputs_yield_empty_dict(self):
        assert build_rankings_lookup({}, {}, {}) == {}

    def test_build_rankings_lookup_includes_total(self):
        result = build_rankings_lookup(
            {"a::hitter": 1}, {"a::hitter": 2}, {"a::hitter": 3}, {"a::hitter": 4}
        )
        assert result["a::hitter"] == {
            "rest_of_season": 1,
            "preseason": 2,
            "current": 3,
            "total": 4,
        }

    def test_build_rankings_lookup_total_defaults_none(self):
        result = build_rankings_lookup({"a::hitter": 1}, {}, {})
        assert result["a::hitter"]["total"] is None

    def test_build_rankings_lookup_player_only_in_total(self):
        result = build_rankings_lookup({}, {}, {}, {"z::pitcher": 7})
        assert result["z::pitcher"] == {
            "rest_of_season": None,
            "preseason": None,
            "current": None,
            "total": 7,
        }
