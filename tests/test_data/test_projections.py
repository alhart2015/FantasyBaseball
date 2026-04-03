import pytest
import pandas as pd
from pathlib import Path
from fantasy_baseball.data.projections import blend_projections, match_roster_to_projections
from fantasy_baseball.models.player import Player, HitterStats, PitcherStats
from fantasy_baseball.utils.name_utils import normalize_name


class TestBlendProjections:
    def test_blend_two_systems_equal_weight(self, fixtures_dir):
        hitters, pitchers, _ = blend_projections(
            fixtures_dir,
            systems=["steamer", "zips"],
        )
        assert len(hitters) == 4
        assert len(pitchers) == 3

    def test_blended_counting_stats_are_averaged(self, fixtures_dir):
        hitters, pitchers, _ = blend_projections(
            fixtures_dir,
            systems=["steamer", "zips"],
        )
        judge = hitters[hitters["name"] == "Aaron Judge"].iloc[0]
        # Steamer: 45 HR, ZiPS: 42 HR -> avg = 43.5
        assert judge["hr"] == pytest.approx(43.5)
        # Steamer: 110 R, ZiPS: 105 R -> avg = 107.5
        assert judge["r"] == pytest.approx(107.5)

    def test_blended_avg_recomputed_from_components(self, fixtures_dir):
        hitters, pitchers, _ = blend_projections(
            fixtures_dir,
            systems=["steamer", "zips"],
        )
        judge = hitters[hitters["name"] == "Aaron Judge"].iloc[0]
        # Steamer: 160 H / 550 AB, ZiPS: 155 H / 545 AB
        # Blended: 157.5 H / 547.5 AB = .2877
        expected_avg = 157.5 / 547.5
        assert judge["avg"] == pytest.approx(expected_avg, abs=0.001)

    def test_blended_era_recomputed_from_components(self, fixtures_dir):
        hitters, pitchers, _ = blend_projections(
            fixtures_dir,
            systems=["steamer", "zips"],
        )
        cole = pitchers[pitchers["name"] == "Gerrit Cole"].iloc[0]
        # Steamer: 70 ER / 200 IP, ZiPS: 72 ER / 195 IP
        # Blended: 71 ER / 197.5 IP -> ERA = 71 * 9 / 197.5 = 3.234
        expected_era = 71.0 * 9 / 197.5
        assert cole["era"] == pytest.approx(expected_era, abs=0.01)

    def test_blended_whip_recomputed_from_components(self, fixtures_dir):
        hitters, pitchers, _ = blend_projections(
            fixtures_dir,
            systems=["steamer", "zips"],
        )
        cole = pitchers[pitchers["name"] == "Gerrit Cole"].iloc[0]
        # Steamer: (56 BB + 154 H) / 200 IP = 1.05
        # ZiPS: (57 BB + 154 H) / 195 IP = 1.08
        # Blended: (56.5 BB + 154 H) / 197.5 IP
        expected_whip = (56.5 + 154.0) / 197.5
        assert cole["whip"] == pytest.approx(expected_whip, abs=0.01)

    def test_custom_weights(self, fixtures_dir):
        hitters, pitchers, _ = blend_projections(
            fixtures_dir,
            systems=["steamer", "zips"],
            weights={"steamer": 0.75, "zips": 0.25},
        )
        judge = hitters[hitters["name"] == "Aaron Judge"].iloc[0]
        # Steamer: 45 HR * 0.75 + ZiPS: 42 HR * 0.25 = 33.75 + 10.5 = 44.25
        assert judge["hr"] == pytest.approx(44.25)

    def test_single_system(self, fixtures_dir):
        hitters, pitchers, _ = blend_projections(
            fixtures_dir,
            systems=["steamer"],
        )
        judge = hitters[hitters["name"] == "Aaron Judge"].iloc[0]
        assert judge["hr"] == 45

    def test_blend_preserves_mlbam_id_in_metadata(self, fixtures_dir):
        """mlbam_id should be available in loaded system DataFrames."""
        from fantasy_baseball.data.fangraphs import load_projection_set
        hitters, pitchers = load_projection_set(fixtures_dir, "steamer")
        assert "mlbam_id" in hitters.columns, "mlbam_id missing from hitter columns"
        judge = hitters[hitters["name"] == "Aaron Judge"].iloc[0]
        assert judge["mlbam_id"] == 592450

    def test_missing_system_raises_error(self, fixtures_dir):
        with pytest.raises(FileNotFoundError, match="No projection files found for system"):
            blend_projections(
                fixtures_dir,
                systems=["steamer", "nonexistent"],
            )

    def test_missing_directory_raises_error(self, tmp_path):
        missing = tmp_path / "does_not_exist"
        with pytest.raises(FileNotFoundError, match="Projections directory not found"):
            blend_projections(missing, systems=["steamer"])

    def test_empty_directory_raises_error(self, tmp_path):
        empty_dir = tmp_path / "empty_projections"
        empty_dir.mkdir()
        with pytest.raises(FileNotFoundError, match="No CSV files found"):
            blend_projections(empty_dir, systems=["steamer"])


class TestMatchRosterToProjections:
    def test_requires_name_norm_column(self):
        """match_roster_to_projections fails without _name_norm on projection DFs."""
        roster = [{"name": "Aaron Judge", "positions": ["OF"]}]
        hitters = pd.DataFrame({"name": ["Aaron Judge"], "hr": [45]})
        pitchers = pd.DataFrame()
        with pytest.raises(KeyError, match="_name_norm"):
            match_roster_to_projections(roster, hitters, pitchers)

    def test_matches_with_name_norm(self, fixtures_dir):
        """match_roster_to_projections works when _name_norm is present."""
        hitters, pitchers, _ = blend_projections(fixtures_dir, systems=["steamer"])
        hitters["_name_norm"] = hitters["name"].apply(normalize_name)
        pitchers["_name_norm"] = pitchers["name"].apply(normalize_name)

        roster = [
            {"name": "Aaron Judge", "positions": ["OF"]},
            {"name": "Gerrit Cole", "positions": ["SP"]},
        ]
        matched = match_roster_to_projections(roster, hitters, pitchers)
        assert len(matched) == 2
        names = {p.name for p in matched}
        assert names == {"Aaron Judge", "Gerrit Cole"}

    def test_returns_player_objects(self):
        roster = [
            {"name": "Aaron Judge", "positions": ["OF"], "selected_position": "OF", "player_id": "123", "status": ""},
        ]
        hitters = pd.DataFrame([{
            "name": "Aaron Judge", "_name_norm": "aaron judge",
            "r": 110, "hr": 45, "rbi": 120, "sb": 5, "avg": 0.291,
            "ab": 550, "h": 160, "pa": 650, "player_type": "hitter",
        }])
        pitchers = pd.DataFrame(columns=["name", "_name_norm", "player_type"])

        result = match_roster_to_projections(roster, hitters, pitchers)
        assert len(result) == 1
        assert isinstance(result[0], Player)
        assert result[0].name == "Aaron Judge"
        assert result[0].player_type == "hitter"
        assert isinstance(result[0].ros, HitterStats)
        assert result[0].ros.hr == 45
        assert result[0].ros.avg == 0.291
        assert result[0].positions == ["OF"]

    def test_pitcher_returns_pitcher_stats(self):
        roster = [
            {"name": "Gerrit Cole", "positions": ["SP"], "selected_position": "SP", "player_id": "456", "status": ""},
        ]
        hitters = pd.DataFrame(columns=["name", "_name_norm", "player_type"])
        pitchers = pd.DataFrame([{
            "name": "Gerrit Cole", "_name_norm": "gerrit cole",
            "w": 15, "k": 240, "sv": 0, "ip": 200, "er": 70, "bb": 56, "h_allowed": 154,
            "era": 3.15, "whip": 1.05, "player_type": "pitcher",
        }])

        result = match_roster_to_projections(roster, hitters, pitchers)
        assert len(result) == 1
        assert isinstance(result[0], Player)
        assert result[0].player_type == "pitcher"
        assert isinstance(result[0].ros, PitcherStats)
        assert result[0].ros.k == 240

    def test_unmatched_players_omitted(self):
        roster = [
            {"name": "Nobody Special", "positions": ["OF"], "selected_position": "OF", "player_id": "999", "status": ""},
        ]
        hitters = pd.DataFrame(columns=["name", "_name_norm", "player_type"])
        pitchers = pd.DataFrame(columns=["name", "_name_norm", "player_type"])

        result = match_roster_to_projections(roster, hitters, pitchers)
        assert result == []


class TestNormalizeRosToFullSeason:
    def test_adds_hitter_actuals_to_remaining_games(self):
        from fantasy_baseball.data.projections import (
            normalize_ros_to_full_season, HITTING_COUNTING_COLS,
        )
        df = pd.DataFrame([{
            "name": "Aaron Judge", "mlbam_id": 592450, "player_type": "hitter",
            "pa": 400, "ab": 300, "h": 90, "r": 60, "hr": 25, "rbi": 65, "sb": 4,
        }])
        game_log_totals = {
            592450: {"pa": 100, "ab": 80, "h": 25, "r": 15, "hr": 5, "rbi": 15, "sb": 1},
        }
        result = normalize_ros_to_full_season(df, game_log_totals, "hitter")
        judge = result.iloc[0]
        assert judge["pa"] == 500
        assert judge["ab"] == 380
        assert judge["h"] == 115
        assert judge["r"] == 75
        assert judge["hr"] == 30
        assert judge["rbi"] == 80
        assert judge["sb"] == 5

    def test_adds_pitcher_actuals_to_remaining_games(self):
        from fantasy_baseball.data.projections import normalize_ros_to_full_season
        df = pd.DataFrame([{
            "name": "Gerrit Cole", "mlbam_id": 543037, "player_type": "pitcher",
            "ip": 170, "k": 190, "w": 12, "sv": 0, "er": 60, "bb": 40, "h_allowed": 130,
        }])
        game_log_totals = {
            543037: {"ip": 13, "k": 16, "w": 1, "sv": 0, "er": 5, "bb": 3, "h_allowed": 9},
        }
        result = normalize_ros_to_full_season(df, game_log_totals, "pitcher")
        cole = result.iloc[0]
        assert cole["ip"] == 183
        assert cole["k"] == 206
        assert cole["w"] == 13
        assert cole["er"] == 65
        assert cole["bb"] == 43
        assert cole["h_allowed"] == 139

    def test_no_game_log_leaves_player_unchanged(self):
        from fantasy_baseball.data.projections import normalize_ros_to_full_season
        df = pd.DataFrame([{
            "name": "Rookie Player", "mlbam_id": 999999, "player_type": "hitter",
            "pa": 400, "ab": 300, "h": 90, "r": 60, "hr": 25, "rbi": 65, "sb": 4,
        }])
        result = normalize_ros_to_full_season(df, {}, "hitter")
        assert result.iloc[0]["pa"] == 400
        assert result.iloc[0]["hr"] == 25

    def test_missing_mlbam_id_leaves_player_unchanged(self):
        from fantasy_baseball.data.projections import normalize_ros_to_full_season
        df = pd.DataFrame([{
            "name": "Aaron Judge", "player_type": "hitter",
            "pa": 400, "ab": 300, "h": 90, "r": 60, "hr": 25, "rbi": 65, "sb": 4,
        }])
        game_log_totals = {
            592450: {"pa": 100, "ab": 80, "h": 25, "r": 15, "hr": 5, "rbi": 15, "sb": 1},
        }
        result = normalize_ros_to_full_season(df, game_log_totals, "hitter")
        assert result.iloc[0]["pa"] == 400

    def test_does_not_mutate_input_dataframe(self):
        from fantasy_baseball.data.projections import normalize_ros_to_full_season
        df = pd.DataFrame([{
            "name": "Aaron Judge", "mlbam_id": 592450, "player_type": "hitter",
            "pa": 400, "ab": 300, "h": 90, "r": 60, "hr": 25, "rbi": 65, "sb": 4,
        }])
        game_log_totals = {
            592450: {"pa": 100, "ab": 80, "h": 25, "r": 15, "hr": 5, "rbi": 15, "sb": 1},
        }
        normalize_ros_to_full_season(df, game_log_totals, "hitter")
        assert df.iloc[0]["pa"] == 400


class TestBlendWithNormalizer:
    def test_normalizer_called_for_each_system(self, fixtures_dir):
        """Normalizer callback is invoked per system with correct args."""
        calls = []

        def track_normalizer(system_name, hitters_df, pitchers_df):
            calls.append(system_name)
            return hitters_df, pitchers_df

        blend_projections(
            fixtures_dir,
            systems=["steamer", "zips"],
            normalizer=track_normalizer,
        )
        assert "steamer" in calls
        assert "zips" in calls

    def test_normalizer_modifies_counting_stats_before_blend(self, fixtures_dir):
        """When normalizer bumps a system's stats, the blend reflects it."""
        def bump_zips_hr(system_name, hitters_df, pitchers_df):
            if system_name == "zips":
                hitters_df = hitters_df.copy()
                hitters_df["hr"] = hitters_df["hr"] + 10
            return hitters_df, pitchers_df

        baseline, _, _ = blend_projections(fixtures_dir, systems=["steamer", "zips"])
        bumped, _, _ = blend_projections(
            fixtures_dir, systems=["steamer", "zips"], normalizer=bump_zips_hr,
        )

        judge_base = baseline[baseline["name"] == "Aaron Judge"].iloc[0]["hr"]
        judge_bump = bumped[bumped["name"] == "Aaron Judge"].iloc[0]["hr"]
        # ZiPS HR went up by 10, weight is 0.5 each → blend goes up by 5
        assert judge_bump == pytest.approx(judge_base + 5, abs=0.1)
