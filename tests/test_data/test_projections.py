import pytest
import pandas as pd
from pathlib import Path
from fantasy_baseball.data.projections import blend_projections, match_roster_to_projections
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
        names = {p["name"] for p in matched}
        assert names == {"Aaron Judge", "Gerrit Cole"}
