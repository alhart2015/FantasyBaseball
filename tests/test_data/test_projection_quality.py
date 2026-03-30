import pytest
import pandas as pd
from fantasy_baseball.data.projection_quality import QualityReport, check_projection_quality


class TestQualityReport:
    def test_empty_report(self):
        report = QualityReport()
        assert report.warnings == []
        assert report.exclusions == {}
        assert report.missing_players == {}

    def test_has_warnings(self):
        report = QualityReport(warnings=["test warning"])
        assert len(report.warnings) == 1


class TestCheckProjectionQuality:
    def test_returns_report_with_no_issues(self):
        """Two identical systems should produce no warnings."""
        hitters = pd.DataFrame([
            {"name": "Aaron Judge", "hr": 45, "r": 110, "rbi": 120,
             "sb": 5, "h": 160, "ab": 550, "pa": 650, "fg_id": "1"},
        ])
        pitchers = pd.DataFrame([
            {"name": "Gerrit Cole", "w": 15, "k": 240, "sv": 0,
             "ip": 200, "er": 70, "bb": 56, "h_allowed": 154, "fg_id": "2"},
        ])
        system_dfs = {
            "steamer": (hitters.copy(), pitchers.copy()),
            "zips": (hitters.copy(), pitchers.copy()),
        }
        report = check_projection_quality(system_dfs)
        assert isinstance(report, QualityReport)
        assert report.exclusions == {}


class TestStatOutlierDetection:
    def test_excludes_system_with_all_zero_stat(self):
        """ZiPS SV = 0 for everyone while steamer has real values."""
        steamer_pitchers = pd.DataFrame([
            {"name": "Emmanuel Clase", "w": 4, "k": 70, "sv": 40,
             "ip": 70, "er": 14, "bb": 14, "h_allowed": 49, "fg_id": "1"},
            {"name": "Josh Hader", "w": 3, "k": 80, "sv": 35,
             "ip": 65, "er": 18, "bb": 20, "h_allowed": 45, "fg_id": "2"},
        ])
        zips_pitchers = pd.DataFrame([
            {"name": "Emmanuel Clase", "w": 3, "k": 66, "sv": 0,
             "ip": 68, "er": 15, "bb": 15, "h_allowed": 50, "fg_id": "1"},
            {"name": "Josh Hader", "w": 3, "k": 75, "sv": 0,
             "ip": 63, "er": 20, "bb": 22, "h_allowed": 48, "fg_id": "2"},
        ])
        system_dfs = {
            "steamer": (pd.DataFrame(), steamer_pitchers),
            "zips": (pd.DataFrame(), zips_pitchers),
        }
        report = check_projection_quality(system_dfs)
        assert "zips" in report.exclusions
        assert "sv" in report.exclusions["zips"]

    def test_no_exclusion_when_systems_agree(self):
        """Two systems with similar SV should not trigger exclusion."""
        steamer_pitchers = pd.DataFrame([
            {"name": "Emmanuel Clase", "w": 4, "k": 70, "sv": 40,
             "ip": 70, "er": 14, "bb": 14, "h_allowed": 49, "fg_id": "1"},
        ])
        zips_pitchers = pd.DataFrame([
            {"name": "Emmanuel Clase", "w": 3, "k": 66, "sv": 38,
             "ip": 68, "er": 15, "bb": 15, "h_allowed": 50, "fg_id": "1"},
        ])
        system_dfs = {
            "steamer": (pd.DataFrame(), steamer_pitchers),
            "zips": (pd.DataFrame(), zips_pitchers),
        }
        report = check_projection_quality(system_dfs)
        assert report.exclusions == {}

    def test_warns_on_moderate_deviation(self):
        """System with >50% deviation gets a warning but not exclusion."""
        steamer_hitters = pd.DataFrame([
            {"name": "Player A", "hr": 30, "r": 80, "rbi": 90,
             "sb": 10, "h": 150, "ab": 500, "pa": 600, "fg_id": "1"},
        ])
        zips_hitters = pd.DataFrame([
            {"name": "Player A", "hr": 14, "r": 78, "rbi": 88,
             "sb": 9, "h": 148, "ab": 498, "pa": 598, "fg_id": "1"},
        ])
        system_dfs = {
            "steamer": (steamer_hitters, pd.DataFrame()),
            "zips": (zips_hitters, pd.DataFrame()),
        }
        report = check_projection_quality(system_dfs)
        hr_warnings = [w for w in report.warnings if "hr" in w.lower()]
        assert len(hr_warnings) > 0
        assert "zips" not in report.exclusions or "hr" not in report.exclusions.get("zips", set())

    def test_excludes_all_nan_column(self):
        """A system where a stat column is entirely NaN should be excluded."""
        steamer_pitchers = pd.DataFrame([
            {"name": "Pitcher A", "w": 10, "k": 180, "sv": 0,
             "ip": 180, "er": 60, "bb": 50, "h_allowed": 150, "fg_id": "1"},
        ])
        bad_pitchers = pd.DataFrame([
            {"name": "Pitcher A", "w": 10, "k": 180, "sv": float("nan"),
             "ip": 180, "er": 60, "bb": 50, "h_allowed": 150, "fg_id": "1"},
        ])
        system_dfs = {
            "steamer": (pd.DataFrame(), steamer_pitchers),
            "bad_system": (pd.DataFrame(), bad_pitchers),
        }
        report = check_projection_quality(system_dfs)
        assert "bad_system" in report.exclusions
        assert "sv" in report.exclusions["bad_system"]

    def test_handles_sparse_stat_like_sv(self):
        """SV is 0 for most players — only compare among players with >0 in any system."""
        base_pitchers = pd.DataFrame([
            {"name": "Clase", "w": 4, "k": 70, "sv": 40,
             "ip": 70, "er": 14, "bb": 14, "h_allowed": 49, "fg_id": "1"},
            {"name": "Starter A", "w": 12, "k": 200, "sv": 0,
             "ip": 190, "er": 65, "bb": 50, "h_allowed": 160, "fg_id": "2"},
            {"name": "Starter B", "w": 10, "k": 170, "sv": 0,
             "ip": 175, "er": 60, "bb": 45, "h_allowed": 150, "fg_id": "3"},
        ])
        system_dfs = {
            "steamer": (pd.DataFrame(), base_pitchers.copy()),
            "zips": (pd.DataFrame(), base_pitchers.copy()),
            "atc": (pd.DataFrame(), base_pitchers.copy()),
        }
        report = check_projection_quality(system_dfs)
        for sys_excl in report.exclusions.values():
            assert "sv" not in sys_excl
