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
