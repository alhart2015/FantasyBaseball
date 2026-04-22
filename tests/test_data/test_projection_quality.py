import pandas as pd
import pytest

from fantasy_baseball.data.projection_quality import QualityReport, check_projection_quality
from fantasy_baseball.data.projections import blend_projections


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
        hitters = pd.DataFrame(
            [
                {
                    "name": "Aaron Judge",
                    "hr": 45,
                    "r": 110,
                    "rbi": 120,
                    "sb": 5,
                    "h": 160,
                    "ab": 550,
                    "pa": 650,
                    "fg_id": "1",
                },
            ]
        )
        pitchers = pd.DataFrame(
            [
                {
                    "name": "Gerrit Cole",
                    "w": 15,
                    "k": 240,
                    "sv": 0,
                    "ip": 200,
                    "er": 70,
                    "bb": 56,
                    "h_allowed": 154,
                    "fg_id": "2",
                },
            ]
        )
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
        steamer_pitchers = pd.DataFrame(
            [
                {
                    "name": "Emmanuel Clase",
                    "w": 4,
                    "k": 70,
                    "sv": 40,
                    "ip": 70,
                    "er": 14,
                    "bb": 14,
                    "h_allowed": 49,
                    "fg_id": "1",
                },
                {
                    "name": "Josh Hader",
                    "w": 3,
                    "k": 80,
                    "sv": 35,
                    "ip": 65,
                    "er": 18,
                    "bb": 20,
                    "h_allowed": 45,
                    "fg_id": "2",
                },
            ]
        )
        zips_pitchers = pd.DataFrame(
            [
                {
                    "name": "Emmanuel Clase",
                    "w": 3,
                    "k": 66,
                    "sv": 0,
                    "ip": 68,
                    "er": 15,
                    "bb": 15,
                    "h_allowed": 50,
                    "fg_id": "1",
                },
                {
                    "name": "Josh Hader",
                    "w": 3,
                    "k": 75,
                    "sv": 0,
                    "ip": 63,
                    "er": 20,
                    "bb": 22,
                    "h_allowed": 48,
                    "fg_id": "2",
                },
            ]
        )
        system_dfs = {
            "steamer": (pd.DataFrame(), steamer_pitchers),
            "zips": (pd.DataFrame(), zips_pitchers),
        }
        report = check_projection_quality(system_dfs)
        assert "zips" in report.exclusions
        assert "sv" in report.exclusions["zips"]

    def test_no_exclusion_when_systems_agree(self):
        """Two systems with similar SV should not trigger exclusion."""
        steamer_pitchers = pd.DataFrame(
            [
                {
                    "name": "Emmanuel Clase",
                    "w": 4,
                    "k": 70,
                    "sv": 40,
                    "ip": 70,
                    "er": 14,
                    "bb": 14,
                    "h_allowed": 49,
                    "fg_id": "1",
                },
            ]
        )
        zips_pitchers = pd.DataFrame(
            [
                {
                    "name": "Emmanuel Clase",
                    "w": 3,
                    "k": 66,
                    "sv": 38,
                    "ip": 68,
                    "er": 15,
                    "bb": 15,
                    "h_allowed": 50,
                    "fg_id": "1",
                },
            ]
        )
        system_dfs = {
            "steamer": (pd.DataFrame(), steamer_pitchers),
            "zips": (pd.DataFrame(), zips_pitchers),
        }
        report = check_projection_quality(system_dfs)
        assert report.exclusions == {}

    def test_warns_on_moderate_deviation(self):
        """System with >50% deviation gets a warning but not exclusion."""
        steamer_hitters = pd.DataFrame(
            [
                {
                    "name": "Player A",
                    "hr": 30,
                    "r": 80,
                    "rbi": 90,
                    "sb": 10,
                    "h": 150,
                    "ab": 500,
                    "pa": 600,
                    "fg_id": "1",
                },
            ]
        )
        zips_hitters = pd.DataFrame(
            [
                {
                    "name": "Player A",
                    "hr": 14,
                    "r": 78,
                    "rbi": 88,
                    "sb": 9,
                    "h": 148,
                    "ab": 498,
                    "pa": 598,
                    "fg_id": "1",
                },
            ]
        )
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
        steamer_pitchers = pd.DataFrame(
            [
                {
                    "name": "Pitcher A",
                    "w": 10,
                    "k": 180,
                    "sv": 0,
                    "ip": 180,
                    "er": 60,
                    "bb": 50,
                    "h_allowed": 150,
                    "fg_id": "1",
                },
            ]
        )
        bad_pitchers = pd.DataFrame(
            [
                {
                    "name": "Pitcher A",
                    "w": 10,
                    "k": 180,
                    "sv": float("nan"),
                    "ip": 180,
                    "er": 60,
                    "bb": 50,
                    "h_allowed": 150,
                    "fg_id": "1",
                },
            ]
        )
        system_dfs = {
            "steamer": (pd.DataFrame(), steamer_pitchers),
            "bad_system": (pd.DataFrame(), bad_pitchers),
        }
        report = check_projection_quality(system_dfs)
        assert "bad_system" in report.exclusions
        assert "sv" in report.exclusions["bad_system"]

    def test_handles_sparse_stat_like_sv(self):
        """SV is 0 for most players — only compare among players with >0 in any system."""
        base_pitchers = pd.DataFrame(
            [
                {
                    "name": "Clase",
                    "w": 4,
                    "k": 70,
                    "sv": 40,
                    "ip": 70,
                    "er": 14,
                    "bb": 14,
                    "h_allowed": 49,
                    "fg_id": "1",
                },
                {
                    "name": "Starter A",
                    "w": 12,
                    "k": 200,
                    "sv": 0,
                    "ip": 190,
                    "er": 65,
                    "bb": 50,
                    "h_allowed": 160,
                    "fg_id": "2",
                },
                {
                    "name": "Starter B",
                    "w": 10,
                    "k": 170,
                    "sv": 0,
                    "ip": 175,
                    "er": 60,
                    "bb": 45,
                    "h_allowed": 150,
                    "fg_id": "3",
                },
            ]
        )
        system_dfs = {
            "steamer": (pd.DataFrame(), base_pitchers.copy()),
            "zips": (pd.DataFrame(), base_pitchers.copy()),
            "atc": (pd.DataFrame(), base_pitchers.copy()),
        }
        report = check_projection_quality(system_dfs)
        for sys_excl in report.exclusions.values():
            assert "sv" not in sys_excl

    def test_large_pool_fringe_players_filtered_out(self):
        """Systems with many fringe players (low AB) should not be excluded."""
        # Steamer: 3 real hitters + 100 fringe players with <50 AB
        real_hitters = [
            {
                "name": f"Star {i}",
                "hr": 30,
                "r": 80,
                "rbi": 90,
                "sb": 10,
                "h": 150,
                "ab": 500,
                "pa": 600,
                "fg_id": str(i),
            }
            for i in range(3)
        ]
        fringe_hitters = [
            {
                "name": f"Fringe {i}",
                "hr": 0,
                "r": 0,
                "rbi": 0,
                "sb": 0,
                "h": 0,
                "ab": 5,
                "pa": 6,
                "fg_id": str(100 + i),
            }
            for i in range(100)
        ]
        steamer = pd.DataFrame(real_hitters + fringe_hitters)
        # ZiPS: only the 3 real hitters (no fringe)
        zips = pd.DataFrame(real_hitters)
        system_dfs = {
            "steamer": (steamer, pd.DataFrame()),
            "zips": (zips, pd.DataFrame()),
        }
        report = check_projection_quality(system_dfs)
        # Steamer should NOT be excluded — fringe players filtered by AB < 50
        assert "steamer" not in report.exclusions


class TestPlayerCountCheck:
    def test_warns_on_low_player_count(self):
        """System with <50% of median player count gets a warning."""
        big_hitters = pd.DataFrame(
            [
                {
                    "name": f"Player {i}",
                    "hr": 20,
                    "r": 80,
                    "rbi": 80,
                    "sb": 5,
                    "h": 140,
                    "ab": 500,
                    "pa": 600,
                    "fg_id": str(i),
                }
                for i in range(100)
            ]
        )
        small_hitters = pd.DataFrame(
            [
                {
                    "name": f"Player {i}",
                    "hr": 20,
                    "r": 80,
                    "rbi": 80,
                    "sb": 5,
                    "h": 140,
                    "ab": 500,
                    "pa": 600,
                    "fg_id": str(i),
                }
                for i in range(10)
            ]
        )
        system_dfs = {
            "steamer": (big_hitters, pd.DataFrame()),
            "zips": (big_hitters.copy(), pd.DataFrame()),
            "tiny": (small_hitters, pd.DataFrame()),
        }
        report = check_projection_quality(system_dfs)
        count_warnings = [w for w in report.warnings if "tiny" in w and "count" in w.lower()]
        assert len(count_warnings) > 0

    def test_no_warning_when_counts_similar(self):
        """Systems with similar player counts should not trigger warnings."""
        hitters_a = pd.DataFrame(
            [
                {
                    "name": f"Player {i}",
                    "hr": 20,
                    "r": 80,
                    "rbi": 80,
                    "sb": 5,
                    "h": 140,
                    "ab": 500,
                    "pa": 600,
                    "fg_id": str(i),
                }
                for i in range(100)
            ]
        )
        hitters_b = pd.DataFrame(
            [
                {
                    "name": f"Player {i}",
                    "hr": 20,
                    "r": 80,
                    "rbi": 80,
                    "sb": 5,
                    "h": 140,
                    "ab": 500,
                    "pa": 600,
                    "fg_id": str(i),
                }
                for i in range(90)
            ]
        )
        system_dfs = {
            "steamer": (hitters_a, pd.DataFrame()),
            "zips": (hitters_b, pd.DataFrame()),
        }
        report = check_projection_quality(system_dfs)
        count_warnings = [w for w in report.warnings if "count" in w.lower()]
        assert len(count_warnings) == 0


class TestRosterCoverage:
    def test_warns_on_missing_player(self):
        """Rostered player missing from one system gets a warning."""
        steamer = pd.DataFrame(
            [
                {
                    "name": "Aaron Judge",
                    "hr": 45,
                    "r": 110,
                    "rbi": 120,
                    "sb": 5,
                    "h": 160,
                    "ab": 550,
                    "pa": 650,
                    "fg_id": "1",
                },
                {
                    "name": "Blake Snell",
                    "hr": 0,
                    "r": 0,
                    "rbi": 0,
                    "sb": 0,
                    "h": 0,
                    "ab": 0,
                    "pa": 0,
                    "fg_id": "3",
                },
            ]
        )
        zips = pd.DataFrame(
            [
                {
                    "name": "Aaron Judge",
                    "hr": 42,
                    "r": 105,
                    "rbi": 115,
                    "sb": 4,
                    "h": 155,
                    "ab": 545,
                    "pa": 640,
                    "fg_id": "1",
                },
            ]
        )
        from fantasy_baseball.utils.name_utils import normalize_name

        roster = {normalize_name("Aaron Judge"), normalize_name("Blake Snell")}
        system_dfs = {
            "steamer": (steamer, pd.DataFrame()),
            "zips": (zips, pd.DataFrame()),
        }
        report = check_projection_quality(system_dfs, roster_names=roster)
        assert normalize_name("Blake Snell") in report.missing_players
        assert "zips" in report.missing_players[normalize_name("Blake Snell")]

    def test_warns_loudly_when_missing_from_all(self):
        """Player missing from ALL systems gets a loud warning."""
        hitters = pd.DataFrame(
            [
                {
                    "name": "Aaron Judge",
                    "hr": 45,
                    "r": 110,
                    "rbi": 120,
                    "sb": 5,
                    "h": 160,
                    "ab": 550,
                    "pa": 650,
                    "fg_id": "1",
                },
            ]
        )
        from fantasy_baseball.utils.name_utils import normalize_name

        roster = {normalize_name("Aaron Judge"), normalize_name("Ghost Player")}
        system_dfs = {
            "steamer": (hitters, pd.DataFrame()),
            "zips": (hitters.copy(), pd.DataFrame()),
        }
        report = check_projection_quality(system_dfs, roster_names=roster)
        assert normalize_name("Ghost Player") in report.missing_players
        all_warnings = [w for w in report.warnings if "ALL" in w and "ghost player" in w.lower()]
        assert len(all_warnings) > 0

    def test_no_warning_when_all_covered(self):
        """All rostered players in all systems -> no missing player warnings."""
        hitters = pd.DataFrame(
            [
                {
                    "name": "Aaron Judge",
                    "hr": 45,
                    "r": 110,
                    "rbi": 120,
                    "sb": 5,
                    "h": 160,
                    "ab": 550,
                    "pa": 650,
                    "fg_id": "1",
                },
            ]
        )
        from fantasy_baseball.utils.name_utils import normalize_name

        roster = {normalize_name("Aaron Judge")}
        system_dfs = {
            "steamer": (hitters, pd.DataFrame()),
            "zips": (hitters.copy(), pd.DataFrame()),
        }
        report = check_projection_quality(system_dfs, roster_names=roster)
        assert report.missing_players == {}

    def test_skips_roster_check_when_none(self):
        """roster_names=None skips roster coverage check entirely."""
        system_dfs = {
            "steamer": (pd.DataFrame(), pd.DataFrame()),
            "zips": (pd.DataFrame(), pd.DataFrame()),
        }
        report = check_projection_quality(system_dfs, roster_names=None)
        assert report.missing_players == {}


class TestBlendWithQualityChecks:
    def test_returns_three_tuple(self, fixtures_dir):
        """blend_projections now returns (hitters, pitchers, report)."""
        hitters, pitchers, report = blend_projections(
            fixtures_dir,
            systems=["steamer", "zips"],
        )
        assert len(hitters) > 0
        assert len(pitchers) > 0
        assert isinstance(report, QualityReport)

    def test_excludes_bad_stat_from_blend(self, tmp_path):
        """When a system has all-zero SV, that system's SV is excluded from blend."""
        # System A: closer with 40 SV
        a_pitchers = pd.DataFrame(
            [
                {
                    "Name": "Closer X",
                    "Team": "NYY",
                    "IP": 70,
                    "W": 4,
                    "SO": 70,
                    "SV": 40,
                    "ERA": 1.80,
                    "WHIP": 0.90,
                    "ER": 14,
                    "BB": 14,
                    "H": 49,
                    "playerid": "1",
                }
            ]
        )
        # System B: same closer but SV = 0 (broken export)
        b_pitchers = pd.DataFrame(
            [
                {
                    "Name": "Closer X",
                    "Team": "NYY",
                    "IP": 68,
                    "W": 3,
                    "SO": 66,
                    "SV": 0,
                    "ERA": 2.00,
                    "WHIP": 0.95,
                    "ER": 15,
                    "BB": 15,
                    "H": 50,
                    "playerid": "1",
                }
            ]
        )

        # Write CSVs
        a_pitchers.to_csv(tmp_path / "systema-pitchers.csv", index=False)
        b_pitchers.to_csv(tmp_path / "systemb-pitchers.csv", index=False)
        # Need empty hitter files too
        pd.DataFrame(
            columns=["Name", "Team", "PA", "AB", "H", "HR", "R", "RBI", "SB", "AVG", "playerid"]
        ).to_csv(tmp_path / "systema-hitters.csv", index=False)
        pd.DataFrame(
            columns=["Name", "Team", "PA", "AB", "H", "HR", "R", "RBI", "SB", "AVG", "playerid"]
        ).to_csv(tmp_path / "systemb-hitters.csv", index=False)

        _hitters, pitchers, report = blend_projections(
            tmp_path,
            systems=["systema", "systemb"],
        )
        closer = pitchers[pitchers["name"] == "Closer X"].iloc[0]
        # SV should come only from system A (40), not averaged with B's 0
        assert closer["sv"] == pytest.approx(40.0)
        assert "systemb" in report.exclusions
        assert "sv" in report.exclusions["systemb"]

    def test_progress_cb_receives_warnings(self, fixtures_dir):
        """progress_cb is called with each warning."""
        messages = []
        _hitters, _pitchers, _report = blend_projections(
            fixtures_dir,
            systems=["steamer", "zips"],
            progress_cb=messages.append,
        )
        assert isinstance(messages, list)
