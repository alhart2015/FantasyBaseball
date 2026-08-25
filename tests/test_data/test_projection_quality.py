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
    def test_does_not_crash_on_nan_name_row(self):
        """Regression: a projection df with a NaN name (blank CSV row) reaching
        the roster-coverage check must not raise. Production failure was
        'normalize() argument 2 must be str, not float' in _check_roster_coverage,
        which killed the daily ROS fetch."""
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
                    "fg_id": "1",
                },
                {
                    "name": float("nan"),
                    "hr": 10,
                    "r": 40,
                    "rbi": 50,
                    "sb": 1,
                    "h": 80,
                    "ab": 350,
                    "fg_id": "99",
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
        # Must not raise.
        report = check_projection_quality(system_dfs, roster_names={"aaron judge"})
        assert isinstance(report, QualityReport)

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


class TestEmptySystemExport:
    """A system whose export has zero data rows must be reported.

    Regression for 2026-08-25: ``the-bat-x-pitchers.csv`` arrived header-only
    (0 rows, 675 the run before) and NOTHING warned. ``_check_player_counts``
    guarded its proportional test with ``count > 0``, so the one case that
    should shout loudest -- a completely empty file -- was the only case it
    could not see.
    """

    @staticmethod
    def _hitters(n, offset=0):
        return pd.DataFrame(
            [
                {
                    "name": f"Hitter {i + offset}",
                    "fg_id": str(i + offset),
                    "hr": 20,
                    "r": 80,
                    "rbi": 75,
                    "sb": 8,
                    "h": 140,
                    "ab": 520,
                }
                for i in range(n)
            ]
        )

    @staticmethod
    def _pitchers(n):
        return pd.DataFrame(
            [
                {
                    "name": f"Pitcher {i}",
                    "fg_id": f"p{i}",
                    "w": 10,
                    "k": 150,
                    "sv": 0,
                    "ip": 150,
                    "er": 60,
                    "bb": 45,
                    "h_allowed": 140,
                }
                for i in range(n)
            ]
        )

    def test_empty_pitcher_export_is_reported(self):
        system_dfs = {
            "steamer": (self._hitters(40), self._pitchers(40)),
            "zips": (self._hitters(40), self._pitchers(40)),
            "the-bat-x": (self._hitters(40), pd.DataFrame()),
        }
        report = check_projection_quality(system_dfs)
        empties = [w for w in report.warnings if "NO pitcher rows" in w]
        assert len(empties) == 1, f"expected one empty-export warning, got {report.warnings}"
        assert "the-bat-x" in empties[0]
        # Names the consequence so the reader knows the blend still produced
        # correct numbers, just from fewer systems.
        assert "2 of 3" in empties[0]

    def test_healthy_hitters_do_not_warn_when_only_pitchers_are_empty(self):
        """Emptiness is per player type: the-bat-x hitters were fine that day."""
        system_dfs = {
            "steamer": (self._hitters(40), self._pitchers(40)),
            "zips": (self._hitters(40), self._pitchers(40)),
            "the-bat-x": (self._hitters(40), pd.DataFrame()),
        }
        report = check_projection_quality(system_dfs)
        assert not [w for w in report.warnings if "NO hitter rows" in w]

    def test_all_systems_present_produces_no_empty_warning(self):
        system_dfs = {
            "steamer": (self._hitters(40), self._pitchers(40)),
            "zips": (self._hitters(40), self._pitchers(40)),
        }
        report = check_projection_quality(system_dfs)
        assert not [w for w in report.warnings if "the export is" in w]

    def test_warnings_are_ascii(self):
        """These strings reach print() on a cp1252 console; see CLAUDE.md."""
        system_dfs = {
            "steamer": (self._hitters(40), self._pitchers(40)),
            "zips": (self._hitters(8, offset=100), self._pitchers(40)),
            "the-bat-x": (self._hitters(40), pd.DataFrame()),
        }
        report = check_projection_quality(system_dfs)
        assert report.warnings, "fixture should trip at least one warning"
        for w in report.warnings:
            w.encode("ascii")


class TestEmptyExportMessageAccuracy:
    """The empty-export warning must not assert things that are false.

    The first version said "the remaining N of M systems renormalize to cover
    it" unconditionally. When EVERY system is empty that renders "0 of M" and
    the sentence is wrong twice over: nothing renormalizes, and there are no
    projections of that type at all. It also said "the export is empty" for a
    case it cannot distinguish from a file that was never downloaded.
    """

    @staticmethod
    def _pitchers(n):
        return pd.DataFrame(
            [
                {
                    "name": f"P{i}",
                    "fg_id": f"p{i}",
                    "w": 10,
                    "k": 150,
                    "sv": 0,
                    "ip": 150,
                    "er": 60,
                    "bb": 45,
                    "h_allowed": 140,
                }
                for i in range(n)
            ]
        )

    @staticmethod
    def _hitters(n):
        return pd.DataFrame(
            [
                {
                    "name": f"H{i}",
                    "fg_id": str(i),
                    "hr": 20,
                    "r": 80,
                    "rbi": 75,
                    "sb": 8,
                    "h": 140,
                    "ab": 520,
                }
                for i in range(n)
            ]
        )

    def test_all_systems_empty_does_not_claim_survivors_renormalize(self):
        system_dfs = {
            "steamer": (self._hitters(40), pd.DataFrame()),
            "zips": (self._hitters(40), pd.DataFrame()),
            "the-bat-x": (self._hitters(40), pd.DataFrame()),
        }
        report = check_projection_quality(system_dfs)
        empties = [w for w in report.warnings if "NO pitcher rows" in w]
        assert len(empties) == 3
        for w in empties:
            # Not `"0 of" not in w`: that substring also appears in "10 of 10",
            # so it tests the wrong property the moment a league configures ten
            # systems. Assert the claim itself is absent instead.
            assert "renormalize" not in w, f"nothing renormalizes here: {w}"
            assert "of 3" not in w, f"must not report a survivor count: {w}"
            assert "NO systems have pitcher rows" in w, f"must say none survive: {w}"
            assert "no pitcher projections at all" in w, (
                f"must say the blend has no rows of this type: {w}"
            )

    def test_partial_emptiness_still_names_the_survivors(self):
        system_dfs = {
            "steamer": (self._hitters(40), self._pitchers(40)),
            "zips": (self._hitters(40), self._pitchers(40)),
            "the-bat-x": (self._hitters(40), pd.DataFrame()),
        }
        report = check_projection_quality(system_dfs)
        empties = [w for w in report.warnings if "NO pitcher rows" in w]
        assert len(empties) == 1
        assert "2 of 3" in empties[0]
        assert "renormalize" in empties[0]

    def test_message_does_not_assert_the_file_exists(self):
        """load_projection_set returns an empty frame for a MISSING file too,
        so the warning must not tell the user their download is corrupt."""
        system_dfs = {
            "steamer": (self._hitters(40), self._pitchers(40)),
            "zips": (self._hitters(40), self._pitchers(40)),
            "the-bat-x": (self._hitters(40), pd.DataFrame()),
        }
        report = check_projection_quality(system_dfs)
        w = next(x for x in report.warnings if "NO pitcher rows" in x)
        assert "empty or missing" in w, f"must not claim the file exists: {w}"


class TestQualityWarningsReachCallersWithoutProgressCb:
    """A quality warning must not be invisible to a caller that passes no
    ``progress_cb``.

    Reporting used to be gated entirely on ``progress_cb`` being truthy, so a
    consumer that only wants the blended frames discarded every warning,
    including the empty-export one -- and they discard the returned
    ``QualityReport`` too, so nothing surfaced anywhere. ``ros_anchor`` and
    ``draft_value`` are in that position; ``db`` and ``build_db`` do pass a
    ``progress_cb`` and were never silent.

    Only SYSTEMIC warnings are logged, and only when there is no
    ``progress_cb``: logging everything would double ``build_db``'s output and
    flood ``db``'s per-year loop, since roster-coverage emits one warning per
    missing player, uncapped.
    """

    @staticmethod
    def _snapshot(tmp_path, *, empty_pitchers_for=()):
        import csv

        snap = tmp_path / "2026-01-01"
        snap.mkdir()
        h_cols = ["Name", "Team", "G", "AB", "PA", "H", "R", "HR", "RBI", "SB", "AVG", "PlayerId"]
        p_cols = ["Name", "Team", "W", "SV", "IP", "SO", "ER", "BB", "H", "ERA", "WHIP", "PlayerId"]
        for system in ("steamer", "zips"):
            with open(snap / f"{system}-hitters.csv", "w", newline="", encoding="utf-8") as fh:
                w = csv.writer(fh)
                w.writerow(h_cols)
                for i in range(30):
                    w.writerow([f"H{i}", "NYY", 150, 520, 570, 140, 80, 20, 75, 8, 0.269, i])
            with open(snap / f"{system}-pitchers.csv", "w", newline="", encoding="utf-8") as fh:
                w = csv.writer(fh)
                w.writerow(p_cols)
                if system not in empty_pitchers_for:
                    for i in range(30):
                        w.writerow([f"P{i}", "NYY", 10, 0, 150, 150, 60, 45, 140, 3.60, 1.23, i])
        return snap

    def test_an_empty_export_is_logged_when_no_progress_cb_is_given(self, tmp_path, caplog):
        snap = self._snapshot(tmp_path, empty_pitchers_for=("zips",))
        with caplog.at_level("WARNING", logger="fantasy_baseball.data.projections"):
            blend_projections(snap, ["steamer", "zips"], None, normalizer=None)
        logged = " ".join(r.getMessage() for r in caplog.records)
        assert "NO pitcher rows" in logged, (
            f"empty export was invisible to a caller with no progress_cb: {logged!r}"
        )

    def test_progress_cb_still_receives_the_warning(self, tmp_path):
        snap = self._snapshot(tmp_path, empty_pitchers_for=("zips",))
        seen: list[str] = []
        blend_projections(snap, ["steamer", "zips"], None, progress_cb=seen.append, normalizer=None)
        assert any("NO pitcher rows" in m for m in seen)


class TestSystemicWarningsAreSeparatedFromPerPlayerNoise:
    """Only whole-system warnings are logged for a caller with no progress_cb.

    `_check_roster_coverage` emits one warning per rostered player it cannot
    find, uncapped, and `data/db.py` blends once per season directory. Logging
    every warning would flood that loop and double the output of
    `scripts/build_db.py`, which passes `progress_cb=print`.
    """

    def test_per_player_coverage_warnings_are_not_logged(self, tmp_path, caplog):
        snap = TestQualityWarningsReachCallersWithoutProgressCb._snapshot(
            tmp_path, empty_pitchers_for=("zips",)
        )
        roster = {"nobody projects this guy", "or this one"}
        with caplog.at_level("WARNING", logger="fantasy_baseball.data.projections"):
            blend_projections(snap, ["steamer", "zips"], None, roster_names=roster, normalizer=None)
        logged = " ".join(r.getMessage() for r in caplog.records)
        assert "NO pitcher rows" in logged, "the systemic warning must still be logged"
        assert "nobody projects this guy" not in logged, (
            f"per-player coverage noise must not reach the log: {logged!r}"
        )

    def test_a_caller_with_progress_cb_is_not_also_logged(self, tmp_path, caplog):
        """build_db passes progress_cb=print; it must not get every line twice."""
        snap = TestQualityWarningsReachCallersWithoutProgressCb._snapshot(
            tmp_path, empty_pitchers_for=("zips",)
        )
        seen: list[str] = []
        with caplog.at_level("WARNING", logger="fantasy_baseball.data.projections"):
            blend_projections(
                snap, ["steamer", "zips"], None, progress_cb=seen.append, normalizer=None
            )
        assert any("NO pitcher rows" in m for m in seen)
        assert not [r for r in caplog.records], (
            "a caller that supplied progress_cb must not also be logged to"
        )
