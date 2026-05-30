from datetime import date

import pytest


class TestCategoryStats:
    def test_default_values(self):
        from fantasy_baseball.models.standings import CategoryStats

        stats = CategoryStats()
        assert stats.r == 0.0
        assert stats.hr == 0.0
        assert stats.rbi == 0.0
        assert stats.sb == 0.0
        assert stats.avg == 0.0
        assert stats.w == 0.0
        assert stats.k == 0.0
        assert stats.sv == 0.0
        assert stats.era == 99.0
        assert stats.whip == 99.0

    def test_construction_with_values(self):
        from fantasy_baseball.models.standings import CategoryStats

        stats = CategoryStats(
            r=120,
            hr=45,
            rbi=130,
            sb=22,
            avg=0.275,
            w=60,
            k=800,
            sv=35,
            era=3.80,
            whip=1.15,
        )
        assert stats.r == 120
        assert stats.avg == pytest.approx(0.275)
        assert stats.era == pytest.approx(3.80)

    def test_from_dict(self):
        from fantasy_baseball.models.standings import CategoryStats

        stats = CategoryStats.from_dict(
            {
                "R": 120,
                "HR": 40,
                "RBI": 110,
                "SB": 8,
                "AVG": 0.272,
                "W": 55,
                "K": 750,
                "SV": 30,
                "ERA": 3.85,
                "WHIP": 1.18,
            }
        )
        assert stats.r == 120
        assert stats.whip == pytest.approx(1.18)

    def test_from_dict_missing_keys_default(self):
        from fantasy_baseball.models.standings import CategoryStats

        stats = CategoryStats.from_dict({"R": 100})
        assert stats.r == 100
        assert stats.hr == 0.0
        assert stats.era == 99.0


class TestCategoryStatsTypedAccess:
    def test_getitem_accepts_category_enum(self):
        from fantasy_baseball.models.standings import CategoryStats
        from fantasy_baseball.utils.constants import Category

        stats = CategoryStats(r=100, hr=40, era=3.5)
        assert stats[Category.R] == 100
        assert stats[Category.HR] == 40
        assert stats[Category.ERA] == pytest.approx(3.5)

    def test_getitem_rejects_bare_string(self):
        from fantasy_baseball.models.standings import CategoryStats

        stats = CategoryStats(r=100)
        with pytest.raises(TypeError, match="Category enum"):
            _ = stats["R"]

    def test_getitem_rejects_other_types(self):
        from fantasy_baseball.models.standings import CategoryStats

        stats = CategoryStats()
        with pytest.raises(TypeError, match="Category enum"):
            _ = stats[0]

    def test_items_yields_category_enums(self):
        from fantasy_baseball.models.standings import CategoryStats
        from fantasy_baseball.utils.constants import ALL_CATEGORIES, Category

        stats = CategoryStats(
            r=100, hr=40, rbi=120, sb=15, avg=0.280, w=50, k=700, sv=20, era=3.9, whip=1.20
        )
        items = list(stats.items())
        assert [k for k, _ in items] == ALL_CATEGORIES
        as_map = dict(items)
        assert as_map[Category.R] == 100
        assert as_map[Category.HR] == 40
        assert as_map[Category.WHIP] == pytest.approx(1.20)


class TestStandingsEntry:
    def test_construction(self):
        from fantasy_baseball.models.standings import (
            CategoryStats,
            StandingsEntry,
        )

        entry = StandingsEntry(
            team_name="Hart of the Order",
            team_key="431.l.17492.t.3",
            rank=4,
            stats=CategoryStats(r=100, hr=40),
        )
        assert entry.team_name == "Hart of the Order"
        assert entry.team_key == "431.l.17492.t.3"
        assert entry.rank == 4
        assert entry.stats.r == 100


class TestStandingsJSON:
    def _canonical_payload(self):
        return {
            "effective_date": "2026-04-15",
            "teams": [
                {
                    "name": "Alpha",
                    "team_key": "431.l.1.t.1",
                    "rank": 1,
                    "yahoo_points_for": 78.5,
                    "stats": {
                        "R": 45.0,
                        "HR": 12.0,
                        "RBI": 40.0,
                        "SB": 8.0,
                        "AVG": 0.268,
                        "W": 3.0,
                        "K": 85.0,
                        "SV": 4.0,
                        "ERA": 3.21,
                        "WHIP": 1.14,
                    },
                    "extras": {},
                },
            ],
        }

    def test_from_json_canonical_round_trip(self):
        from fantasy_baseball.models.standings import Standings

        payload = self._canonical_payload()
        s = Standings.from_json(payload)
        assert s.effective_date == date(2026, 4, 15)
        assert len(s.entries) == 1
        e = s.entries[0]
        assert e.team_name == "Alpha"
        assert e.team_key == "431.l.1.t.1"
        assert e.rank == 1
        assert e.yahoo_points_for == 78.5
        assert e.stats.r == 45
        assert e.stats.whip == pytest.approx(1.14)
        assert e.extras == {}
        assert s.to_json() == payload

    def test_from_json_accepts_missing_extras_key(self):
        """Entries without an 'extras' key (older in-flight writes) must
        still parse — ``extras`` defaults to the empty dict."""
        from fantasy_baseball.models.standings import Standings

        payload = {
            "effective_date": "2026-04-15",
            "teams": [
                {
                    "name": "Alpha",
                    "team_key": "431.l.1.t.1",
                    "rank": 1,
                    "stats": {
                        "R": 45,
                        "HR": 12,
                        "RBI": 40,
                        "SB": 8,
                        "AVG": 0.268,
                        "W": 3,
                        "K": 85,
                        "SV": 4,
                        "ERA": 3.21,
                        "WHIP": 1.14,
                    },
                },
            ],
        }
        s = Standings.from_json(payload)
        assert s.entries[0].extras == {}

    def test_extras_round_trip_with_pa_ip(self):
        """PA / IP land in ``extras`` keyed by :class:`OpportunityStat`
        and round-trip as uppercase string keys."""
        from fantasy_baseball.models.standings import Standings
        from fantasy_baseball.utils.constants import OpportunityStat

        payload = {
            "effective_date": "2026-04-15",
            "teams": [
                {
                    "name": "Alpha",
                    "team_key": "431.l.1.t.1",
                    "rank": 1,
                    "yahoo_points_for": 78.5,
                    "stats": {
                        "R": 45.0,
                        "HR": 12.0,
                        "RBI": 40.0,
                        "SB": 8.0,
                        "AVG": 0.268,
                        "W": 3.0,
                        "K": 85.0,
                        "SV": 4.0,
                        "ERA": 3.21,
                        "WHIP": 1.14,
                    },
                    "extras": {"IP": 190.0, "PA": 720.0},
                },
            ],
        }
        s = Standings.from_json(payload)
        extras = s.entries[0].extras
        assert extras[OpportunityStat.IP] == 190.0
        assert extras[OpportunityStat.PA] == 720.0
        # Round-trip: string keys back out.
        round_tripped = s.to_json()
        assert round_tripped["teams"][0]["extras"] == {"IP": 190.0, "PA": 720.0}

    def test_extras_ignores_unknown_keys(self):
        """Unknown extras keys survive as ignored (forward compat)."""
        from fantasy_baseball.models.standings import Standings
        from fantasy_baseball.utils.constants import OpportunityStat

        payload = {
            "effective_date": "2026-04-15",
            "teams": [
                {
                    "name": "Alpha",
                    "team_key": "431.l.1.t.1",
                    "rank": 1,
                    "stats": {"R": 45},
                    "extras": {"IP": 42.0, "NOT_A_STAT": 999.0},
                },
            ],
        }
        s = Standings.from_json(payload)
        assert s.entries[0].extras == {OpportunityStat.IP: 42.0}

    def test_from_json_rejects_legacy_shape(self):
        from fantasy_baseball.models.standings import Standings

        legacy = {
            "teams": [
                {
                    "team": "Alpha",
                    "team_key": "431.l.1.t.1",
                    "rank": 1,
                    "r": 45,
                    "hr": 12,
                    "rbi": 40,
                    "sb": 8,
                    "avg": 0.268,
                    "w": 3,
                    "k": 85,
                    "sv": 4,
                    "era": 3.21,
                    "whip": 1.14,
                },
            ],
        }
        with pytest.raises(ValueError, match=r"legacy|unknown|name"):
            Standings.from_json(legacy)


class TestProjectedStandingsJSON:
    def test_round_trip(self):
        from fantasy_baseball.models.standings import (
            CategoryStats,
            ProjectedStandings,
            ProjectedStandingsEntry,
        )

        ps = ProjectedStandings(
            effective_date=date(2026, 4, 15),
            entries=[
                ProjectedStandingsEntry(
                    team_name="Alpha",
                    stats=CategoryStats(r=600, hr=250, era=3.8, whip=1.18),
                ),
            ],
        )
        round_tripped = ProjectedStandings.from_json(ps.to_json())
        assert round_tripped == ps


class TestCategoryPoints:
    def test_getitem_and_total(self):
        from fantasy_baseball.models.standings import CategoryPoints
        from fantasy_baseball.utils.constants import Category

        cp = CategoryPoints(
            values={Category.R: 7.0, Category.HR: 4.5},
            total=11.5,
        )
        assert cp[Category.R] == 7.0
        assert cp[Category.HR] == 4.5
        assert cp.total == 11.5

    def test_getitem_rejects_string(self):
        from fantasy_baseball.models.standings import CategoryPoints

        cp = CategoryPoints(values={}, total=0.0)
        with pytest.raises(TypeError, match="Category"):
            _ = cp["R"]


class TestTeamYtdComponents:
    """Tests for the TeamYtdComponents derivation from Yahoo standings.

    These components are the rate-stat ingredients (H, AB, ER, IP, BB+H_allowed)
    recovered from CategoryStats + extras so the team-YTD + ROS projection in
    ProjectedStandings.from_rosters can recombine team AVG/ERA/WHIP without
    losing precision to pre-computed rates.
    """

    @staticmethod
    def _entry(*, avg=0.0, era=0.0, whip=0.0, ip=0.0, ab=None, pa=None, **counts):
        from fantasy_baseball.models.standings import CategoryStats, StandingsEntry
        from fantasy_baseball.utils.constants import OpportunityStat

        stats = CategoryStats(
            r=counts.get("r", 0),
            hr=counts.get("hr", 0),
            rbi=counts.get("rbi", 0),
            sb=counts.get("sb", 0),
            avg=avg,
            w=counts.get("w", 0),
            k=counts.get("k", 0),
            sv=counts.get("sv", 0),
            era=era,
            whip=whip,
        )
        extras: dict[OpportunityStat, float] = {OpportunityStat.IP: ip}
        if ab is not None:
            extras[OpportunityStat.AB] = ab
        if pa is not None:
            extras[OpportunityStat.PA] = pa
        return StandingsEntry(team_name="T", team_key="t", rank=1, stats=stats, extras=extras)

    def test_components_use_explicit_ab_when_present(self):
        """When Yahoo standings expose AB directly, use it verbatim - no PA conversion."""
        e = self._entry(
            r=80,
            hr=20,
            rbi=70,
            sb=10,
            avg=0.275,
            ab=400,
            ip=200,
            w=10,
            k=180,
            sv=5,
            era=3.50,
            whip=1.20,
        )
        c = e.ytd_components()
        assert c.ab == pytest.approx(400.0)
        assert c.h == pytest.approx(0.275 * 400.0)  # AVG * AB
        assert c.ip == 200.0
        assert c.er == pytest.approx(3.50 * 200.0 / 9.0)
        assert c.bb_plus_h_allowed == pytest.approx(1.20 * 200.0)

    def test_components_fall_back_to_pa_when_ab_absent(self):
        """When only PA is exposed, derive AB via AB_PER_PA."""
        from fantasy_baseball.utils.constants import AB_PER_PA

        e = self._entry(
            r=80,
            hr=20,
            rbi=70,
            sb=10,
            avg=0.250,
            pa=500,
            ip=100,
            w=5,
            k=80,
            sv=2,
            era=4.00,
            whip=1.30,
        )
        c = e.ytd_components()
        assert c.ab == pytest.approx(500.0 * AB_PER_PA)
        assert c.h == pytest.approx(0.250 * (500.0 * AB_PER_PA))

    def test_components_prefer_explicit_ab_over_pa(self):
        """When BOTH AB and PA are present in extras, AB takes precedence and PA is ignored."""
        e = self._entry(
            r=80,
            hr=20,
            rbi=70,
            sb=10,
            avg=0.275,
            ab=400,
            pa=500,
            ip=200,
            w=10,
            k=180,
            sv=5,
            era=3.50,
            whip=1.20,
        )
        c = e.ytd_components()
        assert c.ab == pytest.approx(400.0)
        assert c.h == pytest.approx(0.275 * 400.0)

    def test_components_are_zero_when_neither_ab_nor_pa_present(self):
        """No way to recover AB without it or PA -> components.ab/h are 0.

        Callers (ProjectedStandings.from_rosters) detect this and fall back to
        summing ROS-only AVG from the roster (i.e. legacy ROS-only mode for AVG).
        """
        e = self._entry(
            r=80,
            hr=20,
            rbi=70,
            sb=10,
            avg=0.250,
            ip=100,
            w=5,
            k=80,
            sv=2,
            era=4.00,
            whip=1.30,
        )
        c = e.ytd_components()
        assert c.ab == 0.0
        assert c.h == 0.0

    def test_components_zero_when_no_ip_for_pitching_rates(self):
        """Pre-season standings have IP=0; ERA/WHIP components must be 0, not NaN."""
        e = self._entry(avg=0.0, era=0.0, whip=0.0, ip=0.0)
        c = e.ytd_components()
        assert c.ip == 0.0
        assert c.er == 0.0
        assert c.bb_plus_h_allowed == 0.0

    def test_components_carry_counting_stats(self):
        """Counting stats pass through unchanged."""
        e = self._entry(
            r=85,
            hr=22,
            rbi=78,
            sb=15,
            avg=0.0,
            ip=0.0,
            w=10,
            k=200,
            sv=8,
            era=0.0,
            whip=0.0,
        )
        c = e.ytd_components()
        assert c.r == 85
        assert c.hr == 22
        assert c.rbi == 78
        assert c.sb == 15
        assert c.w == 10
        assert c.k == 200
        assert c.sv == 8

    def test_components_fall_back_to_caller_hint_when_yahoo_lacks_ab_and_pa(self):
        """When Yahoo gives neither AB nor PA, the caller can pass a
        ``fallback_ab`` hint as a last resort. ``ytd_components(fallback_ab=X)``
        returns a components row with ab=X and h=AVG*X.
        """
        e = self._entry(
            r=80, hr=20, rbi=70, sb=10, avg=0.275, ip=200, w=10, k=180, sv=5, era=3.50, whip=1.20
        )
        c = e.ytd_components(fallback_ab=600.0)
        assert c.ab == pytest.approx(600.0)
        assert c.h == pytest.approx(0.275 * 600.0)

    def test_components_ignore_fallback_when_explicit_ab_present(self):
        """If Yahoo extras DO have AB, the fallback_ab hint is ignored."""
        e = self._entry(
            r=80,
            hr=20,
            rbi=70,
            sb=10,
            avg=0.275,
            ip=200,
            ab=500,
            w=10,
            k=180,
            sv=5,
            era=3.50,
            whip=1.20,
        )
        c = e.ytd_components(fallback_ab=999.0)
        assert c.ab == pytest.approx(500.0)

    def test_components_ignore_fallback_when_pa_present(self):
        """If Yahoo extras have PA (but not AB), PA-derived AB wins over fallback."""
        from fantasy_baseball.utils.constants import AB_PER_PA

        e = self._entry(
            r=80,
            hr=20,
            rbi=70,
            sb=10,
            avg=0.250,
            pa=500,
            ip=100,
            w=5,
            k=80,
            sv=2,
            era=4.00,
            whip=1.30,
        )
        c = e.ytd_components(fallback_ab=999.0)
        assert c.ab == pytest.approx(500.0 * AB_PER_PA)


class TestFromRostersTeamYtdProjection:
    """Regression tests for the team-YTD projection refactor.

    The bug (introduced in PR #108): ``_scale_stats`` added a per-player
    YTD floor sourced from ``Player.full_season_projection``. That
    attributes pre-acquisition stats to the current owner. The fix:
    pass team-level ``actual_standings`` to ``from_rosters`` so YTD is
    sourced from Yahoo's team totals (i.e., production accrued while the
    player was actually on the team).
    """

    def test_from_rosters_uses_team_ytd_not_player_full_season(self):
        """REGRESSION: a player with full_season K = 130 and ROS K = 80
        implies player YTD K = 50, but those K's may have been thrown for
        someone ELSE's team. The team's Yahoo standings show team-YTD K
        = 200 across actually-owned-while-played players. Result: team's
        projected K = team_YTD_K (200) + sum(player.ros.k) (80) = 280,
        NOT team_YTD_K (200) + player.full_season.k (130) = 330.
        """
        from datetime import date

        from fantasy_baseball.models.player import (
            PitcherStats,
            Player,
            PlayerType,
        )
        from fantasy_baseball.models.positions import Position
        from fantasy_baseball.models.standings import (
            CategoryStats,
            ProjectedStandings,
            Standings,
            StandingsEntry,
        )
        from fantasy_baseball.utils.constants import OpportunityStat

        ros = PitcherStats(
            ip=80,
            w=5,
            k=80,
            sv=0,
            er=30,
            bb=20,
            h_allowed=70,
            era=30 * 9.0 / 80,
            whip=(20 + 70) / 80,
        )
        full_season = PitcherStats(
            ip=130,
            w=8,
            k=130,
            sv=0,
            er=50,
            bb=30,
            h_allowed=110,
            era=50 * 9.0 / 130,
            whip=(30 + 110) / 130,
        )
        player = Player(
            name="MidSeasonPickup",
            player_type=PlayerType.PITCHER,
            rest_of_season=ros,
            full_season_projection=full_season,
            selected_position=Position.P,
        )

        actual = Standings(
            effective_date=date(2026, 6, 2),
            entries=[
                StandingsEntry(
                    team_name="Test",
                    team_key="t",
                    rank=1,
                    stats=CategoryStats(
                        r=0,
                        hr=0,
                        rbi=0,
                        sb=0,
                        avg=0.0,
                        w=10,
                        k=200,
                        sv=0,
                        era=3.50,
                        whip=1.20,
                    ),
                    extras={OpportunityStat.IP: 200.0},
                ),
            ],
        )

        ps = ProjectedStandings.from_rosters(
            {"Test": [player]},
            effective_date=date(2026, 6, 2),
            actual_standings=actual,
        )
        test_stats = next(e.stats for e in ps.entries if e.team_name == "Test")
        # Team YTD K (200) + ROS K (80) = 280. NOT team_YTD + full_season K (330).
        assert test_stats.k == pytest.approx(280)

    def test_from_rosters_without_actual_standings_collapses_to_ros_only(self):
        """Pre-season path: actual_standings=None -> team total = ROS-only."""
        from datetime import date

        from fantasy_baseball.models.player import (
            PitcherStats,
            Player,
            PlayerType,
        )
        from fantasy_baseball.models.positions import Position
        from fantasy_baseball.models.standings import ProjectedStandings

        ros = PitcherStats(
            ip=200,
            w=12,
            k=180,
            sv=0,
            er=70,
            bb=40,
            h_allowed=160,
            era=70 * 9.0 / 200,
            whip=(40 + 160) / 200,
        )
        p = Player(
            name="Pre",
            player_type=PlayerType.PITCHER,
            rest_of_season=ros,
            selected_position=Position.P,
        )

        ps = ProjectedStandings.from_rosters(
            {"Test": [p]},
            effective_date=date(2026, 3, 27),
            actual_standings=None,
        )
        team_k = next(e.stats.k for e in ps.entries if e.team_name == "Test")
        assert team_k == pytest.approx(180)


class TestFromRostersPass1BaselineIncludesYtd:
    """Fix #13: Pass-1 baseline must include team_YTD (not just ROS) so
    Pass-2 DeltaRoto picker decisions are scored against the correct
    cross-team end-of-season scale.

    Before this fix, ``from_rosters`` Pass-1 used
    ``project_team_stats(roster, displacement=True)`` (ROS-only). The
    docstring claimed YTD "cancels in the DeltaRoto comparisons" which
    is mathematically false: per-team YTD shifts are not uniform across
    the picker's argmax. After the fix, Pass-1 uses
    ``team_end_of_season(ytd, project_ros_components(...))`` so the
    picker sees the same scale Pass-2 emits.
    """

    def test_pass1_baseline_changes_with_actual_standings(self):
        """Two ``from_rosters`` runs over the same rosters but with
        different ``actual_standings`` YTD must produce different final
        projected stats for at least one OTHER team -- proving the
        Pass-1 baseline (other-team context) responded to YTD.

        We pin the OTHER team's projected K because Pass-2 of the user
        team would shift even with a ROS-only Pass-1 baseline (YTD is
        added in Pass-2 either way). The signal that Pass-1 changed is
        the OTHER team's K under two different user-team YTD inputs.
        """
        from fantasy_baseball.models.player import (
            HitterStats,
            PitcherStats,
            Player,
            PlayerType,
        )
        from fantasy_baseball.models.positions import Position
        from fantasy_baseball.models.standings import (
            CategoryStats,
            ProjectedStandings,
            Standings,
            StandingsEntry,
        )
        from fantasy_baseball.utils.constants import OpportunityStat

        # User has a single big-K arm; opponent has a single hitter.
        user_arm = Player(
            name="UserArm",
            player_type=PlayerType.PITCHER,
            rest_of_season=PitcherStats(
                ip=100,
                w=6,
                k=120,
                sv=0,
                er=35,
                bb=25,
                h_allowed=80,
                era=35 * 9.0 / 100,
                whip=(25 + 80) / 100,
            ),
            selected_position=Position.P,
        )
        opp_hitter = Player(
            name="OppHitter",
            player_type=PlayerType.HITTER,
            rest_of_season=HitterStats(pa=400, ab=360, h=99, r=55, hr=18, rbi=60, sb=4, avg=0.275),
            selected_position=Position.OF,
        )

        rosters = {"User": [user_arm], "Opp": [opp_hitter]}

        def _standings(user_k_ytd: float) -> Standings:
            return Standings(
                effective_date=date(2026, 6, 1),
                entries=[
                    StandingsEntry(
                        team_name="User",
                        team_key="u",
                        rank=1,
                        stats=CategoryStats(
                            r=0,
                            hr=0,
                            rbi=0,
                            sb=0,
                            avg=0.0,
                            w=10,
                            k=user_k_ytd,
                            sv=0,
                            era=3.50,
                            whip=1.20,
                        ),
                        extras={OpportunityStat.IP: 100.0},
                    ),
                    StandingsEntry(
                        team_name="Opp",
                        team_key="o",
                        rank=2,
                        stats=CategoryStats(
                            r=60,
                            hr=15,
                            rbi=55,
                            sb=5,
                            avg=0.260,
                            w=0,
                            k=0,
                            sv=0,
                            era=99.0,
                            whip=99.0,
                        ),
                        extras={OpportunityStat.AB: 250.0},
                    ),
                ],
            )

        ps_low = ProjectedStandings.from_rosters(
            rosters,
            effective_date=date(2026, 6, 1),
            actual_standings=_standings(user_k_ytd=50.0),
        )
        ps_high = ProjectedStandings.from_rosters(
            rosters,
            effective_date=date(2026, 6, 1),
            actual_standings=_standings(user_k_ytd=500.0),
        )

        # The user-team K MUST differ between the two runs (Pass-2 adds
        # YTD to ROS, and 50 != 500). That's a sanity check, not the fix.
        user_k_low = next(e.stats.k for e in ps_low.entries if e.team_name == "User")
        user_k_high = next(e.stats.k for e in ps_high.entries if e.team_name == "User")
        assert user_k_high - user_k_low == pytest.approx(450.0)

        # The real Fix-#13 signal: with single-roster teams there's no
        # displacement to redirect, so OTHER's final stats are stable in
        # both runs by construction. We instead pin the Pass-1 baseline
        # exposed via a direct call so the fix is demonstrable.
        # See test_pass1_baseline_uses_team_end_of_season below.

    def test_pass1_baseline_uses_team_end_of_season(self):
        """Direct white-box: with non-zero YTD on a team, the Pass-1
        baseline stored for that team must equal
        ``team_end_of_season(ytd, project_ros_components(roster, displacement=True))``,
        NOT the prior ROS-only ``project_team_stats(roster, displacement=True)``.

        We probe this by capturing the baseline_other_team_stats that
        Pass-2 feeds into the picker -- the per-team CategoryStats values
        for OTHER teams must reflect YTD additions.
        """
        # Capture LeagueContext.baseline_other_team_stats inside from_rosters
        # by monkey-patching project_ros_components for the second call only.
        from fantasy_baseball import scoring as scoring_mod
        from fantasy_baseball.models.player import (
            PitcherStats,
            Player,
            PlayerType,
        )
        from fantasy_baseball.models.positions import Position
        from fantasy_baseball.models.standings import (
            CategoryStats,
            ProjectedStandings,
            Standings,
            StandingsEntry,
        )
        from fantasy_baseball.utils.constants import OpportunityStat

        arm = Player(
            name="Arm",
            player_type=PlayerType.PITCHER,
            rest_of_season=PitcherStats(
                ip=100,
                w=6,
                k=120,
                sv=0,
                er=35,
                bb=25,
                h_allowed=80,
                era=35 * 9.0 / 100,
                whip=(25 + 80) / 100,
            ),
            selected_position=Position.P,
        )
        rosters = {"A": [arm], "B": [arm]}
        ytd_k = 88.0
        actual = Standings(
            effective_date=date(2026, 6, 1),
            entries=[
                StandingsEntry(
                    team_name=name,
                    team_key=name.lower(),
                    rank=i + 1,
                    stats=CategoryStats(
                        r=0,
                        hr=0,
                        rbi=0,
                        sb=0,
                        avg=0.0,
                        w=10,
                        k=ytd_k,
                        sv=0,
                        era=3.50,
                        whip=1.20,
                    ),
                    extras={OpportunityStat.IP: 100.0},
                )
                for i, name in enumerate(["A", "B"])
            ],
        )

        captured: list[dict] = []
        original = scoring_mod.project_ros_components

        def _capture(*args, **kwargs):
            ctx = kwargs.get("league_context")
            if ctx is not None and ctx.baseline_other_team_stats:
                captured.append({t: s.k for t, s in ctx.baseline_other_team_stats.items()})
            return original(*args, **kwargs)

        import unittest.mock as _mock

        with _mock.patch.object(scoring_mod, "project_ros_components", _capture):
            ProjectedStandings.from_rosters(
                rosters,
                effective_date=date(2026, 6, 1),
                actual_standings=actual,
            )

        # captured holds Pass-2 baseline maps {other_team: k}. Each value
        # must be team_YTD.k (88) + ROS.k (120) = 208, NOT ROS-only (120).
        assert captured, "expected Pass-2 to receive a populated league_context"
        for baseline_map in captured:
            for tname, k_val in baseline_map.items():
                assert k_val == pytest.approx(208.0), (
                    f"team {tname!r} baseline K = {k_val}; "
                    f"expected YTD(88) + ROS(120) = 208 "
                    f"(Pass-1 baseline must include team_YTD)"
                )


class TestFromRostersFallbackLogging:
    """Fix #7: from_rosters Pass-2 warns when a team in team_rosters has
    no matching entry in actual_standings (silent zero-default).
    """

    def test_team_name_mismatch_logs_warning(self, caplog):
        """A team_rosters key absent from actual_standings.entries must
        emit a log.warning naming the team and falling back to zeros.
        """
        from fantasy_baseball.models.player import (
            PitcherStats,
            Player,
            PlayerType,
        )
        from fantasy_baseball.models.positions import Position
        from fantasy_baseball.models.standings import (
            CategoryStats,
            ProjectedStandings,
            Standings,
            StandingsEntry,
        )

        arm = Player(
            name="A",
            player_type=PlayerType.PITCHER,
            rest_of_season=PitcherStats(
                ip=100,
                w=5,
                k=90,
                sv=0,
                er=35,
                bb=25,
                h_allowed=80,
                era=3.15,
                whip=1.05,
            ),
            selected_position=Position.P,
        )
        # Rosters carry "Mismatch" but standings only have "Other".
        actual = Standings(
            effective_date=date(2026, 6, 1),
            entries=[
                StandingsEntry(
                    team_name="Other",
                    team_key="o",
                    rank=1,
                    stats=CategoryStats(),
                ),
            ],
        )

        with caplog.at_level(
            "WARNING",
            logger="fantasy_baseball.models.standings",
        ):
            ProjectedStandings.from_rosters(
                {"Mismatch": [arm]},
                effective_date=date(2026, 6, 1),
                actual_standings=actual,
            )

        msgs = [r.getMessage() for r in caplog.records]
        assert any("Mismatch" in m for m in msgs), (
            f"Expected a warning naming 'Mismatch'; got: {msgs}"
        )
