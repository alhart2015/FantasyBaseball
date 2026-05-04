import math
from datetime import date

import pytest

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
    ProjectedStandingsEntry,
)
from fantasy_baseball.scoring import (
    ALL_CATS,
    LeagueContext,
    _prob_beats,
    build_team_sds,
    compute_roster_breakdown,
    project_team_sds,
    project_team_stats,
    score_roto,
)
from fantasy_baseball.utils.constants import STAT_VARIANCE, Category


def _stats_table(stats_by_team):
    """Build a :class:`ProjectedStandings` from ``{team: {cat_str: value}}``."""
    return ProjectedStandings(
        effective_date=date(2026, 4, 15),
        entries=[
            ProjectedStandingsEntry(team_name=name, stats=CategoryStats.from_dict(stats))
            for name, stats in stats_by_team.items()
        ],
    )


def _hitter(
    name, r=0, hr=0, rbi=0, sb=0, h=0, ab=0, pa=0, positions=None, selected_position=None, status=""
):
    return Player(
        name=name,
        player_type=PlayerType.HITTER,
        positions=positions or [],
        rest_of_season=HitterStats(r=r, hr=hr, rbi=rbi, sb=sb, h=h, ab=ab, pa=pa or ab),
        selected_position=selected_position,
        status=status,
    )


def _pitcher(
    name,
    w=0,
    k=0,
    sv=0,
    ip=0,
    er=0,
    bb=0,
    h_allowed=0,
    positions=None,
    selected_position=None,
    status="",
):
    return Player(
        name=name,
        player_type=PlayerType.PITCHER,
        positions=positions or [],
        rest_of_season=PitcherStats(w=w, k=k, sv=sv, ip=ip, er=er, bb=bb, h_allowed=h_allowed),
        selected_position=selected_position,
        status=status,
    )


class TestProjectTeamStats:
    def test_pitcher_with_util_position_contributes_pitching_stats(self):
        """Regression: pitchers whose Yahoo positions include Util must still
        contribute pitching stats, not be silently dropped or misrouted as
        hitters.  This bug caused 20+ point swings in projected standings
        when Gerrit Cole, Shane Bieber, etc. had their pitching zeroed out.
        """
        roster = [
            _hitter("Hitter A", r=100, hr=30, rbi=90, sb=10, h=150, ab=550),
            _pitcher("Pitcher With Util", w=15, k=200, sv=0, ip=180, er=60, bb=50, h_allowed=150),
        ]
        stats = project_team_stats(roster)
        assert stats[Category.W] == 15
        assert stats[Category.K] == 200
        assert stats[Category.ERA] == pytest.approx(60 * 9 / 180)

    def test_hitter_and_pitcher_both_counted(self):
        roster = [
            _hitter("Hitter", r=80, hr=25, rbi=70, sb=5, h=130, ab=500),
            _pitcher("Pitcher", w=10, k=150, sv=30, ip=60, er=20, bb=15, h_allowed=50),
        ]
        stats = project_team_stats(roster)
        assert stats[Category.R] == 80
        assert stats[Category.HR] == 25
        assert stats[Category.W] == 10
        assert stats[Category.SV] == 30
        assert stats[Category.AVG] == pytest.approx(130 / 500)
        assert stats[Category.ERA] == pytest.approx(20 * 9 / 60)
        assert stats[Category.WHIP] == pytest.approx((15 + 50) / 60)

    def test_empty_roster(self):
        stats = project_team_stats([])
        assert stats[Category.R] == 0
        assert stats[Category.AVG] == 0
        assert stats[Category.ERA] == 99
        assert stats[Category.WHIP] == 99

    def test_pitchers_only(self):
        roster = [
            _pitcher("SP", w=12, k=180, sv=0, ip=200, er=70, bb=50, h_allowed=170),
        ]
        stats = project_team_stats(roster)
        assert stats[Category.R] == 0
        assert stats[Category.AVG] == 0
        assert stats[Category.W] == 12

    def test_hitters_only(self):
        roster = [
            _hitter("H", r=90, hr=35, rbi=100, sb=15, h=160, ab=580),
        ]
        stats = project_team_stats(roster)
        assert stats[Category.W] == 0
        assert stats[Category.ERA] == 99
        assert stats[Category.R] == 90

    def test_player_without_ros_is_skipped(self):
        """Players unmatched to projections have ``rest_of_season=None`` and should
        contribute nothing to team totals rather than raising."""
        roster = [
            _hitter("Good", r=80, hr=25, rbi=70, sb=5, h=130, ab=500),
            Player(name="Unmatched", player_type=PlayerType.HITTER, rest_of_season=None),
        ]
        stats = project_team_stats(roster)
        assert stats[Category.R] == 80
        assert stats[Category.HR] == 25
        assert stats[Category.AVG] == pytest.approx(130 / 500)

    def test_project_team_stats_sums_ros_only_by_default(self):
        """``project_team_stats`` sums ``Player.rest_of_season`` by default (ROS-only).

        A hot-YTD player and a cold-YTD player with identical ROS-remaining
        must produce identical contributions. This is the user-facing fix for
        the lineup optimizer: forward-looking decisions should be made on
        ROS-remaining, not on full-season totals where YTD is locked.
        """
        ros = HitterStats(r=70, hr=20, rbi=60, sb=5, h=100, ab=400, pa=440)
        hot = Player(
            name="Hot",
            player_type=PlayerType.HITTER,
            rest_of_season=ros,
            full_season_projection=HitterStats(r=100, hr=28, rbi=85, sb=7, h=140, ab=520, pa=580),
        )
        cold = Player(
            name="Cold",
            player_type=PlayerType.HITTER,
            rest_of_season=ros,
            full_season_projection=HitterStats(r=73, hr=21, rbi=62, sb=5, h=103, ab=410, pa=450),
        )
        # Default behavior: read rest_of_season → identical
        assert project_team_stats([hot])[Category.R] == 70
        assert project_team_stats([hot])[Category.R] == project_team_stats([cold])[Category.R]

    def test_project_team_stats_full_season_source_for_standings(self):
        """``project_team_stats`` with ``projection_source='full_season_projection'``
        sums full-season (used by ``ProjectedStandings.from_rosters`` to preserve
        end-of-season standings projection until proper standings + ROS combination
        lands in a follow-up phase)."""
        p = Player(
            name="X",
            player_type=PlayerType.HITTER,
            rest_of_season=HitterStats(r=70, hr=20, rbi=60, sb=5, h=100, ab=400, pa=440),
            full_season_projection=HitterStats(r=100, hr=28, rbi=85, sb=7, h=140, ab=520, pa=580),
        )
        stats = project_team_stats([p], projection_source="full_season_projection")
        assert stats[Category.R] == 100
        assert stats[Category.HR] == 28

    def test_project_team_stats_falls_back_to_rest_of_season_when_full_missing(self):
        """When a Player has only ``rest_of_season`` set (no
        ``full_season_projection``), ``projection_source='full_season_projection'``
        must fall back to ``rest_of_season``. Without this, preseason rosters —
        whose matcher only writes to ``rest_of_season`` — produce a board of
        zeros for the preseason-standings widget."""
        p = Player(
            name="Preseason Hitter",
            player_type=PlayerType.HITTER,
            rest_of_season=HitterStats(r=100, hr=30, rbi=90, sb=10, h=140, ab=520, pa=580),
            full_season_projection=None,
        )
        stats = project_team_stats([p], projection_source="full_season_projection")
        assert stats[Category.R] == 100
        assert stats[Category.HR] == 30


class TestProbBeats:
    """Unit tests for the pairwise Gaussian win-probability helper."""

    def test_equal_means_zero_sd_returns_half(self):
        assert _prob_beats(100, 100, 0, 0, higher_is_better=True) == 0.5

    def test_deterministic_win_when_ahead_with_zero_sd(self):
        assert _prob_beats(110, 100, 0, 0, higher_is_better=True) == 1.0

    def test_deterministic_loss_when_behind_with_zero_sd(self):
        assert _prob_beats(90, 100, 0, 0, higher_is_better=True) == 0.0

    def test_equal_means_positive_sd_returns_half(self):
        assert _prob_beats(100, 100, 10, 10, higher_is_better=True) == pytest.approx(0.5)

    def test_one_sd_ahead_equal_variance(self):
        # μ_a - μ_b = 14.14, combined sd = sqrt(100 + 100) = 14.14
        # z = 14.14 / 14.14 = 1.0 → Φ(1.0) ≈ 0.8413
        assert _prob_beats(114.14, 100, 10, 10, higher_is_better=True) == pytest.approx(
            0.8413, abs=1e-3
        )

    def test_inverse_flips_direction(self):
        # Lower is better: A has smaller μ, so A "beats" B.
        assert _prob_beats(3.50, 4.00, 0, 0, higher_is_better=False) == 1.0
        assert _prob_beats(4.00, 3.50, 0, 0, higher_is_better=False) == 0.0

    def test_inverse_equal_means_still_half(self):
        assert _prob_beats(3.75, 3.75, 0.2, 0.2, higher_is_better=False) == pytest.approx(0.5)

    def test_zero_sd_with_negative_diff_returns_zero(self):
        # Degenerate case: combined sd == 0 and μ_a < μ_b.
        assert _prob_beats(50, 100, 0, 0, higher_is_better=True) == 0.0

    def test_complementary_probabilities_sum_to_one(self):
        # P(A > B) + P(B > A) == 1 when means differ.
        p_ab = _prob_beats(110, 100, 5, 5, higher_is_better=True)
        p_ba = _prob_beats(100, 110, 5, 5, higher_is_better=True)
        assert p_ab + p_ba == pytest.approx(1.0)


def _make_hitter(name, **stats):
    """Build a Player with HitterStats for unit tests."""
    return Player(
        name=name,
        player_type=PlayerType.HITTER,
        positions=["OF"],
        selected_position="OF",
        status="",
        rest_of_season=HitterStats(**stats),
    )


def _make_pitcher(name, **stats):
    return Player(
        name=name,
        player_type=PlayerType.PITCHER,
        positions=["SP"],
        selected_position="SP",
        status="",
        rest_of_season=PitcherStats(**stats),
    )


class TestProjectTeamSDs:
    """Per-team-per-category SD from analytical variance propagation."""

    def test_empty_roster_returns_zeros(self):
        sds = project_team_sds([])
        for cat in ALL_CATS:
            assert sds[cat] == 0.0

    def test_single_hitter_counting_stat(self):
        p = _make_hitter("A", r=80, hr=20, rbi=70, sb=10, h=150, ab=500)
        sds = project_team_sds([p])
        # SD_R = CV_r * sqrt(r^2) = CV_r * r  (single player case)
        assert sds[Category.R] == pytest.approx(STAT_VARIANCE["r"] * 80)
        assert sds[Category.HR] == pytest.approx(STAT_VARIANCE["hr"] * 20)

    def test_independence_aggregates_in_quadrature(self):
        a = _make_hitter("A", r=100, hr=0, rbi=0, sb=0, h=0, ab=0)
        b = _make_hitter("B", r=60, hr=0, rbi=0, sb=0, h=0, ab=0)
        sds = project_team_sds([a, b])
        expected = STAT_VARIANCE["r"] * math.sqrt(100**2 + 60**2)
        assert sds[Category.R] == pytest.approx(expected)

    def test_avg_uses_hits_variance_over_total_ab(self):
        a = _make_hitter("A", r=0, hr=0, rbi=0, sb=0, h=150, ab=500)
        b = _make_hitter("B", r=0, hr=0, rbi=0, sb=0, h=100, ab=400)
        sds = project_team_sds([a, b])
        expected = STAT_VARIANCE["h"] * math.sqrt(150**2 + 100**2) / (500 + 400)
        assert sds[Category.AVG] == pytest.approx(expected)

    def test_era_scales_by_nine_over_ip(self):
        a = _make_pitcher("A", w=10, k=180, sv=0, ip=180, er=60, bb=40, h_allowed=140)
        b = _make_pitcher("B", w=8, k=140, sv=0, ip=150, er=55, bb=35, h_allowed=130)
        sds = project_team_sds([a, b])
        expected = 9.0 * STAT_VARIANCE["er"] * math.sqrt(60**2 + 55**2) / (180 + 150)
        assert sds[Category.ERA] == pytest.approx(expected)

    def test_whip_combines_bb_and_h_allowed_variance(self):
        a = _make_pitcher("A", w=0, k=0, sv=0, ip=100, er=0, bb=30, h_allowed=90)
        sds = project_team_sds([a])
        expected = (
            math.sqrt(STAT_VARIANCE["bb"] ** 2 * 30**2 + STAT_VARIANCE["h_allowed"] ** 2 * 90**2)
            / 100
        )
        assert sds[Category.WHIP] == pytest.approx(expected)

    def test_all_ten_categories_present(self):
        p = _make_hitter("A", r=50, hr=10, rbi=40, sb=5, h=100, ab=400)
        sds = project_team_sds([p])
        assert set(sds.keys()) == set(ALL_CATS)

    def test_displacement_kwarg_defaults_true(self):
        # Bench players excluded by default. A bench-slot hitter should
        # not contribute to SDs when displacement=True (default).
        active = _make_hitter("A", r=80, hr=20, rbi=70, sb=10, h=150, ab=500)
        bench = Player(
            name="B",
            player_type=PlayerType.HITTER,
            positions=["OF"],
            selected_position="BN",
            status="",
            rest_of_season=HitterStats(r=80, hr=20, rbi=70, sb=10, h=150, ab=500),
        )
        sds_with_bench = project_team_sds([active, bench])
        sds_active_only = project_team_sds([active])
        assert sds_with_bench[Category.R] == pytest.approx(sds_active_only[Category.R])


class TestScoreRoto:
    def test_two_teams_simple(self):
        stats = _stats_table(
            {
                "A": {
                    "R": 900,
                    "HR": 250,
                    "RBI": 850,
                    "SB": 100,
                    "AVG": 0.270,
                    "W": 80,
                    "K": 1200,
                    "SV": 50,
                    "ERA": 3.50,
                    "WHIP": 1.15,
                },
                "B": {
                    "R": 800,
                    "HR": 200,
                    "RBI": 750,
                    "SB": 80,
                    "AVG": 0.260,
                    "W": 70,
                    "K": 1100,
                    "SV": 40,
                    "ERA": 4.00,
                    "WHIP": 1.25,
                },
            }
        )
        roto = score_roto(stats)
        assert roto["A"].total == 20  # wins every category
        assert roto["B"].total == 10

    def test_fractional_tiebreaker(self):
        stats = _stats_table(
            {
                "A": {
                    "R": 900,
                    "HR": 250,
                    "RBI": 850,
                    "SB": 100,
                    "AVG": 0.270,
                    "W": 80,
                    "K": 1200,
                    "SV": 50,
                    "ERA": 3.50,
                    "WHIP": 1.15,
                },
                "B": {
                    "R": 900,
                    "HR": 250,
                    "RBI": 850,
                    "SB": 100,
                    "AVG": 0.270,
                    "W": 80,
                    "K": 1200,
                    "SV": 50,
                    "ERA": 3.50,
                    "WHIP": 1.15,
                },
            }
        )
        roto = score_roto(stats)
        # Tied in everything — both get 1.5 per cat (avg of 1 and 2)
        assert roto["A"].total == pytest.approx(15.0)
        assert roto["B"].total == pytest.approx(15.0)

    def test_inverse_stats_lower_is_better(self):
        stats = _stats_table(
            {
                "A": {
                    "R": 0,
                    "HR": 0,
                    "RBI": 0,
                    "SB": 0,
                    "AVG": 0,
                    "W": 0,
                    "K": 0,
                    "SV": 0,
                    "ERA": 3.00,
                    "WHIP": 1.10,
                },
                "B": {
                    "R": 0,
                    "HR": 0,
                    "RBI": 0,
                    "SB": 0,
                    "AVG": 0,
                    "W": 0,
                    "K": 0,
                    "SV": 0,
                    "ERA": 4.50,
                    "WHIP": 1.30,
                },
            }
        )
        roto = score_roto(stats)
        assert roto["A"][Category.ERA] == 2  # lower ERA = better = more points
        assert roto["B"][Category.ERA] == 1

    def test_all_categories_present(self):
        stats = _stats_table({"A": {c.value: 1 for c in ALL_CATS}})
        roto = score_roto(stats)
        for c in ALL_CATS:
            assert c in roto["A"].values
        assert roto["A"].total == pytest.approx(sum(roto["A"].values.values()))


# ── Displacement tests ──────────────────────────────────────────────


class TestDisplacementOff:
    """When displacement=False (default), bench/IL players are summed naively."""

    def test_default_displacement_false(self):
        """displacement defaults to False; bench players are counted."""
        bench = _hitter(
            "Bench Guy", r=50, hr=10, rbi=40, sb=5, h=80, ab=300, selected_position=Position.BN
        )
        active = _hitter(
            "Active",
            r=80,
            hr=20,
            rbi=70,
            sb=10,
            h=140,
            ab=500,
            selected_position=Position.OF,
            positions=[Position.OF],
        )
        stats = project_team_stats([active, bench])
        assert stats[Category.R] == 130  # 80 + 50
        assert stats[Category.HR] == 30  # 20 + 10


class TestDisplacementBenchExclusion:
    """Bench players (BN slot, not IL) are excluded when displacement=True."""

    def test_bench_hitter_excluded(self):
        bench = _hitter(
            "Bench", r=50, hr=10, rbi=40, sb=5, h=80, ab=300, selected_position=Position.BN
        )
        active = _hitter(
            "Active",
            r=80,
            hr=20,
            rbi=70,
            sb=10,
            h=140,
            ab=500,
            selected_position=Position.OF,
            positions=[Position.OF],
        )
        stats = project_team_stats([active, bench], displacement=True)
        assert stats[Category.R] == 80
        assert stats[Category.HR] == 20

    def test_bench_pitcher_excluded(self):
        bench = _pitcher(
            "BenchP",
            w=5,
            k=60,
            sv=0,
            ip=80,
            er=30,
            bb=20,
            h_allowed=70,
            selected_position=Position.BN,
        )
        active = _pitcher(
            "ActiveP",
            w=10,
            k=150,
            sv=0,
            ip=180,
            er=60,
            bb=50,
            h_allowed=150,
            selected_position=Position.SP,
            positions=[Position.SP],
        )
        stats = project_team_stats([active, bench], displacement=True)
        assert stats[Category.W] == 10
        assert stats[Category.K] == 150

    def test_il_player_on_bench_slot_with_il_status_not_excluded_as_bench(self):
        """A player on BN slot but with IL status is NOT treated as bench —
        they're treated as IL (for displacement purposes)."""
        il_player = _hitter(
            "IL Guy",
            r=40,
            hr=8,
            rbi=30,
            sb=3,
            h=60,
            ab=200,
            selected_position=Position.BN,
            status="IL",
        )
        active = _hitter(
            "Active",
            r=80,
            hr=20,
            rbi=70,
            sb=10,
            h=140,
            ab=500,
            selected_position=Position.OF,
            positions=[Position.OF],
        )
        # IL Guy is not excluded as bench (has IL status), so displacement
        # logic applies instead. Since Active is the only option, his stats
        # get scaled down. IL Guy counted at full scale.
        stats = project_team_stats([active, il_player], displacement=True)
        # Active (500 ab) displaced by IL Guy (200 ab) -> factor = (500-200)/500 = 0.6
        # Total R = Active*0.6 + IL Guy full = 48 + 40 = 88
        assert stats[Category.R] == pytest.approx(80 * 0.6 + 40)


class TestDisplacementILHitter:
    """IL hitter displaces worst positional match among active hitters."""

    def test_basic_hitter_displacement(self):
        """IL hitter displaces worst active hitter sharing a position."""
        # Active hitters: one good OF, one bad OF
        good_of = _hitter(
            "Good OF",
            r=90,
            hr=30,
            rbi=90,
            sb=10,
            h=160,
            ab=550,
            positions=[Position.OF],
            selected_position=Position.OF,
        )
        bad_of = _hitter(
            "Bad OF",
            r=40,
            hr=8,
            rbi=30,
            sb=2,
            h=80,
            ab=300,
            positions=[Position.OF],
            selected_position=Position.OF,
        )
        # IL hitter: OF eligible, 200 PA on IL
        il_of = _hitter(
            "IL OF",
            r=30,
            hr=5,
            rbi=20,
            sb=1,
            h=50,
            ab=200,
            positions=[Position.OF],
            selected_position=Position.IL,
            status="IL",
        )

        stats = project_team_stats([good_of, bad_of, il_of], displacement=True)

        # bad_of displaced: factor = max(0, 300 - 200) / 300 = 1/3
        # Totals: good_of full + bad_of scaled + IL full
        assert stats[Category.R] == pytest.approx(90 + 40 / 3 + 30)
        assert stats[Category.HR] == pytest.approx(30 + 8 / 3 + 5)
        assert stats[Category.RBI] == pytest.approx(90 + 30 / 3 + 20)

    def test_il_hitter_fallback_to_worst_hitter_overall(self):
        """When no active hitter shares a position, fallback to worst hitter."""
        ss = _hitter(
            "SS guy",
            r=50,
            hr=10,
            rbi=40,
            sb=5,
            h=90,
            ab=350,
            positions=[Position.SS],
            selected_position=Position.SS,
        )
        first = _hitter(
            "1B guy",
            r=70,
            hr=25,
            rbi=80,
            sb=2,
            h=130,
            ab=480,
            positions=[Position.FIRST_BASE],
            selected_position=Position.FIRST_BASE,
        )
        # IL hitter is OF eligible — no active OF exists
        il_of = _hitter(
            "IL OF",
            r=20,
            hr=4,
            rbi=15,
            sb=1,
            h=40,
            ab=150,
            positions=[Position.OF],
            selected_position=Position.IL,
            status="IL10",
        )

        stats = project_team_stats([ss, first, il_of], displacement=True)

        # Fallback: displace worst hitter overall. SS has lower SGP than 1B.
        # SS factor = max(0, 350 - 150) / 350 = 200/350 = 4/7
        # Total R = 1B full + SS scaled + IL full
        assert stats[Category.R] == pytest.approx(70 + 50 * (4 / 7) + 20)

    def test_displacement_caps_at_zero(self):
        """When IL player has more playing time than active, factor is 0."""
        active = _hitter(
            "Active",
            r=40,
            hr=8,
            rbi=30,
            sb=2,
            h=60,
            ab=200,
            positions=[Position.OF],
            selected_position=Position.OF,
        )
        il_player = _hitter(
            "IL Big",
            r=50,
            hr=10,
            rbi=40,
            sb=5,
            h=100,
            ab=400,
            positions=[Position.OF],
            selected_position=Position.IL,
            status="IL60",
        )

        stats = project_team_stats([active, il_player], displacement=True)

        # factor = max(0, 200 - 400) / 200 = 0
        # Active zeroed out, IL counted in full
        assert stats[Category.R] == 50
        assert stats[Category.HR] == 10

    def test_each_active_displaced_at_most_once(self):
        """Two IL hitters can't both displace the same active player."""
        active = _hitter(
            "Only Active",
            r=80,
            hr=20,
            rbi=70,
            sb=10,
            h=140,
            ab=500,
            positions=[Position.OF],
            selected_position=Position.OF,
        )
        il1 = _hitter(
            "IL1",
            r=30,
            hr=5,
            rbi=20,
            sb=1,
            h=50,
            ab=200,
            positions=[Position.OF],
            selected_position=Position.IL,
            status="IL",
        )
        il2 = _hitter(
            "IL2",
            r=20,
            hr=3,
            rbi=15,
            sb=1,
            h=40,
            ab=150,
            positions=[Position.OF],
            selected_position=Position.IL_PLUS,
            status="IL+",
        )

        stats = project_team_stats([active, il1, il2], displacement=True)

        # IL1 has more playing time (200 ab > 150 ab), processed first.
        # active displaced by IL1: factor = (500 - 200) / 500 = 0.6
        # IL2 has no remaining active to displace (only one active, already displaced).
        # Total = active*0.6 + IL1 full + IL2 full
        assert stats[Category.R] == pytest.approx(80 * 0.6 + 30 + 20)
        assert stats[Category.HR] == pytest.approx(20 * 0.6 + 5 + 3)


class TestDisplacementILPitcher:
    """IL pitcher displaces worst active pitcher matching SP/RP role."""

    def test_sp_displaces_sp(self):
        """IL SP (ip>100) displaces worst active SP."""
        good_sp = _pitcher(
            "Good SP",
            w=15,
            k=200,
            sv=0,
            ip=190,
            er=55,
            bb=40,
            h_allowed=150,
            positions=[Position.SP],
            selected_position=Position.SP,
        )
        bad_sp = _pitcher(
            "Bad SP",
            w=5,
            k=80,
            sv=0,
            ip=120,
            er=55,
            bb=40,
            h_allowed=110,
            positions=[Position.SP],
            selected_position=Position.SP,
        )
        il_sp = _pitcher(
            "IL SP",
            w=8,
            k=100,
            sv=0,
            ip=130,
            er=40,
            bb=30,
            h_allowed=100,
            positions=[Position.SP],
            selected_position=Position.IL,
            status="IL15",
        )

        stats = project_team_stats([good_sp, bad_sp, il_sp], displacement=True)

        # bad_sp displaced: factor = max(0, 120 - 130) / 120 = 0
        # Total = good_sp + bad_sp*0 + il_sp full
        assert stats[Category.W] == pytest.approx(15 + 8)
        assert stats[Category.K] == pytest.approx(200 + 100)

    def test_rp_displaces_rp(self):
        """IL RP (ip<=100) displaces worst active RP, not an SP."""
        sp = _pitcher(
            "SP",
            w=12,
            k=180,
            sv=0,
            ip=180,
            er=60,
            bb=45,
            h_allowed=150,
            positions=[Position.SP],
            selected_position=Position.SP,
        )
        rp = _pitcher(
            "RP",
            w=3,
            k=50,
            sv=20,
            ip=60,
            er=20,
            bb=15,
            h_allowed=50,
            positions=[Position.RP],
            selected_position=Position.RP,
        )
        il_rp = _pitcher(
            "IL RP",
            w=1,
            k=20,
            sv=10,
            ip=30,
            er=10,
            bb=8,
            h_allowed=25,
            positions=[Position.RP],
            selected_position=Position.IL,
            status="IL",
        )

        stats = project_team_stats([sp, rp, il_rp], displacement=True)

        # RP displaced: factor = max(0, 60 - 30) / 60 = 0.5
        # Total = SP full + RP*0.5 + IL RP full
        assert stats[Category.W] == pytest.approx(12 + 3 * 0.5 + 1)
        assert stats[Category.SV] == pytest.approx(20 * 0.5 + 10)


class TestDeltaRotoDisplacement:
    """ΔRoto-optimal displacement: when a LeagueContext is provided,
    the picker chooses the active player whose displacement preserves
    the highest team roto pts — not the lowest-SGP candidate.

    This fixes the elite-low-volume-closer problem (the "Mason Miller"
    case): SGP is volume-weighted, so an elite RP with 50 IP has lower
    total SGP than a struggling 150-IP starter, and the legacy picker
    would zero out the closer to make room for an IL SP. ΔRoto sees
    that the closer's saves are worth more roto pts than the SP's
    marginal innings, and picks the SP for displacement instead.
    """

    def _other_team(self, name: str, *, w=70, k=1300, sv=50, era=4.0, whip=1.25):
        """Build a single competitor with hand-set pitching totals.

        Hitting stats fixed at league-average levels; pitching numbers
        vary so the test team's roster decisions actually move the
        roto landscape (otherwise every category is a wash and every
        candidate looks identical).
        """
        return ProjectedStandingsEntry(
            team_name=name,
            stats=CategoryStats(
                r=800,
                hr=220,
                rbi=770,
                sb=120,
                avg=0.255,
                w=w,
                k=k,
                sv=sv,
                era=era,
                whip=whip,
            ),
        )

    def _league_context_for(self, my_team_name: str, n_other_teams: int = 9) -> LeagueContext:
        """Build a LeagueContext with n_other_teams competitors spanning a
        believable roto landscape, so SV/W/ERA differences for the test
        team produce non-degenerate ΔRoto signal.
        """
        # Spread competitors across the SV / ERA / W landscape so the test
        # team's choices have leverage in multiple categories.
        configs = [
            {"w": 60, "k": 1200, "sv": 30, "era": 4.5, "whip": 1.32},
            {"w": 65, "k": 1250, "sv": 40, "era": 4.2, "whip": 1.27},
            {"w": 70, "k": 1300, "sv": 45, "era": 4.0, "whip": 1.25},
            {"w": 75, "k": 1350, "sv": 50, "era": 3.85, "whip": 1.22},
            {"w": 80, "k": 1400, "sv": 55, "era": 3.7, "whip": 1.20},
            {"w": 72, "k": 1280, "sv": 35, "era": 4.1, "whip": 1.26},
            {"w": 68, "k": 1320, "sv": 48, "era": 3.95, "whip": 1.23},
            {"w": 78, "k": 1380, "sv": 52, "era": 3.78, "whip": 1.21},
            {"w": 73, "k": 1290, "sv": 42, "era": 4.05, "whip": 1.24},
        ][:n_other_teams]
        baseline = {
            f"Other {i + 1}": self._other_team(f"Other {i + 1}", **cfg).stats
            for i, cfg in enumerate(configs)
        }
        # SDs per team — use small constants to make the Gaussian sigmoid
        # smooth without dominating mu differences. Real refresh path uses
        # build_team_sds; this simpler shape is sufficient for the picker.
        team_sds = {t: dict.fromkeys(ALL_CATS, 5.0) for t in [*baseline.keys(), my_team_name]}
        return LeagueContext(
            baseline_other_team_stats=baseline,
            team_sds=team_sds,
            team_name=my_team_name,
        )

    def test_il_sp_does_not_zero_elite_rp(self):
        """The Mason Miller fix: an IL SP returning displaces the worst SP,
        not the elite low-volume closer.

        Without league_context (legacy SGP picker): elite_rp gets zeroed
        because its total SGP is lower than even a struggling SP's
        volume-weighted SGP. With league_context: elite_rp keeps full
        contribution, weak_sp absorbs the displacement.
        """
        elite_rp = _pitcher(
            "Elite RP",
            w=4,
            k=95,
            sv=40,
            ip=70,
            er=12,
            bb=15,
            h_allowed=45,
            positions=[Position.RP],
            selected_position=Position.RP,
        )
        weak_sp = _pitcher(
            "Weak SP",
            w=4,
            k=85,
            sv=0,
            ip=130,
            er=87,
            bb=60,
            h_allowed=165,
            positions=[Position.SP],
            selected_position=Position.SP,
        )
        good_sp = _pitcher(
            "Good SP",
            w=14,
            k=190,
            sv=0,
            ip=190,
            er=70,
            bb=50,
            h_allowed=160,
            positions=[Position.SP],
            selected_position=Position.SP,
        )
        # Fill in some hitters so the team has non-degenerate hitting totals
        # — without them, AVG comes back as 0/0 = 0 and the picker sees
        # collapsed Gaussians on the rate cats.
        hitters = [
            _hitter(
                f"Hitter {i}",
                r=80,
                hr=22,
                rbi=77,
                sb=12,
                h=140,
                ab=540,
                positions=[Position.OF],
                selected_position=Position.OF,
            )
            for i in range(10)
        ]
        il_sp = _pitcher(
            "IL SP",
            w=10,
            k=140,
            sv=0,
            ip=150,
            er=60,
            bb=40,
            h_allowed=130,
            positions=[Position.SP],
            selected_position=Position.IL,
            status="IL15",
        )

        roster = [elite_rp, weak_sp, good_sp, *hitters, il_sp]
        ctx = self._league_context_for("My Team")

        # Legacy SGP behavior: weak_sp has lower SGP than elite_rp here
        # (volume-weighted), so SGP picks weak_sp for the SP-role match —
        # but if Hader-style RP-IL was the case, elite_rp would be picked.
        # This test focuses on the SP-IL case to isolate the ΔRoto win:
        # both pickers should pick weak_sp here. So we use compute_roster_
        # breakdown to assert ΔRoto path doesn't accidentally pick elite_rp.
        breakdown_with_ctx = compute_roster_breakdown(
            "My Team",
            roster,
            league_context=ctx,
            projection_source="rest_of_season",
        )
        elite_contrib = next(p for p in breakdown_with_ctx.pitchers if p.name == "Elite RP")
        assert elite_contrib.scale_factor == 1.0, (
            f"ΔRoto picker should not displace the elite closer; "
            f"got sf={elite_contrib.scale_factor}"
        )

    def test_il_rp_uses_delta_roto_picker(self):
        """When an IL RP returns and there's only one active RP (an elite
        closer), the legacy SGP picker would displace the closer (it's the
        only role-match). The ΔRoto picker can pick a struggling SP instead
        if losing the SP costs fewer roto pts than losing the SV-locked
        closer.
        """
        elite_rp = _pitcher(
            "Elite Closer",
            w=4,
            k=95,
            sv=40,
            ip=70,
            er=12,
            bb=15,
            h_allowed=45,
            positions=[Position.RP],
            selected_position=Position.RP,
        )
        weak_sp = _pitcher(
            "Replaceable SP",
            w=5,
            k=110,
            sv=0,
            ip=140,
            er=85,  # ~5.46 ERA
            bb=55,
            h_allowed=160,
            positions=[Position.SP],
            selected_position=Position.SP,
        )
        good_sp = _pitcher(
            "Good SP",
            w=14,
            k=190,
            sv=0,
            ip=190,
            er=70,
            bb=50,
            h_allowed=160,
            positions=[Position.SP],
            selected_position=Position.SP,
        )
        il_rp = _pitcher(
            "Returning Closer",
            w=2,
            k=35,
            sv=18,
            ip=40,
            er=12,
            bb=12,
            h_allowed=30,
            positions=[Position.RP],
            selected_position=Position.IL,
            status="IL",
        )
        # Need hitters so AVG isn't degenerate.
        hitters = [
            _hitter(
                f"Hitter {i}",
                r=80,
                hr=22,
                rbi=77,
                sb=12,
                h=140,
                ab=540,
                positions=[Position.OF],
                selected_position=Position.OF,
            )
            for i in range(10)
        ]
        roster = [elite_rp, weak_sp, good_sp, *hitters, il_rp]

        # Without league_context: SGP picker forces RP role match → elite_rp
        # gets scaled by (70 - 40) / 70 ≈ 0.43.
        breakdown_sgp = compute_roster_breakdown("My Team", roster)
        elite_sgp = next(p for p in breakdown_sgp.pitchers if p.name == "Elite Closer")
        assert elite_sgp.scale_factor < 1.0, "SGP picker should still displace elite_rp"

        # With league_context: ΔRoto picker should choose the cheaper-to-lose
        # weak_sp instead, leaving the elite closer at full scale.
        ctx = self._league_context_for("My Team")
        breakdown_dr = compute_roster_breakdown(
            "My Team",
            roster,
            league_context=ctx,
            projection_source="rest_of_season",
        )
        elite_dr = next(p for p in breakdown_dr.pitchers if p.name == "Elite Closer")
        weak_dr = next(p for p in breakdown_dr.pitchers if p.name == "Replaceable SP")
        assert elite_dr.scale_factor == 1.0, (
            f"ΔRoto picker should preserve the elite closer; got sf={elite_dr.scale_factor}"
        )
        assert weak_dr.scale_factor < 1.0, (
            f"ΔRoto picker should displace the weak SP instead; got sf={weak_dr.scale_factor}"
        )

    def test_no_league_context_preserves_legacy_behavior(self):
        """Calling project_team_stats and compute_roster_breakdown without
        league_context must produce results identical to the pre-Phase-2
        SGP path (backwards compat for optimizer / draft / trade evaluator).
        """
        # Use the same fixture as the SP-displaces-SP test to anchor on a
        # known-good legacy result.
        good_sp = _pitcher(
            "Good SP",
            w=15,
            k=200,
            sv=0,
            ip=190,
            er=55,
            bb=40,
            h_allowed=150,
            positions=[Position.SP],
            selected_position=Position.SP,
        )
        bad_sp = _pitcher(
            "Bad SP",
            w=5,
            k=80,
            sv=0,
            ip=120,
            er=55,
            bb=40,
            h_allowed=110,
            positions=[Position.SP],
            selected_position=Position.SP,
        )
        il_sp = _pitcher(
            "IL SP",
            w=8,
            k=100,
            sv=0,
            ip=130,
            er=40,
            bb=30,
            h_allowed=100,
            positions=[Position.SP],
            selected_position=Position.IL,
            status="IL15",
        )

        # Legacy assertion from test_sp_displaces_sp must still hold.
        stats = project_team_stats([good_sp, bad_sp, il_sp], displacement=True)
        assert stats[Category.W] == pytest.approx(15 + 8)
        assert stats[Category.K] == pytest.approx(200 + 100)


class TestProjectedStandingsTwoPass:
    """``ProjectedStandings.from_rosters`` is the Phase 2 opt-in site for
    ΔRoto-optimal displacement. The two-pass build (SGP baseline →
    ΔRoto-aware) should produce sane standings without crashing on
    rosters with multiple IL pitchers, and should differ from a hand-
    rolled single-pass-SGP build in the expected direction (elite
    closers preserved).
    """

    def test_two_pass_does_not_crash_on_realistic_league(self):
        """Sanity: a 10-team league with some IL pitchers per team builds
        cleanly via the two-pass path."""

        def make_team(suffix: str, *, with_il_sp: bool = False, with_il_rp: bool = False):
            roster = [
                _pitcher(
                    f"SP1 {suffix}",
                    w=12,
                    k=180,
                    sv=0,
                    ip=180,
                    er=68,
                    bb=50,
                    h_allowed=160,
                    positions=[Position.SP],
                    selected_position=Position.SP,
                ),
                _pitcher(
                    f"SP2 {suffix}",
                    w=10,
                    k=150,
                    sv=0,
                    ip=160,
                    er=75,
                    bb=55,
                    h_allowed=155,
                    positions=[Position.SP],
                    selected_position=Position.SP,
                ),
                _pitcher(
                    f"RP {suffix}",
                    w=4,
                    k=90,
                    sv=30,
                    ip=70,
                    er=22,
                    bb=20,
                    h_allowed=55,
                    positions=[Position.RP],
                    selected_position=Position.RP,
                ),
            ]
            for i in range(8):
                roster.append(
                    _hitter(
                        f"H{i} {suffix}",
                        r=75,
                        hr=20,
                        rbi=70,
                        sb=10,
                        h=140,
                        ab=540,
                        positions=[Position.OF],
                        selected_position=Position.OF,
                    )
                )
            if with_il_sp:
                roster.append(
                    _pitcher(
                        f"IL SP {suffix}",
                        w=9,
                        k=130,
                        sv=0,
                        ip=140,
                        er=60,
                        bb=45,
                        h_allowed=130,
                        positions=[Position.SP],
                        selected_position=Position.IL,
                        status="IL15",
                    )
                )
            if with_il_rp:
                roster.append(
                    _pitcher(
                        f"IL RP {suffix}",
                        w=2,
                        k=40,
                        sv=15,
                        ip=35,
                        er=12,
                        bb=10,
                        h_allowed=30,
                        positions=[Position.RP],
                        selected_position=Position.IL,
                        status="IL",
                    )
                )
            return roster

        rosters = {
            f"Team {i}": make_team(str(i), with_il_sp=(i % 3 == 0), with_il_rp=(i % 4 == 0))
            for i in range(10)
        }

        # Should not raise; should return one entry per team with finite stats.
        standings = ProjectedStandings.from_rosters(rosters, effective_date=date(2026, 5, 5))
        assert len(standings.entries) == 10
        for e in standings.entries:
            assert math.isfinite(e.stats.w)
            assert math.isfinite(e.stats.k)
            assert math.isfinite(e.stats.sv)
            assert math.isfinite(e.stats.era)


class TestDisplacementClassification:
    """Slot-first classification: active slot counts at face value;
    IL slot or BN+IL-status triggers displacement."""

    def test_il_slot_triggers_displacement(self):
        """Player on IL slot is treated as IL even without status string."""
        active = _hitter(
            "Active",
            r=80,
            hr=20,
            rbi=70,
            sb=10,
            h=140,
            ab=500,
            positions=[Position.OF],
            selected_position=Position.OF,
        )
        # selected_position=IL but status="" — still counts as IL
        il_player = _hitter(
            "IL slot",
            r=20,
            hr=4,
            rbi=15,
            sb=1,
            h=40,
            ab=150,
            positions=[Position.OF],
            selected_position=Position.IL,
            status="",
        )

        stats = project_team_stats([active, il_player], displacement=True)
        # factor = (500 - 150) / 500 = 0.7
        # Total = active*0.7 + IL full
        assert stats[Category.R] == pytest.approx(80 * 0.7 + 20)

    def test_bn_slot_with_il_status_triggers_displacement(self):
        """BN slot + IL status is still routed to displacement (unchanged).

        Replaces the prior ``test_il_status_on_active_slot_triggers_displacement``,
        whose behavior was intentionally changed by the
        ``projected_standings_active_slot_face_value`` spec.
        """
        active = _hitter(
            "Active",
            r=80,
            hr=20,
            rbi=70,
            sb=10,
            h=140,
            ab=500,
            positions=[Position.OF],
            selected_position=Position.OF,
        )
        # BN slot + IL60 status — still IL-classified, still displaces.
        il_player = _hitter(
            "IL on bench",
            r=30,
            hr=6,
            rbi=20,
            sb=2,
            h=60,
            ab=250,
            positions=[Position.OF],
            selected_position=Position.BN,
            status="IL60",
        )

        stats = project_team_stats([active, il_player], displacement=True)
        # factor = (500 - 250) / 500 = 0.5
        # Total = active*0.5 + IL full
        assert stats[Category.R] == pytest.approx(80 * 0.5 + 30)

    def test_il_status_on_active_slot_counts_at_face_value(self):
        """Active-slotted player with IL status is treated as active, not IL.

        This is the fix for the Soto-in-OF-with-IL10-status bug: the manager
        put him in an active slot, so respect that — no displacement routing.
        """
        active = _hitter(
            "Active",
            r=80,
            hr=20,
            rbi=70,
            sb=10,
            h=140,
            ab=500,
            positions=[Position.OF],
            selected_position=Position.OF,
        )
        # Active slot (OF) + IL status -> treated at face value under new rule
        il_status_active = _hitter(
            "IL status, active slot",
            r=75,
            hr=25,
            rbi=85,
            sb=8,
            h=150,
            ab=540,
            positions=[Position.OF],
            selected_position=Position.OF,
            status="IL10",
        )

        stats = project_team_stats([active, il_status_active], displacement=True)
        # Both count in full — no displacement, no zeroing.
        assert stats[Category.R] == pytest.approx(80 + 75)
        assert stats[Category.HR] == pytest.approx(20 + 25)
        assert stats[Category.RBI] == pytest.approx(70 + 85)
        assert stats[Category.SB] == pytest.approx(10 + 8)


class TestDisplacementDictInputUnaffected:
    """Dict-input callers (draft scripts) bypass displacement entirely."""

    def test_dict_roster_ignores_displacement(self):
        roster = [
            {
                "player_type": PlayerType.HITTER,
                "r": 80,
                "hr": 20,
                "rbi": 70,
                "sb": 10,
                "h": 140,
                "ab": 500,
            },
            {
                "player_type": PlayerType.HITTER,
                "r": 50,
                "hr": 10,
                "rbi": 40,
                "sb": 5,
                "h": 80,
                "ab": 300,
                "selected_position": "BN",
            },
        ]
        stats = project_team_stats(roster, displacement=True)
        # Dicts are never filtered — all summed naively
        assert stats[Category.R] == 130
        assert stats[Category.HR] == 30


class TestDisplacementProcessingOrder:
    """IL players processed in descending playing time order."""

    def test_higher_playing_time_il_processed_first(self):
        """The IL player with more playing time gets first pick of displacement."""
        of1 = _hitter(
            "OF1",
            r=60,
            hr=15,
            rbi=50,
            sb=8,
            h=100,
            ab=400,
            positions=[Position.OF],
            selected_position=Position.OF,
        )
        of2 = _hitter(
            "OF2",
            r=40,
            hr=8,
            rbi=30,
            sb=2,
            h=70,
            ab=280,
            positions=[Position.OF],
            selected_position=Position.OF,
        )
        # IL1 has 300 ab, IL2 has 100 ab
        il1 = _hitter(
            "IL1",
            r=50,
            hr=12,
            rbi=40,
            sb=5,
            h=90,
            ab=300,
            positions=[Position.OF],
            selected_position=Position.IL,
            status="IL",
        )
        il2 = _hitter(
            "IL2",
            r=20,
            hr=3,
            rbi=10,
            sb=1,
            h=30,
            ab=100,
            positions=[Position.OF],
            selected_position=Position.IL,
            status="IL",
        )

        stats = project_team_stats([of1, of2, il1, il2], displacement=True)

        # IL1 (300 ab) processed first, displaces OF2 (worst SGP, 280 ab)
        # OF2 factor = max(0, 280 - 300) / 280 = 0
        # IL2 (100 ab) processed next, displaces OF1 (only remaining)
        # OF1 factor = max(0, 400 - 100) / 400 = 0.75
        # Total = OF1*0.75 + OF2*0 + IL1 full + IL2 full
        assert stats[Category.R] == pytest.approx(60 * 0.75 + 50 + 20)
        assert stats[Category.HR] == pytest.approx(15 * 0.75 + 12 + 3)


class TestDisplacementRoleMatching:
    """Position-role matching: generic slots (UTIL, IF, DH, P, BN, IL) are ignored."""

    def test_generic_positions_ignored_in_matching(self):
        """Players with only UTIL/DH/etc. in their positions list use
        fallback (worst hitter overall)."""
        active1 = _hitter(
            "1B",
            r=60,
            hr=15,
            rbi=50,
            sb=3,
            h=100,
            ab=400,
            positions=[Position.FIRST_BASE],
            selected_position=Position.FIRST_BASE,
        )
        active2 = _hitter(
            "Worst",
            r=30,
            hr=5,
            rbi=20,
            sb=1,
            h=50,
            ab=250,
            positions=[Position.SECOND_BASE],
            selected_position=Position.SECOND_BASE,
        )
        # IL player only has UTIL in positions — no "real" position overlap
        il_util = _hitter(
            "IL Util",
            r=20,
            hr=4,
            rbi=15,
            sb=1,
            h=35,
            ab=150,
            positions=[Position.UTIL],
            selected_position=Position.IL,
            status="IL",
        )

        stats = project_team_stats([active1, active2, il_util], displacement=True)

        # No position match -> fallback to worst hitter overall = active2
        # active2 factor = max(0, 250 - 150) / 250 = 0.4
        # Total = active1 full + active2*0.4 + IL full
        assert stats[Category.R] == pytest.approx(60 + 30 * 0.4 + 20)


class TestDisplacementNoRos:
    """Players with rest_of_season=None are handled gracefully."""

    def test_il_player_without_ros_no_displacement(self):
        """IL player with rest_of_season=None has 0 playing time — no displacement."""
        active = _hitter(
            "Active",
            r=80,
            hr=20,
            rbi=70,
            sb=10,
            h=140,
            ab=500,
            positions=[Position.OF],
            selected_position=Position.OF,
        )
        il_no_ros = Player(
            name="No ROS",
            player_type=PlayerType.HITTER,
            positions=[Position.OF],
            selected_position=Position.IL,
            status="IL",
            rest_of_season=None,
        )
        stats = project_team_stats([active, il_no_ros], displacement=True)
        # IL player has 0 ab -> 0 displacement
        assert stats[Category.R] == 80
        assert stats[Category.HR] == 20


def _twelve_team_dict(r_values):
    """Build ``{team: {R: value, other cats: 0}}`` for 12 teams (mutable)."""
    teams = {}
    for i, r in enumerate(r_values):
        teams[f"T{i + 1}"] = {
            "R": r,
            "HR": 0,
            "RBI": 0,
            "SB": 0,
            "AVG": 0.0,
            "W": 0,
            "K": 0,
            "SV": 0,
            "ERA": 0.0,
            "WHIP": 0.0,
        }
    return teams


def _twelve_team_stats(r_values):
    """Build a :class:`ProjectedStandings` for 12 teams."""
    return _stats_table(_twelve_team_dict(r_values))


def _all_cat_sds(teams, value):
    """Build ``{team: {Category: value}}`` for every team, every category."""
    return {t: {c: float(value) for c in ALL_CATS} for t in teams}


class TestScoreRotoEV:
    """Expected-value roto scoring with projection uncertainty."""

    def test_no_sds_matches_rank_scoring_distinct(self):
        # 12 distinct values → integer points 1..12.
        stats = _twelve_team_stats([100 + i for i in range(12)])
        roto = score_roto(stats)
        # T12 has highest R (111), gets 12 pts.
        assert roto["T12"][Category.R] == pytest.approx(12.0)
        assert roto["T1"][Category.R] == pytest.approx(1.0)

    def test_no_sds_exact_tie_averages_ranks(self):
        # Two teams tied at top: both get avg of 12 and 11 → 11.5.
        vals = [111, 111] + [100 + i for i in range(10)]
        stats = _twelve_team_stats(vals)
        roto = score_roto(stats)
        assert roto["T1"][Category.R] == pytest.approx(11.5)
        assert roto["T2"][Category.R] == pytest.approx(11.5)

    def test_no_sds_three_way_tie_averages(self):
        # Three teams tied at top: avg of 12+11+10 = 11.
        vals = [111, 111, 111] + [100 + i for i in range(9)]
        stats = _twelve_team_stats(vals)
        roto = score_roto(stats)
        for t in ["T1", "T2", "T3"]:
            assert roto[t][Category.R] == pytest.approx(11.0)

    def test_zero_sds_matches_none_path(self):
        stats = _twelve_team_stats([100 + i for i in range(12)])
        team_names = [e.team_name for e in stats.entries]
        roto_none = score_roto(stats)
        zero_sds = _all_cat_sds(team_names, 0.0)
        roto_zero = score_roto(stats, team_sds=zero_sds)
        for t in team_names:
            for cat in ALL_CATS:
                assert roto_zero[t][cat] == pytest.approx(roto_none[t][cat])

    def test_large_sds_collapse_toward_middle(self):
        # Huge sd >> any mu gap → every team's pairwise P ≈ 0.5 → pts ≈ (N+1)/2 = 6.5.
        stats = _twelve_team_stats([100 + i for i in range(12)])
        team_names = [e.team_name for e in stats.entries]
        huge_sds = _all_cat_sds(team_names, 1_000_000)
        roto = score_roto(stats, team_sds=huge_sds)
        for t in team_names:
            assert roto[t][Category.R] == pytest.approx(6.5, abs=0.01)

    def test_monotone_in_own_stat(self):
        # Increasing team i's stat never decreases its EV points.
        dict_stats = _twelve_team_dict([100 + i for i in range(12)])
        team_names = list(dict_stats.keys())
        sds = _all_cat_sds(team_names, 5.0)
        before = score_roto(_stats_table(dict_stats), team_sds=sds)["T5"][Category.R]
        dict_stats["T5"]["R"] = 108  # was 104, now 108
        after = score_roto(_stats_table(dict_stats), team_sds=sds)["T5"][Category.R]
        assert after > before

    def test_total_pts_per_category_invariant(self):
        # Σ pts across teams in a category = N*(N+1)/2 = 78 for N=12.
        stats = _twelve_team_stats([100 + i for i in range(12)])
        team_names = [e.team_name for e in stats.entries]
        sds = _all_cat_sds(team_names, 5.0)
        roto = score_roto(stats, team_sds=sds)
        total_r = sum(roto[t][Category.R] for t in team_names)
        assert total_r == pytest.approx(78.0, abs=1e-6)

    def test_inverse_category_direction(self):
        # ERA: lower is better. Team with lowest ERA gets highest pts.
        dict_stats = _twelve_team_dict([0] * 12)
        for i, t in enumerate(dict_stats):
            dict_stats[t]["ERA"] = 3.0 + i * 0.1
        roto = score_roto(_stats_table(dict_stats))
        assert roto["T1"][Category.ERA] == pytest.approx(12.0)
        assert roto["T12"][Category.ERA] == pytest.approx(1.0)

    def test_small_swap_within_uncertainty_produces_small_delta(self):
        # Two teams tied at 100 R with sd=10 each. Moving 1 R changes
        # pts by only ~0.03, not the full 1.0 of a rank flip.
        dict_stats = _twelve_team_dict([100, 100, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50])
        team_names = list(dict_stats.keys())
        sds = {t: {c: (10.0 if c == Category.R else 1.0) for c in ALL_CATS} for t in team_names}
        before = score_roto(_stats_table(dict_stats), team_sds=sds)["T1"][Category.R]
        dict_stats["T1"]["R"] = 101  # tiny edge
        after = score_roto(_stats_table(dict_stats), team_sds=sds)["T1"][Category.R]
        delta = after - before
        assert 0 < delta < 0.1  # smooth, not a rank flip

    def test_total_includes_all_categories(self):
        stats = _twelve_team_stats([100 + i for i in range(12)])
        team_names = [e.team_name for e in stats.entries]
        roto = score_roto(stats)
        for t in team_names:
            assert roto[t].total == pytest.approx(sum(roto[t].values[c] for c in ALL_CATS))


# ── ProjectedStandings.from_rosters / build_team_sds ────────────────


class TestProjectedStandingsFromRosters:
    """``ProjectedStandings.from_rosters`` wraps ``project_team_stats`` per team."""

    def test_returns_projected_standings(self):
        team_rosters: dict[str, list] = {"Alpha": [], "Beta": []}
        result = ProjectedStandings.from_rosters(team_rosters, effective_date=date(2026, 4, 15))
        assert isinstance(result, ProjectedStandings)
        assert result.effective_date == date(2026, 4, 15)
        assert {e.team_name for e in result.entries} == {"Alpha", "Beta"}

    def test_returns_one_entry_per_team(self):
        rosters = {
            "Team A": [_make_hitter("Player1", r=80, hr=20, rbi=70, sb=10, h=140, ab=500, pa=500)],
            "Team B": [_make_hitter("Player2", r=70, hr=15, rbi=60, sb=8, h=130, ab=490, pa=490)],
        }
        result = ProjectedStandings.from_rosters(rosters, effective_date=date(2026, 4, 15))
        assert len(result.entries) == 2
        team_names = {entry.team_name for entry in result.entries}
        assert team_names == {"Team A", "Team B"}

    def test_each_entry_has_team_name_and_stats(self):
        rosters = {
            "Team A": [_make_hitter("Player1", r=80, hr=20, rbi=70, sb=10, h=140, ab=500, pa=500)],
        }
        result = ProjectedStandings.from_rosters(rosters, effective_date=date(2026, 4, 15))
        entry = result.entries[0]
        assert entry.team_name == "Team A"
        assert isinstance(entry.stats, CategoryStats)

    def test_stats_covers_all_categories(self):
        from fantasy_baseball.utils.constants import ALL_CATEGORIES

        rosters = {
            "Team A": [
                _make_hitter("H1", r=80, hr=20, rbi=70, sb=10, h=140, ab=500, pa=500),
                _make_pitcher("P1", w=10, k=180, sv=0, ip=180, er=70, bb=55, h_allowed=155),
            ],
        }
        result = ProjectedStandings.from_rosters(rosters, effective_date=date(2026, 4, 15))
        stats = result.entries[0].stats
        # CategoryStats exposes every 5x5 roto category by enum.
        for cat in ALL_CATEGORIES:
            assert stats[cat] is not None


class TestBuildTeamSDs:
    """Pure helper wrapping ``project_team_sds`` with scale factor."""

    def test_returns_one_dict_per_team(self):
        rosters = {
            "Team A": [_make_hitter("P1", r=80, hr=20, rbi=70, sb=10, h=140, ab=500, pa=500)],
            "Team B": [_make_hitter("P2", r=70, hr=15, rbi=60, sb=8, h=130, ab=490, pa=490)],
        }
        result = build_team_sds(rosters, sd_scale=1.0)
        assert set(result.keys()) == {"Team A", "Team B"}

    def test_sd_scale_multiplies_each_value(self):
        rosters = {
            "Team A": [_make_hitter("P1", r=80, hr=20, rbi=70, sb=10, h=140, ab=500, pa=500)],
        }
        unscaled = build_team_sds(rosters, sd_scale=1.0)
        scaled = build_team_sds(rosters, sd_scale=0.5)
        for cat, sd in unscaled["Team A"].items():
            assert scaled["Team A"][cat] == pytest.approx(sd * 0.5)

    def test_sd_scale_zero_yields_zero_sds(self):
        rosters = {
            "Team A": [_make_hitter("P1", r=80, hr=20, rbi=70, sb=10, h=140, ab=500, pa=500)],
        }
        result = build_team_sds(rosters, sd_scale=0.0)
        for sd in result["Team A"].values():
            assert sd == 0.0


class TestComputeRosterBreakdown:
    """Per-player breakdowns mirror _apply_displacement classification."""

    def test_active_hitters_all_contribute_fully(self):
        from fantasy_baseball.scoring import (
            ContributionStatus,
            compute_roster_breakdown,
        )

        h1 = _hitter(
            "H1",
            r=60,
            hr=20,
            rbi=70,
            sb=5,
            h=120,
            ab=450,
            positions=[Position.OF],
            selected_position=Position.OF,
        )
        h2 = _hitter(
            "H2",
            r=50,
            hr=15,
            rbi=60,
            sb=3,
            h=110,
            ab=420,
            positions=[Position.FIRST_BASE],
            selected_position=Position.FIRST_BASE,
        )
        breakdown = compute_roster_breakdown("Team A", [h1, h2])

        assert breakdown.team_name == "Team A"
        assert len(breakdown.hitters) == 2
        assert breakdown.pitchers == []
        names = {c.name for c in breakdown.hitters}
        assert names == {"H1", "H2"}
        for c in breakdown.hitters:
            assert c.status == ContributionStatus.ACTIVE
            assert c.scale_factor == 1.0
            assert c.raw_stats["hr"] in (20, 15)

    def test_healthy_bench_tagged_bench_zero_factor(self):
        from fantasy_baseball.scoring import (
            ContributionStatus,
            compute_roster_breakdown,
        )

        active = _hitter(
            "Active",
            r=80,
            hr=20,
            rbi=70,
            sb=10,
            h=140,
            ab=500,
            positions=[Position.OF],
            selected_position=Position.OF,
        )
        bench = _hitter(
            "Bench",
            r=40,
            hr=10,
            rbi=35,
            sb=3,
            h=80,
            ab=300,
            selected_position=Position.BN,
        )
        breakdown = compute_roster_breakdown("Team A", [active, bench])
        by_name = {c.name: c for c in breakdown.hitters}
        assert by_name["Bench"].status == ContributionStatus.BENCH
        assert by_name["Bench"].scale_factor == 0.0
        assert by_name["Active"].status == ContributionStatus.ACTIVE
        assert by_name["Active"].scale_factor == 1.0

    def test_il_slot_tagged_il_full(self):
        from fantasy_baseball.scoring import (
            ContributionStatus,
            compute_roster_breakdown,
        )

        active = _hitter(
            "Active",
            r=80,
            hr=20,
            rbi=70,
            sb=10,
            h=140,
            ab=500,
            positions=[Position.OF],
            selected_position=Position.OF,
        )
        il = _hitter(
            "IL",
            r=30,
            hr=6,
            rbi=20,
            sb=2,
            h=60,
            ab=250,
            positions=[Position.OF],
            selected_position=Position.IL,
        )
        breakdown = compute_roster_breakdown("Team A", [active, il])
        by_name = {c.name: c for c in breakdown.hitters}
        assert by_name["IL"].status == ContributionStatus.IL_FULL
        assert by_name["IL"].scale_factor == 1.0

    def test_bn_plus_il_status_tagged_il_full(self):
        from fantasy_baseball.scoring import (
            ContributionStatus,
            compute_roster_breakdown,
        )

        active = _hitter(
            "Active",
            r=80,
            hr=20,
            rbi=70,
            sb=10,
            h=140,
            ab=500,
            positions=[Position.OF],
            selected_position=Position.OF,
        )
        bn_il = _hitter(
            "BN-IL",
            r=30,
            hr=6,
            rbi=20,
            sb=2,
            h=60,
            ab=250,
            positions=[Position.OF],
            selected_position=Position.BN,
            status="IL60",
        )
        breakdown = compute_roster_breakdown("Team A", [active, bn_il])
        by_name = {c.name: c for c in breakdown.hitters}
        assert by_name["BN-IL"].status == ContributionStatus.IL_FULL
        assert by_name["BN-IL"].scale_factor == 1.0

    def test_displaced_active_tagged_displaced_with_factor(self):
        from fantasy_baseball.scoring import (
            ContributionStatus,
            compute_roster_breakdown,
        )

        # IL player (250 ab) displaces active (500 ab) → factor = 0.5
        active = _hitter(
            "Active",
            r=80,
            hr=20,
            rbi=70,
            sb=10,
            h=140,
            ab=500,
            positions=[Position.OF],
            selected_position=Position.OF,
        )
        il = _hitter(
            "IL",
            r=30,
            hr=6,
            rbi=20,
            sb=2,
            h=60,
            ab=250,
            positions=[Position.OF],
            selected_position=Position.IL,
        )
        breakdown = compute_roster_breakdown("Team A", [active, il])
        by_name = {c.name: c for c in breakdown.hitters}
        assert by_name["Active"].status == ContributionStatus.DISPLACED
        assert by_name["Active"].scale_factor == pytest.approx(0.5)

    def test_missing_ros_tagged_no_projection(self):
        from fantasy_baseball.scoring import (
            ContributionStatus,
            compute_roster_breakdown,
        )

        active_with = _hitter(
            "Active",
            r=80,
            hr=20,
            rbi=70,
            sb=10,
            h=140,
            ab=500,
            positions=[Position.OF],
            selected_position=Position.OF,
        )
        no_ros = Player(
            name="Missing",
            player_type=PlayerType.HITTER,
            positions=[Position.OF],
            selected_position=Position.OF,
            rest_of_season=None,
        )
        breakdown = compute_roster_breakdown("Team A", [active_with, no_ros])
        by_name = {c.name: c for c in breakdown.hitters}
        assert by_name["Missing"].status == ContributionStatus.NO_PROJECTION
        assert by_name["Missing"].scale_factor == 0.0
        assert by_name["Missing"].raw_stats == {}

    def test_raw_stats_prefers_full_season_projection(self):
        """``_raw_stats_for`` reads ``full_season_projection`` when set so the
        breakdown drilldown sums to the same end-of-season totals shown in
        the standings widget. Falls back to ``rest_of_season`` only when
        ``full_season_projection`` is unset (e.g. preseason rosters)."""
        from fantasy_baseball.scoring import compute_roster_breakdown

        both = Player(
            name="Both Fields",
            player_type=PlayerType.HITTER,
            positions=[Position.OF],
            selected_position=Position.OF,
            rest_of_season=HitterStats(r=70, hr=20, rbi=60, sb=5, h=100, ab=400, pa=440),
            full_season_projection=HitterStats(r=100, hr=28, rbi=85, sb=7, h=140, ab=520, pa=580),
        )
        preseason_only = Player(
            name="Preseason Only",
            player_type=PlayerType.HITTER,
            positions=[Position.FIRST_BASE],
            selected_position=Position.FIRST_BASE,
            rest_of_season=HitterStats(r=90, hr=25, rbi=80, sb=8, h=130, ab=480, pa=540),
            full_season_projection=None,
        )
        by_name = {
            c.name: c for c in compute_roster_breakdown("Team A", [both, preseason_only]).hitters
        }
        assert by_name["Both Fields"].raw_stats["r"] == 100  # full_season wins
        assert by_name["Preseason Only"].raw_stats["r"] == 90  # rest_of_season fallback

    def test_counting_stat_sum_invariant(self):
        """Sum of scaled contributions == project_team_stats output."""
        from fantasy_baseball.scoring import (
            compute_roster_breakdown,
            project_team_stats,
        )

        h1 = _hitter(
            "H1",
            r=60,
            hr=20,
            rbi=70,
            sb=5,
            h=120,
            ab=450,
            positions=[Position.OF],
            selected_position=Position.OF,
        )
        h2 = _hitter(
            "H2",
            r=50,
            hr=15,
            rbi=60,
            sb=3,
            h=110,
            ab=420,
            positions=[Position.FIRST_BASE],
            selected_position=Position.FIRST_BASE,
        )
        il = _hitter(
            "IL",
            r=30,
            hr=6,
            rbi=20,
            sb=2,
            h=60,
            ab=250,
            positions=[Position.OF],
            selected_position=Position.IL,
        )
        roster = [h1, h2, il]
        breakdown = compute_roster_breakdown("Team A", roster)
        agg = project_team_stats(roster, displacement=True)

        summed_hr = sum(c.raw_stats.get("hr", 0) * c.scale_factor for c in breakdown.hitters)
        summed_r = sum(c.raw_stats.get("r", 0) * c.scale_factor for c in breakdown.hitters)
        assert summed_hr == pytest.approx(agg.hr)
        assert summed_r == pytest.approx(agg.r)

    def test_rate_stat_component_invariant(self):
        """AVG computed from scaled H and AB matches project_team_stats."""
        from fantasy_baseball.scoring import (
            compute_roster_breakdown,
            project_team_stats,
        )
        from fantasy_baseball.utils.rate_stats import calculate_avg

        h1 = _hitter(
            "H1",
            r=60,
            hr=20,
            rbi=70,
            sb=5,
            h=120,
            ab=450,
            positions=[Position.OF],
            selected_position=Position.OF,
        )
        il = _hitter(
            "IL",
            r=30,
            hr=6,
            rbi=20,
            sb=2,
            h=60,
            ab=250,
            positions=[Position.OF],
            selected_position=Position.IL,
        )
        breakdown = compute_roster_breakdown("Team A", [h1, il])
        agg = project_team_stats([h1, il], displacement=True)

        total_h = sum(c.raw_stats.get("h", 0) * c.scale_factor for c in breakdown.hitters)
        total_ab = sum(c.raw_stats.get("ab", 0) * c.scale_factor for c in breakdown.hitters)
        assert calculate_avg(total_h, total_ab) == pytest.approx(agg.avg)

    def test_pitcher_partitioning(self):
        """Pitchers go into the pitchers list, hitters into the hitters list."""
        from fantasy_baseball.scoring import compute_roster_breakdown

        h = _hitter(
            "H",
            r=60,
            hr=20,
            rbi=70,
            sb=5,
            h=120,
            ab=450,
            positions=[Position.OF],
            selected_position=Position.OF,
        )
        p = _pitcher(
            "P",
            w=12,
            k=180,
            sv=0,
            ip=180,
            er=60,
            bb=45,
            h_allowed=150,
            positions=[Position.SP],
            selected_position=Position.SP,
        )
        breakdown = compute_roster_breakdown("Team A", [h, p])
        assert [c.name for c in breakdown.hitters] == ["H"]
        assert [c.name for c in breakdown.pitchers] == ["P"]

    def test_pitcher_counting_stat_sum_invariant(self):
        """Sum of scaled pitcher contributions == project_team_stats output."""
        from fantasy_baseball.scoring import (
            compute_roster_breakdown,
            project_team_stats,
        )

        p1 = _pitcher(
            "P1",
            w=15,
            k=200,
            sv=0,
            ip=190,
            er=55,
            bb=40,
            h_allowed=150,
            positions=[Position.SP],
            selected_position=Position.SP,
        )
        p2 = _pitcher(
            "P2",
            w=5,
            k=60,
            sv=20,
            ip=60,
            er=20,
            bb=15,
            h_allowed=50,
            positions=[Position.RP],
            selected_position=Position.RP,
        )
        il = _pitcher(
            "IL",
            w=8,
            k=100,
            sv=0,
            ip=130,
            er=40,
            bb=30,
            h_allowed=100,
            positions=[Position.SP],
            selected_position=Position.IL,
        )
        roster = [p1, p2, il]
        breakdown = compute_roster_breakdown("Team A", roster)
        agg = project_team_stats(roster, displacement=True)
        summed_k = sum(c.raw_stats.get("k", 0) * c.scale_factor for c in breakdown.pitchers)
        summed_w = sum(c.raw_stats.get("w", 0) * c.scale_factor for c in breakdown.pitchers)
        assert summed_k == pytest.approx(agg.k)
        assert summed_w == pytest.approx(agg.w)

    def test_era_component_invariant(self):
        """ERA computed from scaled ER and IP matches project_team_stats."""
        from fantasy_baseball.scoring import (
            compute_roster_breakdown,
            project_team_stats,
        )
        from fantasy_baseball.utils.rate_stats import calculate_era

        p = _pitcher(
            "P",
            w=12,
            k=180,
            sv=0,
            ip=180,
            er=60,
            bb=45,
            h_allowed=150,
            positions=[Position.SP],
            selected_position=Position.SP,
        )
        il = _pitcher(
            "IL",
            w=1,
            k=20,
            sv=10,
            ip=30,
            er=10,
            bb=8,
            h_allowed=25,
            positions=[Position.RP],
            selected_position=Position.IL,
        )
        breakdown = compute_roster_breakdown("Team A", [p, il])
        agg = project_team_stats([p, il], displacement=True)
        total_er = sum(c.raw_stats.get("er", 0) * c.scale_factor for c in breakdown.pitchers)
        total_ip = sum(c.raw_stats.get("ip", 0) * c.scale_factor for c in breakdown.pitchers)
        assert calculate_era(total_er, total_ip) == pytest.approx(agg.era)

    def test_whip_component_invariant(self):
        """WHIP computed from scaled BB, H, IP matches project_team_stats."""
        from fantasy_baseball.scoring import (
            compute_roster_breakdown,
            project_team_stats,
        )
        from fantasy_baseball.utils.rate_stats import calculate_whip

        p = _pitcher(
            "P",
            w=12,
            k=180,
            sv=0,
            ip=180,
            er=60,
            bb=45,
            h_allowed=150,
            positions=[Position.SP],
            selected_position=Position.SP,
        )
        il = _pitcher(
            "IL",
            w=1,
            k=20,
            sv=10,
            ip=30,
            er=10,
            bb=8,
            h_allowed=25,
            positions=[Position.RP],
            selected_position=Position.IL,
        )
        breakdown = compute_roster_breakdown("Team A", [p, il])
        agg = project_team_stats([p, il], displacement=True)
        total_bb = sum(c.raw_stats.get("bb", 0) * c.scale_factor for c in breakdown.pitchers)
        total_h = sum(c.raw_stats.get("h_allowed", 0) * c.scale_factor for c in breakdown.pitchers)
        total_ip = sum(c.raw_stats.get("ip", 0) * c.scale_factor for c in breakdown.pitchers)
        assert calculate_whip(total_bb, total_h, total_ip) == pytest.approx(agg.whip)

    def test_no_projection_does_not_perturb_others(self):
        """A missing-ROS player doesn't trigger displacement on other actives."""
        from fantasy_baseball.scoring import (
            ContributionStatus,
            compute_roster_breakdown,
        )

        active_with = _hitter(
            "Active",
            r=80,
            hr=20,
            rbi=70,
            sb=10,
            h=140,
            ab=500,
            positions=[Position.OF],
            selected_position=Position.OF,
        )
        no_ros = Player(
            name="Missing",
            player_type=PlayerType.HITTER,
            positions=[Position.OF],
            selected_position=Position.IL,
            rest_of_season=None,
        )
        breakdown = compute_roster_breakdown("Team A", [active_with, no_ros])
        by_name = {c.name: c for c in breakdown.hitters}
        # Missing is classified IL by slot, but has no ROS → NO_PROJECTION.
        # Critically: it should not have triggered displacement on Active
        # (no playing time to displace with).
        assert by_name["Missing"].status == ContributionStatus.NO_PROJECTION
        assert by_name["Active"].status == ContributionStatus.ACTIVE
        assert by_name["Active"].scale_factor == 1.0

    def test_active_slot_with_il_status_classified_active(self):
        """Regression guard: active slot + IL status → ACTIVE, not IL_FULL.

        This is the scenario the displacement-fix spec exists to handle:
        Yahoo flips a player's status to IL10 while the manager leaves
        them in an active slot; the breakdown must mirror the fix.
        """
        from fantasy_baseball.scoring import (
            ContributionStatus,
            compute_roster_breakdown,
        )

        p = _hitter(
            "Soto-like",
            r=80,
            hr=30,
            rbi=90,
            sb=5,
            h=140,
            ab=500,
            positions=[Position.OF],
            selected_position=Position.OF,
            status="IL10",
        )
        breakdown = compute_roster_breakdown("Team A", [p])
        c = breakdown.hitters[0]
        assert c.status == ContributionStatus.ACTIVE
        assert c.scale_factor == 1.0
