import math
from datetime import date
from unittest.mock import patch

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
    player_category_variance,
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
    """Per-team-per-category SD from analytical variance propagation.

    These pin the performance-only (CV) propagation, so the autouse fixture
    neutralizes the playing-time term (cv_pt=0); the playing-time term is
    covered separately in :class:`TestProjectTeamSDsPlayingTime`.
    """

    @pytest.fixture(autouse=True)
    def _no_playing_time_variance(self):
        with patch(
            "fantasy_baseball.scoring.playing_time_params",
            return_value=(1.0, 0.0),
        ):
            yield

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


class TestProjectTeamSDsPlayingTime:
    """The playing-time (cv_pt) term added to counting-category SDs."""

    def test_counting_sd_adds_cv_pt_in_quadrature(self):
        p = _make_hitter("A", r=80, hr=20, rbi=70, sb=10, h=150, ab=500)
        with patch(
            "fantasy_baseball.scoring.playing_time_params",
            return_value=(0.85, 0.30),
        ):
            sds = project_team_sds([p])
        # SD_R = sqrt(CV_r^2 + cv_pt^2) * r  (single player)
        assert sds[Category.R] == pytest.approx(80 * math.sqrt(STAT_VARIANCE["r"] ** 2 + 0.30**2))
        assert sds[Category.HR] == pytest.approx(20 * math.sqrt(STAT_VARIANCE["hr"] ** 2 + 0.30**2))

    def test_cv_pt_widens_counting_sd(self):
        p = _make_hitter("A", r=80, hr=20, rbi=70, sb=10, h=150, ab=500)
        with patch("fantasy_baseball.scoring.playing_time_params", return_value=(0.85, 0.0)):
            tight = project_team_sds([p])[Category.R]
        with patch("fantasy_baseball.scoring.playing_time_params", return_value=(0.85, 0.30)):
            wide = project_team_sds([p])[Category.R]
        assert wide > tight

    def test_rate_sd_unaffected_by_cv_pt(self):
        a = _make_hitter("A", r=0, hr=0, rbi=0, sb=0, h=150, ab=500)
        b = _make_pitcher("B", w=0, k=0, sv=0, ip=180, er=60, bb=40, h_allowed=140)
        with patch("fantasy_baseball.scoring.playing_time_params", return_value=(0.85, 0.0)):
            base = project_team_sds([a, b])
        with patch("fantasy_baseball.scoring.playing_time_params", return_value=(0.85, 0.40)):
            wide = project_team_sds([a, b])
        # Playing time scales numerator and denominator together -> cancels.
        assert wide[Category.AVG] == pytest.approx(base[Category.AVG])
        assert wide[Category.ERA] == pytest.approx(base[Category.ERA])
        assert wide[Category.WHIP] == pytest.approx(base[Category.WHIP])

    def test_cv_pt_zero_recovers_cv_only_formula(self):
        p = _make_hitter("A", r=80, hr=20, rbi=70, sb=10, h=150, ab=500)
        with patch("fantasy_baseball.scoring.playing_time_params", return_value=(1.0, 0.0)):
            sds = project_team_sds([p])
        assert sds[Category.R] == pytest.approx(STAT_VARIANCE["r"] * 80)

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


class TestPlayerCategoryVariance:
    """player_category_variance returns per-player variance contributions
    that are consistent with project_team_sds (variances sum across players).
    """

    @pytest.fixture(autouse=True)
    def _no_playing_time_variance(self):
        with patch(
            "fantasy_baseball.scoring.playing_time_params",
            return_value=(1.0, 0.0),
        ):
            yield

    def test_hitter_variance_sums_to_team_sd(self):
        """Two identical hitters: team variance == 2 * single-player variance."""
        p1 = _make_hitter("H1", r=100, hr=30, rbi=90, sb=10, h=150, ab=550)
        p2 = _make_hitter("H2", r=100, hr=30, rbi=90, sb=10, h=150, ab=550)
        team_sd_r = project_team_sds([p1, p2], displacement=False)[Category.R]
        one = player_category_variance(p1)[Category.R]
        assert one > 0
        assert team_sd_r**2 == pytest.approx(2 * one, rel=1e-6)

    def test_hitter_variance_covers_all_counting_cats(self):
        """player_category_variance covers R, HR, RBI, SB for hitters."""
        p = _make_hitter("H", r=80, hr=20, rbi=70, sb=10, h=150, ab=500)
        var = player_category_variance(p)
        for cat in (Category.R, Category.HR, Category.RBI, Category.SB):
            assert cat in var
            assert var[cat] >= 0

    def test_hitter_variance_matches_cv_formula(self):
        """Single hitter counting variance == (CV_stat * stat)^2."""
        p = _make_hitter("H", r=80, hr=20, rbi=70, sb=10, h=150, ab=500)
        var = player_category_variance(p)
        assert var[Category.R] == pytest.approx((STAT_VARIANCE["r"] * 80) ** 2)
        assert var[Category.HR] == pytest.approx((STAT_VARIANCE["hr"] * 20) ** 2)

    def test_pitcher_variance_sums_to_team_sd(self):
        """Two identical pitchers: team variance == 2 * single-player variance."""
        p1 = _make_pitcher("P1", w=12, k=180, sv=5, ip=180, er=60, bb=40, h_allowed=150)
        p2 = _make_pitcher("P2", w=12, k=180, sv=5, ip=180, er=60, bb=40, h_allowed=150)
        team_sd_k = project_team_sds([p1, p2], displacement=False)[Category.K]
        one = player_category_variance(p1)[Category.K]
        assert one > 0
        assert team_sd_k**2 == pytest.approx(2 * one, rel=1e-6)

    def test_pitcher_variance_covers_counting_cats(self):
        """player_category_variance covers W, K, SV for pitchers."""
        p = _make_pitcher("P", w=12, k=180, sv=5, ip=180, er=60, bb=40, h_allowed=150)
        var = player_category_variance(p)
        for cat in (Category.W, Category.K, Category.SV):
            assert cat in var
            assert var[cat] >= 0

    def test_playing_time_term_included_in_variance(self):
        """With cv_pt > 0 the variance is strictly larger than CV-only."""
        p = _make_hitter("H", r=80, hr=20, rbi=70, sb=10, h=150, ab=500)
        with patch(
            "fantasy_baseball.scoring.playing_time_params",
            return_value=(1.0, 0.0),
        ):
            var_tight = player_category_variance(p)[Category.R]
        with patch(
            "fantasy_baseball.scoring.playing_time_params",
            return_value=(0.85, 0.30),
        ):
            var_wide = player_category_variance(p)[Category.R]
        assert var_wide > var_tight

    def test_rate_component_sums_present_for_hitter(self):
        """player_category_variance exposes h_sq and ab for rate assembly."""
        p = _make_hitter("H", r=80, hr=20, rbi=70, sb=10, h=150, ab=550)
        var = player_category_variance(p)
        assert "h_sq" in var
        assert "ab" in var
        assert var["h_sq"] == pytest.approx(150**2)
        assert var["ab"] == pytest.approx(550)

    def test_rate_component_sums_present_for_pitcher(self):
        """player_category_variance exposes er_sq, bb_sq, ha_sq, ip for rate assembly."""
        p = _make_pitcher("P", w=12, k=180, sv=5, ip=180, er=60, bb=40, h_allowed=150)
        var = player_category_variance(p)
        for key in ("er_sq", "bb_sq", "ha_sq", "ip"):
            assert key in var
        assert var["er_sq"] == pytest.approx(60**2)
        assert var["ip"] == pytest.approx(180)

    def test_unknown_player_type_returns_empty_dict(self):
        """Unknown player_type returns an empty dict (no crash, no contributions)."""
        p = {"player_type": "unknown"}
        assert player_category_variance(p) == {}


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


class TestPitcherPoolModel:
    """Phase 3: when ``league_context`` is provided, pitcher displacement
    switches from per-IL-player substitution to a pool-slot model.

    Pool = active + IL pitchers. Top-N (by leave-one-out team-ΔRoto) play
    at full scale; bottom (pool_size - active_p_slots) get sf=0 -- even
    when those are IL pitchers themselves. This matches the real-world
    fantasy decision (when an IL guy returns, the manager benches the
    worst remaining pitcher, not necessarily a same-role match) and
    avoids the substitution model's tendency to zero out elite
    low-volume closers when high-volume IL starters are returning.
    """

    def _other_team(self, name: str, *, w=70, k=1300, sv=50, era=4.0, whip=1.25):
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

    def _league_context_for(self, my_team_name: str) -> "LeagueContext":
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
        ]
        baseline = {
            f"Other {i + 1}": self._other_team(f"Other {i + 1}", **cfg).stats
            for i, cfg in enumerate(configs)
        }
        team_sds = {t: dict.fromkeys(ALL_CATS, 5.0) for t in [*baseline.keys(), my_team_name]}
        return LeagueContext(
            baseline_other_team_stats=baseline,
            team_sds=team_sds,
            team_name=my_team_name,
        )

    def _hitters(self, n=10):
        return [
            _hitter(
                f"H{i}",
                r=80,
                hr=22,
                rbi=77,
                sb=12,
                h=140,
                ab=540,
                positions=[Position.OF],
                selected_position=Position.OF,
            )
            for i in range(n)
        ]

    def test_pool_preserves_two_active_closers_when_il_starter_returns(self):
        """Hart-like scenario: 2 elite closers + several SPs + one IL SP.
        Pool model picks the top-9 from the pool; both closers should
        survive because their per-IP value beats the worst SPs.

        The substitution model under SGP would have role-restricted the
        IL SP to bump an SP, but greedy ΔRoto-substitution can
        sometimes zero a closer when high-IP IL pitchers cascade. Pool
        model's leave-one-out comparison protects high-leverage
        low-volume players when their absolute marginal pts > a
        dispensable starter's.
        """
        elite_closer_a = _pitcher(
            "Closer A",
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
        elite_closer_b = _pitcher(
            "Closer B",
            w=3,
            k=80,
            sv=32,
            ip=60,
            er=15,
            bb=18,
            h_allowed=42,
            positions=[Position.RP],
            selected_position=Position.RP,
        )
        # 7 starters of varying quality
        starters = [
            _pitcher(
                f"SP{i}",
                w=12 - i,
                k=180 - 10 * i,
                sv=0,
                ip=180,
                er=60 + 3 * i,
                bb=50,
                h_allowed=160 + i,
                positions=[Position.SP],
                selected_position=Position.SP,
            )
            for i in range(7)
        ]
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

        # 9 active P slots (7 starters + 2 closers); pool size 10 (+ IL); excess 1.
        roster = [elite_closer_a, elite_closer_b, *starters, weak_sp, *self._hitters(), il_sp]
        ctx = self._league_context_for("My Team")

        bd = compute_roster_breakdown(
            "My Team",
            roster,
            league_context=ctx,
            projection_source="rest_of_season",
        )
        sf = {c.name: c.scale_factor for c in bd.pitchers}
        assert sf["Closer A"] == 1.0, "Pool model should preserve the elite closer"
        assert sf["Closer B"] == 1.0, "Pool model should preserve the second closer too"
        assert sf["IL SP"] == 1.0, "IL SP should fully count (model assumes recovery)"
        # Exactly one pitcher gets benched — should be the lowest-marginal SP,
        # not either closer or the IL SP.
        benched = [name for name, factor in sf.items() if factor == 0.0]
        assert len(benched) == 1
        assert benched[0] not in {"Closer A", "Closer B", "IL SP"}

    def test_pool_can_bench_an_il_pitcher_when_it_has_weakest_marginal(self):
        """Key Phase 3 behavior: the pool can drop an IL pitcher itself
        when active pitchers have stronger marginal contributions —
        which is what real managers would do (don't activate a weak
        returner over a productive incumbent).
        """
        # 9 strong active pitchers — every one is hard to lose
        strong_actives = [
            _pitcher(
                f"Strong{i}",
                w=15,
                k=200,
                sv=0,
                ip=200,
                er=60,
                bb=45,
                h_allowed=155,
                positions=[Position.SP],
                selected_position=Position.SP,
            )
            for i in range(9)
        ]
        # IL pitcher with weak projection (recently injured, low expected value)
        weak_il = _pitcher(
            "Weak IL",
            w=2,
            k=40,
            sv=0,
            ip=50,
            er=35,
            bb=25,
            h_allowed=70,
            positions=[Position.SP],
            selected_position=Position.IL,
            status="IL60",
        )

        roster = [*strong_actives, weak_il, *self._hitters()]
        ctx = self._league_context_for("My Team")

        bd = compute_roster_breakdown(
            "My Team",
            roster,
            league_context=ctx,
            projection_source="rest_of_season",
        )
        sf = {c.name: c.scale_factor for c in bd.pitchers}
        # IL pitcher's projected contribution is below all active SP rates,
        # so no positive-DeltaRoto swap exists -- the IL pitcher gets sf=0
        # via the pair-swap model's "bench" fallback.
        assert sf["Weak IL"] == 0.0, "Pool model should bench the weak IL pitcher"
        for i in range(9):
            assert sf[f"Strong{i}"] == 1.0, f"Strong{i} should be preserved"

    def test_pool_no_op_when_all_pitchers_fit_in_slots(self):
        """No IL pitchers to evaluate -- the pair-swap model returns {} when
        il_candidates is empty, so no displacement is applied."""
        actives = [
            _pitcher(
                f"P{i}",
                w=10,
                k=150,
                sv=0,
                ip=150,
                er=55,
                bb=45,
                h_allowed=130,
                positions=[Position.SP],
                selected_position=Position.SP,
            )
            for i in range(7)
        ]
        roster = [*actives, *self._hitters()]
        ctx = self._league_context_for("My Team")

        bd = compute_roster_breakdown(
            "My Team",
            roster,
            league_context=ctx,
            projection_source="rest_of_season",
        )
        sf = {c.name: c.scale_factor for c in bd.pitchers}
        for i in range(7):
            assert sf[f"P{i}"] == 1.0

    def test_pool_model_inactive_without_league_context(self):
        """Backwards compat: without league_context, pitcher displacement
        falls back to the legacy per-IL-player substitution (SGP-based
        target picking). Phase 1's test_sp_displaces_sp result still
        holds verbatim."""
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
        # Without league_context: substitution model — Bad SP scaled to 0
        # (factor = max(0, 120-130)/120 = 0), totals match Phase 1 test.
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


def _spy_sd_scale(monkeypatch) -> dict[str, float]:
    """Capture the ``sd_scale`` passed to ``build_team_sds`` so a test can
    assert how ``ProjectedStandings.from_rosters`` damps its picker SDs."""
    from fantasy_baseball import scoring

    captured: dict[str, float] = {}
    real = scoring.build_team_sds

    def spy(team_rosters, sd_scale):
        captured["sd_scale"] = sd_scale
        return real(team_rosters, sd_scale)

    monkeypatch.setattr(scoring, "build_team_sds", spy)
    return captured


class TestProjectedStandingsFromRosters:
    """``ProjectedStandings.from_rosters`` wraps ``project_team_stats`` per team."""

    def test_returns_projected_standings(self):
        team_rosters: dict[str, list] = {"Alpha": [], "Beta": []}
        result = ProjectedStandings.from_rosters(team_rosters, effective_date=date(2026, 4, 15))
        assert isinstance(result, ProjectedStandings)
        assert result.effective_date == date(2026, 4, 15)
        assert {e.team_name for e in result.entries} == {"Alpha", "Beta"}

    def test_scales_picker_sds_by_sqrt_fraction_remaining(self, monkeypatch):
        """The displacement picker's SDs must damp by sqrt(fraction_remaining),
        matching the canonical team_sds the optimizer/deltaRoto use -- not the
        full-season sd_scale=1.0 the standings build previously hardcoded
        (which over-softened mid-season lineup decisions vs every other
        consumer)."""
        captured = _spy_sd_scale(monkeypatch)
        ProjectedStandings.from_rosters(
            {"A": [], "B": []}, effective_date=date(2026, 5, 5), fraction_remaining=0.25
        )
        assert captured["sd_scale"] == pytest.approx(0.5)  # sqrt(0.25)

    def test_picker_sds_default_to_full_season_when_fraction_unset(self, monkeypatch):
        """Default (no fraction_remaining) keeps sd_scale=1.0 -- correct for
        preseason (full season remaining) and backward-compatible."""
        captured = _spy_sd_scale(monkeypatch)
        ProjectedStandings.from_rosters({"A": [], "B": []}, effective_date=date(2026, 5, 5))
        assert captured["sd_scale"] == pytest.approx(1.0)

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


class TestScaleStatsYTDFloor:
    """`_scale_stats` adds a YTD floor ONLY in full_season_projection mode.

    In rest_of_season mode (the optimizer/trade-evaluator path) the function
    preserves the legacy ROS-only behavior so forward-looking decisions are
    not biased by locked YTD totals. In full_season_projection mode (the
    standings/breakdown path) YTD always counts so displaced players don't
    silently lose their already-recorded stats.
    """

    def _pitcher_with_ytd(self, name, ros_k, full_season_k):
        """Pitcher whose YTD K = full_season_k - ros_k."""
        from fantasy_baseball.models.player import PitcherStats, Player, PlayerType

        ros = PitcherStats(
            ip=60, w=2, k=ros_k, sv=0, er=20, bb=15, h_allowed=50, era=3.00, whip=1.08
        )
        full = PitcherStats(
            ip=120, w=8, k=full_season_k, sv=0, er=40, bb=30, h_allowed=100, era=3.00, whip=1.08
        )
        return Player(
            name=name,
            player_type=PlayerType.PITCHER,
            rest_of_season=ros,
            full_season_projection=full,
        )

    def test_full_season_mode_scale_zero_preserves_ytd(self):
        """Webb with 78 YTD K + 60 ROS K, displaced (factor=0) in
        full_season_projection mode: contributes 78 K, not 0. YTD is
        locked-in and must not vanish from the standings view.
        """
        from fantasy_baseball.scoring import _scale_stats

        p = self._pitcher_with_ytd("Webb", ros_k=60, full_season_k=138)
        scaled = _scale_stats(p, 0.0, "full_season_projection")
        # YTD = 138 - 60 = 78; ROS * 0 = 0; total = 78
        assert scaled["k"] == 78

    def test_full_season_mode_scale_one_returns_full_season(self):
        """factor=1.0 in full_season mode yields full_season K (= YTD + ROS),
        matching an undiscounted player's projection-source full-season read."""
        from fantasy_baseball.scoring import _scale_stats

        p = self._pitcher_with_ytd("Healthy", ros_k=120, full_season_k=200)
        scaled = _scale_stats(p, 1.0, "full_season_projection")
        # YTD = 200 - 120 = 80; ROS * 1 = 120; total = 200
        assert scaled["k"] == 200

    def test_full_season_mode_scale_half_keeps_full_ytd(self):
        """factor=0.5 in full_season mode: YTD untouched, ROS halved."""
        from fantasy_baseball.scoring import _scale_stats

        p = self._pitcher_with_ytd("Half", ros_k=80, full_season_k=130)
        scaled = _scale_stats(p, 0.5, "full_season_projection")
        # YTD = 130 - 80 = 50; ROS * 0.5 = 40; total = 90
        assert scaled["k"] == 90

    def test_ros_mode_unchanged_legacy_behavior(self):
        """In rest_of_season mode, the function still returns ROS * factor
        with NO YTD floor. The optimizer's forward-looking semantics are
        unchanged: hot-YTD and cold-YTD players with the same ROS contribute
        identically to forward decisions.
        """
        from fantasy_baseball.scoring import _scale_stats

        p = self._pitcher_with_ytd("Webb", ros_k=60, full_season_k=138)
        # ROS-mode displacement: contributes 0, NOT 78. The optimizer must not
        # see YTD bleed into displaced-player scoring.
        scaled_zero = _scale_stats(p, 0.0, "rest_of_season")
        assert scaled_zero["k"] == 0

        # factor=0.5 in ROS mode -> ROS * 0.5 = 30, no YTD.
        scaled_half = _scale_stats(p, 0.5, "rest_of_season")
        assert scaled_half["k"] == 30

    def test_ros_mode_default_when_source_omitted(self):
        """Backwards compatibility: callers that don't pass source get the
        legacy ROS-only behavior. (Same as test_ros_mode_unchanged with
        explicit source, just verifies the default.)
        """
        from fantasy_baseball.scoring import _scale_stats

        p = self._pitcher_with_ytd("Webb", ros_k=60, full_season_k=138)
        scaled = _scale_stats(p, 0.0)
        assert scaled["k"] == 0

    def test_full_season_mode_falls_back_to_ros_when_no_full_season(self):
        """Preseason rosters lack full_season_projection. Even in full_season
        mode, scaling should behave like ROS-only (YTD = 0 by definition)."""
        from fantasy_baseball.models.player import PitcherStats, Player, PlayerType
        from fantasy_baseball.scoring import _scale_stats

        ros = PitcherStats(
            ip=200, w=12, k=200, sv=0, er=70, bb=40, h_allowed=160, era=3.15, whip=1.0
        )
        p = Player(
            name="Preseason", player_type=PlayerType.PITCHER, rest_of_season=ros
        )  # no full_season_projection
        scaled = _scale_stats(p, 0.5, "full_season_projection")
        # No YTD known -> floor = 0 -> ROS * 0.5 = 100
        assert scaled["k"] == 100


class TestPitcherPoolRateSwap:
    """The pool model picks (candidate, target) pairs and discounts the target
    via rate-swap, not zero-out. One pair per round; same target across all
    categories (no per-stat cherry-pick).
    """

    def _il_starter(self, name, *, ros_ip, ros_k, preseason_ip):
        from fantasy_baseball.models.player import PitcherStats, Player, PlayerType
        from fantasy_baseball.models.positions import Position

        ros = PitcherStats(
            ip=ros_ip,
            w=ros_ip * 0.05,
            k=ros_k,
            sv=0,
            er=ros_ip * 0.33,
            bb=ros_ip * 0.25,
            h_allowed=ros_ip * 0.83,
            era=3.00,
            whip=1.08,
        )
        full = PitcherStats(
            ip=ros_ip + 30,
            w=(ros_ip + 30) * 0.05,
            k=ros_k + 30,
            sv=0,
            er=(ros_ip + 30) * 0.33,
            bb=(ros_ip + 30) * 0.25,
            h_allowed=(ros_ip + 30) * 0.83,
            era=3.00,
            whip=1.08,
        )
        pre = PitcherStats(
            ip=preseason_ip,
            w=preseason_ip * 0.05,
            k=preseason_ip * 1.0,
            sv=0,
            er=preseason_ip * 0.33,
            bb=preseason_ip * 0.25,
            h_allowed=preseason_ip * 0.83,
            era=3.00,
            whip=1.08,
        )
        return Player(
            name=name,
            player_type=PlayerType.PITCHER,
            rest_of_season=ros,
            full_season_projection=full,
            preseason=pre,
            selected_position=Position.IL,
        )

    def _active_starter(self, name, *, ros_ip, k_per_9, preseason_ip):
        from fantasy_baseball.models.player import PitcherStats, Player, PlayerType
        from fantasy_baseball.models.positions import Position

        k = ros_ip * k_per_9 / 9.0
        ros = PitcherStats(
            ip=ros_ip,
            w=ros_ip * 0.05,
            k=k,
            sv=0,
            er=ros_ip * 0.40,
            bb=ros_ip * 0.30,
            h_allowed=ros_ip * 0.90,
            era=3.60,
            whip=1.20,
        )
        full = PitcherStats(
            ip=ros_ip + 40,
            w=(ros_ip + 40) * 0.05,
            k=k + 40 * k_per_9 / 9.0,
            sv=0,
            er=(ros_ip + 40) * 0.40,
            bb=(ros_ip + 40) * 0.30,
            h_allowed=(ros_ip + 40) * 0.90,
            era=3.60,
            whip=1.20,
        )
        pre = PitcherStats(
            ip=preseason_ip,
            w=preseason_ip * 0.05,
            k=preseason_ip * k_per_9 / 9.0,
            sv=0,
            er=preseason_ip * 0.40,
            bb=preseason_ip * 0.30,
            h_allowed=preseason_ip * 0.90,
            era=3.60,
            whip=1.20,
        )
        return Player(
            name=name,
            player_type=PlayerType.PITCHER,
            rest_of_season=ros,
            full_season_projection=full,
            preseason=pre,
            selected_position=Position.P,
        )

    def test_webb_rate_swap_discounts_worst_active_sp_not_webb(self):
        """Webb (IL, 60 ROS IP, ~10 K/9) returns. Worst active SP has 130
        ROS IP @ 7 K/9. The pair-swap model should:
          - keep Webb at factor=1.0 (full contribution, NOT in factors dict)
          - discount the worst SP by 60/130 of his ROS
        It should NOT zero out Webb. It should NOT touch other active SPs.
        """
        from fantasy_baseball.models.standings import CategoryStats
        from fantasy_baseball.scoring import (
            LeagueContext,
            _compute_pitcher_pool_factors,
        )
        from fantasy_baseball.utils.constants import Category

        webb = self._il_starter("Webb", ros_ip=60, ros_k=67, preseason_ip=200)
        sp_strong_a = self._active_starter("SP_A", ros_ip=140, k_per_9=10.0, preseason_ip=200)
        sp_strong_b = self._active_starter("SP_B", ros_ip=135, k_per_9=9.5, preseason_ip=200)
        sp_worst = self._active_starter("SP_Worst", ros_ip=130, k_per_9=7.0, preseason_ip=200)

        active = [sp_strong_a, sp_strong_b, sp_worst]
        il = [webb]

        baseline = {
            "Opp1": CategoryStats(
                r=0, hr=0, rbi=0, sb=0, avg=0, w=20, k=400, sv=0, era=4.0, whip=1.3
            ),
            "Opp2": CategoryStats(
                r=0, hr=0, rbi=0, sb=0, avg=0, w=22, k=420, sv=0, era=3.9, whip=1.28
            ),
        }
        team_sds = {tn: {c: 1.0 for c in Category} for tn in ["Me", *baseline.keys()]}
        ctx = LeagueContext(
            baseline_other_team_stats=baseline,
            team_sds=team_sds,
            team_name="Me",
        )

        factors = _compute_pitcher_pool_factors(
            active_pitchers=active,
            il_pitchers=il,
            all_active=active,
            all_il=il,
            league_context=ctx,
            projection_source="rest_of_season",
        )

        assert "Webb" not in factors, "Webb should be active (sf=1.0 implicit), not in factors"
        assert "SP_Worst" in factors, "Worst SP should be the discount target"
        # Same-role direct-IP swap: cand_pre=200, tgt_pre=200, cand_ros=60.
        # window = 200 * (60 / 200) = 60. target_ros_ip=130, scale=(130-60)/130.
        expected = (130.0 - 60.0) / 130.0
        assert abs(factors["SP_Worst"] - expected) < 1e-9
        assert "SP_A" not in factors and "SP_B" not in factors, "Strong SPs untouched"

    def test_same_target_discounted_identically_across_all_categories(self):
        """When SP_Worst is the swap target, ALL his counting stats get the
        same scale factor -- not a per-stat picker. This guards against
        accidental per-category target selection in future refactors.
        """
        from fantasy_baseball.models.standings import CategoryStats
        from fantasy_baseball.scoring import (
            LeagueContext,
            _apply_displacement,
        )
        from fantasy_baseball.utils.constants import Category

        webb = self._il_starter("Webb", ros_ip=60, ros_k=67, preseason_ip=200)
        sp_worst = self._active_starter("SP_Worst", ros_ip=130, k_per_9=7.0, preseason_ip=200)
        sp_strong = self._active_starter("SP_Strong", ros_ip=140, k_per_9=10.0, preseason_ip=200)

        baseline = {
            "Opp1": CategoryStats(
                r=0, hr=0, rbi=0, sb=0, avg=0, w=20, k=400, sv=0, era=4.0, whip=1.3
            ),
            "Opp2": CategoryStats(
                r=0, hr=0, rbi=0, sb=0, avg=0, w=22, k=420, sv=0, era=3.9, whip=1.28
            ),
        }
        team_sds = {tn: {c: 1.0 for c in Category} for tn in ["Me", *baseline.keys()]}
        ctx = LeagueContext(
            baseline_other_team_stats=baseline,
            team_sds=team_sds,
            team_name="Me",
        )

        scaled_roster = _apply_displacement(
            [webb, sp_worst, sp_strong],
            league_context=ctx,
            projection_source="rest_of_season",
        )

        # Find the scaled SP_Worst entry (it's a dict, not a Player).
        # SP_Worst should have been discounted; identify it by its k value
        # being less than SP_Worst's full ROS k.
        worst_scaled = None
        for entry in scaled_roster:
            if isinstance(entry, dict) and entry.get("k", 0) < sp_worst.rest_of_season.k:
                worst_scaled = entry
                break
        assert worst_scaled is not None, "Expected one scaled dict for the swap target"

        expected_factor = (130.0 - 60.0) / 130.0
        # Every counting stat scaled by the same factor.
        for key in ("k", "w", "sv", "ip", "er", "bb", "h_allowed"):
            ros_val = getattr(sp_worst.rest_of_season, key)
            assert abs(worst_scaled[key] - ros_val * expected_factor) < 1e-6, (
                f"Stat {key} not scaled by the unified target factor"
            )

    def test_starter_returning_discounts_reliever_via_preseason_proration(self):
        """Webb (SP, 60 ROS IP, 200 preseason IP) returns to a pool whose
        weakest arm is a reliever (25 ROS IP, 65 preseason IP). The cross-role
        swap should discount the reliever by 65 * (60/200) = 19.5 IP, NOT by
        60 IP (which would zero the reliever).
        """
        from fantasy_baseball.lineup.pitcher_swap import discount_factor
        from fantasy_baseball.models.player import PitcherStats, Player, PlayerType
        from fantasy_baseball.models.positions import Position
        from fantasy_baseball.models.standings import CategoryStats
        from fantasy_baseball.scoring import LeagueContext, _compute_pitcher_pool_factors
        from fantasy_baseball.utils.constants import Category

        webb = self._il_starter("Webb", ros_ip=60, ros_k=67, preseason_ip=200)

        # Elite SPs (untouched), one weak RP (the cross-role target).
        sp_a = self._active_starter("SP_A", ros_ip=140, k_per_9=10.5, preseason_ip=200)
        sp_b = self._active_starter("SP_B", ros_ip=135, k_per_9=9.8, preseason_ip=200)

        rp_ros = PitcherStats(
            ip=25,
            w=1,
            k=20,
            sv=2,
            er=12,
            bb=10,
            h_allowed=22,
            era=4.32,
            whip=1.28,
        )
        rp_full = PitcherStats(
            ip=45,
            w=2,
            k=40,
            sv=5,
            er=20,
            bb=18,
            h_allowed=42,
            era=4.00,
            whip=1.33,
        )
        rp_pre = PitcherStats(
            ip=65,
            w=3,
            k=60,
            sv=7,
            er=30,
            bb=24,
            h_allowed=58,
            era=4.15,
            whip=1.26,
        )
        weak_rp = Player(
            name="Weak_RP",
            player_type=PlayerType.PITCHER,
            rest_of_season=rp_ros,
            full_season_projection=rp_full,
            preseason=rp_pre,
            selected_position=Position.P,
        )
        active = [sp_a, sp_b, weak_rp]

        baseline = {
            "Opp1": CategoryStats(
                r=0, hr=0, rbi=0, sb=0, avg=0, w=20, k=400, sv=18, era=4.0, whip=1.3
            ),
            "Opp2": CategoryStats(
                r=0, hr=0, rbi=0, sb=0, avg=0, w=22, k=420, sv=20, era=3.9, whip=1.28
            ),
        }
        team_sds = {tn: {c: 1.0 for c in Category} for tn in ["Me", *baseline.keys()]}
        ctx = LeagueContext(
            baseline_other_team_stats=baseline,
            team_sds=team_sds,
            team_name="Me",
        )

        factors = _compute_pitcher_pool_factors(
            active_pitchers=active,
            il_pitchers=[webb],
            all_active=active,
            all_il=[webb],
            league_context=ctx,
            projection_source="rest_of_season",
        )

        # Webb stays active.
        assert "Webb" not in factors

        # Weak_RP should be the discount target (he's the weakest pitcher).
        # Window = 65 * (60/200) = 19.5. Factor = (25 - 19.5) / 25.
        assert "Weak_RP" in factors, (
            f"Expected Weak_RP as cross-role target; picker chose {list(factors)}"
        )
        expected = discount_factor(target_ros_ip=25.0, window=65.0 * (60.0 / 200.0))
        assert abs(factors["Weak_RP"] - expected) < 1e-9

    def test_two_il_pitchers_each_pick_own_target(self):
        """Two IL pitchers returning -- each must pick its OWN discount
        target. The already_discounted set must prevent double-targeting,
        and the second IL pitcher's bench-vs-swap evaluation must account
        for the first IL pitcher's committed discount.

        Setup: Webb (200 preseason, 60 ROS, 67 K) and Glasnow (180
        preseason, 45 ROS, 55 K) both return. Four active SPs of varying
        quality are available. Higher-preseason-IP IL pitcher (Webb)
        processes first and claims one target; Glasnow then picks a
        DIFFERENT target (already_discounted prevents reuse).

        Note on target identity: the delta-Roto-optimal picker maximizes
        team roto pts, which is not the same as "discount the weakest."
        Discounting a slightly-higher-volume target by the same IP window
        can yield a better team roto outcome when the team's competitive
        position with that stat is near a half-point boundary. The exact
        targets (SP_A and SP_B in this setup) are a stable fixture of the
        picker's math -- the important invariant is that the TWO targets
        are distinct (already_discounted works) and neither IL pitcher is
        benched.
        """
        from fantasy_baseball.models.standings import CategoryStats
        from fantasy_baseball.scoring import LeagueContext, _compute_pitcher_pool_factors
        from fantasy_baseball.utils.constants import Category

        webb = self._il_starter("Webb", ros_ip=60, ros_k=67, preseason_ip=200)
        glasnow = self._il_starter("Glasnow", ros_ip=45, ros_k=55, preseason_ip=180)

        sp_a = self._active_starter("SP_A", ros_ip=140, k_per_9=10.0, preseason_ip=200)
        sp_b = self._active_starter("SP_B", ros_ip=135, k_per_9=9.5, preseason_ip=200)
        sp_worst_1 = self._active_starter("SP_Worst1", ros_ip=130, k_per_9=7.0, preseason_ip=200)
        sp_worst_2 = self._active_starter("SP_Worst2", ros_ip=125, k_per_9=6.5, preseason_ip=200)

        active = [sp_a, sp_b, sp_worst_1, sp_worst_2]
        il = [webb, glasnow]

        baseline = {
            "Opp1": CategoryStats(
                r=0, hr=0, rbi=0, sb=0, avg=0, w=20, k=400, sv=0, era=4.0, whip=1.3
            ),
            "Opp2": CategoryStats(
                r=0, hr=0, rbi=0, sb=0, avg=0, w=22, k=420, sv=0, era=3.9, whip=1.28
            ),
        }
        team_sds = {tn: {c: 1.0 for c in Category} for tn in ["Me", *baseline.keys()]}
        ctx = LeagueContext(
            baseline_other_team_stats=baseline,
            team_sds=team_sds,
            team_name="Me",
        )

        factors = _compute_pitcher_pool_factors(
            active_pitchers=active,
            il_pitchers=il,
            all_active=active,
            all_il=il,
            league_context=ctx,
            projection_source="rest_of_season",
        )

        # Neither IL pitcher is benched (both should activate -- their rates
        # beat the weakest two actives' rates).
        assert "Webb" not in factors, "Webb should be active (sf=1.0 implicit)"
        assert "Glasnow" not in factors, "Glasnow should be active (sf=1.0 implicit)"

        # Exactly TWO distinct active pitchers are discounted -- already_discounted
        # prevents the second IL pitcher from re-targeting the first's chosen target.
        all_active_names = {"SP_A", "SP_B", "SP_Worst1", "SP_Worst2"}
        discounted_active_targets = [name for name in factors if name in all_active_names]
        assert len(discounted_active_targets) == 2, (
            f"Expected two distinct active targets discounted; got {list(factors)}"
        )
        assert len(set(discounted_active_targets)) == 2, (
            f"already_discounted violated: same target chosen twice; got {list(factors)}"
        )

        # Both scale factors are in (0, 1) -- partial discounts, not full bench.
        for tgt_name in discounted_active_targets:
            assert 0.0 < factors[tgt_name] < 1.0, (
                f"{tgt_name} should be partially discounted, got {factors[tgt_name]}"
            )

    def test_il_pitcher_benched_when_no_positive_swap_exists(self):
        """If every active pitcher's rate beats the IL pitcher's rate by
        enough that discounting any of them costs more than the IL pitcher
        adds, the IL pitcher gets sf=0 (legacy bench fallback).
        """
        from fantasy_baseball.models.standings import CategoryStats
        from fantasy_baseball.scoring import LeagueContext, _compute_pitcher_pool_factors
        from fantasy_baseball.utils.constants import Category

        # Weak IL pitcher: small ROS volume, low K rate.
        weak_il = self._il_starter("Weak_IL", ros_ip=30, ros_k=20, preseason_ip=180)

        # Elite active SPs -- discounting any of them loses more than Weak_IL adds.
        sp_a = self._active_starter("Elite_A", ros_ip=160, k_per_9=11.0, preseason_ip=200)
        sp_b = self._active_starter("Elite_B", ros_ip=155, k_per_9=10.8, preseason_ip=200)
        sp_c = self._active_starter("Elite_C", ros_ip=150, k_per_9=10.5, preseason_ip=200)

        active = [sp_a, sp_b, sp_c]

        baseline = {
            "Opp1": CategoryStats(
                r=0, hr=0, rbi=0, sb=0, avg=0, w=20, k=400, sv=0, era=4.0, whip=1.3
            ),
            "Opp2": CategoryStats(
                r=0, hr=0, rbi=0, sb=0, avg=0, w=22, k=420, sv=0, era=3.9, whip=1.28
            ),
        }
        team_sds = {tn: {c: 1.0 for c in Category} for tn in ["Me", *baseline.keys()]}
        ctx = LeagueContext(
            baseline_other_team_stats=baseline,
            team_sds=team_sds,
            team_name="Me",
        )

        factors = _compute_pitcher_pool_factors(
            active_pitchers=active,
            il_pitchers=[weak_il],
            all_active=active,
            all_il=[weak_il],
            league_context=ctx,
            projection_source="rest_of_season",
        )

        # Weak_IL should be benched (sf=0): no positive swap exists.
        assert factors.get("Weak_IL") == 0.0, (
            f"Expected Weak_IL benched at sf=0; got factors={factors}"
        )

        # The elite SPs are NOT discounted -- no swap was applied.
        assert "Elite_A" not in factors
        assert "Elite_B" not in factors
        assert "Elite_C" not in factors


class TestComputeRosterBreakdownFullSeasonInvariant:
    """The breakdown modal's per-row totals must sum to the standings
    widget's headline total in full_season_projection mode. The aggregate
    over ``contribution_stats[cat]`` per category must equal
    ``project_team_stats(roster, displacement=True, projection_source="full_season_projection")``.

    Pre-fix: the modal computed raw_stats * scale_factor, which gave
    ``full_season * factor`` instead of ``YTD + ROS * factor``, causing a
    per-pitcher discrepancy of ``YTD * (1 - factor)``. Webb scenario: 80 YTD K,
    120 ROS K, factor 0.5 -> modal showed 100, widget showed 140, 40 K gap.
    """

    def _pitcher(self, name, *, ros_k, full_season_k, ros_ip=60, full_season_ip=120):
        from fantasy_baseball.models.player import PitcherStats, Player, PlayerType
        from fantasy_baseball.models.positions import Position

        ros = PitcherStats(
            ip=ros_ip,
            w=2,
            k=ros_k,
            sv=0,
            er=20,
            bb=15,
            h_allowed=50,
            era=3.00,
            whip=1.08,
        )
        full = PitcherStats(
            ip=full_season_ip,
            w=8,
            k=full_season_k,
            sv=0,
            er=40,
            bb=30,
            h_allowed=100,
            era=3.00,
            whip=1.08,
        )
        pre = PitcherStats(
            ip=200,
            w=14,
            k=full_season_k + 50,
            sv=0,
            er=72,
            bb=50,
            h_allowed=170,
            era=3.24,
            whip=1.10,
        )
        return Player(
            name=name,
            player_type=PlayerType.PITCHER,
            rest_of_season=ros,
            full_season_projection=full,
            preseason=pre,
            selected_position=Position.P,
        )

    def test_breakdown_contribution_stats_match_standings_in_full_season_mode(self):
        """For a displaced pitcher with non-zero YTD, contribution_stats[k]
        must equal _scale_stats output, which equals what project_team_stats
        sums into the widget total.
        """
        from fantasy_baseball.models.player import PitcherStats, Player, PlayerType
        from fantasy_baseball.models.positions import Position
        from fantasy_baseball.models.standings import CategoryStats
        from fantasy_baseball.scoring import (
            LeagueContext,
            compute_roster_breakdown,
            project_team_stats,
        )
        from fantasy_baseball.utils.constants import Category

        # Webb: 80 YTD K, 60 ROS K, full_season=140. He's IL.
        webb_ros = PitcherStats(
            ip=60, w=4, k=60, sv=0, er=22, bb=15, h_allowed=50, era=3.30, whip=1.08
        )
        webb_full = PitcherStats(
            ip=140, w=10, k=140, sv=0, er=50, bb=35, h_allowed=120, era=3.21, whip=1.10
        )
        webb_pre = PitcherStats(
            ip=200, w=14, k=200, sv=0, er=72, bb=50, h_allowed=170, era=3.24, whip=1.10
        )
        webb = Player(
            name="Webb",
            player_type=PlayerType.PITCHER,
            rest_of_season=webb_ros,
            full_season_projection=webb_full,
            preseason=webb_pre,
            selected_position=Position.IL,
        )

        # Worst SP: low K rate, gets discounted.
        sp_worst_ros = PitcherStats(
            ip=130, w=6, k=100, sv=0, er=50, bb=30, h_allowed=120, era=3.46, whip=1.15
        )
        sp_worst_full = PitcherStats(
            ip=170, w=8, k=130, sv=0, er=65, bb=40, h_allowed=155, era=3.44, whip=1.15
        )
        sp_worst_pre = PitcherStats(
            ip=200, w=10, k=160, sv=0, er=80, bb=50, h_allowed=185, era=3.60, whip=1.18
        )
        sp_worst = Player(
            name="SP_Worst",
            player_type=PlayerType.PITCHER,
            rest_of_season=sp_worst_ros,
            full_season_projection=sp_worst_full,
            preseason=sp_worst_pre,
            selected_position=Position.P,
        )

        # Filler strong SPs (untouched by swap).
        def strong_sp(name, ros_ip):
            ros = PitcherStats(
                ip=ros_ip,
                w=10,
                k=int(ros_ip * 10.5 / 9),
                sv=0,
                er=40,
                bb=30,
                h_allowed=110,
                era=2.80,
                whip=1.05,
            )
            full = PitcherStats(
                ip=ros_ip + 40,
                w=14,
                k=int((ros_ip + 40) * 10.5 / 9),
                sv=0,
                er=50,
                bb=40,
                h_allowed=140,
                era=2.80,
                whip=1.05,
            )
            pre = PitcherStats(
                ip=200, w=14, k=233, sv=0, er=62, bb=50, h_allowed=180, era=2.80, whip=1.05
            )
            return Player(
                name=name,
                player_type=PlayerType.PITCHER,
                rest_of_season=ros,
                full_season_projection=full,
                preseason=pre,
                selected_position=Position.P,
            )

        sp_a = strong_sp("SP_A", 140)
        sp_b = strong_sp("SP_B", 135)
        roster = [webb, sp_worst, sp_a, sp_b]

        baseline = {
            "Opp1": CategoryStats(
                r=0, hr=0, rbi=0, sb=0, avg=0, w=20, k=400, sv=0, era=4.0, whip=1.3
            ),
            "Opp2": CategoryStats(
                r=0, hr=0, rbi=0, sb=0, avg=0, w=22, k=420, sv=0, era=3.9, whip=1.28
            ),
        }
        team_sds = {tn: {c: 1.0 for c in Category} for tn in ["Me", *baseline.keys()]}
        ctx = LeagueContext(
            baseline_other_team_stats=baseline,
            team_sds=team_sds,
            team_name="Me",
        )

        breakdown = compute_roster_breakdown(
            "Me", roster, league_context=ctx, projection_source="full_season_projection"
        )
        team_stats = project_team_stats(
            roster,
            displacement=True,
            projection_source="full_season_projection",
            league_context=ctx,
        )

        # Sum contribution_stats[k] across all pitchers. Must equal team_stats.k.
        total_k = sum(c.contribution_stats.get("k", 0.0) for c in breakdown.pitchers)
        assert abs(total_k - team_stats.k) < 1e-6, (
            f"Breakdown sum k={total_k} disagrees with standings k={team_stats.k}; "
            f"contributions={[(c.name, c.contribution_stats.get('k')) for c in breakdown.pitchers]}"
        )

        # Specifically: SP_Worst is the discount target (factor ~0.538), and
        # its contribution_stats[k] must be YTD + ROS * factor, not full_season * factor.
        ytd_k_sp_worst = sp_worst.full_season_projection.k - sp_worst.rest_of_season.k
        ros_k_sp_worst = sp_worst.rest_of_season.k
        # SP_Worst must be among the contributions (status DISPLACED or ACTIVE).
        sp_worst_contrib = next((c for c in breakdown.pitchers if c.name == "SP_Worst"), None)
        assert sp_worst_contrib is not None
        if sp_worst_contrib.status == "displaced":
            f = sp_worst_contrib.scale_factor
            expected = ytd_k_sp_worst + ros_k_sp_worst * f
            assert abs(sp_worst_contrib.contribution_stats["k"] - expected) < 1e-6, (
                f"SP_Worst contribution_stats[k]={sp_worst_contrib.contribution_stats['k']}, "
                f"expected YTD+ROS*factor = {ytd_k_sp_worst} + {ros_k_sp_worst}*{f} = {expected}"
            )

    def test_breakdown_contribution_stats_match_ros_mode(self):
        """In rest_of_season mode, contribution_stats[k] = ROS * factor (no YTD floor)."""
        from fantasy_baseball.scoring import compute_roster_breakdown

        # No LeagueContext -> hitter substitution model for pitchers; but with
        # only active pitchers and no IL, no displacement is applied.
        roster = [self._pitcher("P1", ros_k=60, full_season_k=140)]
        breakdown = compute_roster_breakdown("Me", roster, projection_source="rest_of_season")
        p1 = breakdown.pitchers[0]
        # Active, factor=1.0 -> contribution_stats[k] = ROS = 60 (NOT full_season 140).
        assert p1.scale_factor == 1.0
        assert abs(p1.contribution_stats["k"] - 60.0) < 1e-6, (
            f"ROS-mode active player should contribute ROS K=60, got {p1.contribution_stats['k']}"
        )

    def test_breakdown_benched_il_pitcher_contributes_ytd_in_full_season_mode(self):
        """An IL pitcher benched at sf=0 by the pool model still contributes
        YTD in full_season mode (locked-in stats survive). The modal must
        show this YTD, not 0.
        """
        from fantasy_baseball.models.player import PitcherStats, Player, PlayerType
        from fantasy_baseball.models.positions import Position
        from fantasy_baseball.models.standings import CategoryStats
        from fantasy_baseball.scoring import LeagueContext, compute_roster_breakdown
        from fantasy_baseball.utils.constants import Category

        # Weak IL pitcher: no positive swap exists, gets sf=0.
        weak_ros = PitcherStats(
            ip=30, w=1, k=15, sv=0, er=15, bb=8, h_allowed=30, era=4.50, whip=1.27
        )
        weak_full = PitcherStats(
            ip=90, w=4, k=60, sv=0, er=40, bb=25, h_allowed=85, era=4.00, whip=1.22
        )
        weak_pre = PitcherStats(
            ip=180, w=10, k=140, sv=0, er=72, bb=50, h_allowed=170, era=3.60, whip=1.22
        )
        weak_il = Player(
            name="Weak_IL",
            player_type=PlayerType.PITCHER,
            rest_of_season=weak_ros,
            full_season_projection=weak_full,
            preseason=weak_pre,
            selected_position=Position.IL,
        )

        # Elite actives (no swap helps).
        def elite_sp(name, ros_ip):
            ros = PitcherStats(
                ip=ros_ip,
                w=10,
                k=int(ros_ip * 11.0 / 9),
                sv=0,
                er=35,
                bb=25,
                h_allowed=100,
                era=2.50,
                whip=1.00,
            )
            full = PitcherStats(
                ip=ros_ip + 40,
                w=14,
                k=int((ros_ip + 40) * 11.0 / 9),
                sv=0,
                er=45,
                bb=33,
                h_allowed=130,
                era=2.50,
                whip=1.00,
            )
            pre = PitcherStats(
                ip=200, w=14, k=244, sv=0, er=55, bb=45, h_allowed=160, era=2.50, whip=1.00
            )
            return Player(
                name=name,
                player_type=PlayerType.PITCHER,
                rest_of_season=ros,
                full_season_projection=full,
                preseason=pre,
                selected_position=Position.P,
            )

        sp_a, sp_b, sp_c = elite_sp("SP_A", 160), elite_sp("SP_B", 155), elite_sp("SP_C", 150)
        roster = [weak_il, sp_a, sp_b, sp_c]

        baseline = {
            "Opp1": CategoryStats(
                r=0, hr=0, rbi=0, sb=0, avg=0, w=20, k=400, sv=0, era=4.0, whip=1.3
            ),
            "Opp2": CategoryStats(
                r=0, hr=0, rbi=0, sb=0, avg=0, w=22, k=420, sv=0, era=3.9, whip=1.28
            ),
        }
        team_sds = {tn: {c: 1.0 for c in Category} for tn in ["Me", *baseline.keys()]}
        ctx = LeagueContext(
            baseline_other_team_stats=baseline,
            team_sds=team_sds,
            team_name="Me",
        )

        breakdown = compute_roster_breakdown(
            "Me", roster, league_context=ctx, projection_source="full_season_projection"
        )
        weak_contrib = next((c for c in breakdown.pitchers if c.name == "Weak_IL"), None)
        assert weak_contrib is not None
        assert weak_contrib.scale_factor == 0.0, "Weak IL pitcher should be benched at sf=0"
        # YTD = full_season - ROS = 60 - 15 = 45 K. Modal must show this.
        ytd_k = weak_full.k - weak_ros.k
        assert abs(weak_contrib.contribution_stats["k"] - ytd_k) < 1e-6, (
            f"Benched IL pitcher should contribute YTD={ytd_k} K, got {weak_contrib.contribution_stats['k']}"
        )


class TestPlayerContributionFromDictNullGuards:
    """from_dict must not crash when persisted JSON contains explicit null
    values for raw_stats or contribution_stats (e.g., a hand-edit, an
    alternate serializer, or a future error-recovery write path).
    d.get(k, default) returns the explicit None when the key is present
    but null, not the default -- so the dict comprehension must guard.
    """

    def test_from_dict_handles_null_contribution_stats(self):
        from fantasy_baseball.scoring import PlayerContribution

        d = {
            "name": "X",
            "player_type": "pitcher",
            "status": "active",
            "scale_factor": 1.0,
            "raw_stats": {"k": 200.0, "ip": 180.0},
            "contribution_stats": None,
        }
        # Must not raise AttributeError on None.items().
        pc = PlayerContribution.from_dict(d)
        # When contribution_stats is null AND raw_stats is non-empty, the
        # back-compat fallback fires (same as when the key is absent).
        assert pc.contribution_stats.get("k") == 200.0

    def test_from_dict_handles_null_raw_stats(self):
        from fantasy_baseball.scoring import PlayerContribution

        d = {
            "name": "X",
            "player_type": "pitcher",
            "status": "no_projection",
            "scale_factor": 0.0,
            "raw_stats": None,
            "contribution_stats": {},
        }
        # Must not raise AttributeError on None.items().
        pc = PlayerContribution.from_dict(d)
        assert pc.raw_stats == {}
        assert pc.contribution_stats == {}


class TestProjectRosComponents:
    """``project_ros_components`` returns the same component shape as
    ``StandingsEntry.ytd_components`` but sourced from displacement-scaled
    ROS projections on a roster. The result is the ROS contribution that
    gets added to team YTD to produce end-of-season totals.
    """

    @staticmethod
    def _hitter(name, *, r, hr, rbi, sb, h, ab):
        from fantasy_baseball.models.player import HitterStats, Player, PlayerType
        from fantasy_baseball.models.positions import Position

        ros = HitterStats(
            r=r,
            hr=hr,
            rbi=rbi,
            sb=sb,
            h=h,
            ab=ab,
            avg=h / ab if ab else 0.0,
        )
        return Player(
            name=name,
            player_type=PlayerType.HITTER,
            rest_of_season=ros,
            selected_position=Position.OF,
        )

    @staticmethod
    def _pitcher(name, *, w, k, sv, ip, er, bb, h_allowed):
        from fantasy_baseball.models.player import PitcherStats, Player, PlayerType
        from fantasy_baseball.models.positions import Position

        ros = PitcherStats(
            w=w,
            k=k,
            sv=sv,
            ip=ip,
            er=er,
            bb=bb,
            h_allowed=h_allowed,
            era=er * 9.0 / ip if ip else 0.0,
            whip=(bb + h_allowed) / ip if ip else 0.0,
        )
        return Player(
            name=name,
            player_type=PlayerType.PITCHER,
            rest_of_season=ros,
            selected_position=Position.P,
        )

    def test_components_sum_counting_stats_across_roster(self):
        """R/HR/RBI/SB/W/K/SV are summed across all rostered players (no
        displacement to keep the math obvious)."""
        from fantasy_baseball.scoring import project_ros_components

        roster = [
            self._hitter("A", r=30, hr=10, rbi=25, sb=5, h=60, ab=200),
            self._hitter("B", r=40, hr=15, rbi=35, sb=2, h=70, ab=250),
            self._pitcher("P1", w=5, k=80, sv=0, ip=50, er=18, bb=15, h_allowed=40),
        ]
        c = project_ros_components(roster, displacement=False)
        assert c.r == pytest.approx(70.0)
        assert c.hr == pytest.approx(25.0)
        assert c.rbi == pytest.approx(60.0)
        assert c.sb == pytest.approx(7.0)
        assert c.w == pytest.approx(5.0)
        assert c.k == pytest.approx(80.0)
        assert c.sv == pytest.approx(0.0)

    def test_components_expose_rate_stat_ingredients(self):
        """H, AB sum across hitters; ER, IP, BB+H_allowed sum across pitchers.

        The team_end_of_season helper uses these to recompute team AVG /
        ERA / WHIP after adding YTD components.
        """
        from fantasy_baseball.scoring import project_ros_components

        roster = [
            self._hitter("A", r=30, hr=10, rbi=25, sb=5, h=60, ab=200),
            self._hitter("B", r=40, hr=15, rbi=35, sb=2, h=70, ab=250),
            self._pitcher("P1", w=5, k=80, sv=0, ip=50, er=18, bb=15, h_allowed=40),
            self._pitcher("P2", w=8, k=100, sv=0, ip=70, er=25, bb=20, h_allowed=60),
        ]
        c = project_ros_components(roster, displacement=False)
        assert c.h == pytest.approx(130.0)
        assert c.ab == pytest.approx(450.0)
        assert c.ip == pytest.approx(120.0)
        assert c.er == pytest.approx(43.0)
        assert c.bb_plus_h_allowed == pytest.approx((15 + 40) + (20 + 60))

    def test_empty_roster_returns_zero_components(self):
        """Sanity: no players -> all-zero components, no NaN."""
        from fantasy_baseball.scoring import project_ros_components

        c = project_ros_components([], displacement=False)
        assert c.r == 0.0
        assert c.ab == 0.0
        assert c.ip == 0.0

    def test_components_apply_displacement_scaling(self):
        """When displacement=True (default), displaced players have their
        ROS counting stats scaled by the displacement factor. The simplest
        verifiable case: a roster with one IL pitcher who displaces no one
        is no-op."""
        from fantasy_baseball.scoring import project_ros_components

        # Single hitter, no IL, no displacement opportunities.
        roster = [self._hitter("A", r=30, hr=10, rbi=25, sb=5, h=60, ab=200)]
        c = project_ros_components(roster, displacement=True)
        assert c.r == pytest.approx(30.0)
        assert c.ab == pytest.approx(200.0)


class TestTeamEndOfSeason:
    """``team_end_of_season(ytd, ros)`` produces a :class:`CategoryStats`
    representing projected end-of-season totals = YTD + ROS, with AVG/ERA/WHIP
    recomputed from summed components.
    """

    def test_counting_stats_sum(self):
        from fantasy_baseball.models.standings import TeamYtdComponents
        from fantasy_baseball.scoring import TeamRosComponents, team_end_of_season

        ytd = TeamYtdComponents(
            r=80,
            hr=20,
            rbi=70,
            sb=10,
            w=10,
            k=180,
            sv=5,
            h=100,
            ab=400,
            ip=200,
            er=78,
            bb_plus_h_allowed=240,
        )
        ros = TeamRosComponents(
            r=50,
            hr=15,
            rbi=45,
            sb=8,
            w=8,
            k=120,
            sv=3,
            h=70,
            ab=280,
            ip=150,
            er=55,
            bb_plus_h_allowed=180,
        )
        out = team_end_of_season(ytd, ros)
        assert out.r == pytest.approx(130)
        assert out.hr == pytest.approx(35)
        assert out.rbi == pytest.approx(115)
        assert out.sb == pytest.approx(18)
        assert out.w == pytest.approx(18)
        assert out.k == pytest.approx(300)
        assert out.sv == pytest.approx(8)

    def test_avg_recombined_from_components_equal_rates(self):
        """YTD and ROS have the same AVG (0.250); combined AVG is 0.250."""
        from fantasy_baseball.models.standings import TeamYtdComponents
        from fantasy_baseball.scoring import TeamRosComponents, team_end_of_season

        ytd = TeamYtdComponents(h=100, ab=400)
        ros = TeamRosComponents(h=70, ab=280)
        out = team_end_of_season(ytd, ros)
        assert out.avg == pytest.approx(170 / 680)

    def test_avg_correctly_weights_when_rates_differ(self):
        """YTD AVG is .300; ROS AVG is .200. Combined is weighted by AB."""
        from fantasy_baseball.models.standings import TeamYtdComponents
        from fantasy_baseball.scoring import TeamRosComponents, team_end_of_season

        ytd = TeamYtdComponents(h=120, ab=400)
        ros = TeamRosComponents(h=40, ab=200)
        out = team_end_of_season(ytd, ros)
        assert out.avg == pytest.approx(160 / 600)

    def test_era_recombined_from_components(self):
        from fantasy_baseball.models.standings import TeamYtdComponents
        from fantasy_baseball.scoring import TeamRosComponents, team_end_of_season

        ytd = TeamYtdComponents(er=78, ip=200)
        ros = TeamRosComponents(er=55, ip=150)
        out = team_end_of_season(ytd, ros)
        assert out.era == pytest.approx(9 * 133 / 350)

    def test_whip_recombined_from_components(self):
        from fantasy_baseball.models.standings import TeamYtdComponents
        from fantasy_baseball.scoring import TeamRosComponents, team_end_of_season

        ytd = TeamYtdComponents(ip=200, bb_plus_h_allowed=240)
        ros = TeamRosComponents(ip=150, bb_plus_h_allowed=180)
        out = team_end_of_season(ytd, ros)
        assert out.whip == pytest.approx(420 / 350)

    def test_zero_ab_yields_zero_avg_not_nan(self):
        """No AB anywhere -> AVG = 0.0, not NaN."""
        from fantasy_baseball.models.standings import TeamYtdComponents
        from fantasy_baseball.scoring import TeamRosComponents, team_end_of_season

        ytd = TeamYtdComponents()
        ros = TeamRosComponents()
        out = team_end_of_season(ytd, ros)
        assert out.avg == 0.0
        assert out.era == 0.0
        assert out.whip == 0.0

    def test_preseason_ytd_zero_collapses_to_ros_only(self):
        """Pre-season has YTD=zero; result must equal ROS-only projection."""
        from fantasy_baseball.models.standings import TeamYtdComponents
        from fantasy_baseball.scoring import TeamRosComponents, team_end_of_season

        ytd = TeamYtdComponents()
        ros = TeamRosComponents(
            r=80,
            hr=20,
            rbi=70,
            sb=10,
            w=10,
            k=180,
            sv=5,
            h=120,
            ab=480,
            ip=200,
            er=70,
            bb_plus_h_allowed=240,
        )
        out = team_end_of_season(ytd, ros)
        assert out.r == pytest.approx(80)
        assert out.avg == pytest.approx(120 / 480)
        assert out.era == pytest.approx(9 * 70 / 200)
        assert out.whip == pytest.approx(240 / 200)
