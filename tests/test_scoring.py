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
    _prob_beats,
    build_projected_standings,
    build_team_sds,
    project_team_stats,
    score_roto,
)
from fantasy_baseball.utils.constants import Category


def _stats_table(stats_by_team):
    """Build a :class:`ProjectedStandings` from ``{team: {cat_str: value}}``."""
    return ProjectedStandings(
        effective_date=date(2026, 4, 15),
        entries=[
            ProjectedStandingsEntry(team_name=name, stats=CategoryStats.from_dict(stats))
            for name, stats in stats_by_team.items()
        ],
    )


def _hitter(name, r=0, hr=0, rbi=0, sb=0, h=0, ab=0, pa=0,
            positions=None, selected_position=None, status=""):
    return Player(
        name=name,
        player_type=PlayerType.HITTER,
        positions=positions or [],
        rest_of_season=HitterStats(r=r, hr=hr, rbi=rbi, sb=sb, h=h, ab=ab, pa=pa or ab),
        selected_position=selected_position,
        status=status,
    )


def _pitcher(name, w=0, k=0, sv=0, ip=0, er=0, bb=0, h_allowed=0,
             positions=None, selected_position=None, status=""):
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
            _pitcher("Pitcher With Util", w=15, k=200, sv=0, ip=180, er=60,
                     bb=50, h_allowed=150),
        ]
        stats = project_team_stats(roster)
        assert stats[Category.W] == 15
        assert stats[Category.K] == 200
        assert stats[Category.ERA] == pytest.approx(60 * 9 / 180)

    def test_hitter_and_pitcher_both_counted(self):
        roster = [
            _hitter("Hitter", r=80, hr=25, rbi=70, sb=5, h=130, ab=500),
            _pitcher("Pitcher", w=10, k=150, sv=30, ip=60, er=20, bb=15,
                     h_allowed=50),
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
            _pitcher("SP", w=12, k=180, sv=0, ip=200, er=70, bb=50,
                     h_allowed=170),
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


from fantasy_baseball.scoring import project_team_sds
from fantasy_baseball.models.player import HitterStats, PitcherStats, Player, PlayerType
from fantasy_baseball.utils.constants import STAT_VARIANCE


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
        for cat in ["R", "HR", "RBI", "SB", "AVG", "W", "K", "SV", "ERA", "WHIP"]:
            assert sds[cat] == 0.0

    def test_single_hitter_counting_stat(self):
        p = _make_hitter("A", r=80, hr=20, rbi=70, sb=10, h=150, ab=500)
        sds = project_team_sds([p])
        # SD_R = CV_r * sqrt(r^2) = CV_r * r  (single player case)
        assert sds["R"] == pytest.approx(STAT_VARIANCE["r"] * 80)
        assert sds["HR"] == pytest.approx(STAT_VARIANCE["hr"] * 20)

    def test_independence_aggregates_in_quadrature(self):
        a = _make_hitter("A", r=100, hr=0, rbi=0, sb=0, h=0, ab=0)
        b = _make_hitter("B", r=60, hr=0, rbi=0, sb=0, h=0, ab=0)
        sds = project_team_sds([a, b])
        expected = STAT_VARIANCE["r"] * math.sqrt(100**2 + 60**2)
        assert sds["R"] == pytest.approx(expected)

    def test_avg_uses_hits_variance_over_total_ab(self):
        a = _make_hitter("A", r=0, hr=0, rbi=0, sb=0, h=150, ab=500)
        b = _make_hitter("B", r=0, hr=0, rbi=0, sb=0, h=100, ab=400)
        sds = project_team_sds([a, b])
        expected = STAT_VARIANCE["h"] * math.sqrt(150**2 + 100**2) / (500 + 400)
        assert sds["AVG"] == pytest.approx(expected)

    def test_era_scales_by_nine_over_ip(self):
        a = _make_pitcher("A", w=10, k=180, sv=0, ip=180, er=60, bb=40, h_allowed=140)
        b = _make_pitcher("B", w=8, k=140, sv=0, ip=150, er=55, bb=35, h_allowed=130)
        sds = project_team_sds([a, b])
        expected = 9.0 * STAT_VARIANCE["er"] * math.sqrt(60**2 + 55**2) / (180 + 150)
        assert sds["ERA"] == pytest.approx(expected)

    def test_whip_combines_bb_and_h_allowed_variance(self):
        a = _make_pitcher("A", w=0, k=0, sv=0, ip=100, er=0, bb=30, h_allowed=90)
        sds = project_team_sds([a])
        expected = math.sqrt(
            STAT_VARIANCE["bb"]**2 * 30**2
            + STAT_VARIANCE["h_allowed"]**2 * 90**2
        ) / 100
        assert sds["WHIP"] == pytest.approx(expected)

    def test_all_ten_categories_present(self):
        p = _make_hitter("A", r=50, hr=10, rbi=40, sb=5, h=100, ab=400)
        sds = project_team_sds([p])
        assert set(sds.keys()) == {"R", "HR", "RBI", "SB", "AVG", "W", "K", "SV", "ERA", "WHIP"}

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
        assert sds_with_bench["R"] == pytest.approx(sds_active_only["R"])


class TestScoreRoto:
    def test_two_teams_simple(self):
        stats = _stats_table({
            "A": {"R": 900, "HR": 250, "RBI": 850, "SB": 100, "AVG": 0.270,
                   "W": 80, "K": 1200, "SV": 50, "ERA": 3.50, "WHIP": 1.15},
            "B": {"R": 800, "HR": 200, "RBI": 750, "SB": 80, "AVG": 0.260,
                   "W": 70, "K": 1100, "SV": 40, "ERA": 4.00, "WHIP": 1.25},
        })
        roto = score_roto(stats)
        assert roto["A"].total == 20  # wins every category
        assert roto["B"].total == 10

    def test_fractional_tiebreaker(self):
        stats = _stats_table({
            "A": {"R": 900, "HR": 250, "RBI": 850, "SB": 100, "AVG": 0.270,
                   "W": 80, "K": 1200, "SV": 50, "ERA": 3.50, "WHIP": 1.15},
            "B": {"R": 900, "HR": 250, "RBI": 850, "SB": 100, "AVG": 0.270,
                   "W": 80, "K": 1200, "SV": 50, "ERA": 3.50, "WHIP": 1.15},
        })
        roto = score_roto(stats)
        # Tied in everything — both get 1.5 per cat (avg of 1 and 2)
        assert roto["A"].total == pytest.approx(15.0)
        assert roto["B"].total == pytest.approx(15.0)

    def test_inverse_stats_lower_is_better(self):
        stats = _stats_table({
            "A": {"R": 0, "HR": 0, "RBI": 0, "SB": 0, "AVG": 0,
                   "W": 0, "K": 0, "SV": 0, "ERA": 3.00, "WHIP": 1.10},
            "B": {"R": 0, "HR": 0, "RBI": 0, "SB": 0, "AVG": 0,
                   "W": 0, "K": 0, "SV": 0, "ERA": 4.50, "WHIP": 1.30},
        })
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
        bench = _hitter("Bench Guy", r=50, hr=10, rbi=40, sb=5, h=80, ab=300,
                        selected_position=Position.BN)
        active = _hitter("Active", r=80, hr=20, rbi=70, sb=10, h=140, ab=500,
                         selected_position=Position.OF,
                         positions=[Position.OF])
        stats = project_team_stats([active, bench])
        assert stats[Category.R] == 130  # 80 + 50
        assert stats[Category.HR] == 30  # 20 + 10


class TestDisplacementBenchExclusion:
    """Bench players (BN slot, not IL) are excluded when displacement=True."""

    def test_bench_hitter_excluded(self):
        bench = _hitter("Bench", r=50, hr=10, rbi=40, sb=5, h=80, ab=300,
                        selected_position=Position.BN)
        active = _hitter("Active", r=80, hr=20, rbi=70, sb=10, h=140, ab=500,
                         selected_position=Position.OF,
                         positions=[Position.OF])
        stats = project_team_stats([active, bench], displacement=True)
        assert stats[Category.R] == 80
        assert stats[Category.HR] == 20

    def test_bench_pitcher_excluded(self):
        bench = _pitcher("BenchP", w=5, k=60, sv=0, ip=80, er=30, bb=20,
                         h_allowed=70, selected_position=Position.BN)
        active = _pitcher("ActiveP", w=10, k=150, sv=0, ip=180, er=60, bb=50,
                          h_allowed=150, selected_position=Position.SP,
                          positions=[Position.SP])
        stats = project_team_stats([active, bench], displacement=True)
        assert stats[Category.W] == 10
        assert stats[Category.K] == 150

    def test_il_player_on_bench_slot_with_il_status_not_excluded_as_bench(self):
        """A player on BN slot but with IL status is NOT treated as bench —
        they're treated as IL (for displacement purposes)."""
        il_player = _hitter("IL Guy", r=40, hr=8, rbi=30, sb=3, h=60, ab=200,
                            selected_position=Position.BN, status="IL")
        active = _hitter("Active", r=80, hr=20, rbi=70, sb=10, h=140, ab=500,
                         selected_position=Position.OF,
                         positions=[Position.OF])
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
        good_of = _hitter("Good OF", r=90, hr=30, rbi=90, sb=10, h=160, ab=550,
                          positions=[Position.OF],
                          selected_position=Position.OF)
        bad_of = _hitter("Bad OF", r=40, hr=8, rbi=30, sb=2, h=80, ab=300,
                         positions=[Position.OF],
                         selected_position=Position.OF)
        # IL hitter: OF eligible, 200 PA on IL
        il_of = _hitter("IL OF", r=30, hr=5, rbi=20, sb=1, h=50, ab=200,
                        positions=[Position.OF],
                        selected_position=Position.IL, status="IL")

        stats = project_team_stats([good_of, bad_of, il_of], displacement=True)

        # bad_of displaced: factor = max(0, 300 - 200) / 300 = 1/3
        # Totals: good_of full + bad_of scaled + IL full
        assert stats[Category.R] == pytest.approx(90 + 40 / 3 + 30)
        assert stats[Category.HR] == pytest.approx(30 + 8 / 3 + 5)
        assert stats[Category.RBI] == pytest.approx(90 + 30 / 3 + 20)

    def test_il_hitter_fallback_to_worst_hitter_overall(self):
        """When no active hitter shares a position, fallback to worst hitter."""
        ss = _hitter("SS guy", r=50, hr=10, rbi=40, sb=5, h=90, ab=350,
                     positions=[Position.SS],
                     selected_position=Position.SS)
        first = _hitter("1B guy", r=70, hr=25, rbi=80, sb=2, h=130, ab=480,
                        positions=[Position.FIRST_BASE],
                        selected_position=Position.FIRST_BASE)
        # IL hitter is OF eligible — no active OF exists
        il_of = _hitter("IL OF", r=20, hr=4, rbi=15, sb=1, h=40, ab=150,
                        positions=[Position.OF],
                        selected_position=Position.IL, status="IL10")

        stats = project_team_stats([ss, first, il_of], displacement=True)

        # Fallback: displace worst hitter overall. SS has lower SGP than 1B.
        # SS factor = max(0, 350 - 150) / 350 = 200/350 = 4/7
        # Total R = 1B full + SS scaled + IL full
        assert stats[Category.R] == pytest.approx(70 + 50 * (4 / 7) + 20)

    def test_displacement_caps_at_zero(self):
        """When IL player has more playing time than active, factor is 0."""
        active = _hitter("Active", r=40, hr=8, rbi=30, sb=2, h=60, ab=200,
                         positions=[Position.OF],
                         selected_position=Position.OF)
        il_player = _hitter("IL Big", r=50, hr=10, rbi=40, sb=5, h=100, ab=400,
                            positions=[Position.OF],
                            selected_position=Position.IL, status="IL60")

        stats = project_team_stats([active, il_player], displacement=True)

        # factor = max(0, 200 - 400) / 200 = 0
        # Active zeroed out, IL counted in full
        assert stats[Category.R] == 50
        assert stats[Category.HR] == 10

    def test_each_active_displaced_at_most_once(self):
        """Two IL hitters can't both displace the same active player."""
        active = _hitter("Only Active", r=80, hr=20, rbi=70, sb=10, h=140, ab=500,
                         positions=[Position.OF],
                         selected_position=Position.OF)
        il1 = _hitter("IL1", r=30, hr=5, rbi=20, sb=1, h=50, ab=200,
                       positions=[Position.OF],
                       selected_position=Position.IL, status="IL")
        il2 = _hitter("IL2", r=20, hr=3, rbi=15, sb=1, h=40, ab=150,
                       positions=[Position.OF],
                       selected_position=Position.IL_PLUS, status="IL+")

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
        good_sp = _pitcher("Good SP", w=15, k=200, sv=0, ip=190, er=55, bb=40,
                           h_allowed=150, positions=[Position.SP],
                           selected_position=Position.SP)
        bad_sp = _pitcher("Bad SP", w=5, k=80, sv=0, ip=120, er=55, bb=40,
                          h_allowed=110, positions=[Position.SP],
                          selected_position=Position.SP)
        il_sp = _pitcher("IL SP", w=8, k=100, sv=0, ip=130, er=40, bb=30,
                         h_allowed=100, positions=[Position.SP],
                         selected_position=Position.IL, status="IL15")

        stats = project_team_stats([good_sp, bad_sp, il_sp], displacement=True)

        # bad_sp displaced: factor = max(0, 120 - 130) / 120 = 0
        # Total = good_sp + bad_sp*0 + il_sp full
        assert stats[Category.W] == pytest.approx(15 + 8)
        assert stats[Category.K] == pytest.approx(200 + 100)

    def test_rp_displaces_rp(self):
        """IL RP (ip<=100) displaces worst active RP, not an SP."""
        sp = _pitcher("SP", w=12, k=180, sv=0, ip=180, er=60, bb=45,
                      h_allowed=150, positions=[Position.SP],
                      selected_position=Position.SP)
        rp = _pitcher("RP", w=3, k=50, sv=20, ip=60, er=20, bb=15,
                      h_allowed=50, positions=[Position.RP],
                      selected_position=Position.RP)
        il_rp = _pitcher("IL RP", w=1, k=20, sv=10, ip=30, er=10, bb=8,
                         h_allowed=25, positions=[Position.RP],
                         selected_position=Position.IL, status="IL")

        stats = project_team_stats([sp, rp, il_rp], displacement=True)

        # RP displaced: factor = max(0, 60 - 30) / 60 = 0.5
        # Total = SP full + RP*0.5 + IL RP full
        assert stats[Category.W] == pytest.approx(12 + 3 * 0.5 + 1)
        assert stats[Category.SV] == pytest.approx(20 * 0.5 + 10)


class TestDisplacementILSlotAndStatus:
    """IL detection uses both selected_position in IL_SLOTS and status in IL_STATUSES."""

    def test_il_slot_triggers_displacement(self):
        """Player on IL slot is treated as IL even without status string."""
        active = _hitter("Active", r=80, hr=20, rbi=70, sb=10, h=140, ab=500,
                         positions=[Position.OF],
                         selected_position=Position.OF)
        # selected_position=IL but status="" — still counts as IL
        il_player = _hitter("IL slot", r=20, hr=4, rbi=15, sb=1, h=40, ab=150,
                            positions=[Position.OF],
                            selected_position=Position.IL, status="")

        stats = project_team_stats([active, il_player], displacement=True)
        # factor = (500 - 150) / 500 = 0.7
        # Total = active*0.7 + IL full
        assert stats[Category.R] == pytest.approx(80 * 0.7 + 20)

    def test_il_status_on_active_slot_triggers_displacement(self):
        """Player with IL status but on an active slot (e.g., Yahoo quirk)
        is treated as IL."""
        active = _hitter("Active", r=80, hr=20, rbi=70, sb=10, h=140, ab=500,
                         positions=[Position.OF],
                         selected_position=Position.OF)
        # selected_position=OF but status="IL60" — still IL
        il_player = _hitter("IL status", r=30, hr=6, rbi=20, sb=2, h=60, ab=250,
                            positions=[Position.OF],
                            selected_position=Position.OF, status="IL60")

        stats = project_team_stats([active, il_player], displacement=True)
        # factor = (500 - 250) / 500 = 0.5
        # Total = active*0.5 + IL full
        assert stats[Category.R] == pytest.approx(80 * 0.5 + 30)


class TestDisplacementDictInputUnaffected:
    """Dict-input callers (draft scripts) bypass displacement entirely."""

    def test_dict_roster_ignores_displacement(self):
        roster = [
            {"player_type": PlayerType.HITTER, "r": 80, "hr": 20, "rbi": 70,
             "sb": 10, "h": 140, "ab": 500},
            {"player_type": PlayerType.HITTER, "r": 50, "hr": 10, "rbi": 40,
             "sb": 5, "h": 80, "ab": 300, "selected_position": "BN"},
        ]
        stats = project_team_stats(roster, displacement=True)
        # Dicts are never filtered — all summed naively
        assert stats[Category.R] == 130
        assert stats[Category.HR] == 30


class TestDisplacementProcessingOrder:
    """IL players processed in descending playing time order."""

    def test_higher_playing_time_il_processed_first(self):
        """The IL player with more playing time gets first pick of displacement."""
        of1 = _hitter("OF1", r=60, hr=15, rbi=50, sb=8, h=100, ab=400,
                      positions=[Position.OF],
                      selected_position=Position.OF)
        of2 = _hitter("OF2", r=40, hr=8, rbi=30, sb=2, h=70, ab=280,
                      positions=[Position.OF],
                      selected_position=Position.OF)
        # IL1 has 300 ab, IL2 has 100 ab
        il1 = _hitter("IL1", r=50, hr=12, rbi=40, sb=5, h=90, ab=300,
                      positions=[Position.OF],
                      selected_position=Position.IL, status="IL")
        il2 = _hitter("IL2", r=20, hr=3, rbi=10, sb=1, h=30, ab=100,
                      positions=[Position.OF],
                      selected_position=Position.IL, status="IL")

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
        active1 = _hitter("1B", r=60, hr=15, rbi=50, sb=3, h=100, ab=400,
                          positions=[Position.FIRST_BASE],
                          selected_position=Position.FIRST_BASE)
        active2 = _hitter("Worst", r=30, hr=5, rbi=20, sb=1, h=50, ab=250,
                          positions=[Position.SECOND_BASE],
                          selected_position=Position.SECOND_BASE)
        # IL player only has UTIL in positions — no "real" position overlap
        il_util = _hitter("IL Util", r=20, hr=4, rbi=15, sb=1, h=35, ab=150,
                          positions=[Position.UTIL],
                          selected_position=Position.IL, status="IL")

        stats = project_team_stats([active1, active2, il_util], displacement=True)

        # No position match -> fallback to worst hitter overall = active2
        # active2 factor = max(0, 250 - 150) / 250 = 0.4
        # Total = active1 full + active2*0.4 + IL full
        assert stats[Category.R] == pytest.approx(60 + 30 * 0.4 + 20)


class TestDisplacementNoRos:
    """Players with rest_of_season=None are handled gracefully."""

    def test_il_player_without_ros_no_displacement(self):
        """IL player with rest_of_season=None has 0 playing time — no displacement."""
        active = _hitter("Active", r=80, hr=20, rbi=70, sb=10, h=140, ab=500,
                         positions=[Position.OF],
                         selected_position=Position.OF)
        il_no_ros = Player(
            name="No ROS", player_type=PlayerType.HITTER,
            positions=[Position.OF],
            selected_position=Position.IL, status="IL",
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
        teams[f"T{i+1}"] = {
            "R": r, "HR": 0, "RBI": 0, "SB": 0, "AVG": 0.0,
            "W": 0, "K": 0, "SV": 0, "ERA": 0.0, "WHIP": 0.0,
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
        # Huge σ >> any μ gap → every team's pairwise P ≈ 0.5 → pts ≈ (N+1)/2 = 6.5.
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
        # Two teams tied at 100 R with σ=10 each. Moving 1 R changes
        # pts by only ~0.03, not the full 1.0 of a rank flip.
        dict_stats = _twelve_team_dict(
            [100, 100, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50]
        )
        team_names = list(dict_stats.keys())
        sds = {
            t: {c: (10.0 if c == Category.R else 1.0) for c in ALL_CATS}
            for t in team_names
        }
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


# ── build_projected_standings / build_team_sds ──────────────────────


class TestBuildProjectedStandings:
    """Pure helper wrapping ``project_team_stats`` for cache output."""

    def test_returns_projected_standings(self):
        from datetime import date

        from fantasy_baseball.models.standings import ProjectedStandings

        team_rosters: dict[str, list] = {"Alpha": [], "Beta": []}
        result = build_projected_standings(
            team_rosters, effective_date=date(2026, 4, 15)
        )
        assert isinstance(result, ProjectedStandings)
        assert result.effective_date == date(2026, 4, 15)
        assert {e.team_name for e in result.entries} == {"Alpha", "Beta"}

    def test_returns_one_entry_per_team(self):
        from datetime import date

        rosters = {
            "Team A": [_make_hitter("Player1", r=80, hr=20, rbi=70, sb=10,
                                    h=140, ab=500, pa=500)],
            "Team B": [_make_hitter("Player2", r=70, hr=15, rbi=60, sb=8,
                                    h=130, ab=490, pa=490)],
        }
        result = build_projected_standings(rosters, effective_date=date(2026, 4, 15))
        assert len(result.entries) == 2
        team_names = {entry.team_name for entry in result.entries}
        assert team_names == {"Team A", "Team B"}

    def test_each_entry_has_team_name_and_stats(self):
        from datetime import date

        from fantasy_baseball.models.standings import CategoryStats

        rosters = {
            "Team A": [_make_hitter("Player1", r=80, hr=20, rbi=70, sb=10,
                                    h=140, ab=500, pa=500)],
        }
        result = build_projected_standings(rosters, effective_date=date(2026, 4, 15))
        entry = result.entries[0]
        assert entry.team_name == "Team A"
        assert isinstance(entry.stats, CategoryStats)

    def test_stats_covers_all_categories(self):
        from datetime import date

        from fantasy_baseball.utils.constants import ALL_CATEGORIES

        rosters = {
            "Team A": [
                _make_hitter("H1", r=80, hr=20, rbi=70, sb=10,
                             h=140, ab=500, pa=500),
                _make_pitcher("P1", w=10, k=180, sv=0, ip=180,
                              er=70, bb=55, h_allowed=155),
            ],
        }
        result = build_projected_standings(rosters, effective_date=date(2026, 4, 15))
        stats = result.entries[0].stats
        # CategoryStats exposes every 5x5 roto category by enum.
        for cat in ALL_CATEGORIES:
            assert stats[cat] is not None


class TestBuildTeamSDs:
    """Pure helper wrapping ``project_team_sds`` with scale factor."""

    def test_returns_one_dict_per_team(self):
        rosters = {
            "Team A": [_make_hitter("P1", r=80, hr=20, rbi=70, sb=10,
                                    h=140, ab=500, pa=500)],
            "Team B": [_make_hitter("P2", r=70, hr=15, rbi=60, sb=8,
                                    h=130, ab=490, pa=490)],
        }
        result = build_team_sds(rosters, sd_scale=1.0)
        assert set(result.keys()) == {"Team A", "Team B"}

    def test_sd_scale_multiplies_each_value(self):
        rosters = {
            "Team A": [_make_hitter("P1", r=80, hr=20, rbi=70, sb=10,
                                    h=140, ab=500, pa=500)],
        }
        unscaled = build_team_sds(rosters, sd_scale=1.0)
        scaled = build_team_sds(rosters, sd_scale=0.5)
        for cat, sd in unscaled["Team A"].items():
            assert scaled["Team A"][cat] == pytest.approx(sd * 0.5)

    def test_sd_scale_zero_yields_zero_sds(self):
        rosters = {
            "Team A": [_make_hitter("P1", r=80, hr=20, rbi=70, sb=10,
                                    h=140, ab=500, pa=500)],
        }
        result = build_team_sds(rosters, sd_scale=0.0)
        for sd in result["Team A"].values():
            assert sd == 0.0
