import math

import pytest

from fantasy_baseball.models.player import (
    HitterStats,
    PitcherStats,
    Player,
    PlayerType,
)
from fantasy_baseball.models.positions import Position
from fantasy_baseball.scoring import ALL_CATS, _prob_beats, project_team_stats, score_roto


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
        assert stats["W"] == 15
        assert stats["K"] == 200
        assert stats["ERA"] == pytest.approx(60 * 9 / 180)

    def test_hitter_and_pitcher_both_counted(self):
        roster = [
            _hitter("Hitter", r=80, hr=25, rbi=70, sb=5, h=130, ab=500),
            _pitcher("Pitcher", w=10, k=150, sv=30, ip=60, er=20, bb=15,
                     h_allowed=50),
        ]
        stats = project_team_stats(roster)
        assert stats["R"] == 80
        assert stats["HR"] == 25
        assert stats["W"] == 10
        assert stats["SV"] == 30
        assert stats["AVG"] == pytest.approx(130 / 500)
        assert stats["ERA"] == pytest.approx(20 * 9 / 60)
        assert stats["WHIP"] == pytest.approx((15 + 50) / 60)

    def test_empty_roster(self):
        stats = project_team_stats([])
        assert stats["R"] == 0
        assert stats["AVG"] == 0
        assert stats["ERA"] == 99
        assert stats["WHIP"] == 99

    def test_pitchers_only(self):
        roster = [
            _pitcher("SP", w=12, k=180, sv=0, ip=200, er=70, bb=50,
                     h_allowed=170),
        ]
        stats = project_team_stats(roster)
        assert stats["R"] == 0
        assert stats["AVG"] == 0
        assert stats["W"] == 12

    def test_hitters_only(self):
        roster = [
            _hitter("H", r=90, hr=35, rbi=100, sb=15, h=160, ab=580),
        ]
        stats = project_team_stats(roster)
        assert stats["W"] == 0
        assert stats["ERA"] == 99
        assert stats["R"] == 90

    def test_player_without_ros_is_skipped(self):
        """Players unmatched to projections have ``rest_of_season=None`` and should
        contribute nothing to team totals rather than raising."""
        roster = [
            _hitter("Good", r=80, hr=25, rbi=70, sb=5, h=130, ab=500),
            Player(name="Unmatched", player_type=PlayerType.HITTER, rest_of_season=None),
        ]
        stats = project_team_stats(roster)
        assert stats["R"] == 80
        assert stats["HR"] == 25
        assert stats["AVG"] == pytest.approx(130 / 500)


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


class TestScoreRoto:
    def test_two_teams_simple(self):
        stats = {
            "A": {"R": 900, "HR": 250, "RBI": 850, "SB": 100, "AVG": 0.270,
                   "W": 80, "K": 1200, "SV": 50, "ERA": 3.50, "WHIP": 1.15},
            "B": {"R": 800, "HR": 200, "RBI": 750, "SB": 80, "AVG": 0.260,
                   "W": 70, "K": 1100, "SV": 40, "ERA": 4.00, "WHIP": 1.25},
        }
        roto = score_roto(stats)
        assert roto["A"]["total"] == 20  # wins every category
        assert roto["B"]["total"] == 10

    def test_fractional_tiebreaker(self):
        stats = {
            "A": {"R": 900, "HR": 250, "RBI": 850, "SB": 100, "AVG": 0.270,
                   "W": 80, "K": 1200, "SV": 50, "ERA": 3.50, "WHIP": 1.15},
            "B": {"R": 900, "HR": 250, "RBI": 850, "SB": 100, "AVG": 0.270,
                   "W": 80, "K": 1200, "SV": 50, "ERA": 3.50, "WHIP": 1.15},
        }
        roto = score_roto(stats)
        # Tied in everything — both get 1.5 per cat (avg of 1 and 2)
        assert roto["A"]["total"] == pytest.approx(15.0)
        assert roto["B"]["total"] == pytest.approx(15.0)

    def test_inverse_stats_lower_is_better(self):
        stats = {
            "A": {"R": 0, "HR": 0, "RBI": 0, "SB": 0, "AVG": 0,
                   "W": 0, "K": 0, "SV": 0, "ERA": 3.00, "WHIP": 1.10},
            "B": {"R": 0, "HR": 0, "RBI": 0, "SB": 0, "AVG": 0,
                   "W": 0, "K": 0, "SV": 0, "ERA": 4.50, "WHIP": 1.30},
        }
        roto = score_roto(stats)
        assert roto["A"]["ERA_pts"] == 2  # lower ERA = better = more points
        assert roto["B"]["ERA_pts"] == 1

    def test_all_categories_present(self):
        stats = {
            "A": {c: 1 for c in ALL_CATS},
        }
        roto = score_roto(stats)
        for c in ALL_CATS:
            assert f"{c}_pts" in roto["A"]
        assert "total" in roto["A"]


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
        assert stats["R"] == 130  # 80 + 50
        assert stats["HR"] == 30  # 20 + 10


class TestDisplacementBenchExclusion:
    """Bench players (BN slot, not IL) are excluded when displacement=True."""

    def test_bench_hitter_excluded(self):
        bench = _hitter("Bench", r=50, hr=10, rbi=40, sb=5, h=80, ab=300,
                        selected_position=Position.BN)
        active = _hitter("Active", r=80, hr=20, rbi=70, sb=10, h=140, ab=500,
                         selected_position=Position.OF,
                         positions=[Position.OF])
        stats = project_team_stats([active, bench], displacement=True)
        assert stats["R"] == 80
        assert stats["HR"] == 20

    def test_bench_pitcher_excluded(self):
        bench = _pitcher("BenchP", w=5, k=60, sv=0, ip=80, er=30, bb=20,
                         h_allowed=70, selected_position=Position.BN)
        active = _pitcher("ActiveP", w=10, k=150, sv=0, ip=180, er=60, bb=50,
                          h_allowed=150, selected_position=Position.SP,
                          positions=[Position.SP])
        stats = project_team_stats([active, bench], displacement=True)
        assert stats["W"] == 10
        assert stats["K"] == 150

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
        assert stats["R"] == pytest.approx(80 * 0.6 + 40)


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
        assert stats["R"] == pytest.approx(90 + 40 / 3 + 30)
        assert stats["HR"] == pytest.approx(30 + 8 / 3 + 5)
        assert stats["RBI"] == pytest.approx(90 + 30 / 3 + 20)

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
        assert stats["R"] == pytest.approx(70 + 50 * (4 / 7) + 20)

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
        assert stats["R"] == 50
        assert stats["HR"] == 10

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
        assert stats["R"] == pytest.approx(80 * 0.6 + 30 + 20)
        assert stats["HR"] == pytest.approx(20 * 0.6 + 5 + 3)


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
        assert stats["W"] == pytest.approx(15 + 8)
        assert stats["K"] == pytest.approx(200 + 100)

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
        assert stats["W"] == pytest.approx(12 + 3 * 0.5 + 1)
        assert stats["SV"] == pytest.approx(20 * 0.5 + 10)


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
        assert stats["R"] == pytest.approx(80 * 0.7 + 20)

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
        assert stats["R"] == pytest.approx(80 * 0.5 + 30)


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
        assert stats["R"] == 130
        assert stats["HR"] == 30


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
        assert stats["R"] == pytest.approx(60 * 0.75 + 50 + 20)
        assert stats["HR"] == pytest.approx(15 * 0.75 + 12 + 3)


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
        assert stats["R"] == pytest.approx(60 + 30 * 0.4 + 20)


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
        assert stats["R"] == 80
        assert stats["HR"] == 20
