import pytest

from fantasy_baseball.lineup.il_return_planner import (
    IlReturnPlanResult,
    Move,
    MovePlan,
    _activate,
    _build_moves,
    _build_pool,
    _counts_against_cap,
    _solve_lineup,
    _tops_differ,
    healthy_rest_of_season,
    plan_il_returns,
    plan_il_returns_scenarios,
    roster_capacity,
)
from fantasy_baseball.models.player import HitterStats, PitcherStats, Player, PlayerType
from fantasy_baseball.models.positions import Position
from fantasy_baseball.models.standings import ProjectedStandings

ROSTER_SLOTS = {
    "C": 1,
    "1B": 1,
    "2B": 1,
    "3B": 1,
    "SS": 1,
    "OF": 3,
    "UTIL": 1,
    "P": 3,
    "BN": 1,
    "IL": 2,
}

TEAM_NAME = "Test Team"


def _standings():
    base = {
        "R": 800,
        "HR": 200,
        "RBI": 800,
        "SB": 100,
        "AVG": 0.260,
        "W": 70,
        "K": 1200,
        "SV": 50,
        "ERA": 3.50,
        "WHIP": 1.20,
        "AB": 5000,
        "H": 1300,
        "IP": 1400,
        "ER": 560,
        "BB": 420,
        "H_ALLOWED": 1300,
    }
    return ProjectedStandings.from_json(
        {
            "effective_date": "2026-04-01",
            "teams": [
                {"name": TEAM_NAME, "stats": dict(base)},
                {"name": "Opponent", "stats": {**base, "SV": 30, "ERA": 3.80}},
            ],
        }
    )


def _contending_standings():
    """Standings where the team is contending in strikeouts (opponent K=450).

    The Webb/Hader roster's pitchers share identical IP/ER/BB/H_allowed/W,
    so K is the only lever distinguishing them. Webb-out lineups top out at
    425 K and Webb-in lineups reach 475-480, so an opponent K of 450 makes
    Webb's strikeouts flip the category -- Webb is then strictly worth
    starting (no value-neutral tie).
    """
    base = {
        "R": 800,
        "HR": 200,
        "RBI": 800,
        "SB": 100,
        "AVG": 0.260,
        "W": 70,
        "K": 1200,
        "SV": 50,
        "ERA": 3.50,
        "WHIP": 1.20,
        "AB": 5000,
        "H": 1300,
        "IP": 1400,
        "ER": 560,
        "BB": 420,
        "H_ALLOWED": 1300,
    }
    return ProjectedStandings.from_json(
        {
            "effective_date": "2026-04-01",
            "teams": [
                {"name": TEAM_NAME, "stats": dict(base)},
                {"name": "Opponent", "stats": {**base, "K": 450, "SV": 30, "ERA": 3.80}},
            ],
        }
    )


def _pitcher(name, slot=None, status=""):
    return Player(
        name=name,
        player_type=PlayerType.PITCHER,
        positions=[Position.P],
        rest_of_season=PitcherStats(
            ip=60.0,
            w=3.0,
            k=60.0,
            sv=0.0,
            er=20.0,
            bb=20.0,
            h_allowed=50.0,
            era=3.00,
            whip=1.17,
        ),
        selected_position=Position.parse(slot) if slot else None,
        status=status,
    )


def _hitter(name, positions, slot=None, **stats):
    return Player(
        name=name,
        player_type=PlayerType.HITTER,
        positions=[Position.parse(p) for p in positions],
        rest_of_season=HitterStats(
            pa=int(stats.get("ab", 500) * 1.15),
            ab=stats.get("ab", 500),
            h=stats.get("h", 130),
            r=stats.get("r", 70),
            hr=stats.get("hr", 20),
            rbi=stats.get("rbi", 70),
            sb=stats.get("sb", 5),
            avg=stats.get("avg", 0.260),
        ),
        selected_position=Position.parse(slot) if slot else None,
    )


def _good_pitcher(name, **stats):
    return Player(
        name=name,
        player_type=PlayerType.PITCHER,
        positions=[Position.P],
        rest_of_season=PitcherStats(
            ip=stats.get("ip", 180.0),
            w=stats.get("w", 14.0),
            k=stats.get("k", 200.0),
            sv=stats.get("sv", 0.0),
            er=stats.get("er", 56.0),
            bb=stats.get("bb", 35.0),
            h_allowed=stats.get("h_allowed", 150.0),
            era=stats.get("era", 2.80),
            whip=stats.get("whip", 1.05),
        ),
    )


class TestCapacity:
    def test_capacity_excludes_il_slots_only(self):
        # 9 hitter slots (C,1B,2B,3B,SS,OF*3,UTIL) + 3 P + 1 BN = 13; IL excluded.
        assert roster_capacity(ROSTER_SLOTS) == 13

    def test_bn_il_status_counts_against_cap(self):
        webb = _pitcher("Webb", slot="BN", status="IL10")
        assert _counts_against_cap(webb) is True

    def test_true_il_slot_does_not_count(self):
        hader = _pitcher("Hader", slot="IL", status="IL15")
        assert _counts_against_cap(hader) is False

    def test_active_slot_counts(self):
        active = _pitcher("Active", slot="P")
        assert _counts_against_cap(active) is True


class TestDataclasses:
    def test_move_to_dict(self):
        m = Move(name="Webb", player_type="pitcher", from_slot="BN", to_slot="P")
        assert m.to_dict() == {
            "name": "Webb",
            "player_type": "pitcher",
            "from_slot": "BN",
            "to_slot": "P",
        }

    def test_move_plan_to_dict_rounds_delta(self):
        plan = MovePlan(
            drops=["Scrub"],
            moves=[Move("Webb", "pitcher", "BN", "P")],
            delta_roto=-0.123,
            band={"mean": -0.12, "sd": 0.4, "p_positive": 0.4, "verdict": "coin-flip"},
        )
        d = plan.to_dict()
        assert d["drops"] == ["Scrub"]
        assert d["delta_roto"] == -0.12
        assert d["moves"][0]["name"] == "Webb"
        assert d["band"]["verdict"] == "coin-flip"

    def test_result_to_dict(self):
        res = IlReturnPlanResult(activating=["Hader"], capacity=13, overflow=1, plans=[])
        d = res.to_dict()
        assert d == {
            "activating": ["Hader"],
            "capacity": 13,
            "overflow": 1,
            "plans": [],
            "warning": None,
        }


class TestActivate:
    def test_activate_clears_il_signals(self):
        hader = _pitcher("Hader", slot="IL", status="IL15")
        cleared = _activate(hader)
        assert cleared.status == ""
        assert cleared.selected_position is None
        assert cleared.name == "Hader"
        # Original is untouched (dataclasses.replace returns a copy).
        assert hader.status == "IL15"


class TestBuildPool:
    def test_pool_includes_counted_bodies_plus_returning_il_slot_players(self):
        active = _pitcher("Active", slot="P")
        webb = _pitcher("Webb", slot="BN", status="IL10")  # counts (BN)
        hader = _pitcher("Hader", slot="IL", status="IL15")  # exempt (IL slot)
        parked = _pitcher("Parked", slot="IL", status="IL60")  # not activated
        roster = [active, webb, hader, parked]

        pool = _build_pool(roster, activating_il=[webb, hader])
        names = {p.name for p in pool}
        # Active + Webb (already counted) + Hader (added from IL). Parked excluded.
        assert names == {"Active", "Webb", "Hader"}
        # Activated players have IL signals cleared.
        webb_p = next(p for p in pool if p.name == "Webb")
        hader_p = next(p for p in pool if p.name == "Hader")
        assert webb_p.status == "" and webb_p.selected_position is None
        assert hader_p.status == "" and hader_p.selected_position is None
        # Non-activated active player is unchanged.
        active_p = next(p for p in pool if p.name == "Active")
        assert active_p.selected_position == Position.P

    def test_two_way_activation_does_not_touch_the_other_row(self):
        """A two-way player is two rows sharing a name; activating his pitcher
        row from IL must not clear the IL signals on -- or dedup away -- his
        active hitter row. Pre-#190 _build_pool keyed on bare name and did both
        (the pitcher row was excluded and the hitter row wrongly activated)."""
        bat = _hitter("Two Way", ["OF"], slot="OF")
        arm = _pitcher("Two Way", slot="IL", status="IL15")
        other = _pitcher("Other", slot="P")
        roster = [bat, arm, other]

        pool = _build_pool(roster, activating_il=[arm])

        # Both rows survive -- the pitcher row was not deduped away by name.
        assert {p.player_key for p in pool} == {
            bat.player_key,
            arm.player_key,
            other.player_key,
        }
        # The hitter row keeps its active slot (NOT activated).
        bat_p = next(p for p in pool if p.player_key == bat.player_key)
        assert bat_p.selected_position == Position.OF
        # The pitcher row is the one activated (IL signals cleared).
        arm_p = next(p for p in pool if p.player_key == arm.player_key)
        assert arm_p.status == "" and arm_p.selected_position is None


class TestSolveLineup:
    def test_solver_returns_active_and_bench(self):
        hitters = [_hitter("OF1", ["OF"])]
        pitchers = [
            _good_pitcher("Ace", k=220),
            _good_pitcher("Mid", k=150, era=3.5, whip=1.2),
            _good_pitcher("Low", k=120, era=4.0, whip=1.3),
        ]
        slots = {"OF": 1, "P": 1, "BN": 1, "IL": 0}
        h_assign, ps, pb = _solve_lineup(
            hitters + pitchers, slots, _standings(), TEAM_NAME, None, 1.0
        )
        assert len(h_assign) == 1
        assert h_assign[0].name == "OF1"
        assert len(ps) == 1  # one P slot
        assert len(pb) == 2  # two pitchers benched
        assert ps[0].name in {"Ace", "Mid", "Low"}


class TestBuildMoves:
    def test_moves_capture_activation_bench_and_drop(self):
        active = _pitcher("Active", slot="P")
        webb = _pitcher("Webb", slot="BN", status="IL10")
        hader = _pitcher("Hader", slot="IL", status="IL15")
        scrub = _pitcher("Scrub", slot="P")
        roster = [active, webb, hader, scrub]
        pool = _build_pool(roster, [webb, hader])
        assert {p.name for p in pool} == {"Active", "Webb", "Hader", "Scrub"}

        from fantasy_baseball.lineup.optimizer import PitcherStarter

        active_player = next(p for p in pool if p.name == "Active")
        webb_player = next(p for p in pool if p.name == "Webb")
        pitcher_starters = [
            PitcherStarter(name="Active", player=active_player, roto_delta=0.0),
            PitcherStarter(name="Webb", player=webb_player, roto_delta=0.0),
        ]
        moves = _build_moves(
            roster=roster,
            pool=pool,
            hitter_assignments=[],
            pitcher_starters=pitcher_starters,
            dropped_keys={scrub.player_key},
        )
        by_name = {m.name: m for m in moves}
        # Webb activates from BN -> P
        assert by_name["Webb"].from_slot == "BN"
        assert by_name["Webb"].to_slot == "P"
        # Hader was IL, not active, not dropped -> goes to BN
        assert by_name["Hader"].from_slot == "IL"
        assert by_name["Hader"].to_slot == "BN"
        # Scrub dropped
        assert by_name["Scrub"].to_slot == "DROP"
        assert by_name["Scrub"].from_slot == "P"
        # Active stays in P -> no move emitted
        assert "Active" not in by_name
        # player_type populated
        assert by_name["Webb"].player_type == "pitcher"


SMALL_SLOTS = {
    "C": 1,
    "1B": 1,
    "2B": 1,
    "3B": 1,
    "SS": 1,
    "OF": 3,
    "UTIL": 1,
    "P": 3,
    "BN": 1,
    "IL": 2,
}  # 9 hitter + 3 P + 1 BN = capacity 13


def _full_hitters():
    specs = [
        ("C1", ["C"]),
        ("1B1", ["1B"]),
        ("2B1", ["2B"]),
        ("3B1", ["3B"]),
        ("SS1", ["SS"]),
        ("OFa", ["OF"]),
        ("OFb", ["OF"]),
        ("OFc", ["OF"]),
        ("UT", ["1B"]),
    ]
    return [
        _hitter(n, pos, r=75, hr=22, rbi=75, sb=8, avg=0.275, ab=520, h=143) for n, pos in specs
    ]


def _webb_hader_roster():
    hitters = _full_hitters()  # 9 counted
    sp1 = _good_pitcher("SP1", k=160, era=3.4, whip=1.15)
    sp2 = _good_pitcher("SP2", k=155, era=3.5, whip=1.18)
    scrub = _pitcher("Scrub", slot="P")  # weak (ip=60,k=60,era=3.0 defaults)
    sp1.selected_position = Position.P
    sp2.selected_position = Position.P
    webb = _good_pitcher("Webb", k=210, era=2.7, whip=1.02)
    webb.selected_position = Position.BN
    webb.status = "IL10"  # BN + IL status -> counts
    hader = _good_pitcher("Hader", k=110, sv=35, era=2.4, whip=0.95)
    hader.selected_position = Position.parse("IL")
    hader.status = "IL15"  # true IL slot -> exempt
    return [*hitters, sp1, sp2, scrub, webb, hader]


class TestPlanIlReturns:
    def test_webb_hader_forces_one_drop_and_benches_a_pitcher(self):
        roster = _webb_hader_roster()
        webb = next(p for p in roster if p.name == "Webb")
        hader = next(p for p in roster if p.name == "Hader")

        result = plan_il_returns(
            roster,
            [webb, hader],
            SMALL_SLOTS,
            projected_standings=_contending_standings(),
            team_name=TEAM_NAME,
            fraction_remaining=1.0,
            team_sds=None,
        )

        assert result.overflow == 1, "activating Hader (IL slot) forces exactly one drop"
        assert result.capacity == 13
        assert result.plans, "expected at least one plan"
        assert len(result.plans) <= 5

        top = result.plans[0]
        assert len(top.drops) == 1
        assert top.drops == ["Scrub"]

        by_name = {m.name: m for m in top.moves}
        assert by_name["Webb"].to_slot == "P"
        assert by_name["Hader"].to_slot == "P"
        benched = [m for m in top.moves if m.to_slot == "BN" and m.player_type == "pitcher"]
        assert len(benched) == 1
        assert by_name["Scrub"].to_slot == "DROP"

    def test_plans_sorted_by_delta_roto_desc(self):
        roster = _webb_hader_roster()
        webb = next(p for p in roster if p.name == "Webb")
        hader = next(p for p in roster if p.name == "Hader")
        result = plan_il_returns(
            roster,
            [webb, hader],
            SMALL_SLOTS,
            projected_standings=_contending_standings(),
            team_name=TEAM_NAME,
            fraction_remaining=1.0,
            team_sds=None,
        )
        deltas = [p.delta_roto for p in result.plans]
        assert deltas == sorted(deltas, reverse=True)
        for p in result.plans:
            assert set(p.band.keys()) == {"mean", "sd", "p_positive", "verdict"}

    def test_no_activation_returns_empty(self):
        roster = _webb_hader_roster()
        result = plan_il_returns(
            roster,
            [],
            SMALL_SLOTS,
            projected_standings=_contending_standings(),
            team_name=TEAM_NAME,
            fraction_remaining=1.0,
            team_sds=None,
        )
        assert result.plans == []
        assert result.activating == []

    def test_open_bench_means_no_drop(self):
        roster = [p for p in _webb_hader_roster() if p.name != "Scrub"]
        webb = next(p for p in roster if p.name == "Webb")
        hader = next(p for p in roster if p.name == "Hader")
        result = plan_il_returns(
            roster,
            [webb, hader],
            SMALL_SLOTS,
            projected_standings=_contending_standings(),
            team_name=TEAM_NAME,
            fraction_remaining=1.0,
            team_sds=None,
        )
        assert result.overflow == 0
        assert len(result.plans) == 1
        assert result.plans[0].drops == []

    def test_activating_a_hitter_is_symmetric(self):
        # Roster: 1 C, 2 OF, 1 P, 1 BN, 1 IL -> capacity = 5.
        # Five players fill all five counted slots; Trout is parked in the IL
        # slot (exempt). Activating Trout brings the pool to 6 > capacity=5,
        # forcing exactly one drop.
        slots = {"C": 1, "OF": 2, "P": 1, "BN": 1, "IL": 1}
        c1 = _hitter("C1", ["C"], r=70, hr=18, rbi=65, sb=4, avg=0.270, ab=500, h=135)
        of1 = _hitter("OF1", ["OF"], r=80, hr=24, rbi=80, sb=10, avg=0.280, ab=540, h=151)
        of2 = _hitter("OF2", ["OF"], r=30, hr=4, rbi=20, sb=1, avg=0.215, ab=300, h=64)  # weakest
        sp1 = _good_pitcher("SP1", k=170)
        bn1 = _hitter("BN1", ["OF"], r=55, hr=12, rbi=50, sb=3, avg=0.250, ab=400, h=100)
        sp1.selected_position = Position.P
        c1.selected_position = Position.C
        of1.selected_position = Position.OF
        of2.selected_position = Position.OF
        bn1.selected_position = Position.BN
        # Elite IL outfielder returning.
        trout = _hitter("Trout", ["OF"], r=110, hr=45, rbi=120, sb=25, avg=0.320, ab=560, h=179)
        trout.selected_position = Position.parse("IL")
        trout.status = "IL15"
        # counted = 5 (c1, of1, of2, sp1, bn1); Trout in IL slot is exempt.
        roster = [c1, of1, of2, sp1, bn1, trout]

        result = plan_il_returns(
            roster,
            [trout],
            slots,
            projected_standings=_standings(),
            team_name=TEAM_NAME,
            fraction_remaining=1.0,
            team_sds=None,
        )

        assert result.capacity == 5
        assert result.overflow == 1  # activating Trout (IL slot) forces one drop
        assert result.plans, "expected at least one plan"
        top = result.plans[0]
        assert len(top.drops) == 1
        # The weakest body (OF2) is the cheapest drop -> top plan (deltaRoto-max,
        # tie-broken by lowest SGP).
        assert top.drops == ["OF2"]
        # Symmetry: the returning hitter is integrated off the IL (no longer "IL").
        by_name = {m.name: m for m in top.moves}
        assert by_name["Trout"].from_slot == "IL"
        assert by_name["Trout"].to_slot != "IL"
        assert by_name["Trout"].player_type == "hitter"
        # The drop is rendered as a DROP move.
        assert by_name["OF2"].to_slot == "DROP"


class TestHealthyRestOfSeason:
    def _cruz(self):
        # Injury-reduced ROS (175 PA) + healthy preseason (543 PA).
        ros = HitterStats(
            pa=175.0,
            ab=154.0,
            h=37.7,
            r=25.0,
            hr=8.0,
            rbi=23.0,
            sb=10.0,
            g=40.0,
            avg=0.245,
            sgp=4.52,
        )
        pre = HitterStats(
            pa=543.0, ab=478.0, h=114.0, r=74.0, hr=23.0, rbi=68.0, sb=28.0, g=127.0, avg=0.239
        )
        return Player(
            name="Cruz",
            player_type=PlayerType.HITTER,
            positions=[Position.OF],
            rest_of_season=ros,
            preseason=pre,
        )

    def test_hitter_scales_volume_preserves_rate_and_clears_sgp(self):
        p = self._cruz()
        out = healthy_rest_of_season(p, fraction_remaining=0.41)
        assert out is not None
        scale = (543.0 * 0.41) / 175.0
        assert out.rest_of_season.pa == pytest.approx(543.0 * 0.41)
        assert out.rest_of_season.hr == pytest.approx(8.0 * scale)
        assert out.rest_of_season.sb == pytest.approx(10.0 * scale)
        assert out.rest_of_season.g == pytest.approx(40.0 * scale)
        assert out.rest_of_season.avg == pytest.approx(0.245)  # rate preserved
        assert out.rest_of_season.sgp is None  # cached SGP cleared
        # Original object untouched (transform returns a copy).
        assert p.rest_of_season.pa == 175.0
        assert p.rest_of_season.sgp == 4.52

    def test_pitcher_scales_ip_and_gs_preserves_rate(self):
        ros = PitcherStats(
            ip=43.0,
            w=3.0,
            k=53.0,
            sv=0.0,
            er=17.0,
            bb=15.0,
            h_allowed=40.0,
            g=9.0,
            gs=9.0,
            era=3.49,
            whip=1.22,
            sgp=3.48,
        )
        pre = PitcherStats(
            ip=150.0,
            w=10.0,
            k=180.0,
            sv=0.0,
            er=60.0,
            bb=45.0,
            h_allowed=130.0,
            g=28.0,
            gs=28.0,
            era=3.60,
            whip=1.17,
        )
        p = Player(
            name="Snell",
            player_type=PlayerType.PITCHER,
            positions=[Position.P],
            rest_of_season=ros,
            preseason=pre,
        )
        out = healthy_rest_of_season(p, fraction_remaining=0.41)
        assert out is not None
        scale = (150.0 * 0.41) / 43.0
        assert out.rest_of_season.ip == pytest.approx(150.0 * 0.41)
        assert out.rest_of_season.k == pytest.approx(53.0 * scale)
        assert out.rest_of_season.gs == pytest.approx(9.0 * scale)
        assert out.rest_of_season.era == pytest.approx(3.49)  # rate preserved
        assert out.rest_of_season.whip == pytest.approx(1.22)
        assert out.rest_of_season.sgp is None

    def test_none_when_no_preseason(self):
        p = self._cruz()
        p.preseason = None
        assert healthy_rest_of_season(p, 0.41) is None

    def test_none_when_current_volume_zero(self):
        p = self._cruz()
        p.rest_of_season = HitterStats(
            pa=0.0, ab=0.0, h=0.0, r=0.0, hr=0.0, rbi=0.0, sb=0.0, g=0.0, avg=0.0
        )
        assert healthy_rest_of_season(p, 0.41) is None

    def test_none_when_not_volume_suppressed(self):
        # preseason.pa * fr = 500 * 0.41 = 205 <= current 300 -> no adjustment.
        p = self._cruz()
        p.rest_of_season = HitterStats(
            pa=300.0, ab=270.0, h=75.0, r=45.0, hr=12.0, rbi=40.0, sb=6.0, g=70.0, avg=0.278
        )
        p.preseason = HitterStats(
            pa=500.0, ab=450.0, h=125.0, r=70.0, hr=20.0, rbi=65.0, sb=10.0, g=150.0, avg=0.278
        )
        assert healthy_rest_of_season(p, 0.41) is None


class TestPlanIlReturnsScenarios:
    def _roster_with_il_hitter(self):
        """9 regular hitters (fill the 9 active hitter slots) + a BN hitter +
        3 pitchers + an IL-slot returnee with a healthy preseason. Counted
        bodies = 9 + 1 BN + 3 P = 13 = SMALL_SLOTS capacity, so activating the
        IL-slot returnee brings the pool to 14 and forces exactly one drop
        (overflow 1)."""
        hitters = _full_hitters()  # 9 counted, fill the 9 hitter active slots
        bn1 = _hitter(
            "BN1", ["OF"], slot="BN", r=40, hr=6, rbi=35, sb=2, avg=0.240, ab=300, h=72
        )  # 13th counted body
        sp1 = _good_pitcher("SP1", k=160, era=3.4, whip=1.15)
        sp2 = _good_pitcher("SP2", k=155, era=3.5, whip=1.18)
        sp1.selected_position = Position.P
        sp2.selected_position = Position.P
        scrub = _pitcher("Scrub", slot="P")  # weak counted pitcher
        # IL-slot returnee: injury ROS (~177 PA, ab*1.15), healthy preseason (543 PA).
        cruz = _hitter(
            "Cruz", ["OF"], slot="IL", r=25, hr=8, rbi=23, sb=10, avg=0.245, ab=154, h=37
        )
        cruz.status = "IL10"
        cruz.preseason = HitterStats(
            pa=543.0, ab=478.0, h=114.0, r=74.0, hr=23.0, rbi=68.0, sb=28.0, g=127.0, avg=0.239
        )
        return [*hitters, bn1, sp1, sp2, scrub, cruz]

    def _call(self, roster, activating):
        return plan_il_returns_scenarios(
            roster,
            activating,
            SMALL_SLOTS,
            projected_standings=_standings(),
            team_name=TEAM_NAME,
            fraction_remaining=0.41,
            team_sds=None,
        )

    def test_suppressed_il_slot_returnee_produces_healthy_scenario(self):
        roster = self._roster_with_il_hitter()
        cruz = next(p for p in roster if p.name == "Cruz")
        res = self._call(roster, [cruz])

        assert res.as_projected.overflow == 1  # IL-slot returnee forces one drop
        assert res.if_healthy is not None
        assert len(res.adjusted) == 1
        adj = res.adjusted[0]
        assert adj["name"] == "Cruz"
        assert adj["vol_unit"] == "PA"
        cruz_pa = cruz.rest_of_season.pa  # _hitter sets pa = int(ab * 1.15) = 177
        assert adj["vol_projected"] == pytest.approx(round(cruz_pa, 1))
        assert adj["vol_healthy"] > adj["vol_projected"]
        # Healthy volume is preseason.pa prorated: 543 * 0.41 ~= 222.6 PA.
        assert adj["vol_healthy"] == pytest.approx(round(543.0 * 0.41, 1))

    def test_healthy_swap_reaches_the_activating_list_not_just_roster(self):
        # Regression: an IL-slot returnee enters _build_pool via the passed
        # activating_il list, so the healthy swap must reach it too. If it only
        # reached the roster copy, if_healthy would equal as_projected exactly.
        roster = self._roster_with_il_hitter()
        cruz = next(p for p in roster if p.name == "Cruz")
        res = self._call(roster, [cruz])
        assert res.if_healthy is not None
        assert res.if_healthy.to_dict() != res.as_projected.to_dict()

    def test_as_projected_reproduces_plan_il_returns_exactly(self):
        roster = self._roster_with_il_hitter()
        cruz = next(p for p in roster if p.name == "Cruz")
        res = self._call(roster, [cruz])
        direct = plan_il_returns(
            roster,
            [cruz],
            SMALL_SLOTS,
            projected_standings=_standings(),
            team_name=TEAM_NAME,
            fraction_remaining=0.41,
            team_sds=None,
        )
        assert res.as_projected.to_dict() == direct.to_dict()

    def test_no_adjustment_yields_null_if_healthy(self):
        # Returnee without preseason -> no healthy scenario, single-list fallback.
        roster = self._roster_with_il_hitter()
        cruz = next(p for p in roster if p.name == "Cruz")
        cruz.preseason = None
        res = self._call(roster, [cruz])
        assert res.if_healthy is None
        assert res.adjusted == []
        assert res.tops_differ is False

    def test_tops_differ_compares_top_drop_sets_order_independent(self):
        def _result(drop_name, drop_type="hitter"):
            return IlReturnPlanResult(
                activating=["Cruz"],
                capacity=13,
                overflow=1,
                plans=[
                    MovePlan(
                        drops=[drop_name],
                        moves=[Move(drop_name, drop_type, "IL", "DROP")],
                        delta_roto=0.1,
                        band={},
                    )
                ],
            )

        a = _result("Cruz")
        b = _result("Scrub")
        same = _result("Cruz")
        empty = IlReturnPlanResult(activating=["Cruz"], capacity=13, overflow=1, plans=[])
        assert _tops_differ(a, b) is True
        assert _tops_differ(a, same) is False  # same dropped body
        assert _tops_differ(a, empty) is False  # no top plan to compare

    def test_tops_differ_distinguishes_two_way_rows_sharing_a_name(self):
        # A two-way player's hitter and pitcher rows share a display name; the
        # comparison keys on (name, player_type) so dropping different rows is
        # NOT read as "same top plan". A bare-name compare would collapse it.
        def _drop(player_type):
            return IlReturnPlanResult(
                activating=["Ohtani"],
                capacity=13,
                overflow=1,
                plans=[
                    MovePlan(
                        drops=["Ohtani"],
                        moves=[Move("Ohtani", player_type, "IL", "DROP")],
                        delta_roto=0.1,
                        band={},
                    )
                ],
            )

        assert _tops_differ(_drop("hitter"), _drop("pitcher")) is True
