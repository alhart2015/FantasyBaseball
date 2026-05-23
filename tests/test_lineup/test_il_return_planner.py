from fantasy_baseball.lineup.il_return_planner import (
    IlReturnPlanResult,
    Move,
    MovePlan,
    _activate,
    _build_moves,
    _build_pool,
    _counts_against_cap,
    _solve_lineup,
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


class TestSolveLineup:
    def test_solver_returns_active_and_bench(self):
        hitters = [_hitter("OF1", ["OF"])]
        pitchers = [
            _good_pitcher("Ace", k=220),
            _good_pitcher("Mid", k=150, era=3.5, whip=1.2),
            _good_pitcher("Low", k=120, era=4.0, whip=1.3),
        ]
        slots = {"OF": 1, "P": 1, "BN": 1, "IL": 0}
        h_assign, ps, pb = _solve_lineup(hitters + pitchers, slots, _standings(), TEAM_NAME, None)
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
            dropped_names={"Scrub"},
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
