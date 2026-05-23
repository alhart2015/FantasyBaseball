from fantasy_baseball.lineup.il_return_planner import (
    IlReturnPlanResult,
    Move,
    MovePlan,
    _counts_against_cap,
    roster_capacity,
)
from fantasy_baseball.models.player import PitcherStats, Player, PlayerType
from fantasy_baseball.models.positions import Position

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
