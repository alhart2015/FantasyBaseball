from fantasy_baseball.models.player import Player, PlayerType, HitterStats, PitcherStats
from fantasy_baseball.models.positions import Position
from fantasy_baseball.lineup.optimizer import HitterAssignment, PitcherStarter


def _hitter(name="H"):
    return Player(
        name=name, player_type=PlayerType.HITTER, positions=[Position.OF],
        rest_of_season=HitterStats(pa=500, ab=450, h=120, r=70, hr=20, rbi=70, sb=10, avg=0.267),
    )


def _pitcher(name="P"):
    return Player(
        name=name, player_type=PlayerType.PITCHER, positions=[Position.SP],
        rest_of_season=PitcherStats(ip=180, w=12, k=180, sv=0, era=3.50, whip=1.20,
                                     er=70, bb=55, h_allowed=160),
    )


class TestHitterAssignment:
    def test_constructs_and_exposes_fields(self):
        p = _hitter("Judge")
        a = HitterAssignment(slot=Position.OF, name="Judge", player=p, roto_delta=1.5)
        assert a.slot is Position.OF
        assert a.name == "Judge"
        assert a.player is p
        assert a.roto_delta == 1.5

    def test_to_dict_rounds_roto_delta(self):
        a = HitterAssignment(slot=Position.OF, name="Judge", player=_hitter(), roto_delta=1.2345)
        assert a.to_dict() == {"slot": "OF", "name": "Judge", "roto_delta": 1.23}


class TestPitcherStarter:
    def test_constructs_and_exposes_fields(self):
        p = _pitcher("Skubal")
        s = PitcherStarter(name="Skubal", player=p, roto_delta=0.8)
        assert s.name == "Skubal"
        assert s.player is p
        assert s.roto_delta == 0.8

    def test_to_dict_rounds_roto_delta(self):
        s = PitcherStarter(name="Skubal", player=_pitcher(), roto_delta=0.5678)
        assert s.to_dict() == {"name": "Skubal", "roto_delta": 0.57}
