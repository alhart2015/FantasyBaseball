import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))

import keeper_trades as script

from fantasy_baseball.analysis.keeper_trades import RosterPlayer


def test_to_roster_players_attaches_keeper_value_and_zero_for_unmatched():
    from fantasy_baseball.models.player import Player, PlayerType
    from fantasy_baseball.sgp.rankings import fg_key, rank_key

    players = [
        Player(name="Juan Soto", player_type=PlayerType.HITTER),  # name match
        Player(name="Two Names", player_type=PlayerType.HITTER, fg_id="999"),  # fg_id match
        Player(name="Nobody Here", player_type=PlayerType.HITTER),  # unmatched
    ]
    keeper_by_key = {
        rank_key("Juan Soto", PlayerType.HITTER): 18.0,
        fg_key("999", PlayerType.HITTER): 12.0,  # fg-based board id
        rank_key("Two Names", PlayerType.HITTER): 1.0,  # decoy: fg_id must win
    }
    out = script.to_roster_players(players, keeper_by_key)
    kv = {p.name: p.keeper_value for p in out}
    assert kv["Juan Soto"] == 18.0
    assert kv["Two Names"] == 12.0  # fg_id-primary beats the name entry
    assert kv["Nobody Here"] == 0.0  # unmatched -> 0.0, never dropped
    assert all(isinstance(p, RosterPlayer) for p in out)


def test_ros_refills_and_drops_picks_top_and_bottom(monkeypatch):
    from fantasy_baseball.models.player import Player, PlayerType

    def pl(name):
        return Player(name=name, player_type=PlayerType.HITTER)

    vals = {"fa_hi": 9.0, "fa_mid": 5.0, "fa_lo": 1.0, "opp_a": 8.0, "opp_b": 2.0, "opp_c": 3.0}
    monkeypatch.setattr(script, "_ros_value", lambda p, denoms: vals[p.name])
    waiver = {pl(n).player_key: pl(n) for n in ("fa_hi", "fa_mid", "fa_lo")}
    opp = [pl("opp_a"), pl("opp_b"), pl("opp_c")]
    adds, drops = script._ros_refills_and_drops(waiver, opp, denoms=None, n=1)
    assert adds == ["fa_hi::hitter"]  # top-1 refill by ROS value
    assert drops == ["opp_b::hitter"]  # bottom-1 drop (2.0 is lowest)
