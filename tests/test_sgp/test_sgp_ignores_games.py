from fantasy_baseball.models.player import HitterStats, PitcherStats
from fantasy_baseball.sgp.player_value import calculate_player_sgp


def test_sgp_unaffected_by_games_fields():
    bh = {"r": 80, "hr": 25, "rbi": 80, "sb": 10, "h": 150, "ab": 550}
    assert calculate_player_sgp(HitterStats.from_dict(bh)) == calculate_player_sgp(
        HitterStats.from_dict({**bh, "g": 150})
    )
    bp = {"w": 10, "k": 180, "ip": 190, "er": 70, "bb": 50, "h_allowed": 160}
    assert calculate_player_sgp(PitcherStats.from_dict(bp)) == calculate_player_sgp(
        PitcherStats.from_dict({**bp, "g": 32, "gs": 32})
    )
