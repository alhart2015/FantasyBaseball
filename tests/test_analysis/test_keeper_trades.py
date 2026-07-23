from fantasy_baseball.analysis import keeper_trades as kt


def rp(name, kv):
    return kt.RosterPlayer(player_id=f"{name}::hitter", name=name, keeper_value=kv)


def test_top3_sum_takes_three_highest():
    players = [rp("a", 10), rp("b", 8), rp("c", 6), rp("d", 4)]
    assert kt.top3_sum(players) == 24.0  # 10 + 8 + 6, ignores d


def test_top3_sum_handles_fewer_than_three():
    assert kt.top3_sum([rp("a", 10), rp("b", 8)]) == 18.0
    assert kt.top3_sum([]) == 0.0
