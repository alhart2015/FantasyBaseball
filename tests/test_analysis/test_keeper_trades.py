from fantasy_baseball.analysis import keeper_trades as kt


def rp(name, kv):
    return kt.RosterPlayer(player_id=f"{name}::hitter", name=name, keeper_value=kv)


def test_top3_sum_takes_three_highest():
    players = [rp("a", 10), rp("b", 8), rp("c", 6), rp("d", 4)]
    assert kt.top3_sum(players) == 24.0  # 10 + 8 + 6, ignores d


def test_top3_sum_handles_fewer_than_three():
    assert kt.top3_sum([rp("a", 10), rp("b", 8)]) == 18.0
    assert kt.top3_sum([]) == 0.0


def test_keeper_viable_packages_ordered_and_improving():
    # opp: stud G(16) + scrubs s1(3), s2(2)  -> top3_before = 21
    G = rp("G", 16)
    opp = [G, rp("s1", 3), rp("s2", 2)]
    # giveable: d3(14), sur(12), low(1)
    giveable = [rp("d3", 14), rp("sur", 12), rp("low", 1)]
    pkgs = list(kt.keeper_viable_packages(G, opp, giveable, kt.top3_sum(opp), max_give=3))
    assert pkgs, "expected at least one viable package"
    assert all(kt.top3_sum([p for p in opp if p is not G] + list(pkg)) > 21 for pkg in pkgs)
    # ordered fewest-players-first
    sizes = [len(pkg) for pkg in pkgs]
    assert sizes == sorted(sizes)
    # among equal-size viable packages, least total keeper_value given comes first
    two_player = [pkg for pkg in pkgs if len(pkg) == 2]
    assert set(two_player[0]) == {giveable[0], giveable[1]}
