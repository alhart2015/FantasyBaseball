from fantasy_baseball.analysis import keeper_trades as kt
from fantasy_baseball.models.player import Player, PlayerType
from fantasy_baseball.models.positions import Position


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


def _league():
    hart = [rp("soto", 18), rp("jrod", 16), rp("cam", 14), rp("woo", 12), rp("wood", 9)]
    spacemen = [rp("judge", 17), rp("g", 8), rp("t", 7)]
    return {"Hart": hart, "Spacemen": spacemen}


def _pass_all(give, receive):
    return kt.GuardrailResult(legal=True, delta_total=-1.0, ok=True)


def test_consolidation_found_both_trios_improve():
    out = kt.generate_consolidation_trades("Hart", _league(), _pass_all, sweetener=False)
    assert out, "expected a suggestion"
    s = next(x for x in out if x.acquire.name == "judge")
    assert s.my_top3_after > s.my_top3_before
    assert s.their_top3_after > s.their_top3_before
    assert s.my_gain == 17 - 14


def test_displaced_keeper_is_free_gain_is_fixed():
    out = kt.generate_consolidation_trades("Hart", _league(), _pass_all, sweetener=False)
    s = next(x for x in out if x.acquire.name == "judge")
    assert s.my_gain == 3.0


def test_guardrail_skips_first_package_takes_next():
    seen = []

    def gr(give, receive):
        seen.append(tuple(p.name for p in give))
        ok = len(give) >= 2 and any(p.name == "wood" for p in give)
        return kt.GuardrailResult(legal=True, delta_total=-1.0, ok=ok)

    out = kt.generate_consolidation_trades("Hart", _league(), gr, sweetener=False)
    s = next(x for x in out if x.acquire.name == "judge")
    assert any(p.name == "wood" for p in s.give)
    assert len(seen) >= 2


def test_no_target_when_no_stud_above_my_third():
    league = {
        "Hart": [rp("soto", 18), rp("jrod", 16), rp("cam", 14)],
        "Weak": [rp("x", 10), rp("y", 5), rp("z", 3)],
    }
    assert kt.generate_consolidation_trades("Hart", league, _pass_all) == []


def test_suggestions_sorted_by_my_gain_desc():
    league = {
        "Hart": [rp("soto", 18), rp("jrod", 16), rp("cam", 14), rp("woo", 12), rp("wood", 9)],
        "A": [rp("big", 20), rp("a1", 3), rp("a2", 2)],
        "B": [rp("mid", 15), rp("b1", 3), rp("b2", 2)],
    }
    out = kt.generate_consolidation_trades("Hart", league, _pass_all, sweetener=False)
    gains = [s.my_gain for s in out]
    assert gains == sorted(gains, reverse=True)
    assert out[0].acquire.name == "big"


def _pl(name, pos):
    return Player(name=name, player_type=PlayerType.HITTER, selected_position=pos)


def test_build_consolidation_proposal_balances_and_sets_active():
    # Hart active: soto(OF), jrod(OF), cam(3B); bench: woo(BN)
    hart = [
        _pl("soto", Position.OF),
        _pl("jrod", Position.OF),
        _pl("cam", Position.THIRD_BASE),
        _pl("woo", Position.BN),
    ]
    prop = kt.build_consolidation_proposal(
        opponent="Spacemen",
        hart_players=hart,
        package_keys=["cam::hitter", "woo::hitter"],  # send 2
        receive_key="judge::hitter",  # get 1
        my_adds_keys=["fa1::hitter"],  # refill N-1 = 1
        opp_drop_keys=["scrub::hitter"],  # opp drops N-1 = 1
    )
    assert prop.send == ["cam::hitter", "woo::hitter"]
    assert prop.receive == ["judge::hitter"]
    assert prop.my_adds == ["fa1::hitter"]
    assert prop.opp_drops == ["scrub::hitter"]
    # cam was active and is sent -> leaves active; judge + fa1 enter; soto/jrod stay
    assert prop.my_active_ids == {"soto::hitter", "jrod::hitter", "judge::hitter", "fa1::hitter"}
    assert prop.opp_active_ids == set()  # empty -> evaluator opp fallback
    assert prop.my_active_ids  # regression: NEVER empty
