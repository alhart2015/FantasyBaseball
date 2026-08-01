import pytest

from fantasy_baseball.analysis.draft_value import ParCurve
from fantasy_baseball.analysis.trade_pick import (
    NextYearValue,
    build_replacement_filler,
    build_trade_scenario,
    find_sent_player,
    pick_ordinal_range,
    pick_value,
    worst_of_type,
)
from fantasy_baseball.models.player import HitterStats, Player, PlayerType
from fantasy_baseball.models.positions import Position


def _curve(n=200):
    # Strictly descending so mean-over-a-range is order-sensitive and testable.
    return ParCurve(drafted_pars=[float(n - i) for i in range(n)], keeper_par=18.0)


def test_ordinal_range_round_2_is_11_to_20():
    # nominal 5, 3 keeper rounds, 10 teams -> drafted round 2 -> ordinals 11..20
    assert pick_ordinal_range(5, 3, 10, 200) == (11, 20)


def test_ordinal_range_first_drafted_round():
    assert pick_ordinal_range(4, 3, 10, 200) == (1, 10)


def test_ordinal_range_keeper_round_rejected():
    with pytest.raises(ValueError, match="keeper round"):
        pick_ordinal_range(3, 3, 10, 200)


def test_ordinal_range_beyond_curve_rejected():
    with pytest.raises(ValueError, match="beyond the par curve"):
        pick_ordinal_range(60, 3, 10, 200)  # drafted round 57 -> lo far past 200


def test_ordinal_range_clamps_upper_bound():
    # drafted round 20 -> ordinals 191..200; a 195-long curve clamps hi to 195.
    assert pick_ordinal_range(23, 3, 10, 195) == (191, 195)


def test_ordinal_range_early_mid_late_partition_the_round():
    lo_e, hi_e = pick_ordinal_range(5, 3, 10, 200, "early")
    lo_m, hi_m = pick_ordinal_range(5, 3, 10, 200, "mid")
    lo_l, hi_l = pick_ordinal_range(5, 3, 10, 200, "late")
    assert lo_e == 11 and hi_e < 20  # early starts at the round's top
    assert hi_l == 20 and lo_l > 11  # late ends at the round's bottom
    assert lo_e <= lo_m <= lo_l and hi_e <= hi_m <= hi_l


def test_pick_value_round_average_and_early_higher():
    par = _curve()
    nv = pick_value(par, 5, 3, 10, "round")
    assert isinstance(nv, NextYearValue)
    assert nv.drafted_round == 2
    assert nv.ordinal_lo == 11 and nv.ordinal_hi == 20
    # mean of par_for_slot(11..20) = mean of drafted_pars[10..19] = mean(190..181) = 185.5
    assert nv.expected_var == pytest.approx(185.5)
    # early third (higher VAR) exceeds the full-round average on a descending curve
    assert nv.early_var > nv.expected_var
    assert nv.keeper_par == 18.0


def _hit(name, *, r=90, hr=30, rbi=95, sb=12, h=165, ab=560, pa=620, g=155):
    line = {"r": r, "hr": hr, "rbi": rbi, "sb": sb, "h": h, "ab": ab, "pa": pa, "g": g}
    return Player(
        name=name,
        player_type=PlayerType.HITTER,
        positions=[Position.OF],
        rest_of_season=HitterStats.from_dict(line),
        full_season_projection=HitterStats.from_dict(line),
    )


def test_find_sent_player_normalized_and_ambiguity():
    roster = [_hit("Julio Rodriguez"), _hit("Someone Else")]
    assert find_sent_player(roster, "julio rodriguez").name == "Julio Rodriguez"
    with pytest.raises(ValueError, match="not on"):
        find_sent_player(roster, "Nobody Here")


def test_replacement_filler_is_neutralized_and_renamed():
    star = _hit("Julio Rodriguez")
    filler = build_replacement_filler(star)
    assert filler.name != star.name
    assert filler.name.startswith("Replacement")
    assert filler.positions == star.positions  # can fill the vacated slot
    # Both lines neutralized below the star's real production (r/hr/rbi drop).
    for col in ("r", "hr", "rbi"):
        assert getattr(filler.rest_of_season, col) < getattr(star.rest_of_season, col)
        assert getattr(filler.full_season_projection, col) < getattr(
            star.full_season_projection, col
        )


def test_worst_of_type_picks_lowest_projection():
    from fantasy_baseball.sgp.denominators import get_sgp_denominators

    good = _hit("Good")
    bad = _hit("Bad", r=30, hr=2, rbi=25, sb=1, h=80, ab=400, pa=440, g=110)
    worst = worst_of_type([good, bad], PlayerType.HITTER, get_sgp_denominators(None))
    assert worst.name == "Bad"


def test_build_trade_scenario_keeps_sizes_and_moves_player():
    # tests/test_analysis is a package (has __init__.py), so import the sibling
    # fixture by its package-qualified name, not a bare module name.
    from tests.test_analysis.test_injury_stress import _synth_inputs

    inputs = _synth_inputs()
    user = inputs.user_team_name
    partner = "Opp"
    sent = find_sent_player(inputs.team_rosters[user], "Star")
    n_user0 = len(inputs.team_rosters[user])
    n_partner0 = len(inputs.team_rosters[partner])

    scen = build_trade_scenario(inputs, sent, partner)

    # user size unchanged: lost Star, gained exactly one filler
    assert len(scen[user]) == n_user0
    assert all(p.name != "Star" for p in scen[user])
    assert sum(p.name.startswith("Replacement") for p in scen[user]) == 1
    # partner size unchanged: gained the intact Star, dropped its worst hitter
    assert len(scen[partner]) == n_partner0
    assert any(p is sent for p in scen[partner])
    # inputs.team_rosters is not mutated
    assert any(p.name == "Star" for p in inputs.team_rosters[user])
