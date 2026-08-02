import dataclasses
import math

import numpy as np
import pytest

from fantasy_baseball.analysis.draft_value import ParCurve
from fantasy_baseball.analysis.trade_pick import (
    NextYearValue,
    build_replacement_filler,
    build_trade_scenario,
    find_sent_player,
    pick_ordinal_range,
    pick_value,
    run_scenario,
    worst_of_type,
)
from fantasy_baseball.mc_roster import build_effective_rosters
from fantasy_baseball.models.player import HitterStats, Player, PlayerType
from fantasy_baseball.models.positions import Position
from fantasy_baseball.models.standings import CategoryStats
from fantasy_baseball.scoring import _classify_roster, build_team_sds
from fantasy_baseball.simulation import simulate_remaining_season_batch


def _reslot(inputs, team, name, slot):
    """Copy of ``inputs`` with one named player moved to ``slot``.

    The shared ``_synth_inputs`` fixture slots every player into a real starting
    spot, so it has no bench and no IL -- exactly the two buckets the slot-
    transplant bugs live in. These tests carve out the bucket they need.
    """
    roster = [
        dataclasses.replace(p, selected_position=slot) if p.name == name else p
        for p in inputs.team_rosters[team]
    ]
    rosters = dict(inputs.team_rosters)
    rosters[team] = roster
    return dataclasses.replace(inputs, team_rosters=rosters)


def _buckets(roster):
    """(active, il, bench) counts -- what the MC actually simulates."""
    active, il, bench = _classify_roster(roster)
    return len(active), len(il), len(bench)


def _moved(roster, player):
    """True if ``player`` himself is on ``roster``, re-slotted but not replaced.

    A traded body is re-slotted into the spot it fills, and ``dataclasses.replace``
    returns a COPY -- so ``is`` no longer finds him. Identity of the projection
    objects is the sharper check anyway: it distinguishes the real player from a
    synthetic filler wearing a similar name, which plain ``==`` would not.
    """
    return any(
        p.name == player.name
        and p.rest_of_season is player.rest_of_season
        and p.full_season_projection is player.full_season_projection
        for p in roster
    )


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
    assert _moved(scen[partner], sent)
    # inputs.team_rosters is not mutated
    assert any(p.name == "Star" for p in inputs.team_rosters[user])


def test_this_year_impact_star_to_rival_does_not_help_you():
    # package-qualified sibling import (tests/test_analysis has __init__.py)
    from fantasy_baseball.analysis.trade_pick import find_sent_player, this_year_impact
    from tests.test_analysis.test_injury_stress import _synth_inputs

    inputs = _synth_inputs()
    sent = find_sent_player(inputs.team_rosters[inputs.user_team_name], "Star")
    # small n_iter for speed; common random numbers keep the delta meaningful
    impact = this_year_impact(inputs, sent, "Opp", n_iter=300, seed=42)

    # trading your star to the only rival must not raise your win% by more than a
    # 1.0 pt fixed-seed tolerance band (expected direction is a decrease).
    assert impact.new_win <= impact.base_win + 1.0
    # all 10 categories reported
    assert len(impact.categories) == 10
    assert {c.category for c in impact.categories} == {
        "R",
        "HR",
        "RBI",
        "SB",
        "AVG",
        "W",
        "K",
        "SV",
        "ERA",
        "WHIP",
    }
    assert impact.n_iter == 300 and impact.seed == 42


def test_resolve_partner_normalizes_and_rejects_self_and_unknown():
    from fantasy_baseball.analysis.trade_pick import resolve_partner

    rosters = {"Hart of the Order": [], "SkeleThor": []}
    assert resolve_partner(rosters, "skelethor", "Hart of the Order") == "SkeleThor"
    with pytest.raises(ValueError, match="yourself"):
        resolve_partner(rosters, "Hart of the Order", "Hart of the Order")
    with pytest.raises(ValueError, match="not a team"):
        resolve_partner(rosters, "Nonexistent", "Hart of the Order")


def test_render_report_is_ascii_and_sign_aware():
    from fantasy_baseball.analysis.trade_pick import (
        CategoryDelta,
        NextYearValue,
        ThisYearImpact,
        TradePickResult,
        render_report,
    )

    cats = [
        CategoryDelta(c, 30.0, 28.0, 80.0, 78.0)
        for c in ("R", "HR", "RBI", "SB", "AVG", "W", "K", "SV", "ERA", "WHIP")
    ]
    loss = TradePickResult(
        sent_name="Julio Rodriguez",
        partner="SkeleThor",
        this_year=ThisYearImpact(62.1, 57.9, 91.0, 88.2, cats, 2000, 42),
        next_year=NextYearValue(5, 3, 2, "round", 4.2, 5.1, 18.4, 11, 20),
    )
    out = render_report(loss)
    assert out.isascii()
    assert "Julio Rodriguez" in out and "SkeleThor" in out
    assert "give up" in out  # loss framing (base_win > new_win)

    gain = TradePickResult(
        sent_name="Spare Part",
        partner="SkeleThor",
        this_year=ThisYearImpact(50.0, 50.5, 80.0, 80.0, cats, 2000, 42),
        next_year=NextYearValue(5, 3, 2, "round", 4.2, 5.1, 18.4, 11, 20),
    )
    assert "roughly neutral" in render_report(gain)  # non-negative delta framing


def test_cli_help_runs():
    import subprocess
    import sys
    from pathlib import Path

    root = Path(__file__).resolve().parents[2]
    script = root / "scripts" / "trade_pick_calc.py"
    result = subprocess.run(
        [sys.executable, str(script), "--help"],
        capture_output=True,
        text=True,
        cwd=str(root),
    )
    assert result.returncode == 0
    assert "--send" in result.stdout and "--pick-round" in result.stdout


def test_build_trade_scenario_two_way_swaps_the_real_players():
    """A player-for-player trade must move BOTH players, not substitute a filler.

    With `received` supplied there is no vacated slot to fill: each side gives one
    and gets one, so sizes hold without inventing a replacement, and neither roster
    should end up holding a synthetic body.
    """
    from tests.test_analysis.test_injury_stress import _synth_inputs

    inputs = _synth_inputs()
    user, partner = inputs.user_team_name, "Opp"
    sent = find_sent_player(inputs.team_rosters[user], "Star")
    received = inputs.team_rosters[partner][0]
    n_user0, n_partner0 = len(inputs.team_rosters[user]), len(inputs.team_rosters[partner])

    scen = build_trade_scenario(inputs, sent, partner, received=received)

    assert len(scen[user]) == n_user0
    assert len(scen[partner]) == n_partner0
    assert _moved(scen[user], received)
    assert not _moved(scen[user], sent)
    assert _moved(scen[partner], sent)
    assert not _moved(scen[partner], received)
    # No filler on either side -- a two-way trade fills its own hole.
    assert not any(p.name.startswith("Replacement") for p in scen[user])
    assert not any(p.name.startswith("Replacement") for p in scen[partner])
    # And nothing was dropped from the partner: the incoming player IS the slot filler.
    assert {p.name for p in inputs.team_rosters[partner]} - {p.name for p in scen[partner]} == {
        received.name
    }


def test_build_trade_scenario_two_way_does_not_mutate_inputs():
    from tests.test_analysis.test_injury_stress import _synth_inputs

    inputs = _synth_inputs()
    user, partner = inputs.user_team_name, "Opp"
    before_user = list(inputs.team_rosters[user])
    before_partner = list(inputs.team_rosters[partner])
    sent = find_sent_player(inputs.team_rosters[user], "Star")
    build_trade_scenario(inputs, sent, partner, received=inputs.team_rosters[partner][0])
    assert inputs.team_rosters[user] == before_user
    assert inputs.team_rosters[partner] == before_partner


def test_build_trade_scenario_rejects_a_received_player_not_on_the_partner():
    """Otherwise the partner silently grows by one and the comparison is unbalanced."""
    from tests.test_analysis.test_injury_stress import _synth_inputs

    inputs = _synth_inputs()
    user, partner = inputs.user_team_name, "Opp"
    sent = find_sent_player(inputs.team_rosters[user], "Star")
    stranger = _hit("Not Theirs")
    with pytest.raises(ValueError, match="not on"):
        build_trade_scenario(inputs, sent, partner, received=stranger)


# ---------------------------------------------------------------------------
# Slot transplant: the incoming body inherits the spot the outgoing body left.
#
# `scoring._classify_roster` reads a body's bucket (active / il / bench) off
# `selected_position`, and the MC only simulates active + IL bodies (healthy
# bench pitchers are dropped outright). So a trade that moves Players between
# rosters without reassigning slots silently moves them between BUCKETS, which
# invents or deletes lineup production the trade never touched.
# ---------------------------------------------------------------------------


def test_filler_inherits_a_benched_sent_players_slot():
    """Trading a BENCH player must not hand the user a free active body.

    Regression: the filler was built with ``selected_position=None`` and
    ``_classify_roster`` is slot-first with no None branch, so it landed in
    ``active`` no matter what the sent player's slot was.
    """
    from tests.test_analysis.test_injury_stress import _synth_inputs

    inputs = _reslot(_synth_inputs(), "Me", "Weak", Position.BN)
    weak = find_sent_player(inputs.team_rosters["Me"], "Weak")
    before = _buckets(inputs.team_rosters["Me"])

    scen = build_trade_scenario(inputs, weak, "Opp")

    assert _buckets(scen["Me"]) == before
    filler = next(p for p in scen["Me"] if p.name.startswith("Replacement"))
    assert filler.selected_position == Position.BN


def test_two_way_swap_preserves_both_sides_bucket_counts():
    """An even swap must not move either side's active/bench/IL structure.

    Regression: the received player was appended carrying the PARTNER's slot, so
    receiving a bench-slotted arm for an active one benched him on the user's
    roster -- and healthy bench pitchers are dropped from the MC entirely, so the
    acquired arm threw zero innings.
    """
    from tests.test_analysis.test_injury_stress import _synth_inputs

    inputs = _reslot(_synth_inputs(), "Opp", "Q8", Position.BN)
    sent = find_sent_player(inputs.team_rosters["Me"], "P7")
    received = find_sent_player(inputs.team_rosters["Opp"], "Q8")
    before_me = _buckets(inputs.team_rosters["Me"])
    before_opp = _buckets(inputs.team_rosters["Opp"])

    scen = build_trade_scenario(inputs, sent, "Opp", received=received)

    assert _buckets(scen["Me"]) == before_me
    assert _buckets(scen["Opp"]) == before_opp
    got = next(p for p in scen["Me"] if p.name == "Q8")
    assert got.selected_position == sent.selected_position
    gone = next(p for p in scen["Opp"] if p.name == "P7")
    assert gone.selected_position == received.selected_position


def test_a_healthy_incoming_body_never_inherits_an_il_slot():
    """Slot travels with the roster spot; health travels with the player."""
    from tests.test_analysis.test_injury_stress import _synth_inputs

    inputs = _reslot(_synth_inputs(), "Me", "Star", Position.IL)
    star = find_sent_player(inputs.team_rosters["Me"], "Star")

    scen = build_trade_scenario(inputs, star, "Opp")

    filler = next(p for p in scen["Me"] if p.name.startswith("Replacement"))
    assert not filler.is_on_il()
    assert filler.selected_position == Position.BN


def test_an_injured_acquisition_is_not_promoted_into_an_active_slot():
    """The reverse guard: inheriting an ACTIVE slot must not heal a hurt player."""
    from tests.test_analysis.test_injury_stress import _synth_inputs

    inputs = _reslot(_synth_inputs(), "Opp", "Q8", Position.IL)
    sent = find_sent_player(inputs.team_rosters["Me"], "P7")
    received = find_sent_player(inputs.team_rosters["Opp"], "Q8")
    assert sent.selected_position not in (Position.BN, Position.IL)  # an active slot

    scen = build_trade_scenario(inputs, sent, "Opp", received=received)

    got = next(p for p in scen["Me"] if p.name == "Q8")
    assert got.is_on_il()
    _active, il, _bench = _classify_roster(scen["Me"])
    assert any(p.name == "Q8" for p in il)


def test_replacement_filler_is_healthy_even_when_the_sent_player_is_not():
    """A waiver body is healthy. ``dataclasses.replace`` carried ``status`` over,
    so the filler for an IL'd star inherited his IL status and was simulated as
    injured -- production the user would actually have."""
    hurt = dataclasses.replace(
        _hit("Julio Rodriguez"), status="IL10", selected_position=Position.IL
    )
    filler = build_replacement_filler(hurt)
    assert filler.status == ""
    assert not filler.is_on_il()


def test_partner_with_nothing_to_drop_benches_the_incoming_player():
    """The partner grows by one (documented, accepted) -- but the extra body must
    not also be a free ACTIVE body carrying the slot it held on the user's team."""
    from tests.test_analysis.test_injury_stress import _synth_inputs

    inputs = _synth_inputs()
    sent = find_sent_player(inputs.team_rosters["Me"], "Star")
    # Strip the partner of every hitter so worst_of_type() finds no drop candidate.
    rosters = dict(inputs.team_rosters)
    rosters["Opp"] = [p for p in rosters["Opp"] if p.player_type != PlayerType.HITTER]
    inputs = dataclasses.replace(inputs, team_rosters=rosters)
    before_active = _buckets(inputs.team_rosters["Opp"])[0]

    scen = build_trade_scenario(inputs, sent, "Opp")

    assert len(scen["Opp"]) == len(inputs.team_rosters["Opp"]) + 1
    assert _buckets(scen["Opp"])[0] == before_active  # no free active body
    joined = next(p for p in scen["Opp"] if p.name == "Star")
    assert joined.selected_position == Position.BN


# ---------------------------------------------------------------------------
# Common random numbers.
#
# The MC draws each team's per-player randomness as one block indexed by LIST
# POSITION, so removing a player from the middle of a roster and appending the
# new body at the end re-rolls every player after him. The report prints
# "common random numbers across both runs"; that claim is only true if the
# scenario roster is the baseline roster with one entry substituted IN PLACE.
# ---------------------------------------------------------------------------


def test_build_trade_scenario_substitutes_in_place():
    from tests.test_analysis.test_injury_stress import _synth_inputs

    inputs = _synth_inputs()
    user, partner = inputs.user_team_name, "Opp"
    before_user = inputs.team_rosters[user]
    before_partner = inputs.team_rosters[partner]
    sent = find_sent_player(before_user, "Star")
    drop = worst_of_type(before_partner, sent.player_type, inputs.denoms)
    i = before_user.index(sent)
    j = before_partner.index(drop)

    scen = build_trade_scenario(inputs, sent, partner)

    assert scen[user][i].name.startswith("Replacement")
    assert [p.name for p in scen[user]][:i] == [p.name for p in before_user][:i]
    assert [p.name for p in scen[user]][i + 1 :] == [p.name for p in before_user][i + 1 :]
    assert scen[partner][j].name == "Star"
    assert [p.name for p in scen[partner]][:j] == [p.name for p in before_partner][:j]
    assert [p.name for p in scen[partner]][j + 1 :] == [p.name for p in before_partner][j + 1 :]


def test_build_trade_scenario_two_way_substitutes_in_place():
    from tests.test_analysis.test_injury_stress import _synth_inputs

    inputs = _synth_inputs()
    user, partner = inputs.user_team_name, "Opp"
    sent = find_sent_player(inputs.team_rosters[user], "Star")
    received = inputs.team_rosters[partner][3]
    i = inputs.team_rosters[user].index(sent)
    j = inputs.team_rosters[partner].index(received)

    scen = build_trade_scenario(inputs, sent, partner, received=received)

    assert scen[user][i].name == received.name
    assert scen[partner][j].name == sent.name
    assert len(scen[user]) == len(inputs.team_rosters[user])
    assert len(scen[partner]) == len(inputs.team_rosters[partner])


def test_a_materially_null_trade_moves_the_win_pct_by_exactly_zero():
    """The sharp end of the CRN claim.

    Swap a player for a stat-for-stat clone of himself: both rosters are
    materially identical afterwards, so a genuinely paired comparison must return
    a bit-identical win%. Any nonzero delta here is pure RNG misalignment, and it
    is the same misalignment that contaminates every real delta this tool prints.
    """
    from tests.test_analysis.test_injury_stress import _synth_inputs

    inputs = _synth_inputs()
    user, partner = inputs.user_team_name, "Opp"
    sent = find_sent_player(inputs.team_rosters[user], "Star")
    clone = dataclasses.replace(sent, name="Star Clone", yahoo_id="999")
    # Park the clone on the partner in place of a body with the same slot, so the
    # ONLY difference between the two runs is which of two identical lines sits
    # in each roster spot.
    victim = inputs.team_rosters[partner][0]
    assert victim.selected_position == sent.selected_position
    rosters = dict(inputs.team_rosters)
    rosters[partner] = [clone if p is victim else p for p in rosters[partner]]
    inputs = dataclasses.replace(inputs, team_rosters=rosters)

    base = run_scenario(inputs, inputs.team_rosters, 300, 42)
    scen_rosters = build_trade_scenario(inputs, sent, partner, received=clone)
    scen = run_scenario(inputs, scen_rosters, 300, 42)

    assert scen["team_results"][user]["first_pct"] == base["team_results"][user]["first_pct"]
    assert scen["team_results"][partner]["first_pct"] == base["team_results"][partner]["first_pct"]


@pytest.mark.parametrize("two_way", [False, True])
def test_scenario_rosters_consume_the_rng_identically(two_way):
    """The structural invariant behind the report's common-random-numbers claim.

    ``simulate_remaining_season_batch`` walks ``team_rosters`` in dict order and
    draws one ``rng.random((n_iter, n_players))`` block per team per player type,
    so the baseline and scenario runs stay on the same stream only if EVERY team
    keeps its dict position, its length, and its hitter/pitcher split -- and stay
    aligned player-for-player only if the bucket counts hold too (the ROS-direct
    engine sizes its draw off the active + IL bodies). Assert all of it.

    Not asserted, because it is false and should be: that an untouched team's
    ``first_pct`` is unchanged. That is a RANKING against the other teams, and
    two rivals trading really does move everyone else's placement.
    """
    from tests.test_analysis.test_injury_stress import _mk_hitter, _mk_pitcher, _synth_inputs

    base_inputs = _synth_inputs()
    third = [_mk_hitter(f"T{i}", str(90 + i)) for i in range(13)] + [
        _mk_pitcher(f"S{i}", str(110 + i)) for i in range(9)
    ]
    rosters = {**base_inputs.team_rosters, "Third": third}
    inputs = dataclasses.replace(
        base_inputs,
        team_rosters=rosters,
        actual_standings={t: {} for t in rosters},
        eos_baseline={t: CategoryStats() for t in rosters},
        team_sds=build_team_sds(rosters, math.sqrt(base_inputs.fraction_remaining)),
    )
    sent = find_sent_player(inputs.team_rosters["Me"], "Star")
    received = find_sent_player(inputs.team_rosters["Opp"], "O4") if two_way else None

    scen = build_trade_scenario(inputs, sent, "Opp", received=received)

    assert list(scen) == list(inputs.team_rosters)  # dict order drives draw order
    for team, before in inputs.team_rosters.items():
        after = scen[team]
        assert len(after) == len(before), team
        assert [p.player_type for p in after] == [p.player_type for p in before], team
        assert _buckets(after) == _buckets(before), team


def test_untouched_team_keeps_its_own_simulated_production():
    """The other half of the claim: a third team's raw draws are untouched.

    Its placement moves (two rivals traded), but the randomness it is handed must
    not, so its own category totals have to come out bit-identical.
    """
    from tests.test_analysis.test_injury_stress import _mk_hitter, _mk_pitcher, _synth_inputs

    base_inputs = _synth_inputs()
    third = [_mk_hitter(f"T{i}", str(90 + i)) for i in range(13)] + [
        _mk_pitcher(f"S{i}", str(110 + i)) for i in range(9)
    ]
    rosters = {**base_inputs.team_rosters, "Third": third}
    inputs = dataclasses.replace(
        base_inputs,
        team_rosters=rosters,
        actual_standings={t: {} for t in rosters},
        eos_baseline={t: CategoryStats() for t in rosters},
        team_sds=build_team_sds(rosters, math.sqrt(base_inputs.fraction_remaining)),
    )
    sent = find_sent_player(inputs.team_rosters["Me"], "Star")

    def totals(team_rosters):
        eff = build_effective_rosters(
            team_rosters,
            inputs.eos_baseline,
            inputs.team_sds,
            inputs.fraction_remaining,
            denoms=inputs.denoms,
        )
        return simulate_remaining_season_batch(
            inputs.actual_standings,
            {t: [p.to_flat_dict() for p in r] for t, r in team_rosters.items()},
            inputs.fraction_remaining,
            np.random.default_rng(42),
            inputs.h_slots,
            inputs.p_slots,
            120,
            effective_rosters=eff,
        )

    base = totals(inputs.team_rosters)
    scen = totals(build_trade_scenario(inputs, sent, "Opp"))

    for cat, arr in base["Third"].items():
        assert np.array_equal(arr, scen["Third"][cat]), cat
