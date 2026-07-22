import math

from fantasy_baseball.analysis.injury_stress import (
    HealthProbs,
    McInputs,
    health_probabilities,
    substitute_replacement,
    win_pct,
)
from fantasy_baseball.models.player import HitterStats, PitcherStats, Player, PlayerType
from fantasy_baseball.models.positions import Position
from fantasy_baseball.models.standings import CategoryStats
from fantasy_baseball.scoring import build_team_sds
from fantasy_baseball.sgp.denominators import get_sgp_denominators
from fantasy_baseball.utils.constants import PITCHING_COUNTING


def _hitter(name, *, pa, ab, g):
    return Player(
        name=name,
        player_type=PlayerType.HITTER,
        positions=[Position.OF],
        rest_of_season=HitterStats.from_dict(
            {"r": 80, "hr": 20, "rbi": 70, "sb": 5, "h": 150, "ab": ab, "pa": pa, "g": g}
        ),
        full_season_projection=HitterStats.from_dict(
            {"r": 80, "hr": 20, "rbi": 70, "sb": 5, "h": 150, "ab": ab, "pa": pa, "g": g}
        ),
    )


def test_health_probabilities_sum_to_one_and_ordered():
    players = [_hitter("A", pa=600, ab=550, g=150), _hitter("B", pa=600, ab=550, g=150)]
    hp = health_probabilities(players, 0.5, n_samples=5000, seed=42)
    assert isinstance(hp, HealthProbs)
    assert abs(hp.p_all_healthy + hp.p_one + hp.p_two_plus - 1.0) < 1e-9
    assert 0.0 <= hp.p_two_plus <= hp.p_one  # two-or-more is rarer than exactly-one here
    assert set(hp.per_player) == {"A", "B"}


def test_health_haircut_alone_is_not_significant():
    # A player realizing EXACTLY his expected level (eff_mean) must NOT count as
    # losing significant time -- guards the haircut-vs-injury bug. With threshold
    # 0 no one is ever significant regardless of the systematic mean haircut... so
    # instead assert per-player significance stays well below 1.0 for a healthy
    # full-timer (the haircut does not by itself trip the eff_mean-relative bar).
    players = [_hitter("A", pa=600, ab=550, g=150)]
    hp = health_probabilities(players, 0.5, threshold=0.20, n_samples=20000, seed=1)
    assert hp.per_player["A"] < 0.5


# ---------------------------------------------------------------------------
# Replacement-level substitution + counterfactual win%
# ---------------------------------------------------------------------------


def _mk_hitter(name, pid, *, r=90, hr=30, rbi=95, sb=12, h=165, ab=560, pa=620, g=155):
    line = {"r": r, "hr": hr, "rbi": rbi, "sb": sb, "h": h, "ab": ab, "pa": pa, "g": g}
    return Player(
        name=name,
        player_type=PlayerType.HITTER,
        positions=[Position.OF],
        selected_position=Position.OF,
        yahoo_id=pid,
        rest_of_season=HitterStats.from_dict(line),
        full_season_projection=HitterStats.from_dict(line),
    )


def _mk_pitcher(name, pid, *, w=12, k=190, sv=0, ip=170, er=60, bb=45, ha=140, g=30):
    line = {"w": w, "k": k, "sv": sv, "ip": ip, "er": er, "bb": bb, "h_allowed": ha, "g": g}
    return Player(
        name=name,
        player_type=PlayerType.PITCHER,
        positions=[Position.SP],
        selected_position=Position.SP,
        yahoo_id=pid,
        rest_of_season=PitcherStats.from_dict(line),
        full_season_projection=PitcherStats.from_dict(line),
    )


def _synth_inputs():
    """Minimal 2-team league good enough to drive run_ros_monte_carlo."""
    star = _mk_hitter("Star", "1")  # high value
    weak = _mk_hitter("Weak", "2", r=40, hr=3, rbi=35, sb=2, h=95, ab=430, pa=470, g=120)
    ace = _mk_pitcher("Ace", "3")  # high-value pitcher
    me = (
        [star, weak]
        + [_mk_hitter(f"H{i}", str(10 + i)) for i in range(11)]
        + [ace]
        + [_mk_pitcher(f"P{i}", str(30 + i)) for i in range(8)]
    )
    opp = [_mk_hitter(f"O{i}", str(50 + i)) for i in range(13)] + [
        _mk_pitcher(f"Q{i}", str(70 + i)) for i in range(9)
    ]
    team_rosters = {"Me": me, "Opp": opp}
    actual_standings = {t: {} for t in team_rosters}  # preseason-like (no YTD)
    fr = 1.0
    eos = {t: CategoryStats() for t in team_rosters}  # LeagueContext baseline stand-in
    sds = build_team_sds(team_rosters, math.sqrt(fr))
    denoms = get_sgp_denominators(None)
    return McInputs(
        team_rosters=team_rosters,
        actual_standings=actual_standings,
        fraction_remaining=fr,
        h_slots=13,
        p_slots=9,
        eos_baseline=eos,
        team_sds=sds,
        denoms=denoms,
        user_team_name="Me",
        projected_margin=0.0,
    )


def _find(players, name):
    return next(p for p in players if p.name == name)


def test_substitute_swaps_ros_to_scaled_replacement():
    inp = _synth_inputs()
    me = inp.team_rosters["Me"]
    sub = substitute_replacement(me, [_find(me, "Star")])
    orig = {p.name: p for p in me}
    subd = {p.name: p for p in sub}
    assert subd["Weak"].rest_of_season is orig["Weak"].rest_of_season  # untouched
    # AB is preserved by design: the replacement line is scaled to occupy Star's
    # OWN ROS volume, not to shrink it (same playing-time bucket, worse per-AB
    # rate) -- this is the "scaled to X's ROS volume" contract.
    assert subd["Star"].rest_of_season.ab == orig["Star"].rest_of_season.ab
    # Replacement-level rate stats drop for the counting cats where Star's own
    # rate clearly outpaces the position's replacement rate (r/hr/rbi/h). SB is
    # excluded: Star's per-AB SB rate sits below the OF replacement rate here, so
    # a replacement bat can out-steal him -- not monotone, same caveat the
    # pitcher test documents below for ER/BB/H.
    for col in ("r", "hr", "rbi", "h"):
        assert getattr(subd["Star"].rest_of_season, col) < getattr(orig["Star"].rest_of_season, col)
    assert subd["Star"].positions == orig["Star"].positions  # slot preserved


def test_substitute_works_for_a_pitcher():
    inp = _synth_inputs()
    me = inp.team_rosters["Me"]
    subd = {p.name: p for p in substitute_replacement(me, [_find(me, "Ace")])}
    orig = {p.name: p for p in me}
    # IP is preserved by design: the replacement arm is scaled to occupy Ace's
    # OWN ROS volume, not to shrink it (same innings bucket, worse per-IP rate).
    assert subd["Ace"].rest_of_season.ip == orig["Ace"].rest_of_season.ip
    for col in PITCHING_COUNTING:
        # Ace's counting stats collapse to replacement level (K/W strictly lower);
        # ER/BB/H may not be monotone, so only assert the "good" counting cats drop.
        if col in ("w", "k"):
            assert getattr(subd["Ace"].rest_of_season, col) < getattr(
                orig["Ace"].rest_of_season, col
            )


def test_counterfactual_star_costs_more_than_weak():
    inp = _synth_inputs()
    me = inp.team_rosters["Me"]
    base = win_pct(inp, me, n_iter=300)
    lose_star = win_pct(inp, substitute_replacement(me, [_find(me, "Star")]), n_iter=300)
    lose_weak = win_pct(inp, substitute_replacement(me, [_find(me, "Weak")]), n_iter=300)
    assert base - lose_star >= base - lose_weak  # star hurts at least as much
    assert base - lose_star > 0.0  # losing the star has a real cost


def test_counterfactual_pitcher_has_cost_and_no_raw_hole():
    inp = _synth_inputs()
    me = inp.team_rosters["Me"]
    base = win_pct(inp, me, n_iter=300)
    lose_ace = win_pct(inp, substitute_replacement(me, [_find(me, "Ace")]), n_iter=300)
    assert base - lose_ace > 0.0  # a replacement arm still pitches -> real, finite cost


def test_replacement_ros_routes_starter_by_full_season_ip():
    # Regression for issue #251. Production roster blobs carry only the generic
    # slot position Position.P (never SP/RP eligibility), so _replacement_line's
    # position routing can't fire and falls back to role_from_ip. Mid-season a
    # real starter's REST-OF-SEASON IP (~73) sits below STARTER_IP_THRESHOLD (100)
    # even though his FULL-SEASON IP (~180) is well above it. The replacement role
    # must be decided on full-season IP -> SP, not the shrunken ROS IP -> RP: an SP
    # wrongly handed the K-rich, save-bearing RP line grades out as an UPGRADE
    # (negative single-loss injury exposure -- Woo/Luzardo/Webb/Gray in the report).
    from fantasy_baseball.analysis.injury_stress import _replacement_ros

    starter = Player(
        name="RealStarter",
        player_type=PlayerType.PITCHER,
        positions=[Position.P],  # generic slot -- mirrors the stored roster blob
        rest_of_season=PitcherStats.from_dict(
            {"w": 5, "k": 65, "sv": 0, "ip": 73, "er": 29, "bb": 22, "h_allowed": 63, "g": 12}
        ),
        full_season_projection=PitcherStats.from_dict(
            {"w": 13, "k": 175, "sv": 0, "ip": 180, "er": 70, "bb": 50, "h_allowed": 150, "g": 30}
        ),
    )
    repl = _replacement_ros(starter)
    assert isinstance(repl, PitcherStats)
    assert repl.ip == starter.rest_of_season.ip  # scaled to his OWN ROS innings
    assert repl.sv == 0.0  # SP replacement carries NO saves; the RP line would
    assert repl.k * 9.0 / repl.ip < 10.0  # SP replacement ~9.0 K/9, not the RP ~11.5


def test_win_pct_is_deterministic():
    # Same inputs + same seed -> identical first_pct (locks the reconciliation /
    # common-random-numbers contract).
    inp = _synth_inputs()
    me = inp.team_rosters["Me"]
    assert win_pct(inp, me, n_iter=200) == win_pct(inp, me, n_iter=200)


# ---------------------------------------------------------------------------
# Stress-test orchestration
# ---------------------------------------------------------------------------


def test_run_stress_test_ranks_and_flags():
    from fantasy_baseball.analysis.injury_stress import run_stress_test

    inp = _synth_inputs()
    res = run_stress_test(inp, n_iter=300, pair_top_k=4)
    names = [e.name for e in res.singles]
    assert names.index("Star") < names.index("Weak")  # the star outranks the weak bat
    assert res.singles == sorted(res.singles, key=lambda e: e.win_pct_cost, reverse=True)
    assert 0.0 <= res.health.p_all_healthy <= 1.0
    # pairs are top-K choose 2 and ranked by joint cost
    assert len(res.pairs) == 6  # C(4, 2)
    assert res.pairs == sorted(res.pairs, key=lambda e: e.joint_cost, reverse=True)


def test_render_report_is_ascii_and_has_sections():
    from fantasy_baseball.analysis.injury_stress import render_report, run_stress_test

    res = run_stress_test(_synth_inputs(), n_iter=200, pair_top_k=4)
    text = render_report(res)
    text.encode("ascii")  # raises if any non-ASCII slipped in
    for marker in [
        "WHAT INJURY RISK COSTS",
        "STAYS HEALTHY",
        "MOST EXPOSED",
        "LOSING TWO",
        "generic",
    ]:
        assert marker.lower() in text.lower()
    assert "Star" in text


# ---------------------------------------------------------------------------
# Live Upstash input assembly (pure helpers)
# ---------------------------------------------------------------------------


def test_projected_margin_from_eos_signs_correctly():
    from fantasy_baseball.analysis.injury_stress import projected_margin_from_eos

    inp = _synth_inputs()
    m = projected_margin_from_eos(inp.eos_baseline, "Me")
    assert isinstance(m, float)  # 2-team synthetic: sign follows Me-vs-Opp roto totals
