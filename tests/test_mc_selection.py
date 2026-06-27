from fantasy_baseball.mc_selection import (
    compute_active_slot_cols,
    compute_fixed_topk_cols,
    format_attribution_table,
    run_selection_attribution,
)
from fantasy_baseball.models.player import HitterStats, PitcherStats, Player, PlayerType
from fantasy_baseball.models.positions import Position
from fantasy_baseball.models.standings import CategoryStats
from fantasy_baseball.utils.constants import ALL_CATEGORIES


def _hitter(name, slot, r=80):
    return Player(
        name=name,
        player_type=PlayerType.HITTER,
        positions=[Position.OF],
        selected_position=slot,
        rest_of_season=HitterStats(r=r, hr=20, rbi=70, sb=5, h=150, ab=550),
    )


def _pitcher(name, slot, k=150):
    return Player(
        name=name,
        player_type=PlayerType.PITCHER,
        positions=[Position.P],
        selected_position=slot,
        rest_of_season=PitcherStats(w=10, k=k, ip=180, er=70, bb=50, h_allowed=150),
    )


def test_active_slot_cols_excludes_healthy_bench_and_il():
    players = [
        _hitter("H_active", Position.OF),  # hitter col 0 -> active
        _hitter("H_bench", Position.BN),  # hitter col 1 -> excluded (bench)
        _pitcher("P_active", Position.P),  # pitcher col 0 -> active
        _pitcher("P_il", Position.IL),  # pitcher col 1 -> excluded (IL)
    ]
    cols = compute_active_slot_cols(players)
    assert cols["h"].tolist() == [0]
    assert cols["p"].tolist() == [0]


def test_fixed_topk_cols_picks_highest_mean_stats():
    flat = [
        {"player_type": "hitter", "r": 100, "hr": 30, "rbi": 100, "sb": 10},
        {"player_type": "hitter", "r": 50, "hr": 10, "rbi": 40, "sb": 2},
        {"player_type": "hitter", "r": 80, "hr": 25, "rbi": 80, "sb": 8},
        {"player_type": "pitcher", "w": 12, "k": 200, "sv": 0, "ip": 190},
        {"player_type": "pitcher", "w": 5, "k": 90, "sv": 0, "ip": 70},
    ]
    cols = compute_fixed_topk_cols(flat, h_slots=2, p_slots=1)
    assert sorted(cols["h"].tolist()) == [0, 2]
    assert cols["p"].tolist() == [0]


def _hitter_full(name, slot, r=80, rbi=70, g=150, pa=600):
    """Hitter with explicit games/PA so the new-engine bench-fill path has games."""
    return Player(
        name=name,
        player_type=PlayerType.HITTER,
        positions=[Position.OF],
        selected_position=slot,
        rest_of_season=HitterStats.from_dict(
            {"r": r, "hr": 20, "rbi": rbi, "sb": 5, "h": 150, "ab": 550, "pa": pa, "g": g}
        ),
    )


def _context(teams):
    """eos_baseline (team -> CategoryStats) + team_sds for the new-engine arm."""
    eos_baseline = {t: CategoryStats() for t in teams}
    team_sds = {t: {c: 5.0 for c in ALL_CATEGORIES} for t in teams}
    return eos_baseline, team_sds


def test_run_selection_attribution_three_arms_and_ordering():
    deep = [
        _hitter("Star", Position.OF, r=100),
        _hitter("Reg", Position.OF, r=80),
        _hitter("BenchMasher", Position.BN, r=95),
        _pitcher("Ace", Position.P),
    ]
    rosters = {"Deep": deep}
    actuals = {
        "Deep": {
            "R": 0,
            "HR": 0,
            "RBI": 0,
            "SB": 0,
            "AVG": 0,
            "W": 0,
            "K": 0,
            "SV": 0,
            "ERA": 0,
            "WHIP": 0,
        }
    }
    res = run_selection_attribution(
        rosters, actuals, 1.0, h_slots=2, p_slots=1, n_iter=2000, seed=3
    )
    assert set(res) == {"topk_per_iter", "topk_fixed", "active_slot"}
    assert res["active_slot"]["Deep"]["R"] <= res["topk_per_iter"]["Deep"]["R"]
    table = format_attribution_table(res)
    assert "Deep" in table and "active_slot" in table
    assert "new_engine" not in table  # arm skipped without context -> column absent


def _zero_actuals(teams):
    cats = ["R", "HR", "RBI", "SB", "AVG", "W", "K", "SV", "ERA", "WHIP"]
    return {t: {c: 0 for c in cats} for t in teams}


def test_new_engine_arm_present_and_between_floor_and_ceiling():
    # Two active hitters + a STRONG healthy bench bat (higher RBI than a starter)
    # so top-k seats it (ceiling) while active-slot excludes it (floor). The new
    # engine fixes the active set -> the bench bat contributes only a small
    # injury-fill share, landing the new-engine RBI BETWEEN the two.
    deep = [
        _hitter_full("Star", Position.OF, r=100, rbi=100),
        _hitter_full("Reg", Position.OF, r=70, rbi=40),
        _hitter_full("BenchMasher", Position.BN, r=95, rbi=95),
        _pitcher("Ace", Position.P),
    ]
    rosters = {"Deep": deep}
    actuals = _zero_actuals(["Deep"])
    eos_baseline, team_sds = _context(["Deep"])

    res = run_selection_attribution(
        rosters,
        actuals,
        1.0,
        h_slots=2,
        p_slots=1,
        n_iter=2000,
        seed=3,
        eos_baseline=eos_baseline,
        team_sds=team_sds,
    )

    assert "new_engine" in res
    cats = {"R", "HR", "RBI", "SB", "AVG", "W", "K", "SV", "ERA", "WHIP"}
    assert set(res["new_engine"]["Deep"].keys()) == cats

    floor = res["active_slot"]["Deep"]["RBI"]
    ceiling = res["topk_per_iter"]["Deep"]["RBI"]
    new = res["new_engine"]["Deep"]["RBI"]
    tol = 1.0  # mechanism-level: small tolerance, assert ordering not magnitude
    assert floor - tol <= new <= ceiling + tol, (
        f"new_engine RBI {new} not between active_slot {floor} and topk {ceiling}"
    )


def test_format_attribution_table_includes_new_engine_column():
    deep = [
        _hitter_full("Star", Position.OF, r=100, rbi=100),
        _hitter_full("Reg", Position.OF, r=70, rbi=40),
        _hitter_full("BenchMasher", Position.BN, r=95, rbi=95),
        _pitcher("Ace", Position.P),
    ]
    rosters = {"Deep": deep}
    eos_baseline, team_sds = _context(["Deep"])
    res = run_selection_attribution(
        rosters,
        _zero_actuals(["Deep"]),
        1.0,
        h_slots=2,
        p_slots=1,
        n_iter=500,
        seed=3,
        eos_baseline=eos_baseline,
        team_sds=team_sds,
    )
    table = format_attribution_table(res)
    assert "new_engine" in table
