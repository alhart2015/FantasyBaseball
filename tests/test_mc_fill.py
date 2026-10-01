from fantasy_baseball.mc_fill import (
    ActiveSample,
    BenchSample,
    allocate_bench_fill,
)
from fantasy_baseball.mc_roster import ActiveBody, BenchBody
from fantasy_baseball.models.player import Player, PlayerType
from fantasy_baseball.models.positions import Position
from fantasy_baseball.utils.constants import HITTING_COUNTING


def _player(name, pid, pos=Position.OF):
    return Player(
        name=name,
        player_type=PlayerType.HITTER,
        positions=[pos],
        selected_position=pos,
        yahoo_id=pid,
    )


def _active(name, pid, g_ros_adj, factor=1.0, pos=Position.OF):
    return ActiveBody(player=_player(name, pid, pos), factor=factor, g_ros_adj=g_ros_adj)


def _bench(name, pid, g_ros_full, per_game_value, pos=Position.OF):
    return BenchBody(
        player=_player(name, pid, pos),
        g_ros_full=g_ros_full,
        per_game_value=per_game_value,
        eligible_positions=frozenset({pos}),
    )


def _line(**kw):
    return {c: float(kw.get(c, 0.0)) for c in HITTING_COUNTING}


def _bench_sample(b, per_game, capacity=None):
    return BenchSample(
        body=b,
        per_game_counts=_line(**per_game),
        capacity=b.g_ros_full if capacity is None else capacity,
    )


def _no_replacement(_active_body):
    return _line()  # zero replacement -> isolates bench-fill mechanism


def _flat_replacement(val):
    return lambda _b: _line(**{c: val for c in HITTING_COUNTING})


def _realistic_replacement(_b):
    # A real full-season replacement line: counting stats << AB (NOT flat), so the
    # per-game conversion (stat / (ab / PA_PER_GAME)) is meaningful.
    return _line(r=43.0, hr=12.0, rbi=45.0, sb=4.0, h=120.0, ab=516.0)


def test_eligible_bench_gets_nonzero_fill_on_low_availability():
    a = _active("Star", "1", g_ros_adj=80.0)
    b = _bench("Depth", "2", g_ros_full=60.0, per_game_value=2.0)
    res = allocate_bench_fill(
        [ActiveSample(a, frac_missed=0.5)],
        [_bench_sample(b, {"r": 0.5, "h": 1.0, "ab": 4.0})],
        _no_replacement,
    )
    assert res.fill_counts["r"] > 0  # eligible bench fills an injured starter


def test_full_availability_yields_zero_fill():
    a = _active("Star", "1", g_ros_adj=80.0)
    b = _bench("Depth", "2", g_ros_full=60.0, per_game_value=2.0)
    res = allocate_bench_fill(
        [ActiveSample(a, frac_missed=0.0)],
        [_bench_sample(b, {"r": 0.5})],
        _no_replacement,
    )
    assert all(v == 0.0 for v in res.fill_counts.values())  # no injury -> no fill


def test_position_mismatch_routes_to_replacement_not_bench():
    a = _active("OFstar", "1", g_ros_adj=80.0, pos=Position.OF)
    b = _bench("Catcher", "2", g_ros_full=60.0, per_game_value=2.0, pos=Position.C)
    res = allocate_bench_fill(
        [ActiveSample(a, frac_missed=0.5)],  # 40 games missed
        [_bench_sample(b, {"r": 99.0})],  # huge bench rate, but wrong position
        _realistic_replacement,
    )
    # bench (C) cannot fill an OF shortfall -> fill is replacement-SCALE
    # (~43/120 r-per-game * 40 ~ 14 r), NOT the catcher's 99/game (~3960 r).
    assert 0.0 < res.fill_counts["r"] < 100.0


def test_replacement_per_game_not_overscaled():
    # A full-season replacement line (R=43 over ab=516 ~= 120 games) must convert
    # to ~0.36 r/game, NOT 43/PA_PER_GAME (~10/game). 30 missed games, no bench.
    a = _active("OFstar", "1", g_ros_adj=60.0, pos=Position.OF)
    res = allocate_bench_fill(
        [ActiveSample(a, frac_missed=0.5)],  # 0.5 * 60 = 30 games missed
        [],
        _realistic_replacement,
    )
    expected = 43.0 / (516.0 / 4.3) * 30.0  # per-game r * games_missed ~= 10.75
    assert abs(res.fill_counts["r"] - expected) < 1e-6


def test_fill_never_exceeds_bench_capacity():
    # Two OF starters both injured; one bench body eligible for both. Its total
    # contributed games cannot exceed its per-iteration CAPACITY (no longer the
    # static g_ros_full -- capacity = g_ros_full*scale, which CAN exceed g_ros_full).
    a1 = _active("S1", "1", g_ros_adj=100.0, pos=Position.OF)
    a2 = _active("S2", "2", g_ros_adj=100.0, pos=Position.OF)
    cap = 10.0
    b = _bench("Depth", "3", g_ros_full=cap, per_game_value=2.0, pos=Position.OF)
    res = allocate_bench_fill(
        [ActiveSample(a1, frac_missed=1.0), ActiveSample(a2, frac_missed=1.0)],
        [_bench_sample(b, {"r": 1.0}, capacity=cap)],
        _no_replacement,
    )
    assert res.fill_counts["r"] <= cap + 1e-9


def test_capacity_below_g_ros_full_limits_fill_and_cascades():
    # Starter misses 50 games. Best bench bat has capacity 10 (sampled low
    # availability); the residual cascades to the second bench bat.
    a = _active("Star", "1", g_ros_adj=100.0, pos=Position.OF)
    b1 = _bench("D1", "2", g_ros_full=60.0, per_game_value=3.0, pos=Position.OF)
    b2 = _bench("D2", "3", g_ros_full=60.0, per_game_value=1.0, pos=Position.OF)
    res = allocate_bench_fill(
        [ActiveSample(a, frac_missed=0.5)],  # 0.5 * 100 = 50 games missed
        [
            _bench_sample(b1, {"r": 1.0}, capacity=10.0),
            _bench_sample(b2, {"r": 0.5}, capacity=60.0),
        ],
        _no_replacement,
    )
    # b1: 10 games * 1.0 = 10 r; b2: remaining 40 games * 0.5 = 20 r -> 30 r.
    assert abs(res.fill_counts["r"] - 30.0) < 1e-9


def test_zero_capacity_body_skipped_and_cascades():
    # Best bench bat sampled fully unavailable (capacity 0) -> contributes nothing
    # despite the highest rate; the next eligible body covers the shortfall.
    a = _active("Star", "1", g_ros_adj=100.0, pos=Position.OF)
    b1 = _bench("D1", "2", g_ros_full=60.0, per_game_value=3.0, pos=Position.OF)
    b2 = _bench("D2", "3", g_ros_full=60.0, per_game_value=1.0, pos=Position.OF)
    res = allocate_bench_fill(
        [ActiveSample(a, frac_missed=0.5)],  # 50 games missed
        [_bench_sample(b1, {"r": 9.0}, capacity=0.0), _bench_sample(b2, {"r": 0.5}, capacity=60.0)],
        _no_replacement,
    )
    # b1 skipped (cap 0); b2 covers all 50 -> 25 r.
    assert abs(res.fill_counts["r"] - 25.0) < 1e-9


def test_per_game_value_ordering_picks_better_body():
    a = _active("Star", "1", g_ros_adj=20.0, pos=Position.OF)
    good = _bench("Good", "2", g_ros_full=100.0, per_game_value=5.0, pos=Position.OF)
    bad = _bench("Bad", "3", g_ros_full=100.0, per_game_value=1.0, pos=Position.OF)
    res = allocate_bench_fill(
        [ActiveSample(a, frac_missed=1.0)],
        [_bench_sample(good, {"r": 10.0}), _bench_sample(bad, {"r": 0.0})],
        _no_replacement,
    )
    # both have ample capacity for the 20-game shortfall, so the higher per-game
    # body covers it all -> nonzero r from "good", proving it was chosen first.
    assert res.fill_counts["r"] > 0


def test_residual_goes_to_replacement_when_bench_exhausted():
    a = _active("Star", "1", g_ros_adj=100.0, pos=Position.OF)
    b = _bench("Depth", "2", g_ros_full=5.0, per_game_value=2.0, pos=Position.OF)
    res = allocate_bench_fill(
        [ActiveSample(a, frac_missed=1.0)],  # 100 games missed, bench covers 5
        [_bench_sample(b, {"r": 0.0})],  # bench gives 0 r -> all r must be replacement
        _flat_replacement(0.5),
    )
    assert res.fill_counts["r"] > 0  # residual 95 games -> replacement r


def test_displaced_body_fill_bounded_by_g_ros_adj_not_g_ros_full():
    # CONSERVATION: a displaced body (factor 0.5, g_ros_full 80 -> g_ros_adj 40)
    # with frac_missed=1.0 misses at most g_ros_adj=40 games, NOT g_ros_full=80.
    # Pin the multiplier: bench gives exactly 1 r/game with ample capacity, so
    # bench-only fill r == games_missed == frac_missed * g_ros_adj == 40, never 80.
    a = _active("Displaced", "1", g_ros_adj=40.0, factor=0.5, pos=Position.OF)
    b = _bench("Depth", "2", g_ros_full=200.0, per_game_value=2.0, pos=Position.OF)
    res = allocate_bench_fill(
        [ActiveSample(a, frac_missed=1.0)],
        [_bench_sample(b, {"r": 1.0})],
        _no_replacement,
    )
    assert abs(res.fill_counts["r"] - 40.0) < 1e-6  # g_ros_adj (40), NOT g_ros_full (80)


def test_tie_break_by_player_id_ascending():
    # Two equal-per-game-value eligible bodies, only enough shortfall for one game.
    # Deterministic: id "2" (ascending) is chosen, contributing its distinctive rate.
    a = _active("Star", "1", g_ros_adj=1.0, pos=Position.OF)  # 1 game missed at frac 1.0
    b_lo = _bench("LowId", "2", g_ros_full=100.0, per_game_value=3.0, pos=Position.OF)
    b_hi = _bench("HighId", "9", g_ros_full=100.0, per_game_value=3.0, pos=Position.OF)
    res = allocate_bench_fill(
        [ActiveSample(a, frac_missed=1.0)],
        [_bench_sample(b_hi, {"r": 0.0}), _bench_sample(b_lo, {"r": 7.0})],
        _no_replacement,
    )
    assert abs(res.fill_counts["r"] - 7.0) < 1e-6  # id "2" (LowId) chosen, gives 7


# ---- Pitcher fill (#393): season-share units, starters cover starters ----


def _arm(name, pid):
    return Player(
        name=name,
        player_type=PlayerType.PITCHER,
        positions=[Position.P],
        selected_position=Position.P,
        yahoo_id=pid,
    )


def _p_active(name, pid, factor=1.0):
    return ActiveBody(player=_arm(name, pid), factor=factor, g_ros_adj=0.0)


def _p_bench(name, pid, role, value=1.0):
    from fantasy_baseball.mc_roster import BenchPitcherBody

    return BenchPitcherBody(player=_arm(name, pid), role=role, per_share_value=value)


def _p_line(**kw):
    from fantasy_baseball.mc_fill import PITCHER_FILL_COLS

    return {c: float(kw.get(c, 0.0)) for c in PITCHER_FILL_COLS}


def test_pitcher_fill_same_role_bench_covers_missed_share():
    from fantasy_baseball.mc_fill import allocate_pitcher_fill

    ace = _p_active("Ace", "1")
    spot = _p_bench("Spot", "2", "SP")
    out = allocate_pitcher_fill(
        [ActiveSample(body=ace, frac_missed=0.25)],
        [BenchSample(body=spot, per_game_counts=_p_line(ip=80.0, k=70.0), capacity=1.0)],
        role_of=lambda _ab: "SP",
        replacement_per_share=lambda _ab: _p_line(ip=999.0),
    ).fill_counts
    # A quarter of the remaining season, at the bench arm's own per-share line.
    assert out["ip"] == 20.0 and out["k"] == 17.5


def test_pitcher_fill_other_role_bench_is_skipped_for_replacement():
    """Role gate: an injured starter is not covered by a bench reliever when the
    covering role is 'SP'; the residual goes to the replacement streamer."""
    from fantasy_baseball.mc_fill import allocate_pitcher_fill

    ace = _p_active("Ace", "1")
    setup = _p_bench("Setup", "2", "RP")
    out = allocate_pitcher_fill(
        [ActiveSample(body=ace, frac_missed=0.5)],
        [BenchSample(body=setup, per_game_counts=_p_line(ip=60.0, k=80.0), capacity=1.0)],
        role_of=lambda _ab: "SP",
        replacement_per_share=lambda _ab: _p_line(ip=100.0, k=90.0),
    ).fill_counts
    assert out["ip"] == 50.0 and out["k"] == 45.0


def test_pitcher_fill_capacity_is_the_bench_arms_own_availability():
    from fantasy_baseball.mc_fill import allocate_pitcher_fill

    out = allocate_pitcher_fill(
        [
            ActiveSample(body=_p_active("A", "1"), frac_missed=0.5),
            ActiveSample(body=_p_active("B", "2"), frac_missed=0.5),
        ],
        [
            BenchSample(
                body=_p_bench("Spot", "3", "SP"), per_game_counts=_p_line(ip=80.0), capacity=0.6
            )
        ],
        role_of=lambda _ab: "SP",
        replacement_per_share=lambda _ab: _p_line(ip=40.0),
    ).fill_counts
    # 1.0 share missed in total; the bench arm covers 0.6 of it, replacement the other 0.4.
    assert abs(out["ip"] - (0.6 * 80.0 + 0.4 * 40.0)) < 1e-9


def test_pitcher_fill_never_fills_saves():
    from fantasy_baseball.mc_fill import PITCHER_FILL_COLS, allocate_pitcher_fill

    assert "sv" not in PITCHER_FILL_COLS
    out = allocate_pitcher_fill(
        [ActiveSample(body=_p_active("Closer", "1"), frac_missed=1.0)],
        [],
        role_of=lambda _ab: "RP",
        replacement_per_share=lambda _ab: {"sv": 30.0, "ip": 60.0},
    ).fill_counts
    assert "sv" not in out and out["ip"] == 60.0
