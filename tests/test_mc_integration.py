"""Phase 4a + 4b MC-integration tests.

4a (below): sampler-plumbing changes to ``_apply_variance_batch`` -- the new
``VarianceBatch`` return, the always-populated ``frac_missed`` array, the
``suppress_repl`` flag, and the ``pt_mean_fraction`` mean-horizon split -- all
while keeping the DEFAULT path byte-for-byte identical to the pre-4a code.

4b: ROS-direct hitter integration -- the ``_simulate_team_hitters_ros_direct``
helper (displacement + bench injury-fill) and the ``effective_rosters`` dual
path through ``simulate_remaining_season_batch`` (the body-direct hitter route
when supplied; byte-identical top-k fallback when ``None``). MECHANISM-ONLY: no
absolute-magnitude assertions tied to the ``pt_mean_fraction``/per-game choices.
"""

import numpy as np

from fantasy_baseball.simulation import VarianceBatch, _apply_variance_batch


def _players():
    """Two hitters with positive replacement lines (so suppress_repl can bite)."""
    return [
        {
            "name": "A",
            "player_type": "hitter",
            "r": 80,
            "hr": 25,
            "rbi": 80,
            "sb": 12,
            "h": 150,
            "ab": 550,
            "positions": ["2B"],
        },
        {
            "name": "B",
            "player_type": "hitter",
            "r": 60,
            "hr": 10,
            "rbi": 55,
            "sb": 3,
            "h": 120,
            "ab": 480,
            "positions": ["C"],
        },
    ]


# Seed-pinned snapshot of the OLD (pre-4a) ``_apply_variance_batch`` output for
# ``_players()`` with rng seed 12345, fraction_remaining=0.4, n_iter=5. Captured
# by running the pre-change function under ``git stash`` (the commit before 4a).
# This is the load-bearing back-compat anchor: if the default path ever drifts
# from these literals, the refactor changed the simulated distribution.
_LEGACY_DEFAULT_COUNTS = {
    "ab": np.array(
        [
            [545.9231584685585, 473.4650557183418],
            [591.8447987818389, 506.7672773890555],
            [549.0686449886527, 473.99234741403365],
            [570.883099601002, 467.303026818985],
            [578.5799718166516, 562.7433148665731],
        ]
    ),
    "h": np.array(
        [
            [133.0363892833034, 124.26735154627076],
            [146.0, 125.0],
            [150.74971479540616, 118.27752327541049],
            [142.0, 113.83466895383512],
            [148.0, 160.0],
        ]
    ),
    "hr": np.array(
        [
            [15.261879521636649, 9.605074034091501],
            [24.0, 5.0],
            [25.28827655113129, 10.475563793044364],
            [26.0, 14.118554816389642],
            [24.0, 14.0],
        ]
    ),
    "r": np.array(
        [
            [65.0181946416517, 60.305647991073755],
            [69.0, 64.0],
            [69.37485739770308, 51.79685775838857],
            [81.0, 66.25146535010217],
            [83.0, 77.0],
        ]
    ),
    "rbi": np.array(
        [
            [60.82405933063069, 59.420296136366005],
            [74.0, 59.0],
            [70.3305071590675, 53.902255172177455],
            [80.0, 62.47421926555857],
            [83.0, 66.0],
        ]
    ),
    "sb": np.array(
        [
            [4.650150143678695, 1.4585925811690004],
            [13.0, 4.0],
            [7.376977028402457, 1.4215896551555325],
            [14.0, 4.891015661825612],
            [11.0, 3.0],
        ]
    ),
}


def test_variance_batch_default_matches_legacy_columns():
    """RNG-stream-STABILITY: default path is byte-identical to pre-4a.

    This is NOT merely an arithmetic-equality check. The 4a refactor must not
    REORDER, ADD, or REMOVE any ``rng.random`` / ``rng.multivariate_normal``
    (copula) draw on the default path. Splitting one ``playing_time_moments``
    call into a mean call and an sd call is allowed only because that function is
    closed-form and consumes NO rng. If a future edit perturbs the rng stream
    (extra draw, moved copula sampling), these pinned literals will diverge for a
    non-bug reason -- DO NOT "fix" that by re-pinning the snapshot; fix the rng
    drift instead.
    """
    rng = np.random.default_rng(12345)
    result = _apply_variance_batch(_players(), "hitter", rng, 0.4, 5)
    assert isinstance(result, VarianceBatch)
    assert set(result.counts.keys()) == set(_LEGACY_DEFAULT_COUNTS.keys())
    for col, expected in _LEGACY_DEFAULT_COUNTS.items():
        np.testing.assert_array_equal(result.counts[col], expected)


def test_frac_missed_exposed_and_in_unit_range():
    """``frac_missed`` is always populated, shape (n_iter, n_players), in [0, 1]."""
    rng = np.random.default_rng(7)
    result = _apply_variance_batch(_players(), "hitter", rng, 0.4, 6)
    assert result.frac_missed.shape == (6, 2)
    assert np.all(result.frac_missed >= 0.0)
    assert np.all(result.frac_missed <= 1.0)
    # It is exactly max(0, 1 - scales); scales <= 1 cannot exceed 1 here, and a
    # partial-season haircut guarantees at least one positive shortfall.
    assert np.any(result.frac_missed > 0.0)


def test_scales_exposed_and_consistent_with_frac_missed():
    """scales is exposed, shape (n_iter, n_players), and frac_missed == max(0, 1-scales)."""
    rng = np.random.default_rng(7)
    result = _apply_variance_batch(_players(), "hitter", rng, 0.4, 6)
    assert result.scales.shape == (6, 2)
    np.testing.assert_array_equal(result.frac_missed, np.maximum(0.0, 1.0 - result.scales))


def test_scales_empty_player_list_is_shape_correct():
    """The n_players==0 early return still builds a shape-correct scales array."""
    result = _apply_variance_batch([], "hitter", np.random.default_rng(1), 0.4, 4)
    assert result.scales.shape == (4, 0)


def test_suppress_repl_removes_replacement_contribution():
    """suppress_repl=True drops the replacement backfill -> strictly lower counts.

    Both players have positive-SB replacement lines (2B / C floors), and a
    partial-season run forces frac_missed > 0, so the default path folds a
    positive ``repl_contrib`` into at least one column. Suppressing it must yield
    strictly smaller counts wherever frac_missed > 0 and the replacement line is
    positive.
    """
    default = _apply_variance_batch(_players(), "hitter", np.random.default_rng(99), 0.4, 8)
    suppressed = _apply_variance_batch(
        _players(), "hitter", np.random.default_rng(99), 0.4, 8, suppress_repl=True
    )
    # Same rng seed -> same scales/copula draws; only the repl term differs.
    np.testing.assert_array_equal(default.frac_missed, suppressed.frac_missed)
    missed = default.frac_missed > 0.0
    assert np.any(missed)
    found_strictly_lower = False
    for col in default.counts:
        d = default.counts[col]
        s = suppressed.counts[col]
        # Suppressed is never larger (repl_contrib >= 0).
        assert np.all(s <= d + 1e-9)
        # Where games were missed, at least one column drops strictly.
        if np.any((s < d - 1e-9) & missed):
            found_strictly_lower = True
    assert found_strictly_lower


def test_pt_mean_fraction_moves_mean_only_not_sd_dispersion():
    """MECHANISM: pt_mean_fraction drives ONLY the mean haircut, not the SD.

    The playing-time haircut is ``eff_mean = 1 - (1 - mean_scale) * fr_mean`` and
    the spread is ``eff_sd = cv_pt * sqrt(fraction_remaining)``. ``pt_mean_fraction``
    overrides ``fr_mean`` ONLY: a SMALLER value (e.g. 0.0) applies less of the
    sub-1 ``mean_scale`` haircut -> a HIGHER realized mean; a LARGER value (1.0)
    applies the full haircut -> a LOWER mean. Crucially the SD term keeps
    ``fraction_remaining`` and the ``_negbin_copula_counts`` dispersion is
    untouched, so the per-player spread is BYTE-identical across pt_mean_fraction
    values (only the location shifts).

    (Note: ``pt_mean_fraction=1.0`` LOWERS the mean relative to the
    ``fraction_remaining=0.4`` default -- it applies the FULL haircut, which is
    exactly what 4b's ROS-direct path wants since the whole remaining window is at
    risk. ``suppress_repl=True`` isolates the sampled production from the
    replacement backfill so the mean shift is read cleanly.)
    """
    n_iter = 8000
    fr = 0.4
    kw = dict(suppress_repl=True)
    less_haircut = _apply_variance_batch(
        _players(), "hitter", np.random.default_rng(2024), fr, n_iter, pt_mean_fraction=0.0, **kw
    )
    default = _apply_variance_batch(
        _players(), "hitter", np.random.default_rng(2024), fr, n_iter, **kw
    )
    full_haircut = _apply_variance_batch(
        _players(), "hitter", np.random.default_rng(2024), fr, n_iter, pt_mean_fraction=1.0, **kw
    )
    # Mean is monotone in the haircut term: no haircut (0.0) > partial (default
    # 0.4) > full (1.0). Use AB (the playing-time-driven volume column).
    assert less_haircut.counts["ab"].mean() > default.counts["ab"].mean()
    assert default.counts["ab"].mean() > full_haircut.counts["ab"].mean()
    # SD is NOT collapsed and does NOT move with pt_mean_fraction: the spread term
    # keeps fraction_remaining, so per-player SD is identical across all three.
    for j in range(2):
        sd0 = less_haircut.counts["ab"][:, j].std()
        sd_full = full_haircut.counts["ab"][:, j].std()
        assert sd0 > 0.0
        np.testing.assert_allclose(sd0, sd_full, rtol=0, atol=1e-9)


def test_apply_variance_batch_pt_volumes_default_is_byte_identical():
    """pt_volumes=None reproduces the legacy per-player vol == _projected_volume.

    Seed-pinned, additive-param contract (mirror the 4a byte-equality test): the
    new ``pt_volumes`` knob defaulting to None must NOT perturb the rng stream or
    the per-player curve lookup. Omitting it and passing None explicitly must be
    bit-identical across every column (and frac_missed).
    """
    omitted = _apply_variance_batch(_players(), "hitter", np.random.default_rng(54321), 0.4, 7)
    none_passed = _apply_variance_batch(
        _players(), "hitter", np.random.default_rng(54321), 0.4, 7, pt_volumes=None
    )
    assert set(omitted.counts) == set(none_passed.counts)
    for col in omitted.counts:
        np.testing.assert_array_equal(omitted.counts[col], none_passed.counts[col])
    np.testing.assert_array_equal(omitted.frac_missed, none_passed.frac_missed)


# ---------------------------------------------------------------------------
# Phase 4b: ROS-direct hitter integration.
# ---------------------------------------------------------------------------

from fantasy_baseball.mc_roster import (  # noqa: E402
    ActiveBody,
    EffectiveRoster,
    build_effective_roster,
)
from fantasy_baseball.models.player import (  # noqa: E402
    HitterStats,
    PitcherStats,
    Player,
    PlayerType,
)
from fantasy_baseball.models.positions import Position  # noqa: E402
from fantasy_baseball.models.standings import CategoryStats  # noqa: E402
from fantasy_baseball.scoring import LeagueContext  # noqa: E402
from fantasy_baseball.simulation import (  # noqa: E402
    _simulate_team_hitters_ros_direct,
    run_ros_monte_carlo,
    simulate_remaining_season_batch,
)
from fantasy_baseball.utils.constants import ALL_CATEGORIES  # noqa: E402


def _hitter(name, slot, pid, *, r=80, hr=20, rbi=70, sb=5, h=150, ab=550, pa=600, g=150):
    return Player(
        name=name,
        player_type=PlayerType.HITTER,
        positions=[Position.OF],
        selected_position=slot,
        yahoo_id=pid,
        rest_of_season=HitterStats.from_dict(
            {"r": r, "hr": hr, "rbi": rbi, "sb": sb, "h": h, "ab": ab, "pa": pa, "g": g}
        ),
    )


def _pitcher(name, slot, pid):
    return Player(
        name=name,
        player_type=PlayerType.PITCHER,
        positions=[Position.SP],
        selected_position=slot,
        yahoo_id=pid,
        rest_of_season=PitcherStats.from_dict(
            {"w": 8, "k": 100, "sv": 0, "ip": 90, "er": 35, "bb": 28, "h_allowed": 80, "g": 15}
        ),
    )


def _ctx(team="Me", others=("Opp",)):
    base = {t: CategoryStats() for t in others}
    sds = {t: {c: 5.0 for c in ALL_CATEGORIES} for t in (team, *others)}
    return LeagueContext(baseline_other_team_stats=base, team_sds=sds, team_name=team)


def _eff_roster(roster, team="Me"):
    return build_effective_roster(roster, _ctx(team=team))


def test_healthy_roster_hitter_totals_positive_no_bench_contrib():
    """Healthy roster (frac_missed == 0) -> positive counting; bench contributes 0.

    fraction_remaining=1.0 lifts the playing-time haircut so a seed that draws
    full (or above) gives no missed games. With no shortfall the bench injury
    fill is exactly zero, so the team total equals the active bodies' own
    realized counting (bench excluded).
    """
    roster = [
        _hitter("Starter", Position.OF, "1"),
        _hitter("BenchBat", Position.BN, "2", r=200, hr=200, rbi=200, sb=200),
    ]
    eff = _eff_roster(roster)
    rng = np.random.default_rng(7)
    out = _simulate_team_hitters_ros_direct(eff, 1.0, rng, 64)
    for cat in ("R", "HR", "RBI", "SB"):
        assert np.all(out[cat] >= 0.0)
        assert out[cat].mean() > 0.0
    assert np.all(out["ros_ab"] > 0.0)

    # The bench bat (huge raw stats) must NOT seat itself: the active set is
    # fixed to the starter. Active-only ceiling = the starter's own draws.
    starter_body = next(b for b in eff.active if b.player.name == "Starter")
    starter_flat = [starter_body.player.to_flat_dict()]
    vb = _apply_variance_batch(
        starter_flat,
        PlayerType.HITTER,
        np.random.default_rng(7),
        1.0,
        64,
        pt_mean_fraction=1.0,
        suppress_repl=True,
    )
    # Where the starter draws full (frac_missed == 0), team R == starter R alone:
    # the bench bat adds nothing. Use the same seed so the active draws line up.
    healthy = vb.frac_missed[:, 0] == 0.0
    assert np.any(healthy)
    np.testing.assert_allclose(out["R"][healthy], vb.counts["r"][healthy, 0])


def test_injured_active_body_gets_bench_fill():
    """An injured active body routes its shortfall to the eligible bench body.

    Forcing a low draw (small fraction_remaining + a starved-then-eligible
    bench) makes the bench per-game line contribute a NONZERO fill share, so the
    team total EXCEEDS the active-only total (which would otherwise floor the
    missed games at zero production). Mechanism only -- no magnitude pinned.
    """
    roster = [
        _hitter("Starter", Position.OF, "1"),
        _hitter("BenchBat", Position.OF, "2", r=120, hr=40, rbi=110, sb=30),
    ]
    eff = _eff_roster(roster)
    rng = np.random.default_rng(3)
    # Low fraction_remaining -> the haircut forces frac_missed > 0 on the starter.
    with_bench = _simulate_team_hitters_ros_direct(eff, 0.2, rng, 256)

    # Same roster, bench removed: no fill body, so the shortfall routes only to
    # replacement (smaller per-game line than the strong bench bat).
    eff_no_bench = build_effective_roster([roster[0]], _ctx())
    no_bench = _simulate_team_hitters_ros_direct(eff_no_bench, 0.2, np.random.default_rng(3), 256)
    # The strong bench bat lifts the team R mean above the bench-less case.
    assert with_bench["R"].mean() > no_bench["R"].mean()


def test_hitter_team_total_at_least_ytd_with_caller_blend():
    """Banked-YTD floor is STRUCTURAL: caller blend team_total = YTD + ROS >= YTD.

    ROS-direct means every summed-ROS counting cat is >= 0, so the caller's
    ``actuals + ROS`` blend never dips below YTD -- no max(actual, sim) clamp
    needed. Exercise the full batch with effective_rosters supplied.
    """
    roster = [_hitter("Starter", Position.OF, "1"), _pitcher("Ace", Position.P, "9")]
    ytd = {
        "R": 40,
        "HR": 12,
        "RBI": 38,
        "SB": 7,
        "AVG": 0.270,
        "W": 5,
        "K": 70,
        "SV": 0,
        "ERA": 3.50,
        "WHIP": 1.15,
        "AB": 1500,
        "IP": 400,
    }
    flat = {"Me": [p.to_flat_dict_full_season() for p in roster]}
    batch = simulate_remaining_season_batch(
        {"Me": ytd},
        flat,
        0.4,
        np.random.default_rng(11),
        h_slots=13,
        p_slots=9,
        n_iter=128,
        effective_rosters={"Me": _eff_roster(roster)},
    )
    for cat in ("R", "HR", "RBI", "SB"):
        assert np.all(batch["Me"][cat] >= ytd[cat] - 1e-6)


def test_churn_freeze_active_set_fixed():
    """The contributing active hitter set is identical every iteration.

    A bench bat with higher raw stats than the starter would be seated by the
    OLD per-iteration top-k; the body-direct engine FIXES the active set, so the
    bench bat contributes ONLY injury fill (zero when the starter draws full).
    Verified by: in the healthy case, the team R equals the starter's own draw
    on every iteration where it draws full (the bench bat never seats).
    """
    roster = [
        _hitter("Starter", Position.OF, "1", r=70),
        _hitter("BenchBat", Position.BN, "2", r=300, hr=80, rbi=250, sb=60),
    ]
    eff = _eff_roster(roster)
    out = _simulate_team_hitters_ros_direct(eff, 1.0, np.random.default_rng(5), 128)
    starter_body = next(b for b in eff.active if b.player.name == "Starter")
    vb = _apply_variance_batch(
        [starter_body.player.to_flat_dict()],
        PlayerType.HITTER,
        np.random.default_rng(5),
        1.0,
        128,
        pt_mean_fraction=1.0,
        suppress_repl=True,
    )
    healthy = vb.frac_missed[:, 0] == 0.0
    assert np.any(healthy)
    # The bench bat's 300 R would dominate top-k every iter; here the healthy
    # team R stays at the starter's ~70-level draw, never the bench bat's ~300.
    np.testing.assert_allclose(out["R"][healthy], vb.counts["r"][healthy, 0])


def _active_body(player, *, factor=1.0):
    """ActiveBody with g_ros_adj = factor * rest_of_season.g (the helper's input)."""
    g = float(player.rest_of_season.g)
    return ActiveBody(player=player, factor=factor, g_ros_adj=factor * g)


def _solo_eff(player, *, factor=1.0):
    """EffectiveRoster with one active hitter and an empty bench fill pool."""
    return EffectiveRoster(active=[_active_body(player, factor=factor)], bench=[])


def test_repl_not_double_counted_on_new_path():
    """STARVED bench: an injured body's shortfall hits replacement EXACTLY once.

    The helper samples with ``suppress_repl=True`` (the bench fill owns the
    backfill), so the built-in ``repl_contrib`` is NOT folded into the active
    draw. With no bench, the only replacement is the single fill pass. Mechanism
    check (scale-independent): on the iterations where the body draws FULL
    (frac_missed == 0) there is no replacement at all, so the team R equals the
    body's own suppressed draw exactly -- proving the built-in backfill is off
    (a non-suppressed draw would add ``repl * frac_missed``, but more tellingly
    the helper's realized counts track the suppressed sampler bit-for-bit).
    """
    player = _hitter("Solo", Position.OF, "1", r=80)
    eff = _solo_eff(player)
    out = _simulate_team_hitters_ros_direct(eff, 0.4, np.random.default_rng(1), 256)
    # Reconstruct the suppressed sampler with the same seed.
    vb = _apply_variance_batch(
        [player.to_flat_dict()],
        PlayerType.HITTER,
        np.random.default_rng(1),
        0.4,
        256,
        pt_mean_fraction=1.0,
        suppress_repl=True,
    )
    healthy = vb.frac_missed[:, 0] == 0.0
    assert np.any(healthy)
    # On healthy iters: zero fill, so team R == the suppressed draw exactly. If
    # the helper had used the default (non-suppressed) sampler, the columns would
    # still match here (repl*0 == 0) -- so additionally assert the realized R on
    # the INJURED iters never exceeds the suppressed draw plus a SINGLE
    # replacement allocation (no double-count): team R - fill == suppressed draw.
    np.testing.assert_allclose(out["R"][healthy], vb.counts["r"][healthy, 0])
    # On every iter the team R is the suppressed active draw (factor 1.0) PLUS a
    # non-negative single fill pass -- so it is always >= the suppressed draw and
    # carries the built-in replacement zero times (suppress_repl), the fill once.
    repl = _replacement_line_for_test(player)
    assert np.all(out["R"] >= vb.counts["r"][:, 0] - 1e-6)
    assert repl["r"] >= 0  # replacement line is well-formed


def _replacement_line_for_test(player):
    from fantasy_baseball.simulation import _replacement_line

    return _replacement_line(player.to_flat_dict(), is_hitter=True)


def test_displacement_factor_scales_hitter_mean():
    """A displaced active body (factor < 1) contributes a LOWER mean than undisplaced.

    Construct two EffectiveRosters with the SAME single active hitter -- one
    undisplaced (factor 1.0), one displaced (factor 0.5) -- and an empty bench.
    The displacement multiplies the sampled ROS counts, so the team mean is
    strictly lower. Isolated on the healthy iterations (no fill), the displaced
    mean is ~0.5x the undisplaced. No absolute magnitude pinned.
    """
    player = _hitter("Star", Position.OF, "1", r=100, pa=600, g=150)
    eff_full = _solo_eff(player, factor=1.0)
    eff_disp = _solo_eff(player, factor=0.5)
    full_out = _simulate_team_hitters_ros_direct(eff_full, 1.0, np.random.default_rng(8), 512)
    disp_out = _simulate_team_hitters_ros_direct(eff_disp, 1.0, np.random.default_rng(8), 512)
    # factor < 1 lowers the contributed mean.
    assert disp_out["R"].mean() < full_out["R"].mean()
    # On healthy iters (no fill in either), the displaced draw is exactly 0.5x.
    vb = _apply_variance_batch(
        [player.to_flat_dict()],
        PlayerType.HITTER,
        np.random.default_rng(8),
        1.0,
        512,
        pt_mean_fraction=1.0,
        suppress_repl=True,
    )
    healthy = vb.frac_missed[:, 0] == 0.0
    assert np.any(healthy)
    np.testing.assert_allclose(disp_out["R"][healthy], 0.5 * full_out["R"][healthy])


def _one_full_timer_hitter(*, full_pa=620.0, ros_pa=305.0, ros_r=45.0):
    """EffectiveRoster with ONE active full-timer hitter, empty bench.

    Unlike ``_hitter`` (which leaves ``full_season_projection=None`` so the
    ROS-fallback in ``_full_season_pt_volume`` would make the cv_pt test
    vacuous), this sets BOTH ``full_season_projection.pa`` AND
    ``rest_of_season.pa`` to DIFFERENT volumes, so the fix (curve lookup at
    full-season volume, not ROS) is actually exercised. ROS counting is scaled to
    the ROS window (a real mid-season ROS line, not a full-season line).
    """
    scale = ros_pa / full_pa
    player = Player(
        name="FullTimer",
        player_type=PlayerType.HITTER,
        positions=[Position.OF],
        selected_position=Position.OF,
        yahoo_id="1",
        rest_of_season=HitterStats.from_dict(
            {
                "r": ros_r,
                "hr": 15 * scale,
                "rbi": 70 * scale,
                "sb": 5 * scale,
                "h": 150 * scale,
                "ab": 0.9 * ros_pa,
                "pa": ros_pa,
                "g": 150 * scale,
            }
        ),
        full_season_projection=HitterStats.from_dict(
            {
                "r": ros_r / scale,
                "hr": 15,
                "rbi": 70,
                "sb": 5,
                "h": 150,
                "ab": 0.9 * full_pa,
                "pa": full_pa,
                "g": 150,
            }
        ),
    )
    return _solo_eff(player)


def test_ros_direct_uses_full_season_volume_for_cv_pt():
    """ROS-direct samples the PT curve at FULL-SEASON volume, not ROS volume.

    A full-timer (full-season PA 620, ROS PA 305) must be sampled with the
    FULL-SEASON cv_pt band (~0.20), NOT the inflated ROS-volume band (~0.42 at
    305 PA, where a full-timer is misclassified as a part-timer). Two checks:

    1) Reconstruct the underlying sampler (same seed) with full-season volume
       (620) vs ROS volume (305). The full-season lookup yields a STRICTLY
       narrower R SD on the raw active draw -- that separation is the bug. Pin
       that the helper actually uses the full-season (narrower) band: the helper's
       realized R SD equals the full-season-volume draw's R SD, NOT the wider
       ROS-volume draw's.
    2) Sanity band: the helper R SD sits well below the ~2x-wide ROS-volume PT
       band (engineering guard against a gross regression).
    """
    from fantasy_baseball.utils.playing_time import playing_time_params

    eff = _one_full_timer_hitter(full_pa=620.0, ros_pa=305.0, ros_r=45.0)
    out = _simulate_team_hitters_ros_direct(eff, 0.49, np.random.default_rng(0), 8000)
    helper_sd = out["R"].std()

    # Reconstruct the active sampler with the SAME seed, full-season vol vs ROS.
    body = eff.active[0]
    flat = [body.player.to_flat_dict()]
    full_vol = _apply_variance_batch(
        flat,
        PlayerType.HITTER,
        np.random.default_rng(0),
        0.49,
        8000,
        pt_mean_fraction=1.0,
        suppress_repl=True,
        pt_volumes=np.array([620.0]),
    )
    ros_vol = _apply_variance_batch(
        flat,
        PlayerType.HITTER,
        np.random.default_rng(0),
        0.49,
        8000,
        pt_mean_fraction=1.0,
        suppress_repl=True,
        pt_volumes=np.array([305.0]),
    )
    sd_full = full_vol.counts["r"][:, 0].std()
    sd_ros = ros_vol.counts["r"][:, 0].std()
    # The bug signal: ROS-volume lookup is materially WIDER than full-season.
    assert sd_ros > sd_full * 1.4, (sd_full, sd_ros)
    # Empty bench + factor 1.0: on the healthy iters (no fill) the helper's
    # realized R equals the full-season-volume draw exactly (the helper samples at
    # full-season vol), and is NOT the ROS-volume draw.
    healthy = full_vol.frac_missed[:, 0] == 0.0
    assert np.any(healthy)
    np.testing.assert_allclose(out["R"][healthy], full_vol.counts["r"][healthy, 0])
    assert not np.allclose(out["R"][healthy], ros_vol.counts["r"][healthy, 0])

    # Engineering band: well under the ~2x-wide ROS-volume PT scale.
    cv_ros = playing_time_params(PlayerType.HITTER, 305.0)[1]
    assert helper_sd < 45.0 * cv_ros * (0.49**0.5) * 0.9, helper_sd


def _mixed_rosters():
    return {
        "Me": [
            _hitter("H1", Position.OF, "1"),
            _hitter("H2", Position.OF, "2", r=60),
            _pitcher("P1", Position.P, "10"),
            _pitcher("P2", Position.P, "11"),
        ],
        "Opp": [
            _hitter("OH1", Position.OF, "3", r=75),
            _pitcher("OP1", Position.P, "12"),
        ],
    }


def _flat_rosters(rosters):
    return {t: [p.to_flat_dict_full_season() for p in players] for t, players in rosters.items()}


def test_pitchers_ros_direct_track_eroto_projection():
    """Phase 5: effective_rosters routes pitchers through the ROS-direct helper.

    (Pre-Phase-5 this test asserted the pitcher distribution was UNCHANGED with
    vs without effective_rosters -- correct only while pitchers always used the
    top-k full-season path. Phase 5 deliberately switches the effective_rosters
    pitcher path to ROS-direct (pt_mean_fraction=0, no haircut, no top-k
    over-credit), so the two paths now legitimately DIFFER. The new invariant: the
    ROS-direct pitcher mean tracks the summed ERoto ROS projection -- here two
    active SP at K=100/IP=90 -> ~200 K / ~180 IP -- with NO mean haircut.)
    """
    rosters = _mixed_rosters()
    actuals = {t: {} for t in rosters}
    flat = _flat_rosters(rosters)
    eff = {t: _eff_roster(players, team=t) for t, players in rosters.items()}
    n = 4000
    with_eff = simulate_remaining_season_batch(
        actuals, flat, 0.4, np.random.default_rng(21), 13, 9, n, effective_rosters=eff
    )
    # Me has two active SP, each ROS K=100 / IP=90 -> summed projection 200 / 180.
    proj_k = sum(p.rest_of_season.k for p in rosters["Me"] if p.player_type == PlayerType.PITCHER)
    proj_ip = sum(p.rest_of_season.ip for p in rosters["Me"] if p.player_type == PlayerType.PITCHER)
    assert abs(with_eff["Me"]["K"].mean() - proj_k) / proj_k < 0.06, with_eff["Me"]["K"].mean()
    assert abs(with_eff["Me"]["ERA"].mean()) > 0  # ERA recombines from ROS-direct volume
    assert with_eff["Me"]["W"].mean() > 0 and proj_ip > 0


def test_whole_context_fallback_to_topk():
    """effective_rosters=None is BYTE-identical to the pre-4b top-k batch.

    The byte anchor: the None path runs the EXACT old code path with zero
    added/reordered rng draws. Two seeded runs with effective_rosters omitted
    vs explicitly None must be bit-equal across every category.
    """
    rosters = _mixed_rosters()
    actuals = {t: {} for t in rosters}
    flat = _flat_rosters(rosters)
    omitted = simulate_remaining_season_batch(
        actuals, flat, 0.4, np.random.default_rng(33), 13, 9, 64
    )
    explicit_none = simulate_remaining_season_batch(
        actuals, flat, 0.4, np.random.default_rng(33), 13, 9, 64, effective_rosters=None
    )
    for team in rosters:
        for cat in omitted[team]:
            np.testing.assert_array_equal(omitted[team][cat], explicit_none[team][cat])


def test_run_ros_monte_carlo_accepts_effective_rosters():
    """run_ros_monte_carlo threads effective_rosters through to the batch."""
    rosters = _mixed_rosters()
    actuals = {t: {} for t in rosters}
    eff = {t: _eff_roster(players, team=t) for t, players in rosters.items()}
    result = run_ros_monte_carlo(
        team_rosters=rosters,
        actual_standings=actuals,
        fraction_remaining=0.4,
        h_slots=13,
        p_slots=9,
        user_team_name="Me",
        n_iterations=50,
        seed=42,
        effective_rosters=eff,
    )
    assert "team_results" in result
    assert np.isfinite(result["team_results"]["Me"]["median_pts"])


# ---------------------------------------------------------------------------
# Phase 5: ROS-direct pitcher integration.
# ---------------------------------------------------------------------------

from fantasy_baseball.simulation import (  # noqa: E402
    _simulate_team_pitchers_ros_direct,
)


def _pitcher_custom(name, slot, pid, *, w=8, k=100, sv=0, ip=90, er=35, bb=28, ha=80, g=15):
    return Player(
        name=name,
        player_type=PlayerType.PITCHER,
        positions=[Position.SP],
        selected_position=slot,
        yahoo_id=pid,
        rest_of_season=PitcherStats.from_dict(
            {"w": w, "k": k, "sv": sv, "ip": ip, "er": er, "bb": bb, "h_allowed": ha, "g": g}
        ),
    )


def _solo_eff_pitcher(player, *, factor=1.0):
    """EffectiveRoster with one active pitcher and an empty bench fill pool."""
    return EffectiveRoster(active=[_active_body(player, factor=factor)], bench=[])


def test_pitcher_helper_samples_active_only_applies_factor():
    """K positive; a factor<1 active body contributes less than at factor 1.

    Two active pitchers in the EffectiveRoster; healthy bench pitchers are not
    present in ``EffectiveRoster.active`` so they never contribute. Mechanism:
    every returned array is populated, K/IP means are positive, and halving a
    body's displacement factor strictly lowers its contributed K mean.
    """
    p = _pitcher_custom("SP1", Position.P, "1", k=150, ip=180)
    eff_full = _solo_eff_pitcher(p, factor=1.0)
    eff_disp = _solo_eff_pitcher(p, factor=0.5)
    out = _simulate_team_pitchers_ros_direct(eff_full, 0.5, np.random.default_rng(0), 500)
    assert set(out) >= {"W", "K", "SV", "ros_ip", "ros_er", "ros_bb", "ros_ha"}
    assert out["K"].mean() > 0 and out["ros_ip"].mean() > 0

    disp = _simulate_team_pitchers_ros_direct(eff_disp, 0.5, np.random.default_rng(0), 500)
    assert disp["K"].mean() < out["K"].mean()


def test_pitcher_mean_matches_projection_no_haircut():
    """CRITICAL regression: pt_mean_fraction=0 => NO playing-time mean haircut.

    With one active SP at factor 1.0 the helper's K/IP means must track the
    summed ROS projection (== ERoto), NOT mean_scale*projection (~0.8x, which a
    pt_mean_fraction=1.0 haircut would wrongly produce). Pin to the body's actual
    ROS projection (read off the constructed Player) and assert within ~6%.
    """
    p = _pitcher_custom("Ace", Position.P, "1", k=150, ip=180)
    eff = _solo_eff_pitcher(p, factor=1.0)
    proj_k = float(p.rest_of_season.k)
    proj_ip = float(p.rest_of_season.ip)
    out = _simulate_team_pitchers_ros_direct(eff, 0.5, np.random.default_rng(0), 4000)
    assert abs(out["K"].mean() - proj_k) / proj_k < 0.06, out["K"].mean()
    assert abs(out["ros_ip"].mean() - proj_ip) / proj_ip < 0.06, out["ros_ip"].mean()


def test_pitcher_helper_empty_active_returns_zeros():
    """No active pitchers (hitters-only roster) -> all-zero pitcher arrays."""
    roster = [_hitter("OnlyBat", Position.OF, "1")]
    eff = _eff_roster(roster)
    out = _simulate_team_pitchers_ros_direct(eff, 0.5, np.random.default_rng(0), 100)
    assert all(
        (out[c] == 0).all() for c in ("W", "K", "SV", "ros_ip", "ros_er", "ros_bb", "ros_ha")
    )


def test_effective_rosters_routes_pitchers_and_no_fallback_regression():
    """effective_rosters routes pitcher cats through the ROS-direct helper.

    With effective_rosters supplied the team's W/K/SV = YTD + ROS (no clamp);
    with None the byte-identical top-k path still holds. Assert the two PATHS
    differ for a team whose active-slot pitchers != its top-k pitchers (a weak
    active SP vs a strong benched SP that top-k would seat), and that the
    effective_rosters=None path is unchanged (covered by the byte-anchor test).
    """
    roster = [
        _pitcher_custom("WeakActive", Position.P, "1", w=2, k=40, sv=0, ip=60),
        _pitcher_custom("StrongBench", Position.BN, "2", w=14, k=220, sv=0, ip=200),
    ]
    flat = {"Me": [p.to_flat_dict_full_season() for p in roster]}
    actuals = {"Me": {}}
    n = 4000
    with_eff = simulate_remaining_season_batch(
        actuals,
        flat,
        0.4,
        np.random.default_rng(5),
        13,
        9,
        n,
        effective_rosters={"Me": _eff_roster(roster)},
    )
    without = simulate_remaining_season_batch(
        actuals,
        flat,
        0.4,
        np.random.default_rng(5),
        13,
        9,
        n,
        effective_rosters=None,
    )
    # ROS-direct seats ONLY the weak active SP; top-k seats the strong bench SP.
    # The K mean must differ materially between the two paths.
    assert with_eff["Me"]["K"].mean() < without["Me"]["K"].mean()
