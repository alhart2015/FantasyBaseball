"""Phase 4a tests: sampler-plumbing changes to ``_apply_variance_batch``.

These tests lock the pure-additive contract of the 4a refactor: the new
``VarianceBatch`` return, the always-populated ``frac_missed`` array, the
``suppress_repl`` flag, and the ``pt_mean_fraction`` mean-horizon split -- all
while keeping the DEFAULT path byte-for-byte identical to the pre-4a code.
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
