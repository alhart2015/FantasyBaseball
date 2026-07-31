import pandas as pd
import pytest

from fantasy_baseball.keepers.projection import (
    RESIDUAL_QUANTILE_LEVELS,
    SGP_FIT,
    SGP_SD_FIT,
    STD_RESIDUAL_QUANTILES,
    expected_sgp,
    probability_top_n,
    sample_outcomes,
    sgp_sd,
)


def test_the_fit_is_monotone_for_both_pools():
    """The band summary showed pitchers at 90-95 out-earning 95-100. A model that
    reproduced that would say a better composite predicts less value."""
    grid = pd.Series([i / 100 for i in range(101)])
    for kind in SGP_FIT:
        assert expected_sgp(grid, kind).is_monotonic_increasing, kind


def test_hitters_project_above_pitchers_at_the_same_composite():
    """PINS THE SHIPPED FIT, not a derivation -- both sides are constants in the
    module under test. Update on refit. The gap is structural: hitters reach four
    counting categories while no pitcher reaches all three of W/K/SV."""
    top = pd.Series([0.95])
    assert expected_sgp(top, "hitter").iloc[0] > expected_sgp(top, "pitcher").iloc[0]


def test_the_error_term_grows_with_the_composite():
    """A single pooled SD would understate uncertainty exactly at the top of the
    board, which is the only part a keeper decision uses."""
    for kind in SGP_SD_FIT:
        out = sgp_sd(pd.Series([0.1, 0.5, 0.95]), kind)
        assert out.is_monotonic_increasing, kind
        assert out.iloc[0] > 0, kind


def test_the_individual_spread_dwarfs_the_gap_it_has_to_resolve():
    """PINS THE SHIPPED FIT. Why adjacent ranks are coin flips: one rank of
    composite is worth far less than the spread on one player's outcome."""
    one_rank = (
        expected_sgp(pd.Series([0.90]), "hitter").iloc[0]
        - (expected_sgp(pd.Series([0.895]), "hitter").iloc[0])
    )
    assert sgp_sd(pd.Series([0.90]), "hitter").iloc[0] > 20 * one_rank


# --- top-N probability ----------------------------------------------------


def _kinds(n, pool="hitter"):
    return pd.Series([pool] * n)


def test_top_n_probabilities_sum_to_the_number_of_slots():
    """Three slots distribute exactly three players' worth of confidence. A useful
    self-check: any bug that double-counts or drops a player breaks this."""
    means = pd.Series([14.0, 13.0, 12.0, 11.0, 10.0, 6.0])
    sds = pd.Series([5.0] * 6)
    out = probability_top_n(means, sds, _kinds(6), 3)
    assert out.sum() == pytest.approx(3.0, abs=1e-9)


def test_a_better_player_is_likelier_to_make_the_cut():
    means = pd.Series([20.0, 12.0, 4.0])
    out = probability_top_n(means, pd.Series([4.0] * 3), _kinds(3), 1)
    assert out.iloc[0] > out.iloc[1] > out.iloc[2]


def test_everyone_makes_it_when_there_are_at_least_as_many_slots_as_players():
    out = probability_top_n(pd.Series([9.0, 3.0]), pd.Series([4.0, 4.0]), _kinds(2), 2)
    assert out.tolist() == [1.0, 1.0]


def test_a_dominant_player_is_near_certain():
    means = pd.Series([60.0, 10.0, 9.0, 8.0])
    out = probability_top_n(means, pd.Series([4.0] * 4), _kinds(4), 1)
    assert out.iloc[0] > 0.999


def test_identical_players_split_the_slots_evenly():
    out = probability_top_n(pd.Series([10.0] * 4), pd.Series([4.0] * 4), _kinds(4), 2)
    assert out.tolist() == pytest.approx([0.5] * 4, abs=0.02)


def test_top_n_rejects_a_nonpositive_slot_count():
    with pytest.raises(ValueError, match="positive"):
        probability_top_n(pd.Series([1.0]), pd.Series([1.0]), _kinds(1), 0)


def test_top_n_is_reproducible_and_seed_sensitive():
    args = (pd.Series([12.0, 11.5, 11.0]), pd.Series([5.0] * 3), _kinds(3), 2)
    assert probability_top_n(*args).tolist() == probability_top_n(*args).tolist()
    shifted = probability_top_n(*args, seed=7)
    assert shifted.tolist() != probability_top_n(*args).tolist()
    assert shifted.sum() == pytest.approx(2.0, abs=1e-9)


def test_top_n_keeps_the_callers_index():
    means = pd.Series([9.0, 8.0], index=["a", "b"])
    out = probability_top_n(means, pd.Series([3.0, 3.0], index=["a", "b"]), _kinds(2), 1)
    assert out.index.tolist() == ["a", "b"]


# --- residual sampling ----------------------------------------------------


def test_residual_quantiles_line_up_with_their_levels_and_are_sorted():
    """`np.interp` silently misreads a mismatched pair rather than raising."""
    assert list(RESIDUAL_QUANTILE_LEVELS) == sorted(RESIDUAL_QUANTILE_LEVELS)
    assert RESIDUAL_QUANTILE_LEVELS[0] == 0.0
    assert RESIDUAL_QUANTILE_LEVELS[-1] == 1.0
    for pool, quantiles in STD_RESIDUAL_QUANTILES.items():
        assert len(quantiles) == len(RESIDUAL_QUANTILE_LEVELS), pool
        assert list(quantiles) == sorted(quantiles), pool


def test_the_sampled_residuals_are_right_skewed_not_normal():
    """SGP is bounded below, so the left tail is thin and the right tail fat. A
    normal would overstate downside and understate the upside a top-N question
    turns on."""
    outcomes = sample_outcomes(pd.Series([0.0]), pd.Series([1.0]), _kinds(1, "pitcher"))
    z = pd.Series(outcomes[0])
    assert z.skew() > 0.5
    assert (z < -1.5).mean() < 0.05  # far below a normal's 0.067
    assert (z > 1.5).mean() > 0.06


def test_each_pool_samples_its_own_residual_shape():
    kinds = pd.Series(["hitter", "pitcher"])
    outcomes = sample_outcomes(pd.Series([0.0, 0.0]), pd.Series([1.0, 1.0]), kinds)
    # The hitter tail reaches further down; the pitcher tail further up.
    assert outcomes[0].min() < outcomes[1].min()
    assert outcomes[1].max() > outcomes[0].max()


def test_sampling_recovers_the_mean_and_spread_it_was_given():
    outcomes = sample_outcomes(pd.Series([10.0]), pd.Series([4.0]), _kinds(1), draws=60_000)
    assert outcomes.mean() == pytest.approx(10.0, abs=0.15)
    assert outcomes.std() == pytest.approx(4.0, rel=0.06)


# --- positional scarcity --------------------------------------------------
def test_exact_ties_still_credit_exactly_top_n():
    """Only reachable at sd == 0, but the sum-to-top_n invariant is documented, so
    it holds unconditionally. Thresholding on the nth value instead of selecting
    n winners would credit every player tied at the cut."""
    kinds = _kinds(4)
    zero_spread = pd.Series([0.0] * 4)
    for means in ([10.0] * 4, [10.0, 9.0, 9.0, 1.0]):
        out = probability_top_n(pd.Series(means), zero_spread, kinds, 2)
        assert out.sum() == pytest.approx(2.0)
