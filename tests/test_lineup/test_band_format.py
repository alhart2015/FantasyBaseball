# band_class is now keyed on P(helps) thresholds (user-requested change,
# replacing the old +/-1 SD crosses-zero rule). Thresholds are strict:
#   > 0.75  -> "real"      (high confidence the swap helps)
#   < 0.25  -> "downgrade" (high confidence the swap hurts)
#   else    -> "coin-flip" (not enough signal)
# Boundary values 0.75 and 0.25 fall into coin-flip (strictly greater/less).
from fantasy_baseball.lineup.band_format import band_class


def test_band_class_high_p_positive_is_real():
    assert band_class(0.80) == "real"


def test_band_class_very_high_p_positive_is_real():
    assert band_class(0.99) == "real"


def test_band_class_low_p_positive_is_downgrade():
    assert band_class(0.20) == "downgrade"


def test_band_class_very_low_p_positive_is_downgrade():
    assert band_class(0.01) == "downgrade"


def test_band_class_mid_p_positive_is_coin_flip():
    assert band_class(0.50) == "coin-flip"


def test_band_class_boundary_75_is_coin_flip():
    # Strictly greater than 0.75 required for "real"; 0.75 itself is coin-flip.
    assert band_class(0.75) == "coin-flip"


def test_band_class_boundary_25_is_coin_flip():
    # Strictly less than 0.25 required for "downgrade"; 0.25 itself is coin-flip.
    assert band_class(0.25) == "coin-flip"


def test_band_class_just_above_75_is_real():
    assert band_class(0.751) == "real"


def test_band_class_just_below_25_is_downgrade():
    assert band_class(0.249) == "downgrade"
