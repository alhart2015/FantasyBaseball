from fantasy_baseball.lineup.band_format import band_class, band_label


def test_band_class_clears_zero_is_real():
    assert band_class(3.4, 1.1) == "real"  # 3.4 - 1.1 = 2.3 > 0


def test_band_class_straddles_zero_is_coin_flip():
    assert band_class(1.9, 2.3) == "coin-flip"  # 1.9 - 2.3 < 0 < 1.9 + 2.3


def test_band_class_below_zero_is_downgrade():
    assert band_class(-2.0, 1.0) == "downgrade"  # -2.0 + 1.0 = -1.0 < 0


def test_band_label_format():
    assert band_label(1.9, 2.3) == "+1.9 +/- 2.3"
    assert band_label(-0.6, 1.8) == "-0.6 +/- 1.8"
