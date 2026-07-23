from fantasy_baseball.analysis import keeper_value as kv


def test_clamp_ratio_clamps_to_band():
    band = (0.25, 2.5)
    assert kv._clamp_ratio(10.0, 2.0, band, kv.EPS) == 2.5   # 5.0 -> clamp hi
    assert kv._clamp_ratio(1.0, 10.0, band, kv.EPS) == 0.25  # 0.1 -> clamp lo
    assert kv._clamp_ratio(3.0, 4.0, band, kv.EPS) == 0.75   # in-band


def test_clamp_ratio_none_on_tiny_denominator():
    assert kv._clamp_ratio(5.0, 0.0, (0.25, 2.5), kv.EPS) is None


def test_scale_line_scales_scored_fields_and_keeps_flat_on_none():
    anchor = {"r": 100.0, "hr": 30.0, "rbi": 90.0, "sb": 10.0, "ab": 500.0, "avg": 0.280}
    zips_base = {"r": 90.0, "hr": 25.0, "rbi": 80.0, "sb": 0.0, "ab": 450.0, "avg": 0.270}
    zips_y = {"r": 99.0, "hr": 20.0, "rbi": 88.0, "sb": 5.0, "ab": 441.0, "avg": 0.2565}
    out = kv._scale_line(anchor, zips_base, zips_y, "hitter", (0.25, 2.5), kv.EPS)
    assert out["r"] == 100.0 * (99.0 / 90.0)         # 1.10
    assert out["hr"] == 30.0 * (20.0 / 25.0)          # 0.80
    assert round(out["avg"], 4) == round(0.280 * (0.2565 / 0.270), 4)  # rate scaled directly
    assert out["sb"] == 10.0                          # zips_base sb == 0 -> ratio None -> flat
