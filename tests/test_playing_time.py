"""Tests for the shared playing-time model lookup (utils/playing_time.py)."""

from itertools import pairwise
from typing import ClassVar

import numpy as np
import pytest

from fantasy_baseball.models.player import PlayerType
from fantasy_baseball.utils.constants import (
    PLAYING_TIME_CURVES,
    PLAYING_TIME_SHAPE,
    QUANTILE_LEVELS,
)
from fantasy_baseball.utils.playing_time import (
    playing_time_params,
    playing_time_shape,
    scale_from_uniform,
)


class TestPlayingTimeParams:
    def test_hitter_band_center_returns_exact_values(self):
        # First hitter band center from PLAYING_TIME_CURVES.
        band = PLAYING_TIME_CURVES["hitters"][0]
        mean_scale, cv_pt = playing_time_params(PlayerType.HITTER, band["vol"])
        assert mean_scale == pytest.approx(band["mean_scale"])
        assert cv_pt == pytest.approx(band["cv_pt"])

    def test_interpolates_between_band_centers(self):
        lo, hi = PLAYING_TIME_CURVES["hitters"][0], PLAYING_TIME_CURVES["hitters"][1]
        mid_vol = (lo["vol"] + hi["vol"]) / 2
        mean_scale, cv_pt = playing_time_params(PlayerType.HITTER, mid_vol)
        # Strictly between the two band values (mean rises, cv falls with volume).
        assert lo["mean_scale"] < mean_scale < hi["mean_scale"]
        assert hi["cv_pt"] < cv_pt < lo["cv_pt"]

    def test_clamps_below_min_volume(self):
        band = PLAYING_TIME_CURVES["hitters"][0]
        mean_scale, cv_pt = playing_time_params(PlayerType.HITTER, 50.0)
        assert mean_scale == pytest.approx(band["mean_scale"])
        assert cv_pt == pytest.approx(band["cv_pt"])

    def test_clamps_above_max_volume(self):
        band = PLAYING_TIME_CURVES["hitters"][-1]
        mean_scale, cv_pt = playing_time_params(PlayerType.HITTER, 5000.0)
        assert mean_scale == pytest.approx(band["mean_scale"])
        assert cv_pt == pytest.approx(band["cv_pt"])

    def test_nan_volume_maps_to_lowest_band(self):
        # Bad/missing data must land on the conservative (lowest-volume) band,
        # not slip through NaN comparisons to the best band.
        low = PLAYING_TIME_CURVES["hitters"][0]
        mean_scale, cv_pt = playing_time_params(PlayerType.HITTER, float("nan"))
        assert mean_scale == pytest.approx(low["mean_scale"])
        assert cv_pt == pytest.approx(low["cv_pt"])

    def test_pitcher_high_ip_uses_sp_curve(self):
        band = PLAYING_TIME_CURVES["SP"][2]  # vol 147.1
        mean_scale, cv_pt = playing_time_params(PlayerType.PITCHER, band["vol"])
        assert mean_scale == pytest.approx(band["mean_scale"])
        assert cv_pt == pytest.approx(band["cv_pt"])

    def test_pitcher_low_ip_uses_rp_curve(self):
        band = PLAYING_TIME_CURVES["RP"][0]  # vol 48.3
        mean_scale, cv_pt = playing_time_params(PlayerType.PITCHER, band["vol"])
        assert mean_scale == pytest.approx(band["mean_scale"])
        assert cv_pt == pytest.approx(band["cv_pt"])

    def test_sp_rp_split_at_threshold(self):
        # Just below 100 IP -> RP curve (clamped to its top band); at/above -> SP.
        rp_top = PLAYING_TIME_CURVES["RP"][-1]
        sp_bottom = PLAYING_TIME_CURVES["SP"][0]
        rp_mean, _ = playing_time_params(PlayerType.PITCHER, 99.0)
        sp_mean, _ = playing_time_params(PlayerType.PITCHER, 100.0)
        assert rp_mean == pytest.approx(rp_top["mean_scale"])
        assert sp_mean == pytest.approx(sp_bottom["mean_scale"])

    def test_monotonic_within_hitter_curve(self):
        m_low, cv_low = playing_time_params(PlayerType.HITTER, 400.0)
        m_high, cv_high = playing_time_params(PlayerType.HITTER, 600.0)
        assert m_high >= m_low  # more projected PA -> smaller haircut
        assert cv_high <= cv_low  # more projected PA -> tighter

    def test_accepts_string_player_type(self):
        from_enum = playing_time_params(PlayerType.HITTER, 500.0)
        from_str = playing_time_params("hitter", 500.0)
        assert from_enum == from_str


class TestPlayingTimeShape:
    """The empirical standardized-z ladder lookup (the distribution SHAPE)."""

    def test_band_center_returns_stored_ladder(self):
        band = PLAYING_TIME_SHAPE["hitters"][0]
        ladder = playing_time_shape(PlayerType.HITTER, band["vol"])
        assert ladder == pytest.approx(band["z"])

    def test_ladder_length_matches_quantile_levels(self):
        ladder = playing_time_shape(PlayerType.HITTER, 500.0)
        assert len(ladder) == len(QUANTILE_LEVELS)

    def test_ladder_is_nondecreasing(self):
        ladder = playing_time_shape(PlayerType.HITTER, 500.0)
        assert all(b >= a for a, b in pairwise(ladder))

    def test_clamps_below_min_volume(self):
        band = PLAYING_TIME_SHAPE["hitters"][0]
        assert playing_time_shape(PlayerType.HITTER, 50.0) == pytest.approx(band["z"])

    def test_low_ip_uses_rp_ladder(self):
        band = PLAYING_TIME_SHAPE["RP"][0]
        assert playing_time_shape(PlayerType.PITCHER, band["vol"]) == pytest.approx(band["z"])

    def test_interpolates_between_bands_elementwise(self):
        lo = PLAYING_TIME_SHAPE["hitters"][0]
        hi = PLAYING_TIME_SHAPE["hitters"][1]
        mid = playing_time_shape(PlayerType.HITTER, (lo["vol"] + hi["vol"]) / 2)
        for zlo, zmid, zhi in zip(lo["z"], mid, hi["z"], strict=True):
            assert min(zlo, zhi) <= zmid <= max(zlo, zhi)


class TestScaleFromUniform:
    """Pure uniform-draw -> realized PA/IP multiplier (the new MC sampler core)."""

    _LADDER: ClassVar[list[float]] = [-2.0, -1.5, -1.0, -0.5, 0.0, 0.5, 1.0, 1.5, 2.0]

    def test_median_draw_uses_p50_z(self):
        # u = 0.50 is a quantile knot, so z = the p50 ladder entry (0.0 here).
        s = scale_from_uniform(0.9, 0.2, self._LADDER, 0.5, 1.0)
        assert s == pytest.approx(0.9)

    def test_fraction_remaining_zero_returns_one(self):
        # Nothing left to play -> realized == projected regardless of the draw.
        for u in (0.01, 0.3, 0.99):
            assert scale_from_uniform(0.8, 0.4, self._LADDER, u, 0.0) == pytest.approx(1.0)

    def test_monotonic_in_u(self):
        xs = [scale_from_uniform(0.9, 0.3, self._LADDER, u, 1.0) for u in np.linspace(0, 1, 25)]
        assert all(b >= a for a, b in pairwise(xs))

    def test_never_negative(self):
        deep_down = [-5.0, -4.0, -3.0, -2.0, -1.0, 0.0, 1.0, 2.0, 3.0]
        assert scale_from_uniform(0.5, 0.4, deep_down, 0.0, 1.0) >= 0.0

    def test_full_time_hitter_ceiling_is_realistic_not_2x(self):
        # THE regression for the over-performance bug: a full-time hitter's
        # top draw lands near full health (~1.16), NOT the old flat 2.0 clip.
        crv = PLAYING_TIME_CURVES["hitters"][-1]
        z = PLAYING_TIME_SHAPE["hitters"][-1]["z"]
        s = scale_from_uniform(crv["mean_scale"], crv["cv_pt"], z, 0.999, 1.0)
        assert 1.0 < s < 1.25

    def test_reliever_role_change_upside_survives(self):
        # A low-IP reliever CAN spike well above a hitter's ceiling (closer role).
        crv = PLAYING_TIME_CURVES["RP"][0]
        z = PLAYING_TIME_SHAPE["RP"][0]["z"]
        s = scale_from_uniform(crv["mean_scale"], crv["cv_pt"], z, 0.999, 1.0)
        assert s > 1.5
