"""Tests for the shared playing-time model lookup (utils/playing_time.py)."""

import pytest

from fantasy_baseball.models.player import PlayerType
from fantasy_baseball.utils.constants import PLAYING_TIME_CURVES
from fantasy_baseball.utils.playing_time import playing_time_params


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
