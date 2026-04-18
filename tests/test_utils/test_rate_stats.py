"""Tests for rate-stat helpers."""

import pytest

from fantasy_baseball.utils.rate_stats import format_ip


class TestFormatIP:
    @pytest.mark.parametrize(
        "ip,expected",
        [
            (0, "0.0"),
            (1.0, "1.0"),
            (6.0, "6.0"),
            (1 / 3, "0.1"),
            (2 / 3, "0.2"),
            (6 + 1 / 3, "6.1"),
            (6 + 2 / 3, "6.2"),
            (2.3333333, "2.1"),
            (2.6666666, "2.2"),
            (0.9999999, "1.0"),
        ],
    )
    def test_baseball_notation(self, ip, expected):
        assert format_ip(ip) == expected

    def test_none_returns_em_dash(self):
        assert format_ip(None) == "\u2014"

    def test_rolls_over_on_near_third_boundary(self):
        # 5.999 rounds to 18 outs → 6.0, not 5.3
        assert format_ip(5.9999) == "6.0"
