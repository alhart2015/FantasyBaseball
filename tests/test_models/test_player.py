import pytest
import pandas as pd


class TestHitterStats:
    def test_from_dict(self):
        from fantasy_baseball.models.player import HitterStats
        d = {"pa": 650, "ab": 550, "h": 160, "r": 100, "hr": 40, "rbi": 100, "sb": 5, "avg": 0.291}
        stats = HitterStats.from_dict(d)
        assert stats.pa == 650
        assert stats.hr == 40
        assert stats.avg == 0.291

    def test_from_dict_missing_keys_default_to_zero(self):
        from fantasy_baseball.models.player import HitterStats
        stats = HitterStats.from_dict({"hr": 30})
        assert stats.hr == 30
        assert stats.pa == 0
        assert stats.avg == 0

    def test_from_series(self):
        from fantasy_baseball.models.player import HitterStats
        s = pd.Series({"pa": 650, "ab": 550, "h": 160, "r": 100, "hr": 40, "rbi": 100, "sb": 5, "avg": 0.291})
        stats = HitterStats.from_series(s)
        assert stats.hr == 40
        assert stats.avg == 0.291

    def test_to_dict(self):
        from fantasy_baseball.models.player import HitterStats
        stats = HitterStats(pa=650, ab=550, h=160, r=100, hr=40, rbi=100, sb=5, avg=0.291)
        d = stats.to_dict()
        assert d["hr"] == 40
        assert d["avg"] == 0.291
        assert "sgp" not in d  # None sgp excluded

    def test_to_dict_includes_sgp_when_set(self):
        from fantasy_baseball.models.player import HitterStats
        stats = HitterStats(pa=650, ab=550, h=160, r=100, hr=40, rbi=100, sb=5, avg=0.291, sgp=12.5)
        d = stats.to_dict()
        assert d["sgp"] == 12.5

    def test_to_series(self):
        from fantasy_baseball.models.player import HitterStats
        stats = HitterStats(pa=650, ab=550, h=160, r=100, hr=40, rbi=100, sb=5, avg=0.291)
        s = stats.to_series()
        assert s["hr"] == 40
        assert s["player_type"] == "hitter"

    def test_compute_avg_from_components(self):
        from fantasy_baseball.models.player import HitterStats
        stats = HitterStats.from_dict({"h": 150, "ab": 500})
        assert stats.avg == pytest.approx(0.300)


class TestPitcherStats:
    def test_from_dict(self):
        from fantasy_baseball.models.player import PitcherStats
        d = {"ip": 200, "w": 15, "k": 220, "sv": 0, "er": 62, "bb": 40, "h_allowed": 150, "era": 2.79, "whip": 0.95}
        stats = PitcherStats.from_dict(d)
        assert stats.ip == 200
        assert stats.k == 220
        assert stats.era == 2.79

    def test_from_dict_computes_era_whip_from_components(self):
        from fantasy_baseball.models.player import PitcherStats
        stats = PitcherStats.from_dict({"ip": 180, "er": 60, "bb": 40, "h_allowed": 130})
        assert stats.era == pytest.approx(3.0)
        assert stats.whip == pytest.approx((40 + 130) / 180)

    def test_from_series(self):
        from fantasy_baseball.models.player import PitcherStats
        s = pd.Series({"ip": 200, "w": 15, "k": 220, "sv": 0, "er": 62, "bb": 40, "h_allowed": 150, "era": 2.79, "whip": 0.95})
        stats = PitcherStats.from_series(s)
        assert stats.k == 220

    def test_to_dict(self):
        from fantasy_baseball.models.player import PitcherStats
        stats = PitcherStats(ip=200, w=15, k=220, sv=0, er=62, bb=40, h_allowed=150, era=2.79, whip=0.95)
        d = stats.to_dict()
        assert d["k"] == 220
        assert d["era"] == 2.79

    def test_to_series(self):
        from fantasy_baseball.models.player import PitcherStats
        stats = PitcherStats(ip=200, w=15, k=220, sv=0, er=62, bb=40, h_allowed=150, era=2.79, whip=0.95)
        s = stats.to_series()
        assert s["player_type"] == "pitcher"
        assert s["k"] == 220
