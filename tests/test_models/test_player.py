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


class TestRankInfo:
    def test_from_dict(self):
        from fantasy_baseball.models.player import RankInfo
        r = RankInfo.from_dict({"ros": 5, "preseason": 8, "current": 12})
        assert r.ros == 5
        assert r.preseason == 8
        assert r.current == 12

    def test_from_dict_missing_keys(self):
        from fantasy_baseball.models.player import RankInfo
        r = RankInfo.from_dict({"ros": 5})
        assert r.ros == 5
        assert r.preseason is None
        assert r.current is None

    def test_to_dict(self):
        from fantasy_baseball.models.player import RankInfo
        r = RankInfo(ros=5, preseason=8, current=12)
        assert r.to_dict() == {"ros": 5, "preseason": 8, "current": 12}

    def test_empty_rank(self):
        from fantasy_baseball.models.player import RankInfo
        r = RankInfo()
        assert r.ros is None


class TestPlayer:
    def test_from_dict_hitter(self):
        from fantasy_baseball.models.player import Player, HitterStats
        d = {
            "name": "Aaron Judge", "player_type": "hitter",
            "positions": ["OF", "DH"], "team": "NYY",
            "fg_id": "15640", "mlbam_id": 592450,
            "selected_position": "OF", "status": "",
            "wsgp": 12.5,
            "rank": {"ros": 2, "preseason": 1, "current": 3},
            "ros": {"pa": 600, "ab": 500, "h": 145, "r": 95, "hr": 38, "rbi": 92, "sb": 7, "avg": 0.290},
            "preseason": {"pa": 650, "ab": 550, "h": 160, "r": 110, "hr": 45, "rbi": 120, "sb": 5, "avg": 0.291},
        }
        p = Player.from_dict(d)
        assert p.name == "Aaron Judge"
        assert p.player_type == "hitter"
        assert p.fg_id == "15640"
        assert p.mlbam_id == 592450
        assert isinstance(p.ros, HitterStats)
        assert p.ros.hr == 38
        assert isinstance(p.preseason, HitterStats)
        assert p.preseason.hr == 45
        assert p.current is None
        assert p.wsgp == 12.5
        assert p.rank.ros == 2

    def test_from_dict_pitcher(self):
        from fantasy_baseball.models.player import Player, PitcherStats
        d = {
            "name": "Gerrit Cole", "player_type": "pitcher",
            "positions": ["P"], "team": "NYY",
            "ros": {"ip": 190, "w": 14, "k": 200, "sv": 0, "er": 60, "bb": 40, "h_allowed": 140, "era": 2.84, "whip": 0.95},
        }
        p = Player.from_dict(d)
        assert p.player_type == "pitcher"
        assert isinstance(p.ros, PitcherStats)
        assert p.ros.k == 200

    def test_to_dict_roundtrip(self):
        from fantasy_baseball.models.player import Player
        d = {
            "name": "Aaron Judge", "player_type": "hitter",
            "positions": ["OF"], "team": "NYY",
            "fg_id": "15640", "mlbam_id": 592450,
            "wsgp": 12.5,
            "rank": {"ros": 2, "preseason": 1, "current": 3},
            "ros": {"pa": 600, "ab": 500, "h": 145, "r": 95, "hr": 38, "rbi": 92, "sb": 7, "avg": 0.290},
        }
        p = Player.from_dict(d)
        result = p.to_dict()
        assert result["name"] == "Aaron Judge"
        assert result["ros"]["hr"] == 38
        assert result["rank"]["ros"] == 2
        assert result["wsgp"] == 12.5

    def test_from_dict_flat_stats_hitter(self):
        """Player.from_dict handles flat dicts where stats are top-level keys."""
        from fantasy_baseball.models.player import Player
        d = {
            "name": "Aaron Judge", "player_type": "hitter",
            "positions": ["OF"], "team": "NYY",
            "r": 95, "hr": 38, "rbi": 92, "sb": 7, "h": 145, "ab": 500, "pa": 600, "avg": 0.290,
        }
        p = Player.from_dict(d)
        assert p.ros is not None
        assert p.ros.hr == 38

    def test_from_dict_flat_stats_pitcher(self):
        """Player.from_dict handles flat dicts where stats are top-level keys."""
        from fantasy_baseball.models.player import Player
        d = {
            "name": "Gerrit Cole", "player_type": "pitcher",
            "positions": ["P"],
            "ip": 190, "w": 14, "k": 200, "sv": 0, "era": 2.84, "whip": 0.95,
        }
        p = Player.from_dict(d)
        assert p.ros is not None
        assert p.ros.k == 200

    def test_to_series(self):
        from fantasy_baseball.models.player import Player
        d = {
            "name": "Aaron Judge", "player_type": "hitter",
            "positions": ["OF"], "team": "NYY",
            "ros": {"pa": 600, "ab": 500, "h": 145, "r": 95, "hr": 38, "rbi": 92, "sb": 7, "avg": 0.290},
        }
        p = Player.from_dict(d)
        s = p.to_series()
        assert s["name"] == "Aaron Judge"
        assert s["player_type"] == "hitter"
        assert s["hr"] == 38
        assert s["positions"] == ["OF"]


class TestCacheCompatibility:
    def test_to_dict_preserves_nested_ros_and_preseason(self):
        """Verify to_dict includes nested ros and preseason dicts."""
        from fantasy_baseball.models.player import Player, HitterStats, RankInfo
        p = Player(
            name="Aaron Judge",
            player_type="hitter",
            positions=["OF"],
            team="NYY",
            fg_id="15640",
            yahoo_id="12345",
            selected_position="OF",
            ros=HitterStats(pa=600, ab=500, h=145, r=95, hr=38, rbi=92, sb=7, avg=0.290),
            preseason=HitterStats(pa=650, ab=550, h=160, r=110, hr=45, rbi=120, sb=5, avg=0.291),
            wsgp=12.5,
            rank=RankInfo(ros=2, preseason=1, current=3),
            pace={"R": {"actual": 15, "expected": 14, "z_score": 0.5}},
        )
        d = p.to_dict()
        # Core identity
        assert d["name"] == "Aaron Judge"
        assert d["player_type"] == "hitter"
        assert d["player_id"] == "12345"
        # ROS stats in nested dict
        assert d["ros"]["hr"] == 38
        # Preseason in nested dict
        assert d["preseason"]["hr"] == 45
        # wSGP
        assert d["wsgp"] == 12.5
        # Rank
        assert d["rank"]["ros"] == 2
        # Pace stored as "stats"
        assert d["stats"]["R"]["actual"] == 15

    def test_player_from_dict_roundtrip_with_all_fields(self):
        """Full roundtrip: construct Player, serialize, reconstruct, compare."""
        from fantasy_baseball.models.player import Player, HitterStats, RankInfo
        original = Player(
            name="Aaron Judge",
            player_type="hitter",
            positions=["OF"],
            team="NYY",
            fg_id="15640",
            mlbam_id=592450,
            yahoo_id="12345",
            selected_position="OF",
            status="",
            ros=HitterStats(pa=600, ab=500, h=145, r=95, hr=38, rbi=92, sb=7, avg=0.290),
            preseason=HitterStats(pa=650, ab=550, h=160, r=110, hr=45, rbi=120, sb=5, avg=0.291),
            wsgp=12.5,
            rank=RankInfo(ros=2, preseason=1, current=3),
        )
        d = original.to_dict()
        restored = Player.from_dict(d)
        assert restored.name == original.name
        assert restored.player_type == original.player_type
        assert restored.ros.hr == original.ros.hr
        assert restored.preseason.hr == original.preseason.hr
        assert restored.wsgp == original.wsgp
        assert restored.rank.ros == original.rank.ros


class TestSgpComputation:
    def test_hitter_stats_compute_sgp(self):
        from fantasy_baseball.models.player import HitterStats
        stats = HitterStats(pa=650, ab=550, h=160, r=100, hr=40, rbi=100, sb=5, avg=0.291)
        sgp = stats.compute_sgp()
        assert sgp > 0
        assert stats.sgp == sgp  # cached on the instance

    def test_pitcher_stats_compute_sgp(self):
        from fantasy_baseball.models.player import PitcherStats
        stats = PitcherStats(ip=200, w=15, k=220, sv=0, er=62, bb=40, h_allowed=150, era=2.79, whip=0.95)
        sgp = stats.compute_sgp()
        assert sgp > 0
        assert stats.sgp == sgp

    def test_player_compute_wsgp(self):
        from fantasy_baseball.models.player import Player, HitterStats
        p = Player(
            name="Aaron Judge", player_type="hitter",
            ros=HitterStats(pa=650, ab=550, h=160, r=100, hr=40, rbi=100, sb=5, avg=0.291),
        )
        leverage = {"R": 0.1, "HR": 0.1, "RBI": 0.1, "SB": 0.1, "AVG": 0.1,
                    "W": 0.1, "K": 0.1, "SV": 0.1, "ERA": 0.1, "WHIP": 0.1}
        wsgp = p.compute_wsgp(leverage)
        assert wsgp > 0
        assert p.wsgp == wsgp

    def test_player_compute_wsgp_no_ros_returns_zero(self):
        from fantasy_baseball.models.player import Player
        p = Player(name="Unknown", player_type="hitter")
        wsgp = p.compute_wsgp({"R": 0.1, "HR": 0.1, "RBI": 0.1, "SB": 0.1, "AVG": 0.1,
                               "W": 0.1, "K": 0.1, "SV": 0.1, "ERA": 0.1, "WHIP": 0.1})
        assert wsgp == 0.0

    def test_hitter_sgp_matches_calculate_player_sgp(self):
        """Verify our compute_sgp produces same result as the standalone function."""
        from fantasy_baseball.models.player import HitterStats
        from fantasy_baseball.sgp.player_value import calculate_player_sgp
        stats = HitterStats(pa=650, ab=550, h=160, r=100, hr=40, rbi=100, sb=5, avg=0.291)
        our_sgp = stats.compute_sgp()
        standalone_sgp = calculate_player_sgp(stats.to_series())
        assert our_sgp == pytest.approx(standalone_sgp)

    def test_pitcher_sgp_matches_calculate_player_sgp(self):
        from fantasy_baseball.models.player import PitcherStats
        from fantasy_baseball.sgp.player_value import calculate_player_sgp
        stats = PitcherStats(ip=200, w=15, k=220, sv=0, er=62, bb=40, h_allowed=150, era=2.79, whip=0.95)
        our_sgp = stats.compute_sgp()
        standalone_sgp = calculate_player_sgp(stats.to_series())
        assert our_sgp == pytest.approx(standalone_sgp)
