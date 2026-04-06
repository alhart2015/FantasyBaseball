import sqlite3
from fantasy_baseball.lineup.blending import load_game_logs_by_name


def _create_game_logs_table(conn):
    conn.execute("""
        CREATE TABLE IF NOT EXISTS game_logs (
            season INTEGER NOT NULL,
            mlbam_id INTEGER NOT NULL,
            name TEXT NOT NULL,
            team TEXT,
            player_type TEXT NOT NULL,
            date TEXT NOT NULL,
            pa INTEGER, ab INTEGER, h INTEGER, r INTEGER, hr INTEGER,
            rbi INTEGER, sb INTEGER,
            ip REAL, k INTEGER, er INTEGER, bb INTEGER, h_allowed INTEGER,
            w INTEGER, sv INTEGER, gs INTEGER,
            PRIMARY KEY (season, mlbam_id, date)
        )
    """)


class TestLoadGameLogsByName:
    def test_groups_hitter_games_by_normalized_name(self):
        conn = sqlite3.connect(":memory:")
        conn.row_factory = sqlite3.Row
        _create_game_logs_table(conn)
        conn.execute(
            "INSERT INTO game_logs (season, mlbam_id, name, player_type, date, "
            "pa, ab, h, r, hr, rbi, sb) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (2026, 1, "Juan Soto", "hitter", "2026-04-01", 5, 4, 2, 1, 1, 2, 0),
        )
        conn.execute(
            "INSERT INTO game_logs (season, mlbam_id, name, player_type, date, "
            "pa, ab, h, r, hr, rbi, sb) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (2026, 1, "Juan Soto", "hitter", "2026-04-02", 4, 3, 1, 0, 0, 0, 1),
        )
        conn.commit()

        result = load_game_logs_by_name(conn, 2026)
        assert "juan soto::hitter" in result
        assert len(result["juan soto::hitter"]) == 2
        assert result["juan soto::hitter"][0]["date"] == "2026-04-01"
        assert result["juan soto::hitter"][0]["pa"] == 5
        assert result["juan soto::hitter"][0]["hr"] == 1

    def test_groups_pitcher_games_with_g_field(self):
        conn = sqlite3.connect(":memory:")
        conn.row_factory = sqlite3.Row
        _create_game_logs_table(conn)
        conn.execute(
            "INSERT INTO game_logs (season, mlbam_id, name, player_type, date, "
            "ip, k, er, bb, h_allowed, w, sv, gs) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (2026, 2, "Bryan Abreu", "pitcher", "2026-04-01",
             1.0, 1, 2, 1, 2, 0, 0, 0),
        )
        conn.commit()

        result = load_game_logs_by_name(conn, 2026)
        assert "bryan abreu::pitcher" in result
        games = result["bryan abreu::pitcher"]
        assert len(games) == 1
        assert games[0]["g"] == 1  # synthesized from row
        assert games[0]["gs"] == 0
        assert games[0]["ip"] == 1.0

    def test_empty_table_returns_empty_dict(self):
        conn = sqlite3.connect(":memory:")
        conn.row_factory = sqlite3.Row
        _create_game_logs_table(conn)

        result = load_game_logs_by_name(conn, 2026)
        assert result == {}


# ---------------------------------------------------------------------------
# blend_player_with_game_logs tests
# ---------------------------------------------------------------------------

from fantasy_baseball.models.player import Player, PlayerType, HitterStats, PitcherStats
from fantasy_baseball.lineup.blending import blend_player_with_game_logs


def _hitter(name, **kwargs):
    return Player(
        name=name,
        player_type=PlayerType.HITTER,
        positions=["OF"],
        ros=HitterStats(
            pa=kwargs.get("pa", 600), ab=kwargs.get("ab", 540),
            h=kwargs.get("h", 150), r=kwargs.get("r", 80),
            hr=kwargs.get("hr", 25), rbi=kwargs.get("rbi", 80),
            sb=kwargs.get("sb", 10), avg=kwargs.get("avg", 0.278),
        ),
    )


def _pitcher(name, **kwargs):
    return Player(
        name=name,
        player_type=PlayerType.PITCHER,
        positions=["RP"],
        ros=PitcherStats(
            ip=kwargs.get("ip", 65.0), w=kwargs.get("w", 4.0),
            k=kwargs.get("k", 70.0), sv=kwargs.get("sv", 20.0),
            er=kwargs.get("er", 20.0), bb=kwargs.get("bb", 22.0),
            h_allowed=kwargs.get("h_allowed", 50.0),
            era=kwargs.get("era", 2.77), whip=kwargs.get("whip", 1.11),
        ),
    )


class TestBlendPlayerWithGameLogs:
    def test_returns_unchanged_player_when_no_logs(self):
        player = _hitter("Test Hitter", hr=25)
        result = blend_player_with_game_logs(player, [], "2026-04-06")
        assert result.ros.hr == 25
        assert result is not player  # should be a copy

    def test_blends_hitter_stats_toward_actuals(self):
        # Projection: .278 AVG, 25 HR
        player = _hitter("Slugger", pa=600, ab=540, h=150, hr=25, avg=0.278)
        # Actuals: 50 PA, much worse — .100 AVG, 0 HR
        logs = [
            {"date": "2026-04-01", "pa": 5, "ab": 5, "h": 0, "r": 0, "hr": 0, "rbi": 0, "sb": 0},
            {"date": "2026-04-02", "pa": 5, "ab": 5, "h": 1, "r": 0, "hr": 0, "rbi": 0, "sb": 0},
            {"date": "2026-04-03", "pa": 5, "ab": 4, "h": 0, "r": 0, "hr": 0, "rbi": 0, "sb": 0},
            {"date": "2026-04-04", "pa": 5, "ab": 5, "h": 0, "r": 0, "hr": 0, "rbi": 0, "sb": 0},
            {"date": "2026-04-05", "pa": 5, "ab": 4, "h": 0, "r": 0, "hr": 0, "rbi": 0, "sb": 0},
        ] * 2  # 50 PA total
        result = blend_player_with_game_logs(player, logs, "2026-04-06")
        # With 50 PA and reliability of 200, actual weight = 50/250 = 20%
        # HR projection should decrease but still be near projection
        assert result.ros.hr < player.ros.hr
        assert result.ros.hr > 0  # not fully actual (which is 0)
        # AVG should be pulled down
        assert result.ros.avg < player.ros.avg

    def test_blends_pitcher_era_toward_actuals(self):
        # Projection: 2.77 ERA
        player = _pitcher("Reliever", ip=65, era=2.77, er=20)
        # Actuals: 5 IP, terrible ERA (9.00)
        logs = [
            {"date": "2026-04-01", "ip": 1.0, "k": 1, "er": 1, "bb": 1,
             "h_allowed": 2, "w": 0, "sv": 0, "gs": 0, "g": 1},
            {"date": "2026-04-02", "ip": 1.0, "k": 0, "er": 1, "bb": 0,
             "h_allowed": 1, "w": 0, "sv": 0, "gs": 0, "g": 1},
            {"date": "2026-04-03", "ip": 1.0, "k": 1, "er": 1, "bb": 1,
             "h_allowed": 2, "w": 0, "sv": 0, "gs": 0, "g": 1},
            {"date": "2026-04-04", "ip": 1.0, "k": 2, "er": 1, "bb": 0,
             "h_allowed": 1, "w": 0, "sv": 0, "gs": 0, "g": 1},
            {"date": "2026-04-05", "ip": 1.0, "k": 1, "er": 1, "bb": 1,
             "h_allowed": 2, "w": 0, "sv": 0, "gs": 0, "g": 1},
        ]
        result = blend_player_with_game_logs(player, logs, "2026-04-06")
        # ERA should be pulled up from 2.77 toward 9.00 but stay closer to projection
        # With 5 IP and reliability of 120: actual_weight = 5/125 = 4%
        assert result.ros.era > player.ros.era
        assert result.ros.era < 9.0

    def test_preserves_player_metadata(self):
        player = _pitcher("Test Guy", ip=65, era=3.00, sv=20)
        player.team = "HOU"
        player.wsgp = 1.5
        result = blend_player_with_game_logs(player, [], "2026-04-06")
        assert result.name == "Test Guy"
        assert result.team == "HOU"
        assert result.positions == ["RP"]
        assert result.player_type == PlayerType.PITCHER

    def test_handles_player_with_no_ros(self):
        player = Player(name="Nobody", player_type=PlayerType.HITTER, positions=["OF"])
        result = blend_player_with_game_logs(player, [], "2026-04-06")
        assert result.ros is None
