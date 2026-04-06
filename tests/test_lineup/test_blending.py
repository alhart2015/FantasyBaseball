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
        assert "juan soto" in result
        assert len(result["juan soto"]) == 2
        assert result["juan soto"][0]["date"] == "2026-04-01"
        assert result["juan soto"][0]["pa"] == 5
        assert result["juan soto"][0]["hr"] == 1

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
        assert "bryan abreu" in result
        games = result["bryan abreu"]
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
