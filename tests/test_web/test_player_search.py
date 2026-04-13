import json
import sqlite3
from unittest.mock import patch

import pytest
from fantasy_baseball.web.season_app import create_app


@pytest.fixture
def client():
    app = create_app()
    app.config["TESTING"] = True
    app.config["SECRET_KEY"] = "test"
    with app.test_client() as c:
        yield c


def _seed_test_db(conn):
    """Insert test projection data into an in-memory SQLite database."""
    conn.executescript("""
        CREATE TABLE IF NOT EXISTS blended_projections (
            year INTEGER, fg_id TEXT, name TEXT, team TEXT, player_type TEXT,
            pa REAL, ab REAL, h REAL, r REAL, hr REAL, rbi REAL, sb REAL, avg REAL,
            w REAL, k REAL, sv REAL, ip REAL, er REAL, bb REAL, h_allowed REAL,
            era REAL, whip REAL, adp REAL,
            PRIMARY KEY (year, fg_id)
        );
        CREATE TABLE IF NOT EXISTS ros_blended_projections (
            year INTEGER, snapshot_date TEXT, fg_id TEXT, name TEXT, team TEXT,
            player_type TEXT,
            pa REAL, ab REAL, h REAL, r REAL, hr REAL, rbi REAL, sb REAL, avg REAL,
            w REAL, k REAL, sv REAL, ip REAL, er REAL, bb REAL, h_allowed REAL,
            era REAL, whip REAL, adp REAL,
            PRIMARY KEY (year, snapshot_date, fg_id)
        );
        CREATE TABLE IF NOT EXISTS game_logs (
            season INTEGER, mlbam_id INTEGER, name TEXT, team TEXT,
            player_type TEXT, date TEXT,
            pa INTEGER, ab INTEGER, h INTEGER, r INTEGER, hr INTEGER,
            rbi INTEGER, sb INTEGER,
            ip REAL, k INTEGER, er INTEGER, bb INTEGER, h_allowed INTEGER,
            w INTEGER, sv INTEGER, gs INTEGER,
            PRIMARY KEY (season, mlbam_id, date)
        );
        CREATE TABLE IF NOT EXISTS weekly_rosters (
            snapshot_date TEXT, week_num INTEGER, team TEXT, slot TEXT,
            player_name TEXT, positions TEXT,
            PRIMARY KEY (snapshot_date, team, slot, player_name)
        );
        CREATE TABLE IF NOT EXISTS positions (
            name TEXT NOT NULL PRIMARY KEY,
            positions TEXT NOT NULL
        );
    """)
    conn.execute(
        "INSERT INTO ros_blended_projections VALUES "
        "(2026, '2026-04-01', '15640', 'Aaron Judge', 'NYY', 'hitter', "
        "600, 500, 145, 95, 38, 92, 7, 0.290, NULL, NULL, NULL, NULL, NULL, NULL, NULL, NULL, NULL, 5.0)"
    )
    conn.execute(
        "INSERT INTO blended_projections VALUES "
        "(2026, '15640', 'Aaron Judge', 'NYY', 'hitter', "
        "650, 550, 160, 110, 45, 120, 5, 0.291, NULL, NULL, NULL, NULL, NULL, NULL, NULL, NULL, NULL, 5.0)"
    )
    conn.execute(
        "INSERT INTO ros_blended_projections VALUES "
        "(2026, '2026-04-01', '28027', 'Gerrit Cole', 'NYY', 'pitcher', "
        "NULL, NULL, NULL, NULL, NULL, NULL, NULL, NULL, 14, 200, 0, 190, 60, 40, 140, 2.84, 0.95, 20.0)"
    )
    conn.commit()


def test_players_page_renders(client):
    resp = client.get("/players")
    assert resp.status_code == 200
    assert b"pos-filter" in resp.data


def test_search_returns_matching_players(client):
    conn = sqlite3.connect(":memory:")
    conn.row_factory = sqlite3.Row
    _seed_test_db(conn)

    with patch("fantasy_baseball.web.season_routes._get_search_db") as mock_db:
        mock_db.return_value = conn
        resp = client.get("/api/players/search?q=judge")

    assert resp.status_code == 200
    data = json.loads(resp.data)
    assert len(data) == 1
    assert data[0]["name"] == "Aaron Judge"
    assert data[0]["player_type"] == "hitter"
    assert data[0]["rest_of_season"]["hr"] == 38
    assert data[0]["preseason"]["hr"] == 45


def test_search_requires_min_2_chars(client):
    resp = client.get("/api/players/search?q=j")
    assert resp.status_code == 200
    data = json.loads(resp.data)
    assert data == []


def test_search_no_results(client):
    conn = sqlite3.connect(":memory:")
    conn.row_factory = sqlite3.Row
    _seed_test_db(conn)

    with patch("fantasy_baseball.web.season_routes._get_search_db") as mock_db:
        mock_db.return_value = conn
        resp = client.get("/api/players/search?q=nonexistent")

    data = json.loads(resp.data)
    assert data == []
