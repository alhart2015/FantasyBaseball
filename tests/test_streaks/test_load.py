"""Tests for streaks DuckDB upserts and existence queries."""

from datetime import date

import duckdb
import pytest

from fantasy_baseball.streaks.data.load import upsert_hitter_games
from fantasy_baseball.streaks.data.schema import init_schema


@pytest.fixture
def conn():
    c = duckdb.connect(":memory:")
    init_schema(c)
    yield c
    c.close()


def _row(player_id=660271, dt=date(2024, 4, 1), hr=1):
    return {
        "player_id": player_id,
        "name": "Mike Trout",
        "team": "LAA",
        "season": 2024,
        "date": dt,
        "pa": 4,
        "ab": 3,
        "h": 1,
        "hr": hr,
        "r": 1,
        "rbi": 2,
        "sb": 0,
        "bb": 1,
        "k": 1,
    }


def test_upsert_hitter_games_inserts_rows(conn):
    upsert_hitter_games(conn, [_row(), _row(dt=date(2024, 4, 2), hr=0)])
    count = conn.execute("SELECT COUNT(*) FROM hitter_games").fetchone()[0]
    assert count == 2


def test_upsert_hitter_games_is_idempotent(conn):
    upsert_hitter_games(conn, [_row(), _row(dt=date(2024, 4, 2))])
    upsert_hitter_games(conn, [_row(), _row(dt=date(2024, 4, 2))])  # same rows
    count = conn.execute("SELECT COUNT(*) FROM hitter_games").fetchone()[0]
    assert count == 2


def test_upsert_hitter_games_updates_on_pk_collision(conn):
    upsert_hitter_games(conn, [_row(hr=1)])
    upsert_hitter_games(conn, [_row(hr=2)])  # same (player_id, date), new hr value
    hr = conn.execute("SELECT hr FROM hitter_games").fetchone()[0]
    assert hr == 2


def test_upsert_hitter_games_empty_list_is_noop(conn):
    upsert_hitter_games(conn, [])
    count = conn.execute("SELECT COUNT(*) FROM hitter_games").fetchone()[0]
    assert count == 0
