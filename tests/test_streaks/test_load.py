"""Tests for streaks DuckDB upserts and existence queries."""

from datetime import date

import duckdb
import pytest

from fantasy_baseball.streaks.data.load import (
    existing_player_seasons,
    existing_statcast_dates,
    upsert_hitter_games,
    upsert_statcast_pa,
)
from fantasy_baseball.streaks.data.schema import init_schema
from fantasy_baseball.streaks.models import HitterGame, HitterStatcastPA


@pytest.fixture
def conn():
    c = duckdb.connect(":memory:")
    init_schema(c)
    yield c
    c.close()


def _row(player_id=660271, dt=date(2024, 4, 1), hr=1, game_pk=None):
    """Default game_pk derives from the date so distinct dates get distinct
    PKs while same-date rows (used by the collision test) still collide."""
    if game_pk is None:
        game_pk = int(dt.strftime("%Y%m%d"))
    return HitterGame(
        player_id=player_id,
        game_pk=game_pk,
        name="Mike Trout",
        team="LAA",
        season=2024,
        date=dt,
        pa=4,
        ab=3,
        h=1,
        hr=hr,
        r=1,
        rbi=2,
        sb=0,
        bb=1,
        k=1,
        b2=0,
        b3=0,
        sf=0,
        hbp=0,
        ibb=0,
        cs=0,
        gidp=0,
        sh=0,
        ci=0,
        is_home=True,
    )


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


def _statcast_row(player_id=660271, dt=date(2024, 4, 1), pa_index=1, event="single"):
    return HitterStatcastPA(
        player_id=player_id,
        date=dt,
        pa_index=pa_index,
        event=event,
        launch_speed=95.5,
        launch_angle=12.0,
        estimated_woba_using_speedangle=0.45,
        barrel=False,
        at_bat_number=pa_index,
        bb_type=None,
        estimated_ba_using_speedangle=0.40,
        hit_distance_sc=None,
    )


def test_upsert_statcast_pa_inserts(conn):
    upsert_statcast_pa(conn, [_statcast_row(pa_index=1), _statcast_row(pa_index=2)])
    count = conn.execute("SELECT COUNT(*) FROM hitter_statcast_pa").fetchone()[0]
    assert count == 2


def test_upsert_statcast_pa_is_idempotent(conn):
    upsert_statcast_pa(conn, [_statcast_row(pa_index=1)])
    upsert_statcast_pa(conn, [_statcast_row(pa_index=1)])
    count = conn.execute("SELECT COUNT(*) FROM hitter_statcast_pa").fetchone()[0]
    assert count == 1


def test_upsert_statcast_pa_handles_null_event(conn):
    upsert_statcast_pa(conn, [_statcast_row(event=None)])
    out = conn.execute("SELECT event FROM hitter_statcast_pa").fetchone()
    assert out[0] is None


def test_upsert_statcast_pa_empty_noop(conn):
    upsert_statcast_pa(conn, [])
    count = conn.execute("SELECT COUNT(*) FROM hitter_statcast_pa").fetchone()[0]
    assert count == 0


def test_existing_player_seasons_empty(conn):
    assert existing_player_seasons(conn) == set()


def test_existing_player_seasons_returns_distinct_pairs(conn):
    upsert_hitter_games(
        conn,
        [
            _row(player_id=660271, dt=date(2024, 4, 1)),
            _row(player_id=660271, dt=date(2024, 4, 2)),  # same (player, season)
            _row(player_id=545361, dt=date(2024, 4, 1)),
        ],
    )
    pairs = existing_player_seasons(conn)
    assert pairs == {(660271, 2024), (545361, 2024)}


def test_existing_statcast_dates_returns_distinct_dates(conn):
    upsert_statcast_pa(
        conn,
        [
            _statcast_row(pa_index=1, dt=date(2024, 4, 1)),
            _statcast_row(pa_index=2, dt=date(2024, 4, 1)),  # same date
            _statcast_row(pa_index=1, dt=date(2024, 4, 2)),
        ],
    )
    dates = existing_statcast_dates(conn)
    assert dates == {date(2024, 4, 1), date(2024, 4, 2)}
