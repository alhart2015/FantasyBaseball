import sqlite3
import pytest
from fantasy_baseball.data.db import (
    create_tables,
    get_connection,
    save_spoe_results,
    save_spoe_components,
    load_spoe_components,
    get_completed_spoe_weeks,
    get_spoe_results,
)


def test_spoe_results_table_exists():
    conn = get_connection(":memory:")
    create_tables(conn)
    tables = {r[0] for r in conn.execute(
        "SELECT name FROM sqlite_master WHERE type='table'"
    ).fetchall()}
    assert "spoe_results" in tables
    conn.close()


def test_spoe_components_table_exists():
    conn = get_connection(":memory:")
    create_tables(conn)
    tables = {r[0] for r in conn.execute(
        "SELECT name FROM sqlite_master WHERE type='table'"
    ).fetchall()}
    assert "spoe_components" in tables
    conn.close()


def test_save_and_load_spoe_results():
    conn = get_connection(":memory:")
    create_tables(conn)
    results = [
        {"team": "Team A", "category": "R",
         "projected_stat": 50.0, "actual_stat": 60.0,
         "projected_pts": 4.0, "actual_pts": 7.0, "spoe": 3.0},
        {"team": "Team A", "category": "total",
         "projected_stat": None, "actual_stat": None,
         "projected_pts": 40.0, "actual_pts": 45.0, "spoe": 5.0},
    ]
    save_spoe_results(conn, 2026, "2026-03-31", results)
    rows = get_spoe_results(conn, 2026, "2026-03-31")
    assert len(rows) == 2
    assert rows[0]["spoe"] == 3.0
    conn.close()


def test_save_and_load_spoe_components():
    conn = get_connection(":memory:")
    create_tables(conn)
    components = {
        "Team A": {"H": 15.0, "AB": 55.0, "R": 8.0, "IP": 12.0},
        "Team B": {"H": 12.0, "AB": 48.0, "R": 6.0, "IP": 10.0},
    }
    save_spoe_components(conn, 2026, "2026-03-31", components)
    loaded = load_spoe_components(conn, 2026, "2026-03-31")
    assert loaded["Team A"]["H"] == pytest.approx(15.0)
    assert loaded["Team B"]["IP"] == pytest.approx(10.0)
    conn.close()


def test_get_completed_spoe_weeks():
    conn = get_connection(":memory:")
    create_tables(conn)
    results = [
        {"team": "Team A", "category": "total",
         "projected_stat": None, "actual_stat": None,
         "projected_pts": 40.0, "actual_pts": 45.0, "spoe": 5.0},
    ]
    save_spoe_results(conn, 2026, "2026-03-31", results)
    save_spoe_results(conn, 2026, "2026-04-07", results)
    weeks = get_completed_spoe_weeks(conn, 2026)
    assert weeks == {"2026-03-31", "2026-04-07"}
    conn.close()


def test_save_spoe_results_is_idempotent():
    conn = get_connection(":memory:")
    create_tables(conn)
    results = [
        {"team": "Team A", "category": "R",
         "projected_stat": 50.0, "actual_stat": 60.0,
         "projected_pts": 4.0, "actual_pts": 7.0, "spoe": 3.0},
    ]
    save_spoe_results(conn, 2026, "2026-03-31", results)
    results[0]["spoe"] = 4.0
    save_spoe_results(conn, 2026, "2026-03-31", results)
    rows = get_spoe_results(conn, 2026, "2026-03-31")
    assert len(rows) == 1
    assert rows[0]["spoe"] == 4.0
    conn.close()


def test_save_spoe_components_is_idempotent():
    conn = get_connection(":memory:")
    create_tables(conn)
    components = {"Team A": {"H": 15.0}}
    save_spoe_components(conn, 2026, "2026-03-31", components)
    components["Team A"]["H"] = 20.0
    save_spoe_components(conn, 2026, "2026-03-31", components)
    loaded = load_spoe_components(conn, 2026, "2026-03-31")
    assert loaded["Team A"]["H"] == pytest.approx(20.0)
    conn.close()
