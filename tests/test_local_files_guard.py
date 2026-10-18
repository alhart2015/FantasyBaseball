"""The conftest guard keeping tests off the machine's own data files (#496)."""

import json

import pandas as pd
import pytest

from fantasy_baseball.data import db, kv_store
from fantasy_baseball.data.kv_sync import local_destination

from ._fantasy_db import BOARD_FIXTURE
from .conftest import _REAL_LOCAL_FILES, own_fantasy_db_path


def _paths():
    """fantasy.db, the default KV path, and the KV path in use -- resolved without
    opening the store, which would create the file (the real one, under the marker)."""
    return db.DB_PATH, kv_store._DEFAULT_LOCAL_DB, local_destination()


def test_every_test_gets_its_own_data_files(_local_files_root):
    fantasy, default_kv, kv = _paths()
    assert fantasy.parent.parent == _local_files_root and fantasy.name == "fantasy.db"
    assert kv == default_kv.resolve() == (fantasy.parent / "local.db").resolve()
    assert fantasy != _REAL_LOCAL_FILES["db_path"]
    assert default_kv != _REAL_LOCAL_FILES["default_kv"]


@pytest.fixture(scope="module")
def _module_paths():
    return _paths()


def test_wider_fixtures_never_see_the_real_files(_module_paths, _local_files_root):
    """Module/class/session fixtures set up before the per-test guard runs."""
    for path in _module_paths:
        assert path.parent.parent == _local_files_root
    assert _module_paths[0] != _REAL_LOCAL_FILES["db_path"]
    assert _module_paths[1] != _REAL_LOCAL_FILES["default_kv"]


def test_the_fantasy_db_fixture_seeds_this_tests_own_database(fantasy_db, _local_files_root):
    conn = db.get_connection()
    try:
        hitters, pitchers = db.get_blended_projections(conn)
    finally:
        conn.close()
    assert fantasy_db == db.DB_PATH and fantasy_db.parent.parent == _local_files_root
    assert len(hitters) > 500 and len(pitchers) > 500


def test_the_fantasy_db_rebuilds_the_fixture_board(fantasy_db):
    """Every fixture player comes back with its own positions, including players who
    share a name (one positions row per name)."""
    from fantasy_baseball.draft.board import build_board_from_frames

    conn = db.get_connection()
    try:
        hitters, pitchers = db.get_blended_projections(conn)
        positions = db.get_positions(conn)
    finally:
        conn.close()
    board, _scale = build_board_from_frames(hitters, pitchers, positions)
    fixture = pd.DataFrame(json.loads(BOARD_FIXTURE.read_text()))
    want = dict(zip(fixture["player_id"], fixture["positions"].map(list), strict=True))
    got = dict(zip(board["player_id"], board["positions"].map(list), strict=True))
    assert got == want


def test_the_fantasy_db_fixture_refuses_a_path_not_its_own(_local_files_root):
    with pytest.raises(RuntimeError, match="not a test's own file"):
        own_fantasy_db_path(_REAL_LOCAL_FILES["db_path"], _local_files_root)


@pytest.mark.real_local_files
def test_the_marker_sees_the_real_paths(_local_files_root):
    fantasy, default_kv, _ = _paths()
    assert fantasy == _REAL_LOCAL_FILES["db_path"]
    assert fantasy.name == "fantasy.db" and fantasy.parent.name == "data"
    assert default_kv == _REAL_LOCAL_FILES["default_kv"] and default_kv.parent == fantasy.parent
