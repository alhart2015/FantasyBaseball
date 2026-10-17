"""The conftest guard keeping tests off the machine's own data files (#496)."""

import pytest

from fantasy_baseball.data import db, kv_store


def test_every_test_gets_its_own_data_files(tmp_path):
    own = tmp_path / "_local_files"
    assert own / "fantasy.db" == db.DB_PATH
    assert own / "local.db" == kv_store._DEFAULT_LOCAL_DB
    assert kv_store._build_sqlite_kv().path == own / "local.db"


def test_the_fantasy_db_fixture_seeds_this_tests_own_database(fantasy_db, tmp_path):
    hitters, pitchers = db.get_blended_projections(db.get_connection())
    assert fantasy_db == db.DB_PATH and fantasy_db.is_relative_to(tmp_path)
    assert len(hitters) > 500 and len(pitchers) > 500


@pytest.mark.real_local_files
def test_the_marker_sees_the_real_paths(tmp_path):
    assert not db.DB_PATH.is_relative_to(tmp_path)
    assert db.DB_PATH.name == "fantasy.db" and db.DB_PATH.parent.name == "data"
    assert kv_store._DEFAULT_LOCAL_DB.parent == db.DB_PATH.parent
