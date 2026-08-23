"""The startup allowlist over ``--kv-path`` in ``scripts/run_manual_refresh.py``.

The guard used to be a denylist of ONE filename: refuse ``local.db``, accept
everything else. Everything else included ``data/fantasy.db`` -- 25 MB of real
projections, draft results, weekly rosters and game logs
(``fantasy_baseball.data.db.DB_PATH``) -- which a manual run would have opened
and grown ``kv``/``hash_kv`` tables inside. These tests pin the replacement:

* the two protected databases are refused BY NAME, wherever they sit, and the
  names in the script are checked against the constants they mirror rather than
  re-typed here (the script cannot import ``fantasy_baseball`` -- doing so
  before ``main()`` sets ``FANTASY_LOCAL_KV_PATH`` would bind the KV singleton
  to the Yahoo baseline, which is the invariant
  ``test_run_manual_refresh.py`` enforces over the AST);
* an existing file is adopted only if it ALREADY has the exact
  ``kv``/``hash_kv`` shape ``kv_store.SqliteKVStore`` creates -- and that
  expected shape is itself compared against a store built by the real class, so
  a schema change breaks a test instead of a guard;
* a path that does not exist yet is allowed, because ``--dry-run`` runs before
  bootstrap and the live path refuses a missing store further down;
* the inspection is read-only and leaves no sidecars, so refusing a database
  does not modify it -- which is the property the whole guard is for.
"""

from __future__ import annotations

import gc
import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

import run_manual_refresh as drv


def _sidecars(directory: Path) -> list[str]:
    """SQLite sidecar files under ``directory``, sorted."""
    return sorted(p.name for p in directory.glob("*.db-*"))


def _dry_run_argv(kv_path: Path | str) -> list[str]:
    """Enough argv to reach the guard. The guard refuses before any file load."""
    return ["--dry-run", "--kv-path", str(kv_path)]


# --------------------------------------------------------------------------
# The protected names are the real ones, not remembered ones
# --------------------------------------------------------------------------


def test_protected_names_mirror_the_constants_they_stand_in_for():
    """The script hard-codes two filenames; here is where that coupling is pinned."""
    from fantasy_baseball.data.db import DB_PATH
    from fantasy_baseball.data.kv_store import _DEFAULT_LOCAL_DB

    assert DB_PATH.name == drv.APP_DB_NAME
    assert _DEFAULT_LOCAL_DB.name == drv.BASELINE_DB_NAME
    assert set(drv.PROTECTED_DBS) == {DB_PATH.name, _DEFAULT_LOCAL_DB.name}


def test_the_expected_kv_shape_is_what_sqlite_kv_store_really_creates(tmp_path):
    """A drift in ``SqliteKVStore``'s schema must fail here, not in the field."""
    from fantasy_baseball.data.kv_store import SqliteKVStore

    path = tmp_path / "shape.db"
    store = SqliteKVStore(path)
    store.set("cache:standings", "{}")
    store.hset("job_log:manual:1", "step", "done")
    del store
    gc.collect()

    assert drv._kv_store_shape(path) == drv.KV_STORE_SCHEMA


# --------------------------------------------------------------------------
# Rejections
# --------------------------------------------------------------------------


def test_refuses_the_application_database_by_its_real_path(capsys, monkeypatch):
    """The finding that motivated the allowlist: data/fantasy.db was accepted."""
    from fantasy_baseball.data.db import DB_PATH

    monkeypatch.delenv("FANTASY_LOCAL_KV_PATH", raising=False)
    monkeypatch.delenv("FB_SKIP_YAHOO", raising=False)

    assert drv.main(_dry_run_argv(DB_PATH)) == drv.RC_REFUSED
    out = capsys.readouterr().out
    assert "REFUSING TO RUN" in out
    assert "application database" in out
    assert str(DB_PATH.resolve()) in out
    # Refused BEFORE _activate_manual_environment: nothing was pointed anywhere.
    assert "FANTASY_LOCAL_KV_PATH" not in drv.os.environ
    assert "FB_SKIP_YAHOO" not in drv.os.environ


def test_refuses_a_file_named_like_the_application_database_anywhere(capsys, tmp_path):
    """By bare filename, not only at the repo path: a copy is just as wrong."""
    from fantasy_baseball.data.db import DB_PATH

    elsewhere = tmp_path / DB_PATH.name
    assert drv.main(_dry_run_argv(elsewhere)) == drv.RC_REFUSED
    assert "application database" in capsys.readouterr().out


def test_the_baseline_refusal_still_names_the_yahoo_baseline(capsys, tmp_path):
    """The allowlist did not lose the message the older guard printed."""
    from fantasy_baseball.data.kv_store import _DEFAULT_LOCAL_DB

    assert drv.main(_dry_run_argv(_DEFAULT_LOCAL_DB)) == drv.RC_REFUSED
    assert "Yahoo baseline" in capsys.readouterr().out
    assert drv.main(_dry_run_argv(tmp_path / _DEFAULT_LOCAL_DB.name)) == drv.RC_REFUSED
    assert "Yahoo baseline" in capsys.readouterr().out


def test_refuses_an_existing_application_shaped_database_by_shape(capsys, tmp_path):
    """The same schema under a name nobody protected is still not a KV store."""
    from fantasy_baseball.data.db import create_tables, get_connection

    path = tmp_path / "someone_elses.db"
    conn = get_connection(path)
    create_tables(conn)
    conn.close()

    assert drv.main(_dry_run_argv(path)) == drv.RC_REFUSED
    out = capsys.readouterr().out
    assert "is not a KV store" in out
    assert "raw_projections" in out
    assert "bootstrap_manual_kv.py" in out


def test_refusing_a_database_does_not_read_or_write_it(tmp_path):
    """The inspection is read-only AND sidecar-free: refusal leaves no trace."""
    from fantasy_baseball.data.db import create_tables, get_connection

    path = tmp_path / "untouched.db"
    conn = get_connection(path)
    create_tables(conn)
    conn.close()
    before = (path.stat().st_size, path.stat().st_mtime_ns, path.read_bytes())
    assert _sidecars(tmp_path) == []

    assert drv.main(_dry_run_argv(path)) == drv.RC_REFUSED

    assert (path.stat().st_size, path.stat().st_mtime_ns, path.read_bytes()) == before
    assert _sidecars(tmp_path) == []


def test_refuses_an_empty_file_the_operator_touched(capsys, tmp_path):
    """``touch data/manual2.db`` is not a KV store; sqlite would call it a valid one."""
    path = tmp_path / "touched.db"
    path.touch()
    assert drv.main(_dry_run_argv(path)) == drv.RC_REFUSED
    assert "not a SQLite database" in capsys.readouterr().out


def test_refuses_a_non_sqlite_file_wearing_a_db_suffix(capsys, tmp_path):
    path = tmp_path / "notreally.db"
    path.write_text("snapshot_date: 2026-08-22\n", encoding="utf-8")
    assert drv.main(_dry_run_argv(path)) == drv.RC_REFUSED
    assert "not a SQLite database" in capsys.readouterr().out


def test_refuses_a_path_that_is_not_named_like_a_store(capsys, tmp_path):
    """A mistyped --kv-path pointing at a transcription must not become a store."""
    path = tmp_path / "rosters.yaml"
    path.write_text("teams: []\n", encoding="utf-8")
    assert drv.main(_dry_run_argv(path)) == drv.RC_REFUSED
    out = capsys.readouterr().out
    assert "not named like a SQLite store" in out
    assert path.read_text(encoding="utf-8") == "teams: []\n"


def test_refuses_a_directory(capsys, tmp_path):
    path = tmp_path / "store.db"
    path.mkdir()
    assert drv.main(_dry_run_argv(path)) == drv.RC_REFUSED
    assert "is a directory" in capsys.readouterr().out


@pytest.mark.parametrize("name", [n.upper() for n in drv.PROTECTED_DBS])
def test_protected_names_are_matched_case_insensitively(tmp_path, name):
    assert drv.main(_dry_run_argv(tmp_path / name)) == drv.RC_REFUSED


# --------------------------------------------------------------------------
# Acceptances -- an allowlist that refuses everything is not a guard
# --------------------------------------------------------------------------


def test_accepts_a_brand_new_path_the_operator_names(tmp_path):
    """--dry-run runs before bootstrap; main() refuses a missing store later."""
    target = tmp_path / "manual.db"
    assert drv._kv_path_rejection(target) is None
    assert drv._guard_environment(target) == drv.RC_OK
    assert not target.exists()


@pytest.mark.parametrize("suffix", drv.KV_STORE_SUFFIXES)
def test_accepts_every_advertised_store_suffix(tmp_path, suffix):
    assert drv._kv_path_rejection(tmp_path / f"manual{suffix}") is None


def test_accepts_an_existing_kv_store(tmp_path):
    from fantasy_baseball.data.kv_store import SqliteKVStore

    path = tmp_path / "manual.db"
    store = SqliteKVStore(path)
    store.set("cache:standings", "{}")
    del store
    gc.collect()

    assert drv._kv_path_rejection(path) is None
    assert drv._guard_environment(path) == drv.RC_OK


def test_accepts_a_kv_store_another_process_still_holds_open(tmp_path):
    """A live store keeps its schema in an uncheckpointed -wal.

    The zero-side-effect probe cannot see through that and reports NO tables,
    so without the ``-wal`` retry in ``_kv_store_shape`` a perfectly good store
    would be refused whenever the dashboard happened to have it open.
    """
    from fantasy_baseball.data.kv_store import SqliteKVStore

    path = tmp_path / "manual.db"
    store = SqliteKVStore(path)  # deliberately still open below
    store.set("cache:standings", "{}")

    assert Path(str(path) + "-wal").exists()
    assert drv._read_sqlite_schema(path, immutable=True) == {}
    assert drv._kv_path_rejection(path) is None

    del store
    gc.collect()


def test_the_baseline_is_refused_by_name_even_though_its_shape_is_valid(tmp_path):
    """Why the name rule exists: local.db passes the shape test perfectly."""
    from fantasy_baseball.data.kv_store import SqliteKVStore

    lookalike = tmp_path / drv.BASELINE_DB_NAME
    store = SqliteKVStore(lookalike)
    store.set("cache:standings", "{}")
    del store
    gc.collect()

    assert drv._kv_store_shape(lookalike) == drv.KV_STORE_SCHEMA
    assert drv._kv_path_rejection(lookalike) is not None


# --------------------------------------------------------------------------
# The real files on this box, when they are present
# --------------------------------------------------------------------------


def _assert_refused_and_untouched(path: Path) -> None:
    """Both real databases are gitignored, so a fresh clone may not have them."""
    if not path.exists():
        pytest.skip(f"{path} is not present on this machine")

    before = (path.stat().st_size, path.stat().st_mtime_ns)
    sidecars_before = _sidecars(path.parent)

    assert drv._guard_environment(path.resolve()) == drv.RC_REFUSED

    assert (path.stat().st_size, path.stat().st_mtime_ns) == before
    assert _sidecars(path.parent) == sidecars_before


def test_the_real_application_database_is_refused_and_untouched():
    from fantasy_baseball.data.db import DB_PATH

    _assert_refused_and_untouched(DB_PATH)


def test_the_real_baseline_store_is_refused_and_untouched():
    from fantasy_baseball.data.kv_store import _DEFAULT_LOCAL_DB

    _assert_refused_and_untouched(_DEFAULT_LOCAL_DB)
