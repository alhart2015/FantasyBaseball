"""Tests for scripts/bootstrap_manual_kv.py.

The property these tests exist to pin is a safety property, not a feature: the
manual pipeline gets its own KV file, seeded from the Yahoo baseline
(``data/local.db``), and the baseline must come out of the operation
bit-for-bit identical. Every test that runs the bootstrap therefore also checks
the source's row counts, its row-level content digest, and the raw bytes of the
``.db`` file.
"""

import hashlib
import shutil
import sqlite3
import subprocess
import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

import bootstrap_manual_kv as boot


def _file_sha256(path: Path) -> str:
    """SHA-256 of the raw bytes of ``path`` (the ``.db`` file only, no sidecars)."""
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _baseline_view(dest: Path, tmp_path: Path) -> tuple[dict[str, int], str]:
    """``(counts, digest)`` for ``dest`` with the manual breadcrumb removed.

    The bootstrap deliberately adds exactly one row the source does not have --
    the provenance stamp that tells every later process this store is manual and
    must not reach prod Upstash (see ``stamp_manual_provenance``). Fidelity of
    the CARRIED-OVER baseline is still the property these tests pin, so they
    compare against the copy with that one row taken back out.
    """
    from fantasy_baseball.manual.seed import PROVENANCE_KEY

    scratch = tmp_path / "baseline_view.db"
    shutil.copyfile(dest, scratch)
    conn = sqlite3.connect(str(scratch))
    try:
        conn.execute("DELETE FROM kv WHERE key = ?", (PROVENANCE_KEY,))
        conn.commit()
    finally:
        conn.close()
    return boot.row_counts(scratch), boot.content_digest(scratch)


class _SourceSnapshot:
    """Everything about a store that must not change: counts, rows, bytes."""

    def __init__(self, path: Path):
        self.path = path
        self.counts = boot.row_counts(path)
        self.digest = boot.content_digest(path)
        self.file_sha256 = _file_sha256(path)
        self.size = path.stat().st_size

    def assert_unchanged(self) -> None:
        assert boot.row_counts(self.path) == self.counts
        assert boot.content_digest(self.path) == self.digest
        assert _file_sha256(self.path) == self.file_sha256
        assert self.path.stat().st_size == self.size


@pytest.fixture
def open_store():
    """Build real ``SqliteKVStore`` instances and close them at teardown.

    The real class is used rather than a hand-rolled schema so the fixture
    inherits the production ``PRAGMA journal_mode=WAL`` / ``isolation_level=None``
    setup -- which is exactly the configuration that makes ``shutil.copy`` of the
    ``.db`` unsafe and ``Connection.backup()`` necessary. Closing matters on
    Windows: an open handle blocks tmp_path cleanup.
    """
    from fantasy_baseball.data.kv_store import SqliteKVStore

    opened = []

    def _open(path):
        store = SqliteKVStore(path)
        opened.append(store)
        return store

    try:
        yield _open
    finally:
        for store in opened:
            # No public close() on SqliteKVStore; it is a process-lifetime singleton
            # in production. Tests need the handle released.
            store._conn.close()


@pytest.fixture
def source_db(tmp_path, open_store):
    """A populated WAL store, with its writer connection LEFT OPEN.

    Left open on purpose: with WAL and no checkpoint, the rows written here live
    in the ``-wal`` sidecar rather than the ``.db`` file, so a copy that only
    grabs the ``.db`` loses them. The backup path must not.
    """
    path = tmp_path / "source" / "local_baseline.db"
    store = open_store(path)
    store.set("blended_projections:hitters", '[{"name": "Juan Soto"}]')
    store.set("cache:positions", '{"Juan Soto::hitter": ["OF"]}')
    store.set("job_log:manual", "expiring", ex=3600)
    store.hset("standings_history", "2026-08-04", '{"teams": []}')
    store.hset("standings_history", "2026-08-11", '{"teams": []}')
    return path


def _run(source: Path, dest: Path, *extra: str) -> int:
    return boot.main(["--source", str(source), "--dest", str(dest), *extra])


# --------------------------------------------------------------------------
# The copy itself
# --------------------------------------------------------------------------


def test_copies_every_row_including_uncheckpointed_wal_frames(source_db, tmp_path, open_store):
    dest = tmp_path / "manual.db"
    # Guard on the premise of the fixture: the rows really are sitting in the WAL.
    wal = Path(str(source_db) + "-wal")
    assert wal.exists() and wal.stat().st_size > 0, "fixture no longer exercises the WAL path"

    assert _run(source_db, dest) == 0

    counts, digest = _baseline_view(dest, tmp_path)
    assert counts == boot.row_counts(source_db)
    assert digest == boot.content_digest(source_db)
    copy = open_store(dest)
    assert copy.get("blended_projections:hitters") == '[{"name": "Juan Soto"}]'
    assert copy.hgetall("standings_history") == {
        "2026-08-04": '{"teams": []}',
        "2026-08-11": '{"teams": []}',
    }


def test_a_naive_file_copy_would_have_lost_rows(source_db, tmp_path):
    """Pins WHY the script uses the backup API rather than shutil.copy."""
    naive = tmp_path / "naive.db"
    shutil.copyfile(source_db, naive)
    try:
        # Either the copy is short some rows, or -- as happens when even the
        # CREATE TABLE is still in the WAL -- it has no schema at all.
        naive_counts = boot.row_counts(naive)
    except sqlite3.OperationalError:
        naive_counts = None
    assert naive_counts != boot.row_counts(source_db)

    faithful = tmp_path / "manual.db"
    assert _run(source_db, faithful) == 0
    assert _baseline_view(faithful, tmp_path)[0] == boot.row_counts(source_db)


def test_the_copy_is_stamped_manual_and_that_is_the_only_row_added(source_db, tmp_path, open_store):
    """The store must announce itself as manual before anything reads it.

    ``manual_store_active`` gates whether a process may serve prod Upstash
    rosters. The seeder stamps this key only at the END of a successful seed, so
    a store that had been bootstrapped but not yet seeded -- the runbook's own
    sequence, and the state a seed interrupted partway leaves behind -- read as
    Yahoo mode and spliced month-stale prod rosters into a manual page.
    """
    import json

    from fantasy_baseball.manual.seed import MANUAL_SOURCE, PROVENANCE_KEY

    dest = tmp_path / "manual.db"
    assert _run(source_db, dest) == 0

    stamp = json.loads(open_store(dest).get(PROVENANCE_KEY))
    assert stamp["source"] == MANUAL_SOURCE
    assert stamp["yahoo"] is False
    assert stamp["seeded"] is False

    # Exactly one row more than the source, and it is that one.
    assert boot.row_counts(dest)["kv"] == boot.row_counts(source_db)["kv"] + 1
    assert _baseline_view(dest, tmp_path)[1] == boot.content_digest(source_db)


def test_source_is_untouched_by_the_copy(source_db, tmp_path):
    before = _SourceSnapshot(source_db)
    assert _run(source_db, tmp_path / "manual.db") == 0
    before.assert_unchanged()


def test_writes_to_the_copy_do_not_reach_the_source(source_db, tmp_path, open_store):
    """The two stores are separate files -- that separation IS the isolation."""
    dest = tmp_path / "manual.db"
    assert _run(source_db, dest) == 0
    before = _SourceSnapshot(source_db)

    manual = open_store(dest)
    manual.set("cache:roster_audit", '{"drops": []}')
    manual.hset("weekly_rosters_history", "2026-08-22", "[]")

    before.assert_unchanged()
    assert manual.get("cache:roster_audit") == '{"drops": []}'


def test_source_handle_is_read_only(source_db):
    conn = boot.open_readonly(source_db)
    try:
        with pytest.raises(sqlite3.OperationalError, match=r"readonly|read-only|read only"):
            conn.execute("INSERT INTO kv(key, value) VALUES('x', 'y')")
    finally:
        conn.close()


def test_prints_both_resolved_absolute_paths_first(source_db, tmp_path, capsys):
    dest = tmp_path / "manual.db"
    assert _run(source_db, dest) == 0
    out = capsys.readouterr().out
    assert str(boot.resolve_path(source_db)) in out
    assert str(boot.resolve_path(dest)) in out
    # Both paths are printed before any row is read or written.
    assert out.index(str(boot.resolve_path(dest))) < out.index("source rows:")


# --------------------------------------------------------------------------
# Refusals -- each one must leave the filesystem alone
# --------------------------------------------------------------------------


def test_refuses_when_destination_exists_without_force(source_db, tmp_path, open_store):
    dest = tmp_path / "manual.db"
    existing = open_store(dest)
    existing.set("do:not:clobber", "sentinel")
    before = _SourceSnapshot(source_db)
    dest_before = boot.content_digest(dest)

    assert _run(source_db, dest) == 2

    assert boot.content_digest(dest) == dest_before
    assert existing.get("do:not:clobber") == "sentinel"
    before.assert_unchanged()


def test_force_overwrites_the_destination(source_db, tmp_path, open_store):
    dest = tmp_path / "manual.db"
    stale = open_store(dest)
    stale.set("do:not:clobber", "sentinel")
    stale._conn.close()
    before = _SourceSnapshot(source_db)

    assert _run(source_db, dest, "--force") == 0

    fresh = open_store(dest)
    assert fresh.get("do:not:clobber") is None
    assert fresh.get("cache:positions") == '{"Juan Soto::hitter": ["OF"]}'
    before.assert_unchanged()


def test_refuses_when_destination_is_the_source(source_db, capsys):
    before = _SourceSnapshot(source_db)
    assert _run(source_db, source_db, "--force") == 2
    assert "destination is the source" in capsys.readouterr().out
    before.assert_unchanged()


def test_refuses_any_destination_named_local_db(source_db, tmp_path, capsys):
    dest = tmp_path / "elsewhere" / "local.db"
    assert _run(source_db, dest, "--force") == 2
    assert "Yahoo baseline" in capsys.readouterr().out
    assert not dest.exists()


def test_refuses_the_repo_baseline_store(source_db, capsys):
    """``--dest data/local.db`` is the mistake that would destroy the baseline."""
    assert boot.main(["--source", str(source_db), "--dest", "data/local.db", "--force"]) == 2
    out = capsys.readouterr().out
    assert str(boot.DEFAULT_SOURCE.resolve()) in out
    assert "Yahoo baseline" in out


@pytest.mark.parametrize("render", ["true", "1", "false"])
def test_refuses_when_render_is_set(source_db, tmp_path, monkeypatch, capsys, render):
    """Stricter than ``kv_store.is_remote()``: ANY value of RENDER refuses."""
    monkeypatch.setenv("RENDER", render)
    dest = tmp_path / "manual.db"

    assert _run(source_db, dest) == 2

    assert not dest.exists()
    out = capsys.readouterr().out
    assert "RENDER is set" in out
    assert str(boot.resolve_path(dest)) in out  # paths printed even when refusing


def test_refuses_a_missing_source(tmp_path, capsys):
    missing = tmp_path / "nope.db"
    dest = tmp_path / "manual.db"
    assert _run(missing, dest) == 2
    assert "source does not exist" in capsys.readouterr().out
    assert not dest.exists()


# --------------------------------------------------------------------------
# Defaults and repo wiring
# --------------------------------------------------------------------------


def test_defaults_match_the_kv_store_default_path():
    """Drift guard: the default source IS the store ``get_kv()`` builds off Render."""
    from fantasy_baseball.data import kv_store

    assert boot.DEFAULT_SOURCE == kv_store._DEFAULT_LOCAL_DB
    assert boot.DEFAULT_DEST.name == "manual.db"
    assert boot.DEFAULT_DEST.parent == boot.DEFAULT_SOURCE.parent


def test_relative_paths_anchor_at_the_repo_root(tmp_path):
    assert boot.resolve_path("data/manual.db") == (PROJECT_ROOT / "data" / "manual.db").resolve()
    absolute = tmp_path / "manual.db"
    assert boot.resolve_path(absolute) == absolute.resolve()


def test_manual_db_is_gitignored():
    """``data/*.db`` already covers it -- no .gitignore edit is needed."""
    if shutil.which("git") is None:
        pytest.skip("git not on PATH")
    result = subprocess.run(
        ["git", "check-ignore", "-v", "data/manual.db"],
        cwd=PROJECT_ROOT,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, f"data/manual.db is NOT gitignored: {result.stdout}"
    assert "data/*.db" in result.stdout


def test_a_copy_that_cannot_be_stamped_is_removed(source_db, tmp_path, monkeypatch):
    """An unstamped store on disk is worse than no store at all.

    `manual_store_active` reads the ABSENCE of the stamp as "this is the Yahoo
    baseline", so a copy left behind at data/manual.db hands the next process that
    points FANTASY_LOCAL_KV_PATH at it month-stale prod rosters for a manual page --
    the exact splice the stamp exists to prevent, reached through the failure path.
    """
    dest = tmp_path / "manual.db"

    def _boom(_dest):
        raise RuntimeError("no venv, no package")

    monkeypatch.setattr(boot, "stamp_manual_provenance", _boom)

    assert _run(source_db, dest) == 1
    assert not dest.exists(), "an unstamped copy must not survive"


def test_a_copy_that_fails_the_fidelity_check_is_removed(source_db, tmp_path, monkeypatch):
    """Same reasoning: both fidelity branches leave a fully written, unstamped store."""
    dest = tmp_path / "manual.db"
    real = boot.content_digest
    calls: list[Path] = []

    def _digest(path: Path) -> str:
        calls.append(path)
        # Third call is the copy's own digest; make it disagree with the source.
        return "different" if len(calls) == 3 else real(path)

    monkeypatch.setattr(boot, "content_digest", _digest)

    assert _run(source_db, dest) == 1
    assert not dest.exists(), "an unstamped copy must not survive"


def test_the_script_finds_the_package_without_an_active_venv(source_db, tmp_path):
    """It was stdlib-only by design; the stamp made it need `fantasy_baseball`.

    Every other script here inserts src/ on sys.path for exactly this reason. Without
    it the copy and both checks pass and only the stamp dies -- the worst possible
    place to fail.
    """
    src = boot.PROJECT_ROOT / "src"
    assert str(src) in sys.path


def test_bootstrap_and_seed_agree_on_the_seeded_flag(source_db, tmp_path, open_store):
    """`seeded` is the field to branch on, so it must mean the same to both writers.

    They each used to build their own payload: bootstrap wrote `seeded: False`, the
    seeder wrote `seeded_at` and no `seeded` key at all -- so `.get("seeded")` was
    falsy after a SUCCESSFUL seed, the opposite of what it reads like.
    """
    import json

    from fantasy_baseball.manual.seed import PROVENANCE_KEY, stamp_provenance

    dest = tmp_path / "manual.db"
    assert _run(source_db, dest) == 0
    store = open_store(dest)

    assert json.loads(store.get(PROVENANCE_KEY))["seeded"] is False

    stamp_provenance(store, str(dest), seeded=True, teams=10, players=241)

    after = json.loads(store.get(PROVENANCE_KEY))
    assert after["seeded"] is True
    assert after["teams"] == 10
