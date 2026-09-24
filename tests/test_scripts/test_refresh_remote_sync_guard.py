"""``refresh_remote`` must not wipe a non-baseline KV store on its sync-back.

Step 3 of the script calls ``sync_remote_to_local(remote=remote)`` with no
``local=``, so the destination is whatever ``FANTASY_LOCAL_KV_PATH`` resolves
to -- and the sync wipes its destination before refilling it. Running this
script from a manual-pipeline shell would therefore delete the hand-transcribed
manual store silently.

The MESSAGE is shared with ``run_season_dashboard`` -- one hazard, one wording --
but the DECISION stays in each script:
``tests/test_data/test_kv_sync.py::test_default_local_is_get_kv`` pins the
library contract that the default destination is simply whatever ``get_kv()``
returns, and the whole test suite relies on ``FANTASY_LOCAL_KV_PATH`` for
isolation. Narrowing that contract library-side breaks legitimate callers.
"""

import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

import refresh_remote

from fantasy_baseball.data import kv_store


@pytest.fixture(autouse=True)
def _isolated_kv(monkeypatch):
    monkeypatch.setenv("RENDER", "false")
    kv_store._reset_singleton()
    yield
    kv_store._reset_singleton()


def test_refuses_when_destination_is_the_manual_store(monkeypatch, tmp_path):
    manual = tmp_path / "manual.db"
    monkeypatch.setenv("FANTASY_LOCAL_KV_PATH", str(manual))
    kv_store._reset_singleton()

    msg = refresh_remote._sync_destination_refusal()

    assert msg is not None
    assert "REFUSING TO SYNC" in msg
    # WHICH sync: this script's refusal fires at startup, before the remote
    # refresh, so the operator has to be told it is the sync-back at the end
    # that would have done the damage -- and that nothing has run yet.
    assert "sync-back" in msg
    assert "Nothing has run yet" in msg
    assert str(manual) in msg
    # It must say what the operator should do, not just that it failed.
    assert "FANTASY_LOCAL_KV_PATH" in msg
    assert "run_manual_refresh.py" in msg


def test_allows_the_yahoo_baseline(monkeypatch, tmp_path):
    """Relocate the baseline into tmp_path so the real data/local.db is never opened."""
    baseline = tmp_path / "local.db"
    monkeypatch.setattr(kv_store, "_DEFAULT_LOCAL_DB", baseline)
    monkeypatch.setenv("FANTASY_LOCAL_KV_PATH", str(baseline))
    kv_store._reset_singleton()

    assert refresh_remote._sync_destination_refusal() is None


def test_refusal_exit_code_matches_the_other_entry_points(monkeypatch, tmp_path):
    """2 means 'refused, nothing happened' across all three scripts."""
    assert refresh_remote.RC_REFUSED == 2


def test_script_is_ascii_only():
    """cp1252 stdout on this dev box: one non-ASCII glyph crashes the script."""
    raw = (PROJECT_ROOT / "scripts" / "refresh_remote.py").read_bytes()
    assert raw.decode("ascii")


def test_an_already_exported_render_does_not_refuse_the_run(monkeypatch, tmp_path):
    """CLAUDE.md tells operators to export RENDER=true to read Upstash directly.

    Resolving the destination through get_kv() answered according to RENDER, so from
    such a shell the guard saw the Upstash client, found no local file, and refused a
    perfectly legitimate remote refresh before it ran.
    """
    baseline = tmp_path / "local.db"
    monkeypatch.setattr(kv_store, "_DEFAULT_LOCAL_DB", baseline)
    monkeypatch.setenv("FANTASY_LOCAL_KV_PATH", str(baseline))
    monkeypatch.setenv("RENDER", "true")
    kv_store._reset_singleton()

    assert refresh_remote._sync_destination_refusal() is None


def test_the_guard_does_not_create_the_store_it_is_asking_about(monkeypatch, tmp_path):
    """The refusal says nothing was written; asking must not make that false.

    SqliteKVStore.__init__ mkdirs the parent and runs CREATE TABLE, so resolving the
    destination through get_kv() created an empty database and its WAL sidecars
    underneath a message promising no local write.
    """
    manual = tmp_path / "nested" / "manual.db"
    monkeypatch.setenv("FANTASY_LOCAL_KV_PATH", str(manual))
    kv_store._reset_singleton()

    assert refresh_remote._sync_destination_refusal() is not None
    assert not manual.exists()
    assert not manual.parent.exists()


# ---------------------------------------------------------------------------
# Prod holding the published manual store
# ---------------------------------------------------------------------------


def test_refuses_over_published_manual_data():
    msg = refresh_remote.prod_manual_refusal('{"seeded": true}', end_manual=False, skip_yahoo=False)

    assert msg is not None
    assert "publish_manual.py" in msg
    assert "--end-manual" in msg
    assert "Nothing has run yet" in msg


def test_skip_yahoo_is_refused_over_manual_data_too():
    """Stale-data mode recomputes on top of the transcription -- the 409's reason."""
    assert refresh_remote.prod_manual_refusal("{}", end_manual=False, skip_yahoo=True) is not None


def test_end_manual_lets_a_real_refresh_through():
    assert refresh_remote.prod_manual_refusal("{}", end_manual=True, skip_yahoo=False) is None


def test_end_manual_needs_yahoo():
    msg = refresh_remote.prod_manual_refusal("{}", end_manual=True, skip_yahoo=True)
    assert msg is not None and "--skip-yahoo" in msg


def test_a_yahoo_mode_prod_is_unaffected():
    assert refresh_remote.prod_manual_refusal(None, end_manual=False, skip_yahoo=False) is None
    assert refresh_remote.prod_manual_refusal(None, end_manual=False, skip_yahoo=True) is None


def test_end_manual_refuses_when_fb_skip_yahoo_is_set(monkeypatch, capsys):
    """FB_SKIP_YAHOO is stale-data mode without the flag; --end-manual must see it.

    Otherwise the run recomputes over the transcription and then deletes the stamp
    as though a real Yahoo refresh had replaced it.
    """
    from fantasy_baseball.data.cache_keys import MANUAL_PROVENANCE_KEY
    from fantasy_baseball.web import refresh_pipeline

    class _Prod:
        def get(self, key):
            return '{"seeded": true}' if key == MANUAL_PROVENANCE_KEY else None

        def delete(self, key):  # pragma: no cover - must not be reached
            raise AssertionError("stamp deleted on a refused run")

    def _no_refresh(*_a, **_k):  # pragma: no cover - must not be reached
        raise AssertionError("refresh ran on a refused run")

    monkeypatch.setenv("FB_SKIP_YAHOO", "1")
    monkeypatch.setattr(refresh_remote, "_sync_destination_refusal", lambda: None)
    monkeypatch.setattr(kv_store, "build_explicit_upstash_kv", lambda: _Prod())
    monkeypatch.setattr(refresh_pipeline, "RefreshRun", _no_refresh)

    assert refresh_remote.main(["--end-manual"]) == refresh_remote.RC_REFUSED
    assert "FB_SKIP_YAHOO" in capsys.readouterr().out
