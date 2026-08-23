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
