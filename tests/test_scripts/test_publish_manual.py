"""``scripts/publish_manual.py``: the manual store going up to prod.

Prod is stood in for by a second SQLite store; the script reaches it only through
``build_explicit_upstash_kv``, which is patched. The conftest strips real Upstash
creds, so an unpatched path fails closed rather than reaching prod.
"""

import gzip
import json
import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

import publish_manual

from fantasy_baseball.data import kv_store
from fantasy_baseball.data.cache_keys import MANUAL_PROVENANCE_KEY, CacheKey, redis_key
from fantasy_baseball.data.kv_store import SqliteKVStore

META = redis_key(CacheKey.META)
SEEDED = json.dumps(
    {
        "source": "manual-transcription",
        "seeded": True,
        "roster_snapshot_date": "2026-09-14",
        "standings_effective_date": "2026-09-14",
    }
)


@pytest.fixture
def stores(tmp_path, monkeypatch):
    """(manual store path, local store, fake prod) with prod wired into the script."""
    path = tmp_path / "manual.db"
    local = SqliteKVStore(path)
    prod = SqliteKVStore(tmp_path / "prod.db")
    monkeypatch.setattr(kv_store, "build_explicit_upstash_kv", lambda: prod)
    return path, local, prod


def _run(path: Path, tmp_path: Path, *extra: str) -> int:
    return publish_manual.main(
        ["--kv-path", str(path), "--backup-dir", str(tmp_path / "backups"), *extra]
    )


def test_publishes_a_seeded_store_and_backs_up_what_it_replaced(stores, tmp_path):
    path, local, prod = stores
    local.set(MANUAL_PROVENANCE_KEY, SEEDED)
    local.set(META, json.dumps({"_data": {"last_refresh": "2026-09-14 09:04"}}))
    local.set("cache:standings", "manual")
    prod.set("cache:standings", "yahoo 07-27")

    assert _run(path, tmp_path) == publish_manual.RC_OK

    assert prod.get("cache:standings") == "manual"
    assert prod.get(MANUAL_PROVENANCE_KEY) == SEEDED
    [backup] = (tmp_path / "backups").glob("upstash-before-publish-*.json.gz")
    with gzip.open(backup, "rt", encoding="utf-8") as fh:
        saved = json.load(fh)
    assert saved["strings"]["cache:standings"] == "yahoo 07-27"
    assert saved["strings"][MANUAL_PROVENANCE_KEY] is None


def test_dry_run_writes_nothing_anywhere(stores, tmp_path):
    path, local, prod = stores
    local.set(MANUAL_PROVENANCE_KEY, SEEDED)
    local.set("cache:standings", "manual")

    assert _run(path, tmp_path, "--dry-run") == publish_manual.RC_OK

    assert prod.get("cache:standings") is None
    assert prod.get(MANUAL_PROVENANCE_KEY) is None
    assert not (tmp_path / "backups").exists()


def test_refuses_the_yahoo_baseline(stores, tmp_path):
    """No stamp means not a manual store: Yahoo-era data must not go up unmarked."""
    path, local, prod = stores
    local.set("cache:standings", "yahoo")

    assert _run(path, tmp_path) == publish_manual.RC_REFUSED
    assert prod.get("cache:standings") is None


def test_refuses_a_bootstrapped_but_unseeded_store(stores, tmp_path):
    path, local, prod = stores
    local.set(MANUAL_PROVENANCE_KEY, json.dumps({"seeded": False}))
    local.set("cache:standings", "yahoo copy")

    assert _run(path, tmp_path) == publish_manual.RC_REFUSED
    assert prod.get("cache:standings") is None


def test_a_missing_store_is_refused_without_being_created(tmp_path, monkeypatch):
    missing = tmp_path / "nested" / "manual.db"
    monkeypatch.setattr(kv_store, "build_explicit_upstash_kv", lambda: pytest.fail("reached prod"))

    assert _run(missing, tmp_path) == publish_manual.RC_REFUSED
    assert not missing.parent.exists()


def test_refuses_under_render(stores, tmp_path, monkeypatch):
    path, local, _ = stores
    local.set(MANUAL_PROVENANCE_KEY, SEEDED)
    monkeypatch.setenv("RENDER", "false")  # any non-empty value

    assert _run(path, tmp_path) == publish_manual.RC_REFUSED


def test_script_is_ascii_only():
    """cp1252 stdout on this dev box: one non-ASCII glyph crashes the script."""
    raw = (PROJECT_ROOT / "scripts" / "publish_manual.py").read_bytes()
    assert raw.decode("ascii")
