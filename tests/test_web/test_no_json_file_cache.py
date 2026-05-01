"""Regression guardrail: cache reads/writes must go through kv_store, not files.

After the cache-layer unification (commits 42a15ab, 1439470, and the
Phase 3 commit that introduced this test), the dashboard's cache layer
lives entirely in kv_store. ``data/cache/*.json`` is gone; the
``CACHE_FILES`` and ``CACHE_DIR`` constants are gone; and no production
or script code should reach for them again.

These tests fail loudly if anyone re-introduces the JSON-file cache
pattern that produced the local/remote split that bit us in PR #45's
diagnostic thread.
"""

from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]

PRODUCTION_AND_SCRIPT_FILES = [
    "src/fantasy_baseball/web/season_data.py",
    "src/fantasy_baseball/web/season_routes.py",
    "src/fantasy_baseball/web/refresh_pipeline.py",
    "src/fantasy_baseball/data/redis_store.py",
    "src/fantasy_baseball/data/ros_pipeline.py",
    "scripts/smoke_test.py",
    "scripts/compare_sgp_local_vs_remote.py",
]

# Forbidden tokens. Anything that reads/writes JSON files in
# ``data/cache/`` or imports the deleted constants.
FORBIDDEN_TOKENS = [
    "CACHE_FILES",
    'data/cache"',
    "data/cache'",
    'data/cache/"',
    "data/cache/'",
]


def test_no_json_file_cache_in_production():
    """Production and CLI files must not reach for ``data/cache/`` or
    the deleted ``CACHE_FILES`` mapping.

    Cache reads/writes go through ``read_cache`` / ``write_cache`` in
    ``season_data``, which route through ``kv_store.get_kv()`` (Upstash
    on Render, SQLite locally). The JSON-file layer is gone.
    """
    for rel_path in PRODUCTION_AND_SCRIPT_FILES:
        path = REPO_ROOT / rel_path
        assert path.is_file(), f"expected file at {rel_path}"
        text = path.read_text(encoding="utf-8")
        for forbidden in FORBIDDEN_TOKENS:
            assert forbidden not in text, (
                f"{rel_path} references the removed JSON-file cache "
                f"({forbidden!r}). Cache reads/writes must go through "
                "``read_cache``/``write_cache`` in season_data, which "
                "route through kv_store.get_kv()."
            )


def test_cache_dir_constant_removed_from_season_data():
    """The ``CACHE_DIR`` and ``CACHE_FILES`` symbols must not exist on
    ``season_data``. Their presence implied a parallel cache layer that
    confused local-vs-remote freshness debugging.
    """
    from fantasy_baseball.web import season_data

    assert not hasattr(season_data, "CACHE_DIR"), (
        "season_data.CACHE_DIR is back. The JSON-file cache layer was "
        "removed in Phase 3 of the cache refactor; reads/writes go "
        "through kv_store.get_kv()."
    )
    assert not hasattr(season_data, "CACHE_FILES"), (
        "season_data.CACHE_FILES is back. Same as CACHE_DIR — the "
        "filename-mapped JSON-file cache layer is gone."
    )
