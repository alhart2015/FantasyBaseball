"""Tests for scripts/migrate_standings_history.py."""

import json
import sys
from pathlib import Path

import pytest

# Ensure scripts/ is on sys.path
SCRIPTS = Path(__file__).resolve().parents[2] / "scripts"
sys.path.insert(0, str(SCRIPTS))

from migrate_standings_history import (  # type: ignore[import-not-found]  # noqa: E402
    _from_legacy_json,
    rewrite_hash,
)

from fantasy_baseball.data import redis_store  # noqa: E402
from fantasy_baseball.models.standings import Standings  # noqa: E402

LEGACY_DAY = {
    "teams": [
        {
            "team": "Alpha",
            "team_key": "431.l.1.t.1",
            "rank": 1,
            "r": 45,
            "hr": 12,
            "rbi": 40,
            "sb": 8,
            "avg": 0.268,
            "w": 3,
            "k": 85,
            "sv": 4,
            "era": 3.21,
            "whip": 1.14,
        },
    ],
}


def test_from_legacy_json_parses_legacy_shape():
    s = _from_legacy_json(LEGACY_DAY, snapshot_date="2026-04-15")
    assert isinstance(s, Standings)
    assert s.entries[0].team_name == "Alpha"
    assert s.entries[0].stats.r == 45
    assert s.entries[0].stats.whip == pytest.approx(1.14)
    assert s.entries[0].yahoo_points_for is None


def test_rewrite_hash_converts_legacy_entries(fake_redis):
    fake_redis.hset(redis_store.STANDINGS_HISTORY_KEY, "2026-04-15", json.dumps(LEGACY_DAY))
    stats = rewrite_hash(fake_redis)
    assert stats["rewritten"] == 1
    assert stats["skipped"] == 0
    # After rewrite, standard reader works
    reloaded = redis_store.get_standings_day(fake_redis, "2026-04-15")
    assert reloaded is not None
    assert reloaded.entries[0].team_name == "Alpha"


def test_rewrite_hash_is_idempotent(fake_redis):
    fake_redis.hset(redis_store.STANDINGS_HISTORY_KEY, "2026-04-15", json.dumps(LEGACY_DAY))
    rewrite_hash(fake_redis)
    stats = rewrite_hash(fake_redis)
    assert stats["rewritten"] == 0
    assert stats["skipped"] == 1
