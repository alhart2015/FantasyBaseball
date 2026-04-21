"""Tests for scripts/migrate_standings_history.py."""

import json
import sys
from datetime import date
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
from fantasy_baseball.models.standings import (  # noqa: E402
    CategoryStats,
    Standings,
    StandingsEntry,
)

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


def test_rewrite_hash_counts_corrupt_json_as_error(fake_redis):
    fake_redis.hset(redis_store.STANDINGS_HISTORY_KEY, "2026-04-15", "not-json")
    stats = rewrite_hash(fake_redis)
    assert stats["errors"] == 1
    assert stats["rewritten"] == 0
    assert stats["skipped"] == 0


def test_rewrite_hash_counts_non_dict_payload_as_error(fake_redis):
    fake_redis.hset(redis_store.STANDINGS_HISTORY_KEY, "2026-04-15", json.dumps([1, 2, 3]))
    stats = rewrite_hash(fake_redis)
    assert stats["errors"] == 1
    assert stats["rewritten"] == 0
    assert stats["skipped"] == 0


def test_rewrite_hash_handles_mixed_hash(fake_redis):
    canonical = Standings(
        effective_date=date(2026, 4, 16),
        entries=[
            StandingsEntry(
                team_name="Beta",
                team_key="431.l.1.t.2",
                rank=1,
                stats=CategoryStats(r=10),
                yahoo_points_for=None,
            )
        ],
    ).to_json()
    fake_redis.hset(redis_store.STANDINGS_HISTORY_KEY, "2026-04-15", json.dumps(LEGACY_DAY))
    fake_redis.hset(redis_store.STANDINGS_HISTORY_KEY, "2026-04-16", json.dumps(canonical))
    stats = rewrite_hash(fake_redis)
    assert stats["rewritten"] == 1
    assert stats["skipped"] == 1
    assert stats["errors"] == 0


def test_rewrite_hash_handles_legacy_row_missing_era(fake_redis):
    legacy_no_era = {
        "teams": [
            {
                "team": "Delta",
                "team_key": "431.l.1.t.3",
                "rank": 2,
                "r": 30,
                "hr": 5,
                "rbi": 22,
                "sb": 1,
                "avg": 0.240,
                "w": 2,
                "k": 40,
                "sv": 0,
                # era and whip intentionally omitted
            },
        ],
    }
    fake_redis.hset(
        redis_store.STANDINGS_HISTORY_KEY,
        "2026-04-17",
        json.dumps(legacy_no_era),
    )
    stats = rewrite_hash(fake_redis)
    assert stats["rewritten"] == 1
    reloaded = redis_store.get_standings_day(fake_redis, "2026-04-17")
    assert reloaded is not None
    assert reloaded.entries[0].stats.era == 99.0
    assert reloaded.entries[0].stats.whip == 99.0


def test_rewrite_hash_rewrites_intermediate_shape_without_effective_date(fake_redis):
    """Canonical rows inside ``{"teams": [...]}`` but no ``effective_date``
    wrapper. Migrator should inject the hash key as ``effective_date`` and
    rewrite in canonical form — this is the shape that survived the first
    migration pass and kept crashing ``League.from_redis`` afterwards.
    """
    intermediate = {
        "teams": [
            {
                "name": "Epsilon",
                "team_key": "431.l.1.t.4",
                "rank": 3,
                "stats": {
                    "R": 50,
                    "HR": 15,
                    "RBI": 45,
                    "SB": 10,
                    "AVG": 0.270,
                    "W": 4,
                    "K": 90,
                    "SV": 5,
                    "ERA": 3.10,
                    "WHIP": 1.12,
                },
            }
        ],
    }
    fake_redis.hset(
        redis_store.STANDINGS_HISTORY_KEY,
        "2026-04-18",
        json.dumps(intermediate),
    )
    stats = rewrite_hash(fake_redis)
    assert stats["rewritten"] == 1
    assert stats["errors"] == 0
    reloaded = redis_store.get_standings_day(fake_redis, "2026-04-18")
    assert reloaded is not None
    assert reloaded.effective_date == date(2026, 4, 18)
    assert reloaded.entries[0].team_name == "Epsilon"
    assert reloaded.entries[0].stats.r == 50


def test_rewrite_hash_survives_partially_canonical_row(fake_redis):
    # Canonical wrapper + canonical field names on the row (has 'name',
    # has 'stats'), but MISSING team_key and rank. Standings.from_json
    # raises KeyError on this shape. Before the broaden-except fix, this
    # would halt the migration; now the canonical probe falls through
    # to the legacy parser, which fails ValueError (no 'team' key) and
    # gets counted as an error without raising.
    partial = {
        "effective_date": "2026-04-16",
        "teams": [
            {
                "name": "Gamma",
                "stats": {"R": 5},
            }
        ],
    }
    fake_redis.hset(
        redis_store.STANDINGS_HISTORY_KEY,
        "2026-04-16",
        json.dumps(partial),
    )
    stats = rewrite_hash(fake_redis)
    assert stats["errors"] == 1
    assert stats["rewritten"] == 0
    assert stats["skipped"] == 0
