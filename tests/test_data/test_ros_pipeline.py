"""Tests for blend_and_cache_ros — the Redis-backed ROS projections pipeline."""
import shutil
from pathlib import Path

import pandas as pd
import pytest

from fantasy_baseball.data import redis_store
from fantasy_baseball.data.ros_pipeline import blend_and_cache_ros


FIXTURES_DIR = Path(__file__).parent.parent / "fixtures"


def _make_ros_tree(root: Path, year: int, date: str) -> Path:
    """Create root/{year}/rest_of_season/{date}/{steamer-hitters,steamer-pitchers}.csv.

    Mirrors tests/test_data/test_db.py::_make_ros_dir so we can exercise
    the blending pipeline against the same fixture CSVs without reaching
    into test_db.py.
    """
    date_dir = root / str(year) / "rest_of_season" / date
    date_dir.mkdir(parents=True)
    shutil.copy(FIXTURES_DIR / "steamer_hitters.csv", date_dir / "steamer-hitters.csv")
    shutil.copy(FIXTURES_DIR / "steamer_pitchers.csv", date_dir / "steamer-pitchers.csv")
    return date_dir


@pytest.fixture
def projections_dir(tmp_path, monkeypatch):
    """Isolated projections dir + cache dir so write_cache writes to tmp.

    Patches both the module-level ``CACHE_DIR`` and ``write_cache`` /
    ``read_cache``'s default-argument cache-dir capture so write-through
    to disk lands in ``tmp_path/cache`` rather than the real
    ``data/cache/`` directory.
    """
    import fantasy_baseball.web.season_data as season_data
    fake_cache = tmp_path / "cache"
    monkeypatch.setattr(season_data, "CACHE_DIR", fake_cache)
    # Default args were bound at def-time; overwrite them so
    # ``write_cache(...)`` without an explicit cache_dir uses tmp.
    season_data.write_cache.__defaults__ = (fake_cache,)
    season_data.read_cache.__defaults__ = (fake_cache,)
    return tmp_path / "projections"


def test_blend_and_cache_ros_raises_when_ros_dir_missing(projections_dir, monkeypatch):
    """Root dir exists but no rest_of_season subdir → FileNotFoundError."""
    # Nothing created under projections_dir — the year dir is absent.
    monkeypatch.setattr(
        "fantasy_baseball.data.ros_pipeline.get_default_client",
        lambda: None,
    )
    with pytest.raises(FileNotFoundError, match="ROS snapshot dir missing"):
        blend_and_cache_ros(
            projections_dir, ["steamer"], {"steamer": 1.0},
            None, 2026,
        )


def test_blend_and_cache_ros_raises_when_no_date_dirs(projections_dir, monkeypatch):
    """rest_of_season/ exists but contains no date subdirs → FileNotFoundError."""
    (projections_dir / "2026" / "rest_of_season").mkdir(parents=True)
    monkeypatch.setattr(
        "fantasy_baseball.data.ros_pipeline.get_default_client",
        lambda: None,
    )
    with pytest.raises(FileNotFoundError, match="No ROS snapshot dirs"):
        blend_and_cache_ros(
            projections_dir, ["steamer"], {"steamer": 1.0},
            None, 2026,
        )


def test_blend_and_cache_ros_blends_latest_snapshot_and_writes_cache(
    projections_dir, fake_redis, monkeypatch, tmp_path,
):
    """Two date dirs present: the latest one gets blended; result cached."""
    _make_ros_tree(projections_dir, year=2026, date="2026-04-07")
    _make_ros_tree(projections_dir, year=2026, date="2026-04-14")

    monkeypatch.setattr(
        "fantasy_baseball.data.ros_pipeline.get_default_client",
        lambda: fake_redis,
    )
    # Point write_cache's Redis singleton at fake_redis so the
    # write-through lands somewhere observable.
    import fantasy_baseball.web.season_data as season_data
    monkeypatch.setattr(season_data, "_get_redis", lambda: fake_redis)

    hitters_df, pitchers_df = blend_and_cache_ros(
        projections_dir, ["steamer"], {"steamer": 1.0},
        None, 2026,
    )

    # DataFrames: fixture has 4 hitters + 3 pitchers
    assert len(hitters_df) == 4
    assert len(pitchers_df) == 3
    assert (hitters_df["player_type"] == "hitter").all()
    assert (pitchers_df["player_type"] == "pitcher").all()

    # Local cache file written by write_cache.
    import json
    cache_file = tmp_path / "cache" / "ros_projections.json"
    assert cache_file.exists()
    cached = json.loads(cache_file.read_text(encoding="utf-8"))
    assert len(cached["hitters"]) == 4
    assert len(cached["pitchers"]) == 3

    # Redis write-through: cache:ros_projections is populated.
    raw = fake_redis.get("cache:ros_projections")
    assert raw is not None
    cached_redis = json.loads(raw)
    assert len(cached_redis["hitters"]) == 4
    assert len(cached_redis["pitchers"]) == 3


def test_blend_and_cache_ros_normalizes_using_redis_totals(
    projections_dir, fake_redis, monkeypatch,
):
    """game_log_totals from Redis must be added to ROS counting stats.

    The fixture has Aaron Judge (mlbam_id 592450) with ROS HR=45. If we
    pre-seed game_log_totals with 10 accumulated HR, the blended output
    must show HR=55 (the JSON round-trip stringifies keys; the pipeline
    must coerce them back to int).
    """
    _make_ros_tree(projections_dir, year=2026, date="2026-04-07")

    # 10 HR already accumulated for Aaron Judge.
    redis_store.set_game_log_totals(
        fake_redis, "hitters",
        {"592450": {"ab": 40, "h": 12, "r": 8, "hr": 10, "rbi": 25, "sb": 0, "pa": 45}},
    )
    # Gerrit Cole (mlbam_id 543037) has 5 wins, 60 Ks banked.
    redis_store.set_game_log_totals(
        fake_redis, "pitchers",
        {"543037": {"ip": 40.0, "er": 12, "bb": 10, "h_allowed": 30, "k": 60, "w": 5, "sv": 0}},
    )

    monkeypatch.setattr(
        "fantasy_baseball.data.ros_pipeline.get_default_client",
        lambda: fake_redis,
    )
    import fantasy_baseball.web.season_data as season_data
    monkeypatch.setattr(season_data, "_get_redis", lambda: fake_redis)

    hitters_df, pitchers_df = blend_and_cache_ros(
        projections_dir, ["steamer"], {"steamer": 1.0},
        None, 2026,
    )

    judge = hitters_df[hitters_df["name"] == "Aaron Judge"]
    assert len(judge) == 1
    # ROS 45 + actuals 10 = 55 full-season
    assert judge.iloc[0]["hr"] == pytest.approx(55)

    cole = pitchers_df[pitchers_df["name"] == "Gerrit Cole"]
    assert len(cole) == 1
    # ROS 15 W + actuals 5 W = 20 full-season
    assert cole.iloc[0]["w"] == pytest.approx(20)
    # ROS 240 K + actuals 60 K = 300 full-season
    assert cole.iloc[0]["k"] == pytest.approx(300)


def test_blend_and_cache_ros_still_returns_dfs_when_redis_unconfigured(
    projections_dir, fake_redis, monkeypatch,
):
    """None client: blending still succeeds; Redis write-through is a no-op.

    Convention matches Tasks 2-6: readers return empty, writers no-op.
    write_cache still writes to local disk, so DataFrames come back populated.
    ``fake_redis`` is injected but the pipeline sees ``None`` for both its
    own client and ``season_data._get_redis``, so we can assert the Redis
    cache key was never written.
    """
    _make_ros_tree(projections_dir, year=2026, date="2026-04-07")

    monkeypatch.setattr(
        "fantasy_baseball.data.ros_pipeline.get_default_client",
        lambda: None,
    )
    import fantasy_baseball.web.season_data as season_data
    monkeypatch.setattr(season_data, "_get_redis", lambda: None)

    hitters_df, pitchers_df = blend_and_cache_ros(
        projections_dir, ["steamer"], {"steamer": 1.0},
        None, 2026,
    )
    assert len(hitters_df) == 4
    assert len(pitchers_df) == 3
    # Writer no-op: cache:ros_projections must NOT be written when the
    # Redis client is unconfigured.
    assert fake_redis.get("cache:ros_projections") is None
