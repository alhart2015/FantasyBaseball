"""Tests for blend_and_cache_ros — the Redis-backed ROS projections pipeline."""

import shutil
from pathlib import Path

import pytest

from fantasy_baseball.data import redis_store
from fantasy_baseball.data.ros_pipeline import blend_and_cache_ros
from tests._cache_helpers import unwrap_cache_value

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
def projections_dir(tmp_path):
    """Isolated projections dir for the ROS blend tests.

    Pre-cache-refactor this fixture also monkey-patched
    ``season_data.CACHE_DIR`` and ``write_cache.__defaults__`` to redirect
    JSON-file cache writes to ``tmp_path/cache``. After the cache moved
    onto kv_store the redirection is unnecessary — tests that need a KV
    isolation seed it directly via ``monkeypatch.setattr(season_data,
    "get_kv", lambda: fake_redis)`` (see the individual test bodies).
    """
    return tmp_path / "projections"


def test_blend_and_cache_ros_raises_when_ros_dir_missing(projections_dir, monkeypatch):
    """Root dir exists but no rest_of_season subdir → FileNotFoundError."""
    # Nothing created under projections_dir — the year dir is absent.
    monkeypatch.setattr(
        "fantasy_baseball.data.ros_pipeline.get_kv",
        lambda: None,
    )
    with pytest.raises(FileNotFoundError, match="ROS snapshot dir missing"):
        blend_and_cache_ros(
            projections_dir,
            ["steamer"],
            {"steamer": 1.0},
            None,
            2026,
        )


def test_blend_and_cache_ros_raises_when_no_date_dirs(projections_dir, monkeypatch):
    """rest_of_season/ exists but contains no date subdirs → FileNotFoundError."""
    (projections_dir / "2026" / "rest_of_season").mkdir(parents=True)
    monkeypatch.setattr(
        "fantasy_baseball.data.ros_pipeline.get_kv",
        lambda: None,
    )
    with pytest.raises(FileNotFoundError, match="No ROS snapshot dirs"):
        blend_and_cache_ros(
            projections_dir,
            ["steamer"],
            {"steamer": 1.0},
            None,
            2026,
        )


def test_blend_and_cache_ros_blends_latest_snapshot_and_writes_cache(
    projections_dir,
    fake_redis,
    monkeypatch,
    tmp_path,
):
    """Two date dirs present: the latest one gets blended; result cached."""
    _make_ros_tree(projections_dir, year=2026, date="2026-04-07")
    _make_ros_tree(projections_dir, year=2026, date="2026-04-14")

    monkeypatch.setattr(
        "fantasy_baseball.data.ros_pipeline.get_kv",
        lambda: fake_redis,
    )
    # write_cache routes through season_data.get_kv() after Phase 1 of
    # the cache refactor. Patch the season_data binding so the cache
    # write lands in fake_redis where we can assert against it.
    import fantasy_baseball.web.season_data as season_data

    monkeypatch.setattr(season_data, "get_kv", lambda: fake_redis)

    hitters_df, pitchers_df = blend_and_cache_ros(
        projections_dir,
        ["steamer"],
        {"steamer": 1.0},
        None,
        2026,
    )

    # DataFrames: fixture has 4 hitters + 3 pitchers
    assert len(hitters_df) == 4
    assert len(pitchers_df) == 3
    assert (hitters_df["player_type"] == "hitter").all()
    assert (pitchers_df["player_type"] == "pitcher").all()

    # cache:ros_projections is populated in the KV.

    raw = fake_redis.get("cache:ros_projections")
    assert raw is not None
    cached_redis = unwrap_cache_value(raw)
    assert len(cached_redis["hitters"]) == 4
    assert len(cached_redis["pitchers"]) == 3


def test_blend_and_cache_ros_normalizes_using_redis_totals(
    projections_dir,
    fake_redis,
    monkeypatch,
):
    """game_log_totals from Redis must be added to ROS counting stats — but
    only into the cache:full_season_projections blob; the returned
    DataFrame and cache:ros_projections must remain ROS-only.

    # updated: cache:ros_projections is ROS-only per ros_only_decision_projections.md
    The fixture has Aaron Judge (mlbam_id 592450) with ROS HR=45. After
    pre-seeding 10 accumulated HR, the returned (ROS) DataFrame must
    still show HR=45, while cache:full_season_projections must show
    HR=55 (45+10). Pre-fix this test asserted HR=55 on the returned
    DataFrame because the pipeline pre-blend-normalized; the new
    pipeline keeps the returned DataFrame ROS-only and emits the
    YTD-added view through the typed full-season setter.
    """

    _make_ros_tree(projections_dir, year=2026, date="2026-04-07")

    # 10 HR already accumulated for Aaron Judge.
    redis_store.set_game_log_totals(
        fake_redis,
        "hitters",
        {"592450": {"ab": 40, "h": 12, "r": 8, "hr": 10, "rbi": 25, "sb": 0, "pa": 45}},
    )
    # Gerrit Cole (mlbam_id 543037) has 5 wins, 60 Ks banked.
    redis_store.set_game_log_totals(
        fake_redis,
        "pitchers",
        {"543037": {"ip": 40.0, "er": 12, "bb": 10, "h_allowed": 30, "k": 60, "w": 5, "sv": 0}},
    )

    monkeypatch.setattr(
        "fantasy_baseball.data.ros_pipeline.get_kv",
        lambda: fake_redis,
    )
    import fantasy_baseball.web.season_data as season_data

    monkeypatch.setattr(season_data, "get_kv", lambda: fake_redis)

    hitters_df, pitchers_df = blend_and_cache_ros(
        projections_dir,
        ["steamer"],
        {"steamer": 1.0},
        None,
        2026,
    )

    # Returned DataFrame is ROS-only (no YTD added).
    judge = hitters_df[hitters_df["name"] == "Aaron Judge"]
    assert len(judge) == 1
    assert judge.iloc[0]["hr"] == pytest.approx(45)  # ROS-only HR

    cole = pitchers_df[pitchers_df["name"] == "Gerrit Cole"]
    assert len(cole) == 1
    assert cole.iloc[0]["w"] == pytest.approx(15)  # ROS-only W
    assert cole.iloc[0]["k"] == pytest.approx(240)  # ROS-only K

    # cache:full_season_projections holds the YTD-added view.
    full_raw = fake_redis.get("cache:full_season_projections")
    assert full_raw is not None
    full = unwrap_cache_value(full_raw)
    judge_full = next(p for p in full["hitters"] if p["name"] == "Aaron Judge")
    cole_full = next(p for p in full["pitchers"] if p["name"] == "Gerrit Cole")
    assert judge_full["hr"] == pytest.approx(55)  # 45 + 10
    assert cole_full["w"] == pytest.approx(20)  # 15 + 5
    assert cole_full["k"] == pytest.approx(300)  # 240 + 60


def test_blend_writes_both_ros_and_full_season(tmp_path, monkeypatch):
    """blend_and_cache_ros() must write BOTH cache:ros_projections (ROS-only)
    AND cache:full_season_projections (with YTD added)."""
    from fantasy_baseball.data import redis_store as rs
    from fantasy_baseball.data.kv_store import SqliteKVStore, _reset_singleton
    from fantasy_baseball.data.ros_pipeline import blend_and_cache_ros

    monkeypatch.setenv("FANTASY_LOCAL_KV_PATH", str(tmp_path / "kv.db"))
    _reset_singleton()
    kv = SqliteKVStore(tmp_path / "kv.db")
    rs.set_game_log_totals(
        kv,
        "hitters",
        {
            "12345": {
                "r": 30,
                "hr": 5,
                "rbi": 20,
                "sb": 2,
                "h": 30,
                "ab": 100,
                "pa": 110,
                "name": "Test Hitter",
            },
        },
    )
    rs.set_game_log_totals(kv, "pitchers", {})

    proj_dir = tmp_path / "projections"
    date_dir = proj_dir / "2026" / "rest_of_season" / "2026-04-26"
    date_dir.mkdir(parents=True)
    (date_dir / "steamer-hitters.csv").write_text(
        "fg_id,mlbam_id,Name,Team,PA,AB,R,HR,RBI,SB,H,AVG\n"
        "x,12345,Test Hitter,X,440,400,100,25,75,8,110,0.275\n"
    )
    (date_dir / "steamer-pitchers.csv").write_text(
        "fg_id,mlbam_id,Name,Team,IP,W,SO,SV,ER,BB,H,ERA,WHIP\n"
    )

    blend_and_cache_ros(
        projections_dir=proj_dir,
        systems=["steamer"],
        weights=None,
        roster_names=None,
        season_year=2026,
    )

    from fantasy_baseball.data.cache_keys import CacheKey
    from fantasy_baseball.web.season_data import read_cache_dict

    ros = read_cache_dict(CacheKey.ROS_PROJECTIONS)
    full = read_cache_dict(CacheKey.FULL_SEASON_PROJECTIONS)

    assert ros is not None and full is not None
    ros_row = next(p for p in ros["hitters"] if p.get("mlbam_id") == 12345)
    full_row = next(p for p in full["hitters"] if p.get("mlbam_id") == 12345)
    assert ros_row["r"] == 100.0, "ROS cache must be 100 (CSV value, no YTD added)"
    assert full_row["r"] == 130.0, "Full-season cache must be 100+30=130"


def test_blend_warns_and_stamps_snapshot_date_when_stale(
    projections_dir,
    fake_redis,
    monkeypatch,
):
    """A snapshot older than the staleness threshold must emit a loud
    warning (full-season = YTD + ROS would double-count) AND stamp the
    snapshot date into both cache envelopes for provenance. Both blobs are
    still written -- warn-and-proceed, not abort."""
    import datetime as _dt
    import json

    _make_ros_tree(projections_dir, year=2026, date="2026-04-07")
    monkeypatch.setattr("fantasy_baseball.data.ros_pipeline.get_kv", lambda: fake_redis)
    import fantasy_baseball.web.season_data as season_data

    monkeypatch.setattr(season_data, "get_kv", lambda: fake_redis)
    # Pin "today" so the 2026-04-07 snapshot is 13 days stale (> 7).
    monkeypatch.setattr(
        "fantasy_baseball.data.ros_pipeline.local_today",
        lambda: _dt.date(2026, 4, 20),
    )

    msgs: list[str] = []
    blend_and_cache_ros(
        projections_dir,
        ["steamer"],
        {"steamer": 1.0},
        None,
        2026,
        progress_cb=msgs.append,
    )

    assert any("stale" in m.lower() for m in msgs), msgs
    for key in ("cache:ros_projections", "cache:full_season_projections"):
        meta = json.loads(fake_redis.get(key))["_meta"]
        assert meta["_ros_snapshot_date"] == "2026-04-07"


def test_blend_no_stale_warning_when_snapshot_fresh(
    projections_dir,
    fake_redis,
    monkeypatch,
):
    """A snapshot within the staleness threshold emits no warning, but the
    snapshot date is still stamped into the envelope."""
    import datetime as _dt
    import json

    _make_ros_tree(projections_dir, year=2026, date="2026-04-18")
    monkeypatch.setattr("fantasy_baseball.data.ros_pipeline.get_kv", lambda: fake_redis)
    import fantasy_baseball.web.season_data as season_data

    monkeypatch.setattr(season_data, "get_kv", lambda: fake_redis)
    # 2 days after the snapshot -> fresh.
    monkeypatch.setattr(
        "fantasy_baseball.data.ros_pipeline.local_today",
        lambda: _dt.date(2026, 4, 20),
    )

    msgs: list[str] = []
    blend_and_cache_ros(
        projections_dir,
        ["steamer"],
        {"steamer": 1.0},
        None,
        2026,
        progress_cb=msgs.append,
    )

    assert not any("stale" in m.lower() for m in msgs), msgs
    meta = json.loads(fake_redis.get("cache:ros_projections"))["_meta"]
    assert meta["_ros_snapshot_date"] == "2026-04-18"


def test_blend_and_cache_ros_propagates_cache_write_failure(
    projections_dir,
    monkeypatch,
):
    """A KV write failure during caching propagates (fail-loud), so the ROS
    fetch job fails and QStash redelivers rather than silently producing
    un-persisted projections.

    Previously this asserted a None client was a no-op writer; get_kv never
    returns None in the deployed app (it resolves to SQLite off-Render and
    raises on Render without creds), so that dead path was removed -- a
    configured backend that errors on write must surface, not be swallowed.
    """
    import pytest

    _make_ros_tree(projections_dir, year=2026, date="2026-04-07")

    class _WriteFailsKV:
        def get(self, key):
            return None

        def set(self, key, value, **_):
            raise ConnectionError("Upstash unreachable")

    kv = _WriteFailsKV()
    monkeypatch.setattr("fantasy_baseball.data.ros_pipeline.get_kv", lambda: kv)
    import fantasy_baseball.web.season_data as season_data

    monkeypatch.setattr(season_data, "get_kv", lambda: kv)

    with pytest.raises(ConnectionError):
        blend_and_cache_ros(projections_dir, ["steamer"], {"steamer": 1.0}, None, 2026)
