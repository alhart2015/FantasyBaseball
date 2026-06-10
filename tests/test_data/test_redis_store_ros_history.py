"""Tests for ros_projection_history snapshot helpers."""

from fantasy_baseball.data import redis_store

# A full-blob-shaped fixture: extra columns (fg_id, team, avg) that must be
# dropped, and below-threshold players (a phantom AAA bat / a mop-up arm) that
# must be filtered out.
ROS_BLOB = {
    "hitters": [
        {
            "fg_id": "h1",
            "mlbam_id": 111,
            "name": "Star Bat",
            "team": "NYM",
            "pa": 640,
            "r": 95,
            "hr": 30,
            "rbi": 92,
            "sb": 8,
            "h": 165,
            "ab": 560,
            "avg": 0.295,
        },
        {
            "fg_id": "h2",
            "mlbam_id": 222,
            "name": "AAA Phantom",
            "team": "FA",
            "pa": 40,  # below MIN_PA -> dropped
            "r": 5,
            "hr": 1,
            "rbi": 4,
            "sb": 0,
            "h": 9,
            "ab": 36,
            "avg": 0.250,
        },
    ],
    "pitchers": [
        {
            "fg_id": "p1",
            "mlbam_id": 333,
            "name": "Ace Arm",
            "team": "ATL",
            "ip": 180,
            "w": 13,
            "k": 210,
            "sv": 0,
            "er": 62,
            "bb": 45,
            "h_allowed": 150,
            "gs": 30,
            "era": 3.10,
        },
        {
            "fg_id": "p2",
            "mlbam_id": 444,
            "name": "Mopup Arm",
            "team": "FA",
            "ip": 8,  # below MIN_IP -> dropped
            "w": 0,
            "k": 7,
            "sv": 0,
            "er": 5,
            "bb": 4,
            "h_allowed": 10,
            "gs": 0,
            "era": 5.60,
        },
    ],
}


def test_write_and_read_trims_columns_and_players(fake_redis):
    redis_store.write_ros_projection_snapshot(fake_redis, ROS_BLOB, "2026-06-10")
    hist = redis_store.get_ros_projection_history(fake_redis)

    assert list(hist) == ["2026-06-10"]
    snap = hist["2026-06-10"]

    # Below-threshold players filtered out.
    assert [h["name"] for h in snap["hitters"]] == ["Star Bat"]
    assert [p["name"] for p in snap["pitchers"]] == ["Ace Arm"]

    # Only the snapshot columns survive (fg_id/team/avg/gs/era dropped).
    assert set(snap["hitters"][0]) == set(redis_store._ROS_SNAPSHOT_HITTER_COLS)
    assert set(snap["pitchers"][0]) == set(redis_store._ROS_SNAPSHOT_PITCHER_COLS)
    # Values preserved.
    assert snap["hitters"][0]["hr"] == 30
    assert snap["pitchers"][0]["k"] == 210


def test_overwrites_same_date(fake_redis):
    redis_store.write_ros_projection_snapshot(fake_redis, ROS_BLOB, "2026-06-10")
    smaller = {"hitters": [dict(ROS_BLOB["hitters"][0], hr=99)], "pitchers": []}
    redis_store.write_ros_projection_snapshot(fake_redis, smaller, "2026-06-10")

    hist = redis_store.get_ros_projection_history(fake_redis)
    assert list(hist) == ["2026-06-10"]
    assert hist["2026-06-10"]["hitters"][0]["hr"] == 99
    assert hist["2026-06-10"]["pitchers"] == []


def test_multiple_dates(fake_redis):
    redis_store.write_ros_projection_snapshot(fake_redis, ROS_BLOB, "2026-06-03")
    redis_store.write_ros_projection_snapshot(fake_redis, ROS_BLOB, "2026-06-10")
    assert set(redis_store.get_ros_projection_history(fake_redis)) == {"2026-06-03", "2026-06-10"}


def test_write_noops(fake_redis):
    redis_store.write_ros_projection_snapshot(None, ROS_BLOB, "2026-06-10")  # no client
    redis_store.write_ros_projection_snapshot(fake_redis, None, "2026-06-10")  # empty blob
    redis_store.write_ros_projection_snapshot(fake_redis, ROS_BLOB, "")  # blank date
    assert redis_store.get_ros_projection_history(fake_redis) == {}


def test_get_history_none_client_returns_empty():
    assert redis_store.get_ros_projection_history(None) == {}


def test_empty_content_blob_does_not_overwrite(fake_redis):
    """A structurally-present but content-empty blob (refuse-stale fallback)
    must not clobber a previously-good snapshot for the same date."""
    redis_store.write_ros_projection_snapshot(fake_redis, ROS_BLOB, "2026-06-10")
    redis_store.write_ros_projection_snapshot(
        fake_redis, {"hitters": [], "pitchers": []}, "2026-06-10"
    )
    hist = redis_store.get_ros_projection_history(fake_redis)
    # Original snapshot preserved, not replaced with an empty one.
    assert [h["name"] for h in hist["2026-06-10"]["hitters"]] == ["Star Bat"]


def test_non_dict_blob_noop(fake_redis):
    redis_store.write_ros_projection_snapshot(fake_redis, [1, 2, 3], "2026-06-10")  # type: ignore[arg-type]
    assert redis_store.get_ros_projection_history(fake_redis) == {}


def test_nan_volume_dropped(fake_redis):
    blob = {
        "hitters": [
            {"name": "Healthy", "mlbam_id": 1, "pa": 600, "r": 90},
            {"name": "NaN PA", "mlbam_id": 2, "pa": float("nan"), "r": 50},
        ],
        "pitchers": [],
    }
    redis_store.write_ros_projection_snapshot(fake_redis, blob, "2026-06-10")
    snap = redis_store.get_ros_projection_history(fake_redis)["2026-06-10"]
    assert [h["name"] for h in snap["hitters"]] == ["Healthy"]
