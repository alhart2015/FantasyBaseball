"""Regression test for ROS-vs-full-season cache shape.

Captures the bug where ``cache:ros_projections`` held YTD-inflated
full-season stats. After the fix, ``cache:ros_projections`` MUST hold
ROS-remaining-only stats, and ``cache:full_season_projections`` MUST
hold the YTD-added totals.
"""

from __future__ import annotations

import pandas as pd

from fantasy_baseball.data import redis_store as rs


def test_ros_cache_excludes_ytd(tmp_path, monkeypatch):
    """A player with YTD games should have cached ROS == FanGraphs ROS-only.

    Setup: a hitter projected for 100 R remaining-games-only by FanGraphs,
    with 30 R already accumulated YTD. The cached ROS blob should be 100,
    not 130. The cached full-season blob should be 130.
    """
    from fantasy_baseball.data.kv_store import SqliteKVStore

    monkeypatch.setenv("FANTASY_LOCAL_KV_PATH", str(tmp_path / "kv.db"))
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

    # Simulate the FanGraphs ROS-only blend (what the CSV blend produces):
    ros_only_df = pd.DataFrame(
        [
            {
                "name": "Test Hitter",
                "mlbam_id": 12345,
                "fg_id": "x",
                "r": 100.0,
                "hr": 25.0,
                "rbi": 75.0,
                "sb": 8.0,
                "h": 110.0,
                "ab": 400.0,
                "pa": 440.0,
                "avg": 0.275,
                "player_type": "hitter",
                "team": "X",
                "adp": 1,
            }
        ]
    )

    # NEW pipeline (Phase 1) MUST write ROS-only and full-season separately.
    from fantasy_baseball.data.ros_pipeline import write_ros_and_full_season

    write_ros_and_full_season(kv, hitters_ros=ros_only_df, pitchers_ros=pd.DataFrame())

    ros_cache = rs.get_ros_projections(kv)
    full_cache = rs.get_full_season_projections(kv)

    ros_row = next(p for p in ros_cache["hitters"] if p["mlbam_id"] == 12345)
    full_row = next(p for p in full_cache["hitters"] if p["mlbam_id"] == 12345)

    assert ros_row["r"] == 100.0, "ros cache must NOT include YTD"
    assert ros_row["ab"] == 400.0
    assert full_row["r"] == 130.0, "full-season cache MUST include YTD"
    assert full_row["ab"] == 500.0
