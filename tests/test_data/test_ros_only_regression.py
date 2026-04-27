"""Regression test for ROS-vs-full-season cache shape.

Captures the bug where ``cache:ros_projections`` held YTD-inflated
full-season stats. After the fix, ``cache:ros_projections`` MUST hold
ROS-remaining-only stats, and ``cache:full_season_projections`` MUST
hold the YTD-added totals.

Exercises the real entry point — ``blend_and_cache_ros`` — using a
minimal in-tmp projections snapshot, so any future regression in the
pipeline (not just the typed setters) is caught.
"""

from __future__ import annotations


def test_ros_cache_excludes_ytd(tmp_path, monkeypatch):
    """A player with YTD games should have cached ROS == FanGraphs ROS-only.

    Setup: a hitter projected for 100 R remaining-games-only by FanGraphs,
    with 30 R already accumulated YTD. The cached ROS blob should be 100,
    not 130. The cached full-season blob should be 130.
    """
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

    ros_cache = rs.get_ros_projections(kv)
    full_cache = rs.get_full_season_projections(kv)

    assert ros_cache is not None and full_cache is not None
    ros_row = next(p for p in ros_cache["hitters"] if p.get("mlbam_id") == 12345)
    full_row = next(p for p in full_cache["hitters"] if p.get("mlbam_id") == 12345)

    assert ros_row["r"] == 100.0, "ros cache must NOT include YTD"
    assert ros_row["ab"] == 400.0
    assert full_row["r"] == 130.0, "full-season cache MUST include YTD"
    assert full_row["ab"] == 500.0
