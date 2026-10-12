import numpy as np
import pandas as pd
import pytest

from fantasy_baseball.pitcher_ros.saves import (
    fangraphs_shares,
    fit_team_saves,
    score_methods,
    team_games,
    team_weeks,
    usage_answer,
)

A, B = 10, 20


def _games():
    """Two teams, 2023-2025, a game a day for 10 days. Team A wins every game and
    its closer (player 1) saves every one; team B saves none."""
    rows = []
    for season in (2023, 2024, 2025):
        for i in range(10):
            d = pd.Timestamp(season, 4, 1) + pd.Timedelta(days=i)
            pk = season * 100 + i
            rows.append(
                {
                    "player_id": 1,
                    "team_id": A,
                    "game_pk": pk,
                    "game_date": d,
                    "season": season,
                    "sv": 1,
                    "svo": 1,
                    "w": 0,
                }
            )
            rows.append(
                {
                    "player_id": 2,
                    "team_id": A,
                    "game_pk": pk,
                    "game_date": d,
                    "season": season,
                    "sv": 0,
                    "svo": 0,
                    "w": 1,
                }
            )
            rows.append(
                {
                    "player_id": 3,
                    "team_id": B,
                    "game_pk": pk,
                    "game_date": d,
                    "season": season,
                    "sv": 0,
                    "svo": 0,
                    "w": 0,
                }
            )
    return pd.DataFrame(rows)  # fmt: skip


def test_team_games_and_weeks_split_at_the_date():
    tg = team_games(_games())
    assert len(tg) == 2 * 3 * 10
    a = tg[(tg.team_id == A) & (tg.season == 2025)]
    assert (a.sv == 1).all() and (a.w == 1).all()
    grid = pd.DataFrame({"season": [2025], "week": [1], "as_of": [pd.Timestamp(2025, 4, 8)]})
    tw = team_weeks(tg, grid).set_index("team_id")
    assert tw.loc[A, "std_sv"] == 7 and tw.loc[A, "ros_sv"] == 3 and tw.loc[A, "ros_g"] == 3
    assert tw.loc[A, "p1_sv"] == 10 and tw.loc[A, "p1_g"] == 10 and tw.loc[B, "std_w"] == 0


def test_team_saves_are_fit_on_earlier_seasons_only():
    tg = team_games(_games())
    grid = pd.DataFrame(
        {
            "season": [2023, 2024, 2025],
            "week": [0, 0, 0],
            "as_of": pd.to_datetime(["2023-04-01", "2024-04-01", "2025-04-01"]),
        }
    )
    tw = team_weeks(tg, grid)
    # 2023 has no season before it in the store, so nothing before 2024 has inputs.
    with pytest.raises(ValueError, match="no team-week before 2024"):
        fit_team_saves(tw, 2024, {2023, 2024, 2025})
    rate = fit_team_saves(tw, 2025, {2023, 2024, 2025})
    assert len(rate) == 2 and (rate >= 0).all()


def _fg_file(path, rows):
    path.write_text("Name,Team,SV,MLBAMID\n" + "\n".join(rows) + "\n", encoding="utf-8")


def test_fangraphs_shares_are_saves_over_the_team_total(tmp_path):
    _fg_file(tmp_path / "steamer-pitchers.csv", ["A,SDP,30,1", "B,SDP,10,2", "C,,5,3"])
    _fg_file(tmp_path / "zips-pitchers.csv", ["A,SDP,20,1", "B,SDP,20,2"])
    _fg_file(tmp_path / "atc-pitchers.csv", ["A,SDP,,1"])  # no saves projected: left out
    fg = fangraphs_shares(tmp_path)
    assert fg.loc[1, "sv_fg"] == 25 and fg.loc[1, "share_fg"] == pytest.approx((0.75 + 0.5) / 2)
    assert fg.loc[3, "share_fg"] == 0  # no FanGraphs team
    only = fangraphs_shares(tmp_path, ("zips",))
    assert only.loc[2, "share_fg"] == pytest.approx(0.5)
    assert fangraphs_shares(tmp_path, ("oopsy",)) is None


def test_usage_answer_is_his_share_of_his_teams_saves():
    table = pd.DataFrame({"ros_sv": [30, 5, 3, 0]})
    team = pd.Series([40.0, 40.0, 0.0, np.nan])
    share, weight = usage_answer(table, team)
    assert share[0] == 0.75 and weight[0] == 40
    assert np.isnan(share[2]) and weight[2] == 0  # the team saved none: no answer
    assert weight[3] == 0


def test_score_methods_ranks_the_closer_projection():
    units = pd.DataFrame(
        {
            "player_id": range(30),
            "season": 2025,
            "actual": [10.0] * 30,
            "good": [9.0] * 30,
            "bad": [2.0] * 30,
        }
    )
    table, pairs = score_methods(units, ["good", "bad"], ["player_id", "season"], n_boot=200)
    assert table.loc["good", "typical_miss"] == pytest.approx(1.0)
    r = pairs.iloc[0]
    assert (r["a"], r["b"]) == ("good", "bad") and r["diff"] == pytest.approx(1 - 64)
    assert r["sure"] > 0.95
    with pytest.raises(ValueError, match="missing"):
        score_methods(units.assign(good=np.nan), ["good", "bad"], ["player_id", "season"])
