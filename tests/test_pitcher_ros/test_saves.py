import numpy as np
import pandas as pd
import pytest

from fantasy_baseball.pitcher_ros.backtest import _ours_at
from fantasy_baseball.pitcher_ros.saves import (
    SNAPSHOT_WEEK,
    TEAM_COLUMNS,
    fangraphs_shares,
    fangraphs_team_ids,
    fit_team_saves,
    fit_usage_shares,
    net_with_saves,
    scheduled_games_left,
    score_methods,
    team_weeks,
    usage_answer,
)

A, B = 10, 20


def _team_row(player, season, week, team, **cols):
    row = {"player_id": player, "season": season, "week": week, "team_id": team,
           "as_of": pd.Timestamp(season, 4, 1) + pd.Timedelta(weeks=week),
           "season_complete": True}  # fmt: skip
    row.update({c: cols.get(c, 0) for c in TEAM_COLUMNS})
    return row


def _team_table():
    """Team A saves a game a night and wins; team B never saves. 2023-2025, two pitcher
    rows per team-week (the team columns repeat on each)."""
    rows = []
    for season in (2023, 2024, 2025):
        for week in (0, 1):
            played = 7 * week
            first = season == 2023
            for team, sv, w in ((A, 1, 1), (B, 0, 0)):
                for pid in (1, 2):
                    rows.append(
                        _team_row(
                            team + pid,
                            season,
                            week,
                            team,
                            std_team_sv=sv * played,
                            std_team_games=played,
                            std_team_w=w * played,
                            p1_team_sv=0 if first else sv * 21,
                            p1_team_games=0 if first else 21,
                            p1_team_w=0 if first else w * 21,
                            team_games_left=21 - played,
                            ros_team_sv=sv * (21 - played),
                            ros_team_games=21 - played,
                        )
                    )
    return pd.DataFrame(rows)


def test_team_weeks_is_one_row_per_team_and_date():
    tw = team_weeks(_team_table())
    assert len(tw) == 3 * 2 * 2
    assert not tw.duplicated(["team_id", "season", "week"]).any()


def test_team_saves_are_fit_on_earlier_seasons_only():
    tw = team_weeks(_team_table())
    # 2023 has no season before it in the store, so nothing before 2024 has inputs.
    with pytest.raises(ValueError, match="no team-week before 2024"):
        fit_team_saves(tw, 2024)
    rate = fit_team_saves(tw, 2025)
    by_team = tw.loc[rate.index, "team_id"]
    assert len(rate) == 4 and (rate >= 0).all()
    assert rate[by_team == A].min() > rate[by_team == B].max()  # the saving team


def test_scheduled_games_left_counts_every_listed_game():
    sched = pd.DataFrame(
        {
            "season": [2025] * 4,
            "game_date": pd.to_datetime(["2025-04-01", "2025-04-09", "2025-04-10", "2025-04-11"]),
            "home_team_id": [A, A, B, A],
            "away_team_id": [B, B, A, B],
        }
    )
    left = scheduled_games_left(sched, 2025, pd.Timestamp("2025-04-08"))
    assert left[A] == 3 and left[B] == 3


def _fg_file(path, rows):
    path.write_text("Name,Team,SV,MLBAMID,TBF\n" + "\n".join(rows) + "\n", encoding="utf-8")


def test_fangraphs_shares_average_saves_and_shares_over_the_same_systems(tmp_path):
    _fg_file(tmp_path / "steamer-pitchers.csv", ["A,SDP,30,1,260", "B,SDP,10,2,0", "C,,5,3,90"])
    _fg_file(tmp_path / "zips-pitchers.csv", ["A,SDP,20,1,280", "B,,20,2,250"])
    _fg_file(tmp_path / "atc-pitchers.csv", ["A,SDP,,1,300"])  # no saves projected: left out
    fg = fangraphs_shares(tmp_path)
    assert fg.loc[1, "sv_fg"] == 25 and fg.loc[1, "share_fg"] == pytest.approx((0.75 + 1) / 2)
    # On no team in ZiPS: share 0 there, so both averages cover both systems.
    assert fg.loc[2, "sv_fg"] == 15 and fg.loc[2, "share_fg"] == pytest.approx(0.25 / 2)
    assert fg.loc[3, "share_fg"] == 0 and pd.isna(fg.loc[3, "team_fg"])
    assert fg.loc[1, "team_fg"] == "SDP"
    # Batters faced: over the systems with some (Steamer's 0 for B is none).
    assert fg.loc[1, "bf_fg"] == 270 and fg.loc[2, "bf_fg"] == 250
    assert fangraphs_shares(tmp_path, ("oopsy",)) is None


def test_fangraphs_teams_are_matched_to_ours_by_their_pitchers():
    fg = pd.DataFrame({"team_fg": ["SDP", "SDP", "SDP", "NYY"]}, index=[1, 2, 3, 4])
    rows = pd.DataFrame({"player_id": [1, 2, 3, 4], "team_id": [A, A, B, B]})
    ids = fangraphs_team_ids(fg, rows)
    assert ids["SDP"] == A and ids["NYY"] == B


def test_usage_answer_counts_only_saves_for_his_team():
    table = pd.DataFrame({"ros_sv_for_team": [20, 5, 3, 0], "ros_team_sv": [40, 40, 0, 10]})
    share, weight = usage_answer(table)
    assert share[0] == 0.5 and weight[0] == 40
    assert np.isnan(share[2]) and weight[2] == 0  # the team saved none: no answer
    assert share[3] == 0 and weight[3] == 10


def test_usage_shares_pick_their_rounds_on_held_out_pitchers():
    rng = np.random.default_rng(0)
    n = 2000
    table = pd.DataFrame(
        {
            "player_id": rng.integers(0, 300, n),
            "season": rng.choice([2023, 2024, 2025], n),
            "season_complete": True,
        }
    )
    x = pd.DataFrame({"closer": rng.random(n), "noise": rng.random(n)})
    share = (x["closer"] > 0.8).astype(float)
    weight = pd.Series(30.0, index=table.index)
    pred = fit_usage_shares(table, x, share, weight, 2025, seed=1)
    test = table["season"] == 2025
    assert pred.index.equals(table.index[test]) and pred.between(0, 1).all()
    closers = x.loc[test, "closer"] > 0.8
    assert pred[closers].mean() > 0.8 and pred[~closers].mean() < 0.2


def test_score_methods_ranks_the_closer_projection():
    units = pd.DataFrame(
        {"player_id": range(30), "season": 2025, "actual": [10.0] * 30,
         "good": [9.0] * 30, "bad": [2.0] * 30}
    )  # fmt: skip
    table, pairs = score_methods(units, ["good", "bad"], ["player_id", "season"], n_boot=200)
    assert table.loc["good", "typical_miss"] == pytest.approx(1.0)
    r = pairs.iloc[0]
    assert (r["a"], r["b"]) == ("good", "bad") and r["diff"] == pytest.approx(1 - 64)
    assert r["sure"] > 0.95 and r["mse_a"] == pytest.approx(1.0)
    only, none = score_methods(units, ["good"], ["player_id", "season"], pairs=False)
    assert none is None and list(only.columns) == ["mse", "typical_miss"]
    with pytest.raises(ValueError, match="missing"):
        score_methods(units.assign(good=np.nan), ["good", "bad"], ["player_id", "season"])


def test_net_with_saves_swaps_sv_per_span_and_redates_snapshots():
    days = pd.to_datetime(["2025-03-27", "2025-04-03", "2025-04-10"])
    preds = pd.DataFrame(
        {"player_id": [1, 1, 1, 2, 2, 2, 3], "season": 2025, "week": [0, 1, 2, 0, 1, 2, 0],
         "as_of": [*days, *days, days[0]], "k": 0.2, "w": 0.01, "sv": 0.001, "era": 4.0,
         "whip": 1.3}
    )  # fmt: skip
    pre = pd.DataFrame({"player_id": [1, 2], "season": 2025, "mix": [30.0, 3.0],
                        "bf_fg": [250.0, np.nan]})  # fmt: skip
    mid = pd.DataFrame({"player_id": [1, 1], "season": 2025, "mix": [20.0, 10.0],
                        "bf_fg": [200.0, 100.0], "snapshot": ["2025-04-05", "2025-04-06"]})  # fmt: skip
    out = net_with_saves(preds, pre, mid)
    week0 = out[out["week"] == 0].set_index("player_id")["sv"]
    assert len(week0) == 3 and week0[1] == pytest.approx(30 / 250)
    assert week0[2] == 0.001 and week0[3] == 0.001  # no batters faced / no units: the net's
    snaps = out[out["week"] == SNAPSHOT_WEEK]
    # Both snapshots fall in week 1: each gets its own rows, dated to it.
    assert sorted(snaps["as_of"].unique()) == list(pd.to_datetime(["2025-04-05", "2025-04-06"]))
    first = snaps[snaps["as_of"] == pd.Timestamp("2025-04-05")].set_index("player_id")
    assert set(first.index) == {1, 2, 3} and first.loc[1, "sv"] == pytest.approx(0.1)
    assert first.loc[2, "sv"] == 0.001 and first.loc[1, "k"] == 0.2
    with pytest.raises(ValueError, match="more than one row"):
        net_with_saves(preds, pd.concat([pre, pre]), mid)


def test_backtest_reads_each_snapshots_swapped_rows():
    """What backtest._ours_at picks from net_with_saves: the swapped week-0 rows
    preseason; a snapshot's own rows, even one dated like a table week; and on a date
    with no saves units, the net's stats as of then."""
    days = pd.to_datetime(["2025-03-27", "2025-04-03", "2025-04-10"])
    preds = pd.DataFrame(
        {"player_id": 1, "season": 2025, "week": [0, 1, 2], "as_of": days,
         "k": [0.20, 0.21, 0.22], "w": 0.01, "sv": 0.001, "era": 4.0, "whip": 1.3}
    )  # fmt: skip
    pre = pd.DataFrame({"player_id": [1], "season": 2025, "mix": [30.0], "bf_fg": [300.0]})
    mid = pd.DataFrame({"player_id": [1, 1], "season": 2025, "mix": [20.0, 8.0],
                        "bf_fg": [200.0, 100.0], "snapshot": ["2025-04-05", "2025-04-10"]})  # fmt: skip
    ours = {"ours_sv": net_with_saves(preds, pre, mid)}

    def at(when):
        return _ours_at(ours, 2025, when)["ours_sv"].loc[1]

    assert at(None)["sv"] == pytest.approx(0.1) and at(None)["k"] == 0.20
    assert at(pd.Timestamp("2025-04-05"))["sv"] == pytest.approx(0.1)
    assert at(pd.Timestamp("2025-04-05"))["k"] == 0.21
    tie = at(pd.Timestamp("2025-04-10"))  # the snapshot's row, not week 2's
    assert tie["sv"] == pytest.approx(0.08) and tie["k"] == 0.22
    later = at(pd.Timestamp("2025-04-07"))  # no units: week 1's stats, 04-05's SV
    assert later["k"] == 0.21 and later["sv"] == pytest.approx(0.1)
