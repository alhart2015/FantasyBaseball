"""#433: prospect pedigree -- MLB Pipeline lists and June-draft picks as hitter inputs."""

from datetime import date

import numpy as np
import pandas as pd
import pytest

from fantasy_baseball.hitter_ros.pedigree_features import (
    PEDIGREE_FEATURES,
    build_pedigree_features,
)
from fantasy_baseball.pitch_data import pedigree
from fantasy_baseball.pitch_data.pedigree import draft_rows, rankings_rows

STAR, FRINGE, SIGNEE, VET = 1, 2, 3, 4


def _entry(player_id):
    return {"title": "x", "fields": {"playerId": player_id, "eta": "2099"}}


def test_rankings_keep_list_order_and_skip_entries_without_an_id():
    df = rankings_rows(2020, "top100", [_entry(10), {"fields": {}}, _entry(30)])
    assert df["player_id"].tolist() == [10, 30]
    assert df["rank"].tolist() == [1, 3]  # the blank entry still held rank 2
    assert (df["list_size"] == 3).all() and (df["season"] == 2020).all()
    assert "eta" not in df.columns  # today's page: would leak later seasons


def test_a_list_under_another_name_is_found(monkeypatch):
    import requests

    def fake(url, params=None):
        if url.endswith("sel-pr-2014-bluejays"):
            resp = requests.Response()
            resp.status_code = 404
            raise requests.HTTPError(response=resp)
        return {"items": [_entry(5)], "pagination": {}}

    monkeypatch.setattr(pedigree, "_get_json", fake)
    assert [i["fields"]["playerId"] for i in pedigree._list_items(2014, "bluejays")] == [5]


def test_draft_rows_keep_june_picks_with_a_player():
    def pick(pid, number, code="JR", **extra):
        return {
            "person": {"id": pid, "birthDate": "2000-01-01"} if pid else {},
            "pickRound": "1",
            "pickNumber": number,
            "draftType": {"code": code},
            **extra,
        }

    data = {
        "drafts": {
            "rounds": [
                {
                    "picks": [
                        pick(7, 3, signingBonus="100000"),
                        pick(None, 4),
                        pick(8, 5, isPass=True),
                        pick(9, 6, code="NS"),
                    ]
                }
            ]
        }
    }
    df = draft_rows(2019, data)
    assert df["player_id"].tolist() == [7]
    assert df["pick_number"].tolist() == [3] and df["signing_bonus"].tolist() == [100000]
    # A year with every bonus blank still stacks with the others.
    assert str(draft_rows(1999, data)["signing_bonus"].dtype) == "Int64"


def _rankings():
    rows = [
        (2019, "top100", 1, STAR),
        (2019, "pirates", 1, STAR),
        (2020, "pirates", 2, STAR),  # off the Top 100 in 2020
        (2021, "top100", 5, STAR),  # this season's list: must not count for 2021
        (2020, "pirates", 16, FRINGE),
    ]
    df = pd.DataFrame(rows, columns=["season", "list", "rank", "player_id"])
    return df.assign(list_size=np.where(df["list"] == "top100", 100, 30))


def _draft():
    return pd.DataFrame(
        {
            "year": [2015, 2018, 2021, 2009],
            "player_id": [STAR, STAR, STAR, FRINGE],
            "pick_number": [400, 1, 9, 300],
            "birth_date": ["1999-07-01"] * 3 + ["1988-01-01"],
        }
    )


def _table(season=2021):
    ids = [STAR, FRINGE, SIGNEE, VET]
    return pd.DataFrame(
        {
            "player_id": ids * 2,
            "season": season,
            "week": [0] * 4 + [3] * 4,
            "as_of": [date(season, 3, 28)] * 4 + [date(season, 4, 18)] * 4,
            "car_pa": [0, 50, 10, 2000] * 2,
            "std_pa": [0, 0, 0, 0, 40, 30, 20, 60],
            "car_seasons_in_store": 10,
        }
    )


def _row(out, player, week=0):
    return out[(out["player_id"] == player) & (out["week"] == week)].iloc[0]


def test_only_lists_and_drafts_before_the_season_count():
    out = build_pedigree_features(_table(), _rankings(), _draft())
    star = _row(out, STAR)
    assert star["ped_t100_p1"] == 0.0  # off last season's Top 100; 2021's list ignored
    assert star["ped_t100_best"] == 1.0  # No. 1 in 2019
    assert star["ped_org_p1"] == pytest.approx(1 - 1 / 30)
    assert star["ped_org_best"] == 1.0
    assert star["ped_seasons_listed"] == 2
    # His last draft before 2021 is 2018's (No. 1 overall), not 2021's.
    assert star["ped_draft_log_pick"] == 0.0
    assert star["ped_draft_years"] == 3
    assert star["ped_draft_age"] == pytest.approx(19.0, abs=0.01)
    fringe = _row(out, FRINGE)
    assert fringe["ped_org_p1"] == pytest.approx(1 - 15 / 30)
    assert fringe["ped_t100_best"] == 0.0
    # Never ranked, never drafted: lists known (0), draft blank.
    signee = _row(out, SIGNEE)
    assert signee[PEDIGREE_FEATURES[:5]].tolist() == [0.0] * 5
    assert signee[PEDIGREE_FEATURES[5:]].isna().all()
    # The same for every week of the season.
    assert _row(out, STAR, 3)[PEDIGREE_FEATURES].equals(star[PEDIGREE_FEATURES])


def test_before_the_first_lists_the_list_inputs_are_unknown():
    out = build_pedigree_features(_table(2011), _rankings(), _draft())
    assert _row(out, STAR)[PEDIGREE_FEATURES[:5]].isna().all()
    assert _row(out, FRINGE)["ped_draft_log_pick"] == pytest.approx(np.log(300))


def test_vets_get_no_pedigree():
    table = _table()
    out = build_pedigree_features(table, _rankings(), _draft(), vet_min_pa=300)
    assert _row(out, VET)[PEDIGREE_FEATURES].isna().all()
    assert _row(out, STAR)["ped_t100_best"] == 1.0
    # A career count that can't be trusted counts as a vet.
    short = table.assign(car_seasons_in_store=2)
    blank = build_pedigree_features(short, _rankings(), _draft(), vet_min_pa=300)
    assert blank[PEDIGREE_FEATURES].isna().all().all()
    assert list(out.columns) == ["player_id", "season", "week", "as_of", *PEDIGREE_FEATURES]


def test_pedigree_setting_defaults_off():
    pytest.importorskip("torch")
    from fantasy_baseball.hitter_ros.net import NetConfig

    assert NetConfig().pedigree == "none"


def test_a_club_list_past_30_never_ranks_below_off_list():
    rankings = pd.concat(
        [
            _rankings(),
            pd.DataFrame(
                {"season": [2020], "list": ["angels"], "rank": [42], "player_id": [SIGNEE]}
            ).assign(list_size=42),
        ],
        ignore_index=True,
    )
    signee = _row(build_pedigree_features(_table(), rankings, _draft()), SIGNEE)
    assert signee["ped_org_p1"] == 0.0 and signee["ped_org_best"] == 0.0
    assert signee["ped_seasons_listed"] == 1


def test_a_season_missing_from_the_store_is_unknown_not_unlisted():
    # 2020's lists never fetched: a 2021 row can't tell "off the list" from "no data".
    rankings = _rankings().query("season != 2020")
    out = build_pedigree_features(_table(), rankings, _draft())
    assert _row(out, STAR)[PEDIGREE_FEATURES[:5]].isna().all()
    assert _row(out, STAR)["ped_draft_log_pick"] == 0.0


def test_draft_rows_drop_a_repeated_pick():
    pick = {
        "person": {"id": 7, "birthDate": "2000-01-01"},
        "pickRound": "4",
        "pickNumber": 127,
        "draftType": {"code": "JR"},
    }
    df = draft_rows(2008, {"drafts": {"rounds": [{"picks": [pick, dict(pick)]}]}})
    assert df["player_id"].tolist() == [7]
    assert df["round"].tolist() == ["4"]


def test_a_season_fetched_while_current_is_fetched_again_once_over(tmp_path, monkeypatch):
    import os

    calls = []

    def fake(season, name):
        calls.append(name)
        return [_entry(5)]

    monkeypatch.setattr(pedigree, "_list_items", fake)
    path = pedigree.rankings_path(tmp_path, 2026)
    pedigree.fetch_rankings_season(tmp_path, 2026, today=date(2026, 6, 1))
    n_lists = len(calls)
    mid_2026 = date(2026, 6, 1)
    stamp = float(np.datetime64(mid_2026, "s").astype("int64"))
    os.utime(path, (stamp, stamp))
    # Still 2026: the in-season copy is kept.
    pedigree.fetch_rankings_season(tmp_path, 2026, today=date(2026, 9, 1))
    assert len(calls) == n_lists
    # 2026 is over: the edited copy is replaced.
    pedigree.fetch_rankings_season(tmp_path, 2026, today=date(2027, 2, 1))
    assert len(calls) == 2 * n_lists
