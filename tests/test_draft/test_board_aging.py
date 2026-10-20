import json
import logging
import sqlite3

import numpy as np
import pandas as pd
import pytest

from fantasy_baseball.draft import board as board_mod
from fantasy_baseball.pitcher_ros import aging


def _pitchers():
    return pd.DataFrame(
        {
            "fg_id": ["1", "2", "3"],
            "name": ["Old Ace", "Young Ace", "Old Closer"],
            "ip": [180.0, 180.0, 60.0],
            "k": [200.0, 200.0, 70.0],
            "w": [12.0, 12.0, 3.0],
            "sv": [0.0, 0.0, 30.0],
            "er": [60.0, 60.0, 20.0],
            "bb": [45.0, 45.0, 20.0],
            "h_allowed": [150.0, 150.0, 45.0],
            "era": [3.0, 3.0, 3.0],
            "whip": [1.083333, 1.083333, 1.083333],
        }
    )


FIT = {
    ("starter", "era"): aging.Hinge(36.0, 0.1, -0.03, 42.0),
    ("starter", "whip"): aging.Hinge(36.0, 0.03, 0.0, 42.0),
}
IP_FIT = {"reliever": aging.Hinge(36.0, -0.08, 0.66, 41.0)}


def test_totals_follow_the_corrected_rates_and_innings():
    p = _pitchers()
    age = pd.Series([40.0, 28.0, 39.0])
    roles = aging.role_from_projected_ip(p["ip"])
    out = aging.adjust_totals(p, age, roles, FIT, IP_FIT)
    # The old starter: ERA +0.4, WHIP +0.12, same innings; ER and baserunners follow.
    assert out.loc[0, "era"] == pytest.approx(3.4) and out.loc[0, "ip"] == 180.0
    assert out.loc[0, "er"] == pytest.approx(3.4 * 180 / 9)
    assert out.loc[0, "h_allowed"] + out.loc[0, "bb"] == pytest.approx((1.083333 + 0.12) * 180)
    assert out.loc[0, "h_allowed"] / out.loc[0, "bb"] == pytest.approx(150 / 45)
    # The young one is untouched.
    pd.testing.assert_series_equal(out.loc[1], p.loc[1], check_names=False)
    # The old closer: innings cut by 3 x 0.08 / 0.66, saves and strikeouts with them.
    share = 1 - 3 * 0.08 / 0.66
    assert out.loc[2, "ip"] == pytest.approx(60 * share) and out.loc[2, "sv"] == pytest.approx(
        30 * share
    )
    assert out.loc[2, "k"] == pytest.approx(70 * share) and out.loc[2, "era"] == pytest.approx(3.0)


def test_a_fit_round_trips_through_json():
    data = json.loads(
        json.dumps(
            aging.fit_to_dict({**FIT, ("reliever", "k"): aging.Hinge(None, 0.0, 0.0)}, IP_FIT)
        )
    )
    rates, innings = aging.fit_from_dict(data)
    assert rates[("starter", "era")] == FIT[("starter", "era")] and innings == IP_FIT
    assert rates[("reliever", "k")].knot is None and np.isinf(rates[("reliever", "k")].top)


def _conn():
    conn = sqlite3.connect(":memory:")
    conn.row_factory = sqlite3.Row
    conn.execute("CREATE TABLE blended_projections (year INTEGER, player_type TEXT)")
    conn.execute("INSERT INTO blended_projections VALUES (2027, 'pitcher')")
    conn.execute(
        "CREATE TABLE raw_projections (year INTEGER, player_type TEXT, fg_id TEXT, mlbam_id INTEGER)"
    )
    conn.executemany(
        "INSERT INTO raw_projections VALUES (2027, 'pitcher', ?, ?)",
        [("1", 101), ("2", 102), ("3", 103)],
    )
    return conn


def test_the_board_ages_its_pitchers_from_their_ids_and_every_people_file(tmp_path, monkeypatch):
    fit = tmp_path / "fit.json"
    fit.write_text(json.dumps(aging.fit_to_dict(FIT, IP_FIT)))
    people = tmp_path / "people"
    people.mkdir()
    # The ace's birth date is only in the wider file; the closer's only in the narrow one.
    pd.DataFrame({"id": [101, 102], "birthDate": ["1987-01-01", "1999-01-01"]}).to_csv(
        people / "mlb_people_all_2000_2026.csv", index=False
    )
    pd.DataFrame({"id": [103], "birthDate": ["1988-01-01"]}).to_csv(
        people / "mlb_people_all_2025_2025.csv", index=False
    )
    monkeypatch.setattr(board_mod, "PEOPLE_DIR", people)
    out = board_mod.age_correct_pitchers(_conn(), _pitchers(), fit)
    assert out.loc[0, "era"] > 3.0 and out.loc[1, "era"] == 3.0 and out.loc[2, "ip"] < 60.0


def test_no_fit_or_no_people_leaves_the_board_as_it_is(tmp_path, monkeypatch, caplog):
    monkeypatch.setattr(board_mod, "PEOPLE_DIR", tmp_path / "nobody")
    with caplog.at_level(logging.WARNING):
        same = board_mod.age_correct_pitchers(_conn(), _pitchers(), tmp_path / "missing.json")
    pd.testing.assert_frame_equal(same, _pitchers())
    fit = tmp_path / "fit.json"
    fit.write_text(json.dumps(aging.fit_to_dict(FIT, IP_FIT)))
    with caplog.at_level(logging.WARNING):
        same = board_mod.age_correct_pitchers(_conn(), _pitchers(), fit)
    pd.testing.assert_frame_equal(same, _pitchers())
    assert "no pitcher age correction" in caplog.text and "no people file" in caplog.text


def test_the_shipped_fit_reads():
    rates, innings = aging.fit_from_dict(json.loads(board_mod.AGING_FIT.read_text()))
    assert set(innings) == set(aging.ROLES) and ("starter", "era") in rates
