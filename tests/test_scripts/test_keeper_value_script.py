import argparse
import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))

import keeper_value as script


def test_discounts_arg_parses_and_validates():
    assert script._discounts_arg("0.6,0.8,0.9") == [0.6, 0.8, 0.9]
    assert script._discounts_arg("0.5") == [0.5]
    assert script._discounts_arg("1.0") == [1.0]  # 1.0 = no discount, allowed
    for bad in ["0", "1.5", "-0.2", "abc", ""]:
        with pytest.raises(argparse.ArgumentTypeError):
            script._discounts_arg(bad)


def test_parse_args_defaults_and_overrides():
    default = script._parse_args([])
    assert default.horizon == 3
    assert default.discount == [0.60, 0.70, 0.80, 0.90]
    custom = script._parse_args(["--horizon", "2", "--discount", "0.7,0.95"])
    assert custom.horizon == 2
    assert custom.discount == [0.7, 0.95]


def test_load_zips_year_missing_raises_with_url(tmp_path):
    with pytest.raises(FileNotFoundError) as exc:
        script.load_zips_year(tmp_path, 2027)
    assert "fangraphs.com" in str(exc.value)
    assert "2027" in str(exc.value)


def test_load_zips_year_loads_present(tmp_path):
    d = tmp_path / "2027"
    d.mkdir()
    pd.DataFrame(
        [{"Name": "A B", "AB": 500, "H": 150, "HR": 30, "R": 90, "RBI": 95, "SB": 10, "AVG": 0.300}]
    ).to_csv(d / "zips-hitters.csv", index=False)
    # FanGraphs (and ZiPS) exports use "SO" for strikeouts; PITCHING_COLUMN_MAP
    # normalizes SO -> k. A "K" header would NOT be recognized.
    pd.DataFrame(
        [{"Name": "C D", "IP": 180, "W": 14, "SO": 200, "ERA": 3.2, "WHIP": 1.05, "SV": 0}]
    ).to_csv(d / "zips-pitchers.csv", index=False)
    hitters, pitchers = script.load_zips_year(tmp_path, 2027)
    assert not hitters.empty and not pitchers.empty


def test_resolve_candidate_ids_is_collision_safe_by_var():
    # Two same-normalized-name players; find_keeper_match must pick the higher-VAR
    # one, so the highlight resolves to exactly that player_id (never both).
    board = pd.DataFrame(
        [
            {
                "name": "Star Guy",
                "player_id": "star::hitter",
                "name_normalized": "star guy",
                "var": 12.0,
            },
            {
                "name": "Star Guy",
                "player_id": "scrub::hitter",
                "name_normalized": "star guy",
                "var": 1.0,
            },
            {"name": "Other", "player_id": "other::hitter", "name_normalized": "other", "var": 5.0},
        ]
    )
    ids = script.resolve_candidate_ids(board, ["Star Guy"])
    assert ids == {"star::hitter"}  # higher-VAR match only, not the namesake


def test_zips_index_disambiguates_same_name_by_fg_id():
    # Two same-name/same-type hitters distinguished only by fg_id must NOT collapse:
    # each resolves to its own line via lookup_rank(fg_id, ...).
    hitters = pd.DataFrame(
        [
            {"name": "Max Muncy", "fg_id": "111", "hr": 35, "ab": 500},
            {"name": "Max Muncy", "fg_id": "222", "hr": 10, "ab": 300},
        ]
    )
    idx = script.zips_index(hitters, pd.DataFrame())
    indices = {2027: idx}
    a = script._zips_by_year("111", "Max Muncy", "hitter", indices)[2027]
    b = script._zips_by_year("222", "Max Muncy", "hitter", indices)[2027]
    assert a["hr"] == 35 and b["hr"] == 10  # distinct lines, no last-write-wins collapse


def test_zips_by_year_missing_player_is_none():
    idx = script.zips_index(
        pd.DataFrame([{"name": "Someone", "fg_id": "999", "hr": 20, "ab": 400}]), pd.DataFrame()
    )
    got = script._zips_by_year("000", "Nobody Here", "hitter", {2027: idx})
    assert got[2027] is None  # unknown fg_id + unknown name -> None (per_year_var flags it)
