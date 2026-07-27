from pathlib import Path

import pandas as pd

from fantasy_baseball.keepers.savant import (
    fetch_batter_barrels,
    fetch_batter_expected,
    fetch_pitcher_expected,
    fetch_savant_hr,
)


def test_batter_expected_raw_passthrough_no_rename(tmp_path: Path):
    raw = pd.DataFrame({"player_id": [1], "est_woba": [0.350], "woba": [0.330]})
    out = fetch_batter_expected(tmp_path, 2024, fetcher=lambda: raw)
    pd.testing.assert_frame_equal(out, raw)  # est_woba NOT renamed to xwoba
    assert (tmp_path / "savant_batter_expected_2024.csv").exists()


def test_batter_barrels_no_unit_conversion(tmp_path: Path):
    raw = pd.DataFrame({"player_id": [1], "brl_percent": [12.0], "brl_pa": [4.9]})
    out = fetch_batter_barrels(tmp_path, 2024, fetcher=lambda: raw)
    assert out["brl_percent"].iloc[0] == 12.0  # NOT divided by 100
    assert out["brl_pa"].iloc[0] == 4.9


def test_pitcher_expected_raw_passthrough(tmp_path: Path):
    raw = pd.DataFrame({"player_id": [1], "est_woba": [0.300]})
    out = fetch_pitcher_expected(tmp_path, 2024, fetcher=lambda: raw)
    pd.testing.assert_frame_equal(out, raw)


def test_savant_hr_tolerates_empty_pre_2016(tmp_path: Path):
    out = fetch_savant_hr(tmp_path, 2015, fetcher=lambda: pd.DataFrame())
    assert out.empty
    assert not (tmp_path / "savant_hr_2015.csv").exists()


def test_public_api_reexports():
    import fantasy_baseball.keepers as k

    expected = [
        "fetch_or_cache",
        "fetch_mlb_season",
        "fetch_batter_expected",
        "fetch_batter_barrels",
        "fetch_pitcher_expected",
        "fetch_savant_hr",
    ]
    for name in expected:
        assert hasattr(k, name), f"{name} not re-exported from fantasy_baseball.keepers"
        assert name in k.__all__, f"{name} missing from __all__"
