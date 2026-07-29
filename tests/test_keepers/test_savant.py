from pathlib import Path

import pandas as pd

from fantasy_baseball.keepers.savant import (
    fetch_batter_barrels,
    fetch_batter_expected,
    fetch_pitcher_expected,
    fetch_pitcher_pitch_mix,
    fetch_savant_hr,
    tally_pitch_outcomes,
)


def _pitches(*descriptions: str, pitcher: int = 7) -> pd.DataFrame:
    return pd.DataFrame({"pitcher": pitcher, "description": list(descriptions)})


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


def test_foul_tip_is_a_swing_but_not_a_whiff():
    """It is contact. Counting it as a whiff would inflate whiff rate and CSW%
    for exactly the pitchers who generate the most of them."""
    out = tally_pitch_outcomes(_pitches("foul_tip"))
    assert out["swings"].iloc[0] == 1
    assert out["whiffs"].iloc[0] == 0


def test_blocked_swinging_strike_and_missed_bunt_count_as_whiffs():
    out = tally_pitch_outcomes(_pitches("swinging_strike_blocked", "missed_bunt"))
    assert out["whiffs"].iloc[0] == 2
    assert out["swings"].iloc[0] == 2


def test_taken_pitches_are_not_swings():
    out = tally_pitch_outcomes(_pitches("ball", "called_strike", "blocked_ball", "hit_by_pitch"))
    assert out["pitches"].iloc[0] == 4
    assert out["swings"].iloc[0] == 0
    assert out["called_strikes"].iloc[0] == 1


def test_outcomes_tallied_per_pitcher():
    raw = pd.concat([_pitches("swinging_strike", "ball", pitcher=1), _pitches("foul", pitcher=2)])
    out = tally_pitch_outcomes(raw).set_index("player_id")
    assert out.loc[1, "pitches"] == 2
    assert out.loc[1, "whiffs"] == 1
    assert out.loc[2, "pitches"] == 1
    assert out.loc[2, "whiffs"] == 0


def test_pitch_mix_raw_passthrough(tmp_path: Path):
    raw = pd.DataFrame({"player_id": [1], "pitches": [2800], "whiffs": [280]})
    out = fetch_pitcher_pitch_mix(tmp_path, 2026, fetcher=lambda: raw)
    pd.testing.assert_frame_equal(out, raw)
    assert (tmp_path / "savant_pitcher_pitch_mix_2026.csv").exists()


def test_public_api_reexports():
    import fantasy_baseball.keepers as k

    expected = [
        "fetch_or_cache",
        "fetch_mlb_season",
        "fetch_batter_expected",
        "fetch_batter_barrels",
        "fetch_pitcher_expected",
        "fetch_pitcher_pitch_mix",
        "fetch_savant_hr",
    ]
    for name in expected:
        assert hasattr(k, name), f"{name} not re-exported from fantasy_baseball.keepers"
        assert name in k.__all__, f"{name} missing from __all__"
