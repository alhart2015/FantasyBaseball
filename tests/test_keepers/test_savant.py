from datetime import date
from pathlib import Path

import pandas as pd
import pytest

from fantasy_baseball.keepers.savant import (
    _tally_pitch_outcomes,
    fetch_batter_barrels,
    fetch_batter_expected,
    fetch_pitcher_expected,
    fetch_pitcher_pitch_mix,
    fetch_savant_hr,
)


def _pitches(*descriptions: str, pitcher: int = 7, game_type: str = "R") -> pd.DataFrame:
    return pd.DataFrame(
        {"pitcher": pitcher, "description": list(descriptions), "game_type": game_type}
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


def test_foul_tip_is_a_swing_but_not_a_whiff():
    """It is contact. Counting it as a whiff would inflate whiff rate and CSW%
    for exactly the pitchers who generate the most of them."""
    out = _tally_pitch_outcomes(_pitches("foul_tip"))
    assert out["swings"].iloc[0] == 1
    assert out["whiffs"].iloc[0] == 0


def test_blocked_swinging_strike_and_missed_bunt_count_as_whiffs():
    out = _tally_pitch_outcomes(_pitches("swinging_strike_blocked", "missed_bunt"))
    assert out["whiffs"].iloc[0] == 2
    assert out["swings"].iloc[0] == 2


def test_taken_pitches_are_not_swings():
    out = _tally_pitch_outcomes(_pitches("ball", "called_strike", "blocked_ball", "hit_by_pitch"))
    assert out["pitches"].iloc[0] == 4
    assert out["swings"].iloc[0] == 0
    assert out["called_strikes"].iloc[0] == 1


def test_outcomes_tallied_per_pitcher():
    raw = pd.concat([_pitches("swinging_strike", "ball", pitcher=1), _pitches("foul", pitcher=2)])
    out = _tally_pitch_outcomes(raw).set_index("player_id")
    assert out.loc[1, "pitches"] == 2
    assert out.loc[1, "whiffs"] == 1
    assert out.loc[2, "pitches"] == 1
    assert out.loc[2, "whiffs"] == 0


def test_pitch_mix_raw_passthrough(tmp_path: Path):
    raw = pd.DataFrame({"player_id": [1], "pitches": [2800], "whiffs": [280]})
    out = fetch_pitcher_pitch_mix(tmp_path, 2026, fetcher=lambda: raw)
    pd.testing.assert_frame_equal(out, raw)
    # Versioned: a v1 cache predates the spring-training exclusion.
    assert (tmp_path / "savant_pitcher_pitch_mix_2026.v2.csv").exists()
    assert not (tmp_path / "savant_pitcher_pitch_mix_2026.csv").exists()


def test_spring_training_and_postseason_pitches_are_excluded():
    """statcast() returns S and P alongside R and its date window cannot exclude
    them. Spring whiff rates run high against non-roster hitters, and the BBRef
    stats these sit beside are regular-season only."""
    raw = pd.concat(
        [
            _pitches("swinging_strike", "ball"),
            _pitches("swinging_strike", "swinging_strike", game_type="S"),
            _pitches("swinging_strike", game_type="P"),
        ]
    )
    out = _tally_pitch_outcomes(raw)
    assert out["pitches"].iloc[0] == 2
    assert out["whiffs"].iloc[0] == 1


def test_an_all_spring_frame_tallies_to_nothing():
    assert _tally_pitch_outcomes(_pitches("swinging_strike", game_type="S")).empty


def test_pre_2020_balls_in_play_still_count_as_swings():
    """Statcast split balls in play three ways before 2020; the function takes a
    year, so dropping the old spellings would gut the swing denominator."""
    out = _tally_pitch_outcomes(_pitches("hit_into_play_score", "hit_into_play_no_out"))
    assert out["swings"].iloc[0] == 2
    assert out["whiffs"].iloc[0] == 0


def test_all_spring_fold_does_not_crash_the_season_pull(monkeypatch, tmp_path: Path):
    """A run between the first spring game and Opening Day tallies every chunk to
    an empty 0x0 frame. Concat-ing those loses `player_id` and the groupby raised
    KeyError instead of the clean RuntimeError fetch_or_cache is meant to give."""
    import fantasy_baseball.keepers.savant as savant

    spring = _pitches("swinging_strike", "ball", game_type="S")
    monkeypatch.setattr(savant, "local_today", lambda: date(2026, 3, 20))
    monkeypatch.setattr("pybaseball.statcast", lambda **kwargs: spring, raising=False)

    assert savant._savant_pitcher_pitch_mix(2026).empty
    with pytest.raises(RuntimeError, match="returned empty"):
        fetch_pitcher_pitch_mix(
            tmp_path, 2026, fetcher=lambda: savant._savant_pitcher_pitch_mix(2026)
        )


def test_every_season_to_date_fetcher_accepts_a_max_age_override():
    """--refresh works by passing max_age=0, so a fetcher without the parameter
    is silently skipped by it -- which is how the two batter leaderboards went
    un-refreshed while the other three re-pulled."""
    import inspect

    from fantasy_baseball.keepers import bref

    fetchers = (
        fetch_batter_expected,
        fetch_batter_barrels,
        fetch_pitcher_pitch_mix,
        bref.fetch_bref_batting,
        bref.fetch_bref_pitching,
    )
    for fn in fetchers:
        params = inspect.signature(fn).parameters
        assert "max_age" in params, f"{fn.__name__} cannot be refreshed or aged out"
        assert params["max_age"].default is not None, f"{fn.__name__} has no staleness guard"


def test_package_exports_all_resolve():
    """Replaces two partial per-module copies; asserts the whole surface."""
    import fantasy_baseball.keepers as k

    assert k.__all__ == sorted(k.__all__)
    for name in k.__all__:
        assert hasattr(k, name), f"{name} in __all__ but not importable"
