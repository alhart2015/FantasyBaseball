"""#433: zone / chase inputs from one fixed strike-zone box, not Savant's zone numbers."""

from datetime import date

import numpy as np
import pytest

from fantasy_baseball.hitter_ros.features import FIXED_ZONE, input_frame
from fantasy_baseball.hitter_ros.table import build_table
from tests.test_hitter_ros.test_table import HITTER, _row, _season, _write


def _high_strikes(season):
    """Every in-zone pitch crosses the plate above the fixed box's top: Savant's zone
    (from a lowered sz_top, as with 2026's ABS zone) calls it in, the fixed box doesn't."""
    lineups, pitches = season
    for p in pitches:
        if p["zone"] is not None and 1 <= p["zone"] <= 9:
            p["plate_z"] = FIXED_ZONE[2] + 0.3
    return lineups, pitches


@pytest.fixture(scope="module")
def table(tmp_path_factory):
    root = tmp_path_factory.mktemp("store")
    seasons = {2024: _season(2024, date(2024, 4, 1), 10), 2025: _season(2025, date(2025, 4, 1), 21)}
    _write(root, {2024: seasons[2024], 2025: _high_strikes(seasons[2025])})
    return build_table(root)


def test_fixed_box_counts_by_location_not_zone_number(table):
    w1 = _row(table, HITTER, 2025, 1)
    # 3 pitches per game over 7 games: 2 with zone 5 (swinging strike, ball in play),
    # 1 with zone 12. Savant: 14 in the zone. Fixed box: the 14 are too high, so chases.
    assert w1.std_zone_pitches == 14 and w1.std_fzone_pitches == 0
    assert w1.std_chase_pitches == 7 and w1.std_fchase_pitches == 21
    assert w1.std_fchase_swings == w1.std_zone_swings + w1.std_chase_swings
    # Before the change, the two agree.
    p1 = _row(table, HITTER, 2025, 0)
    assert p1.p1_fzone_pitches == p1.p1_zone_pitches > 0


def test_inputs_switch_zone_with_the_same_names(table):
    savant = input_frame(table)
    fixed = input_frame(table, zone="fixed")
    assert list(savant.columns) == list(fixed.columns)
    row = (table.player_id == HITTER) & (table.season == 2025) & (table.week == 1)
    assert savant.loc[row, "std_zone_swing"].iloc[0] == pytest.approx(1.0)
    assert np.isnan(fixed.loc[row, "std_zone_swing"].iloc[0])  # no pitch in the box
    assert fixed.loc[row, "std_chase_swing"].iloc[0] == pytest.approx(14 / 21)
    other = [c for c in savant.columns if "zone" not in c and "chase" not in c]
    assert savant[other].equals(fixed[other])
    with pytest.raises(ValueError):
        input_frame(table, zone="abs")


def test_zone_setting():
    pytest.importorskip("torch")
    from fantasy_baseball.hitter_ros.net import NetConfig

    assert NetConfig(zone="fixed").zone == "fixed"
    with pytest.raises(ValueError):
        NetConfig(zone="abs")


def _pitches(plate_z, sz_bot, sz_top):
    import pandas as pd

    n = len(plate_z)
    return pd.DataFrame(
        {
            "player_id": [1] * n,
            "game_date": [date(2026, 4, 1)] * n,
            "season": [2026] * n,
            "game_pk": [1] * n,
            "at_bat_number": [1] * n,
            "pitch_number": list(range(1, n + 1)),
            "pitch_type": ["FF"] * n,
            "release_speed": [95.0] * n,
            "pfx_x": [0.5] * n,
            "pfx_z": [1.2] * n,
            "release_spin_rate": [2300.0] * n,
            "plate_x": [0.0] * n,
            "plate_z": plate_z,
            "sz_top": sz_top,
            "sz_bot": sz_bot,
            "balls": [0] * n,
            "strikes": [0] * n,
            "outs_when_up": [0] * n,
            "on_1b": [False] * n,
            "on_2b": [False] * n,
            "on_3b": [False] * n,
            "p_throws": ["R"] * n,
            "stand": ["R"] * n,
            "description": ["ball"] * n,
            "type": ["B"] * n,
            "launch_speed_angle": [None] * n,
            "launch_speed": [None] * n,
            "launch_angle": [None] * n,
        }
    )


def test_pitch_height_in_the_fixed_box():
    from fantasy_baseball.hitter_ros.pitch_tokens import tokens_from_pitches

    # The same 3.4 ft pitch: near the top of a 1.6-3.5 ft Savant zone, above a 2026-style
    # zone topping out at 3.2 ft; in the fixed box, it's the same height either way.
    df = _pitches([3.4, 3.4, None], sz_bot=[1.6, 1.6, 1.6], sz_top=[3.5, 3.2, 3.5])
    savant = tokens_from_pitches(df)["loc_up"]
    fixed = tokens_from_pitches(df, zone="fixed")["loc_up"]
    assert savant[0] < 1 < savant[1]
    lo, hi = FIXED_ZONE[1], FIXED_ZONE[2]
    assert fixed[0] == fixed[1] == pytest.approx((3.4 - lo) / (hi - lo))
    assert savant[2] == fixed[2] == 0.5  # no location: the middle, either way
    with pytest.raises(ValueError):
        tokens_from_pitches(df, zone="abs")


def test_fingerprint_tells_zones_apart():
    pytest.importorskip("torch")
    from fantasy_baseball.hitter_ros.pitch_tokens import tokens_from_pitches
    from fantasy_baseball.hitter_ros.pretrain import same_tokens, tokens_fingerprint

    df = _pitches([2.0, 3.0], sz_bot=[1.6, 1.6], sz_top=[3.2, 3.2])
    savant, fixed = tokens_from_pitches(df), tokens_from_pitches(df, zone="fixed")
    recorded = tokens_fingerprint(savant)
    assert same_tokens(recorded, savant) and not same_tokens(recorded, fixed)
    # A run recorded before the height was fingerprinted still matches on what it has.
    old = {k: v for k, v in recorded.items() if k != "loc_up_mean"}
    assert same_tokens(old, savant) and same_tokens(old, fixed)
