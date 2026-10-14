import numpy as np
import pandas as pd
import pytest
import torch

from fantasy_baseball.hitter_ros.pitch_tokens import OUTCOMES
from fantasy_baseball.pitcher_ros.stuff import (
    BB_TYPES,
    LOCATED_FEATURES,
    STUFF_FEATURES,
    StuffConfig,
    fit_stuff,
    load_stuff,
    predict_stuff,
    stuff_frame,
    type_baseline,
)


def _pitch(**kw):
    row = {
        "pitcher": 1,
        "season": 2024,
        "game_date": "2024-05-01",
        "game_pk": 10,
        "at_bat_number": 1,
        "pitch_number": 1,
        "pitch_type": "FF",
        "p_throws": "R",
        "stand": "R",
        "release_speed": 95.0,
        "pfx_x": -0.8,
        "pfx_z": 1.4,
        "release_spin_rate": 2400.0,
        "spin_axis": 210.0,
        "release_pos_x": -2.0,
        "release_pos_z": 6.0,
        "release_extension": 6.5,
        "arm_angle": 45.0,
        "plate_x": 0.0,
        "plate_z": 2.5,
        "balls": 0,
        "strikes": 0,
        "description": "called_strike",
        "type": "S",
        "launch_speed_angle": np.nan,
        "bb_type": None,
        "xwoba": np.nan,
    }
    row.update(kw)
    return row


def test_a_lefty_and_a_righty_with_mirror_image_pitches_look_the_same():
    righty = _pitch()
    lefty = _pitch(
        pitcher=2,
        game_pk=11,
        p_throws="L",
        stand="L",
        pfx_x=0.8,
        release_pos_x=2.0,
        spin_axis=150.0,
    )
    f = stuff_frame(pd.DataFrame([righty, lefty]))
    for c in ("move_arm", "rel_side", "axis_sin", "axis_cos", "same_side"):
        assert f.loc[0, c] == pytest.approx(f.loc[1, c]), c
    # Arm side is + for both: a righty's fastball runs toward third base (pfx_x < 0).
    assert f.loc[0, "move_arm"] == pytest.approx(0.8) and f.loc[0, "rel_side"] == 2.0


def test_offspeed_is_read_against_his_fastball_that_game():
    rows = [
        _pitch(pitch_number=1, release_speed=96.0, pfx_z=1.5),
        _pitch(pitch_number=2, release_speed=94.0, pfx_z=1.3),  # fastball avg 95, 1.4
        _pitch(pitch_number=3, pitch_type="CH", release_speed=86.0, pfx_z=0.4),
        # Another game with no fastball: no reference.
        _pitch(game_pk=12, pitch_type="SL", release_speed=85.0),
    ]
    f = stuff_frame(pd.DataFrame(rows))
    assert f.loc[2, "has_fb"] == 1
    assert f.loc[2, "fb_speed_gap"] == pytest.approx((86 - 95) / 5)
    assert f.loc[2, "fb_up_gap"] == pytest.approx(0.4 - 1.4)
    assert f.loc[3, "has_fb"] == 0 and f.loc[3, "fb_speed_gap"] == 0


def test_missing_tracking_is_zero_with_a_flag_and_labels_follow_the_pitch():
    rows = [
        _pitch(release_spin_rate=np.nan, spin_axis=np.nan, arm_angle=np.nan, season=2012),
        _pitch(
            pitch_number=2,
            description="hit_into_play",
            type="X",
            launch_speed_angle=6.0,
            bb_type="fly_ball",
            xwoba=1.2,
            plate_x=40.0,  # a tracking glitch
        ),
        _pitch(pitch_number=3, description="swinging_strike"),
    ]
    f = stuff_frame(pd.DataFrame(rows))
    assert (f.loc[0, ["has_spin", "has_axis", "has_arm", "spin", "arm_angle"]] == 0).all()
    assert f.loc[0, "graded_contact"] == 0 and f.loc[1, "graded_contact"] == 1
    assert f.loc[1, "outcome"] == OUTCOMES.index("bip_barrel")
    assert f.loc[1, "bb"] == BB_TYPES.index("fly_ball") and f.loc[1, "xwoba"] == pytest.approx(1.2)
    assert f.loc[1, "loc_in"] == -3  # clipped (plate_x + = outside to a righty batter)
    assert f.loc[2, "outcome"] == OUTCOMES.index("whiff")
    assert f.loc[2, "bb"] == -1 and np.isnan(f.loc[2, "xwoba"])
    assert set(LOCATED_FEATURES) <= set(f.columns)


def _synthetic(n=4000, seed=0):
    """Fast fastballs get whiffed, slow ones get hit: something a model can learn."""
    rng = np.random.default_rng(seed)
    speed = rng.uniform(88, 100, n)
    whiff = rng.random(n) < (speed - 86) / 16
    rows = [
        _pitch(
            game_pk=int(i // 20),
            pitch_number=int(i % 20) + 1,
            release_speed=float(s),
            description="swinging_strike" if w else "hit_into_play",
            type="S" if w else "X",
            launch_speed_angle=np.nan if w else 5.0,
            bb_type=None if w else "line_drive",
            xwoba=np.nan if w else 0.5,
        )
        for i, (s, w) in enumerate(zip(speed, whiff, strict=True))
    ]
    return stuff_frame(pd.DataFrame(rows))


def test_the_model_learns_what_the_pitch_type_alone_cannot(tmp_path):
    frame = _synthetic()
    train, test = frame.iloc[:3000], frame.iloc[3000:]
    config = StuffConfig(hidden=(16,), max_epochs=15, patience=3, batch_size=256, lr=1e-2)
    fit = fit_stuff(train, STUFF_FEATURES, config)
    p = predict_stuff(fit, test)
    fast = test["speed"] > 1.5
    assert p.loc[fast, "p_whiff"].mean() > p.loc[~fast, "p_whiff"].mean() + 0.2
    assert np.allclose(p[[f"p_{o}" for o in OUTCOMES]].sum(axis=1), 1, atol=1e-5)
    torch.save(
        {
            "state_dict": fit.model.state_dict(),
            "features": list(STUFF_FEATURES),
            "config": config.to_dict(),
            "season": 2025,
        },
        tmp_path / "model-2025.pt",
    )
    again = predict_stuff(load_stuff(tmp_path / "model-2025.pt"), test)
    pd.testing.assert_frame_equal(p, again, atol=1e-6)
    # Every pitch is one fastball type, so the type-only guess is just the base rate.
    ce, bb_ce, _ = type_baseline(train, test)
    assert ce > 0 and bb_ce >= 0


def test_predictions_are_read_as_graded_contact_era_pitches():
    frame = _synthetic(800)
    config = StuffConfig(hidden=(8,), max_epochs=2, patience=1, batch_size=256)
    fit = fit_stuff(frame, STUFF_FEATURES, config)
    old = frame.assign(graded_contact=np.float32(0))
    pd.testing.assert_frame_equal(predict_stuff(fit, frame), predict_stuff(fit, old))
