import numpy as np
import pandas as pd
import pytest

from fantasy_baseball.pitcher_ros import stuff_features
from fantasy_baseball.pitcher_ros.stuff import BB_TYPES, GROUPS, OUTCOMES
from fantasy_baseball.pitcher_ros.stuff_features import (
    CONTACT_QUALITY,
    STUFF_INPUTS,
    STUFF_RATE_SETS,
    WINDOWS,
    build_stuff_features,
    season_features,
    stuff_inputs,
)

ACE, OTHER = 1, 2


def _pitches(rows):
    """Pitches (pitcher, season, date, whiff chance); every pitch a four-seamer."""
    out = pd.DataFrame(rows, columns=["pitcher", "season", "game_date", "whiff"])
    out["game_date"] = pd.to_datetime(out["game_date"])
    for g in GROUPS:
        out[f"pt_{g}"] = np.float32(g == "ff")
    return out


@pytest.fixture
def fake_model(monkeypatch):
    """The model's prediction is the pitch's own ``whiff`` column: whiff that often, a
    ball in play the rest (all ground balls, xwOBA 0.3)."""

    def predict(fit, pitches):
        p = pd.DataFrame(0.0, index=pitches.index, columns=[f"p_{o}" for o in OUTCOMES])
        p["p_whiff"] = pitches["whiff"].to_numpy()
        p["p_bip_weak"] = 1 - p["p_whiff"]
        for b in BB_TYPES:
            p[f"p_{b}"] = float(b == "ground_ball")
        p["xwoba_bip"] = 0.3
        return p

    monkeypatch.setattr(stuff_features, "predict_stuff", predict)


def _rows(*keys):
    return pd.DataFrame(keys, columns=["player_id", "season", "week", "as_of"]).assign(
        as_of=lambda d: pd.to_datetime(d["as_of"])
    )


def test_windows_are_read_against_last_seasons_league(fake_model):
    pitches = _pitches(
        [
            (ACE, 2023, "2023-05-01", 0.5),
            (ACE, 2024, "2024-05-01", 0.4),
            (ACE, 2024, "2024-05-02", 0.4),
            (OTHER, 2024, "2024-05-01", 0.2),
            (OTHER, 2024, "2024-05-02", 0.2),
            (ACE, 2025, "2025-04-01", 0.6),
            (ACE, 2025, "2025-04-10", 0.9),  # on the as-of date: not yet known
        ]
    )
    f = season_features(_rows((ACE, 2025, 2, "2025-04-10")), pitches, None, 2025).iloc[0]
    league_2024 = (0.4 * 2 + 0.2 * 2) / 4  # every pitch a swing: whiffs per swing
    assert f["p1_stuff_whiff_swing"] == pytest.approx(0.4 - league_2024)
    assert f["p3_stuff_whiff_swing"] == pytest.approx((0.5 + 0.8) / 3 - league_2024)
    assert f["car_stuff_whiff_swing"] == pytest.approx(f["p3_stuff_whiff_swing"])
    assert f["std_stuff_whiff_swing"] == pytest.approx(0.6 - league_2024)
    assert f["p1_stuff_gb_bip"] == pytest.approx(0)  # every ball in play a grounder
    assert np.isnan(f["p1_stuff_off_whiff_swing"])  # he threw no changeups or splitters


def test_a_later_pitch_never_moves_a_row(fake_model):
    base = [(ACE, 2024, "2024-05-01", 0.3), (ACE, 2025, "2025-04-01", 0.6)]
    rows = _rows((ACE, 2025, 1, "2025-04-05"))
    before = season_features(rows, _pitches(base), None, 2025)
    later = _pitches([*base, (ACE, 2025, "2025-04-05", 0.99), (ACE, 2025, "2025-06-01", 0.0)])
    pd.testing.assert_frame_equal(before, season_features(rows, later, None, 2025))


def test_contact_quality_is_blank_before_graded_contact_and_no_model_is_blank(fake_model):
    pitches = _pitches([(ACE, 2014, "2014-05-01", 0.3), (ACE, 2015, "2015-05-01", 0.3)])
    rows = _rows(
        (ACE, 2015, 0, "2015-03-30"), (ACE, 2016, 0, "2016-03-30"), (ACE, 2008, 0, "2008-03-30")
    )
    f = build_stuff_features(rows, pitches, {2015: None, 2016: None}).set_index("season")
    quality = [f"p1_stuff_{r}" for r in CONTACT_QUALITY]
    assert f.loc[2015, quality].isna().all()
    # He throws only four-seamers, so breaking and offspeed stay blank in any season.
    thrown = [c for c in quality if "brk_" not in c and "off_" not in c]
    assert f.loc[2016, thrown].notna().all()
    assert f.loc[2015, "p1_stuff_whiff_swing"] == pytest.approx(0)
    assert f.loc[2008, STUFF_INPUTS].isna().all()


def test_a_rate_set_keeps_every_window_of_just_its_rates():
    core = stuff_inputs("core")
    assert len(core) == len(WINDOWS) * len(STUFF_RATE_SETS["core"])
    assert "car_stuff_whiff_swing" in core and "car_stuff_brk_whiff_swing" not in core
    overall = stuff_inputs("overall")
    assert not any("_fb_" in c or "_brk_" in c or "_off_" in c for c in overall)
    assert set(core) < set(overall) < set(stuff_inputs("all")) == set(STUFF_INPUTS)
