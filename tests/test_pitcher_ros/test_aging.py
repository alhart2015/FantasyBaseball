import numpy as np
import pandas as pd
import pytest

from fantasy_baseball.pitcher_ros import aging


def _rows(rng, n_per_season=400, seasons=(2022, 2023, 2024, 2025), era_slope=0.15, ip_slope=0.05):
    """Synthetic pitcher-seasons: ERA misses and lost innings grow past age 35 for
    starters; relievers have none. Every pitcher shares a -0.1 ERA level miss."""
    parts = []
    for s in seasons:
        n = n_per_season
        age = rng.uniform(23, 42, n)
        role = np.where(rng.uniform(size=n) < 0.5, "starter", "reliever")
        past = np.maximum(0.0, age - 35)
        starter = role == "starter"
        ip = rng.uniform(60, 190, n)
        bf = 4.2 * ip
        p_era = np.full(n, 3.8)
        era = p_era - 0.1 + np.where(starter, era_slope * past, 0.0) + rng.normal(0, 0.4, n)
        proj_ip = ip / np.clip(
            0.8 - np.where(starter, ip_slope * past, 0.0) + rng.normal(0, 0.1, n), 0.2, None
        )
        parts.append(
            pd.DataFrame(
                {
                    "season": s,
                    "age": age,
                    "role": role,
                    "proj_ip": proj_ip,
                    "a_ip": ip,
                    "a_bf": bf,
                    "a_k": 0.24 * bf,
                    "a_w": 0.05 * bf,
                    "a_er": era * ip / 9,
                    "a_h": 0.95 * ip,
                    "a_bb": 0.3 * ip,
                    "p_k": 0.24,
                    "p_w": 0.05,
                    "p_era": p_era,
                    "p_whip": 1.25,
                }
            )
        )
    return pd.concat(parts, ignore_index=True)


def test_hinge_is_flat_to_the_knot_then_straight_then_flat_past_its_top():
    h = aging.Hinge(knot=36.0, slope=0.1, level=-0.03, top=41.0)
    np.testing.assert_allclose(
        h.of(np.array([30.0, 36.0, 38.0, 41.0, 45.0])), [0, 0, 0.2, 0.5, 0.5]
    )
    np.testing.assert_allclose(aging.Hinge(None, 0.0, 1.0).of(np.array([40.0])), [0.0])


def test_rate_fit_finds_the_old_starters_era_miss_and_leaves_the_shared_level_out():
    fit = aging.fit_rates(_rows(np.random.default_rng(0)))
    h = fit[("starter", "era")]
    assert h.knot in (34.0, 36.0) and 0.05 < h.slope < 0.25 and h.level < 0
    # Only the listed corrections are made.
    assert fit[("reliever", "era")].knot is None and fit[("reliever", "k")].knot is None
    rates = pd.DataFrame(
        {"k": [0.25, 0.25], "w": 0.05, "era": [3.5, 3.5], "whip": 1.2}, index=[1, 2]
    )
    out = aging.adjust_rates(
        rates, pd.Series({1: 40.0, 2: 28.0}), pd.Series({1: "starter", 2: "starter"}), fit
    )
    assert out.loc[1, "era"] > 3.5 and out.loc[2, "era"] == 3.5  # no level shift for the young


def test_no_signal_means_no_correction():
    fit = aging.fit_rates(_rows(np.random.default_rng(1), era_slope=0.0), knots=(None, 34.0, 36.0))
    h = fit[("starter", "era")]
    assert h.knot is None or abs(h.slope) < 0.03


def test_innings_fit_takes_old_starters_innings_down_by_their_share():
    rows = _rows(np.random.default_rng(2))
    fit = aging.fit_innings(rows)
    assert fit["starter"].knot is not None and fit["starter"].slope < 0
    ip = pd.Series({1: 180.0, 2: 180.0, 3: 60.0})
    out = aging.adjust_innings(
        ip,
        pd.Series({1: 40.0, 2: 28.0, 3: 40.0}),
        pd.Series({1: "starter", 2: "starter", 3: "reliever"}),
        fit,
    )
    assert out[1] < 180.0 and out[2] == 180.0 and out[3] == pytest.approx(60.0, rel=0.05)
    # No age or role: as he was.
    same = aging.adjust_innings(ip, pd.Series(dtype=float), pd.Series(dtype=object), fit)
    assert same.tolist() == ip.tolist()


def test_role_comes_from_the_start_share():
    out = aging.role(pd.Series({1: 1.0, 2: 0.2, 3: np.nan}))
    assert out[1] == "starter" and out[2] == "reliever" and pd.isna(out[3])


def test_rates_never_go_below_zero():
    fit = {("starter", "k"): aging.Hinge(30.0, -1.0, 0.0, 45.0)}
    out = aging.adjust_rates(
        pd.DataFrame({"k": [0.2]}, index=[1]), pd.Series({1: 40.0}), pd.Series({1: "starter"}), fit
    )
    assert out.loc[1, "k"] == 0.0


def test_one_season_is_refused_not_read_as_no_correction():
    # With one season every knot's held-out loss is 0, so "none" would win silently.
    with pytest.raises(ValueError, match=r"2\+ seasons"):
        aging.fit_rates(_rows(np.random.default_rng(3), seasons=(2024,)))


def test_a_role_with_nothing_to_fit_on_is_refused():
    rows = _rows(np.random.default_rng(4)).assign(p_era=np.nan)
    with pytest.raises(ValueError, match="no starter rows with our era"):
        aging.fit_rates(rows)
