from datetime import date

import numpy as np
import pandas as pd
import pytest

from fantasy_baseball.hitter_ros import milb_features, milb_grade
from fantasy_baseball.hitter_ros.features import check_aligned
from fantasy_baseball.hitter_ros.milb_features import MILB_FEATURES, build_milb_features

AAA = 11


def _counts(pa, avg):
    ab = round(pa * 0.9)
    return {
        "pa": pa,
        "ab": ab,
        "h": round(ab * avg),
        "hr": round(pa * 0.03),
        "r": round(pa * 0.1),
        "rbi": round(pa * 0.1),
        "sb": round(pa * 0.02),
        "bb": round(pa * 0.08),
        "k": round(pa * 0.2),
        "sf": 0,
    }


def _line(season, player, pa, avg, age=24.0, sport=AAA, league=1, **extra):
    return {
        "season": season,
        "player_id": player,
        "sport_id": sport,
        "league_id": league,
        "age": age,
        **_counts(pa, avg),
        **extra,
    }


@pytest.fixture
def factors(monkeypatch):
    """Every level-season grades at 0.5; records the season windows asked for."""
    asked = []

    def fake(levels, seasons):
        asked.append(seasons)
        return pd.DataFrame({s: [0.5] for s in milb_grade.STATS}, index=pd.Index([AAA]))

    monkeypatch.setattr(milb_grade, "MIN_LEAGUE_PA", 1)
    monkeypatch.setattr(milb_features, "level_factors", fake)
    return asked


def _inputs():
    # Player 1 hits .300 and player 2 .200 in one league, so player 1 is 1.2x it.
    season_lines = pd.DataFrame(
        [
            _line(2023, 1, 400, 0.300),
            _line(2023, 2, 400, 0.200, age=26.0),
            _line(2021, 1, 400, 0.300),
            _line(2021, 2, 400, 0.300),  # 2021: player 1 is 1.0x his league
            _line(2018, 1, 400, 0.300),
            _line(2018, 2, 400, 0.100),  # 2018: 1.5x; only in his career window
        ]
    )
    window_lines = pd.DataFrame(
        [
            _line(2024, p, 50, avg, window_end=end)
            for end in (date(2024, 4, 3), date(2024, 4, 10), date(2024, 4, 17))
            for p, avg in ((1, 0.300), (2, 0.200))
        ]
    )
    mlb = pd.DataFrame([_line(2023, 9, 500, 0.250, sport=milb_grade.MLB, league=0)])
    table = pd.DataFrame(
        {
            "player_id": [1, 1, 3],
            "season": [2024, 2024, 2024],
            "week": [0, 2, 0],
            "as_of": pd.to_datetime(["2024-03-28", "2024-04-11", "2024-03-28"]),
        }
    )
    return table, season_lines, window_lines, mlb


def test_features_for_each_window(factors):
    table, season_lines, window_lines, mlb = _inputs()
    out = build_milb_features(table, season_lines, window_lines, mlb).set_index(
        ["player_id", "week"]
    )
    assert list(out.columns) == ["season", "as_of", *MILB_FEATURES]
    week2 = out.loc[(1, 2)]
    # Last season: 1.2x his league, times the 0.5 factor.
    assert week2["milb_p1_avg"] == pytest.approx(0.6, abs=0.01)
    assert week2["milb_p1_log_pa"] == pytest.approx(np.log1p(400))
    # Last three seasons: 2021 (1.0x) and 2023 (1.2x), equal PA; 2018 is too old.
    assert week2["milb_p3_avg"] == pytest.approx(0.5 * 1.1, abs=0.01)
    assert week2["milb_p3_log_pa"] == pytest.approx(np.log1p(800))
    assert week2["milb_car_log_pa"] == pytest.approx(np.log1p(1200))
    assert week2["milb_p1_level"] == 1 and week2["milb_p1_top_level"] == 1
    # Age vs. level: 24 in a 2023 level averaging 25.
    assert week2["milb_p1_age_vs_level"] == pytest.approx(-1.0)


def test_this_season_only_counts_windows_over_before_the_date(factors):
    table, season_lines, window_lines, mlb = _inputs()
    out = build_milb_features(table, season_lines, window_lines, mlb).set_index(
        ["player_id", "week"]
    )
    # Apr 11: the windows ending Apr 3 and Apr 10, not the one ending Apr 17.
    assert out.loc[(1, 2), "milb_std_log_pa"] == pytest.approx(np.log1p(100))
    assert out.loc[(1, 2), "milb_std_avg"] == pytest.approx(0.6, abs=0.01)
    # Opening day: nothing yet this season.
    assert out.loc[(1, 0), "milb_std_log_pa"] == 0
    assert np.isnan(out.loc[(1, 0), "milb_std_avg"])


def test_no_minor_league_pa_is_zero_volume_and_unknown_rates(factors):
    table, season_lines, window_lines, mlb = _inputs()
    out = build_milb_features(table, season_lines, window_lines, mlb).set_index(
        ["player_id", "week"]
    )
    row = out.loc[(3, 0)]
    assert all(row[f"milb_{w}_log_pa"] == 0 for w in milb_features.MILB_WINDOWS)
    assert row[["milb_p1_avg", "milb_car_level", "milb_std_age_vs_level"]].isna().all()


def test_factors_are_walk_forward_and_rows_align(factors):
    table, season_lines, window_lines, mlb = _inputs()
    out = build_milb_features(table, season_lines, window_lines, mlb)
    assert factors == [(2024 - milb_features.FACTOR_SEASONS, 2023)]
    assert check_aligned(table, out) is None


def test_shrink_pulls_a_small_sample_toward_its_levels_average(factors):
    table, season_lines, window_lines, mlb = _inputs()
    raw = build_milb_features(table, season_lines, window_lines, mlb)
    shrunk = build_milb_features(table, season_lines, window_lines, mlb, shrink_pa=400)
    row = (raw["player_id"] == 1) & (raw["week"] == 2)
    # Last season: 400 PA at 0.6 (1.2x, graded), plus 400 PA of the level's average (0.5).
    assert shrunk.loc[row, "milb_p1_avg"].item() == pytest.approx(0.55, abs=0.01)
    assert raw.loc[row, "milb_p1_avg"].item() == pytest.approx(0.6, abs=0.01)
    # Volume, level and age aren't rates: shrinking leaves them alone.
    for col in ("milb_p1_log_pa", "milb_p1_level", "milb_p1_age_vs_level"):
        assert shrunk.loc[row, col].item() == pytest.approx(raw.loc[row, col].item())


def test_vets_can_get_no_minor_league_inputs(factors):
    table, season_lines, window_lines, mlb = _inputs()
    # Player 1 is a vet by week 2 (250 + 60 MLB PA); player 3 never is.
    table = table.assign(car_pa=[250.0, 250.0, 0.0], std_pa=[0.0, 60.0, 0.0])
    out = build_milb_features(table, season_lines, window_lines, mlb, vet_min_pa=300)
    keyed = out.set_index(["player_id", "week"])
    vet = keyed.loc[(1, 2)]
    assert all(vet[f"milb_{w}_log_pa"] == 0 for w in milb_features.MILB_WINDOWS)
    assert (
        vet.drop([f"milb_{w}_log_pa" for w in milb_features.MILB_WINDOWS])
        .drop(["season", "as_of"])
        .isna()
        .all()
    )
    # Opening day he was still under 300: his inputs are there.
    assert keyed.loc[(1, 0), "milb_p1_avg"] == pytest.approx(0.6, abs=0.01)
