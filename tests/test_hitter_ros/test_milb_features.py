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
    table = table.assign(
        car_pa=[250.0, 250.0, 0.0], std_pa=[0.0, 60.0, 0.0], car_seasons_in_store=10
    )
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


def test_load_milb_inputs_says_how_to_fix_a_missing_or_stale_file(factors, tmp_path):
    import json

    from fantasy_baseball.hitter_ros.milb_features import load_milb_inputs

    table, season_lines, window_lines, mlb = _inputs()
    with pytest.raises(ValueError, match=r"build_hitter_ros_milb.py --name x"):
        load_milb_inputs(table, tmp_path, "x")
    built = build_milb_features(table, season_lines, window_lines, mlb)
    path = milb_features.milb_path(tmp_path, "x")
    built.to_parquet(path, index=False)
    path.with_suffix(".json").write_text(json.dumps({"shrink_pa": 5.0}))
    x, options = load_milb_inputs(table, tmp_path, "x")
    assert list(x.columns) == MILB_FEATURES and x.index.equals(table.index)
    assert options == {"shrink_pa": 5.0}
    built.iloc[:1].to_parquet(path, index=False)
    with pytest.raises(ValueError, match="lacks 2 table rows"):
        load_milb_inputs(table, tmp_path, "x")
    # A file from before a feature was added: a fix-it message, not a KeyError.
    built.drop(columns="milb_p1_avg").to_parquet(path, index=False)
    with pytest.raises(ValueError, match="lacks 1 features"):
        load_milb_inputs(table, tmp_path, "x")


def test_a_preset_file_built_with_other_options_is_refused(factors, tmp_path):
    import json

    from fantasy_baseball.hitter_ros.milb_features import MILB_PRESETS, load_milb_inputs

    table, season_lines, window_lines, mlb = _inputs()
    path = milb_features.milb_path(tmp_path, "rookies-s100")
    build_milb_features(table, season_lines, window_lines, mlb).to_parquet(path, index=False)
    with pytest.raises(ValueError, match="unknown options"):  # no options file at all
        load_milb_inputs(table, tmp_path, "rookies-s100")
    path.with_suffix(".json").write_text(json.dumps({"shrink_pa": 0.0, "vets_blank_from": None}))
    with pytest.raises(ValueError, match="not the rookies-s100 preset"):
        load_milb_inputs(table, tmp_path, "rookies-s100")
    path.with_suffix(".json").write_text(json.dumps(MILB_PRESETS["rookies-s100"]))
    _, options = load_milb_inputs(table, tmp_path, "rookies-s100")
    assert options == MILB_PRESETS["rookies-s100"]


def test_too_little_history_to_tell_a_vet_gets_no_inputs(factors):
    table, season_lines, window_lines, mlb = _inputs()
    # Player 1 reads as a rookie (0 MLB PA), but the store has only 2 seasons before.
    table = table.assign(car_pa=0.0, std_pa=0.0, car_seasons_in_store=[2, 2, 10])
    out = build_milb_features(table, season_lines, window_lines, mlb, vet_min_pa=300)
    row = out[(out["player_id"] == 1) & (out["week"] == 2)].iloc[0]
    assert row["milb_p1_log_pa"] == 0 and np.isnan(row["milb_p1_avg"])


def test_the_season_after_2020_has_an_unknown_last_season(factors):
    table, season_lines, window_lines, mlb = _inputs()
    lines = season_lines.assign(season=season_lines["season"] - 2)  # 2021, 2019, 2016
    rows = table.assign(season=2021, as_of=table["as_of"] - pd.DateOffset(years=3))
    out = build_milb_features(rows, lines, window_lines.iloc[:0], mlb)
    row = out[(out["player_id"] == 1) & (out["week"] == 2)].iloc[0]
    # 2020 had no minor leagues: last season is unknown, not "no PA".
    assert np.isnan(row["milb_p1_log_pa"]) and np.isnan(row["milb_p1_avg"])
    assert row["milb_p3_log_pa"] == pytest.approx(np.log1p(400))  # 2019 still counts


def test_no_window_over_before_any_date_does_not_crash(factors):
    table, season_lines, window_lines, mlb = _inputs()
    opening_day = table[table["week"] == 0]  # every minor-league window ends later
    out = build_milb_features(opening_day, season_lines, window_lines, mlb)
    assert (out["milb_std_log_pa"] == 0).all()
    assert out["milb_std_avg"].dtype == "float64"


def test_the_default_is_rookies_only_and_shrunk():
    from fantasy_baseball.hitter_ros.net import NetConfig

    assert NetConfig().milb == "rookies-s100"


def test_minor_league_steal_rates_are_kept_by_default():
    pytest.importorskip("torch")
    from fantasy_baseball.hitter_ros.net import NetConfig

    # #413: dropping them (--no-milb-sb) made rookie SB ordering much worse.
    assert NetConfig().milb_sb is True
