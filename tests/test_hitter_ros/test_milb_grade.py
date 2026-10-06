import numpy as np
import pandas as pd
import pytest

from fantasy_baseball.hitter_ros import milb_grade
from fantasy_baseball.hitter_ros.milb_grade import (
    MLB,
    league_rates,
    level_factors,
    load_milb_lines,
    load_mlb_lines,
    player_levels,
    rates,
    relative_lines,
    translate,
)
from fantasy_baseball.pitch_data.store import connect

AAA, AA, HIGH_A = 11, 12, 13


def _line(season, player, sport, league, pa, avg=0.250, hr=0.03, age=24.0):
    """A line with ``pa`` PA, 90% of them at-bats, hitting ``avg`` with ``hr`` HR/PA."""
    ab = round(pa * 0.9)
    return {
        "season": season,
        "player_id": player,
        "sport_id": sport,
        "league_id": league,
        "pa": pa,
        "ab": ab,
        "h": round(ab * avg),
        "hr": round(pa * hr),
        "r": round(pa * 0.12),
        "rbi": round(pa * 0.11),
        "sb": round(pa * 0.02),
        "bb": round(pa * 0.08),
        "k": round(pa * 0.22),
        "sf": round(pa * 0.01),
        "age": age,
    }


def test_rates_and_empty_denominators():
    r = rates(pd.DataFrame([_line(2024, 1, AAA, 1, 1000, avg=0.300), _line(2024, 2, AAA, 1, 0)]))
    assert r.loc[0, "avg"] == pytest.approx(0.300, abs=1e-3)
    assert r.loc[0, "hr"] == pytest.approx(0.03)
    assert r.loc[1].isna().all()
    assert list(r.columns) == list(milb_grade.STATS)


def test_rates_ignore_non_count_columns():
    frame = pd.DataFrame([{**_line(2024, 1, AAA, 1, 1000, avg=0.300), "league": "PCL"}])
    assert rates(frame).loc[0, "avg"] == pytest.approx(0.300, abs=1e-3)


def test_a_small_league_uses_its_levels_pooled_rates(monkeypatch):
    monkeypatch.setattr(milb_grade, "MIN_LEAGUE_PA", 500)
    lines = pd.DataFrame(
        [
            _line(2024, 1, AAA, 1, 1000, avg=0.300),  # hitter-friendly league
            _line(2024, 2, AAA, 2, 1000, avg=0.200),  # pitcher-friendly league
            _line(2024, 3, AAA, 3, 100, avg=0.400),  # stray PA under a third league
        ]
    )
    lg = league_rates(lines)
    assert lg.loc[(2024, AAA, 1), "avg"] == pytest.approx(0.300, abs=1e-3)
    pooled = (270 + 180 + 36) / (900 + 900 + 90)
    assert lg.loc[(2024, AAA, 3), "avg"] == pytest.approx(pooled)


def test_relative_lines_compare_a_hitter_to_his_own_league(monkeypatch):
    monkeypatch.setattr(milb_grade, "MIN_LEAGUE_PA", 500)
    lines = pd.DataFrame(
        [
            _line(2024, 1, AAA, 1, 1000, avg=0.300),
            _line(2024, 2, AAA, 1, 1000, avg=0.330),
            _line(2024, 3, AAA, 2, 1000, avg=0.220),
            _line(2024, 4, AAA, 2, 1000, avg=0.242),
        ]
    )
    rel = relative_lines(lines).set_index("player_id")["rel_avg"]
    # .330 in the .315 league and .242 in the .231 league are the same: 10% above it.
    assert rel[2] == pytest.approx(rel[4], abs=2e-3)
    assert rel[2] > 1 > rel[1]


def test_player_levels_weight_two_leagues_by_pa_and_measure_age_vs_level():
    rel = pd.DataFrame(
        [
            {**_line(2024, 1, AA, 1, 300, age=21.0), "rel_avg": 1.2},
            {**_line(2024, 1, AA, 2, 100, age=21.0), "rel_avg": 0.8},
            {**_line(2024, 2, AA, 1, 400, age=25.0), "rel_avg": 1.0},
        ]
    )
    for s in milb_grade.STATS:
        if s != "avg":
            rel[f"rel_{s}"] = 1.0
    levels = player_levels(rel).set_index("player_id")
    assert levels.loc[1, "pa"] == 400
    assert levels.loc[1, "rel_avg"] == pytest.approx((1.2 * 300 + 0.8 * 100) / 400)
    # Level mean age, PA-weighted: (21 * 400 + 25 * 400) / 800 = 23.
    assert levels.loc[1, "age_vs_level"] == pytest.approx(-2.0)
    assert levels.loc[2, "age_vs_level"] == pytest.approx(2.0)


def _levels(rows):
    """player_levels-shaped rows: (season, player, sport, pa, rel_avg); other stats 1."""
    frame = pd.DataFrame(rows, columns=["season", "player_id", "sport_id", "pa", "rel_avg"])
    for s in milb_grade.STATS:
        if s != "avg":
            frame[f"rel_{s}"] = 1.0
    return frame


def test_level_factor_is_the_weighted_ratio_over_promoted_players(monkeypatch):
    monkeypatch.setattr(milb_grade, "MIN_PAIRS", 2)
    levels = _levels(
        [
            (2020, 1, AAA, 400, 1.20),
            (2020, 1, MLB, 400, 1.00),  # same season
            (2020, 2, AAA, 400, 1.10),
            (2021, 2, MLB, 200, 0.90),  # the season after
            (2020, 3, AAA, 50, 2.00),  # too few PA to pair
            (2021, 3, MLB, 500, 0.50),
        ]
    )
    f = level_factors(levels, (2020, 2021))
    w1, w2 = 400.0, 2 / (1 / 400 + 1 / 200)
    assert f.loc[AAA, "pairs"] == 2
    assert f.loc[AAA, "avg"] == pytest.approx((w1 * 1.0 + w2 * 0.9) / (w1 * 1.2 + w2 * 1.1))
    assert f.loc[AAA, "hr"] == pytest.approx(1.0)


def test_only_first_promotions_count_each_lower_season_once(monkeypatch):
    monkeypatch.setattr(milb_grade, "MIN_PAIRS", 1)
    levels = _levels(
        [
            # Player 1: AAA then MLB the same season and the next. One pair, same season.
            (2020, 1, AAA, 400, 1.20),
            (2020, 1, MLB, 400, 1.00),
            (2021, 1, MLB, 400, 0.10),
            # Player 2: a veteran with MLB PA (even a few) before his AAA season.
            (2019, 2, MLB, 20, 1.00),
            (2020, 2, AAA, 400, 1.00),
            (2020, 2, MLB, 400, 3.00),
        ]
    )
    f = level_factors(levels, (2019, 2021))
    assert f.loc[AAA, "pairs"] == 1
    assert f.loc[AAA, "avg"] == pytest.approx(1.0 / 1.2)


def test_a_backtest_never_grades_with_seasons_after_its_window(monkeypatch):
    monkeypatch.setattr(milb_grade, "MIN_PAIRS", 1)
    levels = _levels(
        [
            (2020, 1, AAA, 400, 1.20),
            (2020, 1, MLB, 400, 1.00),
            (2021, 2, AAA, 400, 1.00),
            (2022, 2, MLB, 400, 0.10),  # the next season is outside the window
        ]
    )
    f = level_factors(levels, (2020, 2021))
    assert f.loc[AAA, "pairs"] == 1
    assert f.loc[AAA, "avg"] == pytest.approx(1.0 / 1.2)


def test_lower_levels_chain_up_and_thin_levels_are_left_out(monkeypatch):
    monkeypatch.setattr(milb_grade, "MIN_PAIRS", 1)
    levels = _levels(
        [
            (2020, 1, AA, 400, 1.25),
            (2020, 1, MLB, 400, 1.00),  # AA -> MLB: 0.8
            (2020, 2, HIGH_A, 400, 1.00),
            (2020, 2, AA, 400, 0.90),  # A+ -> AA: 0.9
        ]
    )
    f = level_factors(levels, (2020, 2020))
    assert f.loc[AA, "avg"] == pytest.approx(0.8)
    assert f.loc[HIGH_A, "avg"] == pytest.approx(0.9 * 0.8)
    assert f.loc[HIGH_A, "pairs"] == 1
    # No AAA pairs, and nothing below A+ to chain: those levels have no factor.
    assert AAA not in f.index and 14 not in f.index


def test_a_window_without_pairs_gives_no_factors_but_keeps_the_columns():
    levels = _levels([(2020, 1, AAA, 400, 1.2), (2020, 1, MLB, 400, 1.0)])
    f = level_factors(levels, (2001, 2007))
    assert f.empty and list(f.columns) == ["pairs", *milb_grade.STATS]
    assert milb_grade.factor_table(f).empty
    assert translate(levels, f)["mlb_rel_avg"].isna().all()


def test_translate_applies_each_levels_factor():
    levels = _levels([(2020, 1, AAA, 400, 1.2), (2020, 2, 14, 400, 1.2)])
    factors = pd.DataFrame({s: [0.9] for s in milb_grade.STATS}, index=pd.Index([AAA]))
    out = translate(levels, factors).set_index("player_id")["mlb_rel_avg"]
    assert out[1] == pytest.approx(1.08)
    assert np.isnan(out[2])  # no factor for that level


def _write(path, frame):
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(path, index=False)


def test_loaders_drop_pitchers_and_the_mexican_league(tmp_path):
    def api_row(player, league, position_type, pa):
        return {
            "season": 2024,
            "player.id": player,
            "sport_id": AAA,
            "league.id": league,
            "position.type": position_type,
            **{f"stat.{v}": pa for v in milb_grade.COUNTS.values()},
        }

    weekly = pd.DataFrame(
        [
            api_row(1, 117, "Infielder", 10),
            api_row(1, 112, "Infielder", 5),  # traded: a second league
            api_row(2, 117, "Pitcher", 3),
            api_row(3, 125, "Outfielder", 20),  # Mexican League
        ]
    )
    _write(tmp_path / "milb_weekly" / "2024" / "a.parquet", weekly)
    season = pd.DataFrame(
        [{"season": 2024, "player.id": 1, "sport_id": AAA, "stat.age": 23, "stat.hits": 1}]
    )
    _write(tmp_path / "milb_season" / "2024.parquet", season)
    box = {c: 4 for c in milb_grade.COUNTS}
    lineups = pd.DataFrame(
        [
            {"game_date": "2024-05-01", "player_id": 1, "position": "SS", **box},
            {"game_date": "2024-05-01", "player_id": 9, "position": "P", **box},
        ]
    )
    _write(tmp_path / "lineups" / "2024.parquet", lineups)

    conn = connect(tmp_path)
    milb = load_milb_lines(conn).sort_values("league_id")
    assert milb[["player_id", "league_id", "pa", "age"]].values.tolist() == [
        [1, 112, 5, 23],
        [1, 117, 10, 23],
    ]
    mlb = load_mlb_lines(conn)
    assert mlb["player_id"].tolist() == [1] and set(mlb["sport_id"]) == {MLB}


def test_season_totals_match_the_season_loader(tmp_path):
    from fantasy_baseball.hitter_ros.milb_grade import season_totals

    def api_row(window_end, pa):
        return {
            "season": 2024,
            "player.id": 1,
            "sport_id": AAA,
            "league.id": 117,
            "position.type": "Infielder",
            "window_end": window_end,
            **{f"stat.{v}": pa for v in milb_grade.COUNTS.values()},
        }

    weekly = pd.DataFrame(
        [api_row(pd.Timestamp("2024-04-07"), 3), api_row(pd.Timestamp("2024-04-14"), 4)]
    )
    _write(tmp_path / "milb_weekly" / "2024" / "a.parquet", weekly)
    season = pd.DataFrame([{"season": 2024, "player.id": 1, "sport_id": AAA, "stat.age": 23}])
    _write(tmp_path / "milb_season" / "2024.parquet", season)
    conn = connect(tmp_path)
    by_season = load_milb_lines(conn)
    rebuilt = season_totals(load_milb_lines(conn, by_window=True))
    cols = ["season", "player_id", "sport_id", "league_id", *milb_grade.COUNTS, "age"]
    pd.testing.assert_frame_equal(
        rebuilt[cols].astype(float), by_season[cols].astype(float), check_dtype=False
    )
