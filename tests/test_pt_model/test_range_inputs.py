import numpy as np
import pandas as pd
import pytest

from fantasy_baseball.pt_model import range_inputs as ri
from fantasy_baseball.pt_model.ranges import STATS


def _leaderboard(rows):
    """A raw MLB Stats API hitting leaderboard: (id, games, PA, position) per row."""
    return pd.DataFrame(
        [
            {
                "player.id": pid,
                "stat.gamesPlayed": g,
                "stat.plateAppearances": pa,
                "stat.atBats": 0.9 * pa,
                "stat.hits": 0.25 * 0.9 * pa,
                "stat.runs": 0.1 * pa,
                "stat.homeRuns": 0.03 * pa,
                "stat.rbi": 0.1 * pa,
                "stat.stolenBases": 0.02 * pa,
                "position.abbreviation": pos,
            }
            for pid, g, pa, pos in rows
        ]
    )


def test_season_frame_reads_the_counts_and_the_seasons_position():
    frame = ri.season_frame(_leaderboard([(1, 81, 324, "DH"), (2, 10, 30, "C")]))
    assert frame.loc[1, ["g", "pa", "rbi"]].tolist() == [81.0, 324.0, pytest.approx(32.4)]
    assert frame.loc[2, "pa"] == 30.0 and frame.loc[2, "pos"] == "C"  # under 50 PA kept
    with pytest.raises(ValueError, match="repeats"):
        ri.season_frame(_leaderboard([(1, 81, 324, "DH"), (1, 10, 30, "C")]))


def test_load_seasons_reads_the_leaderboard_cache(tmp_path):
    _leaderboard([(1, 81, 324, "SS")]).to_csv(tmp_path / "mlb_hitting_2025.csv", index=False)
    loaded = ri.load_seasons(tmp_path, [2025])
    assert loaded[2025].loc[1, "pa"] == 324.0


def test_season_lines_divide_by_the_seasons_schedule():
    seasons = {
        2020: ri.season_frame(_leaderboard([(1, 30, 120, "SS")])),
        2021: ri.season_frame(_leaderboard([(1, 81, 324, "SS")])),
    }
    lines = ri.season_lines(seasons, [2020, 2021])
    assert lines[2020].loc[1].tolist() == [0.5, 2.0]  # 60 scheduled in 2020
    assert lines[2021].loc[1].tolist() == [0.5, 2.0]
    with pytest.raises(ValueError, match="2019"):
        ri.season_lines(seasons, [2019])


def test_position_comes_from_before_the_season_never_after():
    seasons = {
        2023: ri.season_frame(_leaderboard([(1, 100, 400, "C"), (2, 100, 400, "1B")])),
        2024: ri.season_frame(_leaderboard([(1, 100, 400, "C")])),
        2025: ri.season_frame(_leaderboard([(1, 100, 400, "DH")])),  # moved off catcher
    }
    ids = pd.Index([1, 2, 3])
    assert ri.position_before(seasons, 2025, ids).tolist()[:2] == ["C", "1B"]
    assert ri.position_before(seasons, 2026, ids)[1] == "DH"
    assert pd.isna(ri.position_before(seasons, 2025, ids)[3])


def _setup():
    pt = pd.DataFrame(
        {"pa": [650.0, 600.0, 550.0, 500.0], "ab": [585.0, 540.0, 495.0, 450.0]}, index=[1, 2, 3, 4]
    )
    rates = pd.DataFrame(
        {f"p_{c}": [0.1, 0.1, 0.1] for c in STATS},
        index=pd.MultiIndex.from_tuples(
            [(1, 2026), (3, 2026), (4, 2026)], names=["player_id", "season"]
        ),
    )
    seasons = {y: ri.season_frame(_leaderboard([(1, 145, 600, "C")])) for y in (2023, 2024, 2025)}
    # 2026: 1 had a full year, 4 had 30 PA, 3 never batted.
    seasons[2026] = ri.season_frame(_leaderboard([(1, 150, 620, "C"), (4, 12, 30, "SS")]))
    info = ri.player_info(
        pd.DataFrame(
            {
                "id": [1, 2, 3, 4],
                "birthDate": ["1996-06-30", "1990-01-01", "2002-12-31", "1999-05-01"],
                "mlbDebutDate": ["2015-04-01", "2020-04-01", "2026-04-01", "2025-08-01"],
            }
        )
    )
    return pt, rates, seasons, info


def test_build_rows_ranks_before_dropping_and_reads_real_short_seasons():
    pt, rates, seasons, info = _setup()
    rows, dropped = ri.build_rows(2026, pt, rates, seasons, info, with_actuals=True)
    assert dropped == 1 and rows.index.tolist() == [1, 3, 4]
    assert rows["proj_rank"].tolist() == [1, 3, 4]  # 3 keeps the rank he had with 2 in the list
    assert rows.loc[1, "age"] == pytest.approx(30.0, abs=0.01) and rows.loc[1, "catcher"] == 1.0
    assert rows.loc[1, "ab_pa"] == pytest.approx(0.9)
    # A 30-PA season is 30, not 0; missing from the leaderboard is 0.
    assert rows.loc[4, "a_pa"] == 30.0 and rows.loc[3, "a_pa"] == 0.0
    assert np.isnan(rows.loc[3, "g1"])  # debuted 2026: no 2025 history
    assert rows.loc[4, "g1"] == 0.0 and np.isnan(rows.loc[4, "g2"])  # debuted 2025, didn't bat


def test_build_rows_for_an_unplayed_season_has_no_actuals():
    pt, rates, seasons, info = _setup()
    future = {y: seasons[y] for y in (2024, 2025, 2026)}
    rates27 = rates.rename(index={2026: 2027}, level="season")
    rows, _ = ri.build_rows(2027, pt, rates27, future, info, with_actuals=False)
    assert "a_pa" not in rows and rows.loc[1, "pa1"] == pytest.approx(620 / 162)
    with pytest.raises(ValueError, match="no 2027 season loaded"):
        ri.build_rows(2027, pt, rates27, future, info, with_actuals=True)
    with pytest.raises(ValueError, match="no 2027 rows"):
        ri.build_rows(2027, pt, rates, future, info, with_actuals=False)


def test_load_rates_averages_seeds_and_refuses_two_preseason_rows(tmp_path):
    for seed, hr in ((0, 0.04), (1, 0.06)):
        run = tmp_path / f"r-s{seed}"
        run.mkdir()
        pd.DataFrame(
            {
                "player_id": [1, 1],
                "season": [2026, 2026],
                "week": [0, 5],
                **{c: [hr, 0.5] for c in STATS},
            }
        ).to_parquet(run / "predictions.parquet")
    rates = ri.load_rates([tmp_path / "r-s0", tmp_path / "r-s1"])
    assert rates.loc[(1, 2026), "p_hr"] == pytest.approx(0.05)
    bad = tmp_path / "bad"
    bad.mkdir()
    pd.DataFrame(
        {
            "player_id": [1, 1],
            "season": [2026, 2026],
            "week": [0, 0],
            **{c: [0.1, 0.1] for c in STATS},
        }
    ).to_parquet(bad / "predictions.parquet")
    with pytest.raises(ValueError, match="more than one"):
        ri.load_rates([bad])


def test_history_seasons_are_the_three_before_each():
    assert ri.history_seasons([2022, 2023]) == [2019, 2020, 2021, 2022]
