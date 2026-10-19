import numpy as np
import pandas as pd
import pytest

from fantasy_baseball.pt_model import range_inputs as ri
from fantasy_baseball.pt_model.ranges import STATS


def _stats_file(directory, season, rows):
    pd.DataFrame(rows).to_csv(directory / f"hitters-{season}.csv", index=False)


def _line(mlbam, g, pa, name="X"):
    return {
        "Name": name,
        "MLBAMID": mlbam,
        "G": g,
        "PA": pa,
        "AB": 0.9 * pa,
        "H": 0.25 * 0.9 * pa,
        "R": 0.1 * pa,
        "HR": 0.03 * pa,
        "RBI": 0.1 * pa,
        "SB": 0.02 * pa,
    }


def _panel(rows):
    base = {
        "observed": True,
        "is_pitcher": False,
        "scheduled_games": 162,
        "debut_date": "2015-04-01",
        "primary_position": "SS",
    }
    return pd.DataFrame([{**base, **r} for r in rows])


def test_season_lines_prefer_the_stats_file_and_scale_the_panel_by_its_schedule(tmp_path):
    _stats_file(tmp_path, 2026, [_line(1, 81, 324)])
    panel = _panel(
        [
            {"mlbam_id": 1, "season": 2026, "games": 40, "pa": 160},  # partial: the file wins
            {"mlbam_id": 1, "season": 2020, "games": 30, "pa": 120, "scheduled_games": 60},
            {
                "mlbam_id": 2,
                "season": 2020,
                "games": 60,
                "pa": 240,
                "scheduled_games": 60,
                "is_pitcher": True,
            },
        ]
    )
    lines = ri.season_lines(panel, tmp_path, [2020, 2026])
    assert lines[2026].loc[1].tolist() == [0.5, 2.0]
    assert lines[2020].loc[1].tolist() == [0.5, 2.0] and 2 not in lines[2020].index
    with pytest.raises(ValueError, match="2019"):
        ri.season_lines(panel, tmp_path, [2019])


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


def test_build_rows_ranks_before_dropping_and_reads_a_missing_hitter_as_zero(tmp_path):
    pt = pd.DataFrame({"pa": [650.0, 600.0, 550.0], "ab": [585.0, 540.0, 495.0]}, index=[1, 2, 3])
    rates = pd.DataFrame(
        {f"p_{c}": [0.1, 0.1] for c in STATS},
        index=pd.MultiIndex.from_tuples([(1, 2026), (3, 2026)], names=["player_id", "season"]),
    )
    lines = {y: pd.DataFrame({"g": [0.9], "pa": [4.0]}, index=[1]) for y in (2023, 2024, 2025)}
    info = pd.DataFrame(
        {
            "debut_year": [2015, 2020, 2026],
            "pos": ["C", "SS", "DH"],
            "birth": pd.to_datetime(["1996-06-30", "1990-01-01", "2002-12-31"]),
        },
        index=[1, 2, 3],
    )
    actual = pd.DataFrame(
        {c: [600.0] for c in ("g", "pa", "ab", "h", "r", "hr", "rbi", "sb")}, index=[1]
    )
    rows, dropped = ri.build_rows(2026, pt, rates, lines, info, actual)
    assert dropped == 1 and rows.index.tolist() == [1, 3]
    assert rows["proj_rank"].tolist() == [1, 3]  # 3 keeps the rank he had with 2 in the list
    assert rows.loc[1, "age"] == pytest.approx(30.0, abs=0.01) and rows.loc[1, "catcher"] == 1.0
    # 3 isn't in the season's stats: he batted 0 times.
    assert rows.loc[3, "dh"] == 1.0 and rows.loc[3, "a_pa"] == 0.0
    assert np.isnan(rows.loc[3, "g1"])  # debuted 2026: no 2025 history
    assert rows.loc[1, "ab_pa"] == pytest.approx(0.9)
    with pytest.raises(ValueError, match="no 2027 rows"):
        ri.build_rows(2027, pt, rates, {**lines, 2026: lines[2025]}, info)


def test_history_seasons_are_the_three_before_each():
    assert ri.history_seasons([2022, 2023]) == [2019, 2020, 2021, 2022]
