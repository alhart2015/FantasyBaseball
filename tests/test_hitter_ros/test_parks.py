from datetime import date

import numpy as np
import pandas as pd
import pytest

from fantasy_baseball.hitter_ros import parks
from fantasy_baseball.hitter_ros.milb_grade import COUNTS
from fantasy_baseball.hitter_ros.parks import (
    PARK_FEATURES,
    build_park_features,
    load_player_games,
    load_team_games,
    main_venues,
    park_factors,
)
from fantasy_baseball.pitch_data.store import connect

COORS, PETCO, TOKYO = 19, 2680, 2397
COL, SD = 115, 135


def _team_line(season, pk, venue, home, team, h, pa=40, day=1):
    """One team's batting in one game: ``pa`` PA, 36 AB, ``h`` hits, nothing else."""
    counts = dict.fromkeys(COUNTS, 0)
    counts.update(pa=pa, ab=36, h=h)
    return {
        "season": season,
        "game_pk": pk,
        "game_date": date(season, 5, day),
        "venue_id": venue,
        "home_team_id": home,
        "team_id": team,
        "is_home": team == home,
        **counts,
    }


def _game(season, pk, venue, home, away, h_home, h_away, day=1):
    return [
        _team_line(season, pk, venue, home, home, h_home, day=day),
        _team_line(season, pk, venue, home, away, h_away, day=day),
    ]


def _league(season=2024):
    """Colorado and San Diego play 4 games each way; both teams hit 12 at Coors and 6
    at Petco. Colorado also 'hosts' one game in Tokyo where everyone hits 30."""
    rows = []
    for i in range(4):
        rows += _game(season, season * 1000 + 100 + i, COORS, COL, SD, 12, 12, day=i + 1)
        rows += _game(season, season * 1000 + 200 + i, PETCO, SD, COL, 6, 6, day=i + 10)
    rows += _game(season, season * 1000 + 300, TOKYO, COL, SD, 30, 30, day=20)
    return pd.DataFrame(rows)


def test_main_venue_ignores_a_neutral_site_series():
    main = main_venues(_league())
    assert main[(2024, COL)] == COORS and main[(2024, SD)] == PETCO


def test_park_factor_is_home_over_road_shrunk_toward_one(monkeypatch):
    monkeypatch.setattr(parks, "SHRINK_PA", 320.0)  # = the home PA, so the factor halves
    f = park_factors(_league(), (2024, 2024))
    # Coors: 12/36 at home vs. 6/36 in Colorado's road games -> 2.0 raw. The Tokyo game
    # is neither a Coors game nor a San Diego road game.
    assert f.loc[COORS, "pa"] == 320
    assert f.loc[COORS, "avg"] == pytest.approx(1 + (2.0 - 1) * 0.5)
    assert f.loc[PETCO, "avg"] == pytest.approx(1 + (0.5 - 1) * 0.5)
    assert TOKYO not in f.index


def test_factors_only_use_the_seasons_asked_for():
    games = pd.concat([_league(2023), _league(2024).assign(h=36)])  # 2024: everyone 1.000
    assert park_factors(games, (2023, 2023)).loc[COORS, "avg"] > 1
    assert COORS not in park_factors(games, (2021, 2022)).index


def _players():
    """Player 7 bats 4 PA in every game above for Colorado (2023 and 2024)."""
    rows = []
    for season in (2023, 2024):
        for _, g in _league(season).query("team_id == @COL").iterrows():
            rows.append(
                {
                    "season": season,
                    "player_id": 7,
                    "game_date": g["game_date"],
                    "venue_id": g["venue_id"],
                    "pa": 4,
                }
            )
    return pd.DataFrame(rows)


def _schedule(team_games):
    """The schedule behind ``team_games``: one row per game."""
    return team_games.drop_duplicates(["season", "game_pk"])[
        ["season", "game_pk", "venue_id", "home_team_id"]
    ]


def test_park_inputs_for_a_row(monkeypatch):
    monkeypatch.setattr(parks, "SHRINK_PA", 320.0)
    games = pd.concat([_league(2023), _league(2024)])
    table = pd.DataFrame(
        {
            "player_id": [7, 7],
            "season": [2024, 2024],
            "week": [0, 2],
            "as_of": pd.to_datetime(["2024-05-01", "2024-05-11"]),
            "team_id": [COL, COL],
        }
    )
    out = build_park_features(table, games, _players(), _schedule(games)).set_index("week")
    assert list(out.columns) == ["player_id", "season", "as_of", *PARK_FEATURES]
    # Home park: Coors, measured on 2023 only (the seasons before 2024 in the data).
    assert out.loc[0, "park_home_avg"] == pytest.approx(1.5)
    # Last season: 4 games at Coors (1.5), 4 at Petco (0.75), 1 in Tokyo (no factor: 1).
    assert out.loc[0, "park_p1_avg"] == pytest.approx((4 * 1.5 + 4 * 0.75 + 1) / 9)
    # This season before May 11: the 4 Coors games (May 1-4) and Petco on May 10.
    assert np.isnan(out.loc[0, "park_std_avg"])  # nothing before May 1
    assert out.loc[2, "park_std_avg"] == pytest.approx((4 * 1.5 + 0.75) / 5)


def _write(path, frame):
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(path, index=False)


def test_loaders_join_box_scores_to_ballparks(tmp_path):
    games = pd.DataFrame(
        [
            {
                "game_pk": 1,
                "season": 2024,
                "game_date": date(2024, 5, 1),
                "venue_id": COORS,
                "home_team_id": COL,
                "away_team_id": SD,
                "played": True,
            },
            {
                "game_pk": 2,
                "season": 2024,
                "game_date": date(2024, 5, 2),
                "venue_id": COORS,
                "home_team_id": COL,
                "away_team_id": SD,
                "played": False,
            },
        ]
    )
    _write(tmp_path / "games" / "2024.parquet", games)
    box = dict.fromkeys(COUNTS, 1)
    lineups = pd.DataFrame(
        [
            {"game_pk": 1, "game_date": "2024-05-01", "team_id": COL, "player_id": 7, **box},
            {"game_pk": 1, "game_date": "2024-05-01", "team_id": SD, "player_id": 8, **box},
        ]
    )
    _write(tmp_path / "lineups" / "2024.parquet", lineups)
    conn = connect(tmp_path)
    team = load_team_games(conn).sort_values("team_id")
    assert team[["team_id", "is_home", "venue_id"]].values.tolist() == [
        [COL, True, COORS],
        [SD, False, COORS],
    ]
    assert load_player_games(conn)["player_id"].tolist() == [7, 8] or set(
        load_player_games(conn)["player_id"]
    ) == {7, 8}


def test_load_park_inputs_refuses_a_stale_or_unknown_file(tmp_path):
    import json

    from fantasy_baseball.hitter_ros.parks import build_options, load_park_inputs, parks_path

    table = pd.DataFrame(
        {"player_id": [7], "season": [2024], "week": [0], "as_of": pd.to_datetime(["2024-05-01"])}
    )
    with pytest.raises(ValueError, match="is missing"):
        load_park_inputs(table, tmp_path, "p3")
    built = table.assign(**dict.fromkeys(PARK_FEATURES, 1.0))
    path = parks_path(tmp_path, "p3")
    built.to_parquet(path, index=False)
    with pytest.raises(ValueError, match="unknown options"):
        load_park_inputs(table, tmp_path, "p3")
    path.with_suffix(".json").write_text(json.dumps({**build_options(), "shrink_pa": 1.0}))
    with pytest.raises(ValueError, match="not the current"):
        load_park_inputs(table, tmp_path, "p3")
    path.with_suffix(".json").write_text(json.dumps(build_options()))
    x, options = load_park_inputs(table, tmp_path, "p3")
    assert list(x.columns) == PARK_FEATURES and options == build_options()


def test_park_inputs_are_on_by_default():
    from fantasy_baseball.hitter_ros.net import NetConfig

    assert NetConfig().parks == "p3"


def test_home_park_comes_from_the_schedule_before_any_home_game(monkeypatch):
    """Opening day of a season Colorado opens on the road: no home game played yet, but
    the schedule already says Coors."""
    monkeypatch.setattr(parks, "SHRINK_PA", 320.0)
    games = _league(2023)
    upcoming = pd.DataFrame(
        {
            "season": [2024] * 2,
            "game_pk": [1, 2],
            "venue_id": [COORS, PETCO],
            "home_team_id": [COL, SD],
        }
    )
    table = pd.DataFrame(
        {
            "player_id": [7],
            "season": [2024],
            "week": [0],
            "as_of": pd.to_datetime(["2024-04-01"]),
            "team_id": [COL],
        }
    )
    out = build_park_features(table, games, _players(), pd.concat([_schedule(games), upcoming]))
    assert out.loc[0, "park_home_avg"] == pytest.approx(1.5)


def test_a_doubleheader_at_two_parks_counts_both_games(monkeypatch):
    monkeypatch.setattr(parks, "SHRINK_PA", 320.0)
    games = pd.concat([_league(2023), _league(2024)])
    player = pd.DataFrame(
        {
            "season": 2024,
            "player_id": 7,
            "game_date": [date(2024, 6, 1)] * 2,
            "venue_id": [COORS, PETCO],
            "pa": [4, 4],
        }
    )
    table = pd.DataFrame(
        {
            "player_id": [7],
            "season": [2024],
            "week": [5],
            "as_of": pd.to_datetime(["2024-06-02"]),
            "team_id": [COL],
        }
    )
    out = build_park_features(table, games, player, _schedule(games))
    assert out.loc[0, "park_std_avg"] == pytest.approx((1.5 + 0.75) / 2)


def test_a_season_without_a_games_file_is_an_error(tmp_path):
    box = dict.fromkeys(COUNTS, 1)
    for season in (2023, 2024):
        lineups = pd.DataFrame(
            [
                {
                    "game_pk": season,
                    "game_date": f"{season}-05-01",
                    "team_id": COL,
                    "player_id": 7,
                    **box,
                }
            ]
        )
        _write(tmp_path / "lineups" / f"{season}.parquet", lineups)
    with pytest.raises(ValueError, match="no games files"):
        load_team_games(connect(tmp_path))
    games = pd.DataFrame(
        [
            {
                "game_pk": 2024,
                "season": 2024,
                "game_date": date(2024, 5, 1),
                "venue_id": COORS,
                "home_team_id": COL,
                "away_team_id": SD,
                "played": True,
            }
        ]
    )
    _write(tmp_path / "games" / "2024.parquet", games)
    with pytest.raises(ValueError, match=r"no games file for seasons \[2023\]"):
        load_player_games(connect(tmp_path))
