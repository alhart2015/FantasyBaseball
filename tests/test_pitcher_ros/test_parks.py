from datetime import date

import pandas as pd
import pytest

from fantasy_baseball.hitter_ros.parks import build_park_features
from fantasy_baseball.pitch_data.store import connect
from fantasy_baseball.pitcher_ros.parks import load_park_inputs, load_pitcher_games

COORS, PETCO = 19, 2680
COL, SD = 115, 135
PITCHER = 7


def _write(path, frame):
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(path, index=False)


def _game(pk, day, venue=COORS, played=True):
    return {
        "game_pk": pk,
        "season": 2024,
        "game_date": date(2024, 5, day),
        "venue_id": venue,
        "home_team_id": COL,
        "away_team_id": SD,
        "played": played,
    }


def _line(pk, day, bf, resumed=None, player=PITCHER):
    return {
        "game_pk": pk,
        "game_date": f"2024-05-{day:02d}",
        "resumed_date": resumed,
        "team_id": COL,
        "player_id": player,
        "bf": bf,
    }


def _store(tmp_path, games, lines):
    _write(tmp_path / "games" / "2024.parquet", pd.DataFrame(games))
    _write(tmp_path / "pitching" / "2024.parquet", pd.DataFrame(lines))
    # The ballpark loaders check every box-score season has its games file.
    lineups = pd.DataFrame([{"game_pk": 1, "game_date": "2024-05-01", "team_id": COL}])
    _write(tmp_path / "lineups" / "2024.parquet", lineups)
    return connect(tmp_path)


def test_pitcher_games_are_his_batters_faced_at_each_ballpark(tmp_path):
    conn = _store(
        tmp_path,
        [_game(1, 1), _game(2, 2, venue=PETCO), _game(3, 3, played=False), _game(4, 4)],
        [
            _line(1, 1, bf=24),
            _line(2, 2, bf=5),
            _line(3, 3, bf=9),  # a game that was never played
            _line(4, 4, bf=0, player=8),  # warmed up, faced nobody
        ],
    )
    games = load_pitcher_games(conn).sort_values("game_date")
    assert games[["player_id", "venue_id", "pa"]].values.tolist() == [
        [PITCHER, COORS, 24],
        [PITCHER, PETCO, 5],
    ]
    assert (games["season"] == 2024).all()


def test_a_suspended_game_counts_on_the_day_it_was_finished(tmp_path):
    """As stats.load_games dates it: only from then is his whole line known."""
    conn = _store(tmp_path, [_game(1, 1)], [_line(1, 1, bf=12, resumed="2024-05-20")])
    assert pd.to_datetime(load_pitcher_games(conn)["game_date"]).tolist() == [
        pd.Timestamp("2024-05-20")
    ]


def test_parks_he_pitched_in_are_weighted_by_batters_faced(monkeypatch):
    """The hitter build reads ``pa`` as the weight: for a pitcher, his batters faced."""
    factors = pd.DataFrame(
        {s: [1.3, 0.9] for s in ("avg", "hr", "r", "babip", "k", "bb")} | {"pa": [6000.0] * 2},
        index=pd.Index([COORS, PETCO], dtype="int64", name="venue_id"),
    )
    monkeypatch.setattr(
        "fantasy_baseball.hitter_ros.parks.park_factors", lambda *a, **k: factors.copy()
    )
    pitcher_games = pd.DataFrame(
        {
            "season": [2024, 2024, 2025],
            "player_id": [PITCHER] * 3,
            "game_date": pd.to_datetime(["2024-05-01", "2024-05-02", "2025-04-02"]),
            "venue_id": [COORS, PETCO, PETCO],
            "pa": [30, 10, 20],
        }
    )
    table = pd.DataFrame(
        {
            "player_id": [PITCHER],
            "season": [2025],
            "week": [1],
            "as_of": pd.to_datetime(["2025-04-10"]),
            "team_id": pd.array([SD], dtype="Int64"),
        }
    )
    schedule = pd.DataFrame(
        {"season": [2025], "game_pk": [9], "venue_id": [PETCO], "home_team_id": [SD]}
    )
    team_games = schedule.assign(game_date=date(2025, 4, 1), team_id=SD, is_home=True)
    out = build_park_features(table, team_games, pitcher_games, schedule).iloc[0]
    assert out["park_home_r"] == pytest.approx(0.9)
    assert out["park_p1_r"] == pytest.approx((30 * 1.3 + 10 * 0.9) / 40)
    assert out["park_std_r"] == pytest.approx(0.9)


def test_a_missing_park_file_says_how_to_build_the_pitcher_one(tmp_path):
    table = pd.DataFrame(
        {"player_id": [7], "season": [2024], "week": [0], "as_of": pd.to_datetime(["2024-05-01"])}
    )
    with pytest.raises(ValueError, match=r"build_pitcher_ros_parks.py --name p3"):
        load_park_inputs(table, tmp_path, "p3")
