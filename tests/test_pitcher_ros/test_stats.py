import pandas as pd

from fantasy_baseball.pitch_data.store import PITCHING_STAT_COLUMNS
from fantasy_baseball.pitcher_ros.stats import load_games


def test_load_games_dates_a_suspended_game_to_the_day_it_finished(tmp_path):
    base = {c: 0 for c in PITCHING_STAT_COLUMNS}
    rows = [
        {**base, "player_id": 1, "team_id": 7, "game_pk": 10, "game_date": "2014-05-22",
         "resumed_date": "2014-09-01", "bf": 4, "outs": 3},
        {**base, "player_id": 2, "team_id": 7, "game_pk": 11, "game_date": "2014-05-23",
         "resumed_date": None, "bf": 5, "outs": 3},
    ]  # fmt: skip
    (tmp_path / "pitching").mkdir()
    pd.DataFrame(rows).to_parquet(tmp_path / "pitching" / "2014.parquet")
    games = load_games(tmp_path).set_index("player_id")
    assert games.loc[1, "game_date"] == pd.Timestamp("2014-09-01")
    assert games.loc[2, "game_date"] == pd.Timestamp("2014-05-23")
    assert games.loc[1, "season"] == 2014 and games.loc[1, "g"] == 1
