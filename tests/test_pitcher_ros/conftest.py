import pandas as pd

from fantasy_baseball.pitcher_ros.stats import COUNTS


def game(player, day, **counts):
    """One pitcher-game line as ``stats.load_games`` returns it; counts default to 0."""
    when = pd.Timestamp(day)
    row = {"player_id": player, "game_date": when, "season": when.year}
    row.update({c: counts.get(c, 1 if c == "g" else 0) for c in COUNTS})
    return row


def league_games():
    """2022-2024 league: one season line a year, 1000 BF, 700 outs, 250 K, 10 W, 5 SV,
    100 ER, 200 H, 80 BB -- so K/BF .25, W/BF .01, SV/BF .005, ERA 27*100/700, WHIP
    3*280/700 = 1.2."""
    return [
        game(900, f"{y}-06-01", bf=1000, outs=700, k=250, w=10, sv=5, er=100, h=200, bb=80)
        for y in (2022, 2023, 2024)
    ]
