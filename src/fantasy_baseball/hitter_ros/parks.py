"""Park factors and park inputs for the hitter table (#433).

The model had no park input, only the team's run rate, and #433 found it projects
Coors Field hitters about 9 points of AVG too low. Parks are keyed by ballpark (the
schedule's venue), not team: teams move (the A's to Sacramento in 2025), play a season
elsewhere (the 2025 Rays at Steinbrenner Field) and play neutral-site series (Tokyo,
London, Seoul).

**Park factor** (``park_factors``) of a ballpark, per stat in ``PARK_STATS``: both teams'
rate in games there, over the same teams' rate in the home team's road games, pooled
over a range of seasons. Only home games at the home team's main park that season count
(``main_venues``), so a Tokyo series doesn't count as a Wrigley game -- nor as anyone's
road game. The factor is shrunk toward 1 by ``SHRINK_PA`` plate appearances of neutral
park: one season's home PA is about 6,000, so one season of data counts half, three
seasons about three quarters. A park with no games in the range has no factor.

**Inputs** per table row in season S (``build_park_features``), every factor from the
``PARK_SEASONS`` seasons before S only:

* ``park_home_<stat>``: his team's main park this season (the team of his next game),
  from the season's schedule, which is published in advance -- so a team that opens on
  the road still has one. NaN for a new park.
* ``park_<w>_<stat>`` for ``w`` in std / p1 / p3: the PA-weighted factor of the parks he
  batted in this season before the date, last season, and the three seasons before --
  so the model can take a park out of his past stats. A park without a factor counts
  as neutral (1); NaN with no PA in the window.
"""

from __future__ import annotations

from pathlib import Path

import duckdb
import pandas as pd

from fantasy_baseball.hitter_ros.features import ROW_KEYS, load_feature_file
from fantasy_baseball.hitter_ros.milb_grade import COUNTS, rates

PARK_STATS = ("avg", "hr", "r", "babip", "k", "bb")
PARK_SEASONS = 3
SHRINK_PA = 6000.0
PARK_WINDOWS = ("home", "std", "p1", "p3")
PARK_FEATURES = [f"park_{w}_{s}" for w in PARK_WINDOWS for s in PARK_STATS]


def parks_path(root: Path, name: str) -> Path:
    """Where ``scripts/build_hitter_ros_parks.py --name <name>`` writes the features."""
    return root / f"parks_{name}.parquet"


def build_options() -> dict[str, float]:
    """The settings a park file is built with; saved next to it as ``parks_<name>.json``."""
    return {"park_seasons": float(PARK_SEASONS), "shrink_pa": SHRINK_PA}


def load_park_inputs(
    table: pd.DataFrame, root: Path, name: str
) -> tuple[pd.DataFrame, dict[str, float]]:
    """``PARK_FEATURES`` from ``parks_<name>.parquet`` under ``root``, aligned to
    ``table``'s rows, and the file's build options. ValueError, saying how to fix it,
    when the file is missing, stale (built with other ``build_options``, e.g. before a
    constant changed), lacks a feature, or was built for another table."""
    return load_feature_file(
        table,
        parks_path(root, name),
        PARK_FEATURES,
        f"run scripts/build_hitter_ros_parks.py --name {name}",
        expected_options=build_options(),
        expected_label="the current",
    )


def check_games_cover_lineups(conn: duckdb.DuckDBPyConnection) -> None:
    """ValueError unless every season with box scores has its ``games`` file: the park
    loaders join the two, so a missing season would silently drop out."""
    rerun = "run scripts/fetch_pitch_data.py --start <first> --end <last> --only lineups"
    views = {r[0] for r in conn.execute("SELECT view_name FROM duckdb_views()").fetchall()}
    if "games" not in views:
        raise ValueError(f"the store has no games files (each game's ballpark); {rerun}")
    missing = [
        r[0]
        for r in conn.execute(
            """
            SELECT DISTINCT year(CAST(game_date AS DATE)) AS season FROM lineups
            EXCEPT SELECT DISTINCT season FROM games ORDER BY 1
            """
        ).fetchall()
    ]
    if missing:
        raise ValueError(f"no games file for seasons {missing}; {rerun}")


def load_schedule(conn: duckdb.DuckDBPyConnection) -> pd.DataFrame:
    """Every regular-season game on the schedule, played or not: season, game, ballpark
    and home team."""
    check_games_cover_lineups(conn)
    return conn.execute("SELECT season, game_pk, venue_id, home_team_id FROM games").df()


def load_team_games(conn: duckdb.DuckDBPyConnection) -> pd.DataFrame:
    """One row per played game x team: that team's batting counts (``milb_grade.COUNTS``,
    every batter), with the game's season, date, ballpark, home team and whether this
    team was at home."""
    check_games_cover_lineups(conn)
    sums = ", ".join(f"sum(l.{c}) AS {c}" for c in COUNTS)
    return conn.execute(
        f"""
        SELECT g.season, g.game_pk, g.game_date, g.venue_id, g.home_team_id,
               l.team_id, l.team_id = g.home_team_id AS is_home, {sums}
        FROM lineups l JOIN games g USING (game_pk)
        WHERE g.played
        GROUP BY ALL
        """
    ).df()


def load_player_games(conn: duckdb.DuckDBPyConnection) -> pd.DataFrame:
    """One row per played game x batter with a PA: season, date, ballpark and his PA."""
    check_games_cover_lineups(conn)
    return conn.execute(
        """
        SELECT g.season, l.player_id, g.game_date, g.venue_id, sum(l.pa) AS pa
        FROM lineups l JOIN games g USING (game_pk)
        WHERE g.played
        GROUP BY ALL HAVING sum(l.pa) > 0
        """
    ).df()


def main_venues(games: pd.DataFrame) -> pd.Series:
    """Each team-season's main ballpark: where it has the most home games in ``games``
    (rows with ``season``, ``game_pk``, ``home_team_id`` and ``venue_id``; several rows
    per game are fine). Index (season, team_id)."""
    one_per_game = games.drop_duplicates(["season", "game_pk"])
    counts = one_per_game.groupby(["season", "home_team_id", "venue_id"]).size()
    top = (
        counts.reset_index(name="n")
        .sort_values(["n", "venue_id"], kind="stable")
        .drop_duplicates(["season", "home_team_id"], keep="last")
    )
    return top.set_index(["season", "home_team_id"])["venue_id"].rename_axis(["season", "team_id"])


def park_factors(
    team_games: pd.DataFrame, seasons: tuple[int, int], main: pd.Series | None = None
) -> pd.DataFrame:
    """Shrunk factor per ballpark (index ``venue_id``) and ``PARK_STATS`` stat, from
    the seasons in ``seasons`` (inclusive), plus ``pa`` (home PA behind it). ``main``:
    ``main_venues`` of the played games, if already computed."""
    first, last = seasons
    games = team_games[team_games["season"].between(first, last)]
    main = main_venues(games) if main is None else main
    # Each game's home team's main park that season; a home game there counts.
    home_main = main.reindex(pd.MultiIndex.from_frame(games[["season", "home_team_id"]])).to_numpy()
    at_main = games[games["venue_id"].to_numpy() == home_main]
    counts = list(COUNTS)
    home = at_main.groupby("venue_id")[counts].sum()
    # The same home teams' road games (both teams' batting), credited to their park. Only
    # games at the home team's main park: a neutral site is nobody's road environment.
    road_games = at_main.loc[~at_main["is_home"], ["season", "game_pk", "team_id"]].rename(
        columns={"team_id": "visitor"}
    )
    road_games = road_games.assign(
        venue_id=main.reindex(
            pd.MultiIndex.from_frame(road_games[["season", "visitor"]])
        ).to_numpy()
    )
    road = at_main.merge(
        road_games[["game_pk", "venue_id"]].dropna(), on="game_pk", suffixes=("_game", "")
    )
    road = road.groupby("venue_id")[counts].sum()
    both = home.index.intersection(road.index)
    raw = rates(home.loc[both]) / rates(road.loc[both])
    weight = home.loc[both, "pa"] / (home.loc[both, "pa"] + SHRINK_PA)
    shrunk = 1.0 + (raw[list(PARK_STATS)] - 1.0).mul(weight, axis=0)
    out = shrunk.assign(pa=home.loc[both, "pa"].astype(float))
    out.index = out.index.astype("int64")
    return out


def _weighted_factors(games: pd.DataFrame, factors: pd.DataFrame) -> pd.DataFrame:
    """Per game row: ``pa`` and PA x the ballpark's factor for each ``PARK_STATS`` stat;
    a park without a factor counts as 1 (neutral)."""
    f = factors.reindex(games["venue_id"])[list(PARK_STATS)].fillna(1.0).set_axis(games.index)
    pa = games["pa"].astype(float)
    return f.mul(pa, axis=0).assign(pa=pa)


def _environment(
    games: pd.DataFrame, factors: pd.DataFrame, keys: list[str], window: str
) -> pd.DataFrame:
    """PA-weighted park factor of ``games`` per ``keys`` group."""
    sums = _weighted_factors(games, factors).groupby([games[k] for k in keys]).sum()
    out = sums[list(PARK_STATS)].div(sums["pa"], axis=0)
    return out.add_prefix(f"park_{window}_").reset_index()


def build_park_features(
    table: pd.DataFrame,
    team_games: pd.DataFrame,
    player_games: pd.DataFrame,
    schedule: pd.DataFrame,
) -> pd.DataFrame:
    """``PARK_FEATURES`` for every ``table`` row (which needs ``team_id``), keyed by
    ``ROW_KEYS`` plus ``as_of``. ``schedule``: ``load_schedule`` (the home park of the
    row's season comes from it, played or not)."""
    rows = table[[*ROW_KEYS, "as_of", "team_id"]].copy()
    played_main = main_venues(team_games)
    scheduled_main = main_venues(schedule)
    parts = []
    for season in sorted(int(s) for s in rows["season"].unique()):
        f = park_factors(team_games, (season - PARK_SEASONS, season - 1), played_main)
        here = rows[rows["season"] == season]
        venue = scheduled_main.reindex(
            pd.MultiIndex.from_arrays([here["season"], here["team_id"].astype("Int64")])
        ).to_numpy()
        home = f.reindex(pd.Series(venue, dtype="Int64"))[list(PARK_STATS)]
        out = here[[*ROW_KEYS, "as_of"]].reset_index(drop=True)
        out[[f"park_home_{s}" for s in PARK_STATS]] = home.to_numpy()
        players = here["player_id"].unique()
        past = player_games[
            player_games["player_id"].isin(players)
            & player_games["season"].between(season - PARK_SEASONS, season - 1)
        ]
        p1 = _environment(past[past["season"] == season - 1], f, ["player_id"], "p1")
        p3 = _environment(past, f, ["player_id"], "p3")
        out = out.merge(p1, on="player_id", how="left").merge(p3, on="player_id", how="left")
        out = out.merge(_std(here, player_games, f), on=ROW_KEYS, how="left")
        parts.append(out)
    return pd.concat(parts, ignore_index=True)[[*ROW_KEYS, "as_of", *PARK_FEATURES]]


def _std(rows: pd.DataFrame, player_games: pd.DataFrame, factors: pd.DataFrame) -> pd.DataFrame:
    """park_std_* per row: the parks of his games this season before the as-of date."""
    season = int(rows["season"].iloc[0])
    games = player_games[
        (player_games["season"] == season) & player_games["player_id"].isin(rows["player_id"])
    ]
    weighted = _weighted_factors(games, factors)
    # One row per player-date (a doubleheader can be at two parks), then running totals.
    daily = (
        weighted.groupby([games["player_id"].astype("int64"), pd.to_datetime(games["game_date"])])
        .sum()
        .rename_axis(["player_id", "date"])
        .reset_index()
        .sort_values(["player_id", "date"])
    )
    cols = ["pa", *PARK_STATS]
    daily[cols] = daily.groupby("player_id")[cols].cumsum()
    daily["date"] = daily["date"].astype("datetime64[ns]")
    left = rows[[*ROW_KEYS, "as_of"]].assign(
        as_of=pd.to_datetime(rows["as_of"]).astype("datetime64[ns]"),
        player_id=rows["player_id"].astype("int64"),
    )
    at = pd.merge_asof(
        left.sort_values("as_of"),
        daily.sort_values("date"),
        left_on="as_of",
        right_on="date",
        by="player_id",
        allow_exact_matches=False,  # a game on the as-of date hasn't happened yet
    )
    out = at[ROW_KEYS].copy()
    for s in PARK_STATS:
        out[f"park_std_{s}"] = at[s] / at["pa"].where(at["pa"] > 0)
    return out
