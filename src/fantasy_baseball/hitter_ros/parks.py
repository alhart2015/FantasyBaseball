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
road game. The factor is
shrunk toward 1 by ``SHRINK_PA`` plate appearances of neutral park: one season's home
PA is about 6,000, so one season of data counts half, three seasons about three
quarters. A park with no games in the range has no factor.

**Inputs** per table row in season S (``build_park_features``), every factor from the
``PARK_SEASONS`` seasons before S only:

* ``park_home_<stat>``: his team's main park this season (the team of his next game;
  the season's schedule is known in advance). NaN for a new park.
* ``park_<w>_<stat>`` for ``w`` in std / p1 / p3: the PA-weighted factor of the parks he
  batted in this season before the date, last season, and the three seasons before --
  so the model can take a park out of his past stats. A park without a factor counts
  as neutral (1); NaN with no PA in the window.
"""

from __future__ import annotations

import json
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


def load_park_inputs(table: pd.DataFrame, root: Path, name: str) -> pd.DataFrame:
    """``PARK_FEATURES`` from ``parks_<name>.parquet`` under ``root``, aligned to
    ``table``'s rows. ValueError, saying how to fix it, when the file is missing, stale
    (built with other ``build_options``, e.g. before a constant changed), lacks a
    feature, or was built for another table."""
    path = parks_path(root, name)
    rebuild = f"run scripts/build_hitter_ros_parks.py --name {name}"
    options_path = path.with_suffix(".json")
    if path.exists():
        built = json.loads(options_path.read_text()) if options_path.exists() else None
        if built != build_options():
            raise ValueError(
                f"{path} was built with {built or 'unknown options'}, not the current "
                f"{build_options()}; {rebuild}"
            )
    return load_feature_file(table, path, PARK_FEATURES, rebuild)


def load_team_games(conn: duckdb.DuckDBPyConnection) -> pd.DataFrame:
    """One row per played game x team: that team's batting counts (``milb_grade.COUNTS``,
    every batter), with the game's season, date, ballpark, home team and whether this
    team was at home."""
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
    return conn.execute(
        """
        SELECT g.season, l.player_id, g.game_date, g.venue_id, sum(l.pa) AS pa
        FROM lineups l JOIN games g USING (game_pk)
        WHERE g.played
        GROUP BY ALL HAVING sum(l.pa) > 0
        """
    ).df()


def main_venues(team_games: pd.DataFrame) -> pd.Series:
    """Each team-season's main ballpark: where it played the most home games. Index
    (season, team_id)."""
    home = team_games[team_games["is_home"]].drop_duplicates(["season", "game_pk"])
    counts = home.groupby(["season", "home_team_id", "venue_id"]).size()
    top = (
        counts.reset_index(name="n")
        .sort_values("n")
        .drop_duplicates(["season", "home_team_id"], keep="last")
    )
    return top.set_index(["season", "home_team_id"])["venue_id"].rename_axis(["season", "team_id"])


def park_factors(team_games: pd.DataFrame, seasons: tuple[int, int]) -> pd.DataFrame:
    """Shrunk factor per ballpark (index ``venue_id``) and ``PARK_STATS`` stat, from
    the seasons in ``seasons`` (inclusive), plus ``pa`` (home PA behind it)."""
    first, last = seasons
    games = team_games[team_games["season"].between(first, last)]
    main = main_venues(games)
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


def _environment(
    games: pd.DataFrame, factors: pd.DataFrame, keys: list[str], window: str
) -> pd.DataFrame:
    """PA-weighted park factor of ``games`` per ``keys`` group; a park without a factor
    counts as 1."""
    f = factors.reindex(games["venue_id"])[list(PARK_STATS)].fillna(1.0).set_axis(games.index)
    pa = games["pa"].astype(float)
    group = [games[k] for k in keys]
    total = pa.groupby(group).sum()
    out = pd.DataFrame({s: (f[s] * pa).groupby(group).sum() / total for s in PARK_STATS})
    return out.add_prefix(f"park_{window}_").reset_index()


def build_park_features(
    table: pd.DataFrame, team_games: pd.DataFrame, player_games: pd.DataFrame
) -> pd.DataFrame:
    """``PARK_FEATURES`` for every ``table`` row (which needs ``team_id``), keyed by
    ``ROW_KEYS`` plus ``as_of``."""
    rows = table[[*ROW_KEYS, "as_of", "team_id"]].copy()
    main = main_venues(team_games)
    parts = []
    for season in sorted(int(s) for s in rows["season"].unique()):
        f = park_factors(team_games, (season - PARK_SEASONS, season - 1))
        here = rows[rows["season"] == season]
        venue = main.reindex(
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
    ].sort_values("game_date")
    f = factors.reindex(games["venue_id"])[list(PARK_STATS)].fillna(1.0).set_axis(games.index)
    pa = games["pa"].astype(float)
    by_player = games["player_id"]
    cum = pd.DataFrame(
        {
            "player_id": by_player.astype("int64"),
            "date": pd.to_datetime(games["game_date"]).astype("datetime64[ns]"),
            "pa": pa.groupby(by_player).cumsum(),
            **{s: (f[s] * pa).groupby(by_player).cumsum() for s in PARK_STATS},
        }
    )
    left = rows[[*ROW_KEYS, "as_of"]].assign(
        as_of=pd.to_datetime(rows["as_of"]).astype("datetime64[ns]"),
        player_id=rows["player_id"].astype("int64"),
    )
    at = pd.merge_asof(
        left.sort_values("as_of"),
        cum.sort_values("date"),
        left_on="as_of",
        right_on="date",
        by="player_id",
        allow_exact_matches=False,  # a game on the as-of date hasn't happened yet
    )
    out = at[ROW_KEYS].copy()
    for s in PARK_STATS:
        out[f"park_std_{s}"] = at[s] / at["pa"].where(at["pa"] > 0)
    return out
