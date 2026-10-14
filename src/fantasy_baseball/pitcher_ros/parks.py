"""Park inputs for the pitcher table (#465), on the hitter net's park factors (#433).

A park factor is the same number from either side of the ball: both teams' batting in
games at the park over the same teams' road games (``hitter_ros.parks.park_factors``),
so a park that adds hits for batters adds them against pitchers. Inputs per table row,
from the ``PARK_SEASONS`` seasons before the row's season only, exactly as the hitter
net's (``hitter_ros.parks.build_park_features``):

* ``park_home_<stat>``: his team's main park this season (the team of his next game),
  from the published schedule.
* ``park_<w>_<stat>`` for ``w`` in std / p1 / p3: the factor of the parks he pitched in
  this season before the date, last season, and the three seasons before, weighted by
  the batters he faced there (a batter faced is a plate appearance against him, so the
  hitter code's ``pa`` weight is his ``bf``).
"""

from __future__ import annotations

from pathlib import Path

import duckdb
import pandas as pd

from fantasy_baseball.hitter_ros.features import load_feature_file
from fantasy_baseball.hitter_ros.parks import (
    PARK_FEATURES,
    build_options,
    check_games_cover_lineups,
)
from fantasy_baseball.pitcher_ros.stats import GAME_DATE_SQL


def parks_path(root: Path, name: str) -> Path:
    """Where ``scripts/build_pitcher_ros_parks.py --name <name>`` writes the features."""
    return root / f"parks_{name}.parquet"


def load_pitcher_games(conn: duckdb.DuckDBPyConnection) -> pd.DataFrame:
    """One row per played game x pitcher who faced a batter: season, ``player_id``, date,
    ballpark and his batters faced as ``pa``. Dated like ``stats.load_games`` (a
    suspended game on the day it was finished), so a row's history cuts off where its
    other inputs do."""
    check_games_cover_lineups(conn)
    return conn.execute(
        f"""
        SELECT g.season, p.player_id, p.line_date AS game_date, g.venue_id, sum(p.bf) AS pa
        FROM (SELECT game_pk, player_id, bf, {GAME_DATE_SQL} AS line_date FROM pitching) p
        JOIN games g USING (game_pk)
        WHERE g.played
        GROUP BY ALL HAVING sum(p.bf) > 0
        """
    ).df()


def load_park_inputs(
    table: pd.DataFrame, root: Path, name: str
) -> tuple[pd.DataFrame, dict[str, float]]:
    """``PARK_FEATURES`` from ``parks_<name>.parquet`` under ``root``, aligned to
    ``table``'s rows, and the file's build options. ValueError, saying how to fix it,
    when the file is missing, stale, lacks a feature, or was built for another table."""
    return load_feature_file(
        table,
        parks_path(root, name),
        PARK_FEATURES,
        f"run scripts/build_pitcher_ros_parks.py --name {name}",
        expected_options=build_options(),
        expected_label="the current",
    )
