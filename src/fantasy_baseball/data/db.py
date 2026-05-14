"""SQLite database for fantasy baseball data."""

import json
import re
import sqlite3
from pathlib import Path

import pandas as pd

from fantasy_baseball.data.projections import blend_projections
from fantasy_baseball.models.player import PlayerType
from fantasy_baseball.utils.name_utils import normalize_name

DB_PATH = Path(__file__).resolve().parents[3] / "data" / "fantasy.db"

SCHEMA = """
CREATE TABLE IF NOT EXISTS raw_projections (
    year        INTEGER NOT NULL,
    system      TEXT NOT NULL,
    player_type TEXT NOT NULL,
    name        TEXT NOT NULL,
    team        TEXT,
    fg_id       TEXT,
    mlbam_id    INTEGER,
    pa REAL, ab REAL, h REAL, r REAL, hr REAL,
    rbi REAL, sb REAL, cs REAL, bb REAL, so REAL,
    avg REAL, obp REAL, slg REAL, ops REAL, iso REAL,
    babip REAL, woba REAL, wrc_plus REAL, war REAL,
    w REAL, l REAL, sv REAL, ip REAL, er REAL,
    k REAL, bb_p REAL, h_allowed REAL,
    era REAL, whip REAL, fip REAL, k9 REAL, bb9 REAL,
    hr_p REAL, war_p REAL,
    adp REAL, g REAL,
    UNIQUE (year, system, player_type, fg_id)
);

CREATE INDEX IF NOT EXISTS idx_raw_name ON raw_projections(year, name);

CREATE TABLE IF NOT EXISTS blended_projections (
    year        INTEGER NOT NULL,
    fg_id       TEXT NOT NULL,
    name        TEXT NOT NULL,
    team        TEXT,
    player_type TEXT NOT NULL,
    pa REAL, ab REAL, h REAL,
    r REAL, hr REAL, rbi REAL, sb REAL,
    avg REAL,
    w REAL, k REAL, sv REAL,
    ip REAL, er REAL, bb REAL, h_allowed REAL,
    era REAL, whip REAL,
    adp REAL,
    PRIMARY KEY (year, fg_id)
);

CREATE TABLE IF NOT EXISTS draft_results (
    year    INTEGER NOT NULL,
    pick    INTEGER NOT NULL,
    round   INTEGER NOT NULL,
    team    TEXT NOT NULL,
    player  TEXT NOT NULL,
    fg_id   TEXT,
    PRIMARY KEY (year, pick)
);

CREATE TABLE IF NOT EXISTS weekly_rosters (
    snapshot_date TEXT NOT NULL,
    week_num     INTEGER,
    team         TEXT NOT NULL,
    slot         TEXT NOT NULL,
    player_name  TEXT NOT NULL,
    positions    TEXT,
    status       TEXT,
    yahoo_id     TEXT,
    PRIMARY KEY (snapshot_date, team, slot, player_name)
);

CREATE TABLE IF NOT EXISTS standings (
    year          INTEGER NOT NULL,
    snapshot_date TEXT NOT NULL,
    team          TEXT NOT NULL,
    team_key      TEXT,
    rank          INTEGER,
    r REAL, hr REAL, rbi REAL, sb REAL, avg REAL,
    w REAL, k REAL, sv REAL, era REAL, whip REAL,
    PRIMARY KEY (year, snapshot_date, team)
);

CREATE TABLE IF NOT EXISTS positions (
    name       TEXT NOT NULL PRIMARY KEY,
    positions  TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS ros_blended_projections (
    year          INTEGER NOT NULL,
    snapshot_date TEXT NOT NULL,
    fg_id         TEXT NOT NULL,
    name          TEXT NOT NULL,
    team          TEXT,
    player_type   TEXT NOT NULL,
    pa REAL, ab REAL, h REAL,
    r REAL, hr REAL, rbi REAL, sb REAL,
    avg REAL,
    w REAL, k REAL, sv REAL,
    ip REAL, er REAL, bb REAL, h_allowed REAL,
    era REAL, whip REAL,
    adp REAL,
    PRIMARY KEY (year, snapshot_date, fg_id)
);
"""


def get_connection(db_path=None):
    """Return a sqlite3 connection. Defaults to DB_PATH."""
    path = db_path or DB_PATH
    conn = sqlite3.connect(str(path))
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA foreign_keys=ON")
    conn.row_factory = sqlite3.Row
    return conn


def _add_column_if_missing(conn, table: str, column: str, col_type: str) -> None:
    """Add a column to an existing table if it's not already there.

    Idempotent — safe to call on every connection.
    """
    try:
        info = conn.execute(f"PRAGMA table_info({table})").fetchall()
    except sqlite3.OperationalError:
        return  # Table doesn't exist yet; CREATE will handle it
    if not info:
        return
    existing = {r["name"] for r in info}
    if column not in existing:
        conn.execute(f"ALTER TABLE {table} ADD COLUMN {column} {col_type}")
        conn.commit()


def create_tables(conn):
    """Create all tables (idempotent via IF NOT EXISTS)."""
    # Ensure row_factory is set for dict-like access below.
    conn.row_factory = sqlite3.Row

    # Migrate weekly_rosters if PK is missing player_name (old schema
    # used (snapshot_date, team, slot) which silently dropped duplicate
    # slots like multiple OFs or Ps).
    try:
        info = conn.execute("PRAGMA table_info(weekly_rosters)").fetchall()
        if info:
            pk_cols = {r["name"] for r in info if r["pk"] > 0}
            if "player_name" not in pk_cols:
                conn.execute("DROP TABLE weekly_rosters")
                conn.commit()
    except sqlite3.OperationalError:
        pass  # Table doesn't exist yet; CREATE below will handle it
    conn.executescript(SCHEMA)
    conn.commit()

    # Idempotent column additions for tables that predate later schema bumps.
    _add_column_if_missing(conn, "weekly_rosters", "status", "TEXT")
    _add_column_if_missing(conn, "weekly_rosters", "yahoo_id", "TEXT")
    _add_column_if_missing(conn, "standings", "team_key", "TEXT")


# FanGraphs CSV column → DB column, for hitters
_HITTER_COLS = {
    "Name": "name",
    "Team": "team",
    "PlayerId": "fg_id",
    "MLBAMID": "mlbam_id",
    "PA": "pa",
    "AB": "ab",
    "H": "h",
    "R": "r",
    "HR": "hr",
    "RBI": "rbi",
    "SB": "sb",
    "CS": "cs",
    "BB": "bb",
    "SO": "so",
    "AVG": "avg",
    "OBP": "obp",
    "SLG": "slg",
    "OPS": "ops",
    "ISO": "iso",
    "BABIP": "babip",
    "wOBA": "woba",
    "wRC+": "wrc_plus",
    "WAR": "war",
    "ADP": "adp",
    "G": "g",
}

# FanGraphs CSV column → DB column, for pitchers.
# H, HR, BB, SO map to different DB columns than for hitters.
_PITCHER_COLS = {
    "Name": "name",
    "Team": "team",
    "PlayerId": "fg_id",
    "MLBAMID": "mlbam_id",
    "W": "w",
    "L": "l",
    "SV": "sv",
    "IP": "ip",
    "ER": "er",
    "SO": "k",
    "BB": "bb_p",
    "H": "h_allowed",
    "HR": "hr_p",
    "ERA": "era",
    "WHIP": "whip",
    "FIP": "fip",
    "K/9": "k9",
    "BB/9": "bb9",
    "WAR": "war_p",
    "ADP": "adp",
    "G": "g",
}

# All DB columns in raw_projections (used to filter to only known columns)
_DB_COLUMNS = {
    "year",
    "system",
    "player_type",
    "name",
    "team",
    "fg_id",
    "mlbam_id",
    "pa",
    "ab",
    "h",
    "r",
    "hr",
    "rbi",
    "sb",
    "cs",
    "bb",
    "so",
    "avg",
    "obp",
    "slg",
    "ops",
    "iso",
    "babip",
    "woba",
    "wrc_plus",
    "war",
    "w",
    "l",
    "sv",
    "ip",
    "er",
    "k",
    "bb_p",
    "h_allowed",
    "era",
    "whip",
    "fip",
    "k9",
    "bb9",
    "hr_p",
    "war_p",
    "adp",
    "g",
}

# Pattern: system name is everything before -hitters or -pitchers
_FILENAME_RE = re.compile(r"^(?P<system>.+?)-(?P<ptype>hitters|pitchers)")


def _parse_csv_filename(stem: str):
    """Return (system, player_type) from a CSV filename stem, or (None, None)."""
    m = _FILENAME_RE.match(stem)
    if not m:
        return None, None
    system = m.group("system")
    player_type = PlayerType.HITTER if m.group("ptype") == "hitters" else PlayerType.PITCHER
    return system, player_type


def load_raw_projections(conn, projections_dir):
    """Scan projections_dir for year subdirectories, read FanGraphs CSVs,
    and insert rows into raw_projections (INSERT OR IGNORE on duplicates).

    projections_dir should be a Path-like pointing to the parent of the year
    folders (e.g. ``data/projections/``).
    """
    projections_dir = Path(projections_dir)

    for year_dir in sorted(projections_dir.iterdir()):
        if not year_dir.is_dir() or not year_dir.name.isdigit():
            continue
        year = int(year_dir.name)

        for csv_path in sorted(year_dir.glob("*.csv")):
            system, player_type = _parse_csv_filename(csv_path.stem)
            if system is None:
                continue  # unrecognised filename, skip silently

            col_map = _HITTER_COLS if player_type == PlayerType.HITTER else _PITCHER_COLS

            try:
                df = pd.read_csv(csv_path, dtype={"PlayerId": str, "MLBAMID": str})
            except Exception:
                continue  # skip malformed files

            # Strip BOM from column names (FanGraphs sometimes exports UTF-8-BOM)
            df.columns = [c.lstrip("\ufeff") for c in df.columns]

            # Rename only the columns that are present in both the CSV and the mapping
            rename = {fg: db for fg, db in col_map.items() if fg in df.columns}
            df = df.rename(columns=rename)

            # Attach metadata
            df["year"] = year
            df["system"] = system
            df["player_type"] = player_type

            # Keep only columns that exist in the DB schema
            keep = [c for c in df.columns if c in _DB_COLUMNS]
            df = df[keep]

            # Convert mlbam_id to integer where possible, coercing errors to NaN
            if "mlbam_id" in df.columns:
                df["mlbam_id"] = pd.to_numeric(df["mlbam_id"], errors="coerce")

            # Insert rows; INSERT OR IGNORE handles the UNIQUE constraint
            placeholders = ", ".join("?" * len(keep))
            col_names = ", ".join(keep)
            insert_sql = (
                f"INSERT OR IGNORE INTO raw_projections ({col_names}) VALUES ({placeholders})"
            )
            rows = [
                tuple(None if pd.isna(v) else v for v in row)
                for row in df.itertuples(index=False, name=None)
            ]
            conn.executemany(insert_sql, rows)

    conn.commit()


# ---------------------------------------------------------------------------
# Historical data loaders
# ---------------------------------------------------------------------------

# Pattern to strip FanGraphs-style "(Batter)" / "(Pitcher)" suffixes from names
_PLAYER_SUFFIX_RE = re.compile(r"\s*\((?:Batter|Pitcher)\)\s*$", re.IGNORECASE)


def load_draft_results(conn, drafts_path) -> None:
    """Load draft picks from ``drafts_path`` (JSON) into ``draft_results``.

    The JSON must be structured as::

        {"2023": [{"pick": 1, "round": 1, "team": "...", "player": "..."}, ...], ...}

    "(Batter)" and "(Pitcher)" suffixes are stripped from player names before
    insertion.  For each pick an ``fg_id`` lookup is attempted against
    ``raw_projections`` using ``normalize_name``.  If multiple rows match the
    highest-ADP (lowest numeric value) row wins; if no match exists the field
    is left NULL.

    Uses INSERT OR IGNORE so repeated calls are idempotent.
    """
    drafts_path = Path(drafts_path)
    data = json.loads(drafts_path.read_text(encoding="utf-8"))

    rows = []
    for year_str, picks in data.items():
        year = int(year_str)
        for pick in picks:
            player_raw = pick["player"]
            player_name = _PLAYER_SUFFIX_RE.sub("", player_raw)
            norm = normalize_name(player_name)

            # Attempt fg_id resolution from raw_projections for that year.
            # Fetch all rows for the year, then filter client-side by normalized name
            # so that normalize_name is applied consistently in Python.
            candidates = conn.execute(
                "SELECT name, fg_id, adp FROM raw_projections WHERE year = ?",
                (year,),
            ).fetchall()
            matched = [r for r in candidates if normalize_name(r["name"]) == norm and r["fg_id"]]

            fg_id = None
            if matched:
                # Pick the row with the lowest ADP (best-ranked); nulls sort last
                def _adp_key(r):
                    a = r["adp"]
                    return (a is None, a if a is not None else 0)

                best = min(matched, key=_adp_key)
                fg_id = best["fg_id"]

            rows.append((year, pick["pick"], pick["round"], pick["team"], player_name, fg_id))

    conn.executemany(
        "INSERT OR IGNORE INTO draft_results (year, pick, round, team, player, fg_id) "
        "VALUES (?, ?, ?, ?, ?, ?)",
        rows,
    )
    conn.commit()


def load_standings(conn, standings_path) -> None:
    """Load historical standings from ``standings_path`` (JSON) into ``standings``.

    Expected structure::

        {
          "2023": {
            "standings": [
              {"name": "...", "team_key": "...", "rank": 1,
               "stats": {"R": 900, "HR": 250, ...}}
            ]
          }
        }

    Each team is inserted with ``snapshot_date = 'final'``.  Stat keys in the
    JSON are case-insensitive and mapped to their lowercase DB column names.
    Uses INSERT OR IGNORE so repeated calls are idempotent.
    """
    standings_path = Path(standings_path)
    data = json.loads(standings_path.read_text(encoding="utf-8"))

    rows = []
    for year_str, year_data in data.items():
        year = int(year_str)
        for entry in year_data.get("standings", []):
            stats = {k.lower(): v for k, v in entry.get("stats", {}).items()}
            rows.append(
                (
                    year,
                    "final",
                    entry["name"],
                    entry.get("rank"),
                    stats.get("r"),
                    stats.get("hr"),
                    stats.get("rbi"),
                    stats.get("sb"),
                    stats.get("avg"),
                    stats.get("w"),
                    stats.get("k"),
                    stats.get("sv"),
                    stats.get("era"),
                    stats.get("whip"),
                )
            )

    conn.executemany(
        "INSERT OR IGNORE INTO standings "
        "(year, snapshot_date, team, rank, r, hr, rbi, sb, avg, w, k, sv, era, whip) "
        "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
        rows,
    )
    conn.commit()


def load_weekly_rosters(conn, rosters_dir) -> None:
    """Load weekly roster snapshots from JSON files in ``rosters_dir``.

    Each ``*.json`` file must contain::

        {
          "snapshot_date": "2026-03-23",
          "week_num": 1,
          "team": "Hart of the Order",
          "roster": {
            "C":  {"name": "Ivan Herrera", "positions": ["C", "Util"]},
            "OF": {"name": "Juan Soto",    "positions": ["OF", "Util"]},
            ...
          }
        }

    Each slot key ("C", "OF", "P2", …) becomes one row in ``weekly_rosters``.
    The ``positions`` list is joined with ``", "``.

    Uses INSERT OR IGNORE so repeated calls are idempotent.
    """
    rosters_dir = Path(rosters_dir)
    rows = []
    for json_path in sorted(rosters_dir.glob("*.json")):
        data = json.loads(json_path.read_text(encoding="utf-8"))
        snapshot_date = data["snapshot_date"]
        week_num = data.get("week_num")
        team = data["team"]
        for slot, player_info in data.get("roster", {}).items():
            player_name = player_info.get("name", "")
            positions_list = player_info.get("positions", [])
            positions_str = ", ".join(positions_list) if positions_list else None
            rows.append((snapshot_date, week_num, team, slot, player_name, positions_str))

    conn.executemany(
        "INSERT OR IGNORE INTO weekly_rosters "
        "(snapshot_date, week_num, team, slot, player_name, positions) "
        "VALUES (?, ?, ?, ?, ?, ?)",
        rows,
    )
    conn.commit()


def load_positions(conn, positions: dict[str, list[str]]) -> None:
    """Load position eligibility into the positions table.

    ``positions`` is a dict mapping player name to a list of position strings.
    Uses INSERT OR REPLACE so repeated calls are idempotent.
    """
    rows = [(name, ", ".join(pos_list)) for name, pos_list in positions.items()]
    conn.executemany(
        "INSERT OR REPLACE INTO positions (name, positions) VALUES (?, ?)",
        rows,
    )
    conn.commit()


def get_roster_names(conn) -> set[str] | None:
    """Get normalized names of all rostered players from the latest roster snapshot.

    Returns a set of normalized player names (with Yahoo suffixes like
    "(Batter)" stripped), or None if no roster data exists.
    """
    rows = conn.execute(
        "SELECT DISTINCT player_name FROM weekly_rosters "
        "WHERE snapshot_date = (SELECT MAX(snapshot_date) FROM weekly_rosters)"
    ).fetchall()
    if not rows:
        return None
    return {normalize_name(_PLAYER_SUFFIX_RE.sub("", r["player_name"])) for r in rows}


def get_positions(conn) -> dict[str, list[str]]:
    """Read position eligibility from the database.

    Returns a dict mapping player name to list of position strings,
    matching the format of ``load_positions_cache()``.
    """
    rows = conn.execute("SELECT name, positions FROM positions").fetchall()
    return {row["name"]: [p.strip() for p in row["positions"].split(",")] for row in rows}


# Ordered list of columns in blended_projections.
_BLENDED_TABLE_COLS = [
    "year",
    "fg_id",
    "name",
    "team",
    "player_type",
    "pa",
    "ab",
    "h",
    "r",
    "hr",
    "rbi",
    "sb",
    "avg",
    "w",
    "k",
    "sv",
    "ip",
    "er",
    "bb",
    "h_allowed",
    "era",
    "whip",
    "adp",
]


def _df_to_insert_rows(
    df: pd.DataFrame,
    table_cols: list[str],
    **extra_columns,
) -> tuple[list[str], list[tuple]]:
    """Prepare a blended projection DataFrame for insertion.

    Adds any **extra_columns** (e.g. year=2026, snapshot_date='2026-03-30'),
    ensures ``fg_id`` exists (falls back to ``name``), selects only columns
    present in *table_cols*, and returns ``(column_names, rows)`` ready for
    ``executemany``.
    """
    df = df.copy()
    for col, val in extra_columns.items():
        df[col] = val

    if "fg_id" not in df.columns or df["fg_id"].isna().all():
        df["fg_id"] = df["name"]
    else:
        df["fg_id"] = df["fg_id"].fillna(df["name"])

    keep = [c for c in table_cols if c in df.columns]
    df = df[keep]

    rows = [
        tuple(None if pd.isna(v) else v for v in row)
        for row in df.itertuples(index=False, name=None)
    ]
    return keep, rows


def _insert_blended_dfs(conn, table_name, dfs, row_converter):
    """Insert hitter/pitcher DataFrames into a blended projections table."""
    for df in dfs:
        if df.empty:
            continue
        col_names, rows = row_converter(df)
        if not rows:
            continue
        placeholders = ", ".join("?" * len(col_names))
        insert_sql = (
            f"INSERT OR REPLACE INTO {table_name} ({', '.join(col_names)}) VALUES ({placeholders})"
        )
        conn.executemany(insert_sql, rows)


def load_blended_projections(
    conn,
    projections_dir,
    systems: list[str],
    weights: dict[str, float] | None = None,
    roster_names: set[str] | None = None,
    progress_cb=None,
) -> None:
    """Scan projections_dir for year subdirectories, blend projections for each
    year using the requested systems, and insert into blended_projections.

    Years where the requested systems have no files (e.g. a future year with
    only hitter CSVs) are skipped silently so one bad year doesn't abort the
    whole load.

    Uses INSERT OR REPLACE so repeated calls are idempotent.
    """
    projections_dir = Path(projections_dir)

    for year_dir in sorted(projections_dir.iterdir()):
        if not year_dir.is_dir() or not year_dir.name.isdigit():
            continue
        year = int(year_dir.name)

        try:
            hitters_df, pitchers_df, _ = blend_projections(
                year_dir,
                systems,
                weights,
                roster_names=roster_names,
                progress_cb=progress_cb,
            )
        except Exception:
            continue

        # Default-arg binding captures the current year so the closure
        # isn't sensitive to the loop variable rebinding (B023).
        def to_rows(df, year=year):
            return _df_to_insert_rows(df, _BLENDED_TABLE_COLS, year=year)

        _insert_blended_dfs(conn, "blended_projections", (hitters_df, pitchers_df), to_rows)

    conn.commit()


def get_blended_projections(
    conn,
    year: int | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Read blended projections from the database.

    Returns (hitters_df, pitchers_df) matching the format produced by
    ``blend_projections()`` in projections.py.

    If *year* is None, uses the maximum year in the table (current season).
    """
    if year is None:
        row = conn.execute("SELECT MAX(year) as y FROM blended_projections").fetchone()
        year = row["y"] if row and row["y"] is not None else 0

    hitters = pd.read_sql_query(
        "SELECT * FROM blended_projections WHERE year = ? AND player_type = 'hitter'",
        conn,
        params=(year,),
    )
    pitchers = pd.read_sql_query(
        "SELECT * FROM blended_projections WHERE year = ? AND player_type = 'pitcher'",
        conn,
        params=(year,),
    )

    # Drop the year column (not part of blend_projections output).
    # Keep player_type — downstream code (backfill, SGP, player_id) requires it.
    for df in (hitters, pitchers):
        if "year" in df.columns:
            df.drop(columns=["year"], inplace=True)

    return hitters, pitchers
