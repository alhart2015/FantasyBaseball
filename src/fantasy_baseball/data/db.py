"""SQLite database for fantasy baseball data."""

import json
import re
import sqlite3
from concurrent.futures import ThreadPoolExecutor, as_completed
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

CREATE TABLE IF NOT EXISTS game_logs (
    season       INTEGER NOT NULL,
    mlbam_id     INTEGER NOT NULL,
    name         TEXT NOT NULL,
    team         TEXT,
    player_type  TEXT NOT NULL,          -- hitter, pitcher
    date         TEXT NOT NULL,          -- YYYY-MM-DD
    -- Hitter stats
    pa INTEGER, ab INTEGER, h INTEGER, r INTEGER, hr INTEGER,
    rbi INTEGER, sb INTEGER,
    -- Pitcher stats
    ip REAL, k INTEGER, er INTEGER, bb INTEGER, h_allowed INTEGER,
    w INTEGER, sv INTEGER, gs INTEGER,
    PRIMARY KEY (season, mlbam_id, date)
);

CREATE INDEX IF NOT EXISTS idx_game_logs_name ON game_logs(name);
CREATE INDEX IF NOT EXISTS idx_game_logs_date ON game_logs(season, date);

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

CREATE TABLE IF NOT EXISTS transactions (
    year            INTEGER NOT NULL,
    transaction_id  TEXT NOT NULL,
    timestamp       TEXT,
    team            TEXT NOT NULL,
    team_key        TEXT,
    type            TEXT NOT NULL,
    add_name        TEXT,
    add_player_id   TEXT,
    add_positions   TEXT,
    drop_name       TEXT,
    drop_player_id  TEXT,
    drop_positions  TEXT,
    add_wsgp        REAL,
    drop_wsgp       REAL,
    value           REAL,
    paired_with     TEXT,
    PRIMARY KEY (year, transaction_id)
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
    "year", "system", "player_type",
    "name", "team", "fg_id", "mlbam_id",
    "pa", "ab", "h", "r", "hr", "rbi", "sb", "cs", "bb", "so",
    "avg", "obp", "slg", "ops", "iso", "babip", "woba", "wrc_plus", "war",
    "w", "l", "sv", "ip", "er", "k", "bb_p", "h_allowed",
    "era", "whip", "fip", "k9", "bb9", "hr_p", "war_p",
    "adp", "g",
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
                f"INSERT OR IGNORE INTO raw_projections ({col_names}) "
                f"VALUES ({placeholders})"
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
            rows.append((
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
            ))

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


# ---------------------------------------------------------------------------
# Live append functions (called from the dashboard refresh)
# ---------------------------------------------------------------------------


def append_roster_snapshot(conn, roster, snapshot_date, week_num, team) -> None:
    """Insert one row per player into weekly_rosters.

    ``roster`` is a list of player dicts::

        [{"name": "...", "selected_position": "OF", "positions": ["OF", "Util"],
          "status": "IL10", "player_id": "12345"}, ...]

    ``slot`` is taken from ``player["selected_position"]``. ``positions``
    is the player's eligible positions joined with ``", "``. ``status``
    and ``player_id`` are optional — missing keys write as NULL.

    Uses INSERT OR IGNORE so repeated calls with the same
    (snapshot_date, team, slot, player_name) are idempotent.
    """
    rows = []
    for player in roster:
        slot = player["selected_position"]
        positions_str = ", ".join(player.get("positions", []))
        status = player.get("status")
        yahoo_id = player.get("player_id")
        rows.append((
            snapshot_date,
            week_num,
            team,
            slot,
            player["name"],
            positions_str or None,
            status if status is not None else None,
            yahoo_id if yahoo_id is not None else None,
        ))

    conn.executemany(
        "INSERT OR IGNORE INTO weekly_rosters "
        "(snapshot_date, week_num, team, slot, player_name, positions, "
        " status, yahoo_id) "
        "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
        rows,
    )
    conn.commit()


def append_standings_snapshot(conn, standings, year, snapshot_date) -> None:
    """Insert one row per team into standings.

    ``standings`` is a list of team dicts::

        [{"name": "...", "rank": 1, "stats": {"R": 100, "HR": 30, ...}}, ...]

    Stat keys are case-insensitive.  Uses INSERT OR IGNORE so repeated calls
    with the same (year, snapshot_date, team) are idempotent.
    """
    rows = []
    for entry in standings:
        stats = {k.lower(): v for k, v in entry.get("stats", {}).items()}
        rows.append((
            year,
            snapshot_date,
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
        ))

    conn.executemany(
        "INSERT OR IGNORE INTO standings "
        "(year, snapshot_date, team, rank, r, hr, rbi, sb, avg, w, k, sv, era, whip) "
        "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
        rows,
    )
    conn.commit()


def insert_transactions(conn, transactions):
    """Insert scored transaction rows. Uses INSERT OR IGNORE for idempotency."""
    rows = []
    for t in transactions:
        rows.append((
            t["year"], t["transaction_id"], t.get("timestamp"),
            t["team"], t.get("team_key"), t["type"],
            t.get("add_name"), t.get("add_player_id"), t.get("add_positions"),
            t.get("drop_name"), t.get("drop_player_id"), t.get("drop_positions"),
            t.get("add_wsgp"), t.get("drop_wsgp"), t.get("value"),
            t.get("paired_with"),
        ))
    conn.executemany(
        "INSERT OR IGNORE INTO transactions "
        "(year, transaction_id, timestamp, team, team_key, type, "
        "add_name, add_player_id, add_positions, "
        "drop_name, drop_player_id, drop_positions, "
        "add_wsgp, drop_wsgp, value, paired_with) "
        "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
        rows,
    )
    conn.commit()


def get_transaction_ids(conn, year):
    """Return set of transaction_ids already stored for a year."""
    rows = conn.execute(
        "SELECT transaction_id FROM transactions WHERE year = ?",
        (year,),
    ).fetchall()
    return {r["transaction_id"] for r in rows}


def get_all_transactions(conn, year):
    """Load all transactions for a year, ordered by timestamp."""
    rows = conn.execute(
        "SELECT * FROM transactions WHERE year = ? ORDER BY timestamp",
        (year,),
    ).fetchall()
    return [dict(r) for r in rows]


def update_transaction_pairing(conn, year, txn_id_a, txn_id_b):
    """Mark two transactions as paired with each other."""
    conn.execute(
        "UPDATE transactions SET paired_with = ? "
        "WHERE year = ? AND transaction_id = ?",
        (txn_id_b, year, txn_id_a),
    )
    conn.execute(
        "UPDATE transactions SET paired_with = ? "
        "WHERE year = ? AND transaction_id = ?",
        (txn_id_a, year, txn_id_b),
    )
    conn.commit()


def load_positions(conn, positions: dict[str, list[str]]) -> None:
    """Load position eligibility into the positions table.

    ``positions`` is a dict mapping player name to a list of position strings.
    Uses INSERT OR REPLACE so repeated calls are idempotent.
    """
    rows = [
        (name, ", ".join(pos_list))
        for name, pos_list in positions.items()
    ]
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
    return {
        normalize_name(_PLAYER_SUFFIX_RE.sub("", r["player_name"]))
        for r in rows
    }


def get_positions(conn) -> dict[str, list[str]]:
    """Read position eligibility from the database.

    Returns a dict mapping player name to list of position strings,
    matching the format of ``load_positions_cache()``.
    """
    rows = conn.execute("SELECT name, positions FROM positions").fetchall()
    return {
        row["name"]: [p.strip() for p in row["positions"].split(",")]
        for row in rows
    }


# Ordered list of columns in blended_projections.
_BLENDED_TABLE_COLS = [
    "year", "fg_id", "name", "team", "player_type",
    "pa", "ab", "h", "r", "hr", "rbi", "sb", "avg",
    "w", "k", "sv", "ip", "er", "bb", "h_allowed",
    "era", "whip", "adp",
]

# ROS table has the same columns with snapshot_date added after year.
_ROS_TABLE_COLS = [_BLENDED_TABLE_COLS[0], "snapshot_date"] + _BLENDED_TABLE_COLS[1:]


def _df_to_insert_rows(
    df: pd.DataFrame, table_cols: list[str], **extra_columns,
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
            f"INSERT OR REPLACE INTO {table_name} ({', '.join(col_names)}) "
            f"VALUES ({placeholders})"
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
                year_dir, systems, weights,
                roster_names=roster_names, progress_cb=progress_cb,
            )
        except Exception:
            continue

        def to_rows(df):
            return _df_to_insert_rows(df, _BLENDED_TABLE_COLS, year=year)

        _insert_blended_dfs(conn, "blended_projections", (hitters_df, pitchers_df), to_rows)

    conn.commit()


def load_ros_projections(
    conn,
    projections_dir,
    systems: list[str],
    weights: dict[str, float] | None = None,
    roster_names: set[str] | None = None,
    progress_cb=None,
) -> None:
    """Scan projections_dir for ROS snapshot directories and load them.

    Directory structure expected::

        projections_dir/{year}/ros/{YYYY-MM-DD}/{system}-hitters.csv

    For each date subdirectory found, calls ``blend_projections()`` and inserts
    the result into ``ros_blended_projections``.  Date dirs that are missing the
    requested system files are skipped silently.  Uses INSERT OR REPLACE so
    repeated calls are idempotent.
    """
    projections_dir = Path(projections_dir)

    from fantasy_baseball.data.projections import normalize_ros_to_full_season
    from datetime import date

    # All FanGraphs ROS exports are remaining-games-only — every system gets
    # normalized to full-season by adding accumulated actuals from game_logs.
    hitter_totals, pitcher_totals = get_season_totals(conn, date.today().year)

    def _normalizer(system_name, hitters_df, pitchers_df):
        if progress_cb:
            progress_cb(f"Normalizing {system_name} ROS → full-season")
        h = normalize_ros_to_full_season(hitters_df, hitter_totals, PlayerType.HITTER)
        p = normalize_ros_to_full_season(pitchers_df, pitcher_totals, PlayerType.PITCHER)
        return h, p

    for year_dir in sorted(projections_dir.iterdir()):
        if not year_dir.is_dir() or not year_dir.name.isdigit():
            continue
        year = int(year_dir.name)

        ros_dir = year_dir / "ros"
        if not ros_dir.is_dir():
            continue

        for date_dir in sorted(ros_dir.iterdir()):
            if not date_dir.is_dir():
                continue
            snapshot_date = date_dir.name

            try:
                hitters_df, pitchers_df, _ = blend_projections(
                    date_dir, systems, weights,
                    roster_names=roster_names, progress_cb=progress_cb,
                    normalizer=_normalizer,
                )
            except Exception as exc:
                # Surface what blew up so we can diagnose it from the job log
                # instead of silently dropping a snapshot. Uses progress_cb if
                # available so the message lands in the same JobLogger entries
                # the rest of the pipeline writes to.
                import traceback
                msg = (
                    f"ERROR loading {snapshot_date}: "
                    f"{type(exc).__name__}: {exc}"
                )
                if progress_cb:
                    progress_cb(msg)
                    progress_cb(
                        f"ERROR traceback: {traceback.format_exc().splitlines()[-3:]}"
                    )
                continue

            def to_rows(df, _y=year, _sd=snapshot_date):
                return _df_to_insert_rows(df, _ROS_TABLE_COLS, year=_y, snapshot_date=_sd)

            _insert_blended_dfs(
                conn, "ros_blended_projections", (hitters_df, pitchers_df), to_rows,
            )

    conn.commit()


def get_ros_projections(
    conn, year: int | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Read the latest ROS blended projections from the database.

    Finds the maximum ``snapshot_date`` for the requested *year* (or maximum
    year when *year* is None), then returns ``(hitters_df, pitchers_df)``
    matching the format produced by ``blend_projections()``.

    Returns empty DataFrames if no ROS data exists.
    """
    if year is None:
        row = conn.execute(
            "SELECT MAX(year) as y FROM ros_blended_projections"
        ).fetchone()
        year = row["y"] if row and row["y"] is not None else None

    if year is None:
        empty = pd.DataFrame()
        return empty, empty

    row = conn.execute(
        "SELECT MAX(snapshot_date) as d FROM ros_blended_projections WHERE year = ?",
        (year,),
    ).fetchone()
    snapshot_date = row["d"] if row and row["d"] is not None else None

    if snapshot_date is None:
        empty = pd.DataFrame()
        return empty, empty

    hitters = pd.read_sql_query(
        "SELECT * FROM ros_blended_projections "
        "WHERE year = ? AND snapshot_date = ? AND player_type = 'hitter'",
        conn, params=(year, snapshot_date),
    )
    pitchers = pd.read_sql_query(
        "SELECT * FROM ros_blended_projections "
        "WHERE year = ? AND snapshot_date = ? AND player_type = 'pitcher'",
        conn, params=(year, snapshot_date),
    )

    # Drop year and snapshot_date — not part of blend_projections output.
    # Keep player_type — downstream code requires it.
    for df in (hitters, pitchers):
        for col in ("year", "snapshot_date"):
            if col in df.columns:
                df.drop(columns=[col], inplace=True)

    return hitters, pitchers


def get_blended_projections(
    conn, year: int | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Read blended projections from the database.

    Returns (hitters_df, pitchers_df) matching the format produced by
    ``blend_projections()`` in projections.py.

    If *year* is None, uses the maximum year in the table (current season).
    """
    if year is None:
        row = conn.execute(
            "SELECT MAX(year) as y FROM blended_projections"
        ).fetchone()
        year = row["y"] if row and row["y"] is not None else 0

    hitters = pd.read_sql_query(
        "SELECT * FROM blended_projections WHERE year = ? AND player_type = 'hitter'",
        conn, params=(year,),
    )
    pitchers = pd.read_sql_query(
        "SELECT * FROM blended_projections WHERE year = ? AND player_type = 'pitcher'",
        conn, params=(year,),
    )

    # Drop the year column (not part of blend_projections output).
    # Keep player_type — downstream code (backfill, SGP, player_id) requires it.
    for df in (hitters, pitchers):
        if "year" in df.columns:
            df.drop(columns=["year"], inplace=True)

    return hitters, pitchers


def get_season_totals(
    conn, season: int,
) -> tuple[dict[int, dict], dict[int, dict]]:
    """Get accumulated season stats from game_logs, keyed by mlbam_id.

    Returns (hitter_totals, pitcher_totals) where each is
    {mlbam_id: {stat: value}}.
    """
    hitter_totals = {}
    rows = conn.execute(
        "SELECT mlbam_id, SUM(pa) as pa, SUM(ab) as ab, SUM(h) as h, "
        "SUM(r) as r, SUM(hr) as hr, SUM(rbi) as rbi, SUM(sb) as sb "
        "FROM game_logs WHERE season = ? AND player_type = 'hitter' "
        "GROUP BY mlbam_id", (season,)
    ).fetchall()
    for row in rows:
        hitter_totals[row["mlbam_id"]] = {
            "pa": row["pa"] or 0, "ab": row["ab"] or 0, "h": row["h"] or 0,
            "r": row["r"] or 0, "hr": row["hr"] or 0, "rbi": row["rbi"] or 0,
            "sb": row["sb"] or 0,
        }

    pitcher_totals = {}
    rows = conn.execute(
        "SELECT mlbam_id, SUM(ip) as ip, SUM(k) as k, SUM(w) as w, SUM(sv) as sv, "
        "SUM(er) as er, SUM(bb) as bb, SUM(h_allowed) as h_allowed "
        "FROM game_logs WHERE season = ? AND player_type = 'pitcher' "
        "GROUP BY mlbam_id", (season,)
    ).fetchall()
    for row in rows:
        pitcher_totals[row["mlbam_id"]] = {
            "ip": row["ip"] or 0, "k": row["k"] or 0, "w": row["w"] or 0,
            "sv": row["sv"] or 0, "er": row["er"] or 0, "bb": row["bb"] or 0,
            "h_allowed": row["h_allowed"] or 0,
        }

    return hitter_totals, pitcher_totals


def load_projections_for_date(
    conn, year: int, target_date: str
) -> tuple["pd.DataFrame", "pd.DataFrame"]:
    """Find the best ROS blended projections for a target date.

    Queries ros_blended_projections for the MAX snapshot_date <= target_date.
    Falls back to blended_projections (preseason) if no ROS data exists.
    Used by transaction scoring to look up the projections that were
    available when a historical transaction happened.
    """
    import pandas as pd
    from fantasy_baseball.utils.name_utils import normalize_name

    row = conn.execute(
        "SELECT MAX(snapshot_date) as best_date "
        "FROM ros_blended_projections "
        "WHERE year = ? AND snapshot_date <= ?",
        (year, target_date),
    ).fetchone()

    best_date = row["best_date"] if row else None

    if best_date is not None:
        rows = conn.execute(
            "SELECT * FROM ros_blended_projections "
            "WHERE year = ? AND snapshot_date = ?",
            (year, best_date),
        ).fetchall()
        df = pd.DataFrame([dict(r) for r in rows])
    else:
        rows = conn.execute(
            "SELECT * FROM blended_projections WHERE year = ?",
            (year,),
        ).fetchall()
        df = pd.DataFrame([dict(r) for r in rows])

    if df.empty:
        empty = pd.DataFrame()
        return empty, empty

    df["_name_norm"] = df["name"].apply(normalize_name)

    hitters_df = df[df["player_type"] == "hitter"].reset_index(drop=True)
    pitchers_df = df[df["player_type"] == "pitcher"].reset_index(drop=True)

    return hitters_df, pitchers_df


def fetch_and_load_game_logs(
    conn, season: int, progress_cb=None
) -> int:
    """Fetch game logs for all MLB players and insert into game_logs table.

    Incrementally updates — only inserts games not already in the DB.
    Uses the MLB Stats API to get all team rosters, then fetches per-player
    game logs via the existing fetch_player_game_log() function.

    Returns:
        Number of new game log rows inserted.
    """
    import statsapi

    from fantasy_baseball.analysis.game_logs import fetch_player_game_log

    # Get the latest date we have per player so we can skip up-to-date ones
    existing = {}
    for row in conn.execute(
        "SELECT mlbam_id, MAX(date) as last_date FROM game_logs "
        "WHERE season = ? GROUP BY mlbam_id", (season,)
    ):
        existing[row["mlbam_id"]] = row["last_date"]

    # Get all MLB teams
    if progress_cb:
        progress_cb("Fetching MLB team rosters...")
    teams_data = statsapi.get("teams", {"sportId": 1, "season": season})
    teams_list = teams_data.get("teams", [])

    # Build player list from rosters
    players = []
    seen_ids = set()
    for team in teams_list:
        team_id = team["id"]
        team_abbrev = team.get("abbreviation", "")
        try:
            roster_data = statsapi.get(
                "team_roster",
                {"teamId": team_id, "rosterType": "fullSeason", "season": season},
            )
        except Exception:
            try:
                roster_data = statsapi.get(
                    "team_roster",
                    {"teamId": team_id, "rosterType": "active", "season": season},
                )
            except Exception:
                continue

        for entry in roster_data.get("roster", []):
            person = entry.get("person", {})
            mlbam_id = person.get("id")
            if not mlbam_id or mlbam_id in seen_ids:
                continue
            seen_ids.add(mlbam_id)

            pos_type = entry.get("position", {}).get("type", "")
            player_type = PlayerType.PITCHER if pos_type == "Pitcher" else PlayerType.HITTER

            players.append({
                "mlbam_id": mlbam_id,
                "name": person.get("fullName", ""),
                "team": team_abbrev,
                "player_type": player_type,
            })

    if progress_cb:
        progress_cb(f"Found {len(players)} MLB players, fetching game logs...")

    def _fetch_one(player):
        mid = player["mlbam_id"]
        group = "hitting" if player["player_type"] == PlayerType.HITTER else "pitching"
        try:
            games = fetch_player_game_log(mid, season, group)
        except Exception:
            return (player, [])
        return (player, games or [])

    new_rows = 0
    done_count = 0
    _HITTER_INSERT = (
        "INSERT OR IGNORE INTO game_logs "
        "(season, mlbam_id, name, team, player_type, date, pa, ab, h, r, hr, rbi, sb) "
        "VALUES (?, ?, ?, ?, 'hitter', ?, ?, ?, ?, ?, ?, ?, ?)"
    )
    _PITCHER_INSERT = (
        "INSERT OR IGNORE INTO game_logs "
        "(season, mlbam_id, name, team, player_type, date, ip, k, er, bb, h_allowed, w, sv, gs) "
        "VALUES (?, ?, ?, ?, 'pitcher', ?, ?, ?, ?, ?, ?, ?, ?, ?)"
    )

    # Fetch in parallel, insert as results arrive to limit memory usage
    with ThreadPoolExecutor(max_workers=15) as pool:
        futures = [pool.submit(_fetch_one, p) for p in players]
        for future in as_completed(futures):
            player, games = future.result()
            done_count += 1

            if games:
                mid = player["mlbam_id"]
                last_date = existing.get(mid)
                if last_date:
                    games = [g for g in games if g["date"] > last_date]

                if games:
                    common = (season, mid, player["name"], player["team"])
                    if player["player_type"] == PlayerType.HITTER:
                        rows = [
                            (*common, g["date"], g.get("pa"), g.get("ab"), g.get("h"),
                             g.get("r"), g.get("hr"), g.get("rbi"), g.get("sb"))
                            for g in games
                        ]
                        conn.executemany(_HITTER_INSERT, rows)
                    else:
                        rows = [
                            (*common, g["date"], g.get("ip"), g.get("k"), g.get("er"),
                             g.get("bb"), g.get("h_allowed"), g.get("w"), g.get("sv"), g.get("gs"))
                            for g in games
                        ]
                        conn.executemany(_PITCHER_INSERT, rows)
                    new_rows += len(rows)

            if done_count % 50 == 0:
                conn.commit()
                if progress_cb:
                    progress_cb(f"Game logs: {done_count}/{len(players)} players...")

    conn.commit()
    return new_rows
