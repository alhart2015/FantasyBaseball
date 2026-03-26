"""SQLite database for fantasy baseball data."""

import json
import re
import sqlite3
from pathlib import Path

import pandas as pd

from fantasy_baseball.data.projections import blend_projections
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
    PRIMARY KEY (snapshot_date, team, slot)
);

CREATE TABLE IF NOT EXISTS standings (
    year          INTEGER NOT NULL,
    snapshot_date TEXT NOT NULL,
    team          TEXT NOT NULL,
    rank          INTEGER,
    r REAL, hr REAL, rbi REAL, sb REAL, avg REAL,
    w REAL, k REAL, sv REAL, era REAL, whip REAL,
    PRIMARY KEY (year, snapshot_date, team)
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


def create_tables(conn):
    """Create all tables (idempotent via IF NOT EXISTS)."""
    conn.executescript(SCHEMA)
    conn.commit()


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
    player_type = "hitter" if m.group("ptype") == "hitters" else "pitcher"
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

            col_map = _HITTER_COLS if player_type == "hitter" else _PITCHER_COLS

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


# Ordered list of columns in blended_projections (excluding the PRIMARY KEY pair
# which we always supply explicitly).
_BLENDED_TABLE_COLS = [
    "year", "fg_id", "name", "team", "player_type",
    "pa", "ab", "h", "r", "hr", "rbi", "sb", "avg",
    "w", "k", "sv", "ip", "er", "bb", "h_allowed",
    "era", "whip", "adp",
]


def _df_to_blended_rows(df: pd.DataFrame, year: int) -> tuple[list[str], list[tuple]]:
    """Prepare a blended projection DataFrame for insertion.

    Adds the ``year`` column, ensures ``fg_id`` exists (falls back to ``name``),
    selects only columns present in the table schema, and returns
    ``(column_names, rows)`` ready for ``executemany``.
    """
    df = df.copy()
    df["year"] = year

    # Ensure fg_id — fall back to name if the column is absent or all-null
    if "fg_id" not in df.columns or df["fg_id"].isna().all():
        df["fg_id"] = df["name"]
    else:
        df["fg_id"] = df["fg_id"].fillna(df["name"])

    # Select only columns that exist in both the DataFrame and the table schema
    keep = [c for c in _BLENDED_TABLE_COLS if c in df.columns]
    df = df[keep]

    rows = [
        tuple(None if pd.isna(v) else v for v in row)
        for row in df.itertuples(index=False, name=None)
    ]
    return keep, rows


def load_blended_projections(
    conn,
    projections_dir,
    systems: list[str],
    weights: dict[str, float] | None = None,
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
            hitters_df, pitchers_df = blend_projections(year_dir, systems, weights)
        except Exception:
            # Missing files, unrecognised format, etc. — skip this year.
            continue

        for df in (hitters_df, pitchers_df):
            if df.empty:
                continue
            col_names, rows = _df_to_blended_rows(df, year)
            if not rows:
                continue
            placeholders = ", ".join("?" * len(col_names))
            insert_sql = (
                f"INSERT OR REPLACE INTO blended_projections ({', '.join(col_names)}) "
                f"VALUES ({placeholders})"
            )
            conn.executemany(insert_sql, rows)

    conn.commit()
