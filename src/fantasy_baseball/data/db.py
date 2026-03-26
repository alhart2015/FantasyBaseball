"""SQLite database for fantasy baseball data."""

import sqlite3
from pathlib import Path

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
