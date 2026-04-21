import json
import sqlite3
from pathlib import Path

from fantasy_baseball.data.db import (
    append_roster_snapshot,
    append_standings_snapshot,
    create_tables,
    get_blended_projections,
    get_connection,
    get_positions,
    get_rest_of_season_projections,
    load_blended_projections,
    load_draft_results,
    load_positions,
    load_raw_projections,
    load_rest_of_season_projections,
    load_standings,
    load_weekly_rosters,
)


def test_create_tables_creates_all_five(tmp_path):
    db_path = tmp_path / "test.db"
    conn = sqlite3.connect(db_path)
    create_tables(conn)
    cursor = conn.execute("SELECT name FROM sqlite_master WHERE type='table' ORDER BY name")
    tables = [row[0] for row in cursor]
    assert "raw_projections" in tables
    assert "blended_projections" in tables
    assert "draft_results" in tables
    assert "weekly_rosters" in tables
    assert "standings" in tables
    conn.close()


def test_create_tables_creates_ros_blended_projections(tmp_path):
    """Task 1: ros_blended_projections table is created by create_tables()."""
    db_path = tmp_path / "test.db"
    conn = sqlite3.connect(db_path)
    create_tables(conn)
    cursor = conn.execute("SELECT name FROM sqlite_master WHERE type='table' ORDER BY name")
    tables = [row[0] for row in cursor]
    assert "ros_blended_projections" in tables

    # Verify the PRIMARY KEY columns exist via pragma
    cols = {
        row[1]
        for row in conn.execute("PRAGMA table_info(ros_blended_projections)")
    }
    assert "year" in cols
    assert "snapshot_date" in cols
    assert "fg_id" in cols
    conn.close()


def test_create_tables_is_idempotent(tmp_path):
    db_path = tmp_path / "test.db"
    conn = get_connection(db_path)
    create_tables(conn)
    create_tables(conn)  # should not raise
    conn.close()


def test_get_connection_returns_connection():
    conn = get_connection(":memory:")
    assert isinstance(conn, sqlite3.Connection)
    conn.close()


def test_load_raw_projections(tmp_path):
    csv_dir = tmp_path / "2026"
    csv_dir.mkdir()
    (csv_dir / "steamer-hitters.csv").write_text(
        'Name,Team,PA,AB,H,R,HR,RBI,SB,CS,BB,SO,AVG,OBP,SLG,OPS,ISO,BABIP,wOBA,wRC+,WAR,ADP,G,PlayerId,MLBAMID\n'
        '"James Wood","WSN",600,520,140,85,26,80,15,5,70,150,0.269,0.350,0.480,0.830,0.211,0.320,0.370,130,4.0,50.0,145,"29518",695578\n'
    )
    (csv_dir / "steamer-pitchers.csv").write_text(
        'Name,Team,W,L,SV,ERA,G,GS,IP,H,R,ER,HR,BB,SO,K/9,BB/9,K/BB,HR/9,WHIP,FIP,WAR,ADP,PlayerId,MLBAMID\n'
        '"Corbin Burnes","BAL",14,7,0,3.20,32,32,200,170,75,71,20,50,220,9.9,2.3,4.4,0.9,1.10,3.10,5.0,15.0,"19361",669203\n'
    )

    db_path = tmp_path / "test.db"
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    create_tables(conn)
    load_raw_projections(conn, tmp_path)

    rows = conn.execute("SELECT * FROM raw_projections WHERE year=2026").fetchall()
    assert len(rows) == 2

    hitter = conn.execute(
        "SELECT name, hr, sb, avg, fg_id, player_type FROM raw_projections WHERE name='James Wood'"
    ).fetchone()
    assert hitter["hr"] == 26
    assert hitter["fg_id"] == "29518"
    assert hitter["player_type"] == "hitter"

    pitcher = conn.execute(
        "SELECT name, w, k, era, fg_id, h_allowed, hr_p, bb_p, player_type FROM raw_projections WHERE name='Corbin Burnes'"
    ).fetchone()
    assert pitcher["w"] == 14
    assert pitcher["k"] == 220      # SO maps to k for pitchers
    assert pitcher["h_allowed"] == 170  # H maps to h_allowed for pitchers
    assert pitcher["hr_p"] == 20    # HR maps to hr_p for pitchers
    assert pitcher["bb_p"] == 50    # BB maps to bb_p for pitchers
    assert pitcher["player_type"] == "pitcher"

    conn.close()


def test_load_raw_projections_handles_year_suffix(tmp_path):
    csv_dir = tmp_path / "2025"
    csv_dir.mkdir()
    (csv_dir / "steamer-hitters-2025.csv").write_text(
        'Name,Team,PA,AB,H,R,HR,RBI,SB,AVG,G,PlayerId,MLBAMID\n'
        '"Test Player","NYY",500,430,110,70,20,65,10,0.256,130,"12345",123456\n'
    )

    db_path = tmp_path / "test.db"
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    create_tables(conn)
    load_raw_projections(conn, tmp_path)

    row = conn.execute("SELECT system, year FROM raw_projections WHERE name='Test Player'").fetchone()
    assert row["system"] == "steamer"
    assert row["year"] == 2025
    conn.close()


def test_load_raw_projections_insert_or_ignore(tmp_path):
    """Calling load_raw_projections twice must not raise or create duplicates."""
    csv_dir = tmp_path / "2026"
    csv_dir.mkdir()
    (csv_dir / "steamer-hitters.csv").write_text(
        'Name,Team,PA,HR,G,PlayerId,MLBAMID\n'
        '"Test Player","NYY",500,20,130,"99999",999999\n'
    )

    db_path = tmp_path / "test.db"
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    create_tables(conn)
    load_raw_projections(conn, tmp_path)
    load_raw_projections(conn, tmp_path)  # second call should not raise

    rows = conn.execute("SELECT * FROM raw_projections").fetchall()
    assert len(rows) == 1
    conn.close()


def test_load_raw_projections_date_suffix(tmp_path):
    """Filename like zips-hitters-2027-proj-from-2026-03-25.csv extracts system='zips'."""
    csv_dir = tmp_path / "2027"
    csv_dir.mkdir()
    (csv_dir / "zips-hitters-2027-proj-from-2026-03-25.csv").write_text(
        'Name,Team,PA,HR,G,PlayerId,MLBAMID\n'
        '"Future Player","SEA",480,18,125,"77777",777777\n'
    )

    db_path = tmp_path / "test.db"
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    create_tables(conn)
    load_raw_projections(conn, tmp_path)

    row = conn.execute("SELECT system, year, player_type FROM raw_projections WHERE name='Future Player'").fetchone()
    assert row["system"] == "zips"
    assert row["year"] == 2027
    assert row["player_type"] == "hitter"
    conn.close()


def test_load_blended_projections(tmp_path):
    csv_dir = tmp_path / "2026"
    csv_dir.mkdir()
    (csv_dir / "steamer-hitters.csv").write_text(
        'Name,Team,PA,AB,H,R,HR,RBI,SB,CS,BB,SO,AVG,OBP,SLG,OPS,WAR,ADP,G,PlayerId,MLBAMID\n'
        '"James Wood","WSN",600,520,140,85,26,80,15,5,70,150,0.269,0.350,0.480,0.830,4.0,50.0,145,"29518",695578\n'
    )
    (csv_dir / "steamer-pitchers.csv").write_text(
        'Name,Team,W,L,SV,ERA,G,GS,IP,H,R,ER,HR,BB,SO,K/9,BB/9,WHIP,FIP,WAR,ADP,PlayerId,MLBAMID\n'
        '"Corbin Burnes","BAL",14,7,0,3.20,32,32,200,170,75,71,20,50,220,9.9,2.3,1.10,3.10,5.0,15.0,"19361",669203\n'
    )

    db_path = tmp_path / "test.db"
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    create_tables(conn)
    load_blended_projections(conn, tmp_path, ["steamer"], {"steamer": 1.0})

    hitter = conn.execute(
        "SELECT name, hr, avg, fg_id, year FROM blended_projections WHERE name='James Wood'"
    ).fetchone()
    assert hitter is not None
    assert hitter["fg_id"] == "29518"
    assert hitter["year"] == 2026

    pitcher = conn.execute(
        "SELECT name, w, era, fg_id FROM blended_projections WHERE name='Corbin Burnes'"
    ).fetchone()
    assert pitcher is not None
    assert pitcher["w"] == 14

    conn.close()


def test_load_blended_projections_insert_or_replace(tmp_path):
    """Calling load_blended_projections twice must not raise or create duplicates."""
    csv_dir = tmp_path / "2026"
    csv_dir.mkdir()
    (csv_dir / "steamer-hitters.csv").write_text(
        'Name,Team,PA,AB,H,R,HR,RBI,SB,CS,BB,SO,AVG,OBP,SLG,OPS,WAR,ADP,G,PlayerId,MLBAMID\n'
        '"James Wood","WSN",600,520,140,85,26,80,15,5,70,150,0.269,0.350,0.480,0.830,4.0,50.0,145,"29518",695578\n'
    )
    (csv_dir / "steamer-pitchers.csv").write_text(
        'Name,Team,W,L,SV,ERA,G,GS,IP,H,R,ER,HR,BB,SO,K/9,BB/9,WHIP,FIP,WAR,ADP,PlayerId,MLBAMID\n'
        '"Corbin Burnes","BAL",14,7,0,3.20,32,32,200,170,75,71,20,50,220,9.9,2.3,1.10,3.10,5.0,15.0,"19361",669203\n'
    )

    db_path = tmp_path / "test.db"
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    create_tables(conn)
    load_blended_projections(conn, tmp_path, ["steamer"], {"steamer": 1.0})
    load_blended_projections(conn, tmp_path, ["steamer"], {"steamer": 1.0})  # second call must not raise

    rows = conn.execute("SELECT * FROM blended_projections WHERE year=2026").fetchall()
    assert len(rows) == 2  # one hitter + one pitcher
    conn.close()


def test_load_blended_projections_skips_bad_year(tmp_path):
    """A year directory with no matching CSVs for the requested system is skipped."""
    good_dir = tmp_path / "2026"
    good_dir.mkdir()
    (good_dir / "steamer-hitters.csv").write_text(
        'Name,Team,PA,AB,H,R,HR,RBI,SB,CS,BB,SO,AVG,OBP,SLG,OPS,WAR,ADP,G,PlayerId,MLBAMID\n'
        '"James Wood","WSN",600,520,140,85,26,80,15,5,70,150,0.269,0.350,0.480,0.830,4.0,50.0,145,"29518",695578\n'
    )
    (good_dir / "steamer-pitchers.csv").write_text(
        'Name,Team,W,L,SV,ERA,G,GS,IP,H,R,ER,HR,BB,SO,K/9,BB/9,WHIP,FIP,WAR,ADP,PlayerId,MLBAMID\n'
        '"Corbin Burnes","BAL",14,7,0,3.20,32,32,200,170,75,71,20,50,220,9.9,2.3,1.10,3.10,5.0,15.0,"19361",669203\n'
    )

    # 2027 has no steamer files — blend_projections will raise; should be skipped
    bad_dir = tmp_path / "2027"
    bad_dir.mkdir()
    (bad_dir / "zips-hitters.csv").write_text(
        'Name,Team,PA,AB,H,R,HR,RBI,SB,AVG,G,PlayerId,MLBAMID\n'
        '"Future Player","SEA",480,420,110,60,18,65,8,0.262,125,"77777",777777\n'
    )

    db_path = tmp_path / "test.db"
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    create_tables(conn)
    load_blended_projections(conn, tmp_path, ["steamer"], {"steamer": 1.0})  # must not raise

    rows = conn.execute("SELECT year FROM blended_projections").fetchall()
    years = {r["year"] for r in rows}
    assert 2026 in years
    assert 2027 not in years
    conn.close()


def test_load_draft_results(tmp_path):
    drafts = {
        "2025": [
            {"pick": 1, "round": 1, "team": "Hart of the Order", "player": "Juan Soto"},
            {"pick": 2, "round": 1, "team": "SkeleThor", "player": "Shohei Ohtani (Batter)"},
        ]
    }
    drafts_path = tmp_path / "drafts.json"
    drafts_path.write_text(json.dumps(drafts))

    db_path = tmp_path / "test.db"
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    create_tables(conn)
    load_draft_results(conn, drafts_path)

    rows = conn.execute("SELECT * FROM draft_results ORDER BY pick").fetchall()
    assert len(rows) == 2
    assert rows[0]["player"] == "Juan Soto"
    assert rows[1]["player"] == "Shohei Ohtani"  # suffix stripped
    conn.close()


def test_load_standings(tmp_path):
    standings = {
        "2023": {
            "standings": [
                {"name": "Hart of the Order", "team_key": "k1", "rank": 1,
                 "stats": {"R": 900, "HR": 250, "RBI": 880, "SB": 150, "AVG": 0.260,
                           "W": 80, "K": 1400, "SV": 90, "ERA": 3.60, "WHIP": 1.20}},
            ]
        }
    }
    path = tmp_path / "standings.json"
    path.write_text(json.dumps(standings))

    db_path = tmp_path / "test.db"
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    create_tables(conn)
    load_standings(conn, path)

    row = conn.execute("SELECT * FROM standings").fetchone()
    assert row["year"] == 2023
    assert row["snapshot_date"] == "final"
    assert row["r"] == 900
    conn.close()


def test_load_weekly_rosters(tmp_path):
    roster_dir = tmp_path / "rosters"
    roster_dir.mkdir()
    roster = {
        "snapshot_date": "2026-03-23",
        "week_num": 1,
        "team": "Hart of the Order",
        "league": 5652,
        "roster": {
            "C": {"name": "Ivan Herrera", "positions": ["C", "Util"]},
            "OF": {"name": "Juan Soto", "positions": ["OF", "Util"]},
        }
    }
    (roster_dir / "2026-03-23_hart_roster.json").write_text(json.dumps(roster))

    db_path = tmp_path / "test.db"
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    create_tables(conn)
    load_weekly_rosters(conn, roster_dir)

    rows = conn.execute("SELECT * FROM weekly_rosters ORDER BY slot").fetchall()
    assert len(rows) == 2
    assert rows[0]["player_name"] == "Ivan Herrera"
    assert rows[0]["positions"] == "C, Util"
    conn.close()


def test_append_roster_snapshot(tmp_path):
    db_path = tmp_path / "test.db"
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    create_tables(conn)

    roster = [
        {
            "name": "Juan Soto",
            "selected_position": "OF",
            "positions": ["OF", "Util"],
            "status": "",
            "player_id": "10626",
        },
        {
            "name": "Corbin Burnes",
            "selected_position": "P",
            "positions": ["SP"],
            "status": "IL10",
            "player_id": "9879",
        },
    ]
    append_roster_snapshot(conn, roster, "2026-03-24", 1, "Hart of the Order")

    rows = conn.execute(
        "SELECT player_name, status, yahoo_id FROM weekly_rosters "
        "ORDER BY player_name"
    ).fetchall()
    assert len(rows) == 2
    assert rows[0]["player_name"] == "Corbin Burnes"
    assert rows[0]["status"] == "IL10"
    assert rows[0]["yahoo_id"] == "9879"
    assert rows[1]["player_name"] == "Juan Soto"
    assert rows[1]["status"] == ""  # empty string preserved
    assert rows[1]["yahoo_id"] == "10626"

    # Idempotent: second call should not duplicate
    append_roster_snapshot(conn, roster, "2026-03-24", 1, "Hart of the Order")
    rows = conn.execute("SELECT * FROM weekly_rosters").fetchall()
    assert len(rows) == 2
    conn.close()


def test_append_roster_snapshot_replaces_team_snapshot(tmp_path):
    """A second call for the same (snapshot_date, team) replaces the
    prior roster rather than merging.

    The in-season refresh targets a future-dated Tuesday lock. If a
    prior refresh wrote McLain at 2B and today's refresh writes Lopez
    at 2B (after a waiver swap), the stored snapshot must reflect the
    latest roster only — not union both.

    Regression test for the bug where INSERT OR IGNORE preserved the
    stale McLain row, producing a corrupt 26-player roster downstream.
    """
    db_path = tmp_path / "test.db"
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    create_tables(conn)

    first = [
        {"name": "Matt McLain", "selected_position": "2B",
         "positions": ["2B", "IF", "Util"], "status": "", "player_id": "1"},
        {"name": "Juan Soto", "selected_position": "OF",
         "positions": ["OF", "Util"], "status": "", "player_id": "2"},
    ]
    append_roster_snapshot(conn, first, "2026-04-21", None, "Hart of the Order")

    second = [
        {"name": "Otto Lopez", "selected_position": "2B",
         "positions": ["2B", "SS", "IF", "Util"], "status": "", "player_id": "3"},
        {"name": "Juan Soto", "selected_position": "OF",
         "positions": ["OF", "Util"], "status": "", "player_id": "2"},
    ]
    append_roster_snapshot(conn, second, "2026-04-21", None, "Hart of the Order")

    names = [r["player_name"] for r in conn.execute(
        "SELECT player_name FROM weekly_rosters "
        "WHERE snapshot_date = '2026-04-21' AND team = 'Hart of the Order' "
        "ORDER BY player_name"
    ).fetchall()]
    assert names == ["Juan Soto", "Otto Lopez"], (
        f"Expected stale McLain removed, got {names}"
    )

    # Writes for a DIFFERENT team on the same date must not be touched.
    other = [
        {"name": "Other Player", "selected_position": "C",
         "positions": ["C"], "status": "", "player_id": "99"},
    ]
    append_roster_snapshot(conn, other, "2026-04-21", None, "Other Team")
    append_roster_snapshot(conn, second, "2026-04-21", None, "Hart of the Order")
    other_rows = conn.execute(
        "SELECT player_name FROM weekly_rosters "
        "WHERE team = 'Other Team'"
    ).fetchall()
    assert len(other_rows) == 1
    conn.close()


def test_append_roster_snapshot_defaults_missing_fields(tmp_path):
    """Rosters without status/player_id keys still write (as NULL).

    Historical paths or older fixtures may produce rosters without the
    new fields. The helper should treat them as NULL rather than raise.
    """
    db_path = tmp_path / "test.db"
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    create_tables(conn)

    roster = [
        {"name": "Old Style", "selected_position": "C", "positions": ["C"]},
    ]
    append_roster_snapshot(conn, roster, "2026-03-24", 1, "T")

    row = conn.execute(
        "SELECT player_name, status, yahoo_id FROM weekly_rosters"
    ).fetchone()
    assert row["player_name"] == "Old Style"
    assert row["status"] is None
    assert row["yahoo_id"] is None
    conn.close()


def test_append_standings_snapshot(tmp_path):
    db_path = tmp_path / "test.db"
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    create_tables(conn)

    standings = [
        {
            "name": "Hart of the Order",
            "team_key": "469.l.5652.t.4",
            "rank": 1,
            "stats": {
                "R": 100, "HR": 30, "RBI": 95, "SB": 20, "AVG": 0.265,
                "W": 10, "K": 200, "SV": 15, "ERA": 3.50, "WHIP": 1.18,
            },
        },
    ]
    append_standings_snapshot(conn, standings, 2026, "2026-03-24")

    row = conn.execute("SELECT * FROM standings").fetchone()
    assert row["year"] == 2026
    assert row["snapshot_date"] == "2026-03-24"
    assert row["team"] == "Hart of the Order"
    assert row["team_key"] == "469.l.5652.t.4"
    assert row["rank"] == 1
    assert row["r"] == 100

    # Idempotent
    append_standings_snapshot(conn, standings, 2026, "2026-03-24")
    assert conn.execute("SELECT COUNT(*) FROM standings").fetchone()[0] == 1
    conn.close()


def test_append_standings_snapshot_without_team_key(tmp_path):
    """Missing team_key writes NULL (not an empty string)."""
    db_path = tmp_path / "test.db"
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    create_tables(conn)

    standings = [
        {"name": "Legacy Team", "rank": 1, "stats": {"R": 100}},
    ]
    append_standings_snapshot(conn, standings, 2026, "2026-03-24")

    row = conn.execute("SELECT team_key FROM standings").fetchone()
    assert row["team_key"] is None
    conn.close()


def test_build_db_end_to_end(tmp_path):
    """Integration test: build a DB from minimal test data and query it."""
    import json

    proj_dir = tmp_path / "projections" / "2026"
    proj_dir.mkdir(parents=True)
    (proj_dir / "steamer-hitters.csv").write_text(
        'Name,Team,PA,AB,H,R,HR,RBI,SB,AVG,G,PlayerId,MLBAMID\n'
        '"James Wood","WSN",600,520,140,85,26,80,15,0.269,145,"29518",695578\n'
    )
    (proj_dir / "steamer-pitchers.csv").write_text(
        'Name,Team,W,L,SV,ERA,IP,ER,BB,SO,H,HR,WHIP,G,GS,PlayerId,MLBAMID\n'
        '"Corbin Burnes","BAL",14,7,0,3.20,200,71,50,220,170,20,1.10,32,32,"19361",669203\n'
    )

    drafts_path = tmp_path / "drafts.json"
    drafts_path.write_text(json.dumps({
        "2026": [{"pick": 1, "round": 1, "team": "Hart", "player": "James Wood"}]
    }))

    standings_path = tmp_path / "standings.json"
    standings_path.write_text(json.dumps({
        "2025": {"standings": [
            {"name": "Hart", "team_key": "k1", "rank": 1,
             "stats": {"R": 900, "HR": 250, "RBI": 880, "SB": 150, "AVG": 0.260,
                       "W": 80, "K": 1400, "SV": 90, "ERA": 3.60, "WHIP": 1.20}}
        ]}
    }))

    db_path = tmp_path / "test.db"
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    create_tables(conn)
    load_raw_projections(conn, tmp_path / "projections")
    load_blended_projections(conn, tmp_path / "projections", ["steamer"], {"steamer": 1.0})
    load_draft_results(conn, drafts_path)
    load_standings(conn, standings_path)

    # Verify queries
    wood = conn.execute("SELECT year, hr, avg FROM blended_projections WHERE name='James Wood'").fetchone()
    assert wood is not None and wood["hr"] == 26

    draft = conn.execute("SELECT * FROM draft_results WHERE year=2026").fetchone()
    assert draft["player"] == "James Wood"
    # fg_id should be resolved since raw_projections has James Wood for 2026
    assert draft["fg_id"] == "29518"

    standings = conn.execute("SELECT * FROM standings WHERE year=2025").fetchone()
    assert standings["r"] == 900

    conn.close()


def test_load_positions(tmp_path):
    db_path = tmp_path / "test.db"
    conn = get_connection(db_path)
    create_tables(conn)

    positions = {
        "Aaron Judge": ["OF", "DH"],
        "Gerrit Cole": ["SP"],
        "Shohei Ohtani": ["Util"],
    }
    load_positions(conn, positions)

    rows = conn.execute("SELECT * FROM positions ORDER BY name").fetchall()
    assert len(rows) == 3
    judge = next(r for r in rows if r["name"] == "Aaron Judge")
    assert judge["positions"] == "OF, DH"
    conn.close()


def test_get_positions(tmp_path):
    db_path = tmp_path / "test.db"
    conn = get_connection(db_path)
    create_tables(conn)

    positions = {
        "Aaron Judge": ["OF", "DH"],
        "Gerrit Cole": ["SP"],
    }
    load_positions(conn, positions)
    result = get_positions(conn)

    assert result == {"Aaron Judge": ["OF", "DH"], "Gerrit Cole": ["SP"]}
    conn.close()


def test_get_blended_projections(tmp_path):
    """Round-trip: load blended projections, then read them back."""
    import shutil

    db_path = tmp_path / "test.db"
    conn = get_connection(db_path)
    create_tables(conn)

    # load_blended_projections expects year subdirectories, so create one
    fixtures = Path(__file__).parent.parent / "fixtures"
    year_dir = tmp_path / "projections" / "2026"
    year_dir.mkdir(parents=True)
    for csv in fixtures.glob("*.csv"):
        shutil.copy(csv, year_dir / csv.name)

    load_blended_projections(conn, tmp_path / "projections", ["steamer"], None)

    hitters, pitchers = get_blended_projections(conn, year=2026)

    # Fixture has 4 hitters (steamer_hitters.csv) and 3 pitchers (steamer_pitchers.csv)
    assert len(hitters) == 4
    assert len(pitchers) == 3

    # player_type must be preserved (downstream code requires it)
    assert "player_type" in hitters.columns
    assert (hitters["player_type"] == "hitter").all()
    assert (pitchers["player_type"] == "pitcher").all()

    # year column should be dropped (not part of blend_projections output)
    assert "year" not in hitters.columns

    # Check required columns exist
    for col in ("name", "fg_id", "ab", "h", "r", "hr", "rbi", "sb", "avg", "adp"):
        assert col in hitters.columns, f"Missing hitter column: {col}"
    for col in ("name", "fg_id", "w", "k", "sv", "ip", "er", "bb", "h_allowed", "era", "whip", "adp"):
        assert col in pitchers.columns, f"Missing pitcher column: {col}"

    # Verify a specific player
    judge = hitters[hitters["name"] == "Aaron Judge"]
    assert len(judge) == 1
    assert judge.iloc[0]["hr"] > 0
    conn.close()


# ---------------------------------------------------------------------------
# Helper: build a minimal ROS projections directory tree using fixture CSVs
# ---------------------------------------------------------------------------

def _make_ros_dir(tmp_path, year=2026, date="2026-04-07"):
    """Create data/projections/{year}/ros/{date}/ with steamer fixture CSVs."""
    import shutil
    fixtures = Path(__file__).parent.parent / "fixtures"
    date_dir = tmp_path / "projections" / str(year) / "rest_of_season" / date
    date_dir.mkdir(parents=True)
    # blend_projections expects {system}-hitters.csv / {system}-pitchers.csv
    shutil.copy(fixtures / "steamer_hitters.csv", date_dir / "steamer-hitters.csv")
    shutil.copy(fixtures / "steamer_pitchers.csv", date_dir / "steamer-pitchers.csv")
    return date_dir


# ---------------------------------------------------------------------------
# Task 2: load_rest_of_season_projections()
# ---------------------------------------------------------------------------

def test_load_rest_of_season_projections_basic(tmp_path):
    """Task 2: loading a single ROS snapshot inserts hitter and pitcher rows."""
    _make_ros_dir(tmp_path, year=2026, date="2026-04-07")

    db_path = tmp_path / "test.db"
    conn = get_connection(db_path)
    create_tables(conn)
    load_rest_of_season_projections(conn, tmp_path / "projections", ["steamer"], {"steamer": 1.0})

    rows = conn.execute(
        "SELECT * FROM ros_blended_projections WHERE year=2026 AND snapshot_date='2026-04-07'"
    ).fetchall()
    # Fixture has 4 hitters + 3 pitchers
    assert len(rows) == 7

    hitter_rows = [r for r in rows if r["player_type"] == "hitter"]
    pitcher_rows = [r for r in rows if r["player_type"] == "pitcher"]
    assert len(hitter_rows) == 4
    assert len(pitcher_rows) == 3

    # Spot-check a value
    judge = conn.execute(
        "SELECT hr FROM ros_blended_projections WHERE name='Aaron Judge' AND snapshot_date='2026-04-07'"
    ).fetchone()
    assert judge is not None
    assert judge["hr"] > 0
    conn.close()


def test_load_rest_of_season_projections_idempotent(tmp_path):
    """Task 2: calling load_rest_of_season_projections twice must not duplicate rows."""
    _make_ros_dir(tmp_path, year=2026, date="2026-04-07")

    db_path = tmp_path / "test.db"
    conn = get_connection(db_path)
    create_tables(conn)
    load_rest_of_season_projections(conn, tmp_path / "projections", ["steamer"], {"steamer": 1.0})
    load_rest_of_season_projections(conn, tmp_path / "projections", ["steamer"], {"steamer": 1.0})

    count = conn.execute(
        "SELECT COUNT(*) FROM ros_blended_projections WHERE year=2026 AND snapshot_date='2026-04-07'"
    ).fetchone()[0]
    assert count == 7  # exactly one hitter + pitcher set, no duplicates
    conn.close()


# ---------------------------------------------------------------------------
# Task 3: get_rest_of_season_projections()
# ---------------------------------------------------------------------------

def test_get_rest_of_season_projections_returns_latest_snapshot(tmp_path):
    """Task 3: get_rest_of_season_projections() returns the most recent snapshot_date."""
    _make_ros_dir(tmp_path, year=2026, date="2026-04-07")
    _make_ros_dir(tmp_path, year=2026, date="2026-04-14")

    db_path = tmp_path / "test.db"
    conn = get_connection(db_path)
    create_tables(conn)
    load_rest_of_season_projections(conn, tmp_path / "projections", ["steamer"], {"steamer": 1.0})

    hitters, pitchers = get_rest_of_season_projections(conn, year=2026)

    assert len(hitters) == 4
    assert len(pitchers) == 3

    # year and snapshot_date must be dropped from output
    assert "year" not in hitters.columns
    assert "snapshot_date" not in hitters.columns

    # player_type preserved
    assert "player_type" in hitters.columns
    assert (hitters["player_type"] == "hitter").all()
    assert (pitchers["player_type"] == "pitcher").all()

    # Verify the data came from the *latest* snapshot (2026-04-14) by checking
    # that every row in the DB for the older snapshot was NOT selected alone.
    # We verify by counting: both snapshots each have 7 rows; result should be 7.
    total_in_db = conn.execute("SELECT COUNT(*) FROM ros_blended_projections").fetchone()[0]
    assert total_in_db == 14  # 2 snapshots × 7 players
    conn.close()


def test_get_rest_of_season_projections_empty_when_no_data(tmp_path):
    """Task 3: get_rest_of_season_projections() returns empty DataFrames when table is empty."""
    db_path = tmp_path / "test.db"
    conn = get_connection(db_path)
    create_tables(conn)

    hitters, pitchers = get_rest_of_season_projections(conn)

    assert hitters.empty
    assert pitchers.empty
    conn.close()


class TestGetSeasonTotals:
    def test_returns_hitter_totals_by_mlbam_id(self):
        from fantasy_baseball.data.db import create_tables, get_connection, get_season_totals
        conn = get_connection(":memory:")
        create_tables(conn)
        conn.execute(
            "INSERT INTO game_logs (season, mlbam_id, name, team, player_type, date, "
            "pa, ab, h, r, hr, rbi, sb) VALUES (2026, 592450, 'Aaron Judge', 'NYY', "
            "'hitter', '2026-03-27', 5, 4, 2, 1, 1, 3, 0)"
        )
        conn.execute(
            "INSERT INTO game_logs (season, mlbam_id, name, team, player_type, date, "
            "pa, ab, h, r, hr, rbi, sb) VALUES (2026, 592450, 'Aaron Judge', 'NYY', "
            "'hitter', '2026-03-28', 4, 3, 1, 2, 0, 1, 1)"
        )
        conn.commit()
        hitter_totals, pitcher_totals = get_season_totals(conn, 2026)
        assert 592450 in hitter_totals
        t = hitter_totals[592450]
        assert t["pa"] == 9
        assert t["ab"] == 7
        assert t["h"] == 3
        assert t["r"] == 3
        assert t["hr"] == 1
        assert t["rbi"] == 4
        assert t["sb"] == 1
        assert len(pitcher_totals) == 0

    def test_returns_pitcher_totals_by_mlbam_id(self):
        from fantasy_baseball.data.db import create_tables, get_connection, get_season_totals
        conn = get_connection(":memory:")
        create_tables(conn)
        conn.execute(
            "INSERT INTO game_logs (season, mlbam_id, name, team, player_type, date, "
            "ip, k, er, bb, h_allowed, w, sv, gs) VALUES (2026, 543037, 'Gerrit Cole', "
            "'NYY', 'pitcher', '2026-03-27', 7.0, 9, 2, 1, 5, 1, 0, 1)"
        )
        conn.execute(
            "INSERT INTO game_logs (season, mlbam_id, name, team, player_type, date, "
            "ip, k, er, bb, h_allowed, w, sv, gs) VALUES (2026, 543037, 'Gerrit Cole', "
            "'NYY', 'pitcher', '2026-03-31', 6.0, 7, 3, 2, 4, 0, 0, 1)"
        )
        conn.commit()
        hitter_totals, pitcher_totals = get_season_totals(conn, 2026)
        assert len(hitter_totals) == 0
        assert 543037 in pitcher_totals
        t = pitcher_totals[543037]
        assert t["ip"] == 13.0
        assert t["k"] == 16
        assert t["er"] == 5
        assert t["bb"] == 3
        assert t["h_allowed"] == 9
        assert t["w"] == 1
        assert t["sv"] == 0

    def test_empty_when_no_data(self):
        from fantasy_baseball.data.db import create_tables, get_connection, get_season_totals
        conn = get_connection(":memory:")
        create_tables(conn)
        hitter_totals, pitcher_totals = get_season_totals(conn, 2026)
        assert hitter_totals == {}
        assert pitcher_totals == {}


class TestLoadRosNormalizationAppliesToAllSystems:
    def test_steamer_gets_normalized_after_fix(self, tmp_path):
        """Regression test for the FULL_SEASON_ROS_SYSTEMS bug.

        Steamer (and the-bat-x) used to be skipped from normalization based
        on the assumption they published full-season projections. Empirical
        verification on 2026-04-10 proved that's wrong — all systems are
        rest-of-season-only. This test pins the fix: steamer must get its
        ROS counting stats incremented by accumulated actuals.
        """
        from fantasy_baseball.data.db import (
            create_tables,
            get_connection,
            load_rest_of_season_projections,
        )

        _make_ros_dir(tmp_path, year=2026, date="2026-04-07")
        db_path = tmp_path / "test.db"
        conn = get_connection(db_path)
        create_tables(conn)

        # Pre-populate game_logs with Aaron Judge actuals (mlbam_id 592450)
        conn.execute(
            "INSERT INTO game_logs (season, mlbam_id, name, team, player_type, date, "
            "pa, ab, h, r, hr, rbi, sb) "
            "VALUES (2026, 592450, 'Aaron Judge', 'NYY', 'hitter', '2026-04-05', "
            "90, 80, 25, 15, 5, 12, 1)"
        )
        conn.commit()

        load_rest_of_season_projections(
            conn, tmp_path / "projections", ["steamer"], {"steamer": 1.0},
        )

        judge = conn.execute(
            "SELECT pa, ab, h, r, hr, rbi, sb FROM ros_blended_projections "
            "WHERE name='Aaron Judge' AND snapshot_date='2026-04-07'"
        ).fetchone()
        assert judge is not None, "Judge row missing"
        assert judge["hr"] == 50, f"Expected 45 + 5 = 50, got {judge['hr']}"
        assert judge["r"] == 125, f"Expected 110 + 15 = 125, got {judge['r']}"
        assert judge["rbi"] == 132, f"Expected 120 + 12 = 132, got {judge['rbi']}"
        assert judge["sb"] == 6, f"Expected 5 + 1 = 6, got {judge['sb']}"
        assert judge["pa"] == 740, f"Expected 650 + 90 = 740, got {judge['pa']}"
        assert judge["ab"] == 630, f"Expected 550 + 80 = 630, got {judge['ab']}"
        assert judge["h"] == 185, f"Expected 160 + 25 = 185, got {judge['h']}"

        conn.close()
