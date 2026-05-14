import json
import sqlite3
from pathlib import Path

from fantasy_baseball.data.db import (
    create_tables,
    get_blended_projections,
    get_connection,
    get_positions,
    load_blended_projections,
    load_draft_results,
    load_positions,
    load_raw_projections,
    load_weekly_rosters,
)


def test_create_tables_creates_core_tables(tmp_path):
    db_path = tmp_path / "test.db"
    conn = sqlite3.connect(db_path)
    create_tables(conn)
    cursor = conn.execute("SELECT name FROM sqlite_master WHERE type='table' ORDER BY name")
    tables = [row[0] for row in cursor]
    assert "raw_projections" in tables
    assert "blended_projections" in tables
    assert "draft_results" in tables
    assert "weekly_rosters" in tables
    assert "positions" in tables
    conn.close()


def test_create_tables_creates_ros_blended_projections(tmp_path):
    db_path = tmp_path / "test.db"
    conn = sqlite3.connect(db_path)
    create_tables(conn)
    cursor = conn.execute("SELECT name FROM sqlite_master WHERE type='table' ORDER BY name")
    tables = [row[0] for row in cursor]
    assert "ros_blended_projections" in tables

    # Verify the PRIMARY KEY columns exist via pragma
    cols = {row[1] for row in conn.execute("PRAGMA table_info(ros_blended_projections)")}
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
        "Name,Team,PA,AB,H,R,HR,RBI,SB,CS,BB,SO,AVG,OBP,SLG,OPS,ISO,BABIP,wOBA,wRC+,WAR,ADP,G,PlayerId,MLBAMID\n"
        '"James Wood","WSN",600,520,140,85,26,80,15,5,70,150,0.269,0.350,0.480,0.830,0.211,0.320,0.370,130,4.0,50.0,145,"29518",695578\n'
    )
    (csv_dir / "steamer-pitchers.csv").write_text(
        "Name,Team,W,L,SV,ERA,G,GS,IP,H,R,ER,HR,BB,SO,K/9,BB/9,K/BB,HR/9,WHIP,FIP,WAR,ADP,PlayerId,MLBAMID\n"
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
    assert pitcher["k"] == 220  # SO maps to k for pitchers
    assert pitcher["h_allowed"] == 170  # H maps to h_allowed for pitchers
    assert pitcher["hr_p"] == 20  # HR maps to hr_p for pitchers
    assert pitcher["bb_p"] == 50  # BB maps to bb_p for pitchers
    assert pitcher["player_type"] == "pitcher"

    conn.close()


def test_load_raw_projections_handles_year_suffix(tmp_path):
    csv_dir = tmp_path / "2025"
    csv_dir.mkdir()
    (csv_dir / "steamer-hitters-2025.csv").write_text(
        "Name,Team,PA,AB,H,R,HR,RBI,SB,AVG,G,PlayerId,MLBAMID\n"
        '"Test Player","NYY",500,430,110,70,20,65,10,0.256,130,"12345",123456\n'
    )

    db_path = tmp_path / "test.db"
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    create_tables(conn)
    load_raw_projections(conn, tmp_path)

    row = conn.execute(
        "SELECT system, year FROM raw_projections WHERE name='Test Player'"
    ).fetchone()
    assert row["system"] == "steamer"
    assert row["year"] == 2025
    conn.close()


def test_load_raw_projections_insert_or_ignore(tmp_path):
    """Calling load_raw_projections twice must not raise or create duplicates."""
    csv_dir = tmp_path / "2026"
    csv_dir.mkdir()
    (csv_dir / "steamer-hitters.csv").write_text(
        'Name,Team,PA,HR,G,PlayerId,MLBAMID\n"Test Player","NYY",500,20,130,"99999",999999\n'
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
        'Name,Team,PA,HR,G,PlayerId,MLBAMID\n"Future Player","SEA",480,18,125,"77777",777777\n'
    )

    db_path = tmp_path / "test.db"
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    create_tables(conn)
    load_raw_projections(conn, tmp_path)

    row = conn.execute(
        "SELECT system, year, player_type FROM raw_projections WHERE name='Future Player'"
    ).fetchone()
    assert row["system"] == "zips"
    assert row["year"] == 2027
    assert row["player_type"] == "hitter"
    conn.close()


def test_load_blended_projections(tmp_path):
    csv_dir = tmp_path / "2026"
    csv_dir.mkdir()
    (csv_dir / "steamer-hitters.csv").write_text(
        "Name,Team,PA,AB,H,R,HR,RBI,SB,CS,BB,SO,AVG,OBP,SLG,OPS,WAR,ADP,G,PlayerId,MLBAMID\n"
        '"James Wood","WSN",600,520,140,85,26,80,15,5,70,150,0.269,0.350,0.480,0.830,4.0,50.0,145,"29518",695578\n'
    )
    (csv_dir / "steamer-pitchers.csv").write_text(
        "Name,Team,W,L,SV,ERA,G,GS,IP,H,R,ER,HR,BB,SO,K/9,BB/9,WHIP,FIP,WAR,ADP,PlayerId,MLBAMID\n"
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
        "Name,Team,PA,AB,H,R,HR,RBI,SB,CS,BB,SO,AVG,OBP,SLG,OPS,WAR,ADP,G,PlayerId,MLBAMID\n"
        '"James Wood","WSN",600,520,140,85,26,80,15,5,70,150,0.269,0.350,0.480,0.830,4.0,50.0,145,"29518",695578\n'
    )
    (csv_dir / "steamer-pitchers.csv").write_text(
        "Name,Team,W,L,SV,ERA,G,GS,IP,H,R,ER,HR,BB,SO,K/9,BB/9,WHIP,FIP,WAR,ADP,PlayerId,MLBAMID\n"
        '"Corbin Burnes","BAL",14,7,0,3.20,32,32,200,170,75,71,20,50,220,9.9,2.3,1.10,3.10,5.0,15.0,"19361",669203\n'
    )

    db_path = tmp_path / "test.db"
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    create_tables(conn)
    load_blended_projections(conn, tmp_path, ["steamer"], {"steamer": 1.0})
    load_blended_projections(
        conn, tmp_path, ["steamer"], {"steamer": 1.0}
    )  # second call must not raise

    rows = conn.execute("SELECT * FROM blended_projections WHERE year=2026").fetchall()
    assert len(rows) == 2  # one hitter + one pitcher
    conn.close()


def test_load_blended_projections_skips_bad_year(tmp_path):
    """A year directory with no matching CSVs for the requested system is skipped."""
    good_dir = tmp_path / "2026"
    good_dir.mkdir()
    (good_dir / "steamer-hitters.csv").write_text(
        "Name,Team,PA,AB,H,R,HR,RBI,SB,CS,BB,SO,AVG,OBP,SLG,OPS,WAR,ADP,G,PlayerId,MLBAMID\n"
        '"James Wood","WSN",600,520,140,85,26,80,15,5,70,150,0.269,0.350,0.480,0.830,4.0,50.0,145,"29518",695578\n'
    )
    (good_dir / "steamer-pitchers.csv").write_text(
        "Name,Team,W,L,SV,ERA,G,GS,IP,H,R,ER,HR,BB,SO,K/9,BB/9,WHIP,FIP,WAR,ADP,PlayerId,MLBAMID\n"
        '"Corbin Burnes","BAL",14,7,0,3.20,32,32,200,170,75,71,20,50,220,9.9,2.3,1.10,3.10,5.0,15.0,"19361",669203\n'
    )

    # 2027 has no steamer files — blend_projections will raise; should be skipped
    bad_dir = tmp_path / "2027"
    bad_dir.mkdir()
    (bad_dir / "zips-hitters.csv").write_text(
        "Name,Team,PA,AB,H,R,HR,RBI,SB,AVG,G,PlayerId,MLBAMID\n"
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
        },
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


def test_build_db_end_to_end(tmp_path):
    """Integration test: build a DB from minimal test data and query it."""
    import json

    proj_dir = tmp_path / "projections" / "2026"
    proj_dir.mkdir(parents=True)
    (proj_dir / "steamer-hitters.csv").write_text(
        "Name,Team,PA,AB,H,R,HR,RBI,SB,AVG,G,PlayerId,MLBAMID\n"
        '"James Wood","WSN",600,520,140,85,26,80,15,0.269,145,"29518",695578\n'
    )
    (proj_dir / "steamer-pitchers.csv").write_text(
        "Name,Team,W,L,SV,ERA,IP,ER,BB,SO,H,HR,WHIP,G,GS,PlayerId,MLBAMID\n"
        '"Corbin Burnes","BAL",14,7,0,3.20,200,71,50,220,170,20,1.10,32,32,"19361",669203\n'
    )

    drafts_path = tmp_path / "drafts.json"
    drafts_path.write_text(
        json.dumps({"2026": [{"pick": 1, "round": 1, "team": "Hart", "player": "James Wood"}]})
    )

    db_path = tmp_path / "test.db"
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    create_tables(conn)
    load_raw_projections(conn, tmp_path / "projections")
    load_blended_projections(conn, tmp_path / "projections", ["steamer"], {"steamer": 1.0})
    load_draft_results(conn, drafts_path)

    # Verify queries
    wood = conn.execute(
        "SELECT year, hr, avg FROM blended_projections WHERE name='James Wood'"
    ).fetchone()
    assert wood is not None and wood["hr"] == 26

    draft = conn.execute("SELECT * FROM draft_results WHERE year=2026").fetchone()
    assert draft["player"] == "James Wood"
    # fg_id should be resolved since raw_projections has James Wood for 2026
    assert draft["fg_id"] == "29518"

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
    for col in (
        "name",
        "fg_id",
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
    ):
        assert col in pitchers.columns, f"Missing pitcher column: {col}"

    # Verify a specific player
    judge = hitters[hitters["name"] == "Aaron Judge"]
    assert len(judge) == 1
    assert judge.iloc[0]["hr"] > 0
    conn.close()
