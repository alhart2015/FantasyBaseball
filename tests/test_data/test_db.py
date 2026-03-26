import sqlite3
from fantasy_baseball.data.db import create_tables, get_connection, load_raw_projections, load_blended_projections, DB_PATH


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


def test_create_tables_is_idempotent(tmp_path):
    db_path = tmp_path / "test.db"
    conn = sqlite3.connect(db_path)
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
