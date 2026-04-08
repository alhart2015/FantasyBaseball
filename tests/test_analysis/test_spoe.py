import pytest
from fantasy_baseball.data.db import create_tables, get_connection


def _seed_rosters(conn):
    """Insert test roster data for two teams across two weeks."""
    conn.executemany(
        "INSERT INTO weekly_rosters "
        "(snapshot_date, week_num, team, slot, player_name, positions) "
        "VALUES (?, ?, ?, ?, ?, ?)",
        [
            ("2026-03-31", None, "Team A", "OF", "Juan Soto", "OF, Util"),
            ("2026-03-31", None, "Team A", "P", "Aaron Nola", "SP"),
            ("2026-03-31", None, "Team B", "1B", "Freddie Freeman", "1B, Util"),
            ("2026-03-31", None, "Team B", "P", "Logan Webb", "SP"),
            ("2026-04-07", None, "Team A", "OF", "Freeman Jr", "OF"),
            ("2026-04-07", None, "Team A", "P", "Aaron Nola", "SP"),
            ("2026-04-07", None, "Team B", "1B", "Freddie Freeman", "1B, Util"),
            ("2026-04-07", None, "Team B", "P", "Logan Webb", "SP"),
        ],
    )
    conn.commit()


def _seed_projections(conn):
    """Insert test ROS blended projections."""
    conn.executemany(
        "INSERT INTO ros_blended_projections "
        "(year, snapshot_date, fg_id, name, team, player_type, "
        "pa, ab, h, r, hr, rbi, sb, avg, w, k, sv, ip, er, bb, h_allowed, era, whip, adp) "
        "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
        [
            (2026, "2026-03-30", "1", "Juan Soto", "NYM", "hitter",
             600, 550, 165, 100, 30, 90, 5, 0.300, 0, 0, 0, 0, 0, 0, 0, 0, 0, 10),
            (2026, "2026-03-30", "2", "Aaron Nola", "PHI", "pitcher",
             0, 0, 0, 0, 0, 0, 0, 0, 14, 200, 0, 190, 70, 45, 160, 3.32, 1.08, 50),
            (2026, "2026-04-05", "1", "Juan Soto", "NYM", "hitter",
             580, 530, 155, 95, 28, 85, 4, 0.292, 0, 0, 0, 0, 0, 0, 0, 0, 0, 10),
            (2026, "2026-04-05", "2", "Aaron Nola", "PHI", "pitcher",
             0, 0, 0, 0, 0, 0, 0, 0, 13, 190, 0, 180, 68, 42, 155, 3.40, 1.09, 50),
        ],
    )
    conn.commit()


def _seed_game_logs(conn):
    """Insert test game log data."""
    conn.executemany(
        "INSERT INTO game_logs "
        "(season, mlbam_id, name, team, player_type, date, "
        "pa, ab, h, r, hr, rbi, sb, ip, k, er, bb, h_allowed, w, sv, gs) "
        "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
        [
            (2026, 1, "Juan Soto", "NYM", "hitter", "2026-03-28",
             5, 4, 2, 1, 1, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0),
            (2026, 1, "Juan Soto", "NYM", "hitter", "2026-03-29",
             4, 3, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0),
            (2026, 2, "Aaron Nola", "PHI", "pitcher", "2026-03-29",
             0, 0, 0, 0, 0, 0, 0, 6.0, 7, 2, 1, 5, 1, 0, 1),
            (2026, 1, "Juan Soto", "NYM", "hitter", "2026-04-05",
             4, 4, 3, 2, 1, 3, 0, 0, 0, 0, 0, 0, 0, 0, 0),
        ],
    )
    conn.commit()


class TestLoadRostersForDate:
    def test_returns_all_teams(self):
        from fantasy_baseball.analysis.spoe import load_rosters_for_date
        conn = get_connection(":memory:")
        create_tables(conn)
        _seed_rosters(conn)
        rosters = load_rosters_for_date(conn, "2026-03-31")
        assert set(rosters.keys()) == {"Team A", "Team B"}
        conn.close()

    def test_splits_positions_string(self):
        from fantasy_baseball.analysis.spoe import load_rosters_for_date
        conn = get_connection(":memory:")
        create_tables(conn)
        _seed_rosters(conn)
        rosters = load_rosters_for_date(conn, "2026-03-31")
        soto = [p for p in rosters["Team A"] if p["name"] == "Juan Soto"][0]
        assert soto["positions"] == ["OF", "Util"]
        conn.close()

    def test_roster_changes_across_weeks(self):
        from fantasy_baseball.analysis.spoe import load_rosters_for_date
        conn = get_connection(":memory:")
        create_tables(conn)
        _seed_rosters(conn)
        week1 = load_rosters_for_date(conn, "2026-03-31")
        week2 = load_rosters_for_date(conn, "2026-04-07")
        week1_names = {p["name"] for p in week1["Team A"]}
        week2_names = {p["name"] for p in week2["Team A"]}
        assert "Juan Soto" in week1_names
        assert "Juan Soto" not in week2_names
        assert "Freeman Jr" in week2_names
        conn.close()


class TestLoadProjectionsForDate:
    def test_selects_latest_snapshot_before_target(self):
        from fantasy_baseball.analysis.spoe import load_projections_for_date
        conn = get_connection(":memory:")
        create_tables(conn)
        _seed_projections(conn)
        hitters, pitchers = load_projections_for_date(conn, 2026, "2026-04-07")
        assert len(hitters) == 1
        assert hitters.iloc[0]["hr"] == 28  # from 04-05 snapshot
        conn.close()

    def test_adds_name_norm_column(self):
        from fantasy_baseball.analysis.spoe import load_projections_for_date
        conn = get_connection(":memory:")
        create_tables(conn)
        _seed_projections(conn)
        hitters, pitchers = load_projections_for_date(conn, 2026, "2026-03-31")
        assert "_name_norm" in hitters.columns
        assert hitters.iloc[0]["_name_norm"] == "juan soto"
        conn.close()

    def test_falls_back_to_preseason_if_no_ros(self):
        from fantasy_baseball.analysis.spoe import load_projections_for_date
        conn = get_connection(":memory:")
        create_tables(conn)
        conn.execute(
            "INSERT INTO blended_projections "
            "(year, fg_id, name, team, player_type, "
            "pa, ab, h, r, hr, rbi, sb, avg, w, k, sv, ip, er, bb, h_allowed, era, whip, adp) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (2026, "1", "Juan Soto", "NYM", "hitter",
             600, 550, 165, 100, 30, 90, 5, 0.300, 0, 0, 0, 0, 0, 0, 0, 0, 0, 10),
        )
        conn.commit()
        hitters, pitchers = load_projections_for_date(conn, 2026, "2026-03-31")
        assert len(hitters) == 1
        assert hitters.iloc[0]["name"] == "Juan Soto"
        conn.close()


class TestAggregateGameLogsBefore:
    def test_sums_stats_before_date(self):
        from fantasy_baseball.analysis.spoe import aggregate_game_logs_before
        conn = get_connection(":memory:")
        create_tables(conn)
        _seed_game_logs(conn)
        totals = aggregate_game_logs_before(conn, 2026, "2026-04-01")
        soto = totals["juan soto"]
        assert soto["h"] == pytest.approx(3.0)
        assert soto["hr"] == pytest.approx(1.0)
        assert soto["ab"] == pytest.approx(7.0)
        conn.close()

    def test_excludes_games_on_or_after_date(self):
        from fantasy_baseball.analysis.spoe import aggregate_game_logs_before
        conn = get_connection(":memory:")
        create_tables(conn)
        _seed_game_logs(conn)
        totals = aggregate_game_logs_before(conn, 2026, "2026-04-01")
        soto = totals["juan soto"]
        assert soto["r"] == pytest.approx(1.0)
        conn.close()

    def test_includes_pitchers(self):
        from fantasy_baseball.analysis.spoe import aggregate_game_logs_before
        conn = get_connection(":memory:")
        create_tables(conn)
        _seed_game_logs(conn)
        totals = aggregate_game_logs_before(conn, 2026, "2026-04-01")
        nola = totals["aaron nola"]
        assert nola["ip"] == pytest.approx(6.0)
        assert nola["k"] == pytest.approx(7.0)
        assert nola["w"] == pytest.approx(1.0)
        conn.close()
