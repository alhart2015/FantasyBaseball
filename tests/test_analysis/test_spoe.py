import pytest
from fantasy_baseball.data.db import (
    create_tables,
    get_connection,
    save_spoe_components,
    save_spoe_results,
)
from fantasy_baseball.models.player import Player, HitterStats, PitcherStats, PlayerType


def _make_test_hitter(name, r, hr, rbi, sb, h, ab):
    return Player(
        name=name, player_type=PlayerType.HITTER, positions=["OF"],
        ros=HitterStats(r=r, hr=hr, rbi=rbi, sb=sb, h=h, ab=ab, avg=h / ab if ab else 0),
    )


def _make_test_pitcher(name, w, k, sv, ip, er, bb, h_allowed):
    return Player(
        name=name, player_type=PlayerType.PITCHER, positions=["SP"],
        ros=PitcherStats(
            w=w, k=k, sv=sv, ip=ip, er=er, bb=bb, h_allowed=h_allowed,
            era=er * 9 / ip if ip else 0, whip=(bb + h_allowed) / ip if ip else 0,
        ),
    )


class TestProjectTeamWeek:
    def test_scales_counting_stats_by_weekly_fraction(self):
        from fantasy_baseball.analysis.spoe import project_team_week
        roster = [_make_test_hitter("Hitter", 100, 30, 90, 10, 150, 500)]
        game_log_totals = {}
        components = project_team_week(roster, game_log_totals, days_remaining=175)
        assert components["r"] == pytest.approx(100 * 7 / 175)
        assert components["hr"] == pytest.approx(30 * 7 / 175)
        assert components["h"] == pytest.approx(150 * 7 / 175)

    def test_subtracts_actuals_before_scaling(self):
        from fantasy_baseball.analysis.spoe import project_team_week
        roster = [_make_test_hitter("Hitter", 100, 30, 90, 10, 150, 500)]
        game_log_totals = {"hitter": {"hr": 5, "r": 10, "rbi": 8, "sb": 1,
                                       "h": 20, "ab": 60}}
        components = project_team_week(roster, game_log_totals, days_remaining=175)
        assert components["hr"] == pytest.approx(25 * 7 / 175)

    def test_clamps_remaining_to_zero(self):
        from fantasy_baseball.analysis.spoe import project_team_week
        roster = [_make_test_hitter("Hitter", 100, 30, 90, 10, 150, 500)]
        game_log_totals = {"hitter": {"hr": 35, "r": 10, "rbi": 8, "sb": 1,
                                       "h": 20, "ab": 60}}
        components = project_team_week(roster, game_log_totals, days_remaining=175)
        assert components["hr"] == pytest.approx(0.0)

    def test_sums_hitters_and_pitchers(self):
        from fantasy_baseball.analysis.spoe import project_team_week
        roster = [
            _make_test_hitter("Hitter", 100, 30, 90, 10, 150, 500),
            _make_test_pitcher("Pitcher", 14, 200, 0, 190, 70, 45, 160),
        ]
        components = project_team_week(roster, {}, days_remaining=175)
        assert components["r"] == pytest.approx(100 * 7 / 175)
        assert components["w"] == pytest.approx(14 * 7 / 175)
        assert components["k"] == pytest.approx(200 * 7 / 175)
        assert components["ip"] == pytest.approx(190 * 7 / 175)

    def test_unmatched_player_contributes_nothing(self):
        from fantasy_baseball.analysis.spoe import project_team_week
        unmatched = Player(
            name="Ghost", player_type=PlayerType.HITTER, positions=["OF"],
        )
        roster = [unmatched]
        components = project_team_week(roster, {}, days_remaining=175)
        assert components["r"] == pytest.approx(0.0)


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


def _seed_standings(conn):
    """Insert test standings data for two teams."""
    conn.executemany(
        "INSERT INTO standings "
        "(year, snapshot_date, team, rank, r, hr, rbi, sb, avg, w, k, sv, era, whip) "
        "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
        [
            (2026, "2026-03-31", "Team A", 1, 30, 8, 25, 3, 0.280, 3, 40, 2, 3.20, 1.10),
            (2026, "2026-03-31", "Team B", 2, 25, 6, 20, 5, 0.260, 2, 35, 4, 3.80, 1.25),
        ],
    )
    conn.commit()


def _make_test_config():
    from fantasy_baseball.config import LeagueConfig

    return LeagueConfig(
        league_id=1,
        num_teams=2,
        game_code="mlb",
        team_name="Team A",
        draft_position=1,
        keepers=[],
        roster_slots={},
        projection_systems=["steamer"],
        projection_weights={"steamer": 1.0},
        season_year=2026,
        season_start="2026-03-27",
        season_end="2026-09-28",
    )


def _seed_full_scenario(conn):
    """Seed all data needed for a compute_spoe run: rosters, projections,
    game logs, and standings for one week (2026-03-31)."""
    # Only week 1 rosters (2026-03-31)
    conn.executemany(
        "INSERT INTO weekly_rosters "
        "(snapshot_date, week_num, team, slot, player_name, positions) "
        "VALUES (?, ?, ?, ?, ?, ?)",
        [
            ("2026-03-31", None, "Team A", "OF", "Juan Soto", "OF, Util"),
            ("2026-03-31", None, "Team A", "P", "Aaron Nola", "SP"),
            ("2026-03-31", None, "Team B", "1B", "Freddie Freeman", "1B, Util"),
            ("2026-03-31", None, "Team B", "P", "Logan Webb", "SP"),
        ],
    )
    conn.commit()

    # Projections that cover both teams' players
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
            (2026, "2026-03-30", "3", "Freddie Freeman", "LAD", "hitter",
             600, 540, 170, 95, 25, 85, 8, 0.315, 0, 0, 0, 0, 0, 0, 0, 0, 0, 15),
            (2026, "2026-03-30", "4", "Logan Webb", "SF", "pitcher",
             0, 0, 0, 0, 0, 0, 0, 0, 12, 170, 0, 200, 75, 50, 170, 3.38, 1.10, 60),
        ],
    )
    conn.commit()

    # Game logs before 2026-03-31 (minimal)
    conn.executemany(
        "INSERT INTO game_logs "
        "(season, mlbam_id, name, team, player_type, date, "
        "pa, ab, h, r, hr, rbi, sb, ip, k, er, bb, h_allowed, w, sv, gs) "
        "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
        [
            (2026, 1, "Juan Soto", "NYM", "hitter", "2026-03-28",
             5, 4, 2, 1, 1, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0),
        ],
    )
    conn.commit()

    _seed_standings(conn)


class TestComponentsToRotoStats:
    def test_counting_stats_pass_through(self):
        from fantasy_baseball.analysis.spoe import components_to_roto_stats

        components = {
            "r": 50.0, "hr": 15.0, "rbi": 45.0, "sb": 8.0,
            "h": 100.0, "ab": 400.0,
            "w": 10.0, "k": 150.0, "sv": 5.0,
            "ip": 180.0, "er": 60.0, "bb": 40.0, "h_allowed": 150.0,
        }
        result = components_to_roto_stats(components)
        assert result["R"] == 50.0
        assert result["HR"] == 15.0
        assert result["RBI"] == 45.0
        assert result["SB"] == 8.0
        assert result["W"] == 10.0
        assert result["K"] == 150.0
        assert result["SV"] == 5.0

    def test_rate_stats_computed_from_components(self):
        from fantasy_baseball.analysis.spoe import components_to_roto_stats

        components = {
            "r": 50.0, "hr": 15.0, "rbi": 45.0, "sb": 8.0,
            "h": 100.0, "ab": 400.0,
            "w": 10.0, "k": 150.0, "sv": 5.0,
            "ip": 180.0, "er": 60.0, "bb": 40.0, "h_allowed": 150.0,
        }
        result = components_to_roto_stats(components)
        assert result["AVG"] == pytest.approx(100.0 / 400.0)
        assert result["ERA"] == pytest.approx(60.0 * 9 / 180.0)
        assert result["WHIP"] == pytest.approx((40.0 + 150.0) / 180.0)

    def test_zero_ip_defaults(self):
        from fantasy_baseball.analysis.spoe import components_to_roto_stats

        components = {
            "r": 0.0, "hr": 0.0, "rbi": 0.0, "sb": 0.0,
            "h": 0.0, "ab": 0.0,
            "w": 0.0, "k": 0.0, "sv": 0.0,
            "ip": 0.0, "er": 0.0, "bb": 0.0, "h_allowed": 0.0,
        }
        result = components_to_roto_stats(components)
        assert result["AVG"] == 0.0
        assert result["ERA"] == 99.0
        assert result["WHIP"] == 99.0


class TestComputeSpoe:
    def test_produces_results_for_all_teams_and_categories(self):
        from fantasy_baseball.analysis.spoe import compute_spoe
        from fantasy_baseball.data.db import get_spoe_results

        conn = get_connection(":memory:")
        create_tables(conn)
        _seed_full_scenario(conn)
        config = _make_test_config()

        compute_spoe(conn, config)

        results = get_spoe_results(conn, 2026, "2026-03-31")
        assert len(results) > 0

        # Each team should have 10 categories + 1 total = 11 entries
        team_a_results = [r for r in results if r["team"] == "Team A"]
        team_b_results = [r for r in results if r["team"] == "Team B"]
        assert len(team_a_results) == 11
        assert len(team_b_results) == 11

        # Verify all 10 categories present
        cats_a = {r["category"] for r in team_a_results}
        expected_cats = {"R", "HR", "RBI", "SB", "AVG", "W", "K", "SV", "ERA", "WHIP", "total"}
        assert cats_a == expected_cats
        conn.close()

    def test_spoe_is_actual_minus_projected(self):
        from fantasy_baseball.analysis.spoe import compute_spoe
        from fantasy_baseball.data.db import get_spoe_results

        conn = get_connection(":memory:")
        create_tables(conn)
        _seed_full_scenario(conn)
        config = _make_test_config()

        compute_spoe(conn, config)

        results = get_spoe_results(conn, 2026, "2026-03-31")
        for r in results:
            if r["category"] != "total":
                assert r["spoe"] == pytest.approx(
                    r["actual_pts"] - r["projected_pts"]
                ), f"SPOE mismatch for {r['team']} {r['category']}"

    def test_total_spoe_is_sum_of_categories(self):
        from fantasy_baseball.analysis.spoe import compute_spoe
        from fantasy_baseball.data.db import get_spoe_results

        conn = get_connection(":memory:")
        create_tables(conn)
        _seed_full_scenario(conn)
        config = _make_test_config()

        compute_spoe(conn, config)

        results = get_spoe_results(conn, 2026, "2026-03-31")
        for team in ("Team A", "Team B"):
            team_results = [r for r in results if r["team"] == team]
            cat_spoes = [r["spoe"] for r in team_results if r["category"] != "total"]
            total_row = [r for r in team_results if r["category"] == "total"][0]
            assert total_row["spoe"] == pytest.approx(sum(cat_spoes))

    def test_skips_completed_weeks_except_current(self):
        from fantasy_baseball.analysis.spoe import compute_spoe
        from fantasy_baseball.data.db import get_spoe_results

        conn = get_connection(":memory:")
        create_tables(conn)
        config = _make_test_config()

        # -- Seed week 1 (2026-03-31) with full data --
        _seed_full_scenario(conn)

        # Run compute_spoe to generate week 1 results
        compute_spoe(conn, config)
        week1_results = get_spoe_results(conn, 2026, "2026-03-31")
        assert len(week1_results) > 0

        # -- Seed week 2 data (2026-04-07) --
        conn.executemany(
            "INSERT INTO weekly_rosters "
            "(snapshot_date, week_num, team, slot, player_name, positions) "
            "VALUES (?, ?, ?, ?, ?, ?)",
            [
                ("2026-04-07", None, "Team A", "OF", "Juan Soto", "OF, Util"),
                ("2026-04-07", None, "Team A", "P", "Aaron Nola", "SP"),
                ("2026-04-07", None, "Team B", "1B", "Freddie Freeman", "1B, Util"),
                ("2026-04-07", None, "Team B", "P", "Logan Webb", "SP"),
            ],
        )
        conn.executemany(
            "INSERT INTO standings "
            "(year, snapshot_date, team, rank, r, hr, rbi, sb, avg, w, k, sv, era, whip) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            [
                (2026, "2026-04-07", "Team A", 1, 60, 16, 50, 6, 0.275, 6, 80, 4, 3.10, 1.08),
                (2026, "2026-04-07", "Team B", 2, 50, 12, 40, 10, 0.255, 4, 70, 8, 3.90, 1.30),
            ],
        )
        conn.commit()

        # Run again -- week 1 should be skipped (already completed),
        # week 2 (the current week) should be computed
        compute_spoe(conn, config)

        # Week 1 results should be unchanged
        week1_after = get_spoe_results(conn, 2026, "2026-03-31")
        assert len(week1_after) == len(week1_results)
        for orig, after in zip(
            sorted(week1_results, key=lambda r: (r["team"], r["category"])),
            sorted(week1_after, key=lambda r: (r["team"], r["category"])),
        ):
            assert orig["spoe"] == pytest.approx(after["spoe"])

        # Week 2 results should exist
        week2_results = get_spoe_results(conn, 2026, "2026-04-07")
        assert len(week2_results) > 0
        conn.close()


class TestProrateSpoeIntegration:
    """End-to-end: compute_spoe → load components → prorate_spoe."""

    def test_prorated_projected_stats_are_lower_mid_week(self):
        """When days_played < 7, prorated projected stats should be lower
        than the full-week projected stats stored in the DB."""
        from fantasy_baseball.analysis.spoe import (
            ALL_COMPONENTS,
            compute_spoe,
            prorate_spoe,
        )
        from fantasy_baseball.data.db import (
            get_spoe_results,
            load_spoe_components,
        )

        conn = get_connection(":memory:")
        create_tables(conn)
        _seed_full_scenario(conn)
        config = _make_test_config()

        compute_spoe(conn, config)

        # Full-week results from DB
        full_results = get_spoe_results(conn, 2026, "2026-03-31")
        full_proj_r = {
            r["team"]: r["projected_stat"]
            for r in full_results if r["category"] == "R"
        }

        # Prorate to 3 days
        current = load_spoe_components(conn, 2026, "2026-03-31")
        previous = {t: {c: 0.0 for c in ALL_COMPONENTS} for t in current}
        actual_stats = {
            r["team"]: {}
            for r in full_results if r["category"] == "R"
        }
        # Load actual stats from standings
        for r in full_results:
            if r["category"] != "total":
                actual_stats.setdefault(r["team"], {})[r["category"]] = r["actual_stat"]

        prorated = prorate_spoe(current, previous, actual_stats, days_played=3)
        prorated_proj_r = {
            r["team"]: r["projected_stat"]
            for r in prorated if r["category"] == "R"
        }

        for team in full_proj_r:
            assert prorated_proj_r[team] < full_proj_r[team], (
                f"{team}: prorated R ({prorated_proj_r[team]:.2f}) should be "
                f"less than full-week R ({full_proj_r[team]:.2f})"
            )

        conn.close()
