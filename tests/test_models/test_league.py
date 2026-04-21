from datetime import date

import pytest


def _team(name: str, team_key: str = ""):
    from fantasy_baseball.models.team import Team
    return Team(name=name, team_key=team_key, rosters=[])


def _snap(d: date, *teams: tuple[str, int]):
    from fantasy_baseball.models.standings import (
        CategoryStats,
        Standings,
        StandingsEntry,
    )
    entries = [
        StandingsEntry(name, f"k-{name}", rank, CategoryStats(r=100 + rank))
        for name, rank in teams
    ]
    return Standings(effective_date=d, entries=entries)


class TestLeagueBasics:
    def test_construction(self):
        from fantasy_baseball.models.league import League
        league = League(season_year=2026, teams=[], standings=[])
        assert league.season_year == 2026
        assert league.teams == []
        assert league.standings == []


class TestTeamLookups:
    def _league(self):
        from fantasy_baseball.models.league import League
        return League(
            season_year=2026,
            teams=[_team("Hart of the Order", "k-hart"), _team("Rivals", "k-riv")],
            standings=[],
        )

    def test_team_by_name(self):
        league = self._league()
        t = league.team_by_name("Rivals")
        assert t.name == "Rivals"
        assert t.team_key == "k-riv"

    def test_team_by_name_unknown_raises(self):
        league = self._league()
        with pytest.raises(KeyError, match="Unknown team"):
            league.team_by_name("Nobody")

    def test_team_by_key(self):
        league = self._league()
        t = league.team_by_key("k-hart")
        assert t.name == "Hart of the Order"

    def test_team_by_key_unknown_raises(self):
        league = self._league()
        with pytest.raises(KeyError, match="Unknown team_key"):
            league.team_by_key("k-nobody")


class TestStandingsLookups:
    def _league(self):
        from fantasy_baseball.models.league import League
        return League(
            season_year=2026,
            teams=[],
            standings=[
                _snap(date(2026, 3, 31), ("T1", 1)),
                _snap(date(2026, 4, 7), ("T1", 2)),
                _snap(date(2026, 4, 14), ("T1", 3)),
            ],
        )

    def test_latest_standings_returns_max_date(self):
        league = self._league()
        snap = league.latest_standings()
        assert snap.effective_date == date(2026, 4, 14)

    def test_latest_standings_raises_on_empty(self):
        from fantasy_baseball.models.league import League
        league = League(season_year=2026, teams=[], standings=[])
        with pytest.raises(ValueError, match="no standings"):
            league.latest_standings()

    def test_standings_as_of_exact_match(self):
        league = self._league()
        snap = league.standings_as_of(date(2026, 4, 7))
        assert snap is not None
        assert snap.effective_date == date(2026, 4, 7)

    def test_standings_as_of_between_snapshots(self):
        league = self._league()
        snap = league.standings_as_of(date(2026, 4, 10))
        assert snap is not None
        assert snap.effective_date == date(2026, 4, 7)

    def test_standings_as_of_before_first_returns_none(self):
        league = self._league()
        assert league.standings_as_of(date(2026, 1, 1)) is None

    def test_standings_as_of_after_last_returns_latest(self):
        league = self._league()
        snap = league.standings_as_of(date(2026, 6, 1))
        assert snap is not None
        assert snap.effective_date == date(2026, 4, 14)


def _standings(effective_date: str, teams: list[tuple]):
    """Build a canonical ``Standings`` object for fixture use.

    Each tuple is: ``(team, team_key, rank, r, hr, rbi, sb, avg,
    w, k, sv, era, whip)``.
    """
    from datetime import date as _date

    from fantasy_baseball.models.standings import (
        CategoryStats,
        Standings,
        StandingsEntry,
    )

    entries = [
        StandingsEntry(
            team_name=t,
            team_key=tk,
            rank=rk,
            stats=CategoryStats(
                r=r, hr=hr, rbi=rbi, sb=sb, avg=avg,
                w=w, k=k, sv=sv, era=era, whip=whip,
            ),
        )
        for (t, tk, rk, r, hr, rbi, sb, avg, w, k, sv, era, whip) in teams
    ]
    return Standings(
        effective_date=_date.fromisoformat(effective_date),
        entries=entries,
    )


@pytest.fixture
def redis_with_data(fake_redis, monkeypatch):
    """fake_redis pre-populated with fixture data for from_redis tests.

    Patches ``kv_store.get_kv`` so ``League.from_redis`` reads test data
    via the central KV entry point.
    """
    from fantasy_baseball.data import kv_store, redis_store

    monkeypatch.setattr(kv_store, "get_kv", lambda: fake_redis)

    # --- Rosters: two teams, three snapshots (Hart x2 + Rivals x1) ---
    redis_store.write_roster_snapshot(
        fake_redis, "2026-04-07", "Hart of the Order",
        [
            {"slot": "C",  "player_name": "Ivan Herrera",  "positions": "C, Util",  "status": "",    "yahoo_id": "11"},
            {"slot": "OF", "player_name": "Juan Soto",     "positions": "OF, Util", "status": "",    "yahoo_id": "12"},
            {"slot": "BN", "player_name": "Marcus Semien", "positions": "2B, Util", "status": "DTD", "yahoo_id": "13"},
        ],
    )
    redis_store.write_roster_snapshot(
        fake_redis, "2026-04-14", "Hart of the Order",
        [
            {"slot": "C",  "player_name": "Ivan Herrera",  "positions": "C, Util",  "status": "", "yahoo_id": "11"},
            {"slot": "OF", "player_name": "Juan Soto",     "positions": "OF, Util", "status": "", "yahoo_id": "12"},
            {"slot": "2B", "player_name": "Marcus Semien", "positions": "2B, Util", "status": "", "yahoo_id": "13"},
        ],
    )
    redis_store.write_roster_snapshot(
        fake_redis, "2026-04-14", "Rivals",
        [
            {"slot": "C", "player_name": "William Contreras", "positions": "C, Util", "status": "", "yahoo_id": "99"},
        ],
    )

    # --- Standings: two snapshots, two teams each ---
    redis_store.write_standings_snapshot(
        fake_redis,
        _standings("2026-04-07", [
            ("Hart of the Order", "k-hart", 3,
             100, 40, 110, 15, 0.270, 55, 750, 30, 3.90, 1.18),
            ("Rivals", "k-riv", 5,
             95,  35, 100, 12, 0.265, 50, 700, 28, 4.10, 1.22),
        ]),
    )
    redis_store.write_standings_snapshot(
        fake_redis,
        _standings("2026-04-14", [
            ("Hart of the Order", "k-hart", 2,
             120, 45, 130, 20, 0.275, 60, 820, 33, 3.85, 1.15),
            ("Rivals", "k-riv", 4,
             110, 38, 115, 16, 0.268, 56, 760, 31, 3.95, 1.19),
        ]),
    )

    yield fake_redis


class TestLeagueFromRedis:
    def test_loads_teams_and_standings(self, redis_with_data):
        from fantasy_baseball.models.league import League
        league = League.from_redis(season_year=2026)
        assert {t.name for t in league.teams} == {"Hart of the Order", "Rivals"}
        assert len(league.standings) == 2

    def test_season_year_stored(self, redis_with_data):
        from fantasy_baseball.models.league import League
        league = League.from_redis(season_year=2026)
        assert league.season_year == 2026

    def test_team_has_sorted_rosters(self, redis_with_data):
        from fantasy_baseball.models.league import League
        league = League.from_redis(season_year=2026)
        team = league.team_by_name("Hart of the Order")
        dates = [r.effective_date for r in team.rosters]
        assert date(2026, 4, 7) in dates
        assert date(2026, 4, 14) in dates

    def test_roster_entries_have_parsed_positions(self, redis_with_data):
        from fantasy_baseball.models.league import League
        from fantasy_baseball.models.positions import Position
        league = League.from_redis(season_year=2026)
        team = league.team_by_name("Hart of the Order")
        apr7 = team.roster_as_of(date(2026, 4, 7))
        assert apr7 is not None
        by_name = {e.name: e for e in apr7.entries}
        herrera = by_name["Ivan Herrera"]
        assert herrera.positions == [Position.C, Position.UTIL]
        assert herrera.selected_position is Position.C
        assert herrera.yahoo_id == "11"

    def test_status_nulls_become_empty_string(self, redis_with_data):
        """Redis serialization stores status="" rather than NULL.

        Empty-string status round-trips back as empty string, and
        non-empty statuses like "DTD" are preserved verbatim.
        """
        from fantasy_baseball.models.league import League
        league = League.from_redis(season_year=2026)
        team = league.team_by_name("Hart of the Order")
        apr7 = team.roster_as_of(date(2026, 4, 7))
        assert apr7 is not None
        by_name = {e.name: e for e in apr7.entries}
        assert by_name["Ivan Herrera"].status == ""
        assert by_name["Marcus Semien"].status == "DTD"

    def test_standings_entries_populated(self, redis_with_data):
        from fantasy_baseball.models.league import League
        league = League.from_redis(season_year=2026)
        snap = league.standings_as_of(date(2026, 4, 14))
        assert snap is not None
        lookup = snap.by_team()
        hart = lookup["Hart of the Order"]
        assert hart.team_key == "k-hart"
        assert hart.rank == 2
        assert hart.stats.r == 120
        assert hart.stats.era == pytest.approx(3.85)

    def test_team_appears_in_weekly_rosters_but_not_standings(
        self, redis_with_data, fake_redis,
    ):
        """A team with roster rows but no standings row still loads."""
        from fantasy_baseball.data import redis_store
        redis_store.write_roster_snapshot(
            fake_redis, "2026-04-14", "Ghost Team",
            [
                {"slot": "OF", "player_name": "Nobody", "positions": "OF, Util",
                 "status": "", "yahoo_id": ""},
            ],
        )

        from fantasy_baseball.models.league import League
        league = League.from_redis(season_year=2026)
        ghost = league.team_by_name("Ghost Team")
        assert len(ghost.rosters) == 1
        assert ghost.team_key == ""  # no standings row to provide one

    def test_team_appears_in_standings_but_not_weekly_rosters(
        self, redis_with_data, fake_redis,
    ):
        """A team with standings rows but no roster rows still loads."""
        from fantasy_baseball.data import redis_store
        # Merge into existing 2026-04-14 snapshot by re-writing the full payload.
        redis_store.write_standings_snapshot(
            fake_redis,
            _standings("2026-04-14", [
                ("Hart of the Order", "k-hart", 2,
                 120, 45, 130, 20, 0.275, 60, 820, 33, 3.85, 1.15),
                ("Rivals", "k-riv", 4,
                 110, 38, 115, 16, 0.268, 56, 760, 31, 3.95, 1.19),
                ("Standings Only", "k-so", 6,
                 80, 30, 90, 10, 0.260, 45, 650, 22, 4.20, 1.25),
            ]),
        )

        from fantasy_baseball.models.league import League
        league = League.from_redis(season_year=2026)
        so = league.team_by_name("Standings Only")
        assert so.rosters == []
        assert so.team_key == "k-so"

    def test_unknown_position_token_raises(self, redis_with_data, fake_redis):
        """Unknown position tokens in roster entries surface as errors."""
        from fantasy_baseball.data import redis_store
        redis_store.write_roster_snapshot(
            fake_redis, "2026-04-14", "Hart of the Order",
            [
                {"slot": "QB", "player_name": "Bad Data",
                 "positions": "QB, Util", "status": "", "yahoo_id": ""},
            ],
        )

        from fantasy_baseball.models.league import League
        with pytest.raises(ValueError, match="Unknown position"):
            League.from_redis(season_year=2026)

    def test_team_key_backfills_from_standings(self, redis_with_data):
        """team_key on Team comes from the standings payload."""
        from fantasy_baseball.models.league import League
        league = League.from_redis(season_year=2026)
        assert league.team_by_name("Hart of the Order").team_key == "k-hart"
        assert league.team_by_name("Rivals").team_key == "k-riv"
