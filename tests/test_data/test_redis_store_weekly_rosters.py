"""Tests for weekly_rosters_history helpers."""
from fantasy_baseball.data import redis_store

ENTRY_A_TEAM_1 = {
    "slot": "SS", "player_name": "Bobby Witt Jr.",
    "positions": "SS", "status": "", "yahoo_id": "10764",
}
ENTRY_B_TEAM_1 = {
    "slot": "1B", "player_name": "Vladimir Guerrero Jr.",
    "positions": "1B", "status": "", "yahoo_id": "10621",
}
ENTRY_A_TEAM_2 = {
    "slot": "OF", "player_name": "Juan Soto",
    "positions": "OF", "status": "", "yahoo_id": "10765",
}


def test_write_roster_snapshot_single_team(fake_redis):
    redis_store.write_roster_snapshot(
        fake_redis, "2026-04-15", "Alpha", [ENTRY_A_TEAM_1, ENTRY_B_TEAM_1]
    )
    day = redis_store.get_weekly_roster_day(fake_redis, "2026-04-15")
    assert day == [
        {**ENTRY_A_TEAM_1, "team": "Alpha"},
        {**ENTRY_B_TEAM_1, "team": "Alpha"},
    ]


def test_write_roster_snapshot_merges_multiple_teams(fake_redis):
    redis_store.write_roster_snapshot(
        fake_redis, "2026-04-15", "Alpha", [ENTRY_A_TEAM_1]
    )
    redis_store.write_roster_snapshot(
        fake_redis, "2026-04-15", "Beta", [ENTRY_A_TEAM_2]
    )
    day = redis_store.get_weekly_roster_day(fake_redis, "2026-04-15")
    assert len(day) == 2
    teams = {row["team"] for row in day}
    assert teams == {"Alpha", "Beta"}


def test_write_roster_snapshot_overwrites_same_team_same_day(fake_redis):
    redis_store.write_roster_snapshot(
        fake_redis, "2026-04-15", "Alpha", [ENTRY_A_TEAM_1, ENTRY_B_TEAM_1]
    )
    # Second write for Alpha on same day replaces Alpha's entries.
    redis_store.write_roster_snapshot(
        fake_redis, "2026-04-15", "Alpha", [ENTRY_B_TEAM_1]
    )
    day = redis_store.get_weekly_roster_day(fake_redis, "2026-04-15")
    assert day == [{**ENTRY_B_TEAM_1, "team": "Alpha"}]


def test_get_latest_weekly_rosters_picks_max_date(fake_redis):
    redis_store.write_roster_snapshot(
        fake_redis, "2026-04-08", "Alpha", [ENTRY_A_TEAM_1]
    )
    redis_store.write_roster_snapshot(
        fake_redis, "2026-04-15", "Alpha", [ENTRY_B_TEAM_1]
    )
    latest = redis_store.get_latest_weekly_rosters(fake_redis)
    assert latest == [{**ENTRY_B_TEAM_1, "team": "Alpha"}]


def test_get_weekly_roster_history_returns_all_dates(fake_redis):
    redis_store.write_roster_snapshot(fake_redis, "2026-04-08", "Alpha", [ENTRY_A_TEAM_1])
    redis_store.write_roster_snapshot(fake_redis, "2026-04-15", "Alpha", [ENTRY_B_TEAM_1])
    history = redis_store.get_weekly_roster_history(fake_redis)
    assert set(history.keys()) == {"2026-04-08", "2026-04-15"}
    assert history["2026-04-15"] == [{**ENTRY_B_TEAM_1, "team": "Alpha"}]


def test_get_weekly_roster_history_empty(fake_redis):
    assert redis_store.get_weekly_roster_history(fake_redis) == {}


def test_write_roster_snapshot_none_client_noop():
    # Should not raise.
    redis_store.write_roster_snapshot(None, "2026-04-15", "Alpha", [ENTRY_A_TEAM_1])


def test_get_latest_weekly_rosters_none_client_returns_empty():
    assert redis_store.get_latest_weekly_rosters(None) == []


def test_get_weekly_roster_history_none_client_returns_empty():
    assert redis_store.get_weekly_roster_history(None) == {}


def test_get_weekly_roster_day_none_client_returns_empty():
    assert redis_store.get_weekly_roster_day(None, "2026-04-15") == []


def test_get_weekly_roster_day_ignores_corrupt_json(fake_redis):
    fake_redis.hset(
        redis_store.WEEKLY_ROSTERS_HISTORY_KEY,
        "2026-04-15",
        "not json {{{",
    )
    assert redis_store.get_weekly_roster_day(fake_redis, "2026-04-15") == []


def test_get_weekly_roster_history_skips_corrupt_entries(fake_redis):
    fake_redis.hset(
        redis_store.WEEKLY_ROSTERS_HISTORY_KEY,
        "2026-04-08",
        "not json {{{",
    )
    redis_store.write_roster_snapshot(
        fake_redis, "2026-04-15", "Alpha", [ENTRY_A_TEAM_1]
    )
    history = redis_store.get_weekly_roster_history(fake_redis)
    assert set(history.keys()) == {"2026-04-15"}
    assert history["2026-04-15"] == [{**ENTRY_A_TEAM_1, "team": "Alpha"}]


def test_write_roster_snapshot_recovers_from_corrupt_day(fake_redis):
    fake_redis.hset(
        redis_store.WEEKLY_ROSTERS_HISTORY_KEY,
        "2026-04-15",
        "not json {{{",
    )
    redis_store.write_roster_snapshot(
        fake_redis, "2026-04-15", "Alpha", [ENTRY_A_TEAM_1]
    )
    day = redis_store.get_weekly_roster_day(fake_redis, "2026-04-15")
    assert day == [{**ENTRY_A_TEAM_1, "team": "Alpha"}]


# ---------------------------------------------------------------------------
# get_latest_roster_names: Redis-backed replacement for db.get_roster_names
# ---------------------------------------------------------------------------

ENTRY_WITH_BATTER_SUFFIX = {
    "slot": "SS", "player_name": "Bobby Witt Jr. (Batter)",
    "positions": "SS", "status": "", "yahoo_id": "10764",
}
ENTRY_WITH_PITCHER_SUFFIX = {
    "slot": "SP", "player_name": "Gerrit Cole (Pitcher)",
    "positions": "SP", "status": "", "yahoo_id": "10123",
}


def test_get_latest_roster_names_none_on_empty_hash(fake_redis):
    assert redis_store.get_latest_roster_names(fake_redis) is None


def test_get_latest_roster_names_none_on_none_client():
    assert redis_store.get_latest_roster_names(None) is None


def test_get_latest_roster_names_strips_suffixes_and_unions_teams(fake_redis):
    redis_store.write_roster_snapshot(
        fake_redis, "2026-04-15", "Alpha",
        [ENTRY_WITH_BATTER_SUFFIX, ENTRY_WITH_PITCHER_SUFFIX],
    )
    redis_store.write_roster_snapshot(
        fake_redis, "2026-04-15", "Beta",
        [{"slot": "OF", "player_name": "Juan Soto (Batter)",
          "positions": "OF", "status": "", "yahoo_id": "10765"}],
    )
    names = redis_store.get_latest_roster_names(fake_redis)
    # Suffixes stripped, names normalized (lowercased, accents removed).
    assert names == {"bobby witt jr.", "gerrit cole", "juan soto"}


def test_get_latest_roster_names_picks_latest_snapshot(fake_redis):
    redis_store.write_roster_snapshot(
        fake_redis, "2026-04-08", "Alpha", [ENTRY_WITH_BATTER_SUFFIX]
    )
    redis_store.write_roster_snapshot(
        fake_redis, "2026-04-15", "Alpha", [ENTRY_WITH_PITCHER_SUFFIX]
    )
    names = redis_store.get_latest_roster_names(fake_redis)
    # Only the 2026-04-15 entry's name should be present.
    assert names == {"gerrit cole"}
