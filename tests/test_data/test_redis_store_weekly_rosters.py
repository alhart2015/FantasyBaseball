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
